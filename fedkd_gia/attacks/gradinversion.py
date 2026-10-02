"""GradInversion (Yin et al., CVPR 2021, "See through Gradients") against the FedKD student.

Reference implementation: breaching (JonasGeiping/breaching, attack=seethroughgradients), adapted:
  * label restoration: argsort(min_m dL/dW_fc[n, m])[:K]  (paper Eq. 8)
  * L_grad: squared L2 distance over all shared layers, normalized by ||g*||^2
  * R_fidelity: total variation + l2 (the BN prior of Eq. 10 does not exist: ViTs have no BatchNorm)
  * R_group: G seeds optimized jointly, each pulled towards the pixel-wise mean of all seeds
    ("lazy" group consistency; RANSAC-flow registration is not used)
  * Adam, lr 0.1, cosine decay, 50 warm-up steps, Langevin noise after every step

FedKD-specific threat model: the attacker holds the global student and the client's student
gradient, but not the client's private teacher, so it simulates the distillation part of the
loss with a surrogate:
  none   : L_sim = CE_s                                (attacker ignores distillation)
  dinov2 : L_sim = (1+e^b) CE_s + e^a * L_hid(student, DINOv2 ViT-B/14)
  vitb   : L_sim = (1+e^b) CE_s + e^a * L_hid(student, ImageNet ViT-B/16, never trained in FL)
Neither surrogate has a 10-class head, so KL(p_t||p_s) is approximated by CE (teachers are ~98%
accurate, p_t ~ one-hot) and absorbed into the learnable weight (1+e^b). e^a stands in for the
unknown adaptive weight 2/(CE_t+CE_s). The hidden projector W_h is private too, so the attacker
optimizes its own W_h (and a, b) jointly with the images.
"""
import math

import torch
import torch.nn.functional as F

from models import HiddenProjector, ViTWithInternals

SURROGATES = {
    # name: (timm arch, input size giving 14x14 patches = the student's token grid, mean, std)
    "dinov2": ("vit_base_patch14_dinov2.lvd142m", 196, (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    "vitb": ("vit_base_patch16_224", 224, (0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
}


class Surrogate(torch.nn.Module):
    def __init__(self, name):
        super().__init__()
        arch, self.size, mean, std = SURROGATES[name]
        kw = {"img_size": self.size} if name == "dinov2" else {}
        self.net = ViTWithInternals(arch, num_classes=0, pretrained=True, **kw)
        self.register_buffer("mean", torch.tensor(mean).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor(std).view(1, 3, 1, 1))
        for p in self.parameters():
            p.requires_grad_(False)
        self.eval()

    def forward(self, x):
        """x is in the student's normalization (mean=std=0.5)."""
        x = x * 0.5 + 0.5
        if x.shape[-1] != self.size:
            x = F.interpolate(x, size=(self.size, self.size), mode="bilinear", align_corners=False)
        x = (x - self.mean) / self.std
        _, hidden, attn = self.net(x, return_internals=True)
        return hidden, attn


def hidden_loss(s_hidden, s_attn, t_hidden, t_attn, projector):
    ratio = len(t_hidden) // len(s_hidden)
    loss = 0.0
    for i in range(len(s_hidden)):
        j = (i + 1) * ratio - 1
        loss = loss + F.mse_loss(projector(i, s_hidden[i]), t_hidden[j])
        ta = t_attn[j]
        b, h, n, _ = ta.shape
        hs = s_attn[i].shape[1]
        loss = loss + F.mse_loss(s_attn[i], ta.view(b, hs, h // hs, n, n).mean(2))
    return loss


def restore_labels(target_grads, head_weight_name, k):
    """Paper Eq. 8: the K classes with the most negative per-row minimum of the FC weight gradient."""
    return target_grads[head_weight_name].min(dim=-1)[0].argsort()[:k]


def total_variation(x):
    dx = (x[:, :, :, 1:] - x[:, :, :, :-1]).abs().mean()
    dy = (x[:, :, 1:, :] - x[:, :, :-1, :]).abs().mean()
    return dx + dy


class GradInversion:
    def __init__(self, student, target_grads, labels, surrogate=None, cfg=None):
        self.cfg = cfg
        self.student = student.eval()
        for p in self.student.parameters():
            p.requires_grad_(False)
        self.names = list(target_grads)
        self.params = dict(self.student.named_parameters())
        self.target = [target_grads[n] for n in self.names]
        self.target_norm = sum((g * g).sum() for g in self.target)
        self.labels = labels
        self.surrogate = surrogate

    def _grad_match(self, x, nuisance):
        for n in self.names:
            self.params[n].requires_grad_(True)
        if self.surrogate is None:
            logits = self.student(x)
            loss = F.cross_entropy(logits, self.labels)
        else:
            logits, s_hidden, s_attn = self.student(x, return_internals=True)
            t_hidden, t_attn = self.surrogate(x)
            hid = hidden_loss(s_hidden, s_attn, t_hidden, t_attn, nuisance["projector"])
            loss = (1 + nuisance["b"].exp()) * F.cross_entropy(logits, self.labels) + nuisance["a"].exp() * hid
        grads = torch.autograd.grad(loss, [self.params[n] for n in self.names], create_graph=True)
        for n in self.names:
            self.params[n].requires_grad_(False)
        diff = sum(((g - t) ** 2).sum() for g, t in zip(grads, self.target))
        return diff / self.target_norm

    def run(self, shape, seeds, log_every=500, log=print):
        c = self.cfg
        device = self.labels.device
        xs, nuisances, opts = [], [], []
        for s in seeds:
            g = torch.Generator(device="cpu").manual_seed(s)
            x = torch.randn(shape, generator=g).to(device).clamp_(-1, 1).requires_grad_(True)
            nz = {}
            groups = [{"params": [x], "lr": c.lr}]
            if self.surrogate is not None:
                torch.manual_seed(s)
                t_dim = self.surrogate.net.embed_dim
                nz["projector"] = HiddenProjector(self.student.depth, self.student.embed_dim, t_dim).to(device)
                nz["a"] = torch.tensor(math.log(2.0), device=device, requires_grad=True)
                nz["b"] = torch.tensor(0.0, device=device, requires_grad=True)
                groups.append({"params": list(nz["projector"].parameters()) + [nz["a"], nz["b"]], "lr": c.nuisance_lr})
            xs.append(x); nuisances.append(nz); opts.append(torch.optim.Adam(groups))

        def lr_factor(it):
            if it < c.warmup:
                return (it + 1) / c.warmup
            return 0.5 * (1 + math.cos(math.pi * (it - c.warmup) / max(1, c.iterations - c.warmup)))

        group_start = int(c.group_start * c.iterations)
        history = []
        for it in range(c.iterations):
            f = lr_factor(it)
            consensus = torch.stack([x.detach() for x in xs]).mean(0) if len(xs) > 1 else None
            rec = []
            for x, nz, opt in zip(xs, nuisances, opts):
                for grp, base in zip(opt.param_groups, (c.lr, c.nuisance_lr)):
                    grp["lr"] = base * f
                opt.zero_grad(set_to_none=True)
                l_grad = self._grad_match(x, nz)
                loss = c.alpha_grad * l_grad + c.alpha_tv * total_variation(x) + c.alpha_l2 * (x ** 2).mean()
                if consensus is not None and it >= group_start:
                    loss = loss + c.alpha_group * ((x - consensus) ** 2).mean()
                loss.backward()
                opt.step()
                with torch.no_grad():
                    x.add_(c.alpha_noise * c.lr * f * torch.randn_like(x)).clamp_(-1, 1)
                rec.append(l_grad.item())
            if it % log_every == 0 or it == c.iterations - 1:
                history.append({"it": it, "grad_loss": rec})
                log(f"    it {it:6d}  grad-match {[round(v, 5) for v in rec]}")
        final = [x.detach() for x in xs]
        return final, history
