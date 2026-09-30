"""FedKD core (Wu et al., 2022, "Communication-efficient federated learning via knowledge distillation").

Each client holds a private mentor (teacher, ViT-B) and a shared mentee (student, ViT-S). Both are
trained locally with adaptive mutual distillation; only the student's update is sent to the server,
which averages it. The dynamic SVD compression of the paper is intentionally disabled here: the
server receives the raw student update, which is also the strongest position for a gradient
inversion attacker.

Loss, following the official implementation (model_bert.py) and paper Eq. 1-6:
    L = CE_t + CE_s + (KL_t + KL_s + 2 * L_hid) / (CE_t + CE_s)
  KL_s = KL(p_t || p_s)  mentee learns from mentor (target detached)
  KL_t = KL(p_s || p_t)  mentor learns from mentee (target detached)
  L_hid = sum_l MSE(W_h H_s^l, H_t^l) + MSE(A_s^l, A_t^l)  (trains both models)
The adaptive weight 1/(CE_t+CE_s) is detached so it rescales, not reverses, the gradients.
Attention maps of the 12-head teacher are averaged over consecutive head pairs to match the
6-head student.
"""
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from data import CachedImages
from models import shared_state


def fedkd_loss(t_out, s_out, projector, y):
    t_logits, t_hidden, t_attn = t_out
    s_logits, s_hidden, s_attn = s_out
    t_logits, s_logits = t_logits.float(), s_logits.float()

    ce_t = F.cross_entropy(t_logits, y)
    ce_s = F.cross_entropy(s_logits, y)
    kl_s = F.kl_div(F.log_softmax(s_logits, 1), F.softmax(t_logits.detach(), 1), reduction="batchmean")
    kl_t = F.kl_div(F.log_softmax(t_logits, 1), F.softmax(s_logits.detach(), 1), reduction="batchmean")

    ratio = len(t_hidden) // len(s_hidden)
    hid = 0.0
    for i in range(len(s_hidden)):
        j = (i + 1) * ratio - 1
        hid = hid + F.mse_loss(projector(i, s_hidden[i]).float(), t_hidden[j].float())
        ta = t_attn[j]
        b, h, n, _ = ta.shape
        hs = s_attn[i].shape[1]
        ta = ta.view(b, hs, h // hs, n, n).mean(2)
        hid = hid + F.mse_loss(s_attn[i].float(), ta.float())

    adaptive = (ce_t + ce_s).detach()
    total = ce_t + ce_s + (kl_t + kl_s + 2 * hid) / adaptive
    parts = {"ce_t": ce_t.item(), "ce_s": ce_s.item(), "kl_t": kl_t.item(), "kl_s": kl_s.item(),
             "hid": float(hid), "loss": total.item()}
    return total, t_logits, s_logits, parts


def client_update(teacher, student, projector, images, labels, train_idx, args, device, seed):
    """Run `args.local_epochs` of FedKD local training. Returns the student delta and train stats."""
    start = {k: v.clone() for k, v in shared_state(student).items()}
    teacher.train(); student.train(); projector.train()
    params = [p for m in (teacher, student, projector) for p in m.parameters() if p.requires_grad]
    opt = torch.optim.Adam(params, lr=args.lr)
    use_amp = device.type == "cuda" and args.amp
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    g = torch.Generator().manual_seed(seed)
    loader = DataLoader(CachedImages(images, labels, train_idx), batch_size=args.batch_size, shuffle=True,
                        generator=g, num_workers=args.num_workers, pin_memory=device.type == "cuda",
                        drop_last=False, persistent_workers=args.num_workers > 0)
    sums, n_seen, correct_t, correct_s, steps = {}, 0, 0, 0, 0
    for _ in range(args.local_epochs):
        for x, y in loader:
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            with torch.autocast(device.type, dtype=torch.float16, enabled=use_amp):
                t_out = teacher(x, return_internals=True)
                s_out = student(x, return_internals=True)
                loss, t_logits, s_logits, parts = fedkd_loss(t_out, s_out, projector, y)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            for k, v in parts.items():
                sums[k] = sums.get(k, 0.0) + v
            steps += 1
            n_seen += len(y)
            correct_t += (t_logits.argmax(1) == y).sum().item()
            correct_s += (s_logits.argmax(1) == y).sum().item()
            if args.max_steps and steps >= args.max_steps:
                break
        if args.max_steps and steps >= args.max_steps:
            break
    delta = {k: (v - start[k]).float().cpu() for k, v in shared_state(student).items()}
    stats = {k: v / max(steps, 1) for k, v in sums.items()}
    stats.update(train_acc_t=correct_t / max(n_seen, 1), train_acc_s=correct_s / max(n_seen, 1), steps=steps)
    return delta, stats


def aggregate(global_state, deltas):
    """Server: unweighted mean of client student updates (as in the official code)."""
    new_state = {}
    for k, v in global_state.items():
        new_state[k] = v + torch.stack([d[k] for d in deltas]).mean(0)
    return new_state


@torch.no_grad()
def predict(model, images, labels, idx, args, device):
    model.eval()
    loader = DataLoader(CachedImages(images, labels, idx), batch_size=args.eval_batch_size, shuffle=False,
                        num_workers=args.num_workers, pin_memory=device.type == "cuda")
    preds, ys, loss_sum = [], [], 0.0
    use_amp = device.type == "cuda" and args.amp
    for x, y in loader:
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        with torch.autocast(device.type, dtype=torch.float16, enabled=use_amp):
            logits = model(x).float()
        loss_sum += F.cross_entropy(logits, y, reduction="sum").item()
        preds.append(logits.argmax(1).cpu())
        ys.append(y.cpu())
    preds, ys = torch.cat(preds).numpy(), torch.cat(ys).numpy()
    return preds, ys, loss_sum / max(len(ys), 1)


def macro_f1(preds, ys, num_classes):
    f1s = []
    for c in range(num_classes):
        tp = np.sum((preds == c) & (ys == c))
        fp = np.sum((preds == c) & (ys != c))
        fn = np.sum((preds != c) & (ys == c))
        if tp + fp + fn == 0:
            continue
        f1s.append(2 * tp / (2 * tp + fp + fn))
    return float(np.mean(f1s)) if f1s else 0.0
