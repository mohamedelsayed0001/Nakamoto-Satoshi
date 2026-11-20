"""Record-level DP-SGD for FedKD local training (Abadi et al. 2016), accounted with dp.dpsgd_epsilon.

Every local step:
  1. Poisson sampling: each local training image joins the batch independently with probability q.
  2. Per-image gradient of the full FedKD loss w.r.t. ALL locally trained parameters jointly (the trainable part
     of the private teacher, the student and the hidden projector). The teacher must be inside the mechanism:
     it is trained on the same images and shapes the student's gradient through distillation, so a non-private
     teacher would leak into the released student.
  3. Each per-image gradient is clipped to L2 <= C, the clipped gradients are summed, N(0, (sigma C)^2) is added
     to every coordinate, and the sum is divided by the expected batch size q * N.
  4. AdamW step on that noisy gradient.
All model states (and therefore the released student update) are post-processing of the noisy gradients, so
the whole training run of a client is (epsilon, delta)-DP for each of its images; the per-sample adaptive
FedKD weight 1/(CE_t + CE_s) is computed from that image alone so no image affects another image's gradient.

Per-image gradients use torch.func (vmap over grad) in chunks; if that fails on the platform the code falls
back to an exact per-image backward loop.
"""
import math

import numpy as np
import torch
from torch.func import functional_call, grad, vmap

from fedkd import fedkd_loss
from models import shared_state

LOSS_SCALE = 8.0  # static fp16 loss scale (1024 overflowed: per-image norms are ~125); unscaled before clipping


def _trainable(modules):
    return {f"{m}.{n}": p for m, mod in modules.items() for n, p in mod.named_parameters() if p.requires_grad}


def _sub(params, prefix):
    k = len(prefix) + 1
    return {name[k:]: v for name, v in params.items() if name.startswith(prefix + ".")}


def make_sample_grad(teacher, student, projector, use_amp, device_type):
    """Returns f(params, x, y) -> (per-image grads dict, (t_logits, s_logits, parts)) for a batch, via vmap."""
    def single(params, x, y):
        x, y = x.unsqueeze(0), y.unsqueeze(0)
        with torch.autocast(device_type, dtype=torch.float16, enabled=use_amp):
            t_out = functional_call(teacher, _sub(params, "t"), (x,), {"return_internals": True})
            s_out = functional_call(student, _sub(params, "s"), (x,), {"return_internals": True})
            pp = _sub(params, "p")
            proj = lambda i, h: functional_call(projector.maps[i], {k[len(f"maps.{i}."):]: v for k, v in pp.items()
                                                                   if k.startswith(f"maps.{i}.")}, (h,))
            loss, t_logits, s_logits, parts = fedkd_loss(t_out, s_out, proj, y, with_parts=False)
        return loss * LOSS_SCALE, (t_logits[0].detach(), s_logits[0].detach(), parts)

    return vmap(grad(single, has_aux=True), in_dims=(None, 0, 0), randomness="different")


def per_sample_grads_loop(teacher, student, projector, params, x, y, use_amp, device_type):
    """Exact fallback: one backward pass per image."""
    names = list(params)
    out = {n: [] for n in names}
    t_l, s_l, parts_l = [], [], []
    for i in range(len(y)):
        with torch.autocast(device_type, dtype=torch.float16, enabled=use_amp):
            loss, t_logits, s_logits, parts = fedkd_loss(teacher(x[i:i + 1], return_internals=True),
                                                         student(x[i:i + 1], return_internals=True),
                                                         projector, y[i:i + 1], with_parts=False)
        gs = torch.autograd.grad(loss * LOSS_SCALE, [params[n] for n in names], allow_unused=True)
        for n, g in zip(names, gs):
            out[n].append(torch.zeros_like(params[n]) if g is None else g)
        t_l.append(t_logits[0].detach()); s_l.append(s_logits[0].detach()); parts_l.append(parts)
    return {n: torch.stack(v) for n, v in out.items()}, (torch.stack(t_l), torch.stack(s_l), torch.stack(parts_l))


def dpsgd_client_update(teacher, student, projector, images, labels, train_idx, args, device, seed):
    """DP-SGD version of fedkd.client_update. Returns the (already private) student delta and train stats."""
    from data import CachedImages  # local import keeps module import light

    start = {k: v.clone() for k, v in shared_state(student).items()}
    teacher.train(); student.train(); projector.train()
    modules = {"t": teacher, "s": student, "p": projector}
    params = _trainable(modules)
    teacher_lr = getattr(args, "teacher_lr", None) or args.lr
    groups = [{"params": [p for n, p in params.items() if n.startswith("t.")], "lr": teacher_lr},
              {"params": [p for n, p in params.items() if not n.startswith("t.")], "lr": args.lr}]
    opt = torch.optim.AdamW(groups, lr=args.lr, weight_decay=getattr(args, "weight_decay", 0.0))
    use_amp = device.type == "cuda" and bool(args.amp)

    train_idx = np.asarray(train_idx)
    n_train = len(train_idx)
    q = args.dp_batch_size / n_train
    steps_per_epoch = int(round(1 / q))
    ds = CachedImages(images, labels, train_idx)
    rng = np.random.default_rng(seed)
    noise_gen = torch.Generator(device=device).manual_seed(seed + 11)
    vmapped = None if getattr(args, "dp_loop", 0) else make_sample_grad(teacher, student, projector, use_amp, device.type)

    sums, n_seen, correct_t, correct_s, steps, n_clipped, norm_sum = {}, 0, 0, 0, 0, 0, 0.0
    part_names = ["ce_t", "ce_s", "kl_t", "kl_s", "hid", "loss"]
    for _ in range(args.local_epochs):
        for _ in range(steps_per_epoch):
            chosen = np.flatnonzero(rng.random(n_train) < q)
            acc = {n: torch.zeros_like(p) for n, p in params.items()}
            for c0 in range(0, len(chosen), args.dp_chunk):
                xs, ys = zip(*(ds[i] for i in chosen[c0:c0 + args.dp_chunk]))
                x = torch.stack(xs).to(device, non_blocking=True)
                y = torch.as_tensor(ys, device=device)
                if vmapped is not None:
                    try:
                        g, (t_logits, s_logits, parts) = vmapped({n: p.detach() for n, p in params.items()}, x, y)
                    except Exception as e:  # noqa: BLE001
                        print(f"vmap per-sample gradients failed ({type(e).__name__}: {e}); using per-image loop",
                              flush=True)
                        vmapped = None
                if vmapped is None:
                    g, (t_logits, s_logits, parts) = per_sample_grads_loop(teacher, student, projector, params, x, y,
                                                                           use_amp, device.type)
                # Per-image L2 norm over every trainable tensor jointly, after removing the loss scale.
                sq = sum((v.float() / LOSS_SCALE).pow(2).flatten(1).sum(1) for v in g.values())
                norms = sq.sqrt()
                finite = torch.isfinite(norms)
                factor = torch.where(finite, (args.dp_clip / (norms + 1e-12)).clamp(max=1.0), torch.zeros_like(norms))
                factor = factor / LOSS_SCALE
                for n, v in g.items():
                    acc[n] += torch.einsum("b,b...->...", factor, torch.nan_to_num(v.float(), 0.0, 0.0, 0.0))
                n_clipped += int((norms[finite] > args.dp_clip).sum())
                norm_sum += float(norms[finite].sum())
                pm = parts.float().mean(0) if parts.dim() == 2 else parts.float()
                for k, v in zip(part_names, pm.tolist()):
                    sums[k] = sums.get(k, 0.0) + v * len(y)
                n_seen += len(y)
                correct_t += (t_logits.argmax(1) == y).sum().item()
                correct_s += (s_logits.argmax(1) == y).sum().item()
                del g
            expected = q * n_train
            for n, p in params.items():
                noise = torch.randn(p.shape, generator=noise_gen, device=device) * (args.dp_sigma * args.dp_clip)
                p.grad = (acc[n] + noise) / expected
            opt.step()
            opt.zero_grad(set_to_none=True)
            steps += 1
            if args.max_steps and steps >= args.max_steps:
                break
        if args.max_steps and steps >= args.max_steps:
            break

    delta = {k: (v - start[k]).float().cpu() for k, v in shared_state(student).items()}
    stats = {k: v / max(n_seen, 1) for k, v in sums.items()}
    stats.update(train_acc_t=correct_t / max(n_seen, 1), train_acc_s=correct_s / max(n_seen, 1), steps=steps,
                 update_norm=math.sqrt(sum(float(d.pow(2).sum()) for d in delta.values())),
                 dp_q=q, dp_images=n_seen, dp_frac_clipped=n_clipped / max(n_seen, 1),
                 dp_mean_grad_norm=norm_sum / max(n_seen, 1))
    return delta, stats
