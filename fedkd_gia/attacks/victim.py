"""Victim side: produce the ground-truth student gradient a FedKD client would upload for one batch.

For each attacked client we take the round-30 global student and that client's round-30 private
teacher + hidden projector, draw one batch of `batch_size` images with distinct labels from the
client's training split (GradInversion's label restoration assumes non-repeating labels), and
compute the gradient of the full FedKD loss w.r.t. the student's shared (trainable) parameters.

Only the output of this script (global student, gradients, batch indices) is given to the
attacker; the private teachers never leave this step.
"""
import argparse
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data import NUM_CLASSES, load_cache, num_classes, prepare_cache  # noqa: E402
from dp import clip_and_noise, flat_norm  # noqa: E402
from dpsgd import LOSS_SCALE, _trainable, make_sample_grad  # noqa: E402
from fedkd import fedkd_loss  # noqa: E402
from models import HiddenProjector, build_student, build_teacher  # noqa: E402


def to_input(images, idx):
    x = torch.from_numpy(np.stack([np.array(images[i]) for i in idx])).permute(0, 3, 1, 2).float() / 255
    return (x - 0.5) / 0.5


def load_student(global_state_path, device, n_classes=NUM_CLASSES, trainable_blocks=-1):
    student = build_student(n_classes, pretrained=True, freeze_embeddings=True, trainable_blocks=trainable_blocks)
    student.load_state_dict(torch.load(global_state_path, map_location="cpu"), strict=False)
    return student.to(device).eval()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data_root", required=True)
    p.add_argument("--cache_dir", default="cache")
    p.add_argument("--baseline_dir", required=True, help="baseline run dir with ckpt/ and splits.json")
    p.add_argument("--out_dir", default="victim")
    p.add_argument("--clients", default="", help="comma list; default = first --num_clients drivers")
    p.add_argument("--num_clients", type=int, default=10)
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--defense", choices=["none", "gaussian", "dpsgd"], default="none")
    p.add_argument("--clip", type=float, default=1.0, help="L2 bound on the released student gradient")
    p.add_argument("--sigma", type=float, default=0.0, help="noise multiplier (noise std = sigma * clip)")
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    prepare_cache(args.data_root, args.cache_dir)
    images, meta = load_cache(args.cache_dir)
    nc = num_classes(args.cache_dir)
    labels = meta.label.values
    with open(os.path.join(args.baseline_dir, "splits.json")) as f:
        splits = json.load(f)
    clients = args.clients.split(",") if args.clients else sorted(splits)[: args.num_clients]

    os.makedirs(args.out_dir, exist_ok=True)
    with open(os.path.join(args.baseline_dir, "config.json")) as f:
        train_cfg = json.load(f)
    gpath = os.path.join(args.baseline_dir, "ckpt", "global_student.pt")
    # Only the parameters the clients actually train and share are in the released gradient.
    student = load_student(gpath, device, nc, train_cfg.get("student_trainable_blocks", -1))
    torch.save(torch.load(gpath, map_location="cpu"), os.path.join(args.out_dir, "global_student.pt"))
    if args.defense == "dpsgd":
        # One DP-SGD step exactly as in training: same sigma, per-image clip and jointly clipped parameter set.
        args.sigma, args.clip = train_cfg["dp_sigma"], train_cfg["dp_clip"]
        print(f"DP-SGD victim step: sigma={args.sigma:.4f} clip={args.clip}", flush=True)
    teacher = build_teacher(nc, pretrained=False,
                            trainable_blocks=train_cfg.get("teacher_trainable_blocks", -1)).to(device)
    projector = HiddenProjector(student.depth, student.embed_dim, teacher.embed_dim).train_last(
        train_cfg.get("projector_trainable_layers", -1)).to(device)
    params = {n: p_ for n, p_ in student.named_parameters() if p_.requires_grad}

    index = {}
    for ci, cid in enumerate(clients):
        cs = torch.load(os.path.join(args.baseline_dir, "ckpt", f"client_{cid}.pt"), map_location="cpu")
        teacher.load_state_dict(cs["teacher"])
        projector.load_state_dict(cs["projector"])
        teacher.eval(); projector.eval()

        rng = np.random.default_rng(args.seed * 1000 + ci)
        train_idx = np.array(splits[cid]["train"])
        chosen_labels = rng.choice(nc, size=args.batch_size, replace=False)
        idx = [int(rng.choice(train_idx[labels[train_idx] == c])) for c in chosen_labels]

        x = to_input(images, idx).to(device)
        y = torch.as_tensor(labels[idx], device=device)
        loss, _, _, parts = fedkd_loss(teacher(x, return_internals=True), student(x, return_internals=True), projector, y)
        if args.defense == "dpsgd":
            allp = _trainable({"t": teacher, "s": student, "p": projector})
            g, _ = make_sample_grad(teacher, student, projector, False, device.type)(
                {n: v.detach() for n, v in allp.items()}, x, y)
            norms = sum((v.float() / LOSS_SCALE).pow(2).flatten(1).sum(1) for v in g.values()).sqrt()
            factor = (args.clip / (norms + 1e-12)).clamp(max=1.0) / LOSS_SCALE
            gen = torch.Generator(device=device).manual_seed(args.seed * 1000 + ci + 11)
            grads = {}
            for n in params:  # released part: the student's shared parameters
                summed = torch.einsum("b,b...->...", factor, g["s." + n].float())
                noise = torch.randn(summed.shape, generator=gen, device=device) * (args.sigma * args.clip)
                grads[n] = ((summed + noise) / len(y)).cpu()
            grad_norm = float(norms.mean())
            del g
        else:
            grads = torch.autograd.grad(loss, list(params.values()))
            grads = {n: g.detach().cpu() for n, g in zip(params, grads)}
        # The released gradient gets the same client-side defense as training (no-op unless --defense gaussian),
        # so the attack is run against what a protected client would actually upload.
        if args.defense == "dpsgd":
            pass
        elif args.defense == "gaussian":
            grads, grad_norm = clip_and_noise(grads, args.clip, args.sigma, torch.Generator().manual_seed(args.seed * 1000 + ci))
        else:
            grad_norm = flat_norm(grads)
        torch.save({"indices": idx, "labels": y.cpu().tolist(), "loss_parts": parts, "grads": grads},
                   os.path.join(args.out_dir, f"grad_{cid}.pt"))
        index[cid] = {"indices": idx, "labels": y.cpu().tolist(), "loss_parts": parts, "grad_norm": grad_norm,
                      "defense": args.defense, "clip": args.clip, "sigma": args.sigma}
        print(cid, json.dumps(index[cid]), flush=True)

    with open(os.path.join(args.out_dir, "victims.json"), "w") as f:
        json.dump(index, f, indent=1)


if __name__ == "__main__":
    main()
