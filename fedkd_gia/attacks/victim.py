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
from data import NUM_CLASSES, load_cache, prepare_cache  # noqa: E402
from fedkd import fedkd_loss  # noqa: E402
from models import HiddenProjector, build_student, build_teacher  # noqa: E402


def to_input(images, idx):
    x = torch.from_numpy(np.stack([np.array(images[i]) for i in idx])).permute(0, 3, 1, 2).float() / 255
    return (x - 0.5) / 0.5


def load_student(global_state_path, device):
    student = build_student(NUM_CLASSES, pretrained=True, freeze_embeddings=True)
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
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    prepare_cache(args.data_root, args.cache_dir)
    images, meta = load_cache(args.cache_dir)
    labels = meta.label.values
    with open(os.path.join(args.baseline_dir, "splits.json")) as f:
        splits = json.load(f)
    clients = args.clients.split(",") if args.clients else sorted(splits)[: args.num_clients]

    os.makedirs(args.out_dir, exist_ok=True)
    gpath = os.path.join(args.baseline_dir, "ckpt", "global_student.pt")
    student = load_student(gpath, device)
    torch.save(torch.load(gpath, map_location="cpu"), os.path.join(args.out_dir, "global_student.pt"))
    teacher = build_teacher(NUM_CLASSES, pretrained=False).to(device)
    projector = HiddenProjector(student.depth, student.embed_dim, teacher.embed_dim).to(device)
    params = {n: p_ for n, p_ in student.named_parameters() if p_.requires_grad}

    index = {}
    for ci, cid in enumerate(clients):
        cs = torch.load(os.path.join(args.baseline_dir, "ckpt", f"client_{cid}.pt"), map_location="cpu")
        teacher.load_state_dict(cs["teacher"])
        projector.load_state_dict(cs["projector"])
        teacher.eval(); projector.eval()

        rng = np.random.default_rng(args.seed * 1000 + ci)
        train_idx = np.array(splits[cid]["train"])
        chosen_labels = rng.choice(NUM_CLASSES, size=args.batch_size, replace=False)
        idx = [int(rng.choice(train_idx[labels[train_idx] == c])) for c in chosen_labels]

        x = to_input(images, idx).to(device)
        y = torch.as_tensor(labels[idx], device=device)
        loss, _, _, parts = fedkd_loss(teacher(x, return_internals=True), student(x, return_internals=True), projector, y)
        grads = torch.autograd.grad(loss, list(params.values()))
        torch.save({"indices": idx, "labels": y.cpu().tolist(), "loss_parts": parts,
                    "grads": {n: g.detach().cpu() for n, g in zip(params, grads)}},
                   os.path.join(args.out_dir, f"grad_{cid}.pt"))
        index[cid] = {"indices": idx, "labels": y.cpu().tolist(), "loss_parts": parts}
        print(cid, json.dumps(index[cid]), flush=True)

    with open(os.path.join(args.out_dir, "victims.json"), "w") as f:
        json.dump(index, f, indent=1)


if __name__ == "__main__":
    main()
