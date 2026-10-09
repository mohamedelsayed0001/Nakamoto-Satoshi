"""Run GradInversion on the victim gradients for every (client, surrogate setting) and score by PSNR.

Inputs: --victim_dir from victim.py (global_student.pt, grad_<cid>.pt). Ground-truth images are
read from the data cache only for evaluation. Results are appended to results.jsonl (one line per
attack, already-finished attacks are skipped on restart) and image grids are saved as PNG.
"""
import argparse
import glob
import json
import os
import sys
import time
from types import SimpleNamespace

import numpy as np
import torch
from PIL import Image
from scipy.optimize import linear_sum_assignment

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from data import load_cache, num_classes, prepare_cache  # noqa: E402
from gradinversion import GradInversion, Surrogate, restore_labels  # noqa: E402
from victim import load_student, to_input  # noqa: E402

HEAD_WEIGHT = "vit.head.weight"


def psnr(a, b):
    """a, b in [0,1], shape (3,H,W)."""
    mse = ((a - b) ** 2).mean().item()
    return 10 * np.log10(1.0 / max(mse, 1e-10))


def score(rec, gt):
    """Order-free PSNR: optimal one-to-one matching of reconstructions to ground-truth images."""
    rec01, gt01 = (rec * 0.5 + 0.5).clamp(0, 1).cpu(), (gt * 0.5 + 0.5).clamp(0, 1).cpu()
    m = np.array([[psnr(r, g) for g in gt01] for r in rec01])
    rows, cols = linear_sum_assignment(-m)
    per_image = [float(m[r, c]) for r, c in zip(rows, cols)]
    return float(np.mean(per_image)), per_image, cols.tolist()


def align(rec, cols):
    """Reorder reconstructions so that column j holds the one matched to ground-truth image j."""
    order = [0] * len(cols)
    for r, c in enumerate(cols):
        order[c] = r
    return rec[order]


def save_grid(path, gt, recs):
    rows = [gt] + recs
    t = torch.cat([torch.cat(list((r * 0.5 + 0.5).clamp(0, 1).cpu()), dim=2) for r in rows], dim=1)
    im = Image.fromarray((t.permute(1, 2, 0).numpy() * 255).astype(np.uint8))
    if gt.shape[-1] < 128:  # small images (CIFAR 32x32): enlarge with nearest-neighbour for viewing only
        im = im.resize((im.width * 4, im.height * 4), Image.NEAREST)
    im.save(path)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data_root", required=True)
    p.add_argument("--cache_dir", default="cache")
    p.add_argument("--victim_dir", required=True)
    p.add_argument("--out_dir", default="runs/gradinversion")
    p.add_argument("--clients", default="", help="comma list; default = all victims")
    p.add_argument("--settings", default="none,dinov2,vitb")
    p.add_argument("--iterations", type=int, default=10000)
    p.add_argument("--num_seeds", type=int, default=4)
    p.add_argument("--lr", type=float, default=0.1)
    p.add_argument("--nuisance_lr", type=float, default=1e-3)
    p.add_argument("--warmup", type=int, default=50)
    p.add_argument("--alpha_grad", type=float, default=1.0)
    p.add_argument("--alpha_tv", type=float, default=1e-4)
    p.add_argument("--alpha_l2", type=float, default=1e-6)
    p.add_argument("--alpha_group", type=float, default=0.01)
    p.add_argument("--alpha_noise", type=float, default=0.01)
    p.add_argument("--group_start", type=float, default=0.25, help="fraction of iterations before R_group")
    p.add_argument("--log_every", type=int, default=500)
    p.add_argument("--time_budget_hours", type=float, default=0)
    p.add_argument("--track_psnr", type=int, default=0,
                   help="log PSNR vs ground truth at every log step (evaluation only)")
    args = p.parse_args()
    cfg = SimpleNamespace(**vars(args))
    t0 = time.time()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.out_dir, exist_ok=True)
    with open(os.path.join(args.out_dir, "config.json"), "w") as f:
        json.dump(vars(args), f, indent=2)
    prepare_cache(args.data_root, args.cache_dir)
    images, _ = load_cache(args.cache_dir)

    student = load_student(os.path.join(args.victim_dir, "global_student.pt"), device, num_classes(args.cache_dir))
    victims = sorted(os.path.basename(f)[5:-3] for f in glob.glob(os.path.join(args.victim_dir, "grad_*.pt")))
    clients = args.clients.split(",") if args.clients else victims
    res_path = os.path.join(args.out_dir, "results.jsonl")
    done = set()
    if os.path.exists(res_path):
        done = {(r["client"], r["setting"]) for r in map(json.loads, open(res_path))}

    surrogates = {}
    attack_times = []
    for setting in args.settings.split(","):
        for cid in clients:
            if (cid, setting) in done:
                continue
            if args.time_budget_hours and attack_times and \
                    time.time() - t0 + np.mean(attack_times) > args.time_budget_hours * 3600:
                print("time budget reached; stopping", flush=True)
                return
            a0 = time.time()
            if setting != "none" and setting not in surrogates:
                surrogates.clear()
                surrogates[setting] = Surrogate(setting).to(device)
            v = torch.load(os.path.join(args.victim_dir, f"grad_{cid}.pt"), map_location=device)
            gt = to_input(images, v["indices"]).to(device)
            true_labels = v["labels"]
            labels = restore_labels(v["grads"], HEAD_WEIGHT, len(true_labels))
            label_acc = len(set(labels.tolist()) & set(true_labels)) / len(true_labels)
            print(f"[{setting}] client {cid}: true labels {sorted(true_labels)} restored {sorted(labels.tolist())}",
                  flush=True)

            attack = GradInversion(student, v["grads"], labels.sort()[0], surrogates.get(setting), cfg)
            seeds = list(range(args.num_seeds))
            monitor = None
            if args.track_psnr:
                def monitor(xs, losses, gt=gt):
                    return {"psnr_consensus": score(torch.stack(xs).mean(0), gt)[0],
                            "psnr_best_seed": score(xs[int(np.argmin(losses))], gt)[0]}
            finals, history = attack.run(gt.shape, seeds, args.log_every, log=lambda s: print(s, flush=True),
                                         monitor=monitor)
            final_losses = history[-1]["grad_loss"]
            best = int(np.argmin(final_losses))
            consensus = torch.stack(finals).mean(0)

            psnr_cons, per_cons, cols_cons = score(consensus, gt)
            psnr_best, per_best, cols_best = score(finals[best], gt)
            seed_psnrs = [score(x, gt)[0] for x in finals]
            # No-information floor: a uniform gray image (the mean of the input range).
            psnr_gray = score(torch.zeros_like(gt), gt)[0]
            rec = {"client": cid, "setting": setting, "label_acc": label_acc,
                   "restored_labels": labels.tolist(), "true_labels": true_labels,
                   "psnr_consensus": psnr_cons, "psnr_best_seed": psnr_best, "psnr_per_seed": seed_psnrs, "psnr_gray_floor": psnr_gray,
                   "psnr_per_image_consensus": per_cons, "psnr_per_image_best_seed": per_best,
                   "final_grad_loss": final_losses, "seconds": time.time() - a0, "history": history}
            save_grid(os.path.join(args.out_dir, f"{setting}_{cid}.png"), gt,
                      [align(consensus, cols_cons), align(finals[best], cols_best)])
            torch.save({"consensus": consensus.cpu(), "seeds": [x.cpu() for x in finals]},
                       os.path.join(args.out_dir, f"{setting}_{cid}_rec.pt"))
            with open(res_path, "a") as f:
                f.write(json.dumps(rec) + "\n")
            attack_times.append(time.time() - a0)
            print(f"RESULT [{setting}] client {cid}: PSNR consensus {psnr_cons:.2f} dB | best seed {psnr_best:.2f} dB"
                  f" | gray floor {psnr_gray:.2f} dB | label acc {label_acc:.2f} | {attack_times[-1] / 60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
