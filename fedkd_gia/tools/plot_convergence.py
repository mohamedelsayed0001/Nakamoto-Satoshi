"""Plot PSNR gain over the gray floor vs. iteration for runs made with --track_psnr 1.

usage: python tools/plot_convergence.py --exp_dir results/S2_full
Writes <exp_dir>/plots/psnr_gain_vs_iteration.png (one panel per client) and
<exp_dir>/plots/psnr_gain_all_clients.png (all clients in one panel).
"""
import argparse
import glob
import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300"]
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"


def tidy(ax):
    ax.grid(True, color=GRID, linewidth=0.8)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.tick_params(colors=MUTED, labelsize=8)
    ax.axhline(0, color=MUTED, linestyle=":", linewidth=1)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--exp_dir", required=True)
    args = p.parse_args()
    runs = {}
    for f in glob.glob(os.path.join(args.exp_dir, "**", "results.jsonl"), recursive=True):
        for line in open(f):
            r = json.loads(line)
            runs[r["client"]] = r
    runs = dict(sorted(runs.items()))
    out = os.path.join(args.exp_dir, "plots")
    os.makedirs(out, exist_ok=True)

    n = len(runs)
    cols = 3
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 3 * rows), sharex=True, sharey=True, squeeze=False)
    for ax, (cid, r) in zip(axes.flat, runs.items()):
        it = [h["it"] for h in r["history"]]
        floor = r["psnr_gray_floor"]
        ax.plot(it, [h["psnr_consensus"] - floor for h in r["history"]], color=COLORS[0], linewidth=2,
                label="seeds averaged")
        ax.plot(it, [h["psnr_best_seed"] - floor for h in r["history"]], color=COLORS[1], linewidth=2,
                linestyle="--", label="best seed")
        tidy(ax)
        ax.set_title(f"client {cid}", color=INK, fontsize=10, loc="left")
    for ax in list(axes.flat)[n:]:
        ax.axis("off")
    for ax in axes[-1]:
        ax.set_xlabel("iteration", color=MUTED, fontsize=9)
    for ax in axes[:, 0]:
        ax.set_ylabel("PSNR minus gray floor (dB)", color=MUTED, fontsize=9)
    axes.flat[0].legend(fontsize=8, frameon=False)
    fig.suptitle("GradInversion S2 (DINOv2 surrogate), 20k iterations: PSNR gain over gray image",
                 color=INK, fontsize=11, x=0.01, ha="left")
    fig.tight_layout()
    fig.savefig(os.path.join(out, "psnr_gain_vs_iteration.png"), dpi=130)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 4.5))
    for color, (cid, r) in zip(COLORS, runs.items()):
        it = [h["it"] for h in r["history"]]
        y = [h["psnr_consensus"] - r["psnr_gray_floor"] for h in r["history"]]
        ax.plot(it, y, color=color, linewidth=2, label=cid)
        ax.annotate(cid, (it[-1], y[-1]), xytext=(4, 0), textcoords="offset points", color=INK, fontsize=8,
                    va="center")
    tidy(ax)
    ax.set_xlabel("iteration", color=MUTED, fontsize=9)
    ax.set_ylabel("PSNR minus gray floor (dB), seeds averaged", color=MUTED, fontsize=9)
    ax.set_title("All clients: PSNR gain over gray image vs iteration", color=INK, fontsize=11, loc="left")
    ax.legend(fontsize=8, frameon=False, ncol=3)
    fig.tight_layout()
    fig.savefig(os.path.join(out, "psnr_gain_all_clients.png"), dpi=130)
    plt.close(fig)
    print("saved plots to", out)


if __name__ == "__main__":
    main()
