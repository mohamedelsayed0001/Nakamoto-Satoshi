"""Attack PSNR gain vs global student accuracy for the client-side Gaussian noise sweep (both datasets)."""
import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

RES = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")
INK, MUTED, GRID, BG = "#1f1f1e", "#6b6a63", "#e4e3de", "#fcfcfb"
COLORS = {"S1": "#2a78d6", "S2": "#eb6834"}  # categorical slots 1-2 of the default validated palette
LABELS = {"none": "no noise", "1e-4": "σ=1e-4 (ε 1.5e9)", "1e-3": "σ=1e-3 (ε 1.5e7)", "1e-2": "σ=1e-2 (ε 1.5e5)"}
PANELS = [
    ("State Farm (frozen teacher)", "SF_frozen", "DP_SF", "statefarm_baseline_frozen", "dp_sf_s1em", ["p002", "p012"],
     {"none": (-12, 8, "right"), "1e-4": (8, 4, "left"), "1e-3": (10, -16, "left"), "1e-2": (8, -18, "left")}),
    ("CIFAR-100 (frozen teacher)", "CIFAR_v3", "DP_C100", "cifar100_baseline_frozen", "dp_c100_s1em", ["c01", "c02"],
     {"none": (-12, 0, "right"), "1e-4": (-12, 4, "right"), "1e-3": (-12, 12, "right"), "1e-2": (-10, -3, "right")}),
]


def last(path):
    return [json.loads(l) for l in open(path) if l.strip()][-1]


def gain(prefix, setting, clients):
    vals = []
    for c in clients:
        r = last(os.path.join(RES, f"{prefix}_{setting}_full", c, "results.jsonl"))
        vals.append(r["psnr_consensus"] - r["psnr_gray_floor"])
    return sum(vals) / len(vals)


def acc(folder):
    return 100 * last(os.path.join(RES, folder, "run", "metrics.jsonl"))["global_student_acc"]


def main():
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), facecolor=BG)
    for ax, (title, atk, dp_atk, base, dp_base, clients, offsets) in zip(axes, PANELS):
        ax.set_facecolor(BG)
        pts = [("none", atk, base)] + [(f"1e-{k}", f"{dp_atk}_s1em{k}", f"{dp_base}{k}") for k in (4, 3, 2)]
        for setting, name in [("S1", "S1: no teacher"), ("S2", "S2: DINOv2 surrogate")]:
            xs = [acc(b) for _, _, b in pts]
            ys = [gain(a, setting, clients) for _, a, _ in pts]
            ax.plot(xs, ys, color=COLORS[setting], lw=2, marker="o", ms=8, markeredgecolor=BG, markeredgewidth=2,
                    label=name, zorder=3)
            if setting == "S2":
                for (lab, _, _), x, y in zip(pts, xs, ys):
                    dx, dy, ha = offsets[lab]
                    ax.annotate(LABELS[lab], (x, y), xytext=(dx, dy), textcoords="offset points", fontsize=9,
                                color=MUTED, ha=ha)
        if dp_atk == "DP_C100" and os.path.exists(os.path.join(RES, "DP_C100_dpsgd_eps8_S1_full")):
            x = acc("dpsgd_c100_eps8_small")
            for setting in ("S1", "S2"):
                ax.scatter([x], [gain("DP_C100_dpsgd_eps8", setting, clients)], marker="D", s=70, color=COLORS[setting],
                           edgecolors=INK, linewidths=1.2, zorder=4)
            ax.annotate("DP-SGD ε=8 (formal)", (x, 0.25), xytext=(10, 14), textcoords="offset points", fontsize=9,
                        color=INK, ha="left", fontweight="bold")
        ax.set_ylim(-0.45, 3.4)
        ax.set_title(title, loc="left", fontsize=12, color=INK)
        ax.set_xlabel("Global student accuracy (%)", color=MUTED)
        ax.set_ylabel("Attack PSNR gain over gray (dB)", color=MUTED)
        ax.grid(color=GRID, lw=1)
        ax.set_axisbelow(True)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        for s in ("left", "bottom"):
            ax.spines[s].set_color(GRID)
        ax.tick_params(colors=MUTED)
        ax.axhline(0, color=MUTED, lw=1, ls=(0, (3, 3)))
        ax.legend(frameon=False, labelcolor=INK, loc="upper left")
    fig.suptitle("Attack leakage vs model accuracy: noise-only (circles) and DP-SGD (diamonds)", x=0.01, ha="left", color=INK,
                 fontsize=12)
    fig.text(0.01, 0.005, "Higher = more leakage, further right = more accurate. Mean of 2 attacked clients, round-30 gradient, "
             "GradInversion 20k it. Noise ε: client-level; DP-SGD ε: per image; 30 rounds, δ=1e-5.",
             fontsize=8, color=MUTED)
    fig.tight_layout(rect=(0, 0.03, 1, 0.95))
    out = os.path.join(RES, "summary_plots", "dp_all_psnr_vs_accuracy.png")
    fig.savefig(out, dpi=150, facecolor=BG)
    print("saved", out)


if __name__ == "__main__":
    main()
