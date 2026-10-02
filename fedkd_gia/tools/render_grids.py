"""Re-render attack image grids locally from saved reconstructions (<setting>_<client>_rec.pt).

Row 1: actual batch, row 2: consensus reconstruction, row 3: best-seed reconstruction, with each
reconstruction placed under the ground-truth image it was matched to.

usage: python tools/render_grids.py --exp_dir results/<exp> --victim_dir <victim dir> --data_root ... --cache_dir ...
"""
import argparse
import glob
import json
import os
import sys

import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "attacks"))
from data import load_cache, prepare_cache  # noqa: E402
from run_gradinversion import align, save_grid, score  # noqa: E402
from victim import to_input  # noqa: E402


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--exp_dir", required=True)
    p.add_argument("--victim_dir", required=True)
    p.add_argument("--data_root", required=True)
    p.add_argument("--cache_dir", required=True)
    args = p.parse_args()
    prepare_cache(args.data_root, args.cache_dir)
    images, _ = load_cache(args.cache_dir)
    victims = json.load(open(os.path.join(args.victim_dir, "victims.json")))
    for res in glob.glob(os.path.join(args.exp_dir, "**", "results.jsonl"), recursive=True):
        d = os.path.dirname(res)
        for r in map(json.loads, open(res)):
            rec = torch.load(os.path.join(d, f"{r['setting']}_{r['client']}_rec.pt"))
            gt = to_input(images, victims[r["client"]]["indices"])
            seeds = rec["seeds"]
            best = seeds[int(np.argmin(r["final_grad_loss"]))]
            _, _, cols_c = score(rec["consensus"], gt)
            _, _, cols_b = score(best, gt)
            out = os.path.join(d, f"{r['setting']}_{r['client']}.png")
            save_grid(out, gt, [align(rec["consensus"], cols_c), align(best, cols_b)])
            print("wrote", out)


if __name__ == "__main__":
    main()
