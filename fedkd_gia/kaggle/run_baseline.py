"""Kaggle entry point (run from a clone of the repo by kernels/*/kernel.py): finds the
competition data, runs/resumes FedKD training.

Resume: if this kernel's own previous output (attached as a kernel source) contains a run with
state.json, training continues from it. Outputs go to /kaggle/working/run.
"""
import glob
import os
import subprocess
import sys

MODE = os.environ.get("FEDKD_MODE", "full")  # "bench" = short speed test, "full" = baseline run
BUDGET_HOURS = os.environ.get("FEDKD_BUDGET_HOURS", "11.2")


def sh(cmd):
    print("+", cmd, flush=True)
    subprocess.run(cmd, shell=True, check=True)


sh("nvidia-smi")
sh("pip install -q 'timm>=1.0.9'")

data_root = os.path.dirname(glob.glob("/kaggle/input/**/driver_imgs_list.csv", recursive=True)[0])
prev = [os.path.dirname(p) for p in glob.glob("/kaggle/input/**/run/state.json", recursive=True)]
print("data_root:", data_root, "| previous runs:", prev, flush=True)

args = [
    sys.executable, "train.py",
    "--data_root", data_root,
    "--cache_dir", "/tmp/fedkd_cache",
    "--out_dir", "/kaggle/working/run",
    "--num_workers", "2",
    "--time_budget_hours", BUDGET_HOURS,
]
if prev:
    args += ["--resume_from", prev[0]]
if MODE == "bench":
    args += ["--rounds", "1", "--max_clients", "4", "--local_epochs", "1", "--max_steps", "60", "--max_eval", "64"]

os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
subprocess.run(args, check=True)
