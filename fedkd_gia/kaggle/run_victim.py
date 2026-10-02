"""Kaggle entry point: build round-30 victim gradients (needs the baseline kernel output attached)."""
import glob
import os
import subprocess
import sys

subprocess.run("pip install -q 'timm>=1.0.9'", shell=True, check=True)
data_root = os.path.dirname(glob.glob("/kaggle/input/**/driver_imgs_list.csv", recursive=True)[0])
baseline = os.path.dirname(os.path.dirname(glob.glob("/kaggle/input/**/run/ckpt/global_student.pt", recursive=True)[0]))
print("data_root:", data_root, "| baseline:", baseline, flush=True)
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
subprocess.run([sys.executable, "attacks/victim.py", "--data_root", data_root, "--cache_dir", "/tmp/fedkd_cache",
                "--baseline_dir", baseline, "--out_dir", "/kaggle/working/victim",
                "--num_clients", os.environ.get("NUM_CLIENTS", "10")], check=True)
