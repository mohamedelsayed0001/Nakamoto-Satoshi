"""Kaggle entry point: GradInversion on victim gradients, one process per GPU.

ATTACK_CLIENTS: comma list of clients for this kernel (split across the GPUs).
ATTACK_ARGS:    extra arguments for attacks/run_gradinversion.py.
Results go to /kaggle/working/gi/gpu<k>/.
"""
import glob
import os
import shlex
import subprocess
import sys

import torch

subprocess.run("pip install -q 'timm>=1.0.9'", shell=True, check=True)
subprocess.Popen("nvidia-smi --query-gpu=timestamp,index,memory.used,memory.total,utilization.gpu "
                 "--format=csv,noheader -l 120 | sed -u 's/^/VRAM /'", shell=True)
data_root = os.path.dirname(glob.glob("/kaggle/input/**/driver_imgs_list.csv", recursive=True)[0])
victim_dir = os.path.dirname(glob.glob("/kaggle/input/**/victims.json", recursive=True)[0])
clients = [c for c in os.environ["ATTACK_CLIENTS"].split(",") if c]
extra = shlex.split(os.environ.get("ATTACK_ARGS", ""))
print("data_root:", data_root, "| victims:", victim_dir, "| clients:", clients, flush=True)

os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# Build the image cache and download surrogate weights once, before the per-GPU workers start.
subprocess.run([sys.executable, "-c", f"from data import prepare_cache; prepare_cache({data_root!r}, '/tmp/fedkd_cache'); "
                "import sys; sys.path.insert(0, 'attacks'); from gradinversion import Surrogate; "
                "[Surrogate(s) for s in ('dinov2', 'vitb')]"], check=True)
n_gpu = max(torch.cuda.device_count(), 1)
procs = []
for k in range(n_gpu):
    part = clients[k::n_gpu]
    if not part:
        continue
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(k))
    cmd = [sys.executable, "attacks/run_gradinversion.py", "--data_root", data_root, "--cache_dir", "/tmp/fedkd_cache",
           "--victim_dir", victim_dir, "--out_dir", f"/kaggle/working/gi/gpu{k}", "--clients", ",".join(part)] + extra
    procs.append(subprocess.Popen(cmd, env=env))
codes = [p.wait() for p in procs]
sys.exit(max(codes))
