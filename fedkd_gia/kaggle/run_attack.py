"""Kaggle entry point: GradInversion on victim gradients, one worker process per GPU.

Either
  ATTACK_CLIENTS: comma list of clients (split across the GPUs), ATTACK_ARGS: extra arguments
    -> results in /kaggle/working/gi/gpu<k>/
or
  ATTACK_JOBS: JSON list of [name, clients, args] -> results in /kaggle/working/gi/<name>/
    jobs are dealt round-robin to the GPUs; each GPU runs its jobs one after another.
"""
import glob
import json
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
n_gpu = max(torch.cuda.device_count(), 1)

if os.environ.get("ATTACK_JOBS"):
    jobs = [(name, clients, shlex.split(a)) for name, clients, a in json.loads(os.environ["ATTACK_JOBS"])]
else:
    clients = [c for c in os.environ["ATTACK_CLIENTS"].split(",") if c]
    extra = shlex.split(os.environ.get("ATTACK_ARGS", ""))
    jobs = [(f"gpu{k}", ",".join(clients[k::n_gpu]), extra) for k in range(n_gpu) if clients[k::n_gpu]]
print("data_root:", data_root, "| victims:", victim_dir, "| jobs:", jobs, flush=True)

os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# Build the image cache and download surrogate weights once, before the per-GPU workers start.
subprocess.run([sys.executable, "-c", f"from data import prepare_cache; prepare_cache({data_root!r}, '/tmp/fedkd_cache'); "
                "import sys; sys.path.insert(0, 'attacks'); from gradinversion import Surrogate; "
                "[Surrogate(s) for s in ('dinov2', 'vitb')]"], check=True)


def command(name, clients, extra):
    return [sys.executable, "attacks/run_gradinversion.py", "--data_root", data_root, "--cache_dir", "/tmp/fedkd_cache",
            "--victim_dir", victim_dir, "--out_dir", f"/kaggle/working/gi/{name}", "--clients", clients] + extra


procs = []
for k in range(n_gpu):
    mine = jobs[k::n_gpu]
    if not mine:
        continue
    script = " && ".join(shlex.join(command(*j)) for j in mine)
    procs.append(subprocess.Popen(script, shell=True, env=dict(os.environ, CUDA_VISIBLE_DEVICES=str(k))))
codes = [p.wait() for p in procs]
sys.exit(max(codes))
