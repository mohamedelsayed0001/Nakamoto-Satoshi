# Kaggle kernel stub: FedKD on CIFAR-100 v3 with record-level DP-SGD inside every client
# (Poisson batches of expected size 64, per-image clip 1, noise calibrated to epsilon = 1 at delta = 1e-5 per client).
import os
import subprocess

os.environ["FEDKD_DATASET"] = "cifar100"
os.environ["FEDKD_ARGS"] = "--split iid --num_clients 30 --local_epochs 1 --rounds 30 --teacher_lr 1e-5 --weight_decay 0.05 --teacher_trainable_blocks 2 --defense dpsgd --dp_epsilon 1 --dp_clip 1 --dp_batch_size 64 --dp_chunk 8"
subprocess.run("rm -rf /tmp/repo && git clone --depth 1 https://github.com/mohamedelsayed0001/Nakamoto-Satoshi /tmp/repo",
               shell=True, check=True)
subprocess.run(["python", "/tmp/repo/fedkd_gia/kaggle/run_baseline.py"], check=True)
