# Kaggle kernel stub: FedKD baseline (CIFAR-100 v3) with the client-side
# Gaussian defense (clip + noise on every released student update), sigma = 1e-2.
import os
import subprocess

os.environ["FEDKD_DATASET"] = "cifar100"
os.environ["FEDKD_ARGS"] = "--split iid --num_clients 30 --local_epochs 1 --rounds 30 --teacher_lr 1e-5 --weight_decay 0.05 --teacher_trainable_blocks 2 --defense gaussian --clip 2 --sigma 1e-2"
subprocess.run("rm -rf /tmp/repo && git clone --depth 1 https://github.com/mohamedelsayed0001/Nakamoto-Satoshi /tmp/repo",
               shell=True, check=True)
subprocess.run(["python", "/tmp/repo/fedkd_gia/kaggle/run_baseline.py"], check=True)
