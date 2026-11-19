# Kaggle kernel stub: one round on 4 clients (no defense) to measure client update norms for the clip bound (cifar).
import os
import subprocess

os.environ["FEDKD_DATASET"] = "cifar100"
os.environ["FEDKD_ARGS"] = "--split iid --num_clients 30 --local_epochs 1 --rounds 1 --max_clients 4 --teacher_lr 1e-5 --weight_decay 0.05 --teacher_trainable_blocks 2"
subprocess.run("rm -rf /tmp/repo && git clone --depth 1 https://github.com/mohamedelsayed0001/Nakamoto-Satoshi /tmp/repo",
               shell=True, check=True)
subprocess.run(["python", "/tmp/repo/fedkd_gia/kaggle/run_baseline.py"], check=True)
