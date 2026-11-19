# Kaggle kernel stub: one round on 4 clients (no defense) to measure client update norms for the clip bound (statefarm).
import os
import subprocess

os.environ["FEDKD_ARGS"] = "--client_test_frac 0.2 --teacher_trainable_blocks 2 --teacher_lr 1e-5 --weight_decay 0.05 --num_workers 0 --rounds 1 --max_clients 4"
subprocess.run("rm -rf /tmp/repo && git clone --depth 1 https://github.com/mohamedelsayed0001/Nakamoto-Satoshi /tmp/repo",
               shell=True, check=True)
subprocess.run(["python", "/tmp/repo/fedkd_gia/kaggle/run_baseline.py"], check=True)
