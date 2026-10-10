# Kaggle kernel stub: State Farm FedKD baseline, 21 driver clients (5 unseen drivers held out) with an
# 80/20 local split per client; teacher lr 1e-5, AdamW weight decay 0.05, only the last 2 teacher blocks
# (+ norm, head) trained. 30 rounds x 5 local epochs.
import os
import subprocess

os.environ["FEDKD_ARGS"] = "--client_test_frac 0.2 --teacher_trainable_blocks 2 --teacher_lr 1e-5 --weight_decay 0.05"
subprocess.run("rm -rf /tmp/repo && git clone --depth 1 https://github.com/mohamedelsayed0001/Nakamoto-Satoshi /tmp/repo",
               shell=True, check=True)
subprocess.run(["python", "/tmp/repo/fedkd_gia/kaggle/run_baseline.py"], check=True)
