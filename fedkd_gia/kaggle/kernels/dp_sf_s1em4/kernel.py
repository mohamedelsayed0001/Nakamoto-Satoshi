# Kaggle kernel stub: FedKD baseline (State Farm frozen teacher) with the client-side
# Gaussian defense (clip + noise on every released student update), sigma = 1e-4.
import os
import subprocess

os.environ["FEDKD_ARGS"] = "--client_test_frac 0.2 --teacher_trainable_blocks 2 --teacher_lr 1e-5 --weight_decay 0.05 --defense gaussian --clip 4 --sigma 1e-4"
subprocess.run("rm -rf /tmp/repo && git clone --depth 1 https://github.com/mohamedelsayed0001/Nakamoto-Satoshi /tmp/repo",
               shell=True, check=True)
subprocess.run(["python", "/tmp/repo/fedkd_gia/kaggle/run_baseline.py"], check=True)
