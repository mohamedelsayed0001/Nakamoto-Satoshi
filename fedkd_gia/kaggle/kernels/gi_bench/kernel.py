# Kaggle kernel stub: clone the repo and run run_attack.py.
import os
import subprocess

os.environ["ATTACK_CLIENTS"] = "p002,p012"
os.environ["ATTACK_ARGS"] = "--iterations 1000 --num_seeds 4 --log_every 100"
subprocess.run("rm -rf /tmp/repo && git clone --depth 1 https://github.com/mohamedelsayed0001/Nakamoto-Satoshi /tmp/repo",
               shell=True, check=True)
subprocess.run(["python", "/tmp/repo/fedkd_gia/kaggle/run_attack.py"], check=True)
