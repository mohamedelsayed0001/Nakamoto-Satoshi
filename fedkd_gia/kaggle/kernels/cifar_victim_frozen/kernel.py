# Kaggle kernel stub: clone the repo and run run_victim.py.
import os
import subprocess

os.environ["NUM_CLIENTS"] = "6"
subprocess.run("rm -rf /tmp/repo && git clone --depth 1 https://github.com/mohamedelsayed0001/Nakamoto-Satoshi /tmp/repo",
               shell=True, check=True)
subprocess.run(["python", "/tmp/repo/fedkd_gia/kaggle/run_victim.py"], check=True)
