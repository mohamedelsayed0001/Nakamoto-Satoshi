# Kaggle kernel stub: clone the repo and run the FedKD baseline in "baseline" mode.
import os
import subprocess

os.environ["FEDKD_MODE"] = "full"
subprocess.run("rm -rf /tmp/repo && git clone --depth 1 https://github.com/mohamedelsayed0001/Nakamoto-Satoshi /tmp/repo",
               shell=True, check=True)
subprocess.run(["python", "/tmp/repo/fedkd_gia/kaggle/run_baseline.py"], check=True)
