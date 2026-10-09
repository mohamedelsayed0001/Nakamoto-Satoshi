# Kaggle kernel stub: FedKD baseline on CIFAR-100, 30 IID clients (80/20 local split), 1 local epoch, 30 rounds.
import os
import subprocess

os.environ["FEDKD_DATASET"] = "cifar100"
os.environ["FEDKD_ARGS"] = "--split iid --num_clients 30 --local_epochs 1 --rounds 30"
subprocess.run("rm -rf /tmp/repo && git clone --depth 1 https://github.com/mohamedelsayed0001/Nakamoto-Satoshi /tmp/repo",
               shell=True, check=True)
subprocess.run(["python", "/tmp/repo/fedkd_gia/kaggle/run_baseline.py"], check=True)
