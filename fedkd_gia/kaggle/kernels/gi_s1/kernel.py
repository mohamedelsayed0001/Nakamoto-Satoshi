# Kaggle kernel stub: GradInversion full run, setting none, 10 clients.
import os
import subprocess

os.environ["ATTACK_CLIENTS"] = "p002,p012,p014,p015,p016,p021,p022,p024,p026,p035"
os.environ["ATTACK_ARGS"] = "--settings none --iterations 4000 --num_seeds 2 --alpha_tv 1 --alpha_noise 0 --log_every 500 --time_budget_hours 11.3"
subprocess.run("rm -rf /tmp/repo && git clone --depth 1 https://github.com/mohamedelsayed0001/Nakamoto-Satoshi /tmp/repo",
               shell=True, check=True)
subprocess.run(["python", "/tmp/repo/fedkd_gia/kaggle/run_attack.py"], check=True)
