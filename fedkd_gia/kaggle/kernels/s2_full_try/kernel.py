# Kaggle kernel stub: S2_full - GradInversion with DINOv2 surrogate, 20k iterations, clients p002,p021.
import os
import subprocess

os.environ["ATTACK_CLIENTS"] = "p002,p021"
os.environ["ATTACK_ARGS"] = "--settings dinov2 --iterations 20000 --num_seeds 2 --alpha_tv 1 --alpha_noise 0 --log_every 1000 --track_psnr 1 --time_budget_hours 11.5"
subprocess.run("rm -rf /tmp/repo && git clone --depth 1 https://github.com/mohamedelsayed0001/Nakamoto-Satoshi /tmp/repo",
               shell=True, check=True)
subprocess.run(["python", "/tmp/repo/fedkd_gia/kaggle/run_attack.py"], check=True)
