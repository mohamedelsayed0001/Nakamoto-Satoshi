# Kaggle kernel stub: CIFAR-100 GradInversion 20k iterations, S2 (DINOv2) then S1 (no teacher), clients c02, c03.
import os
import subprocess

os.environ["ATTACK_JOBS"] = "[[\"S2_full/c02\", \"c02\", \"--settings dinov2 --iterations 20000 --num_seeds 2 --alpha_tv 1 --alpha_noise 0 --log_every 1000 --track_psnr 1 --time_budget_hours 11.5\"], [\"S2_full/c03\", \"c03\", \"--settings dinov2 --iterations 20000 --num_seeds 2 --alpha_tv 1 --alpha_noise 0 --log_every 1000 --track_psnr 1 --time_budget_hours 11.5\"], [\"S1_full/c02\", \"c02\", \"--settings none --iterations 20000 --num_seeds 2 --alpha_tv 1 --alpha_noise 0 --log_every 1000 --track_psnr 1 --time_budget_hours 11.5\"], [\"S1_full/c03\", \"c03\", \"--settings none --iterations 20000 --num_seeds 2 --alpha_tv 1 --alpha_noise 0 --log_every 1000 --track_psnr 1 --time_budget_hours 11.5\"]]"
subprocess.run("rm -rf /tmp/repo && git clone --depth 1 https://github.com/mohamedelsayed0001/Nakamoto-Satoshi /tmp/repo",
               shell=True, check=True)
subprocess.run(["python", "/tmp/repo/fedkd_gia/kaggle/run_attack.py"], check=True)
