# Kaggle kernel stub: GradInversion tuning (S1, client p002): TV weight x Langevin noise.
import os
import subprocess

os.environ["ATTACK_JOBS"] = "[[\"tv1e-2_noise0.01\", \"p002\", \"--settings none --iterations 2000 --num_seeds 2 --log_every 250 --alpha_tv 1e-2 --alpha_noise 0.01\"], [\"tv1e-2_noise0\", \"p002\", \"--settings none --iterations 2000 --num_seeds 2 --log_every 250 --alpha_tv 1e-2 --alpha_noise 0\"], [\"tv1e-1_noise0.01\", \"p002\", \"--settings none --iterations 2000 --num_seeds 2 --log_every 250 --alpha_tv 1e-1 --alpha_noise 0.01\"], [\"tv1e-1_noise0\", \"p002\", \"--settings none --iterations 2000 --num_seeds 2 --log_every 250 --alpha_tv 1e-1 --alpha_noise 0\"], [\"tv1_noise0.01\", \"p002\", \"--settings none --iterations 2000 --num_seeds 2 --log_every 250 --alpha_tv 1 --alpha_noise 0.01\"], [\"tv1_noise0\", \"p002\", \"--settings none --iterations 2000 --num_seeds 2 --log_every 250 --alpha_tv 1 --alpha_noise 0\"]]"
subprocess.run("rm -rf /tmp/repo && git clone --depth 1 https://github.com/mohamedelsayed0001/Nakamoto-Satoshi /tmp/repo",
               shell=True, check=True)
subprocess.run(["python", "/tmp/repo/fedkd_gia/kaggle/run_attack.py"], check=True)
