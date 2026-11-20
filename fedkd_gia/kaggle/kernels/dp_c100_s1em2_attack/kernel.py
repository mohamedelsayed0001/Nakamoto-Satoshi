# Kaggle kernel stub: CIFAR-100 v3, defense sigma=1e-2: build the clipped+noised victim gradient (clip 40),
# then GradInversion S2 and S1 at 20k iterations on c01, c02.
import os
import subprocess

os.environ["VICTIM_ARGS"] = "--defense gaussian --clip 40 --sigma 1e-2 --clients c01,c02"
os.environ["VICTIM_DIR"] = "/kaggle/working/victim"
os.environ["ATTACK_JOBS"] = "[[\"S2_full/c01\", \"c01\", \"--settings dinov2 --iterations 20000 --num_seeds 2 --alpha_tv 1 --alpha_noise 0 --log_every 1000 --track_psnr 1 --time_budget_hours 11\"], [\"S2_full/c02\", \"c02\", \"--settings dinov2 --iterations 20000 --num_seeds 2 --alpha_tv 1 --alpha_noise 0 --log_every 1000 --track_psnr 1 --time_budget_hours 11\"], [\"S1_full/c01\", \"c01\", \"--settings none --iterations 20000 --num_seeds 2 --alpha_tv 1 --alpha_noise 0 --log_every 1000 --track_psnr 1 --time_budget_hours 11\"], [\"S1_full/c02\", \"c02\", \"--settings none --iterations 20000 --num_seeds 2 --alpha_tv 1 --alpha_noise 0 --log_every 1000 --track_psnr 1 --time_budget_hours 11\"]]"
subprocess.run("rm -rf /tmp/repo && git clone --depth 1 https://github.com/mohamedelsayed0001/Nakamoto-Satoshi /tmp/repo",
               shell=True, check=True)
subprocess.run(["python", "/tmp/repo/fedkd_gia/kaggle/run_victim.py"], check=True)
subprocess.run(["python", "/tmp/repo/fedkd_gia/kaggle/run_attack.py"], check=True)
