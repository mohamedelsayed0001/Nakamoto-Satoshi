# Kaggle kernel stub: FedKD on CIFAR-100 v3 with record-level DP-SGD (epsilon = 8, delta = 1e-5 per client),
# parameter-efficient retry: student trains/shares only its last block + norm + head, teacher only norm + head,
# projector only its last map (2.19M parameters inside the mechanism instead of 39M); learning rates x10.
import os
import subprocess

os.environ["FEDKD_DATASET"] = "cifar100"
os.environ["FEDKD_ARGS"] = "--split iid --num_clients 30 --local_epochs 1 --rounds 30 --lr 3e-4 --teacher_lr 1e-4 --weight_decay 0.05 --teacher_trainable_blocks 0 --student_trainable_blocks 1 --projector_trainable_layers 1 --defense dpsgd --dp_epsilon 8 --dp_clip 1 --dp_batch_size 64 --dp_chunk 8"
subprocess.run("rm -rf /tmp/repo && git clone --depth 1 https://github.com/mohamedelsayed0001/Nakamoto-Satoshi /tmp/repo",
               shell=True, check=True)
subprocess.run(["python", "/tmp/repo/fedkd_gia/kaggle/run_baseline.py"], check=True)
