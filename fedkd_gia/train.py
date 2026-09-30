"""Federated FedKD training on State Farm (one client per driver), with checkpoint/resume.

Clients are simulated in one process. With several GPUs, clients of a round are trained in
parallel (one worker thread per GPU, each owning its own teacher/student/projector modules).
All per-client state (private teacher + hidden projector) and the global student are
checkpointed after every round, so a run can continue in a later Kaggle session.
"""
import argparse
import json
import os
import queue
import threading
import time

import numpy as np
import torch

from data import NUM_CLASSES, load_cache, make_client_splits, prepare_cache
from fedkd import aggregate, client_update, macro_f1, predict
from models import HiddenProjector, build_student, build_teacher, shared_state


def parse_args(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--data_root", required=True, help="folder with driver_imgs_list.csv and imgs/train")
    p.add_argument("--cache_dir", default="cache")
    p.add_argument("--out_dir", default="runs/baseline")
    p.add_argument("--resume_from", default=None, help="previous out_dir to continue from (read-only ok)")
    p.add_argument("--rounds", type=int, default=30)
    p.add_argument("--local_epochs", type=int, default=5)
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--eval_batch_size", type=int, default=64)
    p.add_argument("--lr", type=float, default=3e-5)
    p.add_argument("--test_frac", type=float, default=0.2)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--num_workers", type=int, default=2)
    p.add_argument("--amp", type=int, default=1)
    p.add_argument("--pretrained", type=int, default=1)
    p.add_argument("--freeze_embeddings", type=int, default=1)
    p.add_argument("--max_clients", type=int, default=0, help="debug: use only the first N drivers")
    p.add_argument("--max_steps", type=int, default=0, help="debug: cap local steps per client per round")
    p.add_argument("--max_eval", type=int, default=0, help="debug: cap test images per client")
    p.add_argument("--time_budget_hours", type=float, default=0, help="stop cleanly before exceeding this")
    return p.parse_args(argv)


class Worker:
    """Model buffers living on one device; client state is loaded into them per task."""

    def __init__(self, device, args):
        self.device = device
        self.teacher = build_teacher(NUM_CLASSES, args.pretrained, args.freeze_embeddings).to(device)
        self.student = build_student(NUM_CLASSES, args.pretrained, args.freeze_embeddings).to(device)
        self.projector = HiddenProjector(self.student.depth, self.student.embed_dim, self.teacher.embed_dim).to(device)


def init_client_state(args, worker):
    """Every client starts from the same pretrained teacher and a fresh projector."""
    torch.manual_seed(args.seed)
    t = build_teacher(NUM_CLASSES, args.pretrained, args.freeze_embeddings)
    pr = HiddenProjector(worker.student.depth, worker.student.embed_dim, worker.teacher.embed_dim)
    return {"teacher": t.state_dict(), "projector": pr.state_dict()}


def client_path(ckpt_dir, cid):
    return os.path.join(ckpt_dir, f"client_{cid}.pt")


def save_atomic(obj, path):
    tmp = path + ".tmp"
    torch.save(obj, tmp)
    os.replace(tmp, path)


def main(args):
    t0 = time.time()
    os.makedirs(args.out_dir, exist_ok=True)
    ckpt_dir = os.path.join(args.out_dir, "ckpt")
    os.makedirs(ckpt_dir, exist_ok=True)
    with open(os.path.join(args.out_dir, "config.json"), "w") as f:
        json.dump(vars(args), f, indent=2)

    prepare_cache(args.data_root, args.cache_dir)
    images, meta = load_cache(args.cache_dir)
    labels = meta.label.values
    splits = make_client_splits(meta, args.test_frac, args.seed)
    clients = sorted(splits)[: args.max_clients or None]
    with open(os.path.join(args.out_dir, "splits.json"), "w") as f:
        json.dump({c: {k: v.tolist() for k, v in s.items()} for c, s in splits.items()}, f)
    print(f"{len(clients)} clients, train sizes: {[len(splits[c]['train']) for c in clients]}", flush=True)

    devices = [torch.device(f"cuda:{i}") for i in range(torch.cuda.device_count())] or [torch.device("cpu")]
    workers = [Worker(d, args) for d in devices]
    print(f"devices: {devices}", flush=True)

    # ---- state: resume or init ----
    start_round = 0
    src = args.resume_from or args.out_dir
    state_file = os.path.join(src, "state.json")
    if os.path.exists(state_file):
        with open(state_file) as f:
            start_round = json.load(f)["completed_rounds"]
        global_state = torch.load(os.path.join(src, "ckpt", "global_student.pt"), map_location="cpu")
        for cid in clients:
            if src != args.out_dir:
                save_atomic(torch.load(client_path(os.path.join(src, "ckpt"), cid), map_location="cpu"),
                            client_path(ckpt_dir, cid))
        if src != args.out_dir:
            for name in ("metrics.jsonl", "state.json"):
                if os.path.exists(os.path.join(src, name)):
                    with open(os.path.join(src, name)) as fi, open(os.path.join(args.out_dir, name), "w") as fo:
                        fo.write(fi.read())
            save_atomic(global_state, os.path.join(ckpt_dir, "global_student.pt"))
        print(f"resumed after round {start_round}", flush=True)
    else:
        global_state = {k: v.cpu().clone() for k, v in shared_state(workers[0].student).items()}
        init = init_client_state(args, workers[0])
        for cid in clients:
            save_atomic(init, client_path(ckpt_dir, cid))
        save_atomic(global_state, os.path.join(ckpt_dir, "global_student.pt"))

    round_times = []
    for rnd in range(start_round, args.rounds):
        if args.time_budget_hours and round_times:
            est = np.mean(round_times[-3:])
            if time.time() - t0 + est > args.time_budget_hours * 3600:
                print(f"time budget reached before round {rnd + 1}; stopping for resume", flush=True)
                break
        r0 = time.time()

        # ---- local training (parallel over devices) ----
        tasks = queue.Queue()
        for cid in clients:
            tasks.put(cid)
        results, lock, errors = {}, threading.Lock(), []

        def run(worker):
            while True:
                try:
                    cid = tasks.get_nowait()
                except queue.Empty:
                    return
                try:
                    cs = torch.load(client_path(ckpt_dir, cid), map_location="cpu")
                    worker.teacher.load_state_dict(cs["teacher"])
                    worker.projector.load_state_dict(cs["projector"])
                    worker.student.load_state_dict(global_state, strict=False)
                    seed = args.seed * 100003 + rnd * 1009 + clients.index(cid)
                    delta, stats = client_update(worker.teacher, worker.student, worker.projector, images, labels,
                                                 splits[cid]["train"], args, worker.device, seed)
                    # Private mentor accuracy on the client's own held-out images (after local training).
                    te_idx = splits[cid]["test"][: args.max_eval or None]
                    tp, ty, tl = predict(worker.teacher, images, labels, te_idx, args, worker.device)
                    stats.update(teacher_test_acc=float((tp == ty).mean()), teacher_test_loss=tl)
                    save_atomic({"teacher": {k: v.cpu() for k, v in worker.teacher.state_dict().items()},
                                 "projector": {k: v.cpu() for k, v in worker.projector.state_dict().items()}},
                                client_path(ckpt_dir, cid))
                    with lock:
                        results[cid] = (delta, stats)
                    print(f"  round {rnd + 1} client {cid}: {json.dumps({k: round(v, 4) for k, v in stats.items()})}",
                          flush=True)
                except Exception as e:  # surface worker errors in the main thread
                    errors.append(e)
                    return

        threads = [threading.Thread(target=run, args=(w,)) for w in workers]
        for th in threads:
            th.start()
        for th in threads:
            th.join()
        if errors:
            raise errors[0]

        # ---- server aggregation ----
        global_state = aggregate(global_state, [results[c][0] for c in clients])

        # ---- evaluation of the global student ----
        w = workers[0]
        w.student.load_state_dict(global_state, strict=False)
        per_client, all_p, all_y = {}, [], []
        for cid in clients:
            te_idx = splits[cid]["test"][: args.max_eval or None]
            p_, y_, l_ = predict(w.student, images, labels, te_idx, args, w.device)
            per_client[cid] = {"student_acc": float((p_ == y_).mean()), "student_loss": l_,
                               **{k: results[cid][1][k] for k in ("teacher_test_acc", "train_acc_s", "train_acc_t", "loss")}}
            all_p.append(p_); all_y.append(y_)
        all_p, all_y = np.concatenate(all_p), np.concatenate(all_y)
        rec = {
            "round": rnd + 1,
            "global_student_acc": float((all_p == all_y).mean()),
            "global_student_macro_f1": macro_f1(all_p, all_y, NUM_CLASSES),
            "mean_client_student_acc": float(np.mean([v["student_acc"] for v in per_client.values()])),
            "mean_client_teacher_acc": float(np.mean([v["teacher_test_acc"] for v in per_client.values()])),
            "round_seconds": time.time() - r0,
            "per_client": per_client,
        }
        save_atomic(global_state, os.path.join(ckpt_dir, "global_student.pt"))
        with open(os.path.join(args.out_dir, "metrics.jsonl"), "a") as f:
            f.write(json.dumps(rec) + "\n")
        with open(os.path.join(args.out_dir, "state.json"), "w") as f:
            json.dump({"completed_rounds": rnd + 1}, f)
        round_times.append(time.time() - r0)
        print(f"ROUND {rnd + 1}/{args.rounds}: global student acc {rec['global_student_acc']:.4f} "
              f"F1 {rec['global_student_macro_f1']:.4f} | mean teacher acc {rec['mean_client_teacher_acc']:.4f} "
              f"| {rec['round_seconds'] / 60:.1f} min", flush=True)

    done = os.path.exists(os.path.join(args.out_dir, "state.json")) and \
        json.load(open(os.path.join(args.out_dir, "state.json")))["completed_rounds"] >= args.rounds
    print("TRAINING COMPLETE" if done else "PAUSED (resume with --resume_from)", flush=True)


if __name__ == "__main__":
    main(parse_args())
