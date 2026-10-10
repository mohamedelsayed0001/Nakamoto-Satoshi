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

from data import load_cache, make_client_splits, make_driver_splits, make_iid_splits, num_classes, prepare_cache
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
    p.add_argument("--teacher_lr", type=float, default=None, help="private teacher learning rate (default: --lr)")
    p.add_argument("--weight_decay", type=float, default=0.0, help="AdamW decoupled weight decay for all models")
    p.add_argument("--teacher_trainable_blocks", type=int, default=-1,
                   help="train only the last N teacher blocks (+ norm, head); -1 = all blocks")
    p.add_argument("--split", choices=["driver", "within", "iid"], default="driver",
                   help="driver: held-out drivers are an unseen test set (split a); within: 80/20 inside each driver")
    p.add_argument("--holdout_drivers", default="p064,p066,p072,p075,p081")
    p.add_argument("--client_test_frac", type=float, default=0.0,
                   help="driver split: also keep this fraction of every client's images as a local test set")
    p.add_argument("--num_clients", type=int, default=30, help="iid split: number of clients")
    p.add_argument("--teacher_eval_size", type=int, default=1000,
                   help="driver split: held-out images per teacher evaluation (all of them in the final round)")
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
        self.teacher = build_teacher(args.num_classes, args.pretrained, args.freeze_embeddings,
                                     args.teacher_trainable_blocks).to(device)
        self.student = build_student(args.num_classes, args.pretrained, args.freeze_embeddings).to(device)
        self.projector = HiddenProjector(self.student.depth, self.student.embed_dim, self.teacher.embed_dim).to(device)


def init_client_state(args, worker):
    """Every client starts from the same pretrained teacher and a fresh projector."""
    torch.manual_seed(args.seed)
    t = build_teacher(args.num_classes, args.pretrained, args.freeze_embeddings)
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
    args.num_classes = num_classes(args.cache_dir)
    labels = meta.label.values
    if args.split == "driver":
        splits, holdout_idx = make_driver_splits(meta, args.holdout_drivers.split(","), args.client_test_frac, args.seed)
        rng = np.random.default_rng(args.seed)
        teacher_eval_idx = np.sort(rng.choice(holdout_idx, min(args.teacher_eval_size, len(holdout_idx)),
                                              replace=False))
        with open(os.path.join(args.out_dir, "holdout.json"), "w") as f:
            json.dump({"drivers": args.holdout_drivers.split(","), "indices": holdout_idx.tolist(),
                       "teacher_eval_subset": teacher_eval_idx.tolist()}, f)
    elif args.split == "iid":
        splits, holdout_idx = make_iid_splits(meta, args.num_clients, args.test_frac, args.seed)
        teacher_eval_idx = None
    else:
        splits = make_client_splits(meta, args.test_frac, args.seed)
        holdout_idx = teacher_eval_idx = None
    clients = sorted(splits)[: args.max_clients or None]
    with open(os.path.join(args.out_dir, "splits.json"), "w") as f:
        json.dump({c: {k: v.tolist() for k, v in s.items()} for c, s in splits.items()}, f)
    msg = f"split={args.split}: {len(clients)} clients, train sizes: {[len(splits[c]['train']) for c in clients]}"
    if holdout_idx is not None:
        msg += f", held-out drivers {args.holdout_drivers} ({len(holdout_idx)} images)"
    print(msg, flush=True)

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
                    if worker.device.type == "cuda":
                        torch.cuda.reset_peak_memory_stats(worker.device)
                    seed = args.seed * 100003 + rnd * 1009 + clients.index(cid)
                    delta, stats = client_update(worker.teacher, worker.student, worker.projector, images, labels,
                                                 splits[cid]["train"], args, worker.device, seed)
                    # Private mentor after local training: on unseen drivers (split a; all of them in the
                    # final round) or on the client's own held-out images (within split).
                    if args.split == "driver":
                        te_idx = holdout_idx if rnd + 1 == args.rounds else teacher_eval_idx
                    else:
                        te_idx = splits[cid]["test"]
                    te_idx = te_idx[: args.max_eval or None]
                    tp, ty, tl = predict(worker.teacher, images, labels, te_idx, args, worker.device)
                    stats.update(teacher_test_acc=float((tp == ty).mean()), teacher_test_loss=tl,
                                 teacher_test_macro_f1=macro_f1(tp, ty, args.num_classes))
                    if args.split == "driver" and len(splits[cid]["test"]):
                        # Driver split with a local 80/20 split: also score the teacher on its own local test.
                        lp, ly, _ = predict(worker.teacher, images, labels, splits[cid]["test"][: args.max_eval or None],
                                            args, worker.device)
                        stats.update(teacher_local_test_acc=float((lp == ly).mean()),
                                     teacher_local_test_macro_f1=macro_f1(lp, ly, args.num_classes))
                    if worker.device.type == "cuda":
                        stats.update(gpu=worker.device.index,
                                     peak_alloc_gb=torch.cuda.max_memory_allocated(worker.device) / 2**30,
                                     peak_reserved_gb=torch.cuda.max_memory_reserved(worker.device) / 2**30)
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
        keep = ("teacher_test_acc", "teacher_test_macro_f1", "train_acc_s", "train_acc_t", "loss")
        per_client, per_driver = {}, {}
        if args.split == "driver":
            idx = holdout_idx[: args.max_eval or None]
            all_p, all_y, _ = predict(w.student, images, labels, idx, args, w.device)
            subj = meta.subject.values[idx]
            for d in sorted(set(subj)):
                m = subj == d
                per_driver[d] = {"student_acc": float((all_p[m] == all_y[m]).mean()), "n": int(m.sum())}
            local_p, local_y = [], []
            for cid in clients:
                per_client[cid] = {k: results[cid][1][k] for k in keep + ("teacher_local_test_acc",
                                                                          "teacher_local_test_macro_f1")
                                   if k in results[cid][1]}
                if len(splits[cid]["test"]):
                    p_, y_, _ = predict(w.student, images, labels, splits[cid]["test"][: args.max_eval or None],
                                        args, w.device)
                    per_client[cid]["student_local_test_acc"] = float((p_ == y_).mean())
                    local_p.append(p_)
                    local_y.append(y_)
            driver_local_acc = (float((np.concatenate(local_p) == np.concatenate(local_y)).mean())
                                if local_p else None)
        else:
            all_p, all_y = [], []
            for cid in clients:
                te_idx = splits[cid]["test"][: args.max_eval or None]
                p_, y_, l_ = predict(w.student, images, labels, te_idx, args, w.device)
                per_client[cid] = {"student_acc": float((p_ == y_).mean()), "student_loss": l_,
                                   **{k: results[cid][1][k] for k in keep}}
                all_p.append(p_)
                all_y.append(y_)
            all_p, all_y = np.concatenate(all_p), np.concatenate(all_y)
        local_test_acc = driver_local_acc if args.split == "driver" else None
        if args.split == "iid":
            # Headline number: the official test set no client trains on; keep the pooled local-test accuracy.
            local_test_acc = float((all_p == all_y).mean())
            all_p, all_y, _ = predict(w.student, images, labels, holdout_idx[: args.max_eval or None], args, w.device)
        rec = {
            "round": rnd + 1,
            "split": args.split,
            "local_test_student_acc": local_test_acc,
            "mean_client_teacher_local_test_acc": (float(np.mean([v["teacher_local_test_acc"] for v in per_client.values()]))
                                                   if all("teacher_local_test_acc" in v for v in per_client.values()) else None),
            "global_student_acc": float((all_p == all_y).mean()),
            "global_student_macro_f1": macro_f1(all_p, all_y, args.num_classes),
            "mean_client_student_acc": float(np.mean([v["student_acc"] for v in (per_driver or per_client).values()])),
            "mean_client_teacher_acc": float(np.mean([v["teacher_test_acc"] for v in per_client.values()])),
            "mean_client_teacher_macro_f1": float(np.mean([v["teacher_test_macro_f1"] for v in per_client.values()])),
            "per_heldout_driver": per_driver,
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
              f"F1 {rec['mean_client_teacher_macro_f1']:.4f} "
              f"| {rec['round_seconds'] / 60:.1f} min", flush=True)

    done = os.path.exists(os.path.join(args.out_dir, "state.json")) and \
        json.load(open(os.path.join(args.out_dir, "state.json")))["completed_rounds"] >= args.rounds
    print("TRAINING COMPLETE" if done else "PAUSED (resume with --resume_from)", flush=True)


if __name__ == "__main__":
    main(parse_args())
