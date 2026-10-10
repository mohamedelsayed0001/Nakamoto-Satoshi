"""Datasets: State Farm (one client per driver) and CIFAR-100 (IID clients).

`prepare_cache` resizes every labeled image once to IMG_SIZE and stores a uint8 array so that
training epochs do not re-decode 640x480 JPEGs. No augmentation is applied (no horizontal flips:
left/right-hand classes such as c1 vs c3 would be swapped).
"""
import json
import os

import numpy as np
import pandas as pd
import torch
from PIL import Image
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset

IMG_SIZE = 224
NUM_CLASSES = 10
# Both timm augreg ViT checkpoints use mean=std=0.5.
MEAN = 0.5
STD = 0.5


def prepare_cache(data_root, cache_dir, num_workers=4):
    """Build images.npy (N,H,W,3 uint8), meta.csv and dataset.json in cache_dir.

    data_root containing driver_imgs_list.csv -> State Farm (resized to IMG_SIZE);
    any other data_root -> CIFAR-100 downloaded there via torchvision (kept at native 32x32;
    the models upsample to their input size).
    """
    os.makedirs(cache_dir, exist_ok=True)
    img_path = os.path.join(cache_dir, "images.npy")
    meta_path = os.path.join(cache_dir, "meta.csv")
    if os.path.exists(img_path) and os.path.exists(meta_path):
        return img_path, meta_path
    if not os.path.exists(os.path.join(data_root, "driver_imgs_list.csv")):
        return prepare_cifar100(data_root, cache_dir)
    with open(os.path.join(cache_dir, "dataset.json"), "w") as f:
        json.dump({"name": "statefarm", "num_classes": NUM_CLASSES}, f)

    meta = pd.read_csv(os.path.join(data_root, "driver_imgs_list.csv"))
    meta = meta.sort_values(["subject", "classname", "img"]).reset_index(drop=True)
    meta["label"] = meta["classname"].str[1:].astype(int)
    paths = [os.path.join(data_root, "imgs", "train", c, f) for c, f in zip(meta.classname, meta.img)]

    tmp_path = img_path + ".tmp"
    arr = np.lib.format.open_memmap(tmp_path, mode="w+", dtype=np.uint8, shape=(len(paths), IMG_SIZE, IMG_SIZE, 3))

    def load(i):
        with Image.open(paths[i]) as im:
            arr[i] = np.asarray(im.convert("RGB").resize((IMG_SIZE, IMG_SIZE), Image.BILINEAR))

    from concurrent.futures import ThreadPoolExecutor
    with ThreadPoolExecutor(num_workers) as ex:
        list(ex.map(load, range(len(paths))))
    arr.flush()
    del arr
    os.replace(tmp_path, img_path)
    meta.to_csv(meta_path, index=False)
    return img_path, meta_path


def prepare_cifar100(data_root, cache_dir):
    """CIFAR-100 (50k train + 10k official test) as one uint8 array; meta has label and split."""
    from torchvision.datasets import CIFAR100
    tr = CIFAR100(data_root, train=True, download=True)
    te = CIFAR100(data_root, train=False, download=True)
    images = np.concatenate([tr.data, te.data])
    meta = pd.DataFrame({"label": np.concatenate([tr.targets, te.targets]).astype(int),
                         "split": ["train"] * len(tr.data) + ["test"] * len(te.data)})
    np.save(os.path.join(cache_dir, "images.npy"), images)
    meta.to_csv(os.path.join(cache_dir, "meta.csv"), index=False)
    with open(os.path.join(cache_dir, "dataset.json"), "w") as f:
        json.dump({"name": "cifar100", "num_classes": 100}, f)
    return os.path.join(cache_dir, "images.npy"), os.path.join(cache_dir, "meta.csv")


def num_classes(cache_dir):
    path = os.path.join(cache_dir, "dataset.json")
    return json.load(open(path))["num_classes"] if os.path.exists(path) else NUM_CLASSES


def make_iid_splits(meta, n_clients=30, test_frac=0.2, seed=42):
    """IID split of the official training set: every class is dealt round-robin over the clients, then each
    client keeps a stratified train/test split of its share. Returns (splits, official_test_idx)."""
    rng = np.random.default_rng(seed)
    train = meta[meta.split == "train"]
    shares = [[] for _ in range(n_clients)]
    pos = 0  # continue the deal across classes so client sizes differ by at most one image
    for _, grp in train.groupby("label"):
        for i in rng.permutation(grp.index.values):
            shares[pos % n_clients].append(i)
            pos += 1
    splits = {}
    for c, share in enumerate(shares):
        share = np.array(share)
        tr, te = train_test_split(share, test_size=test_frac, stratify=meta.label.values[share], random_state=seed)
        splits[f"c{c:02d}"] = {"train": np.sort(tr), "test": np.sort(te)}
    return splits, np.sort(meta.index[meta.split == "test"].values)


def make_client_splits(meta, test_frac=0.2, seed=42):
    """Split (b): every driver keeps a stratified 80/20 train/test split of their own images.

    Returns {driver_id: {"train": idx array, "test": idx array}} with indices into the cache.
    """
    splits = {}
    for subject, grp in meta.groupby("subject", sort=True):
        tr, te = train_test_split(grp.index.values, test_size=test_frac, stratify=grp.label.values, random_state=seed)
        splits[subject] = {"train": np.sort(tr), "test": np.sort(te)}
    return splits


def make_driver_splits(meta, holdout_drivers, client_test_frac=0.0, seed=42):
    """Split (a): held-out drivers form an unseen test set; every other driver is a client. With
    client_test_frac > 0 each client additionally keeps a stratified local test split of its own images.
    Returns ({driver_id: {"train": idx, "test": idx}}, holdout_idx)."""
    holdout = set(holdout_drivers)
    splits = {}
    for s, g in meta.groupby("subject", sort=True):
        if s in holdout:
            continue
        if client_test_frac > 0:
            tr, te = train_test_split(g.index.values, test_size=client_test_frac, stratify=g.label.values,
                                      random_state=seed)
            splits[s] = {"train": np.sort(tr), "test": np.sort(te)}
        else:
            splits[s] = {"train": np.sort(g.index.values), "test": np.array([], dtype=int)}
    holdout_idx = np.sort(meta.index[meta.subject.isin(holdout)].values)
    return splits, holdout_idx


class CachedImages(Dataset):
    def __init__(self, images, labels, indices):
        self.images = images  # memmap/ndarray (N,H,W,3) uint8
        self.labels = labels
        self.indices = np.asarray(indices)

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, i):
        j = self.indices[i]
        x = torch.from_numpy(np.array(self.images[j])).permute(2, 0, 1).float().div_(255)
        x = (x - MEAN) / STD
        return x, int(self.labels[j])


def load_cache(cache_dir):
    images = np.load(os.path.join(cache_dir, "images.npy"), mmap_mode="r")
    meta = pd.read_csv(os.path.join(cache_dir, "meta.csv"))
    return images, meta
