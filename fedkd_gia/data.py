"""State Farm Distracted Driver data: one federated client per driver (subject).

`prepare_cache` resizes every labeled image once to IMG_SIZE and stores a uint8 array so that
training epochs do not re-decode 640x480 JPEGs. No augmentation is applied (no horizontal flips:
left/right-hand classes such as c1 vs c3 would be swapped).
"""
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
    """Read driver_imgs_list.csv + imgs/train/cX/*.jpg, write images.npy (N,H,W,3 uint8) and meta.csv."""
    os.makedirs(cache_dir, exist_ok=True)
    img_path = os.path.join(cache_dir, "images.npy")
    meta_path = os.path.join(cache_dir, "meta.csv")
    if os.path.exists(img_path) and os.path.exists(meta_path):
        return img_path, meta_path

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


def make_client_splits(meta, test_frac=0.2, seed=42):
    """Split (b): every driver keeps a stratified 80/20 train/test split of their own images.

    Returns {driver_id: {"train": idx array, "test": idx array}} with indices into the cache.
    """
    splits = {}
    for subject, grp in meta.groupby("subject", sort=True):
        tr, te = train_test_split(grp.index.values, test_size=test_frac, stratify=grp.label.values, random_state=seed)
        splits[subject] = {"train": np.sort(tr), "test": np.sort(te)}
    return splits


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
