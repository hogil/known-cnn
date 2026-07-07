import json
import os
import numpy as np
from PIL import Image

ROOT = "E:/data/coco"
_INDEX_CACHE = {}


def _load_index(split):
    if split in _INDEX_CACHE:
        return _INDEX_CACHE[split]
    _INDEX_CACHE[split] = _load_index_uncached(split)
    return _INDEX_CACHE[split]


def _load_index_uncached(split):
    """Parse instances json -> (cats: id->idx0..79, names, img_cats, img_file, img_boxes)."""
    d = json.load(open(os.path.join(ROOT, "annotations", f"instances_{split}.json"),
                       encoding="utf-8"))
    cat_ids = sorted(c["id"] for c in d["categories"])
    cid2idx = {cid: i for i, cid in enumerate(cat_ids)}
    names = [c["name"] for c in sorted(d["categories"], key=lambda c: c["id"])]
    img_file = {im["id"]: im["file_name"] for im in d["images"]}
    img_cats, img_boxes = {}, {}
    for a in d["annotations"]:
        ii = a["image_id"]
        ci = cid2idx[a["category_id"]]
        img_cats.setdefault(ii, set()).add(ci)
        x, y, w, h = a["bbox"]
        img_boxes.setdefault(ii, []).append((ci, (x, y, x + w, y + h)))
    return cid2idx, names, img_cats, img_file, img_boxes


def _load_img(split, fname, size):
    p = os.path.join(ROOT, split, fname)
    im = Image.open(p).convert("RGB").resize((size, size), Image.BILINEAR)
    return np.asarray(im, dtype=np.float32) / 255.0


def _multihot(cats, n=80):
    y = np.zeros(n, dtype=np.float32)
    for c in cats:
        y[c] = 1.0
    return y


def build_single_pool(split="train2017", per_class_cap=30, size=128, seed=0):
    """Natural single-category images -> (X[N,3,S,S], Y[N,80])."""
    rng = np.random.default_rng(seed)
    _, _, img_cats, img_file, _ = _load_index(split)
    by = {}
    for ii, cats in img_cats.items():
        if len(cats) == 1:
            by.setdefault(next(iter(cats)), []).append(ii)
    X, Y = [], []
    for c, ids in sorted(by.items()):
        ids = list(ids)
        rng.shuffle(ids)
        for ii in ids[:per_class_cap]:
            X.append(_load_img(split, img_file[ii], size))
            Y.append(_multihot({c}))
    X = np.stack(X).transpose(0, 3, 1, 2)
    return X.astype(np.float32), np.stack(Y)


def build_single_pool_crops(split="train2017", per_class_cap=30, size=128,
                            seed=0, min_box=32):
    """Per-object bbox crops (balanced coverage for thin-tail classes)."""
    rng = np.random.default_rng(seed)
    _, _, _, img_file, img_boxes = _load_index(split)
    by = {}
    for ii, boxes in img_boxes.items():
        for ci, (x0, y0, x1, y1) in boxes:
            if (x1 - x0) >= min_box and (y1 - y0) >= min_box:
                by.setdefault(ci, []).append((ii, (x0, y0, x1, y1)))
    X, Y = [], []
    for c, items in sorted(by.items()):
        items = list(items)
        rng.shuffle(items)
        for ii, box in items[:per_class_cap]:
            p = os.path.join(ROOT, split, img_file[ii])
            im = Image.open(p).convert("RGB").crop(box).resize((size, size),
                                                               Image.BILINEAR)
            X.append(np.asarray(im, dtype=np.float32) / 255.0)
            Y.append(_multihot({c}))
    X = np.stack(X).transpose(0, 3, 1, 2)
    return X.astype(np.float32), np.stack(Y)


def build_multi(split="val2017", n=500, size=128, seed=1):
    """Natural multi-category images -> (X, Y)."""
    rng = np.random.default_rng(seed)
    _, _, img_cats, img_file, _ = _load_index(split)
    items = [(ii, cats) for ii, cats in img_cats.items() if len(cats) >= 2]
    rng.shuffle(items)
    items = items[:n]
    X = np.stack([_load_img(split, img_file[ii], size) for ii, _ in items])
    Y = np.stack([_multihot(c) for _, c in items])
    return X.transpose(0, 3, 1, 2).astype(np.float32), Y
