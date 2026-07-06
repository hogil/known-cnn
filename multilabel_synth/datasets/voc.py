import os
import xml.etree.ElementTree as ET
import numpy as np
from PIL import Image

VOC_CLASSES = [
    "aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat",
    "chair", "cow", "diningtable", "dog", "horse", "motorbike", "person",
    "pottedplant", "sheep", "sofa", "train", "tvmonitor",
]
CLS2IDX = {c: i for i, c in enumerate(VOC_CLASSES)}
N_CLASSES = len(VOC_CLASSES)


def _voc_root(root):
    return os.path.join(root, "VOCdevkit", "VOC2007")


def _ids(root, split):
    p = os.path.join(_voc_root(root), "ImageSets", "Main", split + ".txt")
    return [l.strip() for l in open(p) if l.strip()]


def _cats(root, img_id):
    r = ET.parse(os.path.join(_voc_root(root), "Annotations", img_id + ".xml")).getroot()
    return set(o.find("name").text for o in r.findall("object"))


def _load_img(root, img_id, size):
    p = os.path.join(_voc_root(root), "JPEGImages", img_id + ".jpg")
    im = Image.open(p).convert("RGB").resize((size, size), Image.BILINEAR)
    return np.asarray(im, dtype=np.float32) / 255.0   # H,W,C


def _multihot(cats):
    y = np.zeros(N_CLASSES, dtype=np.float32)
    for c in cats:
        y[CLS2IDX[c]] = 1.0
    return y


def single_pool_ids(root, split="trainval"):
    """Return {class: [img_id, ...]} for natural single-category images."""
    by = {c: [] for c in VOC_CLASSES}
    for i in _ids(root, split):
        c = _cats(root, i)
        if len(c) == 1:
            by[next(iter(c))].append(i)
    return by


def build_single_pool(root, split="trainval", per_class_cap=60, size=128, seed=0):
    """Natural single-category images -> (X[N,3,S,S] in [0,1], Y[N,20], ids)."""
    rng = np.random.default_rng(seed)
    by = single_pool_ids(root, split)
    picks = []
    for c in VOC_CLASSES:
        ids = list(by[c])
        rng.shuffle(ids)
        picks += [(i, c) for i in ids[:per_class_cap]]
    X = np.stack([_load_img(root, i, size) for i, _ in picks]).transpose(0, 3, 1, 2)
    Y = np.stack([_multihot({c}) for _, c in picks])
    return X.astype(np.float32), Y, [i for i, _ in picks]


def build_multi(root, split="test", n=400, size=128, seed=1, require_pairs=None,
                exclude_pairs=None):
    """Natural multi-category images -> (X, Y).

    require_pairs: if given, keep only images whose category set contains at
      least one of these unordered class-index pairs (for held-out-combo test).
    exclude_pairs: if given, drop images containing any of these pairs (for the
      oracle that must not see held-out combos).
    """
    rng = np.random.default_rng(seed)
    items = []
    for i in _ids(root, split):
        cats = _cats(root, i)
        if len(cats) < 2:
            continue
        idxs = {CLS2IDX[c] for c in cats}
        pairs = {frozenset((a, b)) for a in idxs for b in idxs if a < b}
        if require_pairs is not None and not (pairs & require_pairs):
            continue
        if exclude_pairs is not None and (pairs & exclude_pairs):
            continue
        items.append((i, cats))
    rng.shuffle(items)
    items = items[:n]
    X = np.stack([_load_img(root, i, size) for i, _ in items]).transpose(0, 3, 1, 2)
    Y = np.stack([_multihot(c) for _, c in items])
    return X.astype(np.float32), Y
