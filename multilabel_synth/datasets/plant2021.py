"""Plant Pathology 2021: condition-type multi-label (diseases co-occur on one
leaf). data.csv (gpiosenka mirror) gives labels; images from the resized
mirror, matched by basename. healthy = normal; multi = >=2 disease tokens.
"""
import csv
import io
import glob
import os

import numpy as np
from PIL import Image

CSV = "E:/data/plant2021/data.csv"
CLASSES = ["scab", "frog_eye_leaf_spot", "rust", "powdery_mildew", "complex"]
IDX = {c: i for i, c in enumerate(CLASSES)}


def _index():
    idx = {}
    for p in glob.glob("E:/data/plant2021/**/*.jpg", recursive=True):
        idx.setdefault(os.path.basename(p), p)
    return idx


def load_split(size=128, per_single_cap=800, n_multi=1200, n_normal=3000, seed=0):
    rows = list(csv.DictReader(io.StringIO(open(CSV, encoding="latin-1").read(), newline="")))
    index = _index()
    rng = np.random.default_rng(seed)

    def lab(tokens):
        y = np.zeros(len(CLASSES), np.float32)
        for t in tokens:
            if t in IDX:
                y[IDX[t]] = 1.0
        return y

    singles, multis, normals = {}, [], []
    for r in rows:
        fn = os.path.basename(r["filepaths"])
        if fn not in index:
            continue
        toks = r["labels"].split()
        if toks == ["healthy"]:
            normals.append(fn)
        elif len(toks) == 1:
            singles.setdefault(toks[0], []).append(fn)
        else:
            multis.append((fn, toks))

    def load(fn):
        im = Image.open(index[fn]).convert("RGB").resize((size, size), Image.BILINEAR)
        return (np.asarray(im, np.float32) / 255.0).transpose(2, 0, 1)

    spick = []
    for c, fns in singles.items():
        rng.shuffle(fns)
        spick += [(fn, [c]) for fn in fns[:per_single_cap]]
    rng.shuffle(multis); mpick = multis[:n_multi]
    rng.shuffle(normals); npick = normals[:n_normal]

    spX = np.stack([load(fn) for fn, _ in spick]).astype(np.float32)
    spY = np.stack([lab(t) for _, t in spick])
    mX = np.stack([load(fn) for fn, _ in mpick]).astype(np.float32)
    mY = np.stack([lab(t) for _, t in mpick])
    nX = np.stack([load(fn) for fn in npick]).astype(np.float32)
    nY = np.zeros((len(npick), len(CLASSES)), np.float32)
    return {"single": (spX, spY), "multi": (mX, mY), "normal": (nX, nY),
            "n_single_per_class": {c: len(v) for c, v in singles.items()}}
