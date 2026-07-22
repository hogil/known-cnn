"""Strict FULL-IMAGE SVHN loader for the operator-match paper.

Uses digitStruct ONLY for digit labels and cardinality -- NEVER bbox
coordinates and NEVER crops. Each house-number image is loaded whole and
aspect-ratio letterboxed onto a fixed canvas. This makes SVHN a bbox-free,
mask-free, real public partition-domain benchmark:

  - source-train / source-val : full images with exactly ONE digit (label = it)
  - sealed test               : full images with exactly TWO DISTINCT digits
                                (multi-label = the two)

Classes are digits 1..9 (digit 0 excluded: sparse). No bbox coordinate is ever
read for geometry; only `label` is used.
"""
import os
import numpy as np
from PIL import Image

from .svhn_format1 import parse_digitstruct

CLASSES = tuple(range(1, 10))          # digits 1..9 -> index 0..8
N_CLASSES = len(CLASSES)
_CIDX = {c: i for i, c in enumerate(CLASSES)}


def _letterbox(img, out_h, out_w, fill=0):
    """Aspect-ratio letterbox a PIL RGB image onto (out_h, out_w, 3), no crop."""
    w, h = img.size
    s = min(out_w / w, out_h / h)
    nw, nh = max(1, int(round(w * s))), max(1, int(round(h * s)))
    r = img.resize((nw, nh), Image.BILINEAR)
    canvas = np.full((out_h, out_w, 3), fill, np.uint8)
    y0, x0 = (out_h - nh) // 2, (out_w - nw) // 2
    canvas[y0:y0 + nh, x0:x0 + nw] = np.asarray(r, np.uint8)[..., :3]
    return canvas


def load_full_image_by_cardinality(root_dir, cardinality, canvas_h=64, canvas_w=128,
                                   per_class_cap=None, distinct=True, seed=0):
    """Return (imgs[N,H,W,3] uint8, Y[N,N_CLASSES] float32, meta).

    cardinality=1 -> single-digit source (per_class_cap balances per class).
    cardinality=2 -> multi-digit sealed test (distinct digits, all pairs kept).
    Only `label` from digitStruct is used; bbox coordinates are never read here.
    """
    recs = parse_digitstruct(root_dir)
    rng = np.random.default_rng(seed)
    pool = []                                            # (fname, labelset)
    for fname, digits in recs:
        labs = [d["label"] for d in digits]
        if len(labs) != cardinality:
            continue
        if any(l not in _CIDX for l in labs):            # only 1..9
            continue
        if cardinality > 1 and distinct and len(set(labs)) != cardinality:
            continue
        pool.append((fname, sorted(set(labs))))

    if cardinality == 1 and per_class_cap is not None:
        by = {c: [] for c in CLASSES}
        for fname, labs in pool:
            by[labs[0]].append(fname)
        pool = []
        for c in CLASSES:
            arr = by[c]
            if len(arr) > per_class_cap:
                idx = rng.choice(len(arr), size=per_class_cap, replace=False)
                arr = [arr[i] for i in idx]
            pool += [(fn, [c]) for fn in arr]

    imgs, Y, pairs = [], [], set()
    for fname, labs in pool:
        img = Image.open(os.path.join(root_dir, fname)).convert("RGB")
        imgs.append(_letterbox(img, canvas_h, canvas_w))
        y = np.zeros(N_CLASSES, np.float32)
        for l in labs:
            y[_CIDX[l]] = 1.0
        Y.append(y)
        if cardinality == 2:
            pairs.add(tuple(labs))
    meta = dict(n=len(imgs), cardinality=cardinality,
                n_pairs=len(pairs) if cardinality == 2 else None,
                per_class=({c: sum(1 for _, l in pool if l[0] == c) for c in CLASSES}
                           if cardinality == 1 else None))
    return np.stack(imgs), np.stack(Y).astype(np.float32), meta
