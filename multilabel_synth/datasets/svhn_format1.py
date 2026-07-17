"""SVHN format-1 (full house-number) loader with digitStruct.mat parsing.

SVHN format 1 is the ORIGINAL, REAL photographed data: each image is a full
house-number sign, and digitStruct.mat carries a per-digit bounding box + label.
This is genuinely PARTITION-structured multi-label data -- the digits sit
side-by-side in disjoint horizontal regions, exactly the partition combination
law. We use it two ways:

  1. single-digit pool  : crop each digit by its bbox -> 32x32 grayscale, label
     is that digit's class (a REAL single-digit crop, the shared source for all
     content-blind synthesis arms).
  2. real multi-digit   : crop the union bbox of a k-digit house number ->
     (H x W) grayscale canvas, multi-hot over the 10 digit classes = WHICH digits
     appear (bbox positions are NOT used as labels, only presence). This is the
     REAL partition-law multi-label test set (and the oracle training source).

digitStruct.mat is HDF5 (MATLAB v7.3); parsed here with h5py. SVHN labels digit
'0' as 10 in the .mat -- remapped to 0 so classes are 0..9.
"""
import os
import numpy as np
from PIL import Image


def _digitstruct_path(root_dir):
    p = os.path.join(root_dir, "digitStruct.mat")
    if not os.path.isfile(p):
        raise FileNotFoundError(f"digitStruct.mat not found in {root_dir}")
    return p


def parse_digitstruct(root_dir, limit=None):
    """Parse digitStruct.mat -> list of (filename, digits) where digits is a
    list of dicts {label:int(0..9), top, left, height, width}.

    Uses the standard h5py reference-dereferencing idiom for SVHN v7.3 .mat.
    """
    import h5py
    path = _digitstruct_path(root_dir)
    f = h5py.File(path, "r")
    ds = f["digitStruct"]
    names = ds["name"]
    bboxes = ds["bbox"]
    n = names.shape[0] if limit is None else min(limit, names.shape[0])

    def _name(i):
        ref = names[i][0]
        return "".join(chr(c[0]) for c in f[ref][()])

    def _field(item, key):
        vals = item[key]
        if vals.shape[0] == 1:
            return [int(vals[0][0])]
        return [int(f[vals[j][0]][()][0][0]) for j in range(vals.shape[0])]

    out = []
    for i in range(n):
        ref = bboxes[i][0]
        item = f[ref]
        labels = _field(item, "label")
        tops = _field(item, "top")
        lefts = _field(item, "left")
        heights = _field(item, "height")
        widths = _field(item, "width")
        digits = []
        for lab, t, l, h, w in zip(labels, tops, lefts, heights, widths):
            cls = 0 if lab == 10 else lab           # SVHN: '0' stored as 10
            digits.append(dict(label=int(cls), top=int(t), left=int(l),
                               height=int(h), width=int(w)))
        out.append((_name(i), digits))
    f.close()
    return out


def _crop_gray(img, box, out_size):
    """Crop img (PIL RGB) to box=(l,t,r,b), clamp to bounds, grayscale, resize.

    out_size = (W, H). Returns uint8 HxW array.
    """
    W, H = img.size
    l, t, r, b = box
    l = max(0, min(l, W - 1)); t = max(0, min(t, H - 1))
    r = max(l + 1, min(r, W)); b = max(t + 1, min(b, H))
    crop = img.crop((l, t, r, b)).convert("L").resize(out_size, Image.BICUBIC)
    return np.asarray(crop, dtype=np.uint8)


def load_single_digit_pool(root_dir, digit_size=32, max_images=None,
                           per_class_cap=None, seed=0):
    """Crop every digit by its bbox -> (imgs[N,digit_size,digit_size] uint8,
    labels[N] int 0..9). Real single-digit crops shared by all synthesis arms.

    per_class_cap caps crops per class (balanced pool); None keeps all.
    """
    recs = parse_digitstruct(root_dir, limit=max_images)
    rng = np.random.default_rng(seed)
    buckets = {c: [] for c in range(10)}
    for fname, digits in recs:
        if not digits:
            continue
        path = os.path.join(root_dir, fname)
        img = None
        for d in digits:
            box = (d["left"], d["top"], d["left"] + d["width"],
                   d["top"] + d["height"])
            if img is None:
                img = Image.open(path)
            crop = _crop_gray(img, box, (digit_size, digit_size))
            buckets[d["label"]].append(crop)
    imgs, labels = [], []
    for c in range(10):
        arr = buckets[c]
        if per_class_cap is not None and len(arr) > per_class_cap:
            idx = rng.choice(len(arr), size=per_class_cap, replace=False)
            arr = [arr[i] for i in idx]
        for crop in arr:
            imgs.append(crop); labels.append(c)
    return np.stack(imgs), np.asarray(labels, dtype=int)


def load_multidigit(root_dir, canvas_w=64, canvas_h=32, digit_counts=(2,),
                    require_distinct=True, max_images=None, n_classes=10):
    """Crop union bbox of each house number with len(digits) in digit_counts ->
    (imgs[N,1,canvas_h,canvas_w] float32/255, multihot[N,n_classes] float32).

    multihot = set of digit classes present (presence only, not position/count).
    require_distinct: keep only numbers whose present-digit SET has >=2 classes
    (so the multi-hot carries >=2 positive bits, matching the distinct-pair
    synthesis arms). Set False to keep same-digit numbers like '22'.
    """
    recs = parse_digitstruct(root_dir, limit=max_images)
    X, Y = [], []
    n_same = 0
    for fname, digits in recs:
        if len(digits) not in digit_counts:
            continue
        present = sorted({d["label"] for d in digits})
        if require_distinct and len(present) < 2:
            n_same += 1
            continue
        l = min(d["left"] for d in digits)
        t = min(d["top"] for d in digits)
        r = max(d["left"] + d["width"] for d in digits)
        b = max(d["top"] + d["height"] for d in digits)
        img = Image.open(os.path.join(root_dir, fname))
        crop = _crop_gray(img, (l, t, r, b), (canvas_w, canvas_h))
        y = np.zeros(n_classes, dtype=np.float32)
        for c in present:
            y[c] = 1.0
        X.append(crop); Y.append(y)
    X = np.stack(X)[:, None, :, :].astype(np.float32) / 255.0
    return X, np.stack(Y), dict(n_kept=len(Y), n_same_excluded=n_same)
