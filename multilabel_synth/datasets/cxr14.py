"""ChestX-ray14: condition-type multi-label (findings superimposed on one
chest). Real single-finding train pool + real multi-finding eval + real
No-Finding normals. Label CSV verified: 60,361 normal / 30,963 single /
20,796 multi; 14 findings all have singles.

Image root is resolved at load time (resized mirror or original). The CSV
(Data_Entry_2017.csv, possibly a disguised zip) maps image filename ->
Finding Labels ('A|B' multi, 'No Finding' normal).
"""
import csv
import io
import os
import zipfile
import glob

import numpy as np
from PIL import Image

CSV = "E:/data/chestxray14/Data_Entry_2017.csv"
FINDINGS = ["Atelectasis", "Cardiomegaly", "Effusion", "Infiltration", "Mass",
            "Nodule", "Pneumonia", "Pneumothorax", "Consolidation", "Edema",
            "Emphysema", "Fibrosis", "Pleural_Thickening", "Hernia"]
IDX = {f: i for i, f in enumerate(FINDINGS)}


def _read_csv():
    data = open(CSV, "rb").read()
    if data[:2] == b"PK":
        z = zipfile.ZipFile(io.BytesIO(data))
        text = z.read(z.namelist()[0]).decode("latin-1")
    else:
        text = data.decode("latin-1")
    return list(csv.DictReader(io.StringIO(text, newline="")))


def _find_image_root(candidates=("E:/data/cxr14", "E:/data/chestxray14")):
    for root in candidates:
        if os.path.isdir(root):
            hits = glob.glob(os.path.join(root, "**", "00000001_000.png"), recursive=True)
            if hits:
                return os.path.dirname(hits[0])
            # any png dir
            anypng = glob.glob(os.path.join(root, "**", "*.png"), recursive=True)
            if anypng:
                return None  # signal: build a filename->path index instead
    return None


def _build_index(candidates=("E:/data/cxr14", "E:/data/chestxray14")):
    idx = {}
    for root in candidates:
        for p in glob.glob(os.path.join(root, "**", "*.png"), recursive=True):
            idx.setdefault(os.path.basename(p), p)
    return idx


def load_split(size=128, per_single_cap=400, n_multi=2000, n_normal=3000,
               n_holdout=0, seed=0):
    rows = _read_csv()
    col = [c for c in rows[0] if c and "Finding" in c][0]
    fcol = [c for c in rows[0] if c and "Image" in c][0]
    index = _build_index()
    if not index:
        raise FileNotFoundError("no CXR images found under E:/data/cxr14 or chestxray14")

    def label(ls):
        y = np.zeros(14, np.float32)
        for x in ls:
            if x in IDX:
                y[IDX[x]] = 1.0
        return y

    singles, multis, normals = [], [], []
    for r in rows:
        fn = r[fcol]
        if fn not in index:
            continue
        labs = r[col].split("|")
        if labs == ["No Finding"]:
            normals.append(fn)
        elif len(labs) == 1:
            singles.append((fn, labs))
        elif len(labs) >= 2:
            multis.append((fn, labs))
    rng = np.random.default_rng(seed)

    # single pool with per-class cap
    by = {}
    for fn, labs in singles:
        by.setdefault(labs[0], []).append(fn)
    spick = []
    for c, fns in by.items():
        rng.shuffle(fns)
        spick += [(fn, [c]) for fn in fns[:per_single_cap]]
    rng.shuffle(multis)
    mpick = multis[:n_multi]
    rng.shuffle(normals)
    npick = normals[:n_normal]

    def load(fn):
        im = Image.open(index[fn]).convert("L").resize((size, size), Image.BILINEAR)
        return (np.asarray(im, np.float32) / 255.0)[None]

    def stack(items):
        X = np.stack([load(fn) for fn, _ in items])
        Y = np.stack([label(l) for _, l in items])
        return X.astype(np.float32), Y

    spX, spY = stack(spick)
    mX, mY = stack(mpick)
    nX = np.stack([load(fn) for fn in npick]).astype(np.float32)
    nY = np.zeros((len(npick), 14), np.float32)
    return {"single": (spX, spY), "multi": (mX, mY), "normal": (nX, nY),
            "n_single_per_class": {c: len(v) for c, v in by.items()}}
