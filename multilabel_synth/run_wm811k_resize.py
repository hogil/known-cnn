"""Paper-C real-data anchor: WM-811K (cca 8-class, real wafer maps with
categorical pixel values {0,128,255}). Tests the categorical-resize thesis on
public real data with the same small CNN:
  A: NEAREST integer-factor downsize -> one-hot 3ch  (category-preserving + encoding)
  C: NEAREST downsize, grayscale 1ch                 (category-preserving)
  B: BICUBIC downsize, grayscale 1ch                 (interpolated — thesis: worst)
"""
import argparse
import glob
import os

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import TensorDataset, DataLoader

from .models.small_cnn import SmallCNN

ROOT = "E:/data/images/cca"


def load_all(size=56):
    classes = sorted(c for c in os.listdir(ROOT)
                     if os.path.isdir(os.path.join(ROOT, c)))
    Xa, Xc, Xb, y = [], [], [], []
    for ci, c in enumerate(classes):
        for p in glob.glob(os.path.join(ROOT, c, "*.png")):
            im = Image.open(p)
            g = np.array(im)[:, :, 0]                       # {0,128,255}
            near = np.array(Image.fromarray(g).resize((size, size), Image.NEAREST))
            bic = np.array(Image.fromarray(g).resize((size, size), Image.BICUBIC))
            cat = np.zeros_like(near, dtype=np.int64)
            cat[near == 128] = 1
            cat[near == 255] = 2
            onehot = np.eye(3, dtype=np.float32)[cat].transpose(2, 0, 1)
            Xa.append(onehot)
            Xc.append((near[None].astype(np.float32)) / 255.0)
            Xb.append((bic[None].astype(np.float32)) / 255.0)
            y.append(ci)
    return (np.stack(Xa), np.stack(Xc), np.stack(Xb),
            np.array(y, np.int64), classes)


def macro_f1(pred, y, k):
    f1s = []
    for c in range(k):
        tp = int(((pred == c) & (y == c)).sum())
        fp = int(((pred == c) & (y != c)).sum())
        fn = int(((pred != c) & (y == c)).sum())
        if tp + fp + fn == 0:
            continue
        p = tp / (tp + fp) if tp + fp else 0.0
        r = tp / (tp + fn) if tp + fn else 0.0
        f1s.append(2 * p * r / (p + r) if p + r else 0.0)
    return float(np.mean(f1s))


def run(X, y, k, in_ch, seed, epochs=30, bs=32):
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(X))
    ntr = int(0.7 * len(X))
    tr, te = idx[:ntr], idx[ntr:]
    torch.manual_seed(seed)
    model = SmallCNN(num_classes=k, in_ch=in_ch)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    lf = nn.CrossEntropyLoss()
    dl = DataLoader(TensorDataset(torch.from_numpy(X[tr]), torch.from_numpy(y[tr])),
                    batch_size=bs, shuffle=True)
    for _ in range(epochs):
        model.train()
        for xb, yb in dl:
            opt.zero_grad()
            lf(model(xb), yb).backward()
            opt.step()
    model.eval()
    with torch.no_grad():
        pred = model(torch.from_numpy(X[te])).argmax(1).numpy()
    return macro_f1(pred, y[te], k)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    ap.add_argument("--epochs", type=int, default=30)
    args = ap.parse_args()
    Xa, Xc, Xb, y, classes = load_all()
    print(f"WM-811K cca: {len(y)} maps, {len(classes)} classes", flush=True)
    for name, X, ch in [("A_nearest_onehot", Xa, 3),
                        ("C_nearest_gray", Xc, 1),
                        ("B_bicubic_gray", Xb, 1)]:
        f1s = [run(X, y, len(classes), ch, s, epochs=args.epochs)
               for s in args.seeds]
        print(f"{name:18s} macroF1={np.mean(f1s):.4f}+-{np.std(f1s):.4f} "
              f"({', '.join(f'{v:.4f}' for v in f1s)})", flush=True)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
