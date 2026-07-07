"""C1: COCO top-20 subset retry. The 80-class run collapsed at CPU scale;
restrict to the 20 classes with the largest natural-single pools, project
labels onto those 20 dims (images may contain other unlabeled objects —
honest protocol note), and rerun the arm comparison at a scale where signal
is attainable on CPU.
"""
import os
import argparse

import numpy as np
from PIL import Image

from .datasets.coco import _load_index, _load_img, ROOT
from .synthesis.voc_arms import synth_copypaste, synth_arm
from .run_voc import train_model_voc, eval_voc
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from .models.resnet import build_resnet18


def build_pools(split, top, cap, size, seed, min_box=32):
    rng = np.random.default_rng(seed)
    _, _, img_cats, img_file, img_boxes = _load_index(split)
    K = len(top)
    remap = {c: i for i, c in enumerate(top)}
    # natural singles among top classes
    natX, natY = [], []
    by = {}
    for ii, cats in img_cats.items():
        if len(cats) == 1:
            c = next(iter(cats))
            if c in remap:
                by.setdefault(c, []).append(ii)
    for c, ids in sorted(by.items()):
        ids = list(ids)
        rng.shuffle(ids)
        for ii in ids[:cap]:
            natX.append(_load_img(split, img_file[ii], size))
            y = np.zeros(K, np.float32); y[remap[c]] = 1.0
            natY.append(y)
    natX = np.stack(natX).transpose(0, 3, 1, 2).astype(np.float32)
    natY = np.stack(natY)
    # crops
    byc = {}
    for ii, boxes in img_boxes.items():
        for ci, (x0, y0, x1, y1) in boxes:
            if ci in remap and (x1 - x0) >= min_box and (y1 - y0) >= min_box:
                byc.setdefault(ci, []).append((ii, (x0, y0, x1, y1)))
    crX, crY = [], []
    for c, items in sorted(byc.items()):
        items = list(items)
        rng.shuffle(items)
        for ii, box in items[:cap]:
            p = os.path.join(ROOT, split, img_file[ii])
            im = Image.open(p).convert("RGB").crop(box).resize((size, size), Image.BILINEAR)
            crX.append(np.asarray(im, dtype=np.float32) / 255.0)
            y = np.zeros(K, np.float32); y[remap[c]] = 1.0
            crY.append(y)
    crX = np.stack(crX).transpose(0, 3, 1, 2).astype(np.float32)
    crY = np.stack(crY)
    return natX, natY, crX, crY


def build_multi20(split, top, n, size, seed):
    rng = np.random.default_rng(seed)
    _, _, img_cats, img_file, _ = _load_index(split)
    remap = {c: i for i, c in enumerate(top)}
    items = []
    for ii, cats in img_cats.items():
        sel = [c for c in cats if c in remap]
        if len(sel) >= 2:
            items.append((ii, sel))
    rng.shuffle(items)
    items = items[:n]
    X = np.stack([_load_img(split, img_file[ii], size) for ii, _ in items])
    Y = np.zeros((len(items), len(top)), np.float32)
    for k, (_, sel) in enumerate(items):
        for c in sel:
            Y[k, remap[c]] = 1.0
    return X.transpose(0, 3, 1, 2).astype(np.float32), Y


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--k", type=int, default=20)
    ap.add_argument("--cap", type=int, default=100)
    ap.add_argument("--size", type=int, default=128)
    ap.add_argument("--n-train", type=int, default=1500)
    ap.add_argument("--n-test", type=int, default=500)
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--bs", type=int, default=32)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--seeds", nargs="+", type=int, default=[0])
    args = ap.parse_args()

    # top-K classes by natural-single count in train2017
    _, names, img_cats, _, _ = _load_index("train2017")
    from collections import Counter
    per1 = Counter()
    for cats in img_cats.values():
        if len(cats) == 1:
            per1[next(iter(cats))] += 1
    top = [c for c, _ in per1.most_common(args.k)]
    print("top classes:", [names[c] for c in top], flush=True)

    natX, natY, crX, crY = build_pools("train2017", top, args.cap, args.size, seed=0)
    orX, orY = build_multi20("train2017", top, args.n_train, args.size, seed=5)
    teX, teY = build_multi20("val2017", top, args.n_test, args.size, seed=1)
    print(f"singles {len(natX)} crops {len(crX)} oracle {len(orX)} test {len(teX)}", flush=True)

    rng2 = np.random.default_rng(0)
    for arm in ["oracle", "copypaste", "cutmix", "single_only"]:
        for seed in args.seeds:
            if arm == "oracle":
                trX, trY = orX, orY
            elif arm == "single_only":
                trX, trY = natX, natY
            elif arm == "copypaste":
                sX, sY = synth_copypaste(natX, natY, crX, crY, args.n_train, seed)
                trX = np.concatenate([sX, natX]); trY = np.concatenate([sY, natY])
            else:
                sX, sY = synth_arm("cutmix", natX, natY, args.n_train, seed, frac=0.25)
                trX = np.concatenate([sX, natX]); trY = np.concatenate([sY, natY])
            # K-class head
            torch.manual_seed(seed)
            model = build_resnet18(len(top), pretrained=True).to(args.device)
            from .run_voc import IMEAN, ISTD
            opt = torch.optim.Adam(model.parameters(), lr=args.lr)
            lf = nn.BCEWithLogitsLoss()
            Xn = ((trX - IMEAN) / ISTD)
            dl = DataLoader(TensorDataset(torch.from_numpy(Xn), torch.from_numpy(trY)),
                            batch_size=args.bs, shuffle=True)
            for _ in range(args.epochs):
                model.train()
                for xb, yb in dl:
                    xb, yb = xb.to(args.device), yb.to(args.device)
                    opt.zero_grad(); lf(model(xb), yb).backward(); opt.step()
            ev = eval_voc(model, teX, teY, args.bs, args.device)
            idx = rng2.choice(len(trX), size=min(300, len(trX)), replace=False)
            tr = eval_voc(model, trX[idx], (trY[idx] >= 0.5).astype(np.float32),
                          args.bs, args.device)
            print(f"{arm:12s} s{seed} | TRAIN bitF1={tr['bitF1']:.3f} pos={tr['pos']:.3f} "
                  f"neg={tr['neg']:.3f} | EVAL bitF1={ev['bitF1']:.3f} FAR={ev['FAR']:.3f} "
                  f"mAP={ev['mAP']:.3f} pos={ev['pos']:.3f} neg={ev['neg']:.3f}", flush=True)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
