import os
import csv as csvmod
import argparse

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

from .datasets.voc import build_single_pool, build_multi
from .synthesis.voc_arms import synth_arm
from .models.resnet import build_resnet18
from .metrics import compute_map, pos_neg_prob

IMEAN = np.array([0.485, 0.456, 0.406], np.float32).reshape(1, 3, 1, 1)
ISTD = np.array([0.229, 0.224, 0.225], np.float32).reshape(1, 3, 1, 1)
ARMS = ["oracle", "cutmix", "mixup", "single_only"]
FIELDS = ["arm", "seed", "mAP", "pos_prob", "neg_prob", "n_train"]


def _norm(X):
    return (X - IMEAN) / ISTD


def train_eval(trX, trY, teX, teY, epochs, bs, lr, device, seed):
    torch.manual_seed(seed)
    model = build_resnet18(20, pretrained=True).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    lf = nn.BCEWithLogitsLoss()
    dl = DataLoader(TensorDataset(torch.from_numpy(_norm(trX)), torch.from_numpy(trY)),
                    batch_size=bs, shuffle=True)
    for _ in range(epochs):
        model.train()
        for xb, yb in dl:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            lf(model(xb), yb).backward()
            opt.step()
    model.eval()
    P = []
    with torch.no_grad():
        for xb, _ in DataLoader(TensorDataset(torch.from_numpy(_norm(teX)), torch.from_numpy(teY)),
                                batch_size=bs):
            P.append(torch.sigmoid(model(xb.to(device))).cpu().numpy())
    P = np.concatenate(P)
    pos, neg = pos_neg_prob(P, teY)
    return compute_map(P, teY), pos, neg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arms", nargs="+", default=ARMS)
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1])
    ap.add_argument("--per-class-cap", type=int, default=40)
    ap.add_argument("--size", type=int, default=128)
    ap.add_argument("--n-train", type=int, default=800)
    ap.add_argument("--n-test", type=int, default=300)
    ap.add_argument("--epochs", type=int, default=4)
    ap.add_argument("--bs", type=int, default=32)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--cutmix-frac", type=float, default=0.3)
    ap.add_argument("--root", default="E:/data/torchvision")
    ap.add_argument("--out-csv", default="outputs/multilabel_synth/voc_matrix.csv")
    args = ap.parse_args()

    print("loading VOC single pool / oracle multi / test ...", flush=True)
    spX, spY, _ = build_single_pool(args.root, "trainval", args.per_class_cap, args.size, seed=0)
    orX, orY = build_multi(args.root, "trainval", args.n_train, args.size, seed=5)
    teX, teY = build_multi(args.root, "test", args.n_test, args.size, seed=1)
    print(f"single pool {spX.shape[0]}, oracle multi {orX.shape[0]}, test {teX.shape[0]}", flush=True)

    rows = []
    for arm in args.arms:
        for seed in args.seeds:
            if arm == "oracle":
                trX, trY = orX, orY
            elif arm == "single_only":
                trX, trY = spX, spY
            else:
                sX, sY = synth_arm(arm, spX, spY, args.n_train, seed, frac=args.cutmix_frac)
                trX = np.concatenate([sX, spX])
                trY = np.concatenate([sY, spY])
            mAP, pos, neg = train_eval(trX, trY, teX, teY, args.epochs, args.bs,
                                       args.lr, args.device, seed)
            r = {"arm": arm, "seed": seed, "mAP": round(mAP, 4),
                 "pos_prob": round(pos, 4), "neg_prob": round(neg, 4),
                 "n_train": int(trX.shape[0])}
            rows.append(r)
            print(f"{arm:12s} seed={seed} mAP={mAP:.4f} pos={pos:.4f} neg={neg:.4f} n={trX.shape[0]}", flush=True)

    os.makedirs(os.path.dirname(args.out_csv), exist_ok=True)
    with open(args.out_csv, "w", newline="") as f:
        w = csvmod.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        w.writerows(rows)
    print(f"[OUT] {os.path.abspath(args.out_csv)}", flush=True)


if __name__ == "__main__":
    main()
