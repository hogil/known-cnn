import os
import csv as csvmod
import argparse

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

from .datasets.voc import build_single_pool, build_single_pool_crops, build_multi
from .synthesis.voc_arms import synth_arm
from .models.resnet import build_resnet18
from .metrics import compute_map, pos_neg_prob, bit_f1, far

IMEAN = np.array([0.485, 0.456, 0.406], np.float32).reshape(1, 3, 1, 1)
ISTD = np.array([0.229, 0.224, 0.225], np.float32).reshape(1, 3, 1, 1)
ARMS = ["oracle", "cutmix", "mixup", "single_only"]
FIELDS = ["arm", "seed",
          "tr_bitF1", "tr_FAR", "tr_pos", "tr_neg",
          "ev_bitF1", "ev_FAR", "ev_exact", "ev_mAP", "ev_pos", "ev_neg", "n_train"]


def _norm(X):
    return (X - IMEAN) / ISTD


def _predict(model, X, Y, bs, device):
    model.eval()
    P = []
    with torch.no_grad():
        for xb, _ in DataLoader(TensorDataset(torch.from_numpy(_norm(X)), torch.from_numpy(Y)),
                                batch_size=bs):
            P.append(torch.sigmoid(model(xb.to(device))).cpu().numpy())
    return np.concatenate(P)


def train_model_voc(trX, trY, epochs, bs, lr, device, seed):
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
    return model


def eval_voc(model, X, Y, bs, device):
    P = _predict(model, X, Y, bs, device)
    pos, neg = pos_neg_prob(P, Y)
    from .metrics import exact_match
    return {"bitF1": bit_f1(P, Y), "FAR": far(P, Y), "exact": exact_match(P, Y),
            "mAP": compute_map(P, Y), "pos": pos, "neg": neg}


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
    ap.add_argument("--single-mode", choices=["crops", "natural"], default="crops")
    ap.add_argument("--root", default="E:/data/torchvision")
    ap.add_argument("--out-csv", default="outputs/multilabel_synth/voc_matrix.csv")
    args = ap.parse_args()

    print(f"loading VOC single pool (mode={args.single_mode}) / oracle multi / test ...", flush=True)
    if args.single_mode == "crops":
        spX, spY, _ = build_single_pool_crops(args.root, "trainval", args.per_class_cap, args.size, seed=0)
    else:
        spX, spY, _ = build_single_pool(args.root, "trainval", args.per_class_cap, args.size, seed=0)
    orX, orY = build_multi(args.root, "trainval", args.n_train, args.size, seed=5)
    teX, teY = build_multi(args.root, "test", args.n_test, args.size, seed=1)
    print(f"single pool {spX.shape[0]}, oracle multi {orX.shape[0]}, test {teX.shape[0]}", flush=True)

    rng = np.random.default_rng(0)
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
            model = train_model_voc(trX, trY, args.epochs, args.bs, args.lr, args.device, seed)
            # train-set performance on a subsample (targets binarized for metrics)
            idx = rng.choice(len(trX), size=min(300, len(trX)), replace=False)
            tr = eval_voc(model, trX[idx], (trY[idx] >= 0.5).astype(np.float32), args.bs, args.device)
            ev = eval_voc(model, teX, teY, args.bs, args.device)
            r = {"arm": arm, "seed": seed,
                 "tr_bitF1": round(tr["bitF1"], 4), "tr_FAR": round(tr["FAR"], 4),
                 "tr_pos": round(tr["pos"], 4), "tr_neg": round(tr["neg"], 4),
                 "ev_bitF1": round(ev["bitF1"], 4), "ev_FAR": round(ev["FAR"], 4),
                 "ev_exact": round(ev["exact"], 4), "ev_mAP": round(ev["mAP"], 4),
                 "ev_pos": round(ev["pos"], 4), "ev_neg": round(ev["neg"], 4),
                 "n_train": int(trX.shape[0])}
            rows.append(r)
            print(f"{arm:12s} s{seed} | TRAIN bitF1={tr['bitF1']:.3f} FAR={tr['FAR']:.3f} "
                  f"pos={tr['pos']:.3f} neg={tr['neg']:.3f} | EVAL bitF1={ev['bitF1']:.3f} "
                  f"FAR={ev['FAR']:.3f} exact={ev['exact']:.3f} mAP={ev['mAP']:.3f} "
                  f"pos={ev['pos']:.3f} neg={ev['neg']:.3f}", flush=True)

    os.makedirs(os.path.dirname(args.out_csv), exist_ok=True)
    with open(args.out_csv, "w", newline="") as f:
        w = csvmod.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        w.writerows(rows)
    print(f"[OUT] {os.path.abspath(args.out_csv)}", flush=True)


if __name__ == "__main__":
    main()
