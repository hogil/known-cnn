import os
import csv as csvmod
import argparse

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

from .datasets.coco import build_single_pool, build_single_pool_crops, build_multi
from .synthesis.voc_arms import synth_arm, synth_copypaste
from .models.resnet import build_resnet18
from .metrics import pos_neg_prob, bit_f1, far, exact_match, compute_map

IMEAN = np.array([0.485, 0.456, 0.406], np.float32).reshape(1, 3, 1, 1)
ISTD = np.array([0.229, 0.224, 0.225], np.float32).reshape(1, 3, 1, 1)
ARMS = ["oracle", "copypaste", "cutmix", "mixup", "single_only"]
FIELDS = ["arm", "seed",
          "tr_bitF1", "tr_pos", "tr_neg",
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


def train_model(trX, trY, epochs, bs, lr, device, seed):
    torch.manual_seed(seed)
    model = build_resnet18(80, pretrained=True).to(device)
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arms", nargs="+", default=ARMS)
    ap.add_argument("--seeds", nargs="+", type=int, default=[0])
    ap.add_argument("--per-class-cap", type=int, default=30)
    ap.add_argument("--size", type=int, default=128)
    ap.add_argument("--n-train", type=int, default=1500)
    ap.add_argument("--n-test", type=int, default=500)
    ap.add_argument("--epochs", type=int, default=6)
    ap.add_argument("--bs", type=int, default=32)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--cutmix-frac", type=float, default=0.25)
    ap.add_argument("--out-csv", default="outputs/multilabel_synth/coco_matrix.csv")
    args = ap.parse_args()

    print("loading COCO pools (train2017 singles/crops, val2017 multi test) ...", flush=True)
    natX, natY = build_single_pool("train2017", args.per_class_cap, args.size, seed=0)
    crpX, crpY = build_single_pool_crops("train2017", args.per_class_cap, args.size, seed=0)
    orX, orY = build_multi("train2017", args.n_train, args.size, seed=5)
    teX, teY = build_multi("val2017", args.n_test, args.size, seed=1)
    print(f"natural singles {natX.shape[0]}, crops {crpX.shape[0]}, "
          f"oracle multi {orX.shape[0]}, test {teX.shape[0]}", flush=True)

    rng = np.random.default_rng(0)
    rows = []
    for arm in args.arms:
        for seed in args.seeds:
            if arm == "oracle":
                trX, trY = orX, orY
            elif arm == "single_only":
                trX, trY = natX, natY
            elif arm == "copypaste":
                sX, sY = synth_copypaste(natX, natY, crpX, crpY, args.n_train, seed)
                trX = np.concatenate([sX, natX])
                trY = np.concatenate([sY, natY])
            else:
                sX, sY = synth_arm(arm, natX, natY, args.n_train, seed,
                                   frac=args.cutmix_frac)
                trX = np.concatenate([sX, natX])
                trY = np.concatenate([sY, natY])
            model = train_model(trX, trY, args.epochs, args.bs, args.lr,
                                args.device, seed)
            idx = rng.choice(len(trX), size=min(300, len(trX)), replace=False)
            trP = _predict(model, trX[idx], trY[idx], args.bs, args.device)
            trYb = (trY[idx] >= 0.5).astype(np.float32)
            teP = _predict(model, teX, teY, args.bs, args.device)
            tr_pos, tr_neg = pos_neg_prob(trP, trYb)
            ev_pos, ev_neg = pos_neg_prob(teP, teY)
            r = {"arm": arm, "seed": seed,
                 "tr_bitF1": round(bit_f1(trP, trYb), 4),
                 "tr_pos": round(tr_pos, 4), "tr_neg": round(tr_neg, 4),
                 "ev_bitF1": round(bit_f1(teP, teY), 4),
                 "ev_FAR": round(far(teP, teY), 4),
                 "ev_exact": round(exact_match(teP, teY), 4),
                 "ev_mAP": round(compute_map(teP, teY), 4),
                 "ev_pos": round(ev_pos, 4), "ev_neg": round(ev_neg, 4),
                 "n_train": int(len(trX))}
            rows.append(r)
            print(f"{arm:12s} s{seed} | TRAIN bitF1={r['tr_bitF1']:.3f} pos={r['tr_pos']:.3f} "
                  f"neg={r['tr_neg']:.3f} | EVAL bitF1={r['ev_bitF1']:.3f} FAR={r['ev_FAR']:.3f} "
                  f"exact={r['ev_exact']:.3f} mAP={r['ev_mAP']:.3f} "
                  f"pos={r['ev_pos']:.3f} neg={r['ev_neg']:.3f}", flush=True)

    os.makedirs(os.path.dirname(args.out_csv), exist_ok=True)
    with open(args.out_csv, "w", newline="") as f:
        w = csvmod.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        w.writerows(rows)
    print(f"[OUT] {os.path.abspath(args.out_csv)}", flush=True)


if __name__ == "__main__":
    main()
