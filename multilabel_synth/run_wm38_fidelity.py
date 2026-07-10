"""Fidelity CAUSAL intervention on MixedWM38.

Everything is held fixed except one knob: in each synthesized overlay pair
(a, b) we attenuate source a's defect evidence toward the normal-die baseline
by a fraction f, while KEEPING the label {a, b}. f=1.0 is full-fidelity
overlay; f=0.0 fully erases a's evidence yet still labels a as present
(fidelity -> 0). If label fidelity is the causal driver (not a correlate),
downstream bit-F1 and positive recall must fall and false alarms rise
monotonically as f decreases, with all else (operator, data, normals,
neg-target) unchanged.

Defect evidence lives ABOVE the 0.5 normal-die baseline (0=bg, 0.5=normal,
1.0=defect), so attenuation is: a' = where(a>0.5, 0.5 + f*(a-0.5), a).
"""
import argparse
import csv as csvmod

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

from .datasets.mixedwm38 import load_wm38, split_groups
from .models.small_cnn import SmallCNN
from .metrics import bit_f1, pos_neg_prob


def _loader(X, Y, bs, shuffle):
    return DataLoader(TensorDataset(torch.from_numpy(X), torch.from_numpy(Y)),
                      batch_size=bs, shuffle=shuffle)


def _predict(model, X, bs=128, device="cpu"):
    model.eval()
    P = []
    with torch.no_grad():
        for xb, in DataLoader(TensorDataset(torch.from_numpy(X)), batch_size=bs):
            P.append(torch.sigmoid(model(xb.to(device))).cpu().numpy())
    return np.concatenate(P)


def attenuate(a, f):
    """Scale defect evidence (above the 0.5 baseline) by f; keep bg/normal."""
    return np.where(a > 0.5, 0.5 + f * (a - 0.5), a).astype(np.float32)


def synth_ablated(sX, sY, n, seed, f):
    """Overlay pairs with source a attenuated by f; label = union (unchanged).
    Also returns the mean realized survival of a's evidence in the mix."""
    rng = np.random.default_rng(seed)
    lab = sY.argmax(1)
    ii = rng.integers(0, len(sX), size=n * 4)
    jj = rng.integers(0, len(sX), size=n * 4)
    ox, oy, surv = [], [], []
    for a, b in zip(ii, jj):
        if lab[a] == lab[b]:
            continue
        ca = attenuate(sX[a], f)
        cb = sX[b]
        mix = np.maximum(ca, cb)
        # realized survival of a's defect evidence: fraction of a's defect
        # signal (above baseline) that remains above baseline in the mix
        a_ev = (sX[a] - 0.5).clip(min=0)
        kept = ((mix - 0.5).clip(min=0) * (a_ev > 0)).sum()
        surv.append(kept / (a_ev.sum() + 1e-8))
        ox.append(mix)
        oy.append(np.maximum(sY[a], sY[b]))
        if len(ox) >= n:
            break
    return (np.stack(ox).astype(np.float32), np.stack(oy).astype(np.float32),
            float(np.mean(surv)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fracs", nargs="+", type=float,
                    default=[1.0, 0.75, 0.5, 0.25, 0.0])
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    ap.add_argument("--n-train", type=int, default=3000)
    ap.add_argument("--n-single-aug", type=int, default=2000)
    ap.add_argument("--n-synth-normal", type=int, default=2000)
    ap.add_argument("--neg-target", type=float, default=0.03)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--bs", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--n-test", type=int, default=3000)
    ap.add_argument("--out-csv", default=None)
    args = ap.parse_args()

    X, Y = load_wm38()
    g = split_groups(Y)
    rng = np.random.default_rng(12345)
    mixed = rng.permutation(g["mixed"])
    test_idx = rng.choice(mixed[len(mixed) // 2:],
                          size=min(args.n_test, len(mixed) // 2), replace=False)
    teX, teY = X[test_idx], Y[test_idx]
    nrmX = X[g["normal"]]
    sX, sY = X[g["single"]], Y[g["single"]]
    K = sY.shape[1]
    print(f"singles={len(sX)} test={len(teX)} normals={len(nrmX)}", flush=True)

    rows = []
    for f in args.fracs:
        for seed in args.seeds:
            r2 = np.random.default_rng(seed)
            bX, bY, surv = synth_ablated(sX, sY, args.n_train, seed, f)
            aug = r2.choice(len(sX), size=min(args.n_single_aug, len(sX)),
                            replace=False)
            trX = np.concatenate([bX, sX[aug]])
            trY = np.concatenate([bY, sY[aug]])
            if args.n_synth_normal > 0:
                npk = r2.choice(len(sX), size=min(args.n_synth_normal, len(sX)),
                                replace=False)
                nX = np.minimum(sX[npk], 0.5)
                trX = np.concatenate([trX, nX])
                trY = np.concatenate([trY, np.zeros((len(nX), K), np.float32)])
            if args.neg_target > 0:
                trY = trY + (1.0 - trY) * args.neg_target
            trX, trY = trX.astype(np.float32), trY.astype(np.float32)

            torch.manual_seed(seed)
            model = SmallCNN(num_classes=K, in_ch=1).to(args.device)
            opt = torch.optim.Adam(model.parameters(), lr=args.lr)
            lf = nn.BCEWithLogitsLoss()
            for _ in range(args.epochs):
                model.train()
                for xb, yb in _loader(trX, trY, args.bs, True):
                    xb, yb = xb.to(args.device), yb.to(args.device)
                    opt.zero_grad()
                    lf(model(xb), yb).backward()
                    opt.step()
            teP = _predict(model, teX, device=args.device)
            nrmP = _predict(model, nrmX, device=args.device)
            bf = bit_f1(teP, teY)
            pos, neg = pos_neg_prob(teP, teY)
            nfar = float((nrmP >= 0.5).any(axis=1).mean())
            print(f"[f={f:.2f} s{seed}] survival={surv:.3f} | REAL bitF1={bf:.4f} "
                  f"pos={pos:.4f} neg={neg:.4f} NORMAL_FAR={nfar:.4f}", flush=True)
            rows.append({"frac": f, "seed": seed, "survival": round(surv, 4),
                         "bitF1": round(bf, 4), "pos": round(pos, 4),
                         "neg": round(neg, 4), "normal_far": round(nfar, 4)})

    if args.out_csv:
        with open(args.out_csv, "w", newline="") as fh:
            w = csvmod.DictWriter(fh, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
        print(f"[OUT] {args.out_csv}", flush=True)


if __name__ == "__main__":
    main()
