"""Order-extrapolation on MixedWM38: train with PAIR synthesis only vs
pairs+triples(+quads), evaluate real mixed separately by combination order
(2-mix / 3-mix / 4-mix). Tests whether pair-trained synthesis extrapolates to
higher-order combinations it never saw.
"""
import argparse

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

from .datasets.mixedwm38 import load_wm38, split_groups
from .synthesis.wm38_arms import _pair_indices
from .models.small_cnn import SmallCNN
from .metrics import bit_f1, exact_match


def synth_overlay_k(X, Y, n, seed, ks=(2,)):
    """Overlay of k random distinct-class singles, k sampled from ks."""
    rng = np.random.default_rng(seed)
    lab = Y.argmax(1)
    out_x, out_y = [], []
    while len(out_x) < n:
        k = int(ks[int(rng.integers(0, len(ks)))])
        idx = rng.choice(len(X), size=4 * k, replace=False)
        pick, seen = [], set()
        for i in idx:
            if lab[i] not in seen:
                pick.append(i); seen.add(lab[i])
            if len(pick) == k:
                break
        if len(pick) < k:
            continue
        img = X[pick[0]].copy()
        y = Y[pick[0]].copy()
        for i in pick[1:]:
            img = np.maximum(img, X[i])
            y = np.maximum(y, Y[i])
        out_x.append(img); out_y.append(y)
    return np.stack(out_x).astype(np.float32), np.stack(out_y).astype(np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n-train", type=int, default=3000)
    ap.add_argument("--n-single-aug", type=int, default=2000)
    ap.add_argument("--n-synth-normal", type=int, default=2000)
    ap.add_argument("--neg-target", type=float, default=0.03)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--bs", type=int, default=64)
    ap.add_argument("--n-test-per-order", type=int, default=1500)
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()

    X, Y = load_wm38()
    g = split_groups(Y)
    rng = np.random.default_rng(12345)
    mixed = rng.permutation(g["mixed"])
    test_pool = mixed[len(mixed) // 2:]
    nb = Y.sum(1).astype(int)
    tests = {}
    for k in (2, 3, 4):
        pool = np.array([i for i in test_pool if nb[i] == k])
        take = min(args.n_test_per_order, len(pool))
        pick = rng.choice(pool, size=take, replace=False)
        tests[k] = (X[pick], Y[pick])
        print(f"test {k}-mix: {take}", flush=True)
    sX, sY = X[g["single"]], Y[g["single"]]
    nrmX = X[g["normal"]]

    for name, ks in [("pairs-only", (2,)), ("pairs+triples", (2, 3)),
                     ("pairs+tri+quad", (2, 3, 4))]:
        r2 = np.random.default_rng(args.seed)
        bX, bY = synth_overlay_k(sX, sY, args.n_train, args.seed, ks=ks)
        aug = r2.choice(len(sX), args.n_single_aug, replace=False)
        npick = r2.choice(len(sX), args.n_synth_normal, replace=False)
        trX = np.concatenate([bX, sX[aug], np.minimum(sX[npick], 0.5)])
        trY = np.concatenate([bY, sY[aug], np.zeros((args.n_synth_normal, 8), np.float32)])
        trY = trY + (1.0 - trY) * args.neg_target
        torch.manual_seed(args.seed)
        model = SmallCNN(num_classes=8, in_ch=1).to(args.device)
        opt = torch.optim.Adam(model.parameters(), lr=1e-3)
        lf = nn.BCEWithLogitsLoss()
        for _ in range(args.epochs):
            model.train()
            for xb, yb in DataLoader(TensorDataset(torch.from_numpy(trX), torch.from_numpy(trY)),
                                     batch_size=args.bs, shuffle=True):
                xb, yb = xb.to(args.device), yb.to(args.device)
                opt.zero_grad(); lf(model(xb), yb).backward(); opt.step()
        model.eval()
        def pred(Xa):
            P = []
            with torch.no_grad():
                for xb, in DataLoader(TensorDataset(torch.from_numpy(Xa)), batch_size=256):
                    P.append(torch.sigmoid(model(xb.to(args.device))).cpu().numpy())
            return np.concatenate(P)
        parts = []
        for k in (2, 3, 4):
            tX, tY = tests[k]
            P = pred(tX)
            parts.append(f"{k}mix bitF1={bit_f1(P, tY):.4f} exact={exact_match(P, tY):.4f}")
        nfar = float((pred(nrmX) >= 0.5).any(1).mean())
        print(f"[{name:14s}] " + " | ".join(parts) + f" | NORMAL_FAR={nfar:.4f}", flush=True)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
