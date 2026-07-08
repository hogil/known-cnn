"""Combo-support experiment: synthesize only combinations that actually occur
(the 29 real WM38 combos — domain knowledge an engineer has without any
labeled mixed images), vs arbitrary-combination synthesis. Tests whether the
oracle's real privilege (knowledge of the combination support) can be
substituted by cheap domain knowledge, recovering joint (exact) accuracy.
"""
import argparse

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

from .datasets.mixedwm38 import load_wm38, split_groups, combo_key, mixed_combos
from .models.small_cnn import SmallCNN
from .metrics import bit_f1, exact_match


def synth_from_support(X, Y, support, n, seed):
    """Overlay synthesis sampling target combos uniformly from the support."""
    rng = np.random.default_rng(seed)
    lab = Y.argmax(1)
    by = {c: np.where(lab == c)[0] for c in range(Y.shape[1])}
    out_x, out_y = [], []
    while len(out_x) < n:
        combo = support[int(rng.integers(0, len(support)))]
        bits = [i for i, v in enumerate(combo) if v == 1]
        if any(len(by[b]) == 0 for b in bits):
            continue
        img = None
        for b in bits:
            xi = X[int(rng.choice(by[b]))]
            img = xi.copy() if img is None else np.maximum(img, xi)
        out_x.append(img)
        out_y.append(np.array(combo, np.float32))
    return np.stack(out_x).astype(np.float32), np.stack(out_y)


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
        pick = rng.choice(pool, size=min(args.n_test_per_order, len(pool)), replace=False)
        tests[k] = (X[pick], Y[pick])
    sX, sY = X[g["single"]], Y[g["single"]]
    nrmX = X[g["normal"]]

    # combination support = the 29 combos that exist (domain knowledge only —
    # no mixed images or labels are used, just WHICH combos occur)
    support = [list(c) for c in mixed_combos(Y, g["mixed"])]
    print(f"support combos: {len(support)}", flush=True)

    configs = [
        ("support-combos", support),
        ("arbitrary-pairs", None),   # baseline: uniform random pairs
    ]
    for name, sup in configs:
        r2 = np.random.default_rng(args.seed)
        if sup is not None:
            bX, bY = synth_from_support(sX, sY, sup, args.n_train, args.seed)
        else:
            from .synthesis.wm38_arms import synth_wm38
            bX, bY = synth_wm38("overlay", sX, sY, args.n_train, args.seed)
        aug = r2.choice(len(sX), args.n_single_aug, replace=False)
        npick = r2.choice(len(sX), args.n_synth_normal, replace=False)
        trX = np.concatenate([bX, sX[aug], np.minimum(sX[npick], 0.5)])
        trY = np.concatenate([bY, sY[aug], np.zeros((args.n_synth_normal, 8), np.float32)])
        trY = trY + (1.0 - trY) * args.neg_target
        torch.manual_seed(args.seed)
        model = SmallCNN(num_classes=8, in_ch=1)
        opt = torch.optim.Adam(model.parameters(), lr=1e-3)
        lf = nn.BCEWithLogitsLoss()
        for _ in range(args.epochs):
            model.train()
            for xb, yb in DataLoader(TensorDataset(torch.from_numpy(trX), torch.from_numpy(trY)),
                                     batch_size=args.bs, shuffle=True):
                opt.zero_grad()
                lf(model(xb), yb).backward()
                opt.step()
        model.eval()
        def pred(Xa):
            P = []
            with torch.no_grad():
                for xb, in DataLoader(TensorDataset(torch.from_numpy(Xa)), batch_size=256):
                    P.append(torch.sigmoid(model(xb)).numpy())
            return np.concatenate(P)
        parts = []
        for k in (2, 3, 4):
            tX, tY = tests[k]
            P = pred(tX)
            parts.append(f"{k}mix bitF1={bit_f1(P, tY):.4f} exact={exact_match(P, tY):.4f}")
        nfar = float((pred(nrmX) >= 0.5).any(1).mean())
        print(f"[{name:15s}] " + " | ".join(parts) + f" | NORMAL_FAR={nfar:.4f}", flush=True)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
