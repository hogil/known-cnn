"""Stage-2/3 validation on MixedWM38: checkpoint selection by val-F1 vs
val-margin (both computed on a SYNTHETIC val set — no real mixed available in
the realistic protocol), then margin-based rejection at inference.

Stage-2 claim: val-F1 on synthetic val saturates early and picks undertrained
checkpoints; the pos-neg margin keeps growing and picks better ones.
Stage-3 claim: rejecting low-confidence samples cuts residual NORMAL FAR
further at modest coverage cost.
"""
import os
import copy
import argparse

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

from .datasets.mixedwm38 import load_wm38, split_groups
from .synthesis.wm38_arms import synth_wm38
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n-train", type=int, default=3000)
    ap.add_argument("--n-single-aug", type=int, default=2000)
    ap.add_argument("--n-synth-normal", type=int, default=2000)
    ap.add_argument("--neg-target", type=float, default=0.03)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--bs", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--n-test", type=int, default=3000)
    ap.add_argument("--val-mode", choices=["combo", "easy"], default="combo",
                    help="combo: synthetic combos in val (hard). easy: singles+"
                         "normals only (in-dist, saturates like the chip val)")
    args = ap.parse_args()

    X, Y = load_wm38()
    g = split_groups(Y)
    rng = np.random.default_rng(12345)
    mixed = rng.permutation(g["mixed"])
    test_idx = rng.choice(mixed[len(mixed) // 2:], size=args.n_test, replace=False)
    teX, teY = X[test_idx], Y[test_idx]
    nrmX = X[g["normal"]]
    sX, sY = X[g["single"]], Y[g["single"]]

    r2 = np.random.default_rng(args.seed)
    # train = winner recipe (overlay + singles + synthetic normals, neg003)
    bX, bY = synth_wm38("overlay", sX, sY, args.n_train, args.seed)
    aug = r2.choice(len(sX), size=args.n_single_aug, replace=False)
    npick = r2.choice(len(sX), size=args.n_synth_normal, replace=False)
    nX = np.minimum(sX[npick], 0.5)
    trX = np.concatenate([bX, sX[aug], nX])
    trY = np.concatenate([bY, sY[aug], np.zeros((len(nX), 8), np.float32)])
    trY = trY + (1.0 - trY) * args.neg_target

    # val set: "combo" = hard synthetic combos (F1 stays informative);
    # "easy" = singles + normals only (in-dist, saturates like the chip val)
    vs = r2.choice(len(sX), size=300, replace=False)
    vn = r2.choice(len(sX), size=300, replace=False)
    if args.val_mode == "combo":
        vX1, vY1 = synth_wm38("overlay", sX, sY, 600, args.seed + 777)
        vaX = np.concatenate([vX1, sX[vs], np.minimum(sX[vn], 0.5)])
        vaY = np.concatenate([vY1, sY[vs], np.zeros((300, 8), np.float32)])
    else:
        vs2 = r2.choice(len(sX), size=600, replace=False)
        vaX = np.concatenate([sX[vs2], sX[vs], np.minimum(sX[vn], 0.5)])
        vaY = np.concatenate([sY[vs2], sY[vs], np.zeros((300, 8), np.float32)])

    torch.manual_seed(args.seed)
    model = SmallCNN(num_classes=8, in_ch=1).to(args.device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    lf = nn.BCEWithLogitsLoss()

    best_f1, best_m = -1.0, -1e9
    st_f1, st_m, ep_f1, ep_m = None, None, -1, -1
    print("ep | val_f1  val_gap | note", flush=True)
    for ep in range(1, args.epochs + 1):
        model.train()
        for xb, yb in _loader(trX, trY, args.bs, True):
            xb, yb = xb.to(args.device), yb.to(args.device)
            opt.zero_grad()
            lf(model(xb), yb).backward()
            opt.step()
        P = _predict(model, vaX, device=args.device)
        vf1 = bit_f1(P, vaY)
        pos, neg = pos_neg_prob(P, vaY)
        gap = pos - neg
        note = []
        if vf1 > best_f1:
            best_f1, st_f1, ep_f1 = vf1, copy.deepcopy(model.state_dict()), ep
            note.append("F1-pick")
        if gap > best_m:
            best_m, st_m, ep_m = gap, copy.deepcopy(model.state_dict()), ep
            note.append("margin-pick")
        print(f"{ep:2d} | {vf1:.4f}  {gap:.4f} | {'+'.join(note)}", flush=True)

    # stage-2: evaluate both checkpoints on REAL mixed + normals
    for name, st, ep in [("f1-pick", st_f1, ep_f1), ("margin-pick", st_m, ep_m)]:
        model.load_state_dict(st)
        teP = _predict(model, teX, device=args.device)
        nrmP = _predict(model, nrmX, device=args.device)
        nfar = float((nrmP >= 0.5).any(axis=1).mean())
        print(f"[stage2] {name:11s} ep{ep:02d} | REAL bitF1={bit_f1(teP, teY):.4f} "
              f"exact={float(((teP>=0.5)==(teY>=0.5)).all(axis=1).mean()):.4f} "
              f"NORMAL_FAR={nfar:.4f}", flush=True)

    # stage-3: margin reject sweep on margin-picked model
    model.load_state_dict(st_m)
    teP = _predict(model, teX, device=args.device)
    nrmP = _predict(model, nrmX, device=args.device)
    print("[stage3] tau | mixed_coverage bitF1(accepted) | NORMAL_FAR(after reject)", flush=True)
    for tau in (0.5, 0.6, 0.7, 0.8, 0.9):
        acc_m = teP.max(axis=1) >= tau
        acc_n = nrmP.max(axis=1) >= tau
        cov = float(acc_m.mean())
        bf = bit_f1(teP[acc_m], teY[acc_m]) if acc_m.any() else float("nan")
        nfar = float(((nrmP >= 0.5).any(axis=1) & acc_n).mean())
        print(f"[stage3] {tau:.1f} | {cov:.3f} {bf:.4f} | {nfar:.4f}", flush=True)


if __name__ == "__main__":
    main()
