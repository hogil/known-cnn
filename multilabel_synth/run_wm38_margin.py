"""Stage-2/3 validation on MixedWM38: checkpoint selection by val-F1 vs
val-margin (both on a SYNTHETIC val set — no real mixed in the realistic
protocol), then margin/max-prob rejection at inference.

Generalized over the training arm so the chip winning recipe
(fcm_pm + val-margin + nb-reject) can be tested on the PUBLIC WM38 benchmark
against overlay and the equal-condition oracle.

Stage-2: val-F1 on synthetic val saturates early and picks undertrained
checkpoints; the pos-neg margin keeps growing and picks better ones.
Stage-3: rejecting low-max-prob samples (nb-reject) cuts residual NORMAL FAR
at modest coverage cost.
"""
import argparse
import copy
import csv as csvmod

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


def build_train(arm, sX, sY, oracleX, oracleY, args, seed):
    """Assemble training data for one arm, then add synthetic normals +
    neg-target uniformly (same FAR-control treatment for every arm incl.
    oracle, matching the full-scale protocol)."""
    r2 = np.random.default_rng(seed)
    if arm == "oracle":
        pick = r2.choice(len(oracleX), size=min(args.n_train, len(oracleX)),
                         replace=False)
        bX, bY = oracleX[pick], oracleY[pick]
    elif arm == "fcm_pm_pm":
        bX, bY = synth_wm38("fcm_pm", sX, sY, args.n_train, seed,
                            pair_mask=True, mpos=args.mpos)
    else:
        bX, bY = synth_wm38(arm, sX, sY, args.n_train, seed)
    aug = r2.choice(len(sX), size=min(args.n_single_aug, len(sX)), replace=False)
    trX = np.concatenate([bX, sX[aug]])
    trY = np.concatenate([bY, sY[aug]])
    if args.n_synth_normal > 0:
        npick = r2.choice(len(sX), size=min(args.n_synth_normal, len(sX)),
                          replace=False)
        nX = np.minimum(sX[npick], 0.5)
        trX = np.concatenate([trX, nX])
        trY = np.concatenate([trY, np.zeros((len(nX), sY.shape[1]), np.float32)])
    if args.neg_target > 0:
        trY = trY + (1.0 - trY) * args.neg_target
    return trX.astype(np.float32), trY.astype(np.float32)


def build_val(sX, sY, seed, mode):
    r2 = np.random.default_rng(seed + 4242)
    vs = r2.choice(len(sX), size=300, replace=False)
    vn = r2.choice(len(sX), size=300, replace=False)
    if mode == "combo":
        vX1, vY1 = synth_wm38("overlay", sX, sY, 600, seed + 777)
        vaX = np.concatenate([vX1, sX[vs], np.minimum(sX[vn], 0.5)])
        vaY = np.concatenate([vY1, sY[vs], np.zeros((300, sY.shape[1]), np.float32)])
    else:
        vs2 = r2.choice(len(sX), size=600, replace=False)
        vaX = np.concatenate([sX[vs2], sX[vs], np.minimum(sX[vn], 0.5)])
        vaY = np.concatenate([sY[vs2], sY[vs], np.zeros((300, sY.shape[1]), np.float32)])
    return vaX.astype(np.float32), vaY.astype(np.float32)


def run_arm(arm, sX, sY, oracleX, oracleY, teX, teY, nrmX, args, seed, rows):
    trX, trY = build_train(arm, sX, sY, oracleX, oracleY, args, seed)
    vaX, vaY = build_val(sX, sY, seed, args.val_mode)
    torch.manual_seed(seed)
    model = SmallCNN(num_classes=sY.shape[1], in_ch=1).to(args.device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    lf = nn.BCEWithLogitsLoss()
    best_f1, best_m = -1.0, -1e9
    st_f1, st_m, ep_f1, ep_m = None, None, -1, -1
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
        if vf1 > best_f1:
            best_f1, st_f1, ep_f1 = vf1, copy.deepcopy(model.state_dict()), ep
        if gap > best_m:
            best_m, st_m, ep_m = gap, copy.deepcopy(model.state_dict()), ep

    # stage-2: both picks on REAL mixed + real normals
    picks = {}
    for name, st, ep in [("f1", st_f1, ep_f1), ("margin", st_m, ep_m)]:
        model.load_state_dict(st)
        teP = _predict(model, teX, device=args.device)
        nrmP = _predict(model, nrmX, device=args.device)
        bf = bit_f1(teP, teY)
        nfar = float((nrmP >= 0.5).any(axis=1).mean())
        picks[name] = (teP, nrmP, bf, nfar, ep)
        print(f"[{arm:9s} s{seed}] {name}-pick ep{ep:02d} | REAL bitF1={bf:.4f} "
              f"NORMAL_FAR={nfar:.4f}", flush=True)

    # stage-3: nb-reject sweep on the margin-picked model
    teP, nrmP, bf_m, nfar_m, ep_m2 = picks["margin"]
    best_row = None
    for tau in (0.5, 0.6, 0.7, 0.8, 0.9):
        acc_m = teP.max(axis=1) >= tau
        acc_n = nrmP.max(axis=1) >= tau
        cov = float(acc_m.mean())
        bf = bit_f1(teP[acc_m], teY[acc_m]) if acc_m.any() else float("nan")
        nfar = float(((nrmP >= 0.5).any(axis=1) & acc_n).mean())
        print(f"[{arm:9s} s{seed}] stage3 tau{tau:.1f} | cov={cov:.3f} "
              f"bitF1={bf:.4f} NORMAL_FAR={nfar:.4f}", flush=True)
        # record the tau that keeps coverage >= 0.8 with lowest FAR
        if cov >= 0.80 and (best_row is None or nfar < best_row["s3_far"]):
            best_row = {"s3_tau": tau, "s3_cov": cov, "s3_bitF1": bf, "s3_far": nfar}
    if best_row is None:
        best_row = {"s3_tau": float("nan"), "s3_cov": float("nan"),
                    "s3_bitF1": float("nan"), "s3_far": float("nan")}
    rows.append({
        "arm": arm, "seed": seed,
        "f1pick_bitF1": picks["f1"][2], "f1pick_far": picks["f1"][3],
        "marginpick_bitF1": bf_m, "marginpick_far": nfar_m,
        **best_row,
    })


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arms", nargs="+",
                    default=["oracle", "overlay", "fcm_pm", "fcm_pm_pm"])
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    ap.add_argument("--n-train", type=int, default=3000)
    ap.add_argument("--n-single-aug", type=int, default=2000)
    ap.add_argument("--n-synth-normal", type=int, default=2000)
    ap.add_argument("--neg-target", type=float, default=0.03)
    ap.add_argument("--mpos", type=float, default=0.65)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--bs", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--n-test", type=int, default=3000)
    ap.add_argument("--val-mode", choices=["combo", "easy"], default="combo")
    ap.add_argument("--out-csv", default=None)
    args = ap.parse_args()

    X, Y = load_wm38()
    g = split_groups(Y)
    rng = np.random.default_rng(12345)
    mixed = rng.permutation(g["mixed"])
    half = len(mixed) // 2
    oracle_pool, test_pool = mixed[:half], mixed[half:]
    test_idx = rng.choice(test_pool, size=min(args.n_test, len(test_pool)),
                          replace=False)
    teX, teY = X[test_idx], Y[test_idx]
    oracleX, oracleY = X[oracle_pool], Y[oracle_pool]
    nrmX = X[g["normal"]]
    sX, sY = X[g["single"]], Y[g["single"]]
    print(f"singles={len(sX)} oracle_pool={len(oracleX)} test={len(teX)} "
          f"normals={len(nrmX)}", flush=True)

    rows = []
    for arm in args.arms:
        for seed in args.seeds:
            run_arm(arm, sX, sY, oracleX, oracleY, teX, teY, nrmX, args, seed, rows)

    if args.out_csv:
        with open(args.out_csv, "w", newline="") as f:
            w = csvmod.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
        print(f"[OUT] {args.out_csv}", flush=True)


if __name__ == "__main__":
    main()
