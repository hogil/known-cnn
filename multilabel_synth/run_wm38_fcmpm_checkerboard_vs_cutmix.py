"""Fair WM38 head-to-head: does FCM-PM on the CHECKERBOARD layout (random/full
excluded), swept over grid-group count g, grid resolution, and negative-target
label smoothing, beat cutmix?

Reuses the exact building blocks and protocol of run_wm38_margin.py
(synth_wm38 / SmallCNN / bit_f1 / val-margin checkpoint pick / REAL mixed +
REAL normal eval) so every arm is trained and scored identically. neg-target is
applied uniformly to every arm (same FAR-control treatment). Self-contained;
does not modify any existing runner.

Headline context (make_figs, main protocol): cutmix 0.691, FCM-PM 0.663
(that FCM-PM used neg=0 -> FAR blew up). The prior neg-target sweep on the
RANDOM layout (g3 grid9) already reached 0.7225 at neg=0.2 > cutmix, but the
CHECKERBOARD + neg-smoothing cell of the grid was never run. This fills it in.
"""
import argparse
import copy
import csv as csvmod
import os

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


def build_train(cfg, sX, sY, args, seed):
    """One arm's training data + synthetic normals + uniform neg-target.
    cfg = dict(arm, cell_layout, grid, n_groups, neg_target)."""
    r2 = np.random.default_rng(seed)
    if cfg["arm"] == "fcm_pm":
        bX, bY = synth_wm38("fcm", sX, sY, args.n_train, seed,
                            grid=cfg["grid"], n_groups=cfg["n_groups"],
                            pair_mask=True, cell_layout=cfg["cell_layout"],
                            complete_label_scale=args.complete_label_scale,
                            pair_fill=args.fcm_pair_fill)
    else:  # cutmix (or any other single-op arm); grid/groups ignored internally
        bX, bY = synth_wm38(cfg["arm"], sX, sY, args.n_train, seed,
                            grid=cfg["grid"], n_groups=cfg["n_groups"],
                            complete_label_scale=args.complete_label_scale,
                            pair_fill=args.fcm_pair_fill)
    aug = r2.choice(len(sX), size=min(args.n_single_aug, len(sX)), replace=False)
    trX = np.concatenate([bX, sX[aug]])
    trY = np.concatenate([bY, sY[aug]])
    if args.n_synth_normal > 0:
        npick = r2.choice(len(sX), size=min(args.n_synth_normal, len(sX)),
                          replace=False)
        nX = np.minimum(sX[npick], 0.5)
        trX = np.concatenate([trX, nX])
        trY = np.concatenate([trY, np.zeros((len(nX), sY.shape[1]), np.float32)])
    neg = cfg["neg_target"]
    if neg > 0:
        trY = trY + (1.0 - trY) * neg
    return trX.astype(np.float32), trY.astype(np.float32)


def build_val(sX, sY, seed):
    r2 = np.random.default_rng(seed + 4242)
    vs = r2.choice(len(sX), size=300, replace=False)
    vn = r2.choice(len(sX), size=300, replace=False)
    vX1, vY1 = synth_wm38("overlay", sX, sY, 600, seed + 777)
    vaX = np.concatenate([vX1, sX[vs], np.minimum(sX[vn], 0.5)])
    vaY = np.concatenate([vY1, sY[vs], np.zeros((300, sY.shape[1]), np.float32)])
    return vaX.astype(np.float32), vaY.astype(np.float32)


def run_cfg(cfg, sX, sY, teX, teY, nrmX, args, seed):
    trX, trY = build_train(cfg, sX, sY, args, seed)
    vaX, vaY = build_val(sX, sY, seed)
    torch.manual_seed(seed)
    model = SmallCNN(num_classes=sY.shape[1], in_ch=1).to(args.device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    lf = nn.BCEWithLogitsLoss()
    best_m, st_m, ep_m = -1e9, None, -1
    for ep in range(1, args.epochs + 1):
        model.train()
        for xb, yb in _loader(trX, trY, args.bs, True):
            xb, yb = xb.to(args.device), yb.to(args.device)
            opt.zero_grad()
            lf(model(xb), yb).backward()
            opt.step()
        P = _predict(model, vaX, device=args.device)
        pos, neg = pos_neg_prob(P, vaY)
        if pos - neg > best_m:
            best_m, st_m, ep_m = pos - neg, copy.deepcopy(model.state_dict()), ep
    model.load_state_dict(st_m)
    teP = _predict(model, teX, device=args.device)
    nrmP = _predict(model, nrmX, device=args.device)
    bf = bit_f1(teP, teY)
    nfar = float((nrmP >= 0.5).any(axis=1).mean())
    # iso-FAR: threshold each arm to a common NORMAL-FAR target, then read bitF1
    # there (tau = the (1-alpha) quantile of per-normal-sample max prob).
    nrm_max = nrmP.max(axis=1)
    iso = {}
    for a in (0.01, 0.05, 0.10):
        tau = float(np.quantile(nrm_max, 1.0 - a))
        iso[f"bitF1_farq{int(a*100):02d}"] = float(bit_f1(teP, teY, thr=tau))
    print(f"[{cfg['tag']:26s} s{seed}] margin ep{ep_m:02d} | bitF1={bf:.4f} FAR={nfar:.3f} "
          f"| iso-FAR bitF1 @1/5/10%="
          f"{iso['bitF1_farq01']:.3f}/{iso['bitF1_farq05']:.3f}/{iso['bitF1_farq10']:.3f}",
          flush=True)
    return {"tag": cfg["tag"], "arm": cfg["arm"], "cell_layout": cfg["cell_layout"],
            "grid": cfg["grid"], "n_groups": cfg["n_groups"],
            "neg_target": cfg["neg_target"], "seed": seed,
            "real_bitF1": bf, "normal_FAR": nfar, "best_ep": ep_m, **iso}


def build_configs():
    cfgs = []
    for neg in (0.05, 0.10, 0.20):
        cfgs.append({"tag": f"cutmix_neg{neg:.2f}", "arm": "cutmix",
                     "cell_layout": "n/a", "grid": 9, "n_groups": 3, "neg_target": neg})
    for (ng, grid) in [(3, 9), (4, 16), (3, 12)]:
        for neg in (0.05, 0.10, 0.20):
            cfgs.append({"tag": f"fcmpm_cb_g{ng}_grid{grid}_neg{neg:.2f}",
                         "arm": "fcm_pm", "cell_layout": "checkerboard",
                         "grid": grid, "n_groups": ng, "neg_target": neg})
    for neg in (0.10, 0.20):  # random g3grid9 sanity (should reproduce ~0.72)
        cfgs.append({"tag": f"fcmpm_rand_g3_grid9_neg{neg:.2f}", "arm": "fcm_pm",
                     "cell_layout": "random", "grid": 9, "n_groups": 3, "neg_target": neg})
    return cfgs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    ap.add_argument("--n-train", type=int, default=3000)
    ap.add_argument("--n-single-aug", type=int, default=2000)
    ap.add_argument("--n-synth-normal", type=int, default=2000)
    ap.add_argument("--complete-label-scale", type=float, default=1.0)
    ap.add_argument("--fcm-pair-fill", choices=["corner"], default="corner")
    ap.add_argument("--epochs", type=int, default=25)
    ap.add_argument("--bs", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--n-test", type=int, default=3000)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--out-csv",
                    default="outputs/multilabel_synth/wm38_fcmpm_checkerboard_vs_cutmix.csv")
    args = ap.parse_args()

    X, Y = load_wm38()
    g = split_groups(Y)
    rng = np.random.default_rng(12345)
    mixed = rng.permutation(g["mixed"])
    half = len(mixed) // 2
    test_pool = mixed[half:]
    test_idx = rng.choice(test_pool, size=min(args.n_test, len(test_pool)), replace=False)
    teX, teY = X[test_idx], Y[test_idx]
    nrmX = X[g["normal"]]
    sX, sY = X[g["single"]], Y[g["single"]]
    print(f"device={args.device} singles={len(sX)} test={len(teX)} normals={len(nrmX)}",
          flush=True)

    cfgs = build_configs()
    rows = []
    for cfg in cfgs:
        for seed in args.seeds:
            rows.append(run_cfg(cfg, sX, sY, teX, teY, nrmX, args, seed))

    os.makedirs(os.path.dirname(args.out_csv), exist_ok=True)
    with open(args.out_csv, "w", newline="") as f:
        w = csvmod.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"[OUT] {os.path.abspath(args.out_csv)}", flush=True)

    # aggregate
    import collections
    agg = collections.defaultdict(list)
    for r in rows:
        agg[r["tag"]].append((r["real_bitF1"], r["normal_FAR"]))
    print("\n=== WM38 FCM-PM(checkerboard) vs cutmix -- REAL bitF1 (val-margin pick), mean over seeds ===")
    print(f"{'tag':30s} {'bitF1':>7s} {'nFAR':>6s}")
    for tag, vals in sorted(agg.items(), key=lambda kv: -np.mean([v[0] for v in kv[1]])):
        bf = np.mean([v[0] for v in vals]); nf = np.mean([v[1] for v in vals])
        print(f"{tag:30s} {bf:7.4f} {nf:6.3f}")


if __name__ == "__main__":
    main()
