import os
import csv as csvmod
import argparse

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

from .datasets.mixedwm38 import (
    load_wm38, split_groups, mixed_combos, split_holdout_combos, combo_key,
)
from .synthesis.wm38_arms import synth_wm38
from .models.small_cnn import SmallCNN
from .metrics import pos_neg_prob, bit_f1, far, exact_match

ARMS = ["oracle", "fcm_pm", "fcm_pm_pm", "overlay", "cutmix", "mixup", "single_only"]
FIELDS = ["arm", "seed",
          "tr_bitF1", "tr_pos", "tr_neg",
          "ev_bitF1", "ev_FAR", "ev_exact", "ev_pos", "ev_neg",
          "ho_bitF1", "ho_exact", "nrm_FAR", "n_train"]


def _loader(X, Y, bs, shuffle):
    return DataLoader(TensorDataset(torch.from_numpy(X), torch.from_numpy(Y)),
                      batch_size=bs, shuffle=shuffle)


def train_model(trX, trY, epochs, bs, lr, device, seed):
    torch.manual_seed(seed)
    model = SmallCNN(num_classes=trY.shape[1], in_ch=1).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    lf = nn.BCEWithLogitsLoss()
    for _ in range(epochs):
        model.train()
        for xb, yb in _loader(trX, trY, bs, True):
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            lf(model(xb), yb).backward()
            opt.step()
    return model


def predict(model, X, Y, bs, device):
    model.eval()
    P = []
    with torch.no_grad():
        for xb, _ in _loader(X, Y, bs, False):
            P.append(torch.sigmoid(model(xb.to(device))).cpu().numpy())
    return np.concatenate(P)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arms", nargs="+", default=ARMS)
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    ap.add_argument("--n-holdout-combos", type=int, default=6)
    ap.add_argument("--n-train", type=int, default=3000)
    ap.add_argument("--n-single-aug", type=int, default=2000)
    ap.add_argument("--n-test", type=int, default=3000)
    ap.add_argument("--n-holdout-test", type=int, default=1000)
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--bs", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--grid", type=int, default=9)
    ap.add_argument("--n-groups", type=int, default=3)
    ap.add_argument("--neg-target", type=float, default=0.0,
                    help="soft target for negative bits: tgt = y*1 + (1-y)*neg")
    ap.add_argument("--mpos", type=float, default=0.65,
                    help="pair-mask positive target (fcm_pm_pm)")
    ap.add_argument("--out-csv", default="outputs/multilabel_synth/wm38_matrix.csv")
    args = ap.parse_args()

    X, Y = load_wm38()
    g = split_groups(Y)
    rng = np.random.default_rng(12345)

    # mixed -> oracle-pool / test-pool split, then hold out combos
    mixed = rng.permutation(g["mixed"])
    half = len(mixed) // 2
    oracle_pool, test_pool = mixed[:half], mixed[half:]
    combos = mixed_combos(Y, g["mixed"])
    _, hold = split_holdout_combos(combos, args.n_holdout_combos, seed=777)
    hold_set = set(hold)

    oracle_train_idx = np.array([i for i in oracle_pool
                                 if combo_key(Y[i]) not in hold_set])
    test_full_idx = rng.choice(test_pool, size=min(args.n_test, len(test_pool)),
                               replace=False)
    ho_pool = np.array([i for i in test_pool if combo_key(Y[i]) in hold_set])
    test_ho_idx = rng.choice(ho_pool, size=min(args.n_holdout_test, len(ho_pool)),
                             replace=False)
    teX, teY = X[test_full_idx], Y[test_full_idx]
    hoX, hoY = X[test_ho_idx], Y[test_ho_idx]
    nrmX, nrmY = X[g["normal"]], Y[g["normal"]]
    sX_all, sY_all = X[g["single"]], Y[g["single"]]
    print(f"singles={len(sX_all)} oracle_train={len(oracle_train_idx)} "
          f"test_full={len(teX)} test_holdout={len(hoX)} (combos held out: {len(hold)}) "
          f"normals={len(nrmX)}", flush=True)

    rows = []
    for arm in args.arms:
        for seed in args.seeds:
            r2 = np.random.default_rng(seed)
            aug = r2.choice(len(sX_all), size=min(args.n_single_aug, len(sX_all)),
                            replace=False)
            if arm == "oracle":
                pick = r2.choice(oracle_train_idx,
                                 size=min(args.n_train, len(oracle_train_idx)),
                                 replace=False)
                bX, bY = X[pick], Y[pick]
            elif arm == "single_only":
                bX = np.empty((0,) + X.shape[1:], np.float32)
                bY = np.empty((0, Y.shape[1]), np.float32)
            elif arm == "fcm_pm_pm":
                bX, bY = synth_wm38("fcm_pm", sX_all, sY_all, args.n_train, seed,
                                    grid=args.grid, n_groups=args.n_groups,
                                    pair_mask=True, mpos=args.mpos)
            else:
                bX, bY = synth_wm38(arm, sX_all, sY_all, args.n_train, seed,
                                    grid=args.grid, n_groups=args.n_groups)
            trX = np.concatenate([bX, sX_all[aug]])
            trY = np.concatenate([bY, sY_all[aug]])
            if args.neg_target > 0:
                # pos/neg target independence: tgt = y*pos + (1-y)*neg (pos=1)
                trY = trY + (1.0 - trY) * args.neg_target

            model = train_model(trX, trY, args.epochs, args.bs, args.lr,
                                args.device, seed)
            sub = r2.choice(len(trX), size=min(400, len(trX)), replace=False)
            trP = predict(model, trX[sub], trY[sub], args.bs, args.device)
            trYb = (trY[sub] >= 0.5).astype(np.float32)
            teP = predict(model, teX, teY, args.bs, args.device)
            hoP = predict(model, hoX, hoY, args.bs, args.device)
            nrmP = predict(model, nrmX, nrmY, args.bs, args.device)

            tr_pos, tr_neg = pos_neg_prob(trP, trYb)
            ev_pos, ev_neg = pos_neg_prob(teP, teY)
            r = {"arm": arm, "seed": seed,
                 "tr_bitF1": round(bit_f1(trP, trYb), 4),
                 "tr_pos": round(tr_pos, 4), "tr_neg": round(tr_neg, 4),
                 "ev_bitF1": round(bit_f1(teP, teY), 4),
                 "ev_FAR": round(far(teP, teY), 4),
                 "ev_exact": round(exact_match(teP, teY), 4),
                 "ev_pos": round(ev_pos, 4), "ev_neg": round(ev_neg, 4),
                 "ho_bitF1": round(bit_f1(hoP, hoY), 4),
                 "ho_exact": round(exact_match(hoP, hoY), 4),
                 "nrm_FAR": round(float((nrmP >= 0.5).any(axis=1).mean()), 4),
                 "n_train": int(len(trX))}
            rows.append(r)
            print(f"{arm:12s} s{seed} | TRAIN bitF1={r['tr_bitF1']:.3f} pos={r['tr_pos']:.3f} "
                  f"neg={r['tr_neg']:.3f} | EVAL bitF1={r['ev_bitF1']:.3f} FAR={r['ev_FAR']:.3f} "
                  f"exact={r['ev_exact']:.3f} pos={r['ev_pos']:.3f} neg={r['ev_neg']:.3f} | "
                  f"HOLDOUT bitF1={r['ho_bitF1']:.3f} exact={r['ho_exact']:.3f} | "
                  f"NORMAL FAR={r['nrm_FAR']:.3f}", flush=True)

    os.makedirs(os.path.dirname(args.out_csv), exist_ok=True)
    with open(args.out_csv, "w", newline="") as f:
        w = csvmod.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        w.writerows(rows)
    print(f"[OUT] {os.path.abspath(args.out_csv)}", flush=True)


if __name__ == "__main__":
    main()
