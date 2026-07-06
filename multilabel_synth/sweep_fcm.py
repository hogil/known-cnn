import os
import csv as csvmod
import argparse

import numpy as np

from .datasets.multimnist import (
    all_pairs, split_holdout, build_single_pool, synthesize_multi, load_mnist,
)
from .synthesis.arms import synthesize_arm
from .train import train_model, evaluate

FIELDS = ["label", "arm", "opts", "n_seeds",
          "mAP_full_mean", "mAP_full_std",
          "mAP_holdout_mean", "mAP_holdout_std",
          "exact_full_mean", "pos_mean", "neg_mean"]

# FCM-PM tuning sweep + fixed baselines. opts are passed to synthesize_arm.
CONFIGS = [
    ("oracle",        "oracle",      {}),
    ("cutmix_f0.50",  "cutmix",      {"cutmix_frac": 0.50}),
    ("cutmix_f0.25",  "cutmix",      {"cutmix_frac": 0.25}),
    ("copy_paste",    "copy_paste",  {}),
    ("single_only",   "single_only", {}),
    ("fcm_fill",      "fcm_pm",      {"fcm_mode": "fill"}),
    ("fcm_checker_g2","fcm_pm",      {"fcm_mode": "checker", "grid": 2}),
    ("fcm_checker_g3","fcm_pm",      {"fcm_mode": "checker", "grid": 3}),
    ("fcm_checker_g4","fcm_pm",      {"fcm_mode": "checker", "grid": 4}),
    ("fcm_strip_g2",  "fcm_pm",      {"fcm_mode": "strip", "grid": 2}),
    ("fcm_strip_g3",  "fcm_pm",      {"fcm_mode": "strip", "grid": 3}),
    ("fcm_strip_g4",  "fcm_pm",      {"fcm_mode": "strip", "grid": 4}),
]


def _mean_std(xs):
    a = np.array(xs, dtype=float)
    return float(a.mean()), float(a.std())


def run_config(imgs, labels, arm, opts, seeds, train_pairs, all_p,
               per_class_single, n_train, epochs, bs, device,
               tX_full, tY_full, tX_ho, tY_ho):
    allowed = train_pairs if arm == "oracle" else all_p
    fulls, hos, exacts, poss, negs = [], [], [], [], []
    for seed in seeds:
        trX, trY = synthesize_arm(arm, imgs, labels, n=n_train, seed=seed,
                                  allowed_pairs=allowed, **opts)
        spX, spY = build_single_pool(imgs, labels, per_class_single, seed=seed)
        trX = np.concatenate([trX, spX])
        trY = np.concatenate([trY, spY])
        model = train_model(trX, trY, epochs=epochs, bs=bs, device=device, seed=seed)
        f = evaluate(model, tX_full, tY_full, bs=bs, device=device)
        h = evaluate(model, tX_ho, tY_ho, bs=bs, device=device)
        fulls.append(f["mAP"]); hos.append(h["mAP"]); exacts.append(f["exact_match"])
        poss.append(f["pos_prob"]); negs.append(f["neg_prob"])
    fm, fs = _mean_std(fulls)
    hm, hs = _mean_std(hos)
    return {
        "mAP_full_mean": round(fm, 4), "mAP_full_std": round(fs, 4),
        "mAP_holdout_mean": round(hm, 4), "mAP_holdout_std": round(hs, 4),
        "exact_full_mean": round(float(np.mean(exacts)), 4),
        "pos_mean": round(float(np.mean(poss)), 4),
        "neg_mean": round(float(np.mean(negs)), 4),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    ap.add_argument("--n-holdout", type=int, default=9)
    ap.add_argument("--per-class-single", type=int, default=100)
    ap.add_argument("--n-train", type=int, default=2500)
    ap.add_argument("--n-test", type=int, default=1500)
    ap.add_argument("--n-holdout-test", type=int, default=750)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--bs", type=int, default=64)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--mnist-root", default="E:/data/torchvision")
    ap.add_argument("--out-csv", default="outputs/multilabel_synth/fcm_sweep.csv")
    args = ap.parse_args()

    imgs, labels = load_mnist(args.mnist_root, train=True)
    all_p = all_pairs(10)
    train_pairs, holdout_pairs = split_holdout(all_p, args.n_holdout, seed=12345)
    tX_full, tY_full = synthesize_multi(imgs, labels, args.n_test, seed=999, allowed_pairs=all_p)
    tX_ho, tY_ho = synthesize_multi(imgs, labels, args.n_holdout_test, seed=998, allowed_pairs=holdout_pairs)

    rows = []
    for label, arm, opts in CONFIGS:
        r = run_config(imgs, labels, arm, opts, args.seeds, train_pairs, all_p,
                       args.per_class_single, args.n_train, args.epochs, args.bs,
                       args.device, tX_full, tY_full, tX_ho, tY_ho)
        r = {"label": label, "arm": arm, "opts": str(opts), "n_seeds": len(args.seeds), **r}
        rows.append(r)
        print(f"{label:16s} full={r['mAP_full_mean']:.4f}+-{r['mAP_full_std']:.4f} "
              f"holdout={r['mAP_holdout_mean']:.4f}+-{r['mAP_holdout_std']:.4f} "
              f"exact={r['exact_full_mean']:.4f} pos={r['pos_mean']:.4f} neg={r['neg_mean']:.4f}",
              flush=True)

    os.makedirs(os.path.dirname(args.out_csv), exist_ok=True)
    with open(args.out_csv, "w", newline="") as f:
        w = csvmod.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        w.writerows(rows)
    print(f"[OUT] {os.path.abspath(args.out_csv)}")


if __name__ == "__main__":
    main()
