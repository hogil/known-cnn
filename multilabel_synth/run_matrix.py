import os
import csv as csvmod
import argparse

import numpy as np

from .datasets.multimnist import (
    all_pairs, split_holdout, build_single_pool, synthesize_multi, load_mnist,
)
from .synthesis.arms import synthesize_arm
from .train import train_one

ARMS = ["oracle", "fcm_pm", "cutmix", "mixup", "copy_paste", "single_only"]
FIELDS = ["arm", "seed", "mAP_full", "mAP_holdout", "exact_full",
          "pos_prob_full", "neg_prob_full"]


def run_matrix(imgs, labels, arms, seeds, n_holdout,
               per_class_single, n_train, n_test, n_holdout_test,
               epochs, bs, device, out_csv):
    pairs = all_pairs(10)
    train_pairs, holdout_pairs = split_holdout(pairs, n_holdout, seed=12345)

    # fixed evaluation sets (seed-independent)
    testX_full, testY_full = synthesize_multi(imgs, labels, n_test, seed=999,
                                              allowed_pairs=pairs)
    testX_ho, testY_ho = synthesize_multi(imgs, labels, n_holdout_test, seed=998,
                                          allowed_pairs=holdout_pairs)

    rows = []
    for arm in arms:
        # oracle only sees train_pairs (never the held-out combos);
        # synthesis arms may generate every pair.
        allowed = train_pairs if arm == "oracle" else pairs
        for seed in seeds:
            trX, trY = synthesize_arm(arm, imgs, labels,
                                      n=n_train, seed=seed, allowed_pairs=allowed)
            # augment every arm with the single pool (shared single-label signal)
            spX, spY = build_single_pool(imgs, labels, per_class_single, seed=seed)
            trX = np.concatenate([trX, spX])
            trY = np.concatenate([trY, spY])

            full = train_one(trX, trY, testX_full, testY_full,
                             epochs=epochs, bs=bs, device=device, seed=seed)
            ho = train_one(trX, trY, testX_ho, testY_ho,
                           epochs=epochs, bs=bs, device=device, seed=seed)
            rows.append({
                "arm": arm, "seed": seed,
                "mAP_full": round(full["mAP"], 4),
                "mAP_holdout": round(ho["mAP"], 4),
                "exact_full": round(full["exact_match"], 4),
                "pos_prob_full": round(full["pos_prob"], 4),
                "neg_prob_full": round(full["neg_prob"], 4),
            })

    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    with open(out_csv, "w", newline="") as f:
        w = csvmod.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        w.writerows(rows)
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arms", nargs="+", default=ARMS)
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    ap.add_argument("--n-holdout", type=int, default=9)
    ap.add_argument("--per-class-single", type=int, default=200)
    ap.add_argument("--n-train", type=int, default=4000)
    ap.add_argument("--n-test", type=int, default=2000)
    ap.add_argument("--n-holdout-test", type=int, default=1000)
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--bs", type=int, default=64)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--mnist-root", default="E:/data/torchvision")
    ap.add_argument("--out-csv", default="outputs/multilabel_synth/multimnist_matrix.csv")
    args = ap.parse_args()

    imgs, labels = load_mnist(args.mnist_root, train=True)
    rows = run_matrix(imgs, labels, args.arms, args.seeds, args.n_holdout,
                      args.per_class_single, args.n_train, args.n_test,
                      args.n_holdout_test, args.epochs, args.bs, args.device,
                      args.out_csv)
    for r in rows:
        print(r)
    print(f"[OUT] {os.path.abspath(args.out_csv)}")


if __name__ == "__main__":
    main()
