"""Cross-dataset operator-match flip on multiple PUBLIC image datasets.

Generalizes the MNIST operator-match demonstration (run_operator_match.py) to
several public torchvision single-object 28x28 grayscale sources. The builder
logic is dataset-agnostic: it only needs (image, class-label) pairs plus the
class count, so any such dataset is a drop-in source domain.

For EACH dataset we run the identical controlled protocol as the MNIST run:
  - 2x2-cell canvas (grid=2, cell=28 -> 56x56), single-label train / multi-label eval
  - two regimes: PARTITION (disjoint cells) and SUPERPOSITION (same cell, max-union)
  - four content-blind arms: overlay / partition / single_only(floor) / oracle
  - matched operator = the one whose synthesis law equals the domain's true law
      PARTITION regime      -> matched = partition-placement
      SUPERPOSITION regime  -> matched = overlay/max-union
  - 3 seeds, shared SmallCNN + Adam 1e-3 + BCEWithLogits recipe

Operator-match LAW prediction (the flip): matched arm wins in BOTH regimes,
with no mean+/-std overlap vs the mismatched arm, and recovers the oracle.

If a dataset does NOT flip cleanly, that is reported honestly (scope of the law),
not forced.

Produces one consolidated cross-dataset CSV + table + count of datasets on which
the flip holds.
"""
import os
import csv as csvmod
import argparse

from .datasets.multimnist import load_torchvision
from .run_operator_match import (run, print_table, print_verdict, flip_holds,
                                 REGIMES, ARMS, MATCHED, FIELDS)

# (label, torchvision name, split) -- MNIST anchor + additional public datasets.
DEFAULT_DATASETS = [
    ("mnist",          "MNIST",        None),
    ("fashionmnist",   "FashionMNIST", None),
    ("kmnist",         "KMNIST",       None),
    ("emnist_letters", "EMNIST",       "letters"),
]


def print_cross_table(rows, datasets):
    """One consolidated cross-dataset table: rows = dataset x regime."""
    print("\n=== CONSOLIDATED CROSS-DATASET OPERATOR-MATCH FLIP "
          "(bit_F1 mean +/- std over 3 seeds) ===")
    header = ("| Dataset         | Regime        | matched arm | matched bit_F1   "
              "| mismatched bit_F1 | oracle bit_F1 | flip? |")
    dash = ("|-----------------|---------------|-------------|------------------"
            "|-------------------|---------------|-------|")
    print(header)
    print(dash)
    from .run_operator_match import _agg
    n_flip = 0
    for label, _, _ in datasets:
        drows = [r for r in rows if r.get("dataset") == label]
        if not drows:
            continue
        flip_ok, details = flip_holds(rows, label)
        if flip_ok:
            n_flip += 1
        for regime in REGIMES:
            d = details[regime]
            bm, bmstd = d["bm"], d["bmstd"]
            bo, bostd = d["bo"], d["bostd"]
            oc = d["oracle"]
            # per-regime flip cell = matched beats mismatched w/o std overlap
            regime_flip = "yes" if (d["win"]) else "no"
            print(f"| {label:15s} | {regime:13s} | {d['matched']:11s} "
                  f"| {bm:.4f} +/- {bmstd:.4f} | {bo:.4f} +/- {bostd:.4f} "
                  f"| {oc:.4f}        | {regime_flip:5s} |")
    return n_flip


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+", default=None,
                    help="subset of dataset labels to run (default: all)")
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    ap.add_argument("--per-class-single", type=int, default=200)
    ap.add_argument("--n-train", type=int, default=3000)
    ap.add_argument("--n-test", type=int, default=2000)
    ap.add_argument("--n-normal", type=int, default=1000)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--bs", type=int, default=64)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--grid", type=int, default=2)
    ap.add_argument("--cell", type=int, default=28)
    ap.add_argument("--root", default="E:/data/torchvision")
    ap.add_argument("--out-csv",
                    default="outputs/multilabel_synth/operator_match_multidataset.csv")
    args = ap.parse_args()

    datasets = DEFAULT_DATASETS
    if args.datasets:
        want = set(args.datasets)
        datasets = [d for d in DEFAULT_DATASETS if d[0] in want]

    all_rows = []
    ran = []
    for label, tvname, split in datasets:
        try:
            tr_imgs, tr_labels, n_classes = load_torchvision(
                tvname, args.root, train=True, split=split)
            te_imgs, te_labels, _ = load_torchvision(
                tvname, args.root, train=False, split=split)
        except Exception as e:
            print(f"[SKIP] {label} ({tvname}) not available: {repr(e)[:200]}",
                  flush=True)
            continue
        print(f"\n########## {label} ({tvname}"
              f"{'/'+split if split else ''}) n_classes={n_classes} "
              f"train={len(tr_labels)} test={len(te_labels)} ##########",
              flush=True)
        rows = run(tr_imgs, tr_labels, te_imgs, te_labels, ARMS, args.seeds,
                   args.per_class_single, args.n_train, args.n_test,
                   args.n_normal, args.epochs, args.bs, args.device,
                   args.grid, args.cell, out_csv=None,
                   n_classes=n_classes, dataset=label)
        all_rows.extend(rows)
        ran.append((label, tvname, split))
        print_table(rows, label)
        print_verdict(rows, label)

    # consolidated CSV (force-added later by caller)
    if all_rows:
        os.makedirs(os.path.dirname(args.out_csv), exist_ok=True)
        with open(args.out_csv, "w", newline="") as f:
            w = csvmod.DictWriter(f, fieldnames=FIELDS)
            w.writeheader()
            w.writerows(all_rows)

    n_flip = print_cross_table(all_rows, ran)
    n_total = len(ran)
    print(f"\n=== PLAIN VERDICT ===")
    print(f"Operator-match FLIP holds on {n_flip} / {n_total} public datasets.")
    for label, _, _ in ran:
        ok, _ = flip_holds(all_rows, label)
        print(f"  {label:15s}: flip = {'YES' if ok else 'NO'}")
    if n_flip == n_total:
        print("=> The operator-match law is GENERAL across the tested public "
              "datasets (not MNIST-specific).")
    else:
        print("=> The operator-match law holds broadly but not universally; "
              "see per-dataset numbers above for scope.")
    print(f"\n[OUT] {os.path.abspath(args.out_csv)}")


if __name__ == "__main__":
    main()
