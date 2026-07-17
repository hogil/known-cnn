"""Operator-match COMPETITIVENESS sweep vs naive content-blind synthesis baselines.

Strengthens the operator-match demonstration for a broad venue: it is not enough
that the matched content-blind operator beats the single mismatched arm (overlay
vs partition). Here we show the matched operator also beats the obvious naive
content-blind combo-synthesis baselines that a practitioner would reach for when
they do NOT know the domain's true combination law:

  arm 'overlay'     : combos = same-cell max-union (superposition operator)
  arm 'partition'   : combos = distinct-cell placement (partition / FCM-PM op)
  arm 'cutmix'      : combos = random rectangular patch of one single canvas
                      pasted onto the other (standard CutMix), union label
  arm 'mixup'       : combos = 0.5/0.5 pixel average of the two single canvases,
                      union label
  arm 'single_only' : no combos (floor)
  arm 'oracle'      : combos under the domain's TRUE law (upper reference)

In each regime exactly ONE of {overlay, partition} is the matched operator; the
other, plus cutmix and mixup, are content-blind baselines that match neither a
clean partition nor a clean superposition. By the operator-match law the matched
operator should DOMINATE all three content-blind baselines (no std overlap) and
recover the oracle -- i.e. it is the best content-blind choice, not merely better
than overlay.

Everything else (SmallCNN, Adam 1e-3, BCEWithLogits, single sources, canvas,
seeds) is identical to run_operator_match.py; only the arm set is extended. If a
baseline ever ties/beats the matched operator that is reported honestly, since it
would bound the claim.
"""
import os
import csv as csvmod
import argparse

import numpy as np

from .datasets.multimnist import load_torchvision
from .run_operator_match import (run, ARMS_BASELINE, REGIMES, MATCHED,
                                 CONTENT_BLIND, FIELDS, _agg, dominance_holds)

DEFAULT_DATASETS = [
    ("mnist",          "MNIST",        None),
    ("fashionmnist",   "FashionMNIST", None),
    ("kmnist",         "KMNIST",       None),
    ("emnist_letters", "EMNIST",       "letters"),
]

# columns of the consolidated table: role -> (per-regime arm resolver)
COL_ARMS = ["matched", "mismatched", "cutmix", "mixup", "single_only", "oracle"]


def _arm_for(role, regime):
    if role == "matched":
        return MATCHED[regime]
    if role == "mismatched":
        return CONTENT_BLIND[regime][0]     # the other placement/overlay op
    return role                              # cutmix / mixup / single_only / oracle


def print_consolidated(rows, ran):
    """ONE consolidated table: dataset x regime rows, one bit_F1 mean+/-std cell
    per operator column. matched operator's actual arm name is shown."""
    print("\n=== CONSOLIDATED: operator-match vs content-blind baselines "
          "(bit_F1 mean +/- std over seeds) ===")
    head = ("| Dataset         | Regime        | matched arm | "
            "matched          | mismatched       | cutmix           | "
            "mixup            | single_only      | oracle           |")
    dash = ("|-----------------|---------------|-------------|"
            "------------------|------------------|------------------|"
            "-----------------|------------------|------------------|")
    print(head)
    print(dash)
    for label, _, _ in ran:
        for regime in REGIMES:
            cells = []
            for role in COL_ARMS:
                m, s = _agg(rows, regime, _arm_for(role, regime), "bit_f1", label)
                cells.append(f"{m:.4f}+/-{s:.4f}")
            print(f"| {label:15s} | {regime:13s} | {MATCHED[regime]:11s} "
                  f"| {cells[0]:16s} | {cells[1]:16s} | {cells[2]:16s} "
                  f"| {cells[3]:15s} | {cells[4]:16s} | {cells[5]:16s} |")


def print_dominance(rows, ran):
    print("\n=== DOMINANCE VERDICT (matched beats ALL content-blind baselines) ===")
    n_dom = 0
    any_beat = []
    for label, _, _ in ran:
        ok, details = dominance_holds(rows, label)
        n_dom += int(ok)
        for regime in REGIMES:
            d = details[regime]
            bases = ", ".join(
                f"{b}={d['per_base'][b]['mean']:.4f}+/-{d['per_base'][b]['std']:.4f}"
                f"{'(WIN)' if d['per_base'][b]['clean_win'] else '(OVERLAP)'}"
                for b in CONTENT_BLIND[regime])
            print(f"[{label:15s}][{regime:13s}] matched={d['matched']} "
                  f"bit_F1={d['bm']:.4f}+/-{d['bmstd']:.4f} | oracle={d['oracle']:.4f} "
                  f"recover={d['recovers_oracle']} | baselines: {bases} "
                  f"-> matched {'DOMINATES' if d['win'] else 'does NOT dominate'}")
            for base, bb, bbstd in d["beaten_by"]:
                any_beat.append((label, regime, base, bb, bbstd, d['bm'], d['bmstd']))
        print(f"  => {label}: matched dominates all content-blind baselines in "
              f"BOTH regimes = {'YES' if ok else 'NO'}")
    print(f"\n=== PLAIN VERDICT ===")
    print(f"Matched operator dominates ALL content-blind baselines (mismatched, "
          f"cutmix, mixup) with no std overlap AND recovers oracle in EVERY "
          f"regime on {n_dom} / {len(ran)} datasets.")
    if any_beat:
        print("Baselines that tied/beat the matched operator (bounds the claim):")
        for label, regime, base, bb, bbstd, bm, bmstd in any_beat:
            print(f"  {label}/{regime}: {base} bit_F1={bb:.4f}+/-{bbstd:.4f} "
                  f">= matched {bm:.4f}+/-{bmstd:.4f}")
    else:
        print("No content-blind baseline ever tied or beat the matched operator.")
    return n_dom


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+", default=None)
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
                    default="outputs/multilabel_synth/operator_match_baselines.csv")
    args = ap.parse_args()

    datasets = DEFAULT_DATASETS
    if args.datasets:
        want = set(args.datasets)
        datasets = [d for d in DEFAULT_DATASETS if d[0] in want]

    all_rows, ran = [], []
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
        rows = run(tr_imgs, tr_labels, te_imgs, te_labels, ARMS_BASELINE,
                   args.seeds, args.per_class_single, args.n_train, args.n_test,
                   args.n_normal, args.epochs, args.bs, args.device,
                   args.grid, args.cell, out_csv=None,
                   n_classes=n_classes, dataset=label)
        all_rows.extend(rows)
        ran.append((label, tvname, split))

    if all_rows:
        os.makedirs(os.path.dirname(args.out_csv), exist_ok=True)
        with open(args.out_csv, "w", newline="") as f:
            w = csvmod.DictWriter(f, fieldnames=FIELDS)
            w.writeheader()
            w.writerows(all_rows)

    print_consolidated(all_rows, ran)
    print_dominance(all_rows, ran)
    print(f"\n[OUT] {os.path.abspath(args.out_csv)}")


if __name__ == "__main__":
    main()
