"""Operator-match on REAL public remote-sensing land cover (DLRSD / UC-Merced).

A real, public, non-constructed PARTITION benchmark that is NEITHER digits NOR
wafers: an aerial tile's label set is the UNION of land-cover classes occupying
DISJOINT spatial regions (buildings here, trees there, grass there) -- the
partition combination law, on a mainstream remote-sensing dataset.

  SINGLE-LABEL TRAIN pool = real PURE single-class region crops harvested from
                            DLRSD pixel masks (one land-cover class each).
  REAL MULTI-LABEL EVAL   = real UC-Merced tiles with >= 2 of the chosen classes,
                            multi-hot over the class subset (which classes present).

Arms (all share the same real single-class crop pool; differ only in the operator
used to synthesize multi-label training tiles):

  single_only : crops only (floor)
  partition   : tile k crops into k DISJOINT quadrants (matched land-cover law)
  overlay     : per-pixel-MAX k crops into ONE quadrant (superposition, mismatched)
  cutmix      : random-rectangle paste (content-blind)
  mixup       : 0.5/0.5 blend (content-blind)
  oracle      : trained on REAL multi-label tiles (upper reference)

HYPOTHESIS: partition (matched to the disjoint-region land-cover law) beats overlay
AND cutmix/mixup on the REAL multi-label eval, and recovers the oracle.

Honesty gates (same as SVHN): oracle-collapse -> report+stop; partition tied/beaten
-> report plainly. A negative result honestly bounds the law's real-public generality.
"""
import os
import csv as csvmod
import argparse

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

from .datasets.dlrsd import prepare_landcover, DEFAULT_SUBSET
from .synthesis import landcover_ops as ops
from .models.resnet import build_resnet18
from .metrics import bit_f1, far, compute_map, pos_neg_prob

IMEAN = np.array([0.485, 0.456, 0.406], np.float32).reshape(1, 3, 1, 1)
ISTD = np.array([0.229, 0.224, 0.225], np.float32).reshape(1, 3, 1, 1)

ARMS = ["single_only", "partition", "overlay", "cutmix", "mixup", "oracle"]
MATCHED = "partition"
CONTENT_BLIND = ["overlay", "cutmix", "mixup"]
FIELDS = ["dataset", "arm", "matched", "backbone", "seed", "bit_f1", "mAP",
          "far", "normal_far", "pos_prob", "neg_prob", "n_train"]


def _norm(X):
    return (X - IMEAN) / ISTD


def train_model(trX, trY, epochs, bs, lr, device, seed, n_classes, pretrained):
    torch.manual_seed(seed)
    model = build_resnet18(n_classes, pretrained=pretrained).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    lossf = nn.BCEWithLogitsLoss()
    ds = TensorDataset(torch.from_numpy(_norm(trX)), torch.from_numpy(trY))
    loader = DataLoader(ds, batch_size=bs, shuffle=True)
    for _ in range(epochs):
        model.train()
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            lossf(model(xb), yb).backward()
            opt.step()
    return model


def _predict(model, X, bs, device):
    model.eval()
    P = []
    with torch.no_grad():
        for (xb,) in DataLoader(TensorDataset(torch.from_numpy(_norm(X))),
                                batch_size=bs):
            P.append(torch.sigmoid(model(xb.to(device))).cpu().numpy())
    return np.concatenate(P)


def build_train_for_arm(arm, pool_imgs, pool_labels, oracle_imgs, oracle_Y,
                        per_class_single, n_train, seed, cell, grid, n_classes,
                        k_choices):
    spX, spY = ops.build_singles(pool_imgs, pool_labels, per_class_single, seed,
                                 cell, grid, n_classes)
    if arm == "single_only":
        return spX, spY
    if arm == "oracle":
        oX = ops.real_to_chw(oracle_imgs)
        return np.concatenate([oX, spX]), np.concatenate([oracle_Y, spY])
    if arm in ("cutmix", "mixup"):
        cX, cY = ops.build_multi_baseline(pool_imgs, pool_labels, n_train, seed,
                                          arm, cell, grid, n_classes)
        return np.concatenate([cX, spX]), np.concatenate([cY, spY])
    law = "partition" if arm == "partition" else "superposition"
    cX, cY = ops.build_multi(pool_imgs, pool_labels, n_train, seed, law, cell,
                             grid, n_classes, k_choices=k_choices)
    return np.concatenate([cX, spX]), np.concatenate([cY, spY])


def run(data, arms, seeds, per_class_single, n_train, n_normal, epochs, bs, lr,
        device, cell, grid, k_choices, pretrained, out_csv, dataset="landcover"):
    pool_imgs, pool_labels, oracle_imgs, oracle_Y, eval_imgs, eval_Y, names, meta = data
    n_classes = len(names)
    evX = ops.real_to_chw(eval_imgs)
    nX, _ = ops.build_normal(n_normal, cell, grid, n_classes)
    rows = []
    for arm in arms:
        for seed in seeds:
            trX, trY = build_train_for_arm(
                arm, pool_imgs, pool_labels, oracle_imgs, oracle_Y,
                per_class_single, n_train, seed, cell, grid, n_classes, k_choices)
            model = train_model(trX, trY, epochs, bs, lr, device, seed,
                                n_classes, pretrained)
            P = _predict(model, evX, bs, device)
            nP = _predict(model, nX, bs, device)
            pos, neg = pos_neg_prob(P, eval_Y)
            r = {
                "dataset": dataset, "arm": arm, "matched": MATCHED,
                "backbone": "resnet18", "seed": seed,
                "bit_f1": round(float(bit_f1(P, eval_Y)), 4),
                "mAP": round(float(compute_map(P, eval_Y)), 4),
                "far": round(float(far(P, eval_Y)), 4),
                "normal_far": round(float((nP >= 0.5).any(1).mean()), 4),
                "pos_prob": round(float(pos), 4),
                "neg_prob": round(float(neg), 4),
                "n_train": int(trX.shape[0]),
            }
            rows.append(r)
            mark = "MATCH" if arm == MATCHED else ""
            print(f"[landcover] {arm:12s} s{seed} bitF1={r['bit_f1']:.4f} "
                  f"mAP={r['mAP']:.4f} FAR={r['far']:.4f} nFAR={r['normal_far']:.4f} "
                  f"pos={r['pos_prob']:.4f} neg={r['neg_prob']:.4f} "
                  f"n={r['n_train']} {mark}", flush=True)
    if out_csv:
        os.makedirs(os.path.dirname(out_csv), exist_ok=True)
        with open(out_csv, "w", newline="") as f:
            w = csvmod.DictWriter(f, fieldnames=FIELDS)
            w.writeheader(); w.writerows(rows)
    return rows


def _agg(rows, arm, key):
    vals = [r[key] for r in rows if r["arm"] == arm]
    return (float(np.mean(vals)), float(np.std(vals))) if vals else (float("nan"),
                                                                     float("nan"))


def print_table(rows, names, meta):
    print(f"\n=== OPERATOR-MATCH ON REAL DLRSD/UC-MERCED LAND COVER "
          f"(resnet18, mean +/- std over seeds) ===")
    print(f"    subset={names}  eval_pos={meta.get('eval_pos')}")
    print("| Arm         | matched | bit_F1 (mean+/-std) | mAP (mean+/-std)    "
          "| FAR    | nFAR   |")
    print("|-------------|---------|---------------------|---------------------"
          "|--------|--------|")
    for arm in ARMS:
        if not any(r["arm"] == arm for r in rows):
            continue
        bm, bs_ = _agg(rows, arm, "bit_f1")
        mm, ms = _agg(rows, arm, "mAP")
        fm, _ = _agg(rows, arm, "far")
        nm, _ = _agg(rows, arm, "normal_far")
        mk = "yes" if arm == MATCHED else ""
        print(f"| {arm:11s} | {mk:7s} | {bm:.4f} +/- {bs_:.4f}  "
              f"| {mm:.4f} +/- {ms:.4f} | {fm:.4f} | {nm:.4f} |")


def _clean_win_over(rows, metric, tol=0.0):
    """For a metric, list content-blind arms partition does NOT cleanly beat
    (std-bands must be disjoint). Returns (overlaps, beaten_by)."""
    bm, bmstd = _agg(rows, MATCHED, metric)
    overlaps, beaten_by, lines = [], [], []
    for base in CONTENT_BLIND:
        if not any(r["arm"] == base for r in rows):
            continue
        bb, bbstd = _agg(rows, base, metric)
        clean = (bm - bmstd) > (bb + bbstd)
        lines.append(f"    {MATCHED} {bm:.4f}+/-{bmstd:.4f}  vs  {base:8s} "
                     f"{bb:.4f}+/-{bbstd:.4f}  -> clean_win={clean}")
        if bb >= bm:
            beaten_by.append(base)
        if not clean:
            overlaps.append(base)
    return overlaps, beaten_by, lines


def print_verdict(rows, tol=0.02):
    print("\n=== VERDICT (REAL DLRSD/UC-Merced partition-law eval) ===")
    bm, bmstd = _agg(rows, MATCHED, "bit_f1")
    boc, _ = _agg(rows, "oracle", "bit_f1")
    osingle, _ = _agg(rows, "single_only", "bit_f1")
    bmap, _ = _agg(rows, MATCHED, "mAP")
    omap, _ = _agg(rows, "oracle", "mAP")
    # oracle-collapse gate: oracle must clear the single_only floor by a margin
    oracle_collapsed = not (boc > osingle + tol) if not np.isnan(boc) else True

    print("  [bit_F1] partition vs content-blind (base-rate/threshold sensitive):")
    f1_over, f1_beat, f1_lines = _clean_win_over(rows, "bit_f1")
    for ln in f1_lines:
        print(ln)
    print("  [mAP]    partition vs content-blind (threshold-free, base-rate-robust):")
    map_over, map_beat, map_lines = _clean_win_over(rows, "mAP")
    for ln in map_lines:
        print(ln)

    recovers_oracle = bm >= (boc - tol)
    recovers_oracle_map = bmap >= (omap - tol)
    print(f"  single_only floor bit_F1={osingle:.4f}")
    print(f"  oracle bit_F1={boc:.4f} mAP={omap:.4f} -> "
          f"recovers_oracle(bitF1)={recovers_oracle} recovers_oracle(mAP)="
          f"{recovers_oracle_map}; oracle_collapsed={oracle_collapsed}")

    f1_clean = len(f1_over) == 0
    map_clean = len(map_over) == 0
    win = f1_clean and map_clean and recovers_oracle and recovers_oracle_map \
        and not oracle_collapsed
    if oracle_collapsed:
        verdict = ("ORACLE-COLLAPSE: real multi-label oracle did not clear the "
                   "single_only floor -> eval/scale infeasible at this setting")
    elif f1_beat or map_beat:
        verdict = (f"NO-WIN: partition tied/beaten by bitF1:{f1_beat} mAP:{map_beat}")
    elif not (recovers_oracle and recovers_oracle_map):
        if map_clean and f1_clean:
            verdict = ("PARTIAL: partition beats content-blind on BOTH metrics but "
                       "does NOT recover oracle (sim-to-real gap)")
        else:
            verdict = ("PARTIAL/MIXED: partition wins on bit_F1 but ties content-blind "
                       "on base-rate-robust mAP {}, and does NOT recover oracle"
                       .format(map_over))
    elif not (map_clean and f1_clean):
        verdict = (f"TIE: partition best-mean but std-band overlaps on "
                   f"bitF1:{f1_over} mAP:{map_over}")
    else:
        verdict = "WIN: operator-match partition reproduces on REAL land cover"
    print(f"\nPLAIN VERDICT: {verdict}")
    print(f"  partition_clean_win_bitF1={f1_clean}  partition_clean_win_mAP={map_clean}"
          f"  recovers_oracle_bitF1={recovers_oracle}  recovers_oracle_mAP="
          f"{recovers_oracle_map}  oracle_collapsed={oracle_collapsed}  WIN={win}")
    return win


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arms", nargs="+", default=ARMS)
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    ap.add_argument("--subset", nargs="+", default=DEFAULT_SUBSET)
    ap.add_argument("--per-class-single", type=int, default=140)
    ap.add_argument("--n-train", type=int, default=2000)
    ap.add_argument("--n-oracle", type=int, default=1000)
    ap.add_argument("--n-eval", type=int, default=400)
    ap.add_argument("--n-normal", type=int, default=400)
    ap.add_argument("--per-class-cap", type=int, default=140)
    ap.add_argument("--purity", type=float, default=0.6)
    ap.add_argument("--min-side", type=int, default=48)
    ap.add_argument("--cell", type=int, default=64)
    ap.add_argument("--grid", type=int, default=2)
    ap.add_argument("--k-choices", nargs="+", type=int, default=[2])
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--bs", type=int, default=64)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--no-pretrained", action="store_true")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--base", default="E:/data/dlrsd_extracted/DLRSD")
    ap.add_argument("--cache", default=None)
    ap.add_argument("--out-csv",
                    default="outputs/multilabel_synth/operator_match_landcover.csv")
    args = ap.parse_args()

    if args.device == "cuda" and not torch.cuda.is_available():
        print("[warn] cuda unavailable -> cpu", flush=True)
        args.device = "cpu"

    canvas = args.cell * args.grid
    data = prepare_landcover(
        args.base, subset=args.subset, cell=args.cell, canvas=canvas,
        n_eval=args.n_eval, n_oracle=args.n_oracle, per_class_cap=args.per_class_cap,
        min_side=args.min_side, purity=args.purity, seed=0, cache=args.cache)
    print(f"[meta] {data[-1]}", flush=True)
    rows = run(data, args.arms, args.seeds, args.per_class_single, args.n_train,
               args.n_normal, args.epochs, args.bs, args.lr, args.device,
               args.cell, args.grid, tuple(args.k_choices),
               not args.no_pretrained, args.out_csv)
    print_table(rows, data[6], data[7])
    print_verdict(rows)
    print(f"\n[OUT] {os.path.abspath(args.out_csv)}")


if __name__ == "__main__":
    main()
