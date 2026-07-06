"""pos/neg prob diagnostic — given a chip backbone ckpt, compute mean sigmoid prob
on GT-positive bits (pos prob) and GT-negative bits (neg prob), for a dataset root
with class_key subfolders (train pool or eval set).

pos prob = mean sigmoid over (sample,bit) where true_multihot==1  (higher better)
neg prob = mean sigmoid over (sample,bit) where true_multihot==0  (lower better)
Also breaks neg prob down by group (defect-neg / Normal / Invalid / OOD).

Usage:
  python -m chip_multilabel._posneg_prob_diag --model <ckpt> --root <dir> \
         --cap-per-class 50 --device cpu
"""
from __future__ import annotations
import argparse
from collections import defaultdict

import numpy as np
import torch

from .constants import (TRAIN_CLASSES, WAFER_PATTERN_KEYS, OOD_OVERLAY_KEYS,
                        COMBO_KEYS, TRIPLE_COMBO_KEYS, SINGLE_KEYS)
from .eval_dataset import ChipEvalDataset, discover_records_runtime
from .inference_variants import forward_all_logits
from .model_io import load_chip_backbone, select_train_logits


def _group(class_key: str) -> str:
    if class_key == "Normal":
        return "Normal"
    if class_key == "Invalid":
        return "Invalid"
    if class_key in WAFER_PATTERN_KEYS or class_key in OOD_OVERLAY_KEYS:
        return "OOD"
    if class_key in SINGLE_KEYS:
        return "single"
    if class_key in COMBO_KEYS or class_key in TRIPLE_COMBO_KEYS:
        return "combo"
    return "other"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--root", required=True)
    ap.add_argument("--cap-per-class", type=int, default=50)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--num-workers", type=int, default=0)
    ap.add_argument("--sample-seed", type=int, default=42)
    ap.add_argument("--tag", default="")
    args = ap.parse_args()

    device = torch.device(args.device)
    model, meta, keep = load_chip_backbone(args.model, device)

    records = discover_records_runtime(
        args.root,
        n_per_class=args.cap_per_class,
        seed=args.sample_seed,
    )

    ds = ChipEvalDataset(records, img_size=meta["img_size"])
    logits_full = forward_all_logits(model, ds, device,
                                    batch_size=args.batch_size,
                                    num_workers=args.num_workers,
                                    progress_label=f"[pcls] {args.tag or args.root}",
                                    progress_every=50)
    logits = select_train_logits(logits_full, keep)  # (N,4) np
    probs = 1.0 / (1.0 + np.exp(-logits))  # (N,4) sigmoid

    truth = np.stack([r.true_multihot for r in records]).astype(bool)  # (N,4)

    pos_vals = probs[truth]
    neg_vals = probs[~truth]
    pos_prob = float(pos_vals.mean()) if pos_vals.size else float("nan")
    neg_prob = float(neg_vals.mean()) if neg_vals.size else float("nan")

    # neg prob by group
    grp_neg = defaultdict(list)
    grp_pos = defaultdict(list)
    for i, r in enumerate(records):
        g = _group(r.class_key)
        t = truth[i]
        grp_neg[g].extend(probs[i][~t].tolist())
        grp_pos[g].extend(probs[i][t].tolist())

    # per-bit neg prob on the all-negative groups (where ALL leakage shows)
    # fork bit index for the canonical OOD-fork leak diagnostic
    fk_idx = TRAIN_CLASSES.index("fork")

    print(f"\n===== pos/neg prob diag  tag={args.tag or args.root} =====")
    print(f"model backbone={meta['backbone']} img={meta['img_size']} N={len(records)} cap={args.cap_per_class}")
    print(f"OVERALL  pos_prob={pos_prob:.4f}  neg_prob={neg_prob:.4f}")
    print("--- by group ---")
    for g in ["single", "combo", "Normal", "Invalid", "OOD", "other"]:
        if g not in grp_neg and g not in grp_pos:
            continue
        pp = np.mean(grp_pos[g]) if grp_pos[g] else float("nan")
        nn = np.mean(grp_neg[g]) if grp_neg[g] else float("nan")
        npos = len(grp_pos[g]); nneg = len(grp_neg[g])
        print(f"  {g:8s}  pos_prob={pp:.4f} (n={npos:5d})  neg_prob={nn:.4f} (n={nneg:5d})")

    # OOD per-bit mean (which defect bit leaks most on OOD)
    ood_rows = [i for i, r in enumerate(records) if _group(r.class_key) == "OOD"]
    if ood_rows:
        ood_mean = probs[ood_rows].mean(axis=0)
        print("--- OOD per-bit mean prob (all should be LOW) ---")
        print("  " + "  ".join(f"{c}={ood_mean[j]:.3f}" for j, c in enumerate(TRAIN_CLASSES)))

    # PER-CLASS 4-bit mean prob + per-class metric column.
    # metric: pos class (single/combo) -> bit_F1 (mean recall on GT-positive bits @ 0.5);
    #         neg class (Normal/Inv/OOD) -> FAR (fraction of samples where any bit > 0.5).
    print(f"--- PER-CLASS 4-bit mean prob + metric ({'/'.join(TRAIN_CLASSES)} | metric) ---")
    rows_by_key = defaultdict(list)
    for i, r in enumerate(records):
        rows_by_key[r.class_key].append(i)
    order = {"single": 0, "combo": 1, "Normal": 2, "Invalid": 3, "OOD": 4, "other": 5}
    THR = 0.5
    for key in sorted(rows_by_key, key=lambda k: (order.get(_group(k), 9), k)):
        idx = rows_by_key[key]
        p = probs[idx]                                # (n, 4)
        m = p.mean(axis=0)
        gt = truth[idx]                               # (n, 4) bool
        tflag = "".join(str(int(b)) for b in records[idx[0]].true_multihot)
        g = _group(key)
        fired = (p > THR)                             # (n, 4) predicted positive
        pos_min_note = ""
        neg_max_note = ""
        if g in ("single", "combo"):
            # bit_F1 = mean recall on GT-positive bits across samples
            recalls = []
            for j in range(p.shape[1]):
                if gt[:, j].sum():
                    recalls.append(fired[gt[:, j], j].mean())
            metric_name, metric_val = "bit_F1", float(np.mean(recalls)) if recalls else float("nan")
            pos_min = np.where(gt, p, np.inf).min(axis=1)
            neg_max = np.where(~gt, p, -np.inf).max(axis=1)
            pos_min_note = (
                f"  pos_min_p10={np.quantile(pos_min, 0.10):.3f}"
                f"  pos_min_p50={np.quantile(pos_min, 0.50):.3f}"
            )
            if np.isfinite(neg_max).any():
                neg_max_note = (
                    f"  neg_max_p90={np.quantile(neg_max, 0.90):.3f}"
                    f"  neg_max_p95={np.quantile(neg_max, 0.95):.3f}"
                )
        else:
            # FAR = fraction of samples where ANY bit fires (>THR) — all GT negative
            metric_name, metric_val = "FAR", float(fired.any(axis=1).mean())
            neg_max = p.max(axis=1)
            neg_max_note = (
                f"  neg_max_p90={np.quantile(neg_max, 0.90):.3f}"
                f"  neg_max_p95={np.quantile(neg_max, 0.95):.3f}"
            )
        print(f"  PCLS {key:28s} GT[{tflag}] n={len(idx):4d}  "
              + "  ".join(f"{c}={m[j]:.3f}" for j, c in enumerate(TRAIN_CLASSES))
              + f"  {metric_name}={metric_val:.3f}"
              + pos_min_note
              + neg_max_note)


if __name__ == "__main__":
    main()
