# -*- coding: utf-8 -*-
"""Build full-cap pcls diagnostics by reusing run_stage1 eval-split preds.

run_stage1 forwards all sampled records, but preds_chip.parquet stores only the
eval split.  For large caps, this helper forwards only the held-out val split
and combines it with preds_chip.parquet so pcls can cover the full requested
n_per_class without a second full-dataset forward.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from ._bit_metrics import class_key_to_bits
from .constants import (
    COMBO_KEYS,
    OOD_OVERLAY_KEYS,
    SINGLE_KEYS,
    TRAIN_CLASSES,
    TRIPLE_COMBO_KEYS,
    WAFER_PATTERN_KEYS,
)
from .eval_dataset import ChipEvalDataset, discover_records_runtime, stratified_val_eval_split
from .inference_variants import forward_all_logits
from .model_io import load_chip_backbone, select_train_logits


PROB_COLS = [
    "prob_bank_boundary",
    "prob_fork",
    "prob_scratch",
    "prob_scratch_rot",
]


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


def _print_diag(keys: list[str], probs: np.ndarray, tag: str, root: str, cap: int, meta: dict) -> None:
    truth = np.stack([class_key_to_bits(k) for k in keys]).astype(bool)
    pos_vals = probs[truth]
    neg_vals = probs[~truth]
    pos_prob = float(pos_vals.mean()) if pos_vals.size else float("nan")
    neg_prob = float(neg_vals.mean()) if neg_vals.size else float("nan")

    grp_neg = defaultdict(list)
    grp_pos = defaultdict(list)
    rows_by_key = defaultdict(list)
    for i, key in enumerate(keys):
        g = _group(key)
        t = truth[i]
        grp_neg[g].extend(probs[i][~t].tolist())
        grp_pos[g].extend(probs[i][t].tolist())
        rows_by_key[key].append(i)

    print(f"\n===== pos/neg prob diag  tag={tag or root} =====")
    print(f"model backbone={meta['backbone']} img={meta['img_size']} N={len(keys)} cap={cap}")
    print(f"OVERALL  pos_prob={pos_prob:.4f}  neg_prob={neg_prob:.4f}")
    print("--- by group ---")
    for g in ["single", "combo", "Normal", "Invalid", "OOD", "other"]:
        if g not in grp_neg and g not in grp_pos:
            continue
        pp = np.mean(grp_pos[g]) if grp_pos[g] else float("nan")
        nn = np.mean(grp_neg[g]) if grp_neg[g] else float("nan")
        print(f"  {g:8s}  pos_prob={pp:.4f} (n={len(grp_pos[g]):5d})  neg_prob={nn:.4f} (n={len(grp_neg[g]):5d})")

    ood_rows = [i for i, key in enumerate(keys) if _group(key) == "OOD"]
    if ood_rows:
        ood_mean = probs[ood_rows].mean(axis=0)
        print("--- OOD per-bit mean prob (all should be LOW) ---")
        print("  " + "  ".join(f"{c}={ood_mean[j]:.3f}" for j, c in enumerate(TRAIN_CLASSES)))

    print(f"--- PER-CLASS 4-bit mean prob + metric ({'/'.join(TRAIN_CLASSES)} | metric) ---")
    order = {"single": 0, "combo": 1, "Normal": 2, "Invalid": 3, "OOD": 4, "other": 5}
    thr = 0.5
    for key in sorted(rows_by_key, key=lambda k: (order.get(_group(k), 9), k)):
        idx = rows_by_key[key]
        p = probs[idx]
        m = p.mean(axis=0)
        gt = truth[idx]
        tflag = "".join(str(int(b)) for b in class_key_to_bits(key))
        g = _group(key)
        fired = p > thr
        pos_min_note = ""
        neg_max_note = ""
        if g in ("single", "combo"):
            recalls = []
            for j in range(p.shape[1]):
                if gt[:, j].sum():
                    recalls.append(fired[gt[:, j], j].mean())
            metric_name = "bit_F1"
            metric_val = float(np.mean(recalls)) if recalls else float("nan")
            pos_min = np.where(gt, p, np.inf).min(axis=1)
            neg_max = np.where(~gt, p, -np.inf).max(axis=1)
            pos_min_note = f"  pos_min_p10={np.quantile(pos_min, 0.10):.3f}  pos_min_p50={np.quantile(pos_min, 0.50):.3f}"
            if np.isfinite(neg_max).any():
                neg_max_note = f"  neg_max_p90={np.quantile(neg_max, 0.90):.3f}  neg_max_p95={np.quantile(neg_max, 0.95):.3f}"
        else:
            metric_name = "FAR"
            metric_val = float(fired.any(axis=1).mean())
            neg_max = p.max(axis=1)
            neg_max_note = f"  neg_max_p90={np.quantile(neg_max, 0.90):.3f}  neg_max_p95={np.quantile(neg_max, 0.95):.3f}"
        print(
            f"  PCLS {key:28s} GT[{tflag}] n={len(idx):4d}  "
            + "  ".join(f"{c}={m[j]:.3f}" for j, c in enumerate(TRAIN_CLASSES))
            + f"  {metric_name}={metric_val:.3f}"
            + pos_min_note
            + neg_max_note
        )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--root", required=True)
    ap.add_argument("--stage1-preds", required=True)
    ap.add_argument("--cell-id", default="T0__I10")
    ap.add_argument("--cap-per-class", type=int, required=True)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--num-workers", type=int, default=0)
    ap.add_argument("--sample-seed", type=int, default=42)
    ap.add_argument("--split-seed", type=int, default=42)
    ap.add_argument("--val-ratio", type=float, default=0.2)
    ap.add_argument("--tag", default="")
    args = ap.parse_args()

    device = torch.device(args.device)
    model, meta, keep = load_chip_backbone(args.model, device)

    records = discover_records_runtime(args.root, n_per_class=args.cap_per_class, seed=args.sample_seed)
    val_idx, eval_idx = stratified_val_eval_split(records, val_ratio=args.val_ratio, seed=args.split_seed)
    val_records = [records[i] for i in val_idx]
    print(
        f"[pcls_from_stage1] records={len(records)} val={len(val_idx)} eval={len(eval_idx)} "
        f"cap={args.cap_per_class} stage1={args.stage1_preds}",
        flush=True,
    )

    preds = pd.read_parquet(args.stage1_preds)
    if "cell_id" in preds.columns:
        preds = preds[preds["cell_id"] == args.cell_id].copy()
    missing_cols = [c for c in ["class_key", *PROB_COLS] if c not in preds.columns]
    if missing_cols:
        raise SystemExit(f"stage1 preds missing columns: {missing_cols}")
    eval_keys = preds["class_key"].astype(str).tolist()
    eval_probs = preds[PROB_COLS].to_numpy(dtype=np.float32)

    ds = ChipEvalDataset(val_records, img_size=meta["img_size"])
    logits_full = forward_all_logits(
        model,
        ds,
        device,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        progress_label="[pcls_from_stage1] val forward",
        progress_every=50,
    )
    val_logits = select_train_logits(logits_full, keep)
    val_probs = 1.0 / (1.0 + np.exp(-val_logits))
    val_keys = [r.class_key for r in val_records]

    keys = eval_keys + val_keys
    probs = np.concatenate([eval_probs, val_probs.astype(np.float32)], axis=0)
    counts = pd.Series(keys).value_counts().sort_index()
    print("[pcls_from_stage1] combined class counts:")
    for key, count in counts.items():
        print(f"  {key}: {count}")
    _print_diag(keys, probs, args.tag, args.root, args.cap_per_class, meta)


if __name__ == "__main__":
    main()
