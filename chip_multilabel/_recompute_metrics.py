"""Recompute per-class F1 from preds_chip.parquet using class_key as authoritative GT.

★ 260508 — bit_metrics.py was found to exclude 3-class combos (bb+fork+scratch etc)
from per_class_all calculations, causing massive miscounting (e.g., 19C bb F1 = 0.7339
instead of true 0.9928). This script fixes that.

GT bits derived from class_key:
- 'Normal', 'Invalid'                    → [0, 0, 0, 0]
- 'bank_boundary'                        → [1, 0, 0, 0]
- 'bank_boundary+fork'                   → [1, 1, 0, 0]
- 'bank_boundary+fork+scratch'           → [1, 1, 1, 0]   ← was excluded by bit_metrics
- ...
- 'fork+scratch+scratch_rot'             → [0, 1, 1, 1]
- '<X>+ood_<Y>'                          → bits of <X> only (OOD overlay)
- 'CenterCircle'/'RingDots'/etc          → [0, 0, 0, 0]   (wafer-pattern OOD, FAR-only)

ni_chipFAR = (Normal/Invalid GT chip 중 pred bit sum >= 1) / (Normal/Invalid GT chip 수)
"""
from __future__ import annotations
import argparse
import ast
import json
import sys
from pathlib import Path

import pandas as pd

CLASSES = ["bank_boundary", "fork", "scratch", "scratch_rot"]


def parse_set(s):
    if isinstance(s, str):
        return set(ast.literal_eval(s))
    return set(s) if s is not None else set()


def gt_bits_from_class_key(ck: str) -> list[int]:
    """Derive 4-bit GT from class_key string. Strips ood_* tokens (overlay = trained-only bits)."""
    bits = [0, 0, 0, 0]
    if ck in ("Normal", "Invalid"):
        return bits
    # wafer-pattern OOD = no defect bits
    OOD_WAFER = {"CenterCircle", "CenterDonut", "CrescentArc", "RingDots", "Row",
                 "DiagonalSmear", "CrossScratch", "ParallelScratches", "BrokenRing", "Starburst"}
    if ck in OOD_WAFER:
        return bits
    parts = ck.split("+")
    for p in parts:
        if p.startswith("ood_"):
            continue  # overlay decoration
        if p in CLASSES:
            bits[CLASSES.index(p)] = 1
    return bits


def compute_metrics(parquet_path: Path, cell: str = None) -> dict:
    df = pd.read_parquet(parquet_path)
    if cell is None:
        cells = df["cell_id"].unique()
        # pick first
        cell = cells[0]
    df = df[df["cell_id"] == cell].copy()
    df["pred_set"] = df["pred_labels"].apply(parse_set)
    df["gt_bits"] = df["class_key"].apply(gt_bits_from_class_key)
    for c in CLASSES:
        df[f"gt_{c}"] = df["gt_bits"].apply(lambda b, i=CLASSES.index(c): b[i])
        df[f"pred_{c}"] = df["pred_set"].apply(lambda s, c=c: int(c in s))
    n_total = len(df)
    per_class = {}
    for c in CLASSES:
        TP = int(((df[f"gt_{c}"] == 1) & (df[f"pred_{c}"] == 1)).sum())
        FP = int(((df[f"gt_{c}"] == 0) & (df[f"pred_{c}"] == 1)).sum())
        FN = int(((df[f"gt_{c}"] == 1) & (df[f"pred_{c}"] == 0)).sum())
        TN = int(((df[f"gt_{c}"] == 0) & (df[f"pred_{c}"] == 0)).sum())
        P = TP / (TP + FP) if (TP + FP) > 0 else 0.0
        R = TP / (TP + FN) if (TP + FN) > 0 else 0.0
        F1 = 2 * P * R / (P + R) if (P + R) > 0 else 0.0
        per_class[c] = {
            "TP": TP, "FP": FP, "FN": FN, "TN": TN,
            "P": round(P, 4), "R": round(R, 4), "F1": round(F1, 4),
        }
    macro_F1 = sum(per_class[c]["F1"] for c in CLASSES) / 4
    macro_P = sum(per_class[c]["P"] for c in CLASSES) / 4
    macro_R = sum(per_class[c]["R"] for c in CLASSES) / 4
    # micro F1
    total_TP = sum(per_class[c]["TP"] for c in CLASSES)
    total_FP = sum(per_class[c]["FP"] for c in CLASSES)
    total_FN = sum(per_class[c]["FN"] for c in CLASSES)
    micro_P = total_TP / (total_TP + total_FP) if (total_TP + total_FP) > 0 else 0.0
    micro_R = total_TP / (total_TP + total_FN) if (total_TP + total_FN) > 0 else 0.0
    micro_F1 = 2 * micro_P * micro_R / (micro_P + micro_R) if (micro_P + micro_R) > 0 else 0.0
    # ni_chipFAR (Normal/Invalid GT chip 중 pred bit sum >= 1)
    ni_mask = df["class_key"].isin(["Normal", "Invalid"])
    ni_n = int(ni_mask.sum())
    ni_fp = 0
    if ni_n > 0:
        for _, r in df[ni_mask].iterrows():
            if any(c in r["pred_set"] for c in CLASSES):
                ni_fp += 1
    ni_chipFAR = ni_fp / ni_n if ni_n > 0 else 0.0
    # ni_bitFAR (bit-level)
    ni_bit_fp = 0
    if ni_n > 0:
        for _, r in df[ni_mask].iterrows():
            for c in CLASSES:
                if c in r["pred_set"]:
                    ni_bit_fp += 1
    ni_bitFAR = ni_bit_fp / (ni_n * 4) if ni_n > 0 else 0.0
    return {
        "cell": cell,
        "n_total": n_total,
        "ni_n": ni_n,
        "macro_F1": round(macro_F1, 4),
        "macro_P": round(macro_P, 4),
        "macro_R": round(macro_R, 4),
        "micro_F1": round(micro_F1, 4),
        "ni_chipFAR": round(ni_chipFAR, 4),
        "ni_bitFAR": round(ni_bitFAR, 4),
        "ni_FAR_chip_count": ni_fp,
        "per_class": per_class,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--parquet", required=True, type=Path)
    ap.add_argument("--cell", default=None)
    ap.add_argument("--cells", default="T0__I3,T0__I6,T0__I7,T0__I10",
                    help="comma-separated cell list (default: 4 standard inference variants)")
    ap.add_argument("--out", default=None, type=Path)
    args = ap.parse_args()
    df_all = pd.read_parquet(args.parquet)
    available = set(df_all["cell_id"].unique())
    cells = [c.strip() for c in args.cells.split(",") if c.strip() and c.strip() in available]
    out = {}
    for cell in cells:
        out[cell] = compute_metrics(args.parquet, cell)
    if args.out:
        args.out.write_text(json.dumps(out, indent=2))
        print(f"[recompute] saved {args.out}")
    else:
        print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
