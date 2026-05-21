"""Per-bit multi-label metrics (iter 12, 260506).

★ 표준 multi-label classification evaluation:
  - Each chip = 4-bit GT × 4-bit pred → 4 binary classification problems
  - Per-class F1 (4 classes) + CF1 (macro) + OF1 (micro)
  - bit-FAR (NON_DEFECT_GT chips 의 FP bit ratio)
  - chip-FAR (NON_DEFECT_GT chips 의 ≥1 FP bit chip ratio)
  - 3plus_active 빈도 (over-firing 진단)

Reference papers:
  - Wang et al. 2016 CVPR (CNN-RNN) — CF1 / OF1 표준
  - Chen et al. 2019 CVPR (ML-GCN) — multi-label image classification 표준 표
  - Ridnik et al. 2021 ICCV (ASL) — multi-label loss + per-class F1
  - Tsoumakas & Katakis 2007 IJDWM — multi-label evaluation overview

Usage:
    python -m chip_multilabel._bit_metrics \\
        --parquet outputs/stage1_iter12/stage1_<TS>/preds_chip.parquet \\
        --cells T0__I3,T0__I7 [--out outputs/.../bit_metrics.json]
"""
from __future__ import annotations

import argparse
import ast
import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from .constants import (ALL_CLASS_KEYS, COMBO_KEYS, OOD_OVERLAY_KEYS, SINGLE_KEYS,
                        SPECIAL_KEYS, TRAIN_CLASSES, TRIPLE_COMBO_KEYS, WAFER_PATTERN_KEYS)

NON_DEFECT_GT_CLASSES = ("Normal", "Invalid") + WAFER_PATTERN_KEYS  # 6 classes (legacy bundled — Row dropped 260507)
# 260508 — bug fix: TRIPLE_COMBO_KEYS 누락 발견. 4 3-class combo (b+f+sc, b+f+sr, b+sc+sr, f+sc+sr)
# 가 per_class_all 카운트에서 빠져 19C bb F1 0.7339 (실제 0.9928) 같은 환영값 발생. 사용자 directive 정정.
DEFECT_GT_CLASSES = SINGLE_KEYS + COMBO_KEYS + TRIPLE_COMBO_KEYS + OOD_OVERLAY_KEYS  # 18 classes
POSITIVE_GT_CLASSES = SINGLE_KEYS + COMBO_KEYS  # paper-main positive chips: 4 single + 6 two-combo

# 260507 — split NON_DEFECT_GT into 3 groups (analyst Cycle A Step 1)
#   normal_invalid: ('Normal', 'Invalid')           ★ paper main (real-env target)
#   normal_only   : ('Normal',)                     (Normal 200 chip only)
#   ood           : WAFER_PATTERN_KEYS = 5 classes  (1000 chip OOD diagnostic)
NORMAL_INVALID_GT = ("Normal", "Invalid")
NORMAL_ONLY_GT = ("Normal",)
OOD_GT = WAFER_PATTERN_KEYS


def _binary_f1(gt_c: np.ndarray, pred_c: np.ndarray) -> Tuple[int, int, int, int, float, float, float]:
    tp = int(((gt_c == 1) & (pred_c == 1)).sum())
    fp = int(((gt_c == 0) & (pred_c == 1)).sum())
    fn = int(((gt_c == 1) & (pred_c == 0)).sum())
    tn = int(((gt_c == 0) & (pred_c == 0)).sum())
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
    return tp, fp, fn, tn, prec, rec, f1


def class_key_to_bits(class_key: str) -> np.ndarray:
    """class_key (e.g. 'fork+scratch', 'Normal', 'fork+scratch+ood_DiagonalSmear') -> 4-bit GT vector."""
    bits = np.zeros(len(TRAIN_CLASSES), dtype=np.int8)
    if class_key in NON_DEFECT_GT_CLASSES:
        return bits  # all zeros
    if class_key in SINGLE_KEYS:
        bits[TRAIN_CLASSES.index(class_key)] = 1
        return bits
    if class_key in COMBO_KEYS:
        for c in class_key.split("+"):
            bits[TRAIN_CLASSES.index(c)] = 1
        return bits
    if class_key in TRIPLE_COMBO_KEYS:
        # 260508 — bug fix. 3-class combo (b+f+sc, b+f+sr, b+sc+sr, f+sc+sr) 가 missing 이라
        # bb F1 0.7339 환영값 발생했음. trained-class 토큰만 추출.
        for c in class_key.split("+"):
            if c in TRAIN_CLASSES:
                bits[TRAIN_CLASSES.index(c)] = 1
        return bits
    if class_key in OOD_OVERLAY_KEYS:
        # 260507 — 2 trained + 1 OOD overlay. GT bits = 2 trained only ('<a>+<b>+ood_<OOD>').
        trained_part = class_key.split("+ood_")[0]
        for c in trained_part.split("+"):
            bits[TRAIN_CLASSES.index(c)] = 1
        return bits
    # 3plus_active or unknown — should NOT appear in GT (only in pred). zero out defensively.
    return bits


def labels_list_to_bits(labels) -> np.ndarray:
    """Convert pred/true labels list (e.g. ['fork', 'scratch']) -> 4-bit vector."""
    bits = np.zeros(len(TRAIN_CLASSES), dtype=np.int8)
    if isinstance(labels, str):
        # parquet round-trip may stringify list
        try:
            labels = ast.literal_eval(labels)
        except Exception:
            labels = []
    if labels is None:
        return bits
    for c in labels:
        if c in TRAIN_CLASSES:
            bits[TRAIN_CLASSES.index(c)] = 1
    return bits


def compute_bit_metrics(df: pd.DataFrame) -> Dict:
    """Compute all per-bit metrics from a dataframe of chip predictions.

    Required columns: class_key (GT), pred_labels (list of TRAIN_CLASSES names),
                      decision_type (for 3plus_active counting).
    """
    n = len(df)
    gt_bits = np.zeros((n, len(TRAIN_CLASSES)), dtype=np.int8)
    pred_bits = np.zeros((n, len(TRAIN_CLASSES)), dtype=np.int8)
    is_non_defect_gt = np.zeros(n, dtype=bool)
    is_defect_gt = np.zeros(n, dtype=bool)
    is_positive_gt = np.zeros(n, dtype=bool)

    for i, row in enumerate(df.itertuples(index=False)):
        ck = str(row.class_key)
        gt_bits[i] = class_key_to_bits(ck)
        pred_bits[i] = labels_list_to_bits(row.pred_labels)
        if ck in NON_DEFECT_GT_CLASSES:
            is_non_defect_gt[i] = True
        if ck in DEFECT_GT_CLASSES:
            is_defect_gt[i] = True
        if ck in POSITIVE_GT_CLASSES:
            is_positive_gt[i] = True

    # Per-class binary F1 over ALL chips (defect + non-defect)
    per_class: Dict[str, Dict[str, float]] = {}
    for ci, c in enumerate(TRAIN_CLASSES):
        gt_c = gt_bits[:, ci]
        pred_c = pred_bits[:, ci]
        tp, fp, fn, tn, prec, rec, f1 = _binary_f1(gt_c, pred_c)
        per_class[c] = {
            "tp": tp, "fp": fp, "fn": fn, "tn": tn,
            "precision": round(prec, 4), "recall": round(rec, 4),
            "f1": round(f1, 4),
        }

    # CF1 (macro F1 over 4 classes) — paper main
    cf1 = float(np.mean([per_class[c]["f1"] for c in TRAIN_CLASSES]))

    # OF1 (micro F1 over all 4×n bits)
    tp_all = sum(per_class[c]["tp"] for c in TRAIN_CLASSES)
    fp_all = sum(per_class[c]["fp"] for c in TRAIN_CLASSES)
    fn_all = sum(per_class[c]["fn"] for c in TRAIN_CLASSES)
    of1_p = tp_all / (tp_all + fp_all) if (tp_all + fp_all) > 0 else 0.0
    of1_r = tp_all / (tp_all + fn_all) if (tp_all + fn_all) > 0 else 0.0
    of1 = 2 * of1_p * of1_r / (of1_p + of1_r) if (of1_p + of1_r) > 0 else 0.0

    # FAR (False Alarm Rate) — NON_DEFECT_GT chips (GT all-zero) 의 FP
    n_non_def = int(is_non_defect_gt.sum())
    if n_non_def > 0:
        non_def_pred = pred_bits[is_non_defect_gt]  # (n_non_def, 4)
        fp_bits = int((non_def_pred == 1).sum())
        total_bits = 4 * n_non_def
        bit_far = fp_bits / total_bits if total_bits > 0 else 0.0
        chip_with_any_fp = int((non_def_pred.sum(axis=1) > 0).sum())
        chip_far = chip_with_any_fp / n_non_def
    else:
        bit_far = 0.0
        chip_far = 0.0
        fp_bits = 0
        total_bits = 0
        chip_with_any_fp = 0

    # 260507 ★ Cycle A Step 1: split NON_DEFECT_GT into 3 groups
    def _far_for_subset(class_keys: tuple) -> Dict:
        """Compute (bit_FAR, chip_FAR) for a subset of GT classes."""
        mask = np.zeros(n, dtype=bool)
        for i, row in enumerate(df.itertuples(index=False)):
            if str(row.class_key) in class_keys:
                mask[i] = True
        n_sub = int(mask.sum())
        if n_sub == 0:
            return {
                "bit_FAR": 0.0, "chip_FAR": 0.0,
                "FAR_chip_count": 0, "FAR_bit_count": 0,
                "FAR_total_bits": 0, "n_chips": 0,
            }
        sub_pred = pred_bits[mask]
        fp_b = int((sub_pred == 1).sum())
        tot_b = 4 * n_sub
        chip_w = int((sub_pred.sum(axis=1) > 0).sum())
        return {
            "bit_FAR": round(fp_b / tot_b, 4) if tot_b > 0 else 0.0,
            "chip_FAR": round(chip_w / n_sub, 4),
            "FAR_chip_count": chip_w,
            "FAR_bit_count": fp_b,
            "FAR_total_bits": tot_b,
            "n_chips": n_sub,
        }

    far_normal_invalid = _far_for_subset(NORMAL_INVALID_GT)  # ★ paper main
    far_normal_only = _far_for_subset(NORMAL_ONLY_GT)
    far_ood = _far_for_subset(OOD_GT)

    # 260507 — Cycle B: OOD overlay 4 class (2 trained + 1 OOD). GT bits = 2 trained only.
    # ood_overlay_chip_FAR := fraction of overlay chips with extra-bit (pred has any bit
    # not in GT 2-bit set). Stricter than normal chip_FAR — measures noise robustness.
    ood_overlay_per_class: Dict[str, Dict[str, float]] = {}
    ood_overlay_overall = {
        "n_chips": 0, "exact_2bit_count": 0, "partial_1bit_count": 0,
        "over_fire_count": 0, "miss_count": 0,
    }
    for ock in OOD_OVERLAY_KEYS:
        mask = np.zeros(n, dtype=bool)
        for i, row in enumerate(df.itertuples(index=False)):
            if str(row.class_key) == ock:
                mask[i] = True
        n_sub = int(mask.sum())
        if n_sub == 0:
            ood_overlay_per_class[ock] = {
                "n_chips": 0, "exact_2bit_recall": 0.0,
                "partial_1bit_rate": 0.0, "over_fire_rate": 0.0, "miss_rate": 0.0,
                "extra_bits_per_chip": 0.0,
            }
            continue
        sub_gt = gt_bits[mask]
        sub_pred = pred_bits[mask]
        # exact 2-bit: all 4 bits match (both GT bits fire + no extra bit)
        exact_match = (sub_gt == sub_pred).all(axis=1)
        # partial: GT bits fire ⊇ AND ⊋ pred (under-call: fewer bits than GT)
        gt_active_count = sub_gt.sum(axis=1)  # always 2 for overlay GT
        pred_active_count = sub_pred.sum(axis=1)
        # bit-wise: how many of the 2 GT bits fired?
        gt_fired = ((sub_gt == 1) & (sub_pred == 1)).sum(axis=1)  # 0, 1, or 2
        # extra bits fired (pred=1 outside GT)
        extra_bits = ((sub_gt == 0) & (sub_pred == 1)).sum(axis=1)
        # categories (mutually exclusive):
        #   exact_2bit: gt_fired == 2 AND extra_bits == 0
        #   partial_1bit: gt_fired == 1 AND extra_bits == 0
        #   over_fire: gt_fired == 2 AND extra_bits >= 1
        #   miss: gt_fired == 0 (regardless of extras)
        exact_2bit = int(((gt_fired == 2) & (extra_bits == 0)).sum())
        partial_1bit = int(((gt_fired == 1) & (extra_bits == 0)).sum())
        over_fire = int(((gt_fired == 2) & (extra_bits >= 1)).sum())
        partial_1bit_with_extra = int(((gt_fired == 1) & (extra_bits >= 1)).sum())
        miss = int((gt_fired == 0).sum())
        ood_overlay_per_class[ock] = {
            "n_chips": n_sub,
            "exact_2bit_recall": round(exact_2bit / n_sub, 4),
            "partial_1bit_rate": round(partial_1bit / n_sub, 4),
            "partial_1bit_with_extra_rate": round(partial_1bit_with_extra / n_sub, 4),
            "over_fire_rate": round(over_fire / n_sub, 4),
            "miss_rate": round(miss / n_sub, 4),
            "extra_bits_per_chip": round(float(extra_bits.mean()), 4),
        }
        ood_overlay_overall["n_chips"] += n_sub
        ood_overlay_overall["exact_2bit_count"] += exact_2bit
        ood_overlay_overall["partial_1bit_count"] += partial_1bit + partial_1bit_with_extra
        ood_overlay_overall["over_fire_count"] += over_fire
        ood_overlay_overall["miss_count"] += miss

    n_ovl = max(ood_overlay_overall["n_chips"], 1)
    ood_overlay_overall["exact_2bit_recall"] = round(ood_overlay_overall["exact_2bit_count"] / n_ovl, 4)
    ood_overlay_overall["partial_1bit_rate"] = round(ood_overlay_overall["partial_1bit_count"] / n_ovl, 4)
    ood_overlay_overall["over_fire_rate"] = round(ood_overlay_overall["over_fire_count"] / n_ovl, 4)
    ood_overlay_overall["miss_rate"] = round(ood_overlay_overall["miss_count"] / n_ovl, 4)
    # ood_overlay_chip_FAR := fraction with ANY extra bit beyond the 2 GT bits
    ood_overlay_overall["ood_overlay_chip_FAR"] = round(
        ood_overlay_overall["over_fire_count"] / n_ovl, 4
    )

    # 3plus_active 빈도
    decision_type_counts = df["decision_type"].value_counts().to_dict() if "decision_type" in df.columns else {}
    n_3plus = int(decision_type_counts.get("3plus_active", 0))

    # 추가: defect chip 만에서의 macro F1 (참고용 — paper main 은 전체 chip CF1)
    n_def = int(is_defect_gt.sum())
    per_class_def_only: Dict[str, Dict[str, float]] = {}
    if n_def > 0:
        gt_def = gt_bits[is_defect_gt]
        pred_def = pred_bits[is_defect_gt]
        for ci, c in enumerate(TRAIN_CLASSES):
            gt_c = gt_def[:, ci]
            pred_c = pred_def[:, ci]
            tp, fp, fn, _tn, prec, rec, f1 = _binary_f1(gt_c, pred_c)
            per_class_def_only[c] = {
                "tp": tp, "fp": fp, "fn": fn,
                "precision": round(prec, 4), "recall": round(rec, 4), "f1": round(f1, 4),
            }
    cf1_def_only = float(np.mean([per_class_def_only[c]["f1"]
                                  for c in TRAIN_CLASSES])) if per_class_def_only else 0.0

    # Paper-main positive-only bit F1: 4 single + 6 two-combo chips.
    # This intentionally excludes 3-combo and OOD-overlay diagnostics.
    n_positive = int(is_positive_gt.sum())
    per_bit_positive: Dict[str, Dict[str, float]] = {}
    if n_positive > 0:
        gt_pos = gt_bits[is_positive_gt]
        pred_pos = pred_bits[is_positive_gt]
        for ci, c in enumerate(TRAIN_CLASSES):
            gt_c = gt_pos[:, ci]
            pred_c = pred_pos[:, ci]
            tp, fp, fn, _tn, prec, rec, f1 = _binary_f1(gt_c, pred_c)
            per_bit_positive[c] = {
                "tp": tp, "fp": fp, "fn": fn,
                "precision": round(prec, 4), "recall": round(rec, 4), "f1": round(f1, 4),
            }
    cf1_positive = float(np.mean([per_bit_positive[c]["f1"]
                                  for c in TRAIN_CLASSES])) if per_bit_positive else 0.0

    # Per-negative-class FAR: chip-level FP rate for each non-defect class.
    per_class_far: Dict[str, Dict[str, float]] = {}
    for ck in NORMAL_INVALID_GT + OOD_GT:
        mask = np.array([str(row.class_key) == ck for row in df.itertuples(index=False)],
                        dtype=bool)
        n_sub = int(mask.sum())
        if n_sub == 0:
            per_class_far[ck] = {
                "bit_FAR": 0.0, "chip_FAR": 0.0,
                "FAR_chip_count": 0, "FAR_bit_count": 0,
                "FAR_total_bits": 0, "n_chips": 0,
            }
            continue
        sub_pred = pred_bits[mask]
        fp_b = int((sub_pred == 1).sum())
        tot_b = 4 * n_sub
        chip_w = int((sub_pred.sum(axis=1) > 0).sum())
        per_class_far[ck] = {
            "bit_FAR": round(fp_b / tot_b, 4) if tot_b > 0 else 0.0,
            "chip_FAR": round(chip_w / n_sub, 4),
            "FAR_chip_count": chip_w,
            "FAR_bit_count": fp_b,
            "FAR_total_bits": tot_b,
            "n_chips": n_sub,
        }

    return {
        "n_total_chips": int(n),
        "n_defect_gt": n_def,
        "n_positive_gt": n_positive,
        "n_non_defect_gt": n_non_def,
        "per_class_all": per_class,
        "per_class_defect_only": per_class_def_only,
        "per_bit_F1_positive": per_bit_positive,
        "per_class_FAR": per_class_far,
        # ★ paper main — macro F1 = CF1 (Wang 2016 / Chen 2019 명칭)
        "macro_F1": round(cf1, 4),
        "macro_F1_defect_only": round(cf1_def_only, 4),
        "macro_F1_positive": round(cf1_positive, 4),
        # micro F1 = OF1 (overall F1)
        "micro_F1": round(of1, 4),
        # FAR (legacy bundled — Normal+Invalid+5 OOD)
        "bit_FAR": round(bit_far, 4),
        "chip_FAR": round(chip_far, 4),
        "FAR_chip_count": chip_with_any_fp,
        "FAR_bit_count": fp_bits,
        "FAR_total_bits": total_bits,
        # 260507 ★ split FAR (Cycle A Step 1) — paper-worthy main metric
        "normal_invalid_bit_FAR": far_normal_invalid["bit_FAR"],   # ★ paper main
        "normal_invalid_chip_FAR": far_normal_invalid["chip_FAR"],  # ★ paper main
        "normal_invalid_FAR_chip_count": far_normal_invalid["FAR_chip_count"],
        "normal_invalid_n_chips": far_normal_invalid["n_chips"],
        "normal_only_bit_FAR": far_normal_only["bit_FAR"],
        "normal_only_chip_FAR": far_normal_only["chip_FAR"],
        "normal_only_FAR_chip_count": far_normal_only["FAR_chip_count"],
        "normal_only_n_chips": far_normal_only["n_chips"],
        "ood_bit_FAR": far_ood["bit_FAR"],
        "ood_chip_FAR": far_ood["chip_FAR"],
        "ood_FAR_chip_count": far_ood["FAR_chip_count"],
        "ood_n_chips": far_ood["n_chips"],
        # over-firing diagnostic
        "n_3plus_active": n_3plus,
        "frac_3plus_active": round(n_3plus / max(n, 1), 4),
        "decision_type_counts": decision_type_counts,
        # 260507 ★ Cycle B — OOD overlay 4 class metrics (2 trained + 1 OOD; GT = 2 bits)
        "ood_overlay_overall": ood_overlay_overall,
        "ood_overlay_per_class": ood_overlay_per_class,
        "ood_overlay_chip_FAR": ood_overlay_overall["ood_overlay_chip_FAR"],
        "ood_overlay_2bit_recall": ood_overlay_overall["exact_2bit_recall"],
        # legacy alias keys for backward compat
        "CF1_macro_all": round(cf1, 4),
        "OF1_micro_all": round(of1, 4),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--parquet", required=True,
                    help="path to preds_chip.parquet from stage1 output.")
    ap.add_argument("--cells", default="",
                    help="comma-separated cell_id subset (e.g. 'T0__I3,T0__I7'). "
                         "default = all cells in parquet.")
    ap.add_argument("--out", default="",
                    help="output JSON path (default = sibling 'bit_metrics.json' next to parquet).")
    args = ap.parse_args()

    parquet_path = Path(args.parquet)
    df_all = pd.read_parquet(parquet_path)
    cells = sorted(df_all["cell_id"].unique().tolist())
    if args.cells:
        wanted = set(args.cells.split(","))
        cells = [c for c in cells if c in wanted]

    print(f"[bit_metrics] parquet={parquet_path}  rows={len(df_all)}  cells={cells}")

    out: Dict[str, Dict] = {}
    for cell in cells:
        df_cell = df_all[df_all["cell_id"] == cell].reset_index(drop=True)
        metrics = compute_bit_metrics(df_cell)
        out[cell] = metrics
        print(f"\n=== {cell} ===")
        print(f"  eval bit_F1 (positive 4+6) = {metrics['macro_F1_positive']:.4f}  "
              f"(n={metrics['n_positive_gt']})")
        print(f"  macro F1 (= CF1)           = {metrics['macro_F1']:.4f}  ★ paper main")
        print(f"  macro F1 (defect chips)    = {metrics['macro_F1_defect_only']:.4f}")
        print(f"  micro F1 (= OF1)           = {metrics['micro_F1']:.4f}")
        print(f"  bit-FAR (legacy bundled)   = {metrics['bit_FAR']:.4f}  "
              f"({metrics['FAR_bit_count']}/{metrics['FAR_total_bits']} bits)")
        print(f"  chip-FAR (legacy bundled)  = {metrics['chip_FAR']:.4f}  "
              f"({metrics['FAR_chip_count']}/{metrics['n_non_defect_gt']} chips)")
        print(f"  ★ normal_invalid chip-FAR  = {metrics['normal_invalid_chip_FAR']:.4f}  "
              f"({metrics['normal_invalid_FAR_chip_count']}/{metrics['normal_invalid_n_chips']} chips)  ★ paper main")
        print(f"    normal_only chip-FAR     = {metrics['normal_only_chip_FAR']:.4f}  "
              f"({metrics['normal_only_FAR_chip_count']}/{metrics['normal_only_n_chips']} chips)")
        print(f"    ood chip-FAR             = {metrics['ood_chip_FAR']:.4f}  "
              f"({metrics['ood_FAR_chip_count']}/{metrics['ood_n_chips']} chips)  (diagnostic only)")
        print(f"  3plus_active count         = {metrics['n_3plus_active']}  "
              f"({100*metrics['frac_3plus_active']:.2f}%)")
        ovl = metrics.get("ood_overlay_overall", {})
        if ovl.get("n_chips", 0) > 0:
            print(f"  ★ ood_overlay 2bit_recall  = {ovl['exact_2bit_recall']:.4f}  "
                  f"over_fire={ovl['over_fire_rate']:.4f}  "
                  f"partial={ovl['partial_1bit_rate']:.4f}  miss={ovl['miss_rate']:.4f}  "
                  f"(n={ovl['n_chips']})")
        print(f"  eval bit_F1 by class:")
        for c in TRAIN_CLASSES:
            pc = metrics.get("per_bit_F1_positive", {}).get(c, {})
            print(f"    {c:14s}  F1={pc.get('f1', 0.0):.4f}  "
                  f"P={pc.get('precision', 0.0):.4f}  R={pc.get('recall', 0.0):.4f}  "
                  f"TP={pc.get('tp', 0)}  FP={pc.get('fp', 0)}  FN={pc.get('fn', 0)}")
        print(f"  eval FAR by class:")
        for c, pc in metrics.get("per_class_FAR", {}).items():
            print(f"    {c:14s}  chip_FAR={pc.get('chip_FAR', 0.0):.4f}  "
                  f"({pc.get('FAR_chip_count', 0)}/{pc.get('n_chips', 0)} chips)  "
                  f"bit_FAR={pc.get('bit_FAR', 0.0):.4f}")
        print(f"  per-class F1 (all chips):")
        for c in TRAIN_CLASSES:
            pc = metrics["per_class_all"][c]
            print(f"    {c:14s}  F1={pc['f1']:.4f}  P={pc['precision']:.4f}  "
                  f"R={pc['recall']:.4f}  TP={pc['tp']}  FP={pc['fp']}  FN={pc['fn']}")
        print(f"  decision_type counts: {metrics['decision_type_counts']}")

    out_path = Path(args.out) if args.out else (parquet_path.parent / "bit_metrics.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, default=str)
    print(f"\n[bit_metrics] saved → {out_path}")


if __name__ == "__main__":
    main()
