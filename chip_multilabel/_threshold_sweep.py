"""Post-hoc threshold sweep + Pareto frontier (iter 12 Phase 2.5, 260506).

★ User directive: "ls 평가시 pred에서 threshold를 잘못정한거 아닌가?
                   학습시 prob의 확률 분포를 확인하고 그거에 맞춰서 했어야"

스크립트 동작:
  1. stage1 parquet 의 raw prob (sigmoid output) 읽음 — 학습된 모델 logit 그대로 reuse
  2. threshold θ ∈ {0.10, 0.15, ..., 0.95} sweep — single global θ 적용
  3. 각 θ 별 (macro F1, bit-FAR, chip-FAR, fork F1, etc.) 산출
  4. Pareto frontier + composite optima 식별:
       - argmax macro F1 (현재 I3 와 비교)
       - ★ argmax macro F1 s.t. bit-FAR ≤ 5%   (★ 운영 제약 winner)
       - argmax (macro F1 - 1·bit-FAR), (macro F1 - 2·bit-FAR), (macro F1 - bit-FAR)
  5. (옵션) prob histogram per-class

학습 안 함. forward pass 한 번 한 stage1 결과 (probs) reuse. ~수초.

Usage:
    python -m chip_multilabel._threshold_sweep \\
      --parquet outputs/stage1_iter12/stage1_<TS>/preds_chip.parquet \\
      --cell T0__I3 \\
      [--out outputs/stage1_iter12/stage1_<TS>/threshold_sweep.json]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

from .constants import COMBO_KEYS, SINGLE_KEYS, TRAIN_CLASSES, WAFER_PATTERN_KEYS

NON_DEFECT_GT_CLASSES = ("Normal", "Invalid") + WAFER_PATTERN_KEYS  # 7 classes
DEFECT_GT_CLASSES = SINGLE_KEYS + COMBO_KEYS  # 10 classes


def class_key_to_bits(class_key: str) -> np.ndarray:
    bits = np.zeros(len(TRAIN_CLASSES), dtype=np.int8)
    if class_key in NON_DEFECT_GT_CLASSES:
        return bits
    if class_key in SINGLE_KEYS:
        bits[TRAIN_CLASSES.index(class_key)] = 1
        return bits
    if class_key in COMBO_KEYS:
        for c in class_key.split("+"):
            bits[TRAIN_CLASSES.index(c)] = 1
        return bits
    return bits


def evaluate_at_threshold(
    probs: np.ndarray,           # (N, 4)
    gt_bits: np.ndarray,         # (N, 4)
    is_non_defect: np.ndarray,   # (N,)
    is_invalid: np.ndarray,      # (N,) — bypass thresh, declared 'Invalid' regardless
    threshold: float,
) -> Dict:
    """Compute metrics at a given single threshold (per-class same)."""
    # Apply threshold (sigmoid output > θ). For Invalid chips, force pred=0 (handled by stage1).
    # We keep the 4-bit pred as is — Invalid GT chips contribute their bits as 0 (since GT=[0,0,0,0]
    # for NON_DEFECT_GT) and pred bits depend on prob > θ.
    pred = (probs > threshold).astype(np.int8)
    # If invalid heuristic detected, force pred to all-zero (mimic decision_tree behavior)
    pred[is_invalid] = 0

    n = probs.shape[0]
    per_class_f1 = []
    per_class_p = []
    per_class_r = []
    fp_per_class = []
    fn_per_class = []
    tp_per_class = []
    for ci in range(len(TRAIN_CLASSES)):
        gt_c = gt_bits[:, ci]
        pred_c = pred[:, ci]
        tp = int(((gt_c == 1) & (pred_c == 1)).sum())
        fp = int(((gt_c == 0) & (pred_c == 1)).sum())
        fn = int(((gt_c == 1) & (pred_c == 0)).sum())
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
        per_class_f1.append(f1)
        per_class_p.append(prec)
        per_class_r.append(rec)
        tp_per_class.append(tp)
        fp_per_class.append(fp)
        fn_per_class.append(fn)
    macro_f1 = float(np.mean(per_class_f1))

    # micro F1
    tp_all = sum(tp_per_class)
    fp_all = sum(fp_per_class)
    fn_all = sum(fn_per_class)
    of_p = tp_all / (tp_all + fp_all) if (tp_all + fp_all) > 0 else 0.0
    of_r = tp_all / (tp_all + fn_all) if (tp_all + fn_all) > 0 else 0.0
    micro_f1 = 2 * of_p * of_r / (of_p + of_r) if (of_p + of_r) > 0 else 0.0

    # FAR — NON_DEFECT_GT chips
    n_nd = int(is_non_defect.sum())
    if n_nd > 0:
        nd_pred = pred[is_non_defect]
        fp_bits = int((nd_pred == 1).sum())
        bit_far = fp_bits / (4 * n_nd)
        chip_with_fp = int((nd_pred.sum(axis=1) > 0).sum())
        chip_far = chip_with_fp / n_nd
    else:
        bit_far = 0.0
        chip_far = 0.0
        fp_bits = 0
        chip_with_fp = 0

    # 3plus_active
    n_active = pred.sum(axis=1)
    n_3plus = int((n_active >= 3).sum())

    return {
        "threshold": threshold,
        "macro_f1": round(macro_f1, 4),
        "micro_f1": round(micro_f1, 4),
        "bit_far": round(bit_far, 4),
        "chip_far": round(chip_far, 4),
        "f1_bb": round(per_class_f1[0], 4),
        "f1_fork": round(per_class_f1[1], 4),
        "f1_sc": round(per_class_f1[2], 4),
        "f1_sr": round(per_class_f1[3], 4),
        "n_3plus": n_3plus,
        "frac_3plus": round(n_3plus / max(n, 1), 4),
        "fp_bits": fp_bits,
        "chip_with_fp": chip_with_fp,
        "n_non_defect": n_nd,
    }


def find_pareto_frontier(sweep_results: List[Dict]) -> List[Dict]:
    """A point is Pareto-optimal if no other dominates it (higher macro_f1 AND lower bit_far)."""
    frontier = []
    for i, p in enumerate(sweep_results):
        dominated = False
        for j, q in enumerate(sweep_results):
            if i == j:
                continue
            if (q["macro_f1"] >= p["macro_f1"] and q["bit_far"] <= p["bit_far"] and
                (q["macro_f1"] > p["macro_f1"] or q["bit_far"] < p["bit_far"])):
                dominated = True
                break
        if not dominated:
            frontier.append(p)
    return frontier


def find_composite_optima(sweep: List[Dict]) -> Dict[str, Dict]:
    optima = {}
    # F1-only
    optima["argmax_macro_f1"] = max(sweep, key=lambda r: r["macro_f1"])
    # Constrained: bit-FAR <= 5%
    constrained = [r for r in sweep if r["bit_far"] <= 0.05]
    optima["argmax_macro_f1_s.t._bit_far<=5%"] = max(constrained, key=lambda r: r["macro_f1"]) if constrained else None
    # bit-FAR <= 1%
    constrained1 = [r for r in sweep if r["bit_far"] <= 0.01]
    optima["argmax_macro_f1_s.t._bit_far<=1%"] = max(constrained1, key=lambda r: r["macro_f1"]) if constrained1 else None
    # Soft trade-offs
    optima["argmax_macro_f1_minus_1xbit_far"] = max(sweep, key=lambda r: r["macro_f1"] - 1.0 * r["bit_far"])
    optima["argmax_macro_f1_minus_2xbit_far"] = max(sweep, key=lambda r: r["macro_f1"] - 2.0 * r["bit_far"])
    optima["argmax_macro_f1_minus_5xbit_far"] = max(sweep, key=lambda r: r["macro_f1"] - 5.0 * r["bit_far"])
    return optima


def sweep_for_cell(df: pd.DataFrame, thresholds: List[float]) -> Dict:
    """Run sweep for a single cell dataframe."""
    n = len(df)
    probs = df[["prob_bank_boundary", "prob_fork", "prob_scratch", "prob_scratch_rot"]].values
    is_invalid = (df["decision_type"] == "invalid").values

    # GT bits + non_defect mask
    gt_bits = np.zeros((n, len(TRAIN_CLASSES)), dtype=np.int8)
    is_non_defect = np.zeros(n, dtype=bool)
    for i, ck in enumerate(df["class_key"].values):
        gt_bits[i] = class_key_to_bits(str(ck))
        if str(ck) in NON_DEFECT_GT_CLASSES:
            is_non_defect[i] = True

    sweep = []
    for thr in thresholds:
        m = evaluate_at_threshold(probs, gt_bits, is_non_defect, is_invalid, thr)
        sweep.append(m)

    optima = find_composite_optima(sweep)
    pareto = find_pareto_frontier(sweep)

    # Per-class prob histograms (positive vs negative)
    hist = {}
    for ci, c in enumerate(TRAIN_CLASSES):
        pos_mask = gt_bits[:, ci] == 1
        neg_mask = gt_bits[:, ci] == 0
        bins = np.arange(0, 1.01, 0.05)
        hist[c] = {
            "n_pos": int(pos_mask.sum()),
            "n_neg": int(neg_mask.sum()),
            "pos_hist": np.histogram(probs[pos_mask, ci], bins=bins)[0].tolist(),
            "neg_hist": np.histogram(probs[neg_mask, ci], bins=bins)[0].tolist(),
            "bin_edges": bins.round(3).tolist(),
            "pos_p10": float(np.quantile(probs[pos_mask, ci], 0.10)) if pos_mask.any() else None,
            "pos_p50": float(np.quantile(probs[pos_mask, ci], 0.50)) if pos_mask.any() else None,
            "pos_p90": float(np.quantile(probs[pos_mask, ci], 0.90)) if pos_mask.any() else None,
            "neg_p10": float(np.quantile(probs[neg_mask, ci], 0.10)) if neg_mask.any() else None,
            "neg_p50": float(np.quantile(probs[neg_mask, ci], 0.50)) if neg_mask.any() else None,
            "neg_p90": float(np.quantile(probs[neg_mask, ci], 0.90)) if neg_mask.any() else None,
        }

    return {
        "n_chips": n,
        "thresholds": thresholds,
        "sweep": sweep,
        "pareto_frontier": pareto,
        "composite_optima": optima,
        "prob_distribution": hist,
    }


def print_summary(label: str, result: Dict):
    print(f"\n=== {label} ===")
    print(f"  Total chips: {result['n_chips']}")
    print(f"\n  ★ Composite optima:")
    opt = result["composite_optima"]
    keys_print = [
        ("argmax_macro_f1", "argmax macro F1 (current default)"),
        ("argmax_macro_f1_s.t._bit_far<=5%", "argmax macro F1 s.t. bit-FAR ≤ 5% ★"),
        ("argmax_macro_f1_s.t._bit_far<=1%", "argmax macro F1 s.t. bit-FAR ≤ 1%"),
        ("argmax_macro_f1_minus_1xbit_far", "argmax (macro F1 − 1·bit-FAR)"),
        ("argmax_macro_f1_minus_2xbit_far", "argmax (macro F1 − 2·bit-FAR)"),
        ("argmax_macro_f1_minus_5xbit_far", "argmax (macro F1 − 5·bit-FAR)"),
    ]
    for k, lbl in keys_print:
        o = opt.get(k)
        if o is None:
            print(f"    {lbl:55s}  (no feasible θ)")
        else:
            print(f"    {lbl:55s}  θ={o['threshold']:.2f}  "
                  f"macro={o['macro_f1']:.4f}  bit-FAR={o['bit_far']:.4f}  "
                  f"3plus={o['frac_3plus']:.3f}  "
                  f"F1[bb,fork,sc,sr]=[{o['f1_bb']:.2f},{o['f1_fork']:.2f},"
                  f"{o['f1_sc']:.2f},{o['f1_sr']:.2f}]")
    print(f"\n  Pareto frontier ({len(result['pareto_frontier'])} pts):")
    for p in sorted(result["pareto_frontier"], key=lambda r: r["threshold"]):
        print(f"    θ={p['threshold']:.2f}  macro={p['macro_f1']:.4f}  "
              f"bit-FAR={p['bit_far']:.4f}  chip-FAR={p['chip_far']:.4f}  "
              f"3plus={p['frac_3plus']:.3f}")
    print(f"\n  Per-class prob distribution (positive class chips vs negative):")
    for c in TRAIN_CLASSES:
        h = result["prob_distribution"][c]
        print(f"    {c:14s}  pos n={h['n_pos']} p10/50/90={h['pos_p10']:.3f}/{h['pos_p50']:.3f}/{h['pos_p90']:.3f}  "
              f"|  neg n={h['n_neg']} p10/50/90={h['neg_p10']:.3f}/{h['neg_p50']:.3f}/{h['neg_p90']:.3f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--parquet", required=True)
    ap.add_argument("--cell", default="T0__I3",
                    help="cell_id to sweep (uses I3 by default since I3 = sigmoid + F1-max).")
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    parquet_path = Path(args.parquet)
    df_all = pd.read_parquet(parquet_path)
    df = df_all[df_all["cell_id"] == args.cell].reset_index(drop=True)
    if len(df) == 0:
        raise ValueError(f"cell {args.cell} not in parquet (have: {df_all['cell_id'].unique().tolist()})")

    thresholds = [round(t, 3) for t in np.arange(0.10, 0.96, 0.05)]
    print(f"[threshold_sweep] {parquet_path}  cell={args.cell}  n={len(df)}  "
          f"thresholds={len(thresholds)} pts {thresholds[0]}..{thresholds[-1]}")
    result = sweep_for_cell(df, thresholds)
    print_summary(f"{parquet_path.parent.name} / {args.cell}", result)

    out_path = Path(args.out) if args.out else (parquet_path.parent / "threshold_sweep.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, default=str)
    print(f"\n[threshold_sweep] saved → {out_path}")


if __name__ == "__main__":
    main()
