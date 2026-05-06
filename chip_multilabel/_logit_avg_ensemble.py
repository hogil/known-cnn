"""Logit-avg (prob-avg) post-hoc ensemble of two trained chip multi-label models.

Cycle A Step 2 (260507):
  Average sigmoid probabilities of two models (weighted) on the same eval set,
  re-derive thresholds via per-class F1-max on the ensemble probs, run the
  decision_tree to produce pred_labels, then call _bit_metrics.compute_bit_metrics.

This is post-hoc — no retraining, no model loading. Just parquet manipulation.

Usage:
    python -m chip_multilabel._logit_avg_ensemble \
        --parquet-a outputs/T5_T5_v19zpp_seed42_260506_233403/eval_I3/stage1_*/preds_chip.parquet \
        --parquet-b outputs/T9_T9_v19zpp_seed42_260506_235314/eval_I3/stage1_*/preds_chip.parquet \
        --weight-a 0.5 --weight-b 0.5 \
        --label "T5+T9_w0.5_0.5" \
        --out outputs/_iter12_v19zpp_logs/ensemble/T5_T9_w0.5.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from .constants import COMBO_KEYS, TRAIN_CLASSES
from ._bit_metrics import compute_bit_metrics, class_key_to_bits
from .decision_tree import decide
from .metrics import per_class_f1_max_threshold


def _build_multihot_gt(df: pd.DataFrame) -> np.ndarray:
    """Convert class_key column -> (N, 4) multi-hot ground truth."""
    n = len(df)
    gt = np.zeros((n, len(TRAIN_CLASSES)), dtype=np.int8)
    for i, ck in enumerate(df["class_key"].tolist()):
        gt[i] = class_key_to_bits(str(ck))
    return gt


def _decide_from_probs(probs: np.ndarray, thresholds: Dict[str, float],
                      df_in: pd.DataFrame) -> Tuple[List[List[str]], List[str], List[str]]:
    """Run decision_tree per row using ensemble probs + new thresholds.

    Re-uses original df's invalid_score / invalid heuristic decision (heuristic doesn't depend on probs).
    Returns (pred_labels_lists, pred_class_keys, decision_types).
    """
    pred_labels_list: List[List[str]] = []
    pred_class_keys: List[str] = []
    decision_types: List[str] = []
    invalid_scores = df_in["invalid_score"].tolist()
    orig_decision = df_in["decision_type"].tolist()
    for i, p in enumerate(probs):
        is_invalid = (orig_decision[i] == "invalid")
        res = decide(probs=p.tolist(),
                     thresholds=thresholds,
                     invalid_score=float(invalid_scores[i]),
                     is_invalid=is_invalid)
        pred_labels_list.append(list(res.active_labels))
        pred_class_keys.append(res.class_key)
        decision_types.append(res.decision_type)
    return pred_labels_list, pred_class_keys, decision_types


def ensemble_eval(parquet_a: Path, parquet_b: Path,
                  weight_a: float, weight_b: float,
                  label: str) -> Dict:
    df_a = pd.read_parquet(parquet_a).reset_index(drop=True)
    df_b = pd.read_parquet(parquet_b).reset_index(drop=True)
    if len(df_a) != len(df_b):
        raise ValueError(f"row count mismatch: A={len(df_a)} B={len(df_b)}")
    # ensure same chip ordering
    if not (df_a["chip_path"] == df_b["chip_path"]).all():
        # align
        df_b = df_b.set_index("chip_path").loc[df_a["chip_path"]].reset_index()
    prob_cols = [f"prob_{c}" for c in TRAIN_CLASSES]
    pa = df_a[prob_cols].to_numpy(dtype=np.float64)
    pb = df_b[prob_cols].to_numpy(dtype=np.float64)
    w_sum = weight_a + weight_b
    p_ens = (weight_a * pa + weight_b * pb) / w_sum

    # Re-derive per-class F1-max thresholds on ensemble probs vs eval-set GT
    gt = _build_multihot_gt(df_a)  # (N, 4)
    thresholds = per_class_f1_max_threshold(p_ens, gt, list(TRAIN_CLASSES))

    # Run decision_tree
    pred_labels_list, pred_class_keys, dec_types = _decide_from_probs(p_ens, thresholds, df_a)

    # Build new df mimicking parquet schema for compute_bit_metrics
    df_out = pd.DataFrame({
        "cell_id": [label] * len(df_a),
        "chip_path": df_a["chip_path"],
        "class_key": df_a["class_key"],
        "true_labels": df_a["true_labels"],
        "pred_labels": pred_labels_list,
        "pred_class_key": pred_class_keys,
        "decision_type": dec_types,
        **{f"prob_{c}": p_ens[:, ci] for ci, c in enumerate(TRAIN_CLASSES)},
    })

    metrics = compute_bit_metrics(df_out)
    metrics["thresholds"] = thresholds
    metrics["weight_a"] = weight_a
    metrics["weight_b"] = weight_b
    metrics["label"] = label
    return {label: metrics}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--parquet-a", required=True)
    ap.add_argument("--parquet-b", required=True)
    ap.add_argument("--weight-a", type=float, default=0.5)
    ap.add_argument("--weight-b", type=float, default=0.5)
    ap.add_argument("--label", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    out_dict = ensemble_eval(Path(args.parquet_a), Path(args.parquet_b),
                             args.weight_a, args.weight_b, args.label)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out_dict, f, indent=2, default=str)
    m = out_dict[args.label]
    print(f"\n=== {args.label} ===")
    print(f"  CF1                         = {m['macro_F1']:.4f}")
    print(f"  micro F1                    = {m['micro_F1']:.4f}")
    print(f"  ★ normal_invalid chip-FAR  = {m['normal_invalid_chip_FAR']:.4f}")
    print(f"    normal_only chip-FAR     = {m['normal_only_chip_FAR']:.4f}")
    print(f"    ood chip-FAR             = {m['ood_chip_FAR']:.4f}")
    print(f"  fork F1                     = {m['per_class_all']['fork']['f1']:.4f}")
    print(f"  3plus_active                = {m['n_3plus_active']} ({100*m['frac_3plus_active']:.2f}%)")
    print(f"\n[ensemble] saved -> {out_path}")


if __name__ == "__main__":
    main()
