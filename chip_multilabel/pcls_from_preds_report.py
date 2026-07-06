# -*- coding: utf-8 -*-
"""Write per-class probability reports from an existing run_stage1 parquet.

This is CPU-only. It does not forward the model; it reads stored
``preds_chip.parquet`` probabilities and the stored ``pred_labels`` decisions,
then writes the same pcls CSV/Markdown format used by sweep reports.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from ._bit_metrics import class_key_to_bits, compute_bit_metrics, labels_list_to_bits
from .constants import COMBO_KEYS, SINGLE_KEYS, TRAIN_CLASSES, WAFER_PATTERN_KEYS
from .ensemble_vote_report import metric_summary
from .recipe_sweep import pcls_gap_fields, write_pcls_csv, write_pcls_report


PROB_COLS = [
    "prob_bank_boundary",
    "prob_fork",
    "prob_scratch",
    "prob_scratch_rot",
]


def _load_df(preds: Path, cell_id: str) -> pd.DataFrame:
    df = pd.read_parquet(preds)
    if "cell_id" in df.columns:
        df = df[df["cell_id"].astype(str) == cell_id].copy()
    if df.empty:
        raise SystemExit(f"no rows for cell_id={cell_id} in {preds}")
    missing = [c for c in ["class_key", "pred_labels", *PROB_COLS] if c not in df.columns]
    if missing:
        raise SystemExit(f"{preds} missing columns: {missing}")
    return df.reset_index(drop=True)


def _summary_from_stage1(run_dir: Path, cell_id: str) -> dict[str, str] | None:
    path = run_dir / "eval_summary.json"
    if not path.exists():
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    ev = payload.get("eval", {})
    cells = ev.get("all_cells", [])
    cell = next((r for r in cells if r.get("cell_id") == cell_id), None)
    if cell is None:
        cell = {
            "cell_id": ev.get("best_cell_id", cell_id),
            "eval_bit_F1": ev.get("best_eval_bit_F1", 0.0),
            "eval_NI_FAR": ev.get("best_eval_NI_FAR", 0.0),
            "eval_OOD_FAR": ev.get("best_eval_OOD_FAR", 0.0),
            "eval_Total_FAR": ev.get("best_eval_Total_FAR", 0.0),
            "eval_per_bit_F1": ev.get("best_eval_per_bit_F1", {}),
        }
    return {
        "cell": str(cell.get("cell_id", cell_id)),
        "bit_F1": f"{float(cell.get('eval_bit_F1', 0.0)):.4f}",
        "NI_FAR": f"{100.0 * float(cell.get('eval_NI_FAR', 0.0)):.2f}",
        "OOD_FAR": f"{100.0 * float(cell.get('eval_OOD_FAR', 0.0)):.2f}",
        "Total_FAR": f"{100.0 * float(cell.get('eval_Total_FAR', 0.0)):.2f}",
    }


def _rows(df: pd.DataFrame) -> tuple[list[dict[str, str]], dict[str, str], dict[str, str]]:
    probs = df[PROB_COLS].to_numpy(dtype=np.float64)
    keys = df["class_key"].astype(str).to_numpy()
    truth = np.stack([class_key_to_bits(k) for k in keys]).astype(bool)
    pred_bits = np.stack([labels_list_to_bits(x) for x in df["pred_labels"].tolist()]).astype(bool)

    pos = probs[truth]
    neg = probs[~truth]
    diag = {
        "pos_prob": f"{float(pos.mean()):.4f}" if pos.size else "nan",
        "neg_prob": f"{float(neg.mean()):.4f}" if neg.size else "nan",
    }

    preferred = list(SINGLE_KEYS) + list(COMBO_KEYS) + ["Normal", "Invalid"] + list(WAFER_PATTERN_KEYS)
    remaining = sorted(set(keys.tolist()) - set(preferred))
    report_classes = preferred + remaining
    rows: list[dict[str, str]] = []
    for key in report_classes:
        idx = np.flatnonzero(keys == key)
        if len(idx) == 0:
            continue
        p = probs[idx]
        gt_bits_arr = class_key_to_bits(key).astype(bool)
        gt = np.tile(gt_bits_arr, (len(idx), 1))
        gt_bits = "".join(str(int(x)) for x in gt_bits_arr)
        fired = pred_bits[idx]
        row = {
            "class": key,
            "GT": gt_bits,
            "n": str(len(idx)),
            "bank_boundary_prob": f"{float(p[:, 0].mean()):.3f}",
            "fork_prob": f"{float(p[:, 1].mean()):.3f}",
            "scratch_prob": f"{float(p[:, 2].mean()):.3f}",
            "scratch_rot_prob": f"{float(p[:, 3].mean()):.3f}",
        }
        if gt_bits_arr.any():
            recalls = []
            for j in range(len(TRAIN_CLASSES)):
                if gt[:, j].any():
                    recalls.append(float(fired[gt[:, j], j].mean()))
            pos_min = np.where(gt, p, np.inf).min(axis=1)
            neg_max = np.where(~gt, p, -np.inf).max(axis=1)
            row.update(
                {
                    "metric": "bit_F1",
                    "metric_value": f"{float(np.mean(recalls)) if recalls else 0.0:.3f}",
                    "pos_min_p10": f"{float(np.quantile(pos_min, 0.10)):.3f}",
                    "pos_min_p50": f"{float(np.quantile(pos_min, 0.50)):.3f}",
                }
            )
            if np.isfinite(neg_max).any():
                row["neg_max_p90"] = f"{float(np.quantile(neg_max, 0.90)):.3f}"
                row["neg_max_p95"] = f"{float(np.quantile(neg_max, 0.95)):.3f}"
        else:
            neg_max = p.max(axis=1)
            row.update(
                {
                    "metric": "FAR",
                    "metric_value": f"{float(fired.any(axis=1).mean()):.3f}",
                    "neg_max_p90": f"{float(np.quantile(neg_max, 0.90)):.3f}",
                    "neg_max_p95": f"{float(np.quantile(neg_max, 0.95)):.3f}",
                }
            )
        rows.append(row)

    diag.update(pcls_gap_fields(rows))
    metrics = metric_summary(compute_bit_metrics(df))
    return rows, diag, metrics


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--preds", required=True)
    ap.add_argument("--root", required=True)
    ap.add_argument("--mode", required=True, choices=["train", "eval"])
    ap.add_argument("--tag", required=True)
    ap.add_argument("--cell-id", default="T0__I10")
    ap.add_argument("--out-prefix", default="")
    args = ap.parse_args()

    preds = Path(args.preds)
    df = _load_df(preds, args.cell_id)
    rows, diag, metrics = _rows(df)
    stage1_summary = _summary_from_stage1(preds.parent, args.cell_id)
    if stage1_summary:
        metrics.update(stage1_summary)

    prefix = args.out_prefix or f"{args.mode}_pcls_from_preds"
    csv_path = preds.parent / f"{prefix}.csv"
    report_path = preds.parent / f"{prefix}_report.md"
    title = (
        f"{args.tag} -- "
        + ("TRAIN (4 single class)" if args.mode == "train" else "EVAL per-class 4-bit prob (POS = single+combo, NEG = Normal/Invalid/OOD)")
    )
    write_pcls_csv(rows, csv_path)
    write_pcls_report(rows, report_path, title, diag, args.mode, args.root, metrics)
    print(f"csv={csv_path}")
    print(f"report={report_path}")
    print(
        "summary "
        f"bit_F1={metrics.get('bit_F1', '')} Total_FAR={metrics.get('Total_FAR', '')}% "
        f"pos_prob={diag.get('pos_prob', '')} neg_prob={diag.get('neg_prob', '')} "
        f"gap={diag.get('eval_global_gap', '')}"
    )


if __name__ == "__main__":
    main()
