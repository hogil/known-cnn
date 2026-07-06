#!/usr/bin/env python3
"""Sidecar GaussianNB reject report for chip multilabel predictions.

This script never tunes on the target eval labels. It fits a defect-likelihood
model and fixes the reject threshold from a separate calibration
``preds_chip.parquet`` file, then applies that frozen rule to another eval
``preds_chip.parquet`` file.
"""

from __future__ import annotations

import argparse
import ast
import json
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.naive_bayes import GaussianNB


BITS = ["bank_boundary", "fork", "scratch", "scratch_rot"]
PROB_COLS = [f"prob_{b}" for b in BITS]
DEFECT_CLASSES = set(
    BITS
    + [
        "bank_boundary+fork",
        "bank_boundary+scratch",
        "bank_boundary+scratch_rot",
        "fork+scratch",
        "fork+scratch_rot",
        "scratch+scratch_rot",
    ]
)


def parse_labels(value: object) -> list[str]:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return []
    if isinstance(value, list):
        return [str(x) for x in value]
    text = str(value).strip()
    if not text or text in {"[]", "None", "nan"}:
        return []
    try:
        parsed = ast.literal_eval(text)
    except Exception:
        parsed = text.split("+")
    if isinstance(parsed, str):
        return [parsed] if parsed else []
    return [str(x) for x in parsed]


def labels_from_class_key(class_key: object) -> list[str]:
    key = str(class_key)
    if key in DEFECT_CLASSES:
        return key.split("+")
    return []


def labels_to_vec(labels: Iterable[str]) -> np.ndarray:
    present = set(labels)
    return np.array([1 if b in present else 0 for b in BITS], dtype=np.int32)


def bit_f1(y_true: np.ndarray, y_pred: np.ndarray) -> tuple[float, dict[str, float]]:
    scores: dict[str, float] = {}
    for i, bit in enumerate(BITS):
        yt = y_true[:, i].astype(bool)
        yp = y_pred[:, i].astype(bool)
        tp = int(np.logical_and(yt, yp).sum())
        fp = int(np.logical_and(~yt, yp).sum())
        fn = int(np.logical_and(yt, ~yp).sum())
        denom = 2 * tp + fp + fn
        scores[bit] = 0.0 if denom == 0 else (2 * tp / denom)
    return float(np.mean(list(scores.values()))), scores


def read_preds(path: Path, cell: str | None) -> pd.DataFrame:
    df = pd.read_parquet(path)
    missing = [c for c in ["class_key", "pred_labels", *PROB_COLS] if c not in df.columns]
    if missing:
        raise SystemExit(f"{path} missing columns: {missing}")
    if cell and "cell_id" in df.columns:
        df = df[df["cell_id"] == cell].copy()
    elif "cell_id" in df.columns and df["cell_id"].nunique() > 1:
        first = str(df["cell_id"].iloc[0])
        df = df[df["cell_id"] == first].copy()
    if df.empty:
        raise SystemExit(f"{path} has no rows after cell filter")
    return df.reset_index(drop=True)


def fit_defect_nb(calib: pd.DataFrame) -> GaussianNB:
    m = calib["class_key"].astype(str).isin(DEFECT_CLASSES)
    if int(m.sum()) < 2:
        raise SystemExit("calibration preds need at least two defect rows")
    x = np.clip(calib.loc[m, PROB_COLS].to_numpy(dtype=np.float64), 1e-6, 1 - 1e-6)
    y = calib.loc[m, "class_key"].astype(str).to_numpy()
    return GaussianNB(var_smoothing=1e-6).fit(x, y)


def loglik(nb: GaussianNB, df: pd.DataFrame) -> np.ndarray:
    x = np.clip(df.loc[:, PROB_COLS].to_numpy(dtype=np.float64), 1e-6, 1 - 1e-6)
    return nb._joint_log_likelihood(x).max(axis=1)


def choose_tau(calib: pd.DataFrame, calib_ll: np.ndarray, mode: str, pos_quantile: float) -> float:
    is_pos = calib["class_key"].astype(str).isin(DEFECT_CLASSES).to_numpy()
    neg_ll = calib_ll[~is_pos]
    pos_ll = calib_ll[is_pos]
    if mode == "neg-max":
        if len(neg_ll) == 0:
            raise SystemExit("tau-mode neg-max needs negative calibration rows")
        return float(np.max(neg_ll) + 1e-6)
    if mode == "pos-quantile":
        if len(pos_ll) == 0:
            raise SystemExit("tau-mode pos-quantile needs positive calibration rows")
        return float(np.quantile(pos_ll, pos_quantile))
    raise SystemExit(f"unknown tau mode: {mode}")


def summarize(df: pd.DataFrame, ll: np.ndarray, tau: float) -> dict[str, object]:
    accepted = ll >= tau
    true = np.stack([labels_to_vec(labels_from_class_key(k)) for k in df["class_key"]])
    pred_orig = np.stack([labels_to_vec(parse_labels(v)) for v in df["pred_labels"]])
    pred_reject_empty = pred_orig.copy()
    pred_reject_empty[~accepted, :] = 0

    is_pos = true.sum(axis=1) > 0
    is_neg = ~is_pos
    pred_any = pred_orig.sum(axis=1) > 0
    pred_any_reject = pred_reject_empty.sum(axis=1) > 0

    f1_all, per_bit_all = bit_f1(true, pred_reject_empty)
    if int(accepted.sum()) > 0:
        f1_acc, per_bit_acc = bit_f1(true[accepted], pred_orig[accepted])
    else:
        f1_acc, per_bit_acc = 0.0, {b: 0.0 for b in BITS}

    neg_total = max(1, int(is_neg.sum()))
    neg_acc = np.logical_and(is_neg, accepted)
    neg_fp = np.logical_and(neg_acc, pred_any)

    rows = []
    for cls in sorted(df["class_key"].astype(str).unique()):
        m = df["class_key"].astype(str).to_numpy() == cls
        cls_pos = cls in DEFECT_CLASSES
        cls_rejected = int(np.logical_and(m, ~accepted).sum())
        cls_accepted = int(np.logical_and(m, accepted).sum())
        cls_n = int(m.sum())
        if cls_pos:
            false_reject = cls_rejected
            false_accept = 0
        else:
            false_reject = 0
            false_accept = int(np.logical_and(m, np.logical_and(accepted, pred_any)).sum())
        rows.append(
            {
                "class": cls,
                "n": cls_n,
                "accepted": cls_accepted,
                "rejected": cls_rejected,
                "coverage": cls_accepted / max(1, cls_n),
                "reject_rate": cls_rejected / max(1, cls_n),
                "false_reject_pos": false_reject,
                "false_accept_neg": false_accept,
                "ll_p05": float(np.quantile(ll[m], 0.05)),
                "ll_p50": float(np.quantile(ll[m], 0.50)),
                "ll_p95": float(np.quantile(ll[m], 0.95)),
                "ll_max": float(np.max(ll[m])),
            }
        )

    return {
        "n": int(len(df)),
        "tau": float(tau),
        "coverage_total": float(accepted.mean()),
        "coverage_pos": float(accepted[is_pos].mean()) if int(is_pos.sum()) else 0.0,
        "coverage_neg": float(accepted[is_neg].mean()) if int(is_neg.sum()) else 0.0,
        "reject_rate_total": float((~accepted).mean()),
        "reject_rate_pos": float((~accepted[is_pos]).mean()) if int(is_pos.sum()) else 0.0,
        "reject_rate_neg": float((~accepted[is_neg]).mean()) if int(is_neg.sum()) else 0.0,
        "bit_F1_reject_empty": f1_all,
        "bit_F1_accepted_only": f1_acc,
        "per_bit_F1_reject_empty": per_bit_all,
        "per_bit_F1_accepted_only": per_bit_acc,
        "post_reject_Total_FAR": float(100.0 * pred_any_reject[is_neg].mean()) if int(is_neg.sum()) else 0.0,
        "accepted_neg_FAR": float(100.0 * neg_fp.sum() / max(1, int(neg_acc.sum()))),
        "accepted_neg_count": int(neg_acc.sum()),
        "false_accept_neg_count": int(neg_fp.sum()),
        "false_reject_pos_count": int(np.logical_and(is_pos, ~accepted).sum()),
        "pos_count": int(is_pos.sum()),
        "neg_count": int(is_neg.sum()),
        "class_rows": rows,
    }


def format_float(v: object, nd: int = 4) -> str:
    return f"{float(v):.{nd}f}"


def write_report(
    out: Path,
    calib_path: Path,
    eval_path: Path,
    cell: str | None,
    tau_mode: str,
    calib_summary: dict[str, object],
    eval_summary: dict[str, object],
) -> None:
    lines = [
        "# NB Reject Report",
        "",
        f"calib_preds={calib_path}",
        f"eval_preds={eval_path}",
        f"cell={cell or 'auto'}",
        f"tau_mode={tau_mode}",
        f"nb_tau={format_float(eval_summary['tau'], 6)}",
        "",
        "## Summary",
        "",
        "```",
        "| split | n | coverage | pos_cov | neg_cov | bit_F1_reject_empty | bit_F1_accepted | post_reject_FAR | accepted_neg_FAR | false_reject_pos | false_accept_neg |",
        "|-------|---|----------|---------|---------|---------------------|-----------------|-----------------|------------------|------------------|------------------|",
    ]
    for name, summary in [("calib", calib_summary), ("eval", eval_summary)]:
        lines.append(
            "| "
            + " | ".join(
                [
                    name,
                    str(summary["n"]),
                    format_float(summary["coverage_total"]),
                    format_float(summary["coverage_pos"]),
                    format_float(summary["coverage_neg"]),
                    format_float(summary["bit_F1_reject_empty"]),
                    format_float(summary["bit_F1_accepted_only"]),
                    f"{float(summary['post_reject_Total_FAR']):.2f}%",
                    f"{float(summary['accepted_neg_FAR']):.2f}%",
                    str(summary["false_reject_pos_count"]),
                    str(summary["false_accept_neg_count"]),
                ]
            )
            + " |"
        )
    lines.extend(
        [
            "```",
            "",
            "## Eval Class Reject Rows",
            "",
            "```",
            "| class | n | accepted | rejected | coverage | reject_rate | false_reject_pos | false_accept_neg | ll_p05 | ll_p50 | ll_p95 | ll_max |",
            "|-------|---|----------|----------|----------|-------------|------------------|------------------|--------|--------|--------|--------|",
        ]
    )
    for row in eval_summary["class_rows"]:
        lines.append(
            "| "
            + " | ".join(
                [
                    str(row["class"]),
                    str(row["n"]),
                    str(row["accepted"]),
                    str(row["rejected"]),
                    format_float(row["coverage"]),
                    format_float(row["reject_rate"]),
                    str(row["false_reject_pos"]),
                    str(row["false_accept_neg"]),
                    format_float(row["ll_p05"], 2),
                    format_float(row["ll_p50"], 2),
                    format_float(row["ll_p95"], 2),
                    format_float(row["ll_max"], 2),
                ]
            )
            + " |"
        )
    lines.extend(["```", ""])
    out.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--calib-preds", required=True, type=Path)
    ap.add_argument("--eval-preds", required=True, type=Path)
    ap.add_argument("--out-dir", required=True, type=Path)
    ap.add_argument("--cell", default="T0__I10")
    ap.add_argument("--tau-mode", choices=["neg-max", "pos-quantile"], default="neg-max")
    ap.add_argument("--pos-quantile", type=float, default=0.05)
    args = ap.parse_args()

    calib = read_preds(args.calib_preds, args.cell)
    eval_df = read_preds(args.eval_preds, args.cell)
    nb = fit_defect_nb(calib)
    calib_ll = loglik(nb, calib)
    eval_ll = loglik(nb, eval_df)
    tau = choose_tau(calib, calib_ll, args.tau_mode, args.pos_quantile)

    calib_summary = summarize(calib, calib_ll, tau)
    eval_summary = summarize(eval_df, eval_ll, tau)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(calib_summary["class_rows"]).to_csv(args.out_dir / "nb_reject_calib_by_class.csv", index=False)
    pd.DataFrame(eval_summary["class_rows"]).to_csv(args.out_dir / "nb_reject_eval_by_class.csv", index=False)
    payload = {
        "calib_preds": str(args.calib_preds),
        "eval_preds": str(args.eval_preds),
        "cell": args.cell,
        "tau_mode": args.tau_mode,
        "pos_quantile": args.pos_quantile,
        "calib": {k: v for k, v in calib_summary.items() if k != "class_rows"},
        "eval": {k: v for k, v in eval_summary.items() if k != "class_rows"},
    }
    (args.out_dir / "nb_reject_report.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    write_report(
        args.out_dir / "nb_reject_report.md",
        args.calib_preds,
        args.eval_preds,
        args.cell,
        args.tau_mode,
        calib_summary,
        eval_summary,
    )
    print(args.out_dir / "nb_reject_report.md")


if __name__ == "__main__":
    main()
