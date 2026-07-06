"""Post-hoc bit-vote ensemble report from existing stage1 parquet outputs.

This is intentionally inference-free: it combines stored per-chip predictions
from historical runs, then writes the same per-class probability diagnostics
used by the frozen-original sweep reports.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from ._bit_metrics import class_key_to_bits, compute_bit_metrics, labels_list_to_bits
from .constants import COMBO_KEYS, SINGLE_KEYS, TRAIN_CLASSES, WAFER_PATTERN_KEYS, labels_to_class_key
from .recipe_sweep import append_leaderboard, read_leaderboard, write_pcls_csv, write_pcls_report, write_performance_report


REPORT_CLASSES = list(SINGLE_KEYS) + list(COMBO_KEYS) + ["Normal", "Invalid"] + list(WAFER_PATTERN_KEYS)


def _load_cell(path: Path, cell: str) -> pd.DataFrame:
    df = pd.read_parquet(path)
    if "cell_id" in df.columns:
        df = df[df["cell_id"] == cell].reset_index(drop=True)
    if df.empty:
        raise ValueError(f"no rows for cell={cell} in {path}")
    return df


def _align_members(paths: list[Path], cell: str) -> list[pd.DataFrame]:
    members = [_load_cell(p, cell) for p in paths]
    base = members[0]
    chip_order = base["chip_path"].tolist()
    aligned = [base]
    for df in members[1:]:
        if len(df) != len(base) or not (df["chip_path"].reset_index(drop=True) == base["chip_path"]).all():
            df = df.set_index("chip_path").loc[chip_order].reset_index()
        if not (df["class_key"].reset_index(drop=True) == base["class_key"]).all():
            raise ValueError("class_key mismatch after chip_path alignment")
        aligned.append(df.reset_index(drop=True))
    return aligned


def _bits_to_labels(bits: Iterable[int]) -> list[str]:
    return [TRAIN_CLASSES[i] for i, b in enumerate(bits) if int(b) == 1]


def _pred_key(labels: list[str]) -> tuple[str, str]:
    active = frozenset(labels)
    if len(active) >= 4:
        return "4plus_active", "3plus_active"
    if len(active) == 0:
        return "Normal", "normal"
    if len(active) == 1:
        return labels[0], "single"
    return labels_to_class_key(active), "combo"


def build_vote_df(paths: list[Path], cell: str, k: int, tag: str) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    members = _align_members(paths, cell)
    n_models = len(members)
    if k < 1 or k > n_models:
        raise ValueError(f"k must be 1..{n_models}, got {k}")

    pred_stack = []
    for df in members:
        pred_stack.append(np.stack([labels_list_to_bits(x) for x in df["pred_labels"].tolist()]).astype(np.int8))
    votes = np.stack(pred_stack, axis=0).sum(axis=0)
    vote_prob = votes.astype(np.float64) / float(n_models)
    pred_bits = (votes >= k).astype(np.int8)

    base = members[0].reset_index(drop=True)
    pred_labels = [_bits_to_labels(row) for row in pred_bits]
    pred_keys = []
    decision_types = []
    for labels in pred_labels:
        pk, dt = _pred_key(labels)
        pred_keys.append(pk)
        decision_types.append(dt)

    out = pd.DataFrame(
        {
            "cell_id": [tag] * len(base),
            "chip_path": base["chip_path"],
            "class_key": base["class_key"],
            "true_labels": base.get("true_labels", pd.Series([None] * len(base))),
            "pred_labels": pred_labels,
            "pred_class_key": pred_keys,
            "decision_type": decision_types,
            "vote_count_bank_boundary": votes[:, 0],
            "vote_count_fork": votes[:, 1],
            "vote_count_scratch": votes[:, 2],
            "vote_count_scratch_rot": votes[:, 3],
            "prob_bank_boundary": vote_prob[:, 0],
            "prob_fork": vote_prob[:, 1],
            "prob_scratch": vote_prob[:, 2],
            "prob_scratch_rot": vote_prob[:, 3],
        }
    )
    return out, vote_prob, pred_bits


def pcls_rows(df: pd.DataFrame, vote_prob: np.ndarray, pred_bits: np.ndarray) -> list[dict[str, str]]:
    truth = np.stack([class_key_to_bits(str(x)) for x in df["class_key"].tolist()]).astype(bool)
    rows: list[dict[str, str]] = []
    for key in REPORT_CLASSES:
        idx = np.flatnonzero(df["class_key"].to_numpy() == key)
        if len(idx) == 0:
            continue
        gt = truth[idx]
        gt_bits = "".join(str(int(x)) for x in class_key_to_bits(key))
        mean_prob = vote_prob[idx].mean(axis=0)
        fired = pred_bits[idx].astype(bool)
        if "1" in gt_bits:
            recalls = []
            for j in range(len(TRAIN_CLASSES)):
                if gt[:, j].sum():
                    recalls.append(float(fired[gt[:, j], j].mean()))
            metric = "bit_F1"
            metric_value = float(np.mean(recalls)) if recalls else float("nan")
        else:
            metric = "FAR"
            metric_value = float(fired.any(axis=1).mean())
        rows.append(
            {
                "class": key,
                "GT": gt_bits,
                "n": str(len(idx)),
                "bank_boundary_prob": f"{mean_prob[0]:.3f}",
                "fork_prob": f"{mean_prob[1]:.3f}",
                "scratch_prob": f"{mean_prob[2]:.3f}",
                "scratch_rot_prob": f"{mean_prob[3]:.3f}",
                "metric": metric,
                "metric_value": f"{metric_value:.3f}",
            }
        )
    return rows


def diag_summary(df: pd.DataFrame, vote_prob: np.ndarray) -> dict[str, str]:
    truth = np.stack([class_key_to_bits(str(x)) for x in df["class_key"].tolist()]).astype(bool)
    pos = vote_prob[truth]
    neg = vote_prob[~truth]
    return {
        "pos_prob": f"{float(pos.mean()):.4f}" if pos.size else "nan",
        "neg_prob": f"{float(neg.mean()):.4f}" if neg.size else "nan",
    }


def metric_summary(metrics: dict) -> dict[str, str]:
    ni_n = int(metrics.get("normal_invalid_n_chips", 0))
    ni_fp = int(metrics.get("normal_invalid_FAR_chip_count", 0))
    ood_n = int(metrics.get("ood_n_chips", 0))
    ood_fp = int(metrics.get("ood_FAR_chip_count", 0))
    total_n = ni_n + ood_n
    total_fp = ni_fp + ood_fp
    per_bit = metrics.get("per_bit_F1_positive", {})
    return {
        "cell": "ensemble_vote",
        "bit_F1": f"{float(metrics.get('macro_F1_positive', 0.0)):.4f}",
        "NI_FAR": f"{100.0 * float(metrics.get('normal_invalid_chip_FAR', 0.0)):.2f}",
        "OOD_FAR": f"{100.0 * float(metrics.get('ood_chip_FAR', 0.0)):.2f}",
        "Total_FAR": f"{(100.0 * total_fp / total_n) if total_n else 0.0:.2f}",
        "bb_F1": f"{float(per_bit.get('bank_boundary', {}).get('f1', 0.0)):.4f}",
        "fk_F1": f"{float(per_bit.get('fork', {}).get('f1', 0.0)):.4f}",
        "sc_F1": f"{float(per_bit.get('scratch', {}).get('f1', 0.0)):.4f}",
        "sr_F1": f"{float(per_bit.get('scratch_rot', {}).get('f1', 0.0)):.4f}",
    }


def _read_split_json(out_dir: Path, split: str) -> dict | None:
    path = out_dir / f"{split}_metrics.json"
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _maybe_write_combined_report(args: argparse.Namespace, out_dir: Path, n_members: int) -> None:
    train_payload = _read_split_json(out_dir, "train")
    eval_payload = _read_split_json(out_dir, "eval")
    if not train_payload or not eval_payload:
        return

    train_report = out_dir / "train_pcls_report.md"
    eval_report = out_dir / "eval_pcls_report.md"
    if not train_report.exists() or not eval_report.exists():
        return

    train_summary = train_payload.get("summary", {})
    eval_summary = eval_payload.get("summary", {})
    train_diag = train_payload.get("diag", {})
    eval_diag = eval_payload.get("diag", {})

    row = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "dataset": args.dataset,
        "train_root": args.train_root,
        "eval_root": args.eval_root,
        "tag": args.tag,
        "variant": "vote_majority_bits",
        "extra": f"k={args.k}/{n_members} cell={args.cell}",
        "status": "done",
        "ckpt": "ensemble",
        "train_cell": train_summary.get("cell", args.cell),
        "train_bit_F1": train_summary.get("bit_F1", ""),
        "train_NI_FAR": train_summary.get("NI_FAR", ""),
        "train_OOD_FAR": train_summary.get("OOD_FAR", ""),
        "train_Total_FAR": train_summary.get("Total_FAR", ""),
        "train_pos_prob": train_diag.get("pos_prob", ""),
        "train_neg_prob": train_diag.get("neg_prob", ""),
        "eval_cell": eval_summary.get("cell", args.cell),
        "eval_bit_F1": eval_summary.get("bit_F1", ""),
        "eval_NI_FAR": eval_summary.get("NI_FAR", ""),
        "eval_OOD_FAR": eval_summary.get("OOD_FAR", ""),
        "eval_Total_FAR": eval_summary.get("Total_FAR", ""),
        "eval_bb_F1": eval_summary.get("bb_F1", ""),
        "eval_fk_F1": eval_summary.get("fk_F1", ""),
        "eval_sc_F1": eval_summary.get("sc_F1", ""),
        "eval_sr_F1": eval_summary.get("sr_F1", ""),
        "eval_pos_prob": eval_diag.get("pos_prob", ""),
        "eval_neg_prob": eval_diag.get("neg_prob", ""),
        "train_pcls_report": str(train_report),
        "eval_pcls_report": str(eval_report),
        "performance_report": str(out_dir / "performance_report.md"),
        "model": ";".join(str(p) for p in args.parquets),
        "out_dir": str(out_dir),
    }
    write_performance_report(out_dir / "performance_report.md", row, train_report, eval_report)

    lead = Path(args.leaderboard)
    existing_done = {r.get("tag", "") for r in read_leaderboard(lead) if r.get("status") == "done"}
    if args.tag not in existing_done:
        append_leaderboard(lead, row)
        print(f"[ensemble] leaderboard_append={lead}")
    print(f"[ensemble] performance_report={out_dir / 'performance_report.md'}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)
    ap.add_argument("--split", required=True, choices=["train", "eval"])
    ap.add_argument("--root", required=True)
    ap.add_argument("--cell", default="T0__I10")
    ap.add_argument("--k", type=int, required=True)
    ap.add_argument("--parquets", nargs="+", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--dataset", default="frozen_original")
    ap.add_argument("--train-root", default="E:/data/images/classification_chips")
    ap.add_argument("--eval-root", default="E:/data/images/chip_multilabel_v15direct_n2000")
    ap.add_argument("--leaderboard", default=str(Path("outputs/frozen_original/_leaderboard.csv")))
    args = ap.parse_args()

    paths = [Path(p) for p in args.parquets]
    missing = [str(p) for p in paths if not p.exists()]
    if missing:
        raise SystemExit(f"missing parquet(s): {missing}")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    df, vote_prob, pred_bits = build_vote_df(paths, args.cell, args.k, args.tag)
    metrics = compute_bit_metrics(df)
    msum = metric_summary(metrics)
    dsum = diag_summary(df, vote_prob)
    msum["cell"] = args.cell

    df.to_parquet(out_dir / f"{args.split}_ensemble_preds.parquet", index=False)
    (out_dir / f"{args.split}_metrics.json").write_text(
        json.dumps(
            {
                "tag": args.tag,
                "split": args.split,
                "root": args.root,
                "cell": args.cell,
                "k": args.k,
                "n_members": len(paths),
                "metrics": metrics,
                "summary": msum,
                "diag": dsum,
                "parquets": [str(p) for p in paths],
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    rows = pcls_rows(df, vote_prob, pred_bits)
    write_pcls_csv(rows, out_dir / f"{args.split}_pcls.csv")
    report_path = out_dir / f"{args.split}_pcls_report.md"
    title = (
        f"{args.tag} (vote_majority_bits k={args.k}/{len(paths)} cell={args.cell}) -- "
        + ("TRAIN (4 single class)" if args.split == "train" else "EVAL per-class 4-bit prob (POS = single+combo, NEG = Normal/Invalid/OOD)")
    )
    write_pcls_report(rows, report_path, title, dsum, args.split, args.root, msum)

    print(f"[ensemble] {args.tag} split={args.split} cell={args.cell} k={args.k}/{len(paths)}")
    print(f"[ensemble] root={args.root}")
    print(
        "[ensemble] "
        f"bit_F1={msum['bit_F1']} NI_FAR={msum['NI_FAR']}% "
        f"OOD_FAR={msum['OOD_FAR']}% Total_FAR={msum['Total_FAR']}% "
        f"pos_prob={dsum['pos_prob']} neg_prob={dsum['neg_prob']}"
    )
    print(f"[ensemble] report={report_path}")
    _maybe_write_combined_report(args, out_dir, len(paths))
    print(f"[ensemble] done_at={time.strftime('%Y-%m-%d %H:%M:%S')}")


if __name__ == "__main__":
    main()
