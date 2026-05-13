#!/usr/bin/env python
"""iter123: val 2-combo bit-margin audit.

This script does not train. It reloads one or more checkpoints, reconstructs the
same runtime eval split used by run_stage1, and computes the distribution view
needed for S+2 selection:

- val 2-combo per-combo 4-bit mean/std
- val 2-combo bit-wise active/inactive quantiles and margins
- threshold candidates from those margins
- eval single+2-combo bit-F1 and negative FAR for each threshold set
- eval OOD class x 4-bit mean/std

Training policy is unchanged: all model checkpoints should come from 4 single
defect source data only. This script may read eval-only combo/OOD data for
selection diagnostics.
"""
from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import f1_score

from chip_multilabel.constants import (COMBO_KEYS, TRAIN_CLASSES,
                                       TRIPLE_COMBO_KEYS, WAFER_PATTERN_KEYS)
from chip_multilabel.eval_dataset import (ChipEvalDataset,
                                          discover_records_runtime,
                                          stratified_val_eval_split)
from chip_multilabel.inference_variants import forward_all_logits, sigmoid
from chip_multilabel.metrics import per_class_f1_max_threshold
from chip_multilabel.model_io import load_chip_backbone, select_train_logits


BITS = list(TRAIN_CLASSES)


@dataclass(frozen=True)
class ModelSpec:
    tag: str
    path: Path


def parse_model_spec(raw: str) -> ModelSpec:
    """Accept either 'tag=path' or a plain checkpoint path."""
    if "=" in raw:
        tag, path = raw.split("=", 1)
        tag = tag.strip()
        p = Path(path.strip())
    else:
        p = Path(raw.strip())
        tag = infer_tag(p)
    if not tag:
        tag = infer_tag(p)
    return ModelSpec(tag=tag, path=p)


def infer_tag(path: Path) -> str:
    parts = [x for x in path.parts if x]
    if len(parts) >= 2:
        parent = parts[-2]
        stem = path.stem
        parent = re.sub(r"[^A-Za-z0-9_.-]+", "_", parent)
        stem = re.sub(r"[^A-Za-z0-9_.-]+", "_", stem)
        return f"{parent}_{stem}"
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", path.stem)


def read_model_specs(args: argparse.Namespace) -> list[ModelSpec]:
    specs: list[ModelSpec] = []
    for raw in args.model or []:
        specs.append(parse_model_spec(raw))
    if args.models_file:
        p = Path(args.models_file)
        for line in p.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            specs.append(parse_model_spec(line))
    # preserve order while removing duplicate tags
    out: list[ModelSpec] = []
    seen: set[str] = set()
    for spec in specs:
        tag = spec.tag
        if tag in seen:
            tag = f"{tag}_{len(seen)}"
            spec = ModelSpec(tag=tag, path=spec.path)
        seen.add(tag)
        out.append(spec)
    return out


def group_for_class_key(class_key: str, n_true: int) -> str:
    if class_key == "Normal":
        return "Normal"
    if class_key == "Invalid":
        return "Invalid"
    if class_key in WAFER_PATTERN_KEYS or "+ood_" in class_key:
        return "OOD"
    if class_key in TRIPLE_COMBO_KEYS or n_true == 3:
        return "threeC"
    if class_key in COMBO_KEYS or n_true == 2:
        return "twoC"
    if n_true == 1:
        return "single"
    return "other"


def quantile(x: np.ndarray, q: float) -> float:
    if x.size == 0:
        return float("nan")
    return float(np.quantile(x, q))


def describe_prob(x: np.ndarray) -> dict[str, float | int]:
    x = np.asarray(x, dtype=np.float64)
    if x.size == 0:
        return {
            "n": 0,
            "mean": float("nan"),
            "std": float("nan"),
            "min": float("nan"),
            "p01": float("nan"),
            "p05": float("nan"),
            "p10": float("nan"),
            "p50": float("nan"),
            "p90": float("nan"),
            "p95": float("nan"),
            "p99": float("nan"),
            "max": float("nan"),
        }
    return {
        "n": int(x.size),
        "mean": float(x.mean()),
        "std": float(x.std(ddof=0)),
        "min": float(x.min()),
        "p01": quantile(x, 0.01),
        "p05": quantile(x, 0.05),
        "p10": quantile(x, 0.10),
        "p50": quantile(x, 0.50),
        "p90": quantile(x, 0.90),
        "p95": quantile(x, 0.95),
        "p99": quantile(x, 0.99),
        "max": float(x.max()),
    }


def records_frame(records, val_idx: np.ndarray, eval_idx: np.ndarray,
                  probs: np.ndarray) -> pd.DataFrame:
    split = np.full(len(records), "unused", dtype=object)
    split[val_idx] = "val"
    split[eval_idx] = "eval"
    rows = []
    for i, rec in enumerate(records):
        mh = np.asarray(rec.true_multihot, dtype=np.int64)
        n_true = int(mh.sum())
        row = {
            "idx": i,
            "split": split[i],
            "chip_path": rec.chip_path,
            "class_key": rec.class_key,
            "n_true": n_true,
            "group": group_for_class_key(rec.class_key, n_true),
        }
        for bi, bit in enumerate(BITS):
            row[f"true_{bit}"] = int(mh[bi])
            row[f"prob_{bit}"] = float(probs[i, bi])
        row["max_prob"] = float(probs[i].max())
        rows.append(row)
    return pd.DataFrame(rows)


def per_bit_val2_stats(df: pd.DataFrame, tag: str, q_low: float,
                       q_high: float) -> tuple[list[dict], dict[str, float]]:
    val2 = df[(df["split"] == "val") & (df["group"] == "twoC")].copy()
    rows: list[dict] = []
    summary: dict[str, float] = {}
    margins = []
    for bit in BITS:
        pos = val2[val2[f"true_{bit}"] == 1][f"prob_{bit}"].to_numpy()
        neg = val2[val2[f"true_{bit}"] == 0][f"prob_{bit}"].to_numpy()
        pos_stats = describe_prob(pos)
        neg_stats = describe_prob(neg)
        pos_low = quantile(pos, q_low)
        neg_high = quantile(neg, q_high)
        margin = pos_low - neg_high
        margins.append(margin)
        summary[f"{bit}_active_p05"] = quantile(pos, 0.05)
        summary[f"{bit}_active_p10"] = quantile(pos, 0.10)
        summary[f"{bit}_inactive_p95"] = quantile(neg, 0.95)
        summary[f"{bit}_inactive_p90"] = quantile(neg, 0.90)
        summary[f"{bit}_margin_q"] = margin
        rows.append({
            "tag": tag,
            "bit": bit,
            "side": "active",
            "q_low": q_low,
            "q_high": q_high,
            "margin_q": margin,
            **pos_stats,
        })
        rows.append({
            "tag": tag,
            "bit": bit,
            "side": "inactive",
            "q_low": q_low,
            "q_high": q_high,
            "margin_q": margin,
            **neg_stats,
        })

    if len(val2):
        y = val2[[f"true_{b}" for b in BITS]].to_numpy(dtype=bool)
        p = val2[[f"prob_{b}" for b in BITS]].to_numpy(dtype=np.float64)
        all_pos = p[y]
        all_neg = p[~y]
        all_pos_stats = describe_prob(all_pos)
        all_neg_stats = describe_prob(all_neg)
        all_margin = quantile(all_pos, q_low) - quantile(all_neg, q_high)
        summary["all_active_p05"] = quantile(all_pos, 0.05)
        summary["all_active_p10"] = quantile(all_pos, 0.10)
        summary["all_inactive_p95"] = quantile(all_neg, 0.95)
        summary["all_inactive_p90"] = quantile(all_neg, 0.90)
        summary["all_margin_q"] = all_margin
        rows.append({
            "tag": tag,
            "bit": "ALL",
            "side": "active",
            "q_low": q_low,
            "q_high": q_high,
            "margin_q": all_margin,
            **all_pos_stats,
        })
        rows.append({
            "tag": tag,
            "bit": "ALL",
            "side": "inactive",
            "q_low": q_low,
            "q_high": q_high,
            "margin_q": all_margin,
            **all_neg_stats,
        })

    valid_margins = [m for m in margins if m == m]
    summary["val2_min_margin"] = float(np.min(valid_margins)) if valid_margins else float("nan")
    summary["val2_mean_margin"] = float(np.mean(valid_margins)) if valid_margins else float("nan")
    return rows, summary


def combo_stats(df: pd.DataFrame, tag: str) -> list[dict]:
    rows: list[dict] = []
    val2 = df[(df["split"] == "val") & (df["group"] == "twoC")]
    for combo in COMBO_KEYS:
        sub = val2[val2["class_key"] == combo]
        labels = set(combo.split("+"))
        row: dict[str, float | int | str] = {
            "tag": tag,
            "combo": combo,
            "n": int(len(sub)),
        }
        for bit in BITS:
            x = sub[f"prob_{bit}"].to_numpy(dtype=np.float64)
            row[f"{bit}_active"] = int(bit in labels)
            row[f"{bit}_mean"] = float(x.mean()) if x.size else float("nan")
            row[f"{bit}_std"] = float(x.std(ddof=0)) if x.size else float("nan")
            row[f"{bit}_p05"] = quantile(x, 0.05)
            row[f"{bit}_p95"] = quantile(x, 0.95)
        rows.append(row)
    return rows


def ood_stats(df: pd.DataFrame, tag: str) -> list[dict]:
    rows: list[dict] = []
    ood = df[(df["split"] == "eval") & (df["group"] == "OOD")]
    for class_key in sorted(ood["class_key"].unique()):
        sub = ood[ood["class_key"] == class_key]
        row: dict[str, float | int | str] = {
            "tag": tag,
            "class_key": class_key,
            "n": int(len(sub)),
            "max_prob_mean": float(sub["max_prob"].mean()) if len(sub) else float("nan"),
            "max_prob_std": float(sub["max_prob"].std(ddof=0)) if len(sub) else float("nan"),
            "max_prob_p95": quantile(sub["max_prob"].to_numpy(dtype=np.float64), 0.95),
        }
        for bit in BITS:
            x = sub[f"prob_{bit}"].to_numpy(dtype=np.float64)
            row[f"{bit}_mean"] = float(x.mean()) if x.size else float("nan")
            row[f"{bit}_std"] = float(x.std(ddof=0)) if x.size else float("nan")
            row[f"{bit}_p95"] = quantile(x, 0.95)
        rows.append(row)
    return rows


def threshold_sets(df: pd.DataFrame, q_low: float, q_high: float,
                   alphas: Iterable[float]) -> dict[str, dict[str, float]]:
    val2 = df[(df["split"] == "val") & (df["group"] == "twoC")]
    if len(val2) == 0:
        return {}
    y_val2 = val2[[f"true_{b}" for b in BITS]].to_numpy(dtype=np.int64)
    p_val2 = val2[[f"prob_{b}" for b in BITS]].to_numpy(dtype=np.float64)
    out: dict[str, dict[str, float]] = {}
    out["val2_f1max"] = per_class_f1_max_threshold(p_val2, y_val2, BITS)
    for alpha in alphas:
        th: dict[str, float] = {}
        for bit in BITS:
            pos = val2[val2[f"true_{bit}"] == 1][f"prob_{bit}"].to_numpy(dtype=np.float64)
            neg = val2[val2[f"true_{bit}"] == 0][f"prob_{bit}"].to_numpy(dtype=np.float64)
            pos_low = quantile(pos, q_low)
            neg_high = quantile(neg, q_high)
            th[bit] = float(neg_high + alpha * (pos_low - neg_high))
        out[f"margin_a{alpha:.2f}"] = th
    return out


def eval_thresholds(df: pd.DataFrame, tag: str,
                    thresholds_by_name: dict[str, dict[str, float]]) -> list[dict]:
    rows: list[dict] = []
    eval_df = df[df["split"] == "eval"].copy()
    pos = eval_df[eval_df["group"].isin(["single", "twoC"])]
    neg = eval_df[eval_df["group"].isin(["Normal", "Invalid", "OOD"])]
    three = eval_df[eval_df["group"] == "threeC"]
    for name, th in thresholds_by_name.items():
        bit_f1s = []
        three_f1s = []
        per_bit = {}
        per_bit_three = {}
        for bit in BITS:
            y = pos[f"true_{bit}"].to_numpy(dtype=np.int64)
            p = (pos[f"prob_{bit}"].to_numpy(dtype=np.float64) >= th[bit]).astype(np.int64)
            f1 = float(f1_score(y, p, zero_division=0)) if len(pos) else float("nan")
            bit_f1s.append(f1)
            per_bit[f"F1_{bit}"] = f1
            if len(three):
                y3 = three[f"true_{bit}"].to_numpy(dtype=np.int64)
                p3 = (three[f"prob_{bit}"].to_numpy(dtype=np.float64) >= th[bit]).astype(np.int64)
                f13 = float(f1_score(y3, p3, zero_division=0))
            else:
                f13 = float("nan")
            three_f1s.append(f13)
            per_bit_three[f"three_F1_{bit}"] = f13

        if len(pos):
            y_pos = pos[[f"true_{b}" for b in BITS]].to_numpy(dtype=np.int64)
            p_pos = np.stack([
                (pos[f"prob_{b}"].to_numpy(dtype=np.float64) >= th[b]).astype(np.int64)
                for b in BITS
            ], axis=1)
            subset_acc = float((y_pos == p_pos).all(axis=1).mean())
        else:
            subset_acc = float("nan")

        if len(neg):
            fp_mask = np.zeros(len(neg), dtype=bool)
            for bit in BITS:
                fp_mask |= neg[f"prob_{bit}"].to_numpy(dtype=np.float64) >= th[bit]
            total_far = float(fp_mask.mean())
            ni_mask = neg["group"].isin(["Normal", "Invalid"]).to_numpy()
            ood_mask = (neg["group"] == "OOD").to_numpy()
            ni_far = float(fp_mask[ni_mask].mean()) if ni_mask.any() else float("nan")
            ood_far = float(fp_mask[ood_mask].mean()) if ood_mask.any() else float("nan")
        else:
            total_far = ni_far = ood_far = float("nan")

        row = {
            "tag": tag,
            "threshold_set": name,
            "bit_F1_S2": float(np.nanmean(bit_f1s)) if bit_f1s else float("nan"),
            "subset_acc_S2": subset_acc,
            "three_bit_F1_diag": float(np.nanmean(three_f1s)) if three_f1s else float("nan"),
            "total_far_bitonly": total_far,
            "ni_far_bitonly": ni_far,
            "ood_far_bitonly": ood_far,
            **{f"thr_{bit}": float(th[bit]) for bit in BITS},
            **per_bit,
            **per_bit_three,
        }
        rows.append(row)
    return rows


def run_one(spec: ModelSpec, records, val_idx: np.ndarray, eval_idx: np.ndarray,
            args: argparse.Namespace, device: torch.device) -> tuple[dict, list[dict], list[dict], list[dict], list[dict]]:
    if not spec.path.exists():
        raise FileNotFoundError(f"checkpoint not found: {spec.path}")
    model, meta, keep_indices = load_chip_backbone(spec.path, device)
    ds = ChipEvalDataset(records, img_size=int(meta["img_size"]))
    logits_full = forward_all_logits(model, ds, device, batch_size=args.batch_size,
                                     num_workers=args.num_workers, tta=False)
    logits_train = select_train_logits(logits_full, keep_indices)
    probs = sigmoid(logits_train)
    df = records_frame(records, val_idx, eval_idx, probs)

    bit_rows, bit_summary = per_bit_val2_stats(df, spec.tag, args.q_low, args.q_high)
    combo_rows = combo_stats(df, spec.tag)
    ood_rows = ood_stats(df, spec.tag)
    th_sets = threshold_sets(df, args.q_low, args.q_high, args.alphas)
    threshold_rows = eval_thresholds(df, spec.tag, th_sets)

    val2 = df[(df["split"] == "val") & (df["group"] == "twoC")]
    eval_s2 = df[(df["split"] == "eval") & (df["group"].isin(["single", "twoC"]))]
    eval_neg = df[(df["split"] == "eval") & (df["group"].isin(["Normal", "Invalid", "OOD"]))]
    summary = {
        "tag": spec.tag,
        "model": str(spec.path),
        "epoch": meta.get("epoch", -1),
        "backbone": meta.get("backbone", ""),
        "n_val2": int(len(val2)),
        "n_eval_S2": int(len(eval_s2)),
        "n_eval_neg": int(len(eval_neg)),
        **bit_summary,
    }
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return summary, bit_rows, combo_rows, ood_rows, threshold_rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", action="append", default=[],
                    help="Checkpoint spec. Use either PATH or TAG=PATH. Repeatable.")
    ap.add_argument("--models-file", default="",
                    help="Optional text file of PATH or TAG=PATH entries.")
    ap.add_argument("--eval-set", default="D:/project/data/wm-811k/chip_multilabel_v15direct")
    ap.add_argument("--n-per-class", type=int, default=200)
    ap.add_argument("--strength-min", type=float, default=0.0)
    ap.add_argument("--strength-max", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--val-ratio", type=float, default=0.2)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--num-workers", type=int, default=0)
    ap.add_argument("--device", default=None)
    ap.add_argument("--q-low", type=float, default=0.05)
    ap.add_argument("--q-high", type=float, default=0.95)
    ap.add_argument("--alphas", default="0.50,0.65,0.80")
    ap.add_argument("--out-prefix", default="outputs/_iter123_val2_margin")
    args = ap.parse_args()
    args.alphas = [float(x) for x in args.alphas.split(",") if x.strip()]

    specs = read_model_specs(args)
    if not specs:
        raise SystemExit("provide at least one --model or --models-file entry")

    device = torch.device(args.device) if args.device else torch.device(
        "cuda" if torch.cuda.is_available() else "cpu")
    records = discover_records_runtime(
        args.eval_set,
        n_per_class=args.n_per_class,
        strength_min=args.strength_min,
        strength_max=args.strength_max,
        include_classes=None,
        seed=args.seed,
    )
    val_idx, eval_idx = stratified_val_eval_split(records, val_ratio=args.val_ratio,
                                                  seed=args.seed)
    print(f"[iter123] records={len(records)} val={len(val_idx)} eval={len(eval_idx)} device={device}")

    summaries: list[dict] = []
    bit_rows_all: list[dict] = []
    combo_rows_all: list[dict] = []
    ood_rows_all: list[dict] = []
    threshold_rows_all: list[dict] = []
    for spec in specs:
        print(f"[iter123] audit {spec.tag}: {spec.path}")
        summary, bit_rows, combo_rows, ood_rows, threshold_rows = run_one(
            spec, records, val_idx, eval_idx, args, device)
        summaries.append(summary)
        bit_rows_all.extend(bit_rows)
        combo_rows_all.extend(combo_rows)
        ood_rows_all.extend(ood_rows)
        threshold_rows_all.extend(threshold_rows)

    out_prefix = Path(args.out_prefix)
    out_prefix.parent.mkdir(parents=True, exist_ok=True)
    summary_csv = out_prefix.with_name(out_prefix.name + "_summary.csv")
    bit_csv = out_prefix.with_name(out_prefix.name + "_bit_stats.csv")
    combo_csv = out_prefix.with_name(out_prefix.name + "_combo_stats.csv")
    ood_csv = out_prefix.with_name(out_prefix.name + "_ood_stats.csv")
    threshold_csv = out_prefix.with_name(out_prefix.name + "_threshold_eval.csv")
    meta_json = out_prefix.with_name(out_prefix.name + "_meta.json")

    pd.DataFrame(summaries).to_csv(summary_csv, index=False)
    pd.DataFrame(bit_rows_all).to_csv(bit_csv, index=False)
    pd.DataFrame(combo_rows_all).to_csv(combo_csv, index=False)
    pd.DataFrame(ood_rows_all).to_csv(ood_csv, index=False)
    pd.DataFrame(threshold_rows_all).to_csv(threshold_csv, index=False)
    meta_json.write_text(json.dumps({
        "eval_set": args.eval_set,
        "n_per_class": args.n_per_class,
        "seed": args.seed,
        "val_ratio": args.val_ratio,
        "q_low": args.q_low,
        "q_high": args.q_high,
        "alphas": args.alphas,
        "models": [{"tag": s.tag, "path": str(s.path)} for s in specs],
    }, indent=2), encoding="utf-8")

    view_cols = ["tag", "n_val2", "val2_min_margin", "val2_mean_margin",
                 "fork_margin_q", "scratch_margin_q",
                 "fork_active_p05", "scratch_active_p05",
                 "fork_inactive_p95", "scratch_inactive_p95"]
    print("\n=== val2 margin summary ===")
    print(pd.DataFrame(summaries)[view_cols].to_string(index=False))
    print("\n=== threshold eval top rows ===")
    th_df = pd.DataFrame(threshold_rows_all)
    print(th_df[["tag", "threshold_set", "bit_F1_S2", "subset_acc_S2",
                 "three_bit_F1_diag", "total_far_bitonly", "ood_far_bitonly",
                 "thr_fork", "thr_scratch"]].to_string(index=False))
    print(f"\n[iter123] wrote {summary_csv}")
    print(f"[iter123] wrote {bit_csv}")
    print(f"[iter123] wrote {combo_csv}")
    print(f"[iter123] wrote {ood_csv}")
    print(f"[iter123] wrote {threshold_csv}")


if __name__ == "__main__":
    main()
