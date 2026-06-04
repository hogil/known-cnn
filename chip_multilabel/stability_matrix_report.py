#!/usr/bin/env python3
"""Write seed/dataset stability matrix from chip-multilabel leaderboards."""

from __future__ import annotations

import argparse
import csv
import re
from collections import defaultdict
from pathlib import Path
from statistics import mean, stdev


def _float(value: str | None, default: float = 0.0) -> float:
    try:
        return float(value or default)
    except ValueError:
        return default


def _std(values: list[float]) -> float:
    return stdev(values) if len(values) >= 2 else 0.0


def _read_rows(patterns: list[str]) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    paths: list[Path] = []
    for pattern in patterns:
        paths.extend(sorted(Path().glob(pattern)))
    for path in paths:
        dataset = path.parent.name
        with path.open("r", encoding="utf-8", newline="") as f:
            for row in csv.DictReader(f):
                if row.get("status") != "done":
                    continue
                row["_dataset"] = dataset
                rows.append(row)
    return rows


def _condition_key(tag: str) -> tuple[str, str]:
    if tag.startswith("oneaxis_seed_repeat_baseline_"):
        return "baseline", "A100_B100_neg000_p050"
    m = re.match(r"oneaxis_seed_repeat_neg_(neg\d+)_s\d+_", tag)
    if m:
        return "seed_repeat_neg", m.group(1)
    m = re.match(r"oneaxis_seed_repeat_p_(p\d+)_s\d+_", tag)
    if m:
        return "seed_repeat_p", m.group(1)
    m = re.match(r"oneaxis_cutmix_p_(p\d+)_", tag)
    if m:
        return "cutmix_p", m.group(1)
    m = re.match(r"oneaxis_neg_target_(neg\d+)_", tag)
    if m:
        return "neg_target", m.group(1)
    if tag.startswith("targetlabel_weak100_strong100_neg000_"):
        return "baseline", "A100_B100_neg000_p050"
    return "other", tag


def _seed(row: dict[str, str]) -> str:
    seed = row.get("seed") or ""
    if seed:
        return seed
    m = re.search(r"_s(\d+)_", row.get("tag", ""))
    return m.group(1) if m else ""


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--leaderboards", nargs="+", default=["outputs/*/_leaderboard.csv"])
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    rows = _read_rows(args.leaderboards)
    groups: dict[tuple[str, str], list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        axis, value = _condition_key(row.get("tag", ""))
        if axis == "other":
            continue
        groups[(axis, value)].append(row)

    summaries: list[dict[str, object]] = []
    for (axis, value), rs in sorted(groups.items()):
        f1s = [_float(r.get("eval_bit_F1")) for r in rs]
        fars = [_float(r.get("eval_Total_FAR"), 100.0) for r in rs]
        gaps = [_float(r.get("eval_global_gap")) for r in rs]
        datasets = sorted({r.get("_dataset", "") for r in rs})
        seeds = sorted({_seed(r) for r in rs if _seed(r)})
        stable = (
            len(rs) >= 2
            and min(f1s) >= 0.990
            and max(fars) <= 2.0
            and min(gaps) >= 0.10
        )
        summaries.append(
            {
                "axis": axis,
                "value": value,
                "n": len(rs),
                "dataset_n": len(datasets),
                "seed_n": len(seeds),
                "f1_mean": mean(f1s),
                "f1_min": min(f1s),
                "f1_std": _std(f1s),
                "far_mean": mean(fars),
                "far_max": max(fars),
                "gap_mean": mean(gaps),
                "gap_min": min(gaps),
                "gap_std": _std(gaps),
                "stable": stable,
                "datasets": ",".join(datasets),
                "seeds": ",".join(seeds),
            }
        )
    summaries.sort(key=lambda s: (bool(s["stable"]), float(s["f1_min"]), float(s["gap_min"]), -float(s["far_max"])), reverse=True)

    lines: list[str] = []
    lines.append("# FCMPM Seed/Dataset Stability Matrix")
    lines.append("")
    lines.append("이 문서는 단일 최고 row가 아니라 seed와 dataset 반복에서 살아남는 조건을 찾기 위한 stability matrix다.")
    lines.append("")
    lines.append("Stable 판정:")
    lines.append("")
    lines.append("```text")
    lines.append("n >= 2, min(bit_F1) >= 0.990, max(FAR) <= 2.0, min(gap) >= 0.10")
    lines.append("```")
    lines.append("")
    lines.append("## Summary")
    lines.append("")
    lines.append("| axis | value | n | datasets | seeds | F1 mean | F1 min | F1 std | FAR max | gap mean | gap min | gap std | decision |")
    lines.append("|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|")
    for s in summaries:
        decision = "stable-promote" if s["stable"] else "repeat/prune-check"
        lines.append(
            f"| {s['axis']} | {s['value']} | {s['n']} | {s['dataset_n']} | {s['seed_n']} | "
            f"{float(s['f1_mean']):.4f} | {float(s['f1_min']):.4f} | {float(s['f1_std']):.4f} | "
            f"{float(s['far_max']):.2f} | {float(s['gap_mean']):.3f} | {float(s['gap_min']):.3f} | "
            f"{float(s['gap_std']):.3f} | {decision} |"
        )
    lines.append("")
    lines.append("## Interpretation")
    lines.append("")
    lines.append("- High single-run F1 is not enough. Seed-specific OOD tails can destroy gap and FAR.")
    lines.append("- Promote only rows that keep both POS min and NEG max separated across seeds/datasets.")
    lines.append("- Rows with high F1 but high FAR are calibration/tail failures, not champions.")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(out)


if __name__ == "__main__":
    main()
