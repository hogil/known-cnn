#!/usr/bin/env python3
"""Write live experiment audit notes from chip-multilabel leaderboards."""

from __future__ import annotations

import argparse
import csv
import math
import subprocess
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from statistics import mean, stdev


def _float(value: str | None, default: float = 0.0) -> float:
    try:
        return float(value or default)
    except ValueError:
        return default


def _read_rows(patterns: list[str]) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    paths: list[Path] = []
    for pattern in patterns:
        paths.extend(sorted(Path().glob(pattern)))
    for path in paths:
        dataset = path.parent.name
        with path.open("r", encoding="utf-8", newline="") as f:
            for row in csv.DictReader(f):
                row["_dataset"] = dataset
                rows.append(row)
    return rows


def _split_key(tag: str) -> tuple[str, str]:
    if tag.startswith("targetlabel_weak100_strong100_neg000_"):
        return "baseline", "A100_B100_neg000_p050_grid9_g3_cmp100"
    if tag.startswith("oneaxis_"):
        rest = tag[len("oneaxis_") :]
        for axis in (
            "abpos_Avar_B100",
            "neg_target",
            "cutmix_p",
            "loss_variant",
            "seed_repeat_baseline",
            "seed_repeat_neg",
            "seed_repeat_p",
            "grid_g3",
            "group_aligned_grid",
            "twofactor_abpos_neg",
            "twofactor_abpos_p",
            "twofactor_neg_p",
            "twofactor_grid_p",
            "twofactor_loss_neg_p",
            "threefactor_abpos_neg_p",
            "threefactor_abpos_neg_grid",
            "threefactor_loss_neg_p",
        ):
            prefix = axis + "_"
            if rest.startswith(prefix):
                return axis, rest[len(prefix) :].split("_T", 1)[0]
    return "other", tag


def _std(values: list[float]) -> float:
    return stdev(values) if len(values) >= 2 else 0.0


def _current_process_lines() -> list[str]:
    try:
        out = subprocess.check_output(
            [
                "powershell",
                "-NoProfile",
                "-Command",
                "Get-CimInstance Win32_Process | "
                "Where-Object { $_.Name -eq 'python.exe' -and ($_.CommandLine -match 'recipe_sweep|_train_chip_variant|run_stage1|_posneg_prob_diag') } | "
                "Select-Object -ExpandProperty CommandLine",
            ],
            text=True,
            errors="ignore",
        )
    except Exception:
        return []
    return [line.strip() for line in out.splitlines() if line.strip()]


def _summaries(rows: list[dict[str, str]]) -> list[dict[str, object]]:
    grouped: dict[tuple[str, str], list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        if row.get("status") != "done":
            continue
        tag = row.get("tag", "")
        axis, value = _split_key(tag)
        if axis == "other":
            continue
        grouped[(axis, value)].append(row)

    summaries: list[dict[str, object]] = []
    for (axis, value), vals in grouped.items():
        f1s = [_float(v.get("eval_bit_F1")) for v in vals]
        fars = [_float(v.get("eval_Total_FAR"), 100.0) for v in vals]
        gaps = [_float(v.get("eval_global_gap")) for v in vals]
        poss = [_float(v.get("eval_pos_prob")) for v in vals]
        negs = [_float(v.get("eval_neg_prob")) for v in vals]
        datasets = {v.get("_dataset", "") for v in vals}
        summaries.append(
            {
                "axis": axis,
                "value": value,
                "n": len(vals),
                "dataset_n": len(datasets),
                "f1_mean": mean(f1s),
                "f1_std": _std(f1s),
                "far_mean": mean(fars),
                "far_std": _std(fars),
                "gap_mean": mean(gaps),
                "gap_std": _std(gaps),
                "pos_mean": mean(poss),
                "neg_mean": mean(negs),
            }
        )
    return sorted(
        summaries,
        key=lambda x: (
            -float(x["f1_mean"]),
            float(x["far_mean"]),
            -float(x["gap_mean"]),
        ),
    )


def _decision(s: dict[str, object]) -> str:
    f1 = float(s["f1_mean"])
    far = float(s["far_mean"])
    gap = float(s["gap_mean"])
    n = int(s["n"])
    if f1 >= 0.993 and far <= 1.0 and gap >= 0.15:
        return "promote: use for 2-factor/seed repeat"
    if f1 >= 0.993 and far <= 2.0 and n < 3:
        return "repeat: promising but dispersion unknown"
    if f1 < 0.990 or far > 5.0 or gap < 0.0:
        return "prune: delete pth and avoid expansion"
    return "observe: keep evidence, no expansion yet"


def _queue_suggestions(summaries: list[dict[str, object]]) -> list[str]:
    suggestions: list[str] = []
    by_axis = defaultdict(list)
    for s in summaries:
        by_axis[str(s["axis"])].append(s)

    for s in by_axis.get("cutmix_p", []):
        value = str(s["value"])
        if value == "p060" and float(s["f1_mean"]) >= 0.993 and float(s["far_mean"]) <= 1.0:
            suggestions.append("add/confirm: cutmix_p p055 and p065 around p060; seed-repeat p060")
    for s in by_axis.get("neg_target", []):
        value = str(s["value"])
        if value in {"neg002", "neg005"} and float(s["f1_mean"]) >= 0.993:
            suggestions.append(f"combine: {value} x best cutmix_p and {value} x T10")
    for s in by_axis.get("abpos_Avar_B100", []):
        if str(s["value"]) in {"A080_B100", "A070_B100"}:
            suggestions.append(f"prune: {s['value']} A target down-weight is weak; prefer ASL/T10")
    if not suggestions:
        suggestions.append("continue current queue; wait for more completed rows")
    return sorted(set(suggestions))


def _solve_linear_system(a: list[list[float]], b: list[float]) -> list[float] | None:
    n = len(b)
    aug = [row[:] + [rhs] for row, rhs in zip(a, b)]
    for col in range(n):
        pivot = max(range(col, n), key=lambda r: abs(aug[r][col]))
        if abs(aug[pivot][col]) < 1e-12:
            return None
        aug[col], aug[pivot] = aug[pivot], aug[col]
        scale = aug[col][col]
        aug[col] = [v / scale for v in aug[col]]
        for r in range(n):
            if r == col:
                continue
            factor = aug[r][col]
            aug[r] = [rv - factor * cv for rv, cv in zip(aug[r], aug[col])]
    return [aug[i][-1] for i in range(n)]


def _polyfit(xs: list[float], ys: list[float], degree: int) -> list[float] | None:
    if len(xs) < degree + 1:
        return None
    n = degree + 1
    mat: list[list[float]] = []
    rhs: list[float] = []
    for i in range(n):
        mat.append([sum(x ** (i + j) for x in xs) for j in range(n)])
        rhs.append(sum(y * (x ** i) for x, y in zip(xs, ys)))
    return _solve_linear_system(mat, rhs)


def _predict(coef: list[float], x: float) -> float:
    return sum(c * (x ** i) for i, c in enumerate(coef))


def _r2(xs: list[float], ys: list[float], coef: list[float]) -> float:
    ybar = mean(ys)
    ss_tot = sum((y - ybar) ** 2 for y in ys)
    if ss_tot <= 1e-12:
        return 0.0
    ss_res = sum((y - _predict(coef, x)) ** 2 for x, y in zip(xs, ys))
    return 1.0 - ss_res / ss_tot


def _numeric_split(axis: str, value: str) -> float | None:
    if axis == "cutmix_p" and value.startswith("p"):
        return int(value[1:]) / 100.0
    if axis == "neg_target" and value.startswith("neg"):
        return int(value[3:]) / 100.0
    if axis == "abpos_Avar_B100" and value.startswith("A"):
        return int(value[1:4]) / 100.0
    if axis == "baseline":
        # Baseline belongs to all three one-axis numeric curves.
        return None
    return None


def _response_curve_rows(summaries: list[dict[str, object]]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    by_axis: dict[str, list[tuple[float, dict[str, object]]]] = defaultdict(list)

    baseline = next((s for s in summaries if s["axis"] == "baseline"), None)
    for s in summaries:
        axis = str(s["axis"])
        value = str(s["value"])
        x = _numeric_split(axis, value)
        if x is not None:
            by_axis[axis].append((x, s))
    if baseline:
        by_axis["cutmix_p"].append((0.50, baseline))
        by_axis["neg_target"].append((0.00, baseline))
        by_axis["abpos_Avar_B100"].append((1.00, baseline))

    for axis, vals in sorted(by_axis.items()):
        # Average duplicate x values across datasets/seeds.
        by_x: dict[float, list[dict[str, object]]] = defaultdict(list)
        for x, s in vals:
            by_x[x].append(s)
        points: list[tuple[float, float, float, float, float]] = []
        for x, ss in sorted(by_x.items()):
            f1 = mean(float(s["f1_mean"]) for s in ss)
            far = mean(float(s["far_mean"]) for s in ss)
            gap = mean(float(s["gap_mean"]) for s in ss)
            score = f1 + 0.05 * gap if far <= 1.0 else f1 + 0.05 * gap - 0.02 * (far - 1.0)
            points.append((x, f1, far, gap, score))
        if len(points) < 2:
            continue
        xs = [p[0] for p in points]
        ys = [p[4] for p in points]
        lin = _polyfit(xs, ys, 1)
        quad = _polyfit(xs, ys, 2)
        model = "linear"
        r2 = _r2(xs, ys, lin) if lin else 0.0
        x_star = max(points, key=lambda p: p[4])[0]
        note = "best observed"
        if quad:
            qr2 = _r2(xs, ys, quad)
            if qr2 >= r2 + 0.05:
                model = "quadratic"
                r2 = qr2
                a = quad[2]
                b = quad[1]
                if a < 0:
                    vertex = -b / (2 * a)
                    lo, hi = min(xs), max(xs)
                    if lo <= vertex <= hi:
                        x_star = vertex
                        note = "quadratic vertex"
                    else:
                        note = "quadratic, vertex outside range"
        obs = max(points, key=lambda p: p[4])
        rows.append(
            {
                "axis": axis,
                "n": len(points),
                "model": model,
                "r2": r2,
                "observed_best_x": obs[0],
                "observed_score": obs[4],
                "suggested_x": x_star,
                "note": note,
            }
        )
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--leaderboards", nargs="*", default=["outputs/*/_leaderboard.csv"])
    ap.add_argument("--out", default="docs/chip-multilabel/manager_report/LIVE_EXPERIMENT_AUDIT_260603.md")
    args = ap.parse_args()

    rows = _read_rows(args.leaderboards)
    summaries = _summaries(rows)
    process_lines = _current_process_lines()

    lines = [
        "# Live FCMPM Experiment Audit",
        "",
        f"updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        "Purpose: keep training/eval running, record split effects, prune weak checkpoints, and identify next additions.",
        "",
        "## Active Processes",
        "",
    ]
    if process_lines:
        for p in process_lines:
            lines.append(f"- `{p[:260]}`")
    else:
        lines.append("- no active python experiment process found")

    lines.extend(
        [
            "",
            "## Split Summary",
            "",
            "| axis | value | n | dataset_n | F1 mean | F1 std | FAR mean | gap mean | gap std | decision |",
            "|---|---|---:|---:|---:|---:|---:|---:|---:|---|",
        ]
    )
    for s in summaries:
        lines.append(
            f"| {s['axis']} | {s['value']} | {s['n']} | {s['dataset_n']} | "
            f"{float(s['f1_mean']):.4f} | {float(s['f1_std']):.4f} | "
            f"{float(s['far_mean']):.2f} | {float(s['gap_mean']):.3f} | "
            f"{float(s['gap_std']):.3f} | {_decision(s)} |"
        )

    lines.extend(["", "## Queue / Prune Suggestions", ""])
    for suggestion in _queue_suggestions(summaries):
        lines.append(f"- {suggestion}")

    lines.extend(
        [
            "",
            "## Response Curve Fit",
            "",
            "Internal score: if `FAR<=1%`, `score = bit_F1 + 0.05*gap`; otherwise the score subtracts `0.02*(FAR-1)`.",
            "This keeps high-F1 rows from winning if the negative tail leaks.",
            "",
            "| axis | points | fit | R2 | observed best x | suggested x | note |",
            "|---|---:|---|---:|---:|---:|---|",
        ]
    )
    for row in _response_curve_rows(summaries):
        lines.append(
            f"| {row['axis']} | {row['n']} | {row['model']} | {float(row['r2']):.3f} | "
            f"{float(row['observed_best_x']):.4f} | {float(row['suggested_x']):.4f} | {row['note']} |"
        )

    lines.extend(
        [
            "",
            "## Operating Rule",
            "",
            "- Delete low-value `.pth` only; keep CSV/MD/log/probability evidence.",
            "- Expand only conditions with high mean F1, controlled FAR, and stable gap.",
            "- Treat one-off high rows as candidates, not conclusions.",
            "- Prefer ASL/T10 over lowering A target when A target down-weight reduces combo POS min.",
            "",
        ]
    )

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines), encoding="utf-8")
    print(out)


if __name__ == "__main__":
    main()
