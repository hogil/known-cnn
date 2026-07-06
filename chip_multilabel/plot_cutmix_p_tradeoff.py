# -*- coding: utf-8 -*-
"""Plot FCMPM cutmix_p trade-off from leaderboard rows."""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path
from statistics import mean, stdev


def _f(row: dict[str, str], key: str, default: float = float("nan")) -> float:
    try:
        v = row.get(key, "")
        if v == "":
            return default
        return float(v)
    except Exception:
        return default


def _is_cutmix_p_row(row: dict[str, str], mode: str) -> bool:
    tag = row.get("tag", "")
    if mode in {"base", "all"} and tag.startswith("oneaxis_cutmix_p_"):
        return True
    if mode in {"seed", "all"} and tag.startswith("oneaxis_seed_repeat_p_"):
        return True
    return False


def load_rows(paths: list[Path], mode: str) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for path in paths:
        with path.open(newline="", encoding="utf-8") as fp:
            for row in csv.DictReader(fp):
                if _is_cutmix_p_row(row, mode) and row.get("status") == "done":
                    rows.append(row)
    return rows


def summarize(rows: list[dict[str, str]], expected_dataset_n: int | None = None) -> list[dict[str, object]]:
    by_p: dict[float, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        p = _f(row, "cutmix_p")
        if p == p:
            by_p[p].append(row)

    out: list[dict[str, object]] = []
    for p, vals in sorted(by_p.items()):
        f1s = [_f(v, "eval_bit_F1") for v in vals]
        fars = [_f(v, "eval_Total_FAR") for v in vals]
        gaps = [_f(v, "eval_global_gap") for v in vals]
        posmins = [_f(v, "eval_worst_pos_min_prob") for v in vals]
        negmaxs = [_f(v, "eval_worst_neg_max_prob") for v in vals]
        datasets = sorted(set(v.get("dataset", "") for v in vals))
        seeds = sorted(set(v.get("seed", "") for v in vals))
        out.append(
            {
                "p": p,
                "n": len(vals),
                "dataset_n": len(datasets),
                "expected_dataset_n": expected_dataset_n if expected_dataset_n is not None else len(datasets),
                "complete": expected_dataset_n is None or len(datasets) == expected_dataset_n,
                "seed_n": len(seeds),
                "f1_mean": mean(f1s),
                "f1_std": stdev(f1s) if len(f1s) > 1 else 0.0,
                "far_mean": mean(fars),
                "far_max": max(fars),
                "gap_mean": mean(gaps),
                "gap_std": stdev(gaps) if len(gaps) > 1 else 0.0,
                "posmin_mean": mean(posmins),
                "posmin_std": stdev(posmins) if len(posmins) > 1 else 0.0,
                "negmax_mean": mean(negmaxs),
                "negmax_std": stdev(negmaxs) if len(negmaxs) > 1 else 0.0,
            }
        )
    return out


def write_csv(summary: list[dict[str, object]], out: Path) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "p",
        "n",
        "dataset_n",
        "expected_dataset_n",
        "complete",
        "seed_n",
        "f1_mean",
        "f1_std",
        "far_mean",
        "far_max",
        "gap_mean",
        "gap_std",
        "posmin_mean",
        "posmin_std",
        "negmax_mean",
        "negmax_std",
    ]
    with out.open("w", newline="", encoding="utf-8") as fp:
        writer = csv.DictWriter(fp, fieldnames=fields)
        writer.writeheader()
        for row in summary:
            writer.writerow(row)


def plot(summary: list[dict[str, object]], out: Path) -> None:
    import matplotlib.pyplot as plt

    if not summary:
        fig, ax = plt.subplots(figsize=(9.5, 3.0))
        ax.text(0.5, 0.5, "No complete cutmix_p rows yet", ha="center", va="center", fontsize=13)
        ax.axis("off")
        out.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out, dpi=180)
        return

    x = [float(r["p"]) for r in summary]
    pos = [float(r["posmin_mean"]) for r in summary]
    neg = [float(r["negmax_mean"]) for r in summary]
    gap = [float(r["gap_mean"]) for r in summary]
    f1 = [float(r["f1_mean"]) for r in summary]
    far = [float(r["far_mean"]) for r in summary]
    farmax = [float(r["far_max"]) for r in summary]
    n = [int(r["n"]) for r in summary]

    fig, axes = plt.subplots(2, 1, figsize=(9.5, 8.0), sharex=True)

    ax = axes[0]
    ax.plot(x, pos, marker="o", label="weak 2combo min_pos")
    ax.plot(x, neg, marker="o", label="worst NEG max_prob")
    ax.plot(x, gap, marker="o", label="gap = min_pos - neg_max")
    ax.axhline(0.0, color="#666666", linewidth=0.8)
    ax.set_ylabel("probability")
    ax.set_title("FCMPM cutmix_p trade-off: POS exposure vs NEG tail")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="best")
    for xi, yi, ni in zip(x, gap, n):
        ax.annotate(f"n={ni}", (xi, yi), textcoords="offset points", xytext=(0, 7), ha="center", fontsize=8)

    ax = axes[1]
    ax.plot(x, f1, marker="o", label="bit_F1 mean")
    ax2 = ax.twinx()
    ax2.plot(x, far, marker="s", color="#d62728", label="FAR mean")
    ax2.plot(x, farmax, marker="^", color="#8c1d13", linestyle="--", label="FAR max")
    ax.set_ylabel("bit_F1")
    ax2.set_ylabel("FAR (%)")
    ax.set_xlabel("cutmix_p")
    ax.grid(True, alpha=0.25)
    lines, labels = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines + lines2, labels + labels2, loc="best")

    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=180)


def _append_table(lines: list[str], rows: list[dict[str, object]]) -> None:
    lines += [
        "| p | n | dataset_n | seed_n | F1 mean | FAR mean | FAR max | weak 2combo min_pos | worst NEG max | gap mean |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for r in rows:
        lines.append(
            f"| {float(r['p']):.4f} | {int(r['n'])} | {int(r['dataset_n'])} | {int(r['seed_n'])} | "
            f"{float(r['f1_mean']):.4f} | {float(r['far_mean']):.2f} | {float(r['far_max']):.2f} | "
            f"{float(r['posmin_mean']):.3f} | {float(r['negmax_mean']):.3f} | {float(r['gap_mean']):.3f} |"
        )


def write_md(
    summary: list[dict[str, object]],
    plot_summary: list[dict[str, object]],
    plot_path: Path,
    csv_path: Path,
    out: Path,
    expected_dataset_n: int | None = None,
) -> None:
    complete = [r for r in summary if bool(r.get("complete"))]
    incomplete = [r for r in summary if not bool(r.get("complete"))]
    lines = [
        "# FCMPM cutmix_p Trade-off Trend",
        "",
        "This report separates complete multi-dataset rows from in-progress rows. Use the complete rows for conclusions.",
        "",
        f"![cutmix_p trade-off]({plot_path.name})",
        "",
        f"CSV: `{csv_path.name}`",
        "",
    ]
    if expected_dataset_n is not None:
        lines += [f"Expected dataset count: `{expected_dataset_n}`", ""]
    lines += ["## Complete Rows", ""]
    if complete:
        _append_table(lines, complete)
    else:
        lines.append("No complete rows yet. The sweep is still filling missing datasets.")
    lines += ["", "## In-Progress Rows", ""]
    if incomplete:
        _append_table(lines, incomplete)
    else:
        lines.append("None.")
    lines += ["", "## Plot Rows", ""]
    lines.append(
        "The plot uses complete rows only when `--complete-only` is set; otherwise it uses all rows for a live diagnostic."
        if plot_summary
        else "No complete rows were available for plotting yet."
    )
    lines += [
        "",
        "Interpretation:",
        "",
        "- Low `p` can under-expose FCMPM 2combo samples, keeping weak combo `min_pos` low.",
        "- Mid `p` is the target basin: weak combo `min_pos` rises while worst NEG tails remain controlled.",
        "- High `p` can keep F1 high but raise FAR max through OOD/Normal tail leakage.",
        "- Promote only conditions with high F1, low FAR max, positive stable gap, and low seed/dataset variance.",
        "",
    ]
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--leaderboards", nargs="+", required=True)
    ap.add_argument("--out-dir", default="docs/chip-multilabel/manager_report")
    ap.add_argument(
        "--mode",
        choices=("base", "seed", "all"),
        default="base",
        help="base=one row per dataset target, seed=seed-repeat rows, all=mixed diagnostic",
    )
    ap.add_argument(
        "--expected-datasets",
        nargs="*",
        default=None,
        help="Dataset names required for a complete row. Complete rows require dataset_n to match this count.",
    )
    ap.add_argument(
        "--complete-only",
        action="store_true",
        help="Plot only rows whose dataset_n matches --expected-datasets.",
    )
    args = ap.parse_args()

    paths: list[Path] = []
    for pattern in args.leaderboards:
        paths.extend(Path().glob(pattern))
    rows = load_rows(sorted(set(paths)), args.mode)
    expected_dataset_n = len(args.expected_datasets) if args.expected_datasets else None
    summary = summarize(rows, expected_dataset_n=expected_dataset_n)
    plot_summary = [r for r in summary if bool(r.get("complete"))] if args.complete_only else summary

    out_dir = Path(args.out_dir)
    suffix = "" if args.mode == "base" else f"_{args.mode}"
    csv_path = out_dir / f"FCMPM_CUTMIX_P_TRADEOFF_260606{suffix}.csv"
    png_path = out_dir / f"FCMPM_CUTMIX_P_TRADEOFF_260606{suffix}.png"
    md_path = out_dir / f"FCMPM_CUTMIX_P_TRADEOFF_260606{suffix}.md"
    write_csv(summary, csv_path)
    plot(plot_summary, png_path)
    write_md(summary, plot_summary, png_path, csv_path, md_path, expected_dataset_n=expected_dataset_n)
    print(md_path)
    print(png_path)
    print(csv_path)


if __name__ == "__main__":
    main()
