#!/usr/bin/env python3
"""Write a threshold sweep report for the GaussianNB reject sidecar."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from .nb_reject_report import (
    choose_tau,
    fit_defect_nb,
    loglik,
    read_preds,
    summarize,
)


DEFAULT_QUANTILES = "0.0001,0.0005,0.001,0.002,0.005,0.01,0.02"


def _fmt(v: object, nd: int = 4) -> str:
    return f"{float(v):.{nd}f}"


def _table(headers: list[str], rows: list[list[str]]) -> str:
    widths = [len(h) for h in headers]
    for row in rows:
        widths = [max(w, len(c)) for w, c in zip(widths, row)]
    out = [
        "| " + " | ".join(h.ljust(w) for h, w in zip(headers, widths)) + " |",
        "|-" + "-|-".join("-" * w for w in widths) + "-|",
    ]
    out.extend("| " + " | ".join(c.ljust(w) for c, w in zip(row, widths)) + " |" for row in rows)
    return "\n".join(out)


def _parse_quantiles(text: str) -> list[float]:
    vals = []
    for part in text.split(","):
        part = part.strip()
        if not part:
            continue
        vals.append(float(part))
    return vals


def build_report(args: argparse.Namespace) -> tuple[str, list[dict[str, object]]]:
    calib = read_preds(args.calib_preds, args.cell)
    eval_df = read_preds(args.eval_preds, args.cell)
    nb = fit_defect_nb(calib)
    calib_ll = loglik(nb, calib)
    eval_ll = loglik(nb, eval_df)

    rows: list[dict[str, object]] = []
    modes: list[tuple[str, float | None, float]] = []
    if args.include_neg_max:
        modes.append(("neg-max", None, choose_tau(calib, calib_ll, "neg-max", 0.0)))
    for q in _parse_quantiles(args.quantiles):
        modes.append((f"pos-q={q:g}", q, choose_tau(calib, calib_ll, "pos-quantile", q)))

    for mode, q, tau in modes:
        calib_summary = summarize(calib, calib_ll, tau)
        eval_summary = summarize(eval_df, eval_ll, tau)
        rows.append(
            {
                "mode": mode,
                "pos_quantile": q,
                "tau": tau,
                "calib_bit_F1_reject_empty": calib_summary["bit_F1_reject_empty"],
                "calib_post_reject_FAR": calib_summary["post_reject_Total_FAR"],
                "calib_false_reject_pos": calib_summary["false_reject_pos_count"],
                "calib_false_accept_neg": calib_summary["false_accept_neg_count"],
                "eval_bit_F1_reject_empty": eval_summary["bit_F1_reject_empty"],
                "eval_bit_F1_accepted_only": eval_summary["bit_F1_accepted_only"],
                "eval_post_reject_FAR": eval_summary["post_reject_Total_FAR"],
                "eval_accepted_neg_FAR": eval_summary["accepted_neg_FAR"],
                "eval_false_reject_pos": eval_summary["false_reject_pos_count"],
                "eval_false_accept_neg": eval_summary["false_accept_neg_count"],
                "eval_pos_cov": eval_summary["coverage_pos"],
                "eval_neg_cov": eval_summary["coverage_neg"],
            }
        )

    best = sorted(
        rows,
        key=lambda r: (
            float(r["eval_post_reject_FAR"]),
            int(r["eval_false_reject_pos"]),
            -float(r["eval_bit_F1_reject_empty"]),
        ),
    )[0]
    body = [
        [
            str(r["mode"]),
            _fmt(r["tau"], 3),
            _fmt(r["eval_bit_F1_reject_empty"]),
            _fmt(r["eval_bit_F1_accepted_only"]),
            f"{float(r['eval_post_reject_FAR']):.2f}%",
            str(r["eval_false_reject_pos"]),
            str(r["eval_false_accept_neg"]),
            _fmt(r["eval_pos_cov"]),
            _fmt(r["eval_neg_cov"]),
        ]
        for r in rows
    ]
    lines = [
        "# NB Reject Sweep Report",
        "",
        f"calib_preds={args.calib_preds}",
        f"eval_preds={args.eval_preds}",
        f"cell={args.cell or 'auto'}",
        "",
        "## Best Threshold",
        "",
        (
            f"mode={best['mode']} tau={_fmt(best['tau'], 6)} "
            f"eval_bit_F1={_fmt(best['eval_bit_F1_reject_empty'])} "
            f"post_reject_FAR={float(best['eval_post_reject_FAR']):.2f}% "
            f"false_reject_pos={best['eval_false_reject_pos']} "
            f"false_accept_neg={best['eval_false_accept_neg']}"
        ),
        "",
        "## Threshold Sweep",
        "",
        _table(
            [
                "mode",
                "tau",
                "bit_F1_reject",
                "bit_F1_accept",
                "post_FAR",
                "false_reject_pos",
                "false_accept_neg",
                "pos_cov",
                "neg_cov",
            ],
            body,
        ),
        "",
    ]
    return "\n".join(lines), rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--calib-preds", required=True, type=Path)
    ap.add_argument("--eval-preds", required=True, type=Path)
    ap.add_argument("--out-dir", required=True, type=Path)
    ap.add_argument("--cell", default="T0__I10")
    ap.add_argument("--quantiles", default=DEFAULT_QUANTILES)
    ap.add_argument("--include-neg-max", action="store_true")
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    report, rows = build_report(args)
    (args.out_dir / "nb_reject_sweep_report.md").write_text(report, encoding="utf-8")
    pd.DataFrame(rows).to_csv(args.out_dir / "nb_reject_sweep.csv", index=False)
    (args.out_dir / "nb_reject_sweep.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
    print(args.out_dir / "nb_reject_sweep_report.md")


if __name__ == "__main__":
    main()
