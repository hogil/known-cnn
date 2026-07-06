# -*- coding: utf-8 -*-
"""Write a compact report for train/eval per-class sample-count sweeps."""
from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_FAMILY = "cmp10000_p05000_ab090_100_mpos065_s7_ep10"


def _read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", errors="replace", newline="") as f:
        return list(csv.DictReader(f))


def _f(row: dict[str, str], key: str, default: float = 0.0) -> float:
    try:
        return float(row.get(key, "") or default)
    except ValueError:
        return default


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


def _clean_metric(value: str) -> str:
    return value if value not in {"", None} else "-"


def _completed_replays(rows: list[dict[str, str]], family: str) -> list[dict[str, str]]:
    out = []
    for row in rows:
        tag = row.get("tag", "")
        if row.get("status") != "done":
            continue
        if family not in tag:
            continue
        if not tag.startswith("replay_samplecap_") and not tag.startswith("replay_sampletail_"):
            continue
        out.append(row)
    return sorted(
        out,
        key=lambda r: (
            int(r.get("train_eval_n_per_class", "0") or "0"),
            int(r.get("eval_n_per_class", "0") or "0"),
            r.get("tag", ""),
        ),
    )


def _source_rows(rows: list[dict[str, str]], family: str) -> list[dict[str, str]]:
    out = []
    for row in rows:
        tag = row.get("tag", "")
        if row.get("status") != "done":
            continue
        if family not in tag:
            continue
        if not tag.startswith("samplecap_") and not tag.startswith("sampletail_"):
            continue
        out.append(row)
    return sorted(out, key=lambda r: int(r.get("train_cap_per_class", "0") or "0"))


def _replay_table(rows: list[dict[str, str]]) -> str:
    body: list[list[str]] = []
    for row in rows:
        body.append(
            [
                row.get("train_eval_n_per_class", ""),
                row.get("eval_n_per_class", ""),
                _clean_metric(row.get("eval_bit_F1", "")),
                _clean_metric(row.get("eval_Total_FAR", "")),
                _clean_metric(row.get("eval_pos_prob", "")),
                _clean_metric(row.get("eval_neg_prob", "")),
                _clean_metric(row.get("eval_global_gap", "")),
                f"{row.get('eval_worst_pos_class', '')}/{row.get('eval_worst_pos_bit', '')}={row.get('eval_worst_pos_min_prob', '')}",
                f"{row.get('eval_worst_neg_class', '')}/{row.get('eval_worst_neg_bit', '')}={row.get('eval_worst_neg_max_prob', '')}",
                row.get("performance_report", ""),
            ]
        )
    return _table(
        [
            "train",
            "eval",
            "bit_F1",
            "FAR",
            "pos",
            "neg",
            "gap",
            "worst POS",
            "worst NEG",
            "performance_report",
        ],
        body,
    )


def _source_table(rows: list[dict[str, str]]) -> str:
    body: list[list[str]] = []
    for row in rows:
        body.append(
            [
                row.get("train_cap_per_class", ""),
                row.get("eval_n_per_class", ""),
                _clean_metric(row.get("eval_bit_F1", "")),
                _clean_metric(row.get("eval_Total_FAR", "")),
                _clean_metric(row.get("eval_global_gap", "")),
                f"{row.get('eval_worst_pos_class', '')}={row.get('eval_worst_pos_min_prob', '')}",
                f"{row.get('eval_worst_neg_class', '')}={row.get('eval_worst_neg_max_prob', '')}",
                row.get("performance_report", ""),
            ]
        )
    return _table(
        ["train", "eval", "bit_F1", "FAR", "gap", "worst POS", "worst NEG", "performance_report"],
        body,
    )


def _fmt_delta(value: float) -> str:
    sign = "+" if value >= 0 else ""
    return f"{sign}{value:.4f}"


def _trend_analysis(rows: list[dict[str, str]]) -> str:
    if not rows:
        return "No completed rows to analyze."

    by_cell: dict[tuple[int, int], dict[str, str]] = {}
    for row in rows:
        train_cap = int(row.get("train_eval_n_per_class", "0") or "0")
        eval_cap = int(row.get("eval_n_per_class", "0") or "0")
        by_cell[(train_cap, eval_cap)] = row

    train_rows: list[list[str]] = []
    eval_caps = sorted({eval_cap for _, eval_cap in by_cell})
    for eval_cap in eval_caps:
        train_caps = sorted(train_cap for train_cap, e in by_cell if e == eval_cap)
        for prev, cur in zip(train_caps, train_caps[1:]):
            a = by_cell[(prev, eval_cap)]
            b = by_cell[(cur, eval_cap)]
            train_rows.append(
                [
                    str(eval_cap),
                    f"{prev}->{cur}",
                    _fmt_delta(_f(b, "eval_bit_F1") - _f(a, "eval_bit_F1")),
                    _fmt_delta(_f(b, "eval_Total_FAR") - _f(a, "eval_Total_FAR")),
                    _fmt_delta(_f(b, "eval_global_gap") - _f(a, "eval_global_gap")),
                    f"{b.get('eval_worst_pos_class', '')}/{b.get('eval_worst_pos_bit', '')}={b.get('eval_worst_pos_min_prob', '')}",
                    f"{b.get('eval_worst_neg_class', '')}/{b.get('eval_worst_neg_bit', '')}={b.get('eval_worst_neg_max_prob', '')}",
                ]
            )

    eval_rows: list[list[str]] = []
    train_caps = sorted({train_cap for train_cap, _ in by_cell})
    for train_cap in train_caps:
        caps = sorted(eval_cap for t, eval_cap in by_cell if t == train_cap)
        for prev, cur in zip(caps, caps[1:]):
            a = by_cell[(train_cap, prev)]
            b = by_cell[(train_cap, cur)]
            eval_rows.append(
                [
                    str(train_cap),
                    f"{prev}->{cur}",
                    _fmt_delta(_f(b, "eval_bit_F1") - _f(a, "eval_bit_F1")),
                    _fmt_delta(_f(b, "eval_Total_FAR") - _f(a, "eval_Total_FAR")),
                    _fmt_delta(_f(b, "eval_global_gap") - _f(a, "eval_global_gap")),
                    f"{b.get('eval_worst_pos_class', '')}/{b.get('eval_worst_pos_bit', '')}={b.get('eval_worst_pos_min_prob', '')}",
                    f"{b.get('eval_worst_neg_class', '')}/{b.get('eval_worst_neg_bit', '')}={b.get('eval_worst_neg_max_prob', '')}",
                ]
            )

    best_gap = max(rows, key=lambda r: _f(r, "eval_global_gap", -999.0))
    best_far = _best_row(rows)
    lines = [
        "Best gap row: "
        f"train={best_gap.get('train_eval_n_per_class', '')} eval={best_gap.get('eval_n_per_class', '')} "
        f"bit_F1={best_gap.get('eval_bit_F1', '')} FAR={best_gap.get('eval_Total_FAR', '')} "
        f"gap={best_gap.get('eval_global_gap', '')} "
        f"worst_POS={best_gap.get('eval_worst_pos_class', '')}/{best_gap.get('eval_worst_pos_bit', '')}={best_gap.get('eval_worst_pos_min_prob', '')} "
        f"worst_NEG={best_gap.get('eval_worst_neg_class', '')}/{best_gap.get('eval_worst_neg_bit', '')}={best_gap.get('eval_worst_neg_max_prob', '')}",
    ]
    if best_far:
        lines.append(
            "Best FAR/F1 row: "
            f"train={best_far.get('train_eval_n_per_class', '')} eval={best_far.get('eval_n_per_class', '')} "
            f"bit_F1={best_far.get('eval_bit_F1', '')} FAR={best_far.get('eval_Total_FAR', '')} "
            f"gap={best_far.get('eval_global_gap', '')} "
            f"worst_POS={best_far.get('eval_worst_pos_class', '')}/{best_far.get('eval_worst_pos_bit', '')}={best_far.get('eval_worst_pos_min_prob', '')} "
            f"worst_NEG={best_far.get('eval_worst_neg_class', '')}/{best_far.get('eval_worst_neg_bit', '')}={best_far.get('eval_worst_neg_max_prob', '')}"
        )
    lines.extend(
        [
            "",
            "### Train cap deltas at fixed eval cap",
            "",
            _table(
                ["eval", "train step", "d_bit_F1", "d_FAR", "d_gap", "new worst POS", "new worst NEG"],
                train_rows,
            )
            if train_rows
            else "Need at least two train caps at the same eval cap.",
            "",
            "### Eval cap deltas at fixed train cap",
            "",
            _table(
                ["train", "eval step", "d_bit_F1", "d_FAR", "d_gap", "new worst POS", "new worst NEG"],
                eval_rows,
            )
            if eval_rows
            else "Need at least two eval caps at the same train cap.",
        ]
    )
    return "\n".join(lines)


def _nb_reject_summary_table(rows: list[dict[str, str]]) -> str:
    body: list[list[str]] = []
    for row in rows:
        perf = Path(row.get("performance_report", ""))
        if not perf.exists():
            continue
        sweep_json = perf.parent / "nb_reject_sweep_calib_v15_n2000" / "nb_reject_sweep.json"
        if not sweep_json.exists():
            continue
        try:
            payload = json.loads(sweep_json.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if not payload:
            continue
        best = sorted(
            payload,
            key=lambda r: (
                float(r.get("eval_post_reject_FAR", 999.0)),
                int(r.get("eval_false_reject_pos", 10**9)),
                -float(r.get("eval_bit_F1_reject_empty", 0.0)),
            ),
        )[0]
        body.append(
            [
                row.get("train_eval_n_per_class", ""),
                row.get("eval_n_per_class", ""),
                str(best.get("mode", "")),
                f"{float(best.get('eval_bit_F1_reject_empty', 0.0)):.4f}",
                f"{float(best.get('eval_post_reject_FAR', 0.0)):.2f}",
                str(best.get("eval_false_reject_pos", "")),
                str(best.get("eval_false_accept_neg", "")),
                f"{float(best.get('eval_pos_cov', 0.0)):.4f}",
                f"{float(best.get('eval_neg_cov', 0.0)):.4f}",
                str(sweep_json.with_name("nb_reject_sweep_report.md")),
            ]
        )
    if not body:
        return "No NB reject sweep sidecars found yet."
    return _table(
        [
            "train",
            "eval",
            "best reject",
            "post bit_F1",
            "post FAR",
            "false reject POS",
            "false accept NEG",
            "pos cov",
            "neg cov",
            "report",
        ],
        body,
    )


def _active_cells(active_rows: list[list[str]]) -> set[tuple[int, int]]:
    cells: set[tuple[int, int]] = set()
    for row in active_rows:
        if len(row) < 2:
            continue
        stage = row[1]
        if stage not in {"eval_forward", "train_eval", "eval_pcls"}:
            continue
        name = row[0]
        train_match = re.search(r"_tr([0-9]{3})_", name)
        eval_match = re.search(r"_evaln([0-9]+)_", name)
        if train_match and eval_match:
            cells.add((int(train_match.group(1)), int(eval_match.group(1))))
    return cells


def _pending(
    rows: list[dict[str, str]],
    active_rows: list[list[str]],
    train_caps: list[int],
    eval_caps: list[int],
) -> list[str]:
    done = {
        (int(r.get("train_eval_n_per_class", "0") or "0"), int(r.get("eval_n_per_class", "0") or "0"))
        for r in rows
    }
    active = _active_cells(active_rows)
    out = []
    for train_cap in train_caps:
        for eval_cap in eval_caps:
            if (train_cap, eval_cap) not in done and (train_cap, eval_cap) not in active:
                out.append(f"train={train_cap} eval={eval_cap}")
    return out


def _active_replays(replay_dataset: str, family: str, completed_rows: list[dict[str, str]]) -> list[list[str]]:
    root = ROOT / "outputs" / replay_dataset
    if not root.exists():
        return []
    completed_dirs = {Path(r.get("out_dir", "")).resolve() for r in completed_rows if r.get("out_dir")}
    out: list[list[str]] = []
    for d in sorted(root.glob(f"replay_*{family}*"), key=lambda p: p.stat().st_mtime, reverse=True):
        if not d.is_dir():
            continue
        try:
            if d.resolve() in completed_dirs:
                continue
        except OSError:
            pass
        eval_log = d / "eval_external.log"
        train_log = d / "train_eval_external.log"
        diag_log = d / "eval_posneg_pcls_external.log"
        stage = "created"
        progress = "-"
        updated = d.stat().st_mtime
        if eval_log.exists():
            text = eval_log.read_text(encoding="utf-8", errors="replace")
            updated = max(updated, eval_log.stat().st_mtime)
            matches = re.findall(r"\[eval\] forward ([0-9]+)/([0-9]+) chips .*? eta=([0-9]+)s", text)
            if matches:
                done, total, eta = matches[-1]
                pct = 100.0 * int(done) / max(1, int(total))
                progress = f"{done}/{total} ({pct:.1f}%, eta={eta}s)"
                stage = "eval_forward"
            elif "[eval] BEST cell" in text or "[eval] DONE" in text:
                stage = "eval_done"
            else:
                stage = "eval_log"
        if train_log.exists():
            updated = max(updated, train_log.stat().st_mtime)
            stage = "train_eval"
        if diag_log.exists():
            updated = max(updated, diag_log.stat().st_mtime)
            stage = "eval_pcls"
        out.append([d.name, stage, progress, _mtime(updated)])
    return out


def _mtime(ts: float) -> str:
    from datetime import datetime

    return datetime.fromtimestamp(ts).strftime("%Y-%m-%d %H:%M:%S")


def _best_row(rows: list[dict[str, str]]) -> dict[str, str] | None:
    if not rows:
        return None
    return sorted(
        rows,
        key=lambda r: (
            _f(r, "eval_Total_FAR", 999.0),
            -_f(r, "eval_bit_F1"),
            -_f(r, "eval_global_gap"),
        ),
    )[0]


def build_report(args: argparse.Namespace) -> str:
    replay_lead = ROOT / "outputs" / args.replay_dataset / "_leaderboard.csv"
    source_lead = ROOT / "outputs" / args.source_dataset / "_leaderboard.csv"
    replay_rows = _completed_replays(_read_csv(replay_lead), args.family)
    source_rows = _source_rows(_read_csv(source_lead), args.family)
    active_rows = _active_replays(args.replay_dataset, args.family, replay_rows)
    pending = _pending(replay_rows, active_rows, args.train_caps, args.eval_caps)

    lines = [
        f"# Sample Count Matrix Report",
        "",
        f"family={args.family}",
        f"replay_leaderboard={replay_lead}",
        f"source_leaderboard={source_lead}",
        "",
        "## Completed replay rows",
        "",
        _replay_table(replay_rows) if replay_rows else "No completed replay rows.",
        "",
        "## Trend analysis",
        "",
        _trend_analysis(replay_rows),
        "",
        "## NB reject sidecars",
        "",
        _nb_reject_summary_table(replay_rows),
        "",
        "## Active/partial replay rows",
        "",
        _table(["dir", "stage", "progress", "updated"], active_rows) if active_rows else "none",
        "",
        "## Pending replay cells",
        "",
        ", ".join(pending) if pending else "none",
        "",
        "## Source rows used to seed replay",
        "",
        _source_table(source_rows) if source_rows else "No completed source rows.",
    ]

    best = _best_row(replay_rows)
    if best and args.include_best:
        perf = Path(best.get("performance_report", ""))
        lines.extend(["", "## Best completed detailed report", ""])
        if perf.exists():
            lines.append(perf.read_text(encoding="utf-8", errors="replace").strip())
        else:
            lines.append(f"Missing performance_report={perf}")
    return "\n".join(lines).rstrip() + "\n"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source-dataset", default="frozen_iter116J_orig814_v15direct_n2000")
    ap.add_argument("--replay-dataset", default="frozen_iter116J_orig814_eval_n20000")
    ap.add_argument("--family", default=DEFAULT_FAMILY)
    ap.add_argument("--train-caps", type=int, nargs="+", default=[50, 100, 200])
    ap.add_argument("--eval-caps", type=int, nargs="+", default=[200, 2000, 20000])
    ap.add_argument("--out", default="")
    ap.add_argument("--no-include-best", dest="include_best", action="store_false")
    ap.set_defaults(include_best=True)
    args = ap.parse_args()

    out = Path(args.out) if args.out else ROOT / "outputs" / args.replay_dataset / "sample_count_matrix_report.md"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(build_report(args), encoding="utf-8")
    print(out)


if __name__ == "__main__":
    main()
