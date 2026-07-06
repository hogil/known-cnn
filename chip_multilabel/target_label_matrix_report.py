from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path


REPO = Path(__file__).resolve().parents[1]


def _float(row: dict[str, str], key: str, default: float = 0.0) -> float:
    try:
        return float(row.get(key, "") or default)
    except ValueError:
        return default


def _tag_params(tag: str) -> tuple[str, str, str]:
    strong = "?"
    neg = "?"
    p = "?"
    m = re.search(r"strong(\d{3,4})_neg(\d{3,4}).*?_p(\d{5})_", tag)
    if m:
        strong = _decode_pct_tag(m.group(1))
        neg = _decode_pct_tag(m.group(2))
        p = f"{int(m.group(3)) / 10000:.4f}"
    return p, strong, neg


def _decode_pct_tag(value: str) -> str:
    if len(value) == 4:
        return f"{int(value) / 1000:.3f}".rstrip("0").rstrip(".")
    denom = 100
    x = int(value) / denom
    return f"{x:.2f}"


def _read_rows(dataset: str) -> list[dict[str, str]]:
    lead = REPO / "outputs" / dataset / "_leaderboard.csv"
    if not lead.is_file():
        return []
    with lead.open("r", encoding="utf-8", errors="replace", newline="") as f:
        return [
            r for r in csv.DictReader(f)
            if r.get("tag", "").startswith("targetlabel_") and r.get("status") == "done"
        ]


def _table(headers: list[str], rows: list[list[str]]) -> str:
    widths = [
        max(len(str(x)) for x in [h] + [r[i] for r in rows])
        for i, h in enumerate(headers)
    ]
    out = ["| " + " | ".join(h.ljust(widths[i]) for i, h in enumerate(headers)) + " |"]
    out.append("|-" + "-|-".join("-" * w for w in widths) + "-|")
    for r in rows:
        out.append("| " + " | ".join(str(v).ljust(widths[i]) for i, v in enumerate(r)) + " |")
    return "\n".join(out)


def _report_excerpt(path: str, marker: str, limit: int = 120) -> str:
    p = Path(path)
    if not p.is_file():
        return ""
    text = p.read_text(encoding="utf-8", errors="replace")
    if marker in text:
        text = text[text.index(marker):]
    lines = text.splitlines()
    return "\n".join(lines[:limit]).strip()


def build_report(dataset: str) -> str:
    rows = _read_rows(dataset)
    title = f"# Target Label Matrix Report\n\ndataset={dataset}\n"
    if not rows:
        return title + "\nNo completed targetlabel rows yet.\n"

    rows_sorted = sorted(
        rows,
        key=lambda r: (
            _float(r, "eval_Total_FAR", 999.0),
            -_float(r, "eval_bit_F1"),
            -_float(r, "eval_global_gap"),
        ),
    )
    best_far = rows_sorted[0]
    best_gap = max(rows, key=lambda r: _float(r, "eval_global_gap", -999.0))
    best_f1 = max(rows, key=lambda r: _float(r, "eval_bit_F1", -999.0))

    def row_line(r: dict[str, str]) -> list[str]:
        p, strong, neg = _tag_params(r.get("tag", ""))
        return [
            p,
            strong,
            neg,
            r.get("eval_bit_F1", ""),
            r.get("eval_Total_FAR", ""),
            r.get("eval_pos_prob", ""),
            r.get("eval_neg_prob", ""),
            r.get("eval_global_gap", ""),
            f"{r.get('eval_worst_pos_class', '')}/{r.get('eval_worst_pos_bit', '')}={r.get('eval_worst_pos_min_prob', '')}",
            f"{r.get('eval_worst_neg_class', '')}/{r.get('eval_worst_neg_bit', '')}={r.get('eval_worst_neg_max_prob', '')}",
            r.get("tag", ""),
        ]

    headers = [
        "p",
        "strong",
        "neg",
        "bit_F1",
        "FAR",
        "pos",
        "neg_prob",
        "gap",
        "worst POS",
        "worst NEG",
        "tag",
    ]
    best_rows: list[dict[str, str]] = []
    seen_tags: set[str] = set()
    for candidate in (best_far, best_gap, best_f1):
        tag = candidate.get("tag", "")
        if tag in seen_tags:
            continue
        best_rows.append(candidate)
        seen_tags.add(tag)

    out = [
        title,
        f"completed={len(rows)}",
        "",
        "## Best So Far",
        "",
        _table(headers, [row_line(r) for r in best_rows]),
        "",
        "## Completed Rows",
        "",
        _table(headers, [row_line(r) for r in rows_sorted]),
        "",
        "## Best FAR Row TRAIN",
        "",
        _report_excerpt(best_far.get("train_pcls_report", ""), "-- TRAIN", 80),
        "",
        "## Best FAR Row EVAL",
        "",
        _report_excerpt(best_far.get("eval_pcls_report", ""), "-- EVAL", 140),
    ]
    return "\n".join(out).rstrip() + "\n"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="frozen_iter116J_orig814_eval_n20000")
    ap.add_argument("--out", default="")
    args = ap.parse_args()
    text = build_report(args.dataset)
    out = Path(args.out) if args.out else REPO / "outputs" / args.dataset / "_target_label_matrix_report.md"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(text, encoding="utf-8")
    print(out)


if __name__ == "__main__":
    main()
