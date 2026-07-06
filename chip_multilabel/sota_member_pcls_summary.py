# -*- coding: utf-8 -*-
"""Summarize generated per-member pcls reports for a SOTA replay directory."""
from __future__ import annotations

import argparse
import re
from pathlib import Path


def _read_report(path: Path) -> dict[str, str]:
    text = path.read_text(encoding="utf-8", errors="replace")
    out = {"report": str(path), "split": "train" if "train_pcls" in path.name else "eval"}
    if m := re.search(r"root=(.+)", text):
        out["root"] = m.group(1).strip()
    if m := re.search(
        r"bit_F1=([0-9.]+)\s+NI_FAR=([0-9.]+)%\s+OOD_FAR=([0-9.]+)%\s+Total_FAR=([0-9.]+)%\s+cell=([^\\n]+)",
        text,
    ):
        out.update(
            {
                "bit_F1": m.group(1),
                "NI_FAR": m.group(2),
                "OOD_FAR": m.group(3),
                "Total_FAR": m.group(4),
                "cell": m.group(5).strip(),
            }
        )
    if m := re.search(r"OVERALL pos_prob=([0-9.na-]+)\s+neg_prob=([0-9.na-]+)", text):
        out["pos_prob"] = m.group(1)
        out["neg_prob"] = m.group(2)
    if m := re.search(r"global_gap_pos_min_minus_neg_max=([+-]?[0-9.]+)", text):
        out["global_gap"] = m.group(1)
    if m := re.search(r"worst_pos_min=([0-9.]+)\s+([a-z]+)\s+@\s+(.+?)\s+", text):
        out["worst_pos_min"] = m.group(1)
        out["worst_pos_bit"] = m.group(2)
        out["worst_pos_class"] = m.group(3)
    if m := re.search(r"worst_neg_max=([0-9.]+)\s+([a-z]+)\s+@\s+(.+?)\s+", text):
        out["worst_neg_max"] = m.group(1)
        out["worst_neg_bit"] = m.group(2)
        out["worst_neg_class"] = m.group(3)
    return out


def _tag_from_report(root: Path, report: Path) -> str:
    rel = report.relative_to(root)
    parts = rel.parts
    try:
        i = parts.index("_ensemble_member_preds")
        return parts[i + 2]
    except Exception:
        return report.parent.name


def _num(row: dict[str, str], key: str, default: float = -999.0) -> float:
    try:
        return float(row.get(key, ""))
    except Exception:
        return default


def _table(headers: list[str], rows: list[list[str]]) -> str:
    widths = [len(h) for h in headers]
    for row in rows:
        widths = [max(w, len(c)) for w, c in zip(widths, row)]
    lines = [
        "| " + " | ".join(h.ljust(w) for h, w in zip(headers, widths)) + " |",
        "|-" + "-|-".join("-" * w for w in widths) + "-|",
    ]
    lines.extend("| " + " | ".join(c.ljust(w) for c, w in zip(row, widths)) + " |" for row in rows)
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    root = Path(args.root)
    reports = sorted(root.rglob("*pcls_from_preds_report.md"), key=lambda p: p.stat().st_mtime)
    rows = []
    for report in reports:
        row = _read_report(report)
        row["tag"] = _tag_from_report(root, report)
        rows.append(row)

    rows.sort(key=lambda r: (r.get("split", ""), -_num(r, "bit_F1"), _num(r, "Total_FAR", 999.0), -_num(r, "global_gap")))
    headers = [
        "split",
        "tag",
        "bit_F1",
        "Tot FAR",
        "pos",
        "neg",
        "gap",
        "worst_pos",
        "worst_neg",
        "report",
    ]
    table_rows = []
    for r in rows:
        table_rows.append(
            [
                r.get("split", ""),
                r.get("tag", ""),
                r.get("bit_F1", ""),
                r.get("Total_FAR", ""),
                r.get("pos_prob", ""),
                r.get("neg_prob", ""),
                r.get("global_gap", ""),
                (
                    f"{r.get('worst_pos_min', '')} {r.get('worst_pos_bit', '')} @ {r.get('worst_pos_class', '')}"
                    if r.get("worst_pos_min")
                    else ""
                ),
                (
                    f"{r.get('worst_neg_max', '')} {r.get('worst_neg_bit', '')} @ {r.get('worst_neg_class', '')}"
                    if r.get("worst_neg_max")
                    else ""
                ),
                r.get("report", ""),
            ]
        )
    text = "\n".join(
        [
            "# SOTA member pcls summary",
            "",
            f"root={root}",
            "",
            _table(headers, table_rows) if table_rows else "No pcls reports yet.",
            "",
        ]
    )
    out = Path(args.out) if args.out else root / "_sota_member_pcls_summary.md"
    out.write_text(text, encoding="utf-8")
    print(f"summary={out}")
    print(text)


if __name__ == "__main__":
    main()
