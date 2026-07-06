# -*- coding: utf-8 -*-
"""Mine chip-multilabel historical SOTA records into one reproducible report."""
from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SOURCE_FILES = [
    ROOT / "docs/chip-multilabel/tables/paper_main_headline.csv",
    ROOT / "docs/chip-multilabel/tables/all_runs_n2000.csv",
    ROOT / "docs/chip-multilabel/tables/paper_main_ablation.csv",
    ROOT / "docs/chip-multilabel/tables/paper_section5_ablation.csv",
]


def _floatish(value: str | None) -> float | None:
    if value is None:
        return None
    s = str(value).strip().replace("%", "")
    if not s:
        return None
    try:
        return float(s)
    except ValueError:
        return None


def _pick(row: dict[str, str], keys: list[str]) -> float | None:
    for key in keys:
        if key in row:
            v = _floatish(row.get(key))
            if v is not None:
                return v
    return None


def _recipe_hint(text: str) -> str:
    patterns = [
        (r"iter116J", "iter116J T7 LS=0.30 g=3 cmp=0.5"),
        (r"iter26B", "iter26B T7 LS=0.50 g=3 cmp=0.5"),
        (r"iter26D", "iter26D T7 LS=0.40 g=4 cmp=0.5"),
        (r"iter26H", "iter26H T7 LS=0.67 g=3 cmp=0.5 white"),
        (r"iter25", "iter25 T7 LS=0.20/0.30 g=2 cmp=0.5"),
        (r"fcm_margin_g3_cls07_ls30_nopair", "fcm_margin T7 LS=0.30 g=3 cmp=0.7 no-pair"),
        (r"fcm_margin_g3_cls07_ls30_pair", "fcm_margin T7 LS=0.30 g=3 cmp=0.7 pair"),
        (r"fcm_margin_g4_cls05_ls30_nopair", "fcm_margin T7 LS=0.30 g=4 cmp=0.5 no-pair"),
        (r"fcm_margin_g3_cls05_ls40_nopair", "fcm_margin T7 LS=0.40 g=3 cmp=0.5 no-pair"),
        (r"KD", "KD student / ensemble component"),
        (r"ensemble|bag|vote|majority", "ensemble vote"),
    ]
    for pat, hint in patterns:
        if re.search(pat, text, re.IGNORECASE):
            return hint
    return ""


def collect() -> list[dict[str, str]]:
    out: list[dict[str, str]] = []
    for src in SOURCE_FILES:
        if not src.exists():
            continue
        with src.open("r", encoding="utf-8-sig", errors="replace", newline="") as f:
            for row in csv.DictReader(f):
                text = " ".join(str(v) for v in row.values() if v is not None)
                bit = _pick(row, ["bit_F1", "bit_f1", "eval_bit_F1", "bF1_4def"])
                if bit is None:
                    continue
                far = _pick(row, ["total_far_pct", "Tot_FAR", "ni_FAR", "ni_far_pct", "ni_far"])
                key = (
                    row.get("config")
                    or row.get("tag")
                    or row.get("run_tag")
                    or row.get("method")
                    or row.get("iter")
                    or ""
                )
                interesting = (
                    bit >= 0.95
                    or any(k in text for k in ["iter116J", "iter26B", "iter26D", "iter26H", "iter25", "fcm_margin", "iter39", "iter27"])
                )
                if not interesting:
                    continue
                out.append({
                    "source_file": src.name,
                    "record": key,
                    "bit_F1": f"{bit:.4f}",
                    "far_pct": "" if far is None else f"{far:.4f}",
                    "recipe_hint": _recipe_hint(text),
                    "note": text[:260].replace("\n", " "),
                })
    return sorted(out, key=lambda r: (-float(r["bit_F1"]), float(r["far_pct"] or "999")))


def _aligned_table(headers: list[str], rows: list[list[str]]) -> str:
    widths = [len(h) for h in headers]
    for row in rows:
        widths = [max(w, len(c)) for w, c in zip(widths, row)]
    header = "| " + " | ".join(h.ljust(w) for h, w in zip(headers, widths)) + " |"
    sep = "|-" + "-|-".join("-" * w for w in widths) + "-|"
    body = ["| " + " | ".join(c.ljust(w) for c, w in zip(row, widths)) + " |" for row in rows]
    return "\n".join([header, sep, *body])


def write_outputs(rows: list[dict[str, str]], out_dir: Path, top: int) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "_historical_sota_mining.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        fields = ["source_file", "record", "bit_F1", "far_pct", "recipe_hint", "note"]
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)

    headers = ["rank", "bit_F1", "FAR", "source", "record", "recipe_hint"]
    body = [
        [str(i + 1), r["bit_F1"], r["far_pct"], r["source_file"], r["record"], r["recipe_hint"]]
        for i, r in enumerate(rows[:top])
    ]
    md = (
        "# Historical chip-multilabel SOTA mining\n\n"
        "Priority: reproduce frozen-original train/eval first, then expand around single-model and ensemble-component winners.\n\n"
        "```\n"
        f"{_aligned_table(headers, body)}\n"
        "```\n"
    )
    (out_dir / "_historical_sota_mining.md").write_text(md, encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default="outputs/frozen_original")
    ap.add_argument("--top", type=int, default=40)
    args = ap.parse_args()
    rows = collect()
    write_outputs(rows, ROOT / args.out_dir, args.top)
    print(f"mined={len(rows)} out={ROOT / args.out_dir}")


if __name__ == "__main__":
    main()
