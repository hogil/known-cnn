# -*- coding: utf-8 -*-
"""Print frozen-original recipe sweep status with mandatory prob reports."""
from __future__ import annotations

import argparse
import csv
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _f(row: dict[str, str], key: str, default: float = 0.0) -> float:
    try:
        return float(row.get(key, "") or default)
    except ValueError:
        return default


def _read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", errors="replace", newline="") as f:
        return list(csv.DictReader(f))


def _table(headers: list[str], rows: list[list[str]]) -> str:
    widths = [len(h) for h in headers]
    for row in rows:
        widths = [max(w, len(c)) for w, c in zip(widths, row)]
    header = "| " + " | ".join(h.ljust(w) for h, w in zip(headers, widths)) + " |"
    sep = "|-" + "-|-".join("-" * w for w in widths) + "-|"
    body = ["| " + " | ".join(c.ljust(w) for c, w in zip(row, widths)) + " |" for row in rows]
    return "\n".join([header, sep, *body])


def _extract_code_table(md_path: Path) -> str:
    if not md_path.exists():
        return ""
    text = md_path.read_text(encoding="utf-8", errors="replace")
    parts = text.split("```")
    if len(parts) >= 3:
        return parts[1].strip()
    return text.strip()


def _extract_gap(md_path: Path) -> str:
    if not md_path.exists():
        return ""
    text = md_path.read_text(encoding="utf-8", errors="replace")
    marker = "## POS min / NEG max gap analysis"
    if marker not in text:
        return ""
    return text[text.index(marker):].strip()


def _extract_gap_value(md_path: Path) -> float:
    if not md_path.exists():
        return 0.0
    text = md_path.read_text(encoding="utf-8", errors="replace")
    marker = "global_gap_pos_min_minus_neg_max="
    if marker not in text:
        return 0.0
    tail = text.split(marker, 1)[1].split(None, 1)[0]
    try:
        return float(tail)
    except ValueError:
        return 0.0


def _row_eval_gap(r: dict[str, str]) -> float:
    return _extract_gap_value(Path(r.get("eval_pcls_report", "")))


def _row_train_gap(r: dict[str, str]) -> float:
    return _extract_gap_value(Path(r.get("train_pcls_report", "")))


def _recipe_display(r: dict[str, str]) -> str:
    parts = []
    if r.get("variant"):
        parts.append(r["variant"])
    if r.get("LS"):
        parts.append(f"LS={r['LS']}")
    if r.get("n_groups"):
        parts.append(f"g={r['n_groups']}")
    if r.get("cmp_ls"):
        parts.append(f"cmp={r['cmp_ls']}")
    if r.get("seed"):
        parts.append(f"s={r['seed']}")
    params = " ".join(parts)
    return f"{r.get('tag', '')} ({params})" if params else r.get("tag", "")


def _goal_rank_key(r: dict[str, str]) -> tuple[float, float, float]:
    return (_f(r, "eval_Total_FAR", 100.0), -_f(r, "eval_bit_F1"), -_row_eval_gap(r))


def _print_metrics(rows: list[dict[str, str]], top: int) -> None:
    completed = [r for r in rows if r.get("status") == "done"]
    if not completed:
        print("No completed frozen_original leaderboard rows yet.")
        return
    ranked = sorted(completed, key=_goal_rank_key)
    body = []
    for i, r in enumerate(ranked[:top], 1):
        body.append([
            str(i),
            r.get("tag", ""),
            r.get("ckpt", ""),
            r.get("epochs", ""),
            r.get("LS", ""),
            r.get("n_groups", ""),
            r.get("cmp_ls", ""),
            r.get("cutmix_p", ""),
            r.get("seed", ""),
            r.get("eval_cell", ""),
            r.get("train_bit_F1", ""),
            r.get("train_NI_FAR", ""),
            r.get("train_OOD_FAR", ""),
            r.get("train_Total_FAR", ""),
            r.get("eval_bit_F1", ""),
            r.get("eval_NI_FAR", ""),
            r.get("eval_OOD_FAR", ""),
            r.get("eval_Total_FAR", ""),
            r.get("eval_bb_F1", ""),
            r.get("eval_fk_F1", ""),
            r.get("eval_sc_F1", ""),
            r.get("eval_sr_F1", ""),
            r.get("train_pos_prob", ""),
            r.get("train_neg_prob", ""),
            r.get("eval_pos_prob", ""),
            r.get("eval_neg_prob", ""),
            f"{_row_train_gap(r):+.3f}" if r.get("train_pcls_report") else "",
            f"{_row_eval_gap(r):+.3f}" if r.get("eval_pcls_report") else "",
        ])
    print("```")
    print(_table(
        [
            "rank", "Recipe", "ckpt", "ep", "LS", "g", "cmp", "p", "seed",
            "bestI", "train_bit_F1", "train_NI", "train_OOD", "train_Tot",
            "eval_bit_F1", "NI-FAR", "OOD-FAR", "Total FAR", "bb", "fk", "sc", "sr",
            "train_pos", "train_neg", "eval_pos", "eval_neg", "train_gap", "eval_gap",
        ],
        body,
    ))
    print("```")


def _print_reports(rows: list[dict[str, str]], count: int) -> None:
    completed = [r for r in rows if r.get("status") == "done"]
    ranked = sorted(completed, key=_goal_rank_key)
    for r in ranked[:count]:
        print("")
        recipe = _recipe_display(r)
        print(recipe)
        print(f"meta p={r.get('cutmix_p', '')} ep={r.get('epochs', '')} ckpt={r.get('ckpt', '')}")
        print(f"train_root={r.get('train_root', '')}")
        print(f"eval_root ={r.get('eval_root', '')}")
        print(
            f"TRAIN bit_F1={r.get('train_bit_F1', '')} "
            f"NI={r.get('train_NI_FAR', '')}% OOD={r.get('train_OOD_FAR', '')}% "
            f"Tot={r.get('train_Total_FAR', '')}% "
            f"pos_prob={r.get('train_pos_prob', '')} neg_prob={r.get('train_neg_prob', '')}"
        )
        print(f"{recipe} -- TRAIN (4 single class):")
        table = _extract_code_table(Path(r.get("train_pcls_report", "")))
        if table:
            print("```")
            print(table)
            print("```")
        gap = _extract_gap(Path(r.get("train_pcls_report", "")))
        if gap:
            print(gap)
        print("")
        print(
            f"EVAL bit_F1={r.get('eval_bit_F1', '')} "
            f"NI={r.get('eval_NI_FAR', '')}% OOD={r.get('eval_OOD_FAR', '')}% "
            f"Tot={r.get('eval_Total_FAR', '')}% "
            f"pos_prob={r.get('eval_pos_prob', '')} neg_prob={r.get('eval_neg_prob', '')}"
        )
        print(f"{recipe} -- EVAL per-class 4-bit prob (POS = single+combo, NEG = Normal/Invalid/OOD):")
        table = _extract_code_table(Path(r.get("eval_pcls_report", "")))
        if table:
            print("```")
            print(table)
            print("```")
        gap = _extract_gap(Path(r.get("eval_pcls_report", "")))
        if gap:
            print(gap)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="frozen_original")
    ap.add_argument("--top", type=int, default=10)
    ap.add_argument("--reports", type=int, default=1)
    args = ap.parse_args()

    lead = ROOT / "outputs" / args.dataset / "_leaderboard.csv"
    rows = _read_csv(lead)
    print(f"leaderboard={lead}")
    _print_metrics(rows, args.top)
    _print_reports(rows, args.reports)


if __name__ == "__main__":
    main()
