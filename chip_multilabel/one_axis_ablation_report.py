#!/usr/bin/env python3
"""Write a compact report for FCMPM one-axis ablation rows."""

from __future__ import annotations

import argparse
import csv
import re
from collections import defaultdict
from pathlib import Path


def _f(row: dict[str, str], key: str, default: float = 0.0) -> float:
    try:
        return float(row.get(key, "") or default)
    except ValueError:
        return default


def _read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def classify_axis(tag: str) -> tuple[str, str]:
    oneaxis_prefixes = (
        "abpos_Avar_B100",
        "neg_target",
        "cutmix_p",
        "loss_variant",
        "seed_repeat_baseline",
        "seed_repeat_neg",
        "grid_g3",
        "group_aligned_grid",
        "twofactor_abpos_neg",
        "twofactor_abpos_p",
        "twofactor_neg_p",
        "twofactor_grid_p",
        "threefactor_abpos_neg_p",
        "threefactor_abpos_neg_grid",
        "twofactor_loss_neg_p",
        "threefactor_loss_neg_p",
    )
    for axis in oneaxis_prefixes:
        prefix = f"oneaxis_{axis}_"
        if tag.startswith(prefix):
            value = tag[len(prefix):].split("_T7_", 1)[0]
            return axis, value
    if tag.startswith("targetlabel_weak100_strong100_neg000_"):
        return "baseline", "A100_B100_neg000_p050_grid9_g3_cmp100"
    return "other", tag


def score(row: dict[str, str]) -> tuple[float, float, float]:
    """Rank by strong bit_F1, low FAR, then larger POS-min/NEG-max gap."""
    return (
        _f(row, "eval_bit_F1"),
        -_f(row, "eval_Total_FAR", 100.0),
        _f(row, "eval_global_gap", -999.0),
    )


def row_line(row: dict[str, str], axis: str, value: str) -> str:
    return (
        f"| {axis} | {value} | {row.get('eval_bit_F1','')} | "
        f"{row.get('eval_pos_prob','')} | {row.get('eval_neg_prob','')} | "
        f"{row.get('eval_global_gap','')} | "
        f"{row.get('eval_worst_pos_class','')}/{row.get('eval_worst_pos_bit','')}="
        f"{row.get('eval_worst_pos_min_prob','')} | "
        f"{row.get('eval_worst_neg_class','')}/{row.get('eval_worst_neg_bit','')}="
        f"{row.get('eval_worst_neg_max_prob','')} |"
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--leaderboard", default="outputs/frozen_original/_leaderboard.csv")
    ap.add_argument("--out", default="docs/chip-multilabel/manager_report/ONE_AXIS_ABLATION_STATUS_260603.md")
    args = ap.parse_args()

    rows = _read_csv(Path(args.leaderboard))
    selected: list[tuple[str, str, dict[str, str]]] = []
    for row in rows:
        tag = row.get("tag", "")
        if row.get("status") != "done":
            continue
        if tag.startswith("oneaxis_") or tag.startswith("targetlabel_weak100_strong100_neg000_"):
            axis, value = classify_axis(tag)
            selected.append((axis, value, row))

    by_axis: dict[str, list[tuple[str, dict[str, str]]]] = defaultdict(list)
    for axis, value, row in selected:
        by_axis[axis].append((value, row))

    lines = [
        "# FCM-PM One-Axis Ablation Status",
        "",
        "이 문서는 FCM-PM 단일 변수 분리 평가 결과를 누적 정리한다.",
        "",
        "고정 baseline:",
        "",
        "```text",
        "dataset=frozen_original",
        "train=E:/data/images/classification_chips",
        "eval=E:/data/images/chip_multilabel_v15direct_n2000",
        "T7, LS=0.295, g=3, grid=9x9, cmp=1.0, cutmix_p=0.5",
        "A/B target=1.00/1.00, neg target=0.0, mpos=0.65, seed=7",
        "train=200/class, eval=2000/class",
        "```",
        "",
        "진행 원칙:",
        "",
        "1. 단일 변수 분리 평가로 영향 인자를 찾는다.",
        "2. 상위 성능 축 2개를 조합해 2-factor interaction을 본다.",
        "3. 2축 조합에서 안정적인 상위 조건이 나오면 3축 조합으로 확장한다.",
        "4. 매 row는 bit_F1뿐 아니라 POS min / NEG max gap을 같이 본다.",
        "5. 관리자용 표는 오탐률 컬럼을 빼고, probability separation 중심으로 표시한다.",
        "",
        "## Active / Planned Queue",
        "",
        "| phase | axis | values | status |",
        "|---|---|---|---|",
        "| 1-axis | A/B positive target | A=0.90/0.80/0.70, B=1.00 fixed | running / queued |",
        "| 1-axis | neg target | 0.02 / 0.05 / 0.10 | queued |",
        "| 1-axis | cutmix_p | 0.20 / 0.30 / 0.40 / 0.60 / 0.70 / 0.80 | running / queued |",
        "| 1-axis | loss variant | T10 ASL+LS / T4 ASL / T6 BCE->ASL, A/B=1.00/1.00 fixed | queued |",
        "| repeat | seed stability | baseline and neg=0.02/0.05 at seed 13/42/99 | queued |",
        "| 1-axis | grid, g=3 | 3x3 / 6x6 / 12x12, baseline 9x9 | queued |",
        "| 1-axis | group-grid alignment | g=2 grid6 / g=4 grid12, baseline g=3 grid9 | queued |",
        "| existing evidence | cmp | 0.5 / 0.7 / 0.8 / 1.0 | mined, not rerun |",
        "| 2-factor | top 1-axis pairs | neg/p/A-grid plus T10 loss interactions | pending |",
        "| 3-factor | top 2-factor neighborhood | compact T10/neg/p and A/neg/p candidates | pending |",
        "",
        "## Completed Rows",
        "",
        "| axis | value | bit_F1 | pos | neg | gap | worst POS min | worst NEG max |",
        "|---|---|---:|---:|---:|---:|---|---|",
    ]
    for axis, value, row in sorted(selected, key=lambda x: (x[0], x[1])):
        lines.append(row_line(row, axis, value))

    lines.extend(["", "## Axis Best So Far", ""])
    if not by_axis:
        lines.append("No completed one-axis rows yet.")
    else:
        lines.append("| axis | best value | bit_F1 | gap | reason |")
        lines.append("|---|---|---:|---:|---|")
        for axis, vals in sorted(by_axis.items()):
            best_value, best_row = max(vals, key=lambda vr: score(vr[1]))
            reason = (
                f"worst POS {best_row.get('eval_worst_pos_class','')}/"
                f"{best_row.get('eval_worst_pos_bit','')}="
                f"{best_row.get('eval_worst_pos_min_prob','')}; "
                f"worst NEG {best_row.get('eval_worst_neg_class','')}/"
                f"{best_row.get('eval_worst_neg_bit','')}="
                f"{best_row.get('eval_worst_neg_max_prob','')}"
            )
            lines.append(
                f"| {axis} | {best_value} | {best_row.get('eval_bit_F1','')} | "
                f"{best_row.get('eval_global_gap','')} | {reason} |"
            )

    lines.extend(
        [
            "",
            "## Next Stage Rule",
            "",
            "- 관리자 표에는 오탐률 컬럼을 넣지 않는다. 단, 내부 후보 gate에서는 `Total FAR <= 1%`를 같이 본다.",
            "- 1축에서 `bit_F1 >= 0.993`, `Total FAR <= 1%`, `gap`이 baseline보다 개선되는 값을 후보로 둔다.",
            "- 후보가 2개 이상이면 2축 조합을 만든다. 예: `A/B target best` x `neg target best`.",
            "- 2축 조합에서 다시 상위 조건이 안정되면 3축 조합으로 확장한다.",
            "- 이미 충분히 결과가 많은 `cmp` 축은 새로 반복하지 않고 기존 evidence를 사용한다.",
            "",
        ]
    )

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines), encoding="utf-8")
    print(out)


if __name__ == "__main__":
    main()
