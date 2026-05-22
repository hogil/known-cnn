"""Single SOTA report: iter116J-style one-model, one-cell evaluation.

Reads a trained run directory produced by mega_matrix/run_single_sota.sh and
writes only the metrics the chip pipeline uses as primary signals:
eval bit_F1 and eval FAR split.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


BITS = ("bank_boundary", "fork", "scratch", "scratch_rot")
BIT_SHORT = {
    "bank_boundary": "bb",
    "fork": "fork",
    "scratch": "scratch",
    "scratch_rot": "scratch_rot",
}
NEG_ORDER = ("Normal", "Invalid", "CenterDonut", "CrossScratch", "DiagonalSmear", "Starburst")


def _load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _find_stage_dir(run_dir: Path, eval_name: str) -> Path:
    eval_root = run_dir / eval_name
    stages = sorted(eval_root.glob("stage1_*"), key=lambda p: p.name, reverse=True)
    if not stages:
        raise FileNotFoundError(f"no stage1_* directory under {eval_root}")
    return stages[0]


def _fmt_pct(v: float) -> str:
    return f"{v * 100.0:.2f}%"


def _safe_float(v: Any, default: float = 0.0) -> float:
    try:
        return float(v)
    except Exception:
        return default


def write_report(run_dir: Path, stage_dir: Path, out_md: Path, out_png: Path) -> None:
    summary = _load_json(stage_dir / "eval_summary.json")
    bit_far = _load_json(stage_dir / "bit_far_metrics.json")
    history_path = run_dir / "history.json"
    history = _load_json(history_path) if history_path.exists() else []

    cell = "T0__I10"
    if cell not in bit_far:
        available = ", ".join(sorted(bit_far.keys()))
        raise KeyError(f"{cell} missing from bit_far_metrics.json; available={available}")
    m = bit_far[cell]

    per_bit = {b: _safe_float(m.get("per_bit_F1", {}).get(b, 0.0)) for b in BITS}
    per_far_counts = m.get("per_class_FAR_counts", {})

    best_epoch = None
    best_metric = None
    if isinstance(history, list) and history:
        # run_single_sota uses val_margin selection.
        best = max(history, key=lambda r: _safe_float(r.get("val_margin"), float("-inf")))
        best_epoch = best.get("epoch")
        best_metric = _safe_float(best.get("val_margin"))

    lines: list[str] = []
    lines.append("# Single SOTA Eval - iter116J FCM-PM\n\n")
    lines.append(f"- run_dir: `{run_dir}`\n")
    lines.append(f"- eval_dir: `{stage_dir}`\n")
    lines.append("- recipe: ConvNeXtV2-Base FCMAE 384, T7 BCE+LS=0.30, classification_chips only, FCM-PM g=3, pair=masked, I10\n")
    lines.append("- selection: val_margin best_model\n")
    if best_epoch is not None:
        lines.append(f"- selected checkpoint signal: best val_margin epoch={best_epoch}, val_margin={best_metric:.4f}\n")
    lines.append("\n## Eval Performance\n\n")
    lines.append("| cell | eval bit_F1 | eval Total FAR | eval NI FAR | eval OOD FAR |\n")
    lines.append("|---|---:|---:|---:|---:|\n")
    lines.append(
        f"| {cell} | {_safe_float(m.get('bit_F1')):.4f} | "
        f"{_fmt_pct(_safe_float(m.get('Total_FAR')))} | "
        f"{_fmt_pct(_safe_float(m.get('NI_FAR')))} | "
        f"{_fmt_pct(_safe_float(m.get('OOD_FAR')))} |\n"
    )

    lines.append("\n## Eval Bit F1 By Class\n\n")
    lines.append("| class | bit_F1 |\n")
    lines.append("|---|---:|\n")
    for b in BITS:
        lines.append(f"| {b} | {per_bit[b]:.4f} |\n")

    lines.append("\n## Eval FAR By Negative Class\n\n")
    lines.append("| class | FAR | FP/N |\n")
    lines.append("|---|---:|---:|\n")
    for cls in NEG_ORDER:
        stat = per_far_counts.get(cls)
        if not stat:
            continue
        fp = int(stat.get("fp", 0))
        n = int(stat.get("n", 0))
        far = _safe_float(stat.get("chip_FAR"))
        lines.append(f"| {cls} | {_fmt_pct(far)} | {fp}/{n} |\n")

    lines.append("\n## Plot\n\n")
    lines.append(f"![single SOTA bit_F1 and FAR]({out_png.name})\n")

    out_md.write_text("".join(lines), encoding="utf-8")

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    axes[0].bar([BIT_SHORT[b] for b in BITS], [per_bit[b] for b in BITS], color="#2f6f9f")
    axes[0].set_ylim(0.0, 1.0)
    axes[0].set_ylabel("eval bit_F1")
    axes[0].set_title("I10 bit_F1 by defect bit")
    axes[0].grid(axis="y", alpha=0.25)
    for i, b in enumerate(BITS):
        axes[0].text(i, per_bit[b] + 0.015, f"{per_bit[b]:.3f}", ha="center", fontsize=9)

    far_labels = []
    far_vals = []
    for cls in NEG_ORDER:
        stat = per_far_counts.get(cls)
        if not stat:
            continue
        far_labels.append(cls)
        far_vals.append(_safe_float(stat.get("chip_FAR")) * 100.0)
    axes[1].bar(far_labels, far_vals, color="#b55248")
    axes[1].set_ylabel("eval FAR (%)")
    axes[1].set_title("I10 FAR by negative class")
    axes[1].tick_params(axis="x", rotation=35)
    axes[1].grid(axis="y", alpha=0.25)
    top = max(far_vals + [1.0])
    axes[1].set_ylim(0.0, max(1.0, top * 1.2))
    for i, v in enumerate(far_vals):
        axes[1].text(i, v + max(0.03, top * 0.02), f"{v:.2f}%", ha="center", fontsize=8)

    fig.suptitle("Single SOTA iter116J-style eval: I10 only")
    fig.tight_layout()
    fig.savefig(out_png, dpi=140)
    plt.close(fig)

    # Keep a machine-readable one-line summary for shell logs.
    compact = {
        "cell": cell,
        "eval_bit_F1": _safe_float(m.get("bit_F1")),
        "eval_Total_FAR": _safe_float(m.get("Total_FAR")),
        "eval_NI_FAR": _safe_float(m.get("NI_FAR")),
        "eval_OOD_FAR": _safe_float(m.get("OOD_FAR")),
        "stage_dir": str(stage_dir),
        "report": str(out_md),
        "plot": str(out_png),
        "best_epoch_by_val_margin": best_epoch,
    }
    (out_md.parent / "single_sota_summary.json").write_text(
        json.dumps(compact, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--eval-name", default="eval_sota_i10")
    args = ap.parse_args()

    run_dir = Path(args.run_dir).resolve()
    if not run_dir.exists():
        raise FileNotFoundError(run_dir)
    stage_dir = _find_stage_dir(run_dir, args.eval_name)
    out_md = run_dir / "single_sota_summary.md"
    out_png = run_dir / "single_sota_bit_far.png"
    write_report(run_dir, stage_dir, out_md, out_png)
    print(f"[single_sota_report] report={out_md}")
    print(f"[single_sota_report] plot={out_png}")


if __name__ == "__main__":
    main()
