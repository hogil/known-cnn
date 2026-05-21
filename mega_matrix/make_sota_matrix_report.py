"""Build final SOTA-only 12-condition matrix report."""
from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


BITS = ("bank_boundary", "fork", "scratch", "scratch_rot")
NEG = ("Normal", "Invalid", "CenterDonut", "CrossScratch", "DiagonalSmear", "Starburst")
SELS = ("val_f1", "val_margin")


def _sizes(env: str, default: list[int]) -> list[int]:
    raw = os.environ.get(env)
    if not raw:
        return default
    return [int(x) for x in raw.replace(" ", ",").split(",") if x.strip()]


def _load_json(path: Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _fmt(v: float | None, nd: int = 4) -> str:
    if v is None or not math.isfinite(float(v)):
        return "NA"
    return f"{float(v):.{nd}f}"


def _pct(v: float | None) -> str:
    if v is None or not math.isfinite(float(v)):
        return "NA"
    return f"{float(v) * 100.0:.2f}%"


def _find_train_run(group_dir: Path, tn: int) -> Path | None:
    root = group_dir / f"train_n{tn}"
    runs = sorted([p for p in root.glob("*") if p.is_dir() and (p / "history.json").exists()])
    return runs[-1] if runs else None


def _find_eval_stage(eval_root: Path) -> Path | None:
    if not eval_root.exists():
        return None
    stages = sorted(
        [p for p in eval_root.glob("eval_*") if (p / "bit_far_metrics.json").exists()]
        + [p for p in eval_root.glob("stage1_*") if (p / "bit_far_metrics.json").exists()]
    )
    return stages[-1] if stages else None


def collect(group_dir: Path, train_sizes: list[int], eval_sizes: list[int]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    missing: list[str] = []
    for tn in train_sizes:
        run = _find_train_run(group_dir, tn)
        if run is None:
            missing.append(f"train_n{tn}: no training run")
            continue
        for en in eval_sizes:
            for sel in SELS:
                sel_dir = run / "selected" / sel
                selection_path = sel_dir / "selection.json"
                eval_stage = _find_eval_stage(sel_dir / f"eval_n{en}")
                if not selection_path.exists() or eval_stage is None:
                    missing.append(f"train_n{tn} eval_n{en} {sel}")
                    continue
                selection = _load_json(selection_path)
                metrics = _load_json(eval_stage / "bit_far_metrics.json")
                cell = metrics.get("T0__I10")
                if cell is None:
                    missing.append(f"train_n{tn} eval_n{en} {sel}: T0__I10 missing")
                    continue
                per_bit = cell.get("per_bit_F1", {})
                per_far = cell.get("per_class_FAR_counts", {})
                epoch_curve = run / "epoch_curves" / f"eval_n{en}" / "epoch_metrics.png"
                if not epoch_curve.exists() and os.environ.get("SOTA_ALLOW_MISSING_EPOCH_PLOTS", "0") != "1":
                    missing.append(f"train_n{tn} eval_n{en}: missing epoch plot")
                    continue
                row = {
                    "train_n": tn,
                    "eval_n": en,
                    "selection": sel,
                    "run_dir": str(run),
                    "eval_dir": str(eval_stage),
                    "best_epoch": int(selection.get("best_epoch", -1)),
                    "val_f1": float(selection.get("val_f1", float("nan"))),
                    "val_margin": float(selection.get("val_margin", float("nan"))),
                    "bit_F1": float(cell.get("bit_F1", float("nan"))),
                    "total_far": float(cell.get("Total_FAR", float("nan"))),
                    "ni_far": float(cell.get("NI_FAR", float("nan"))),
                    "ood_far": float(cell.get("OOD_FAR", float("nan"))),
                    "per_bit": {b: float(per_bit.get(b, float("nan"))) for b in BITS},
                    "per_far": {
                        c: {
                            "far": float(per_far.get(c, {}).get("chip_FAR", float("nan"))),
                            "fp": int(per_far.get(c, {}).get("fp", 0)),
                            "n": int(per_far.get(c, {}).get("n", 0)),
                        }
                        for c in NEG
                    },
                    "epoch_curve": str(epoch_curve),
                }
                rows.append(row)
    if missing:
        print("[sota-report] missing results:")
        for m in missing:
            print(f"  - {m}")
        raise SystemExit(1)
    return rows


def write_report(group_dir: Path, rows: list[dict[str, Any]]) -> None:
    fig_dir = group_dir / "figs_sota"
    fig_dir.mkdir(parents=True, exist_ok=True)
    out = group_dir / "summary_sota_matrix.md"

    lines: list[str] = []
    lines.append("# SOTA Single-Recipe Matrix\n\n")
    lines.append("## Recipe\n\n")
    lines.append("```text\n")
    lines.append("backbone: convnextv2_base.fcmae_ft_in22k_in1k_384\n")
    lines.append("loss: T7 BCE + label smoothing 0.30\n")
    lines.append("train: 4 single classes only, --no-normal\n")
    lines.append("cutmix: complement, p=0.25, pair=masked, pair-fill=corner, n-groups=3, complete-label-scale=0.5\n")
    lines.append("seed: 1, lr: 1e-4 cosine, batch: 2, accum: 8\n")
    lines.append("eval: I10 only, single4 + 2combo6 + Normal/Invalid/OOD4\n")
    lines.append("```\n\n")

    lines.append("## Final Summary\n\n")
    header = (
        "| train_n | eval_n | selection | best_epoch | val_f1 | val_margin | "
        "eval bit_F1 | Total FAR | NI FAR | OOD FAR | bb F1 | fork F1 | sc F1 | sr F1 | "
        "Normal FAR | Invalid FAR | CD FAR | CS FAR | DS FAR | ST FAR |\n"
    )
    lines.append(header)
    lines.append("|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|\n")
    for r in sorted(rows, key=lambda x: (x["train_n"], x["eval_n"], x["selection"])):
        pf = r["per_far"]
        pb = r["per_bit"]
        lines.append(
            f"| {r['train_n']} | {r['eval_n']} | {r['selection']} | {r['best_epoch']} | "
            f"{_fmt(r['val_f1'])} | {_fmt(r['val_margin'])} | {_fmt(r['bit_F1'])} | "
            f"{_pct(r['total_far'])} | {_pct(r['ni_far'])} | {_pct(r['ood_far'])} | "
            f"{_fmt(pb['bank_boundary'])} | {_fmt(pb['fork'])} | {_fmt(pb['scratch'])} | {_fmt(pb['scratch_rot'])} | "
            f"{_pct(pf['Normal']['far'])} | {_pct(pf['Invalid']['far'])} | "
            f"{_pct(pf['CenterDonut']['far'])} | {_pct(pf['CrossScratch']['far'])} | "
            f"{_pct(pf['DiagonalSmear']['far'])} | {_pct(pf['Starburst']['far'])} |\n"
        )

    lines.append("\n## val_f1 vs val_margin\n\n")
    lines.append("| train_n | eval_n | val_f1 bit_F1 | val_f1 FAR | val_margin bit_F1 | val_margin FAR | d bit_F1 | d FAR |\n")
    lines.append("|---:|---:|---:|---:|---:|---:|---:|---:|\n")
    by_key = {(r["train_n"], r["eval_n"], r["selection"]): r for r in rows}
    for tn in sorted({r["train_n"] for r in rows}):
        for en in sorted({r["eval_n"] for r in rows}):
            a = by_key[(tn, en, "val_f1")]
            b = by_key[(tn, en, "val_margin")]
            lines.append(
                f"| {tn} | {en} | {_fmt(a['bit_F1'])} | {_pct(a['total_far'])} | "
                f"{_fmt(b['bit_F1'])} | {_pct(b['total_far'])} | "
                f"{_fmt(b['bit_F1'] - a['bit_F1'])} | {_pct(b['total_far'] - a['total_far'])} |\n"
            )

    lines.append("\n## Epoch Plots\n\n")
    for tn in sorted({r["train_n"] for r in rows}):
        for en in sorted({r["eval_n"] for r in rows}):
            r = by_key[(tn, en, "val_f1")]
            p = Path(r["epoch_curve"])
            rel = p.relative_to(group_dir) if p.exists() else p
            lines.append(f"- train_n={tn}, eval_n={en}: `{rel}`\n")

    plot_summary(fig_dir, rows)
    lines.append("\n## Summary Plots\n\n")
    for name in ("selection_bit_f1_far.png", "bit_F1_heatmap.png", "total_far_heatmap.png"):
        lines.append(f"![{name}](figs_sota/{name})\n\n")

    with open(out, "w", encoding="utf-8") as f:
        f.writelines(lines)
    with open(group_dir / "summary_sota_matrix.json", "w", encoding="utf-8") as f:
        json.dump(rows, f, indent=2, ensure_ascii=False)
    print(f"[sota-report] wrote {out}")


def plot_summary(fig_dir: Path, rows: list[dict[str, Any]]) -> None:
    train_sizes = sorted({r["train_n"] for r in rows})
    eval_sizes = sorted({r["eval_n"] for r in rows})
    by = {(r["train_n"], r["eval_n"], r["selection"]): r for r in rows}

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    colors = {"val_f1": "#1f77b4", "val_margin": "#2ca02c"}
    for en in eval_sizes:
        for sel in SELS:
            ys = [by[(tn, en, sel)]["bit_F1"] for tn in train_sizes]
            axes[0].plot(train_sizes, ys, marker="o", color=colors[sel],
                         linestyle="-" if sel == "val_f1" else "--",
                         label=f"{sel} eval_n={en}")
            far = [by[(tn, en, sel)]["total_far"] * 100.0 for tn in train_sizes]
            axes[1].plot(train_sizes, far, marker="s", color=colors[sel],
                         linestyle="-" if sel == "val_f1" else "--",
                         label=f"{sel} eval_n={en}")
    axes[0].set_xlabel("train_n per class")
    axes[0].set_ylabel("eval bit_F1")
    axes[0].set_ylim(0, 1.02)
    axes[0].grid(True, alpha=0.25)
    axes[1].set_xlabel("train_n per class")
    axes[1].set_ylabel("Total FAR (%)")
    axes[1].grid(True, alpha=0.25)
    axes[1].legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(fig_dir / "selection_bit_f1_far.png", dpi=140)
    plt.close(fig)

    for metric, title, fname, scale in (
        ("bit_F1", "eval bit_F1", "bit_F1_heatmap.png", 1.0),
        ("total_far", "Total FAR (%)", "total_far_heatmap.png", 100.0),
    ):
        fig, axes = plt.subplots(1, 2, figsize=(10, 4.2), sharey=True)
        for ax, sel in zip(axes, SELS):
            mat = np.zeros((len(train_sizes), len(eval_sizes)), dtype=float)
            for i, tn in enumerate(train_sizes):
                for j, en in enumerate(eval_sizes):
                    mat[i, j] = by[(tn, en, sel)][metric] * scale
            im = ax.imshow(mat, aspect="auto")
            ax.set_title(sel)
            ax.set_xticks(range(len(eval_sizes)), [str(x) for x in eval_sizes])
            ax.set_yticks(range(len(train_sizes)), [str(x) for x in train_sizes])
            ax.set_xlabel("eval_n")
            ax.set_ylabel("train_n")
            for i in range(len(train_sizes)):
                for j in range(len(eval_sizes)):
                    txt = f"{mat[i, j]:.4f}" if metric == "bit_F1" else f"{mat[i, j]:.2f}"
                    ax.text(j, i, txt, ha="center", va="center", color="white", fontsize=8)
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        plt.suptitle(title)
        plt.tight_layout()
        plt.savefig(fig_dir / fname, dpi=140)
        plt.close(fig)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--group-dir", default=os.environ.get("MEGA_GROUP_DIR", ""))
    ap.add_argument("--train-sizes", default=os.environ.get("MEGA_TRAIN_SIZES", "50,100,200,400"))
    ap.add_argument("--eval-sizes", default=os.environ.get("MEGA_EVAL_SIZES", "200,2000,20000"))
    args = ap.parse_args()
    if not args.group_dir:
        raise SystemExit("--group-dir required")
    group_dir = Path(args.group_dir)
    train_sizes = [int(x) for x in args.train_sizes.replace(" ", ",").split(",") if x.strip()]
    eval_sizes = [int(x) for x in args.eval_sizes.replace(" ", ",").split(",") if x.strip()]
    rows = collect(group_dir, train_sizes, eval_sizes)
    write_report(group_dir, rows)
    print("[sota-report] final val_f1 vs val_margin best-model eval performance")
    for r in sorted(rows, key=lambda x: (x["train_n"], x["eval_n"], x["selection"])):
        print(
            f"[sota-report] t{r['train_n']} e{r['eval_n']} {r['selection']} "
            f"ep{r['best_epoch']} bit_F1={r['bit_F1']:.4f} "
            f"FAR={r['total_far']*100.0:.2f}% NI={r['ni_far']*100.0:.2f}% "
            f"OOD={r['ood_far']*100.0:.2f}%",
            flush=True,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
