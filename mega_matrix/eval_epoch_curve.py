"""Evaluate every saved epoch checkpoint on one eval_n condition."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

from chip_multilabel._per_epoch_multi_eval import (
    evaluate,
    format_compact,
    load_multi_val,
)
from chip_multilabel.model_io import load_chip_backbone


class _KeepTrainLogits(torch.nn.Module):
    def __init__(self, model: torch.nn.Module, keep_indices: list[int]):
        super().__init__()
        self.model = model
        self.keep_indices = keep_indices

    def forward(self, x):
        y = self.model(x)
        return y[:, self.keep_indices]


def _load_history(run_dir: Path) -> dict[int, dict[str, Any]]:
    path = run_dir / "history.json"
    if not path.exists():
        return {}
    with open(path, "r", encoding="utf-8") as f:
        rows = json.load(f)
    return {int(r["epoch"]): r for r in rows}


def _json_dumps(v: Any) -> str:
    return json.dumps(v, ensure_ascii=False, sort_keys=True)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--eval-set", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--n-per-class", type=int, required=True)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--num-workers", type=int, default=0)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", default=None)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    run_dir = Path(args.run_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "epoch_metrics.csv"
    md_path = out_dir / "epoch_metrics.md"
    png_path = out_dir / "epoch_metrics.png"
    if csv_path.exists() and png_path.exists() and not args.force:
        print(f"[epoch-eval] exists, skip: {out_dir}", flush=True)
        return 0

    ckpts = sorted(run_dir.glob("epoch_*_model.pth"))
    if not ckpts:
        raise FileNotFoundError(f"no epoch checkpoints under {run_dir}")
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))

    first_model, meta, _ = load_chip_backbone(ckpts[0], device)
    img_size = int(meta["img_size"])
    del first_model
    if device.type == "cuda":
        torch.cuda.empty_cache()

    val_cache = load_multi_val(
        args.eval_set,
        n_per_class=args.n_per_class,
        seed=args.seed,
        img_size=img_size,
    )
    hist = _load_history(run_dir)
    rows: list[dict[str, Any]] = []
    print(
        f"[epoch-eval] eval_set={args.eval_set} "
        f"N={len(val_cache['paths'])} cap={args.n_per_class}/class "
        f"epochs={len(ckpts)}",
        flush=True,
    )

    for ckpt in ckpts:
        ep = int(ckpt.stem.split("_")[1])
        model, _meta, keep_indices = load_chip_backbone(ckpt, device)
        model = _KeepTrainLogits(model, keep_indices).to(device).eval()
        m = evaluate(
            model,
            val_cache,
            device=str(device),
            threshold=0.5,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
        )
        h = hist.get(ep, {})
        row = {
            "epoch": ep,
            "train_loss": h.get("train_loss"),
            "val_f1": h.get("val_f1"),
            "val_margin": h.get("val_margin"),
            "bit_F1": m["bit_F1"],
            "total_far_pct": m["total_far"],
            "ni_far_pct": m["ni_far"],
            "ood_far_pct": m["ood_far"],
            "n_pos": m["n_pos"],
            "n_neg": m["n_neg"],
            "per_bit_f1": m["per_bit_f1"],
            "per_class_far_pct": m["per_class_far"],
        }
        rows.append(row)
        print(f"[epoch-eval] ep {ep:02d} {format_compact(m)}", flush=True)
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    fieldnames = [
        "epoch", "train_loss", "val_f1", "val_margin", "bit_F1",
        "total_far_pct", "ni_far_pct", "ood_far_pct", "n_pos", "n_neg",
        "per_bit_f1", "per_class_far_pct",
    ]
    with open(csv_path, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            rr = dict(r)
            rr["per_bit_f1"] = _json_dumps(rr["per_bit_f1"])
            rr["per_class_far_pct"] = _json_dumps(rr["per_class_far_pct"])
            w.writerow(rr)

    with open(md_path, "w", encoding="utf-8") as f:
        f.write("# Epoch Metrics\n\n")
        f.write(f"- run_dir: `{run_dir}`\n")
        f.write(f"- eval_set: `{args.eval_set}`\n")
        f.write(f"- n_per_class: `{args.n_per_class}`\n\n")
        f.write("| epoch | val_f1 | val_margin | bit_F1 | Total FAR | NI FAR | OOD FAR |\n")
        f.write("|---:|---:|---:|---:|---:|---:|---:|\n")
        for r in rows:
            f.write(
                f"| {r['epoch']} | {float(r['val_f1']):.4f} | "
                f"{float(r['val_margin']):.4f} | {r['bit_F1']:.4f} | "
                f"{r['total_far_pct']:.2f}% | {r['ni_far_pct']:.2f}% | "
                f"{r['ood_far_pct']:.2f}% |\n"
            )

    xs = [int(r["epoch"]) for r in rows]
    bit = [float(r["bit_F1"]) for r in rows]
    far = [float(r["total_far_pct"]) for r in rows]
    ni = [float(r["ni_far_pct"]) for r in rows]
    ood = [float(r["ood_far_pct"]) for r in rows]
    fig, ax1 = plt.subplots(figsize=(9, 4.8))
    ax1.plot(xs, bit, marker="o", color="#1f77b4", label="bit_F1")
    ax1.set_xlabel("epoch")
    ax1.set_ylabel("bit_F1")
    ax1.set_ylim(0.0, 1.02)
    ax1.grid(True, alpha=0.25)
    ax2 = ax1.twinx()
    ax2.plot(xs, far, marker="s", color="#d62728", label="Total FAR")
    ax2.plot(xs, ni, marker="^", color="#ff7f0e", linestyle="--", label="NI FAR")
    ax2.plot(xs, ood, marker="v", color="#2ca02c", linestyle="--", label="OOD FAR")
    ax2.set_ylabel("FAR (%)")
    h1, l1 = ax1.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax1.legend(h1 + h2, l1 + l2, loc="best")
    plt.title("Epoch eval bit_F1 / FAR")
    plt.tight_layout()
    plt.savefig(png_path, dpi=140)
    plt.close(fig)
    print(f"[epoch-eval] saved {csv_path}", flush=True)
    print(f"[epoch-eval] saved {png_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
