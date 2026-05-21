"""Select val_f1 and val_margin checkpoints from one SOTA training run."""
from __future__ import annotations

import argparse
import json
import os
import shutil
from pathlib import Path
from typing import Any


SELECTIONS = {
    "val_f1": "val_f1",
    "val_margin": "val_margin",
}


def _metric(row: dict[str, Any], key: str) -> float:
    v = row.get(key)
    try:
        return float(v)
    except Exception:
        return float("-inf")


def _choose(history: list[dict[str, Any]], metric_key: str) -> dict[str, Any]:
    best = None
    best_v = float("-inf")
    for row in history:
        v = _metric(row, metric_key)
        if v > best_v:
            best = row
            best_v = v
    if best is None:
        raise RuntimeError(f"no valid rows for {metric_key}")
    return best


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    run_dir = Path(args.run_dir)
    hist_path = run_dir / "history.json"
    if not hist_path.exists():
        raise FileNotFoundError(hist_path)
    with open(hist_path, "r", encoding="utf-8") as f:
        history = json.load(f)
    if not isinstance(history, list) or not history:
        raise RuntimeError(f"empty history: {hist_path}")

    selected_root = run_dir / "selected"
    selected_root.mkdir(parents=True, exist_ok=True)
    for label, metric_key in SELECTIONS.items():
        row = _choose(history, metric_key)
        ep = int(row["epoch"])
        src = run_dir / f"epoch_{ep:02d}_model.pth"
        if not src.exists():
            raise FileNotFoundError(
                f"missing epoch checkpoint for {label}: {src}. "
                "Run training with --save-every-epoch."
            )
        dst_dir = selected_root / label
        dst_dir.mkdir(parents=True, exist_ok=True)
        dst = dst_dir / "best_model.pth"
        if args.force and dst.exists():
            dst.unlink()
        if not dst.exists():
            try:
                os.link(src, dst)
            except OSError:
                shutil.copy2(src, dst)
        payload = {
            "selection": label,
            "metric_key": metric_key,
            "best_epoch": ep,
            "metric_value": _metric(row, metric_key),
            "val_f1": _metric(row, "val_f1"),
            "val_margin": _metric(row, "val_margin"),
            "val_acc": _metric(row, "val_acc"),
            "train_loss": _metric(row, "train_loss"),
            "source_checkpoint": str(src),
            "selected_checkpoint": str(dst),
        }
        with open(dst_dir / "selection.json", "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
        print(
            f"[select] {label}: ep{ep:02d} "
            f"{metric_key}={payload['metric_value']:.4f} -> {dst}",
            flush=True,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
