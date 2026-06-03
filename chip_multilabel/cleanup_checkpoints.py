#!/usr/bin/env python3
"""Clean low-value chip-multilabel checkpoints while preserving evidence files."""

from __future__ import annotations

import argparse
import csv
import os
import re
import subprocess
from pathlib import Path


REPO = Path(__file__).resolve().parents[1]
OUTPUTS = REPO / "outputs"


def _float(s: str | None, default: float = 0.0) -> float:
    try:
        return float(s or default)
    except ValueError:
        return default


def _running_tags() -> set[str]:
    tags: set[str] = set()
    try:
        out = subprocess.check_output(
            [
                "powershell",
                "-NoProfile",
                "-Command",
                "Get-CimInstance Win32_Process | "
                "Where-Object { $_.Name -eq 'python.exe' } | "
                "Select-Object -ExpandProperty CommandLine",
            ],
            text=True,
            errors="ignore",
        )
    except Exception:
        return tags
    for line in out.splitlines():
        m = re.search(r"--tag\s+([^\s]+)", line)
        if m:
            tags.add(m.group(1))
        m = re.search(r"outputs[\\/][^\\/\s]+[\\/]([^\\/\s]+)", line)
        if m:
            tags.add(m.group(1))
    return tags


def _leaderboard_rows() -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for lb in OUTPUTS.glob("*/_leaderboard.csv"):
        dataset = lb.parent.name
        with lb.open("r", encoding="utf-8", newline="") as f:
            for row in csv.DictReader(f):
                row["_dataset"] = dataset
                rows.append(row)
    return rows


def _row_dir(row: dict[str, str]) -> Path:
    return OUTPUTS / row["_dataset"] / row.get("tag", "")


def _is_bad(row: dict[str, str], min_f1: float, max_far: float) -> bool:
    if row.get("status") != "done":
        return False
    return _float(row.get("eval_bit_F1")) < min_f1 or _float(row.get("eval_Total_FAR"), 100.0) > max_far


def _inside_outputs(path: Path) -> bool:
    try:
        path.resolve().relative_to(OUTPUTS.resolve())
        return True
    except ValueError:
        return False


def collect_delete_candidates(min_f1: float, max_far: float, running: set[str]) -> list[Path]:
    candidates: set[Path] = set()

    rows = _leaderboard_rows()
    for row in rows:
        tag = row.get("tag", "")
        if not tag or tag in running:
            continue
        root = _row_dir(row)
        if not root.exists():
            continue
        if _is_bad(row, min_f1, max_far):
            # Low-value row: keep evidence logs/csv/md, remove all model weights.
            for p in root.rglob("*.pth"):
                candidates.add(p)
        else:
            # Useful row: keep best checkpoints, remove redundant epoch/final snapshots.
            for p in root.rglob("epoch_*_model.pth"):
                candidates.add(p)
            for p in root.rglob("final_epoch_model.pth"):
                candidates.add(p)

    # Generic cleanup for old dirs not represented in current leaderboards:
    # if a run directory has best_model, epoch/final snapshots are redundant.
    for best in OUTPUTS.rglob("best_model.pth"):
        run_dir = best.parent
        if any(tag and tag in str(run_dir) for tag in running):
            continue
        for p in run_dir.glob("epoch_*_model.pth"):
            candidates.add(p)
        final = run_dir / "final_epoch_model.pth"
        if final.exists():
            candidates.add(final)

    # Smoke/test outputs are never scientific evidence.
    for smoke in (OUTPUTS / "_patch_smoke",):
        if smoke.exists():
            for p in smoke.rglob("*.pth"):
                candidates.add(p)

    return sorted(p for p in candidates if p.exists() and _inside_outputs(p))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--min-f1", type=float, default=0.990)
    ap.add_argument("--max-far", type=float, default=5.0)
    ap.add_argument("--execute", action="store_true")
    args = ap.parse_args()

    running = _running_tags()
    candidates = collect_delete_candidates(args.min_f1, args.max_far, running)
    total = sum(p.stat().st_size for p in candidates if p.exists())
    print(f"running_tags={sorted(running)}")
    print(f"delete_candidates={len(candidates)}")
    print(f"delete_GB={total / (1024 ** 3):.2f}")
    for p in candidates[:200]:
        print(f"{p.stat().st_size / (1024 ** 3):.3f} GB\t{p}")
    if len(candidates) > 200:
        print(f"... {len(candidates) - 200} more")

    if not args.execute:
        print("dry_run=1")
        return

    for p in candidates:
        if p.exists() and _inside_outputs(p):
            p.unlink()
    print("deleted=1")


if __name__ == "__main__":
    main()
