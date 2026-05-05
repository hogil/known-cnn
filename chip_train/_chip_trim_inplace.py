#!/usr/bin/env python3
"""In-place trim: keep top-N chips per class by defect_pixel_ratio, delete rest.

NO archive (user said 그냥 싹 지우고). In-place delete.

Usage:
    python _chip_trim_inplace.py --per-class 100
"""
from __future__ import annotations
import argparse, os, re, random, sys
from pathlib import Path
import numpy as np
from PIL import Image

CLASSES = ['bank_boundary', 'fork', 'invalid_main', 'scratch', 'scratch_rot']
STUB_DIRS = ['particle_blast', 'scratch_21deg']
SRC_DIR = Path("D:/project/data/wm-811k/classification_chips")

BIN_RE = re.compile(r'_[Bb](\d+)\.png$')


def parse_bin(name: str) -> int | None:
    m = BIN_RE.search(name)
    return int(m.group(1)) if m else None


def defect_ratio(path: Path) -> float:
    try:
        img = Image.open(path).convert('RGB')
    except Exception:
        return 0.0
    arr = np.asarray(img)
    return float((arr < 255).any(axis=-1).mean())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--per-class', type=int, default=100)
    ap.add_argument('--min-bin', type=int, default=200)
    ap.add_argument('--min-defect-ratio', type=float, default=0.05)
    ap.add_argument('--scan-cap', type=int, default=2000)
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--dry-run', action='store_true')
    args = ap.parse_args()

    rng = random.Random(args.seed)

    if not SRC_DIR.exists():
        print(f"[error] {SRC_DIR} not found", file=sys.stderr)
        sys.exit(2)

    plans = {}
    print("=== scan + pick top-N ===")
    for cls in CLASSES:
        cls_dir = SRC_DIR / cls
        if not cls_dir.is_dir():
            print(f"  [warn] {cls_dir} missing, skip")
            plans[cls] = []
            continue
        all_files = [p for p in cls_dir.iterdir() if p.suffix.lower() == '.png']
        ok_bin = [p for p in all_files
                  if (b := parse_bin(p.name)) is not None and b >= args.min_bin]
        if len(ok_bin) > args.scan_cap:
            sampled = rng.sample(ok_bin, args.scan_cap)
        else:
            sampled = ok_bin
        scored = [(defect_ratio(p), p) for p in sampled]
        scored = [(r, p) for r, p in scored if r >= args.min_defect_ratio]
        scored.sort(reverse=True, key=lambda x: x[0])
        chosen = [p for _, p in scored[:args.per_class]]
        plans[cls] = chosen
        rs = [r for r, _ in scored[:args.per_class]]
        rng_str = f"{min(rs):.3f}~{max(rs):.3f}" if rs else "n/a"
        print(f"  {cls}: total={len(all_files)} bin>={args.min_bin}={len(ok_bin)} "
              f"chosen={len(chosen)} defect_ratio={rng_str}")

    short = [c for c in CLASSES if len(plans[c]) < args.per_class]
    if short:
        print(f"\n[warn] under target classes: {short}")

    if args.dry_run:
        print("\n[dry-run] no deletions")
        return

    # In-place delete: for each class, delete files not in keep_set
    print("\n=== in-place delete (keep top-N, remove rest) ===")
    for cls in CLASSES:
        cls_dir = SRC_DIR / cls
        if not cls_dir.is_dir():
            continue
        keep_names = {p.name for p in plans[cls]}
        deleted = 0
        for p in cls_dir.iterdir():
            if p.is_file() and p.name not in keep_names:
                p.unlink()
                deleted += 1
        kept = len(list(cls_dir.iterdir()))
        print(f"  {cls}: kept={kept} deleted={deleted}")

    # Delete stub dirs entirely
    print("\n=== delete stub dirs ===")
    import shutil
    for stub in STUB_DIRS:
        d = SRC_DIR / stub
        if d.is_dir():
            n = sum(1 for _ in d.iterdir())
            shutil.rmtree(d)
            print(f"  removed {d} ({n} files)")
        else:
            print(f"  {d} not present, skip")

    # Final summary
    print("\n=== final classification_chips/ ===")
    for cls_dir in sorted(SRC_DIR.iterdir()):
        if cls_dir.is_dir():
            n = len(list(cls_dir.iterdir()))
            print(f"  {cls_dir.name}: {n}")


if __name__ == "__main__":
    main()
