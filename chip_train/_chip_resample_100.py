#!/usr/bin/env python3
"""Resample classification_chips to N per class with pixel-based defect filter.

Goal: from existing huge classification_chips/<class>/ (76k-214k per class),
pick top-N chips with highest defect-pixel ratio per class, write to new dir.

Filters:
- bin in filename >= MIN_BIN (default 200; raise to 250 for stricter)
- defect pixel ratio (non-grade-0 pixels) >= MIN_DEFECT_RATIO (default 0.05)

Existing dataset is preserved by ARCHIVING (mv classification_chips ->
classification_chips_archive_full) before writing new clean small one.

Usage:
    python _chip_resample_100.py --per-class 100
    python _chip_resample_100.py --per-class 200 --min-bin 250 --min-defect-ratio 0.10
"""
from __future__ import annotations
import argparse, os, re, random, shutil, sys
from pathlib import Path
import numpy as np
from PIL import Image

CLASSES = ['bank_boundary', 'fork', 'invalid_main', 'scratch', 'scratch_rot']
SRC_DIR = Path("D:/project/data/wm-811k/classification_chips")
ARCHIVE_DIR = Path("D:/project/data/wm-811k/classification_chips_archive_full")

BIN_RE = re.compile(r'_[Bb](\d+)\.png$')


def parse_bin(name: str) -> int | None:
    m = BIN_RE.search(name)
    return int(m.group(1)) if m else None


def defect_ratio(path: Path) -> float:
    """Fraction of non-grade-0 pixels (assumes grade 0 = (255, 255, 255) white BG).

    Palette PNG: each pixel is grade 0..7. grade 0 = pure white (background).
    Defect chips have some fraction of pixels at grade 1..7 (non-white).
    """
    try:
        img = Image.open(path).convert('RGB')
    except Exception:
        return 0.0
    arr = np.asarray(img)  # (H, W, 3)
    # non-white = any channel < 255
    non_white = (arr < 255).any(axis=-1)
    return float(non_white.mean())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--per-class', type=int, default=100)
    ap.add_argument('--min-bin', type=int, default=200)
    ap.add_argument('--min-defect-ratio', type=float, default=0.05,
                    help='Minimum non-white pixel ratio. 0.05 = 5%% non-white.')
    ap.add_argument('--scan-cap', type=int, default=2000,
                    help='Max files to scan per class (random sample). Saves time.')
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--archive', action='store_true', default=True,
                    help='Move existing SRC_DIR to ARCHIVE_DIR before writing new (default True)')
    ap.add_argument('--no-archive', dest='archive', action='store_false')
    ap.add_argument('--dry-run', action='store_true')
    args = ap.parse_args()

    rng = random.Random(args.seed)

    # 0. Verify source exists
    if not SRC_DIR.exists():
        print(f"[error] {SRC_DIR} not found", file=sys.stderr)
        sys.exit(2)
    for cls in CLASSES:
        if not (SRC_DIR / cls).is_dir():
            print(f"[error] {SRC_DIR / cls} missing", file=sys.stderr)
            sys.exit(2)

    # 1. Plan: scan + filter per class, pick top N by defect_ratio
    plans = {}
    print("=== scan + filter ===")
    for cls in CLASSES:
        cls_dir = SRC_DIR / cls
        all_files = [p for p in cls_dir.iterdir() if p.suffix.lower() == '.png']
        # bin filter
        ok_bin = []
        for p in all_files:
            b = parse_bin(p.name)
            if b is not None and b >= args.min_bin:
                ok_bin.append(p)
        # cap + random sample for scan
        if len(ok_bin) > args.scan_cap:
            sampled = rng.sample(ok_bin, args.scan_cap)
        else:
            sampled = ok_bin
        # defect ratio
        scored = []
        for p in sampled:
            r = defect_ratio(p)
            if r >= args.min_defect_ratio:
                scored.append((r, p))
        scored.sort(reverse=True, key=lambda x: x[0])
        chosen = scored[:args.per_class]
        plans[cls] = chosen
        print(f"  {cls}: total={len(all_files)} bin>={args.min_bin}={len(ok_bin)} "
              f"scanned={len(sampled)} passed_filter={len(scored)} chosen={len(chosen)}")
        if chosen:
            top_r = chosen[0][0]; bot_r = chosen[-1][0]
            print(f"    defect_ratio range: {bot_r:.3f} ~ {top_r:.3f}")

    # 2. Sanity check
    short = [c for c in CLASSES if len(plans[c]) < args.per_class]
    if short:
        print(f"\n[warn] classes under target: {short}")
        print(f"  consider lowering --min-defect-ratio (current {args.min_defect_ratio}) "
              f"or raising --scan-cap (current {args.scan_cap})")

    if args.dry_run:
        print("\n[dry-run] no files moved/copied")
        return

    # 3. Archive existing
    if args.archive:
        if ARCHIVE_DIR.exists():
            print(f"\n[archive] {ARCHIVE_DIR} already exists - leaving SRC_DIR alone, "
                  f"writing new files alongside (skip archive)")
        else:
            print(f"\n[archive] mv {SRC_DIR} -> {ARCHIVE_DIR}")
            shutil.move(str(SRC_DIR), str(ARCHIVE_DIR))
            SRC_DIR.mkdir(parents=True, exist_ok=True)
    # if no archive, write into existing SRC_DIR (additive)

    # 4. Copy chosen files to new SRC_DIR
    print("\n=== copy chosen ===")
    for cls in CLASSES:
        out_cls = SRC_DIR / cls
        out_cls.mkdir(parents=True, exist_ok=True)
        for r, p in plans[cls]:
            # source path now under archive
            if args.archive:
                src = ARCHIVE_DIR / cls / p.name
            else:
                src = p
            dst = out_cls / p.name
            shutil.copy2(src, dst)
        print(f"  {cls}: {len(plans[cls])} copied")

    # 5. Final summary
    print("\n=== final classification_chips ===")
    for cls in CLASSES:
        n = len(list((SRC_DIR / cls).iterdir()))
        print(f"  {cls}: {n}")


if __name__ == "__main__":
    main()
