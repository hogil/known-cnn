#!/usr/bin/env python3
"""Extract 200x200 OOD chips from D: wafer-canvas (unknown/<OOD_class>/*.png).

For each OOD class (CenterDonut/CrossScratch/DiagonalSmear/Starburst):
- Read wafer 6400x6400 palette PNGs
- Crop 32x32 grid of 200x200 chips per wafer
- Filter chips with sufficient defect signal (non-baseline pixel ratio > threshold)
- Random sample N chips per class
- Save to E:\\data\\images\\chip_multilabel_v15direct\\<OOD_class>\\<class>_<idx:04d>.png

Output palette preserved (8-bit colormap).
"""
from __future__ import annotations
import argparse
import random
import sys
from pathlib import Path

import numpy as np
from PIL import Image

CHIP = 200
WAFER = 6400
GRID = WAFER // CHIP   # 32
SRC_ROOT = Path("D:/project/data/wm-811k/unknown")
DST_ROOT = Path("E:/data/images/chip_multilabel_v15direct")
OOD_CLASSES = ("CenterDonut", "CrossScratch", "DiagonalSmear", "Starburst")


def extract_class(cls: str, per_class: int, defect_thresh: float, rng: random.Random):
    src_dir = SRC_ROOT / cls
    dst_dir = DST_ROOT / cls
    dst_dir.mkdir(parents=True, exist_ok=True)

    wafers = sorted(src_dir.glob("*.png"))
    if not wafers:
        print(f"[OOD] {cls}: no wafer source")
        return 0
    print(f"[OOD] {cls}: {len(wafers)} wafers, target={per_class}, thresh={defect_thresh}")

    palette_bytes = None
    candidates = []  # list of (wafer_idx, gy, gx, defect_ratio)
    for wi, wp in enumerate(wafers):
        im = Image.open(wp)
        if im.mode != "P":
            print(f"[OOD] {cls} {wp.name}: skip (mode={im.mode})")
            continue
        if palette_bytes is None:
            palette_bytes = im.getpalette()
        arr = np.array(im, dtype=np.uint8)
        if arr.shape != (WAFER, WAFER):
            print(f"[OOD] {cls} {wp.name}: skip (shape={arr.shape})")
            continue
        for gy in range(GRID):
            for gx in range(GRID):
                y0, x0 = gy * CHIP, gx * CHIP
                tile = arr[y0:y0+CHIP, x0:x0+CHIP]
                # baseline = grade 0 or 1, defect = grade 2+
                defect_ratio = float((tile >= 2).sum()) / (CHIP * CHIP)
                if defect_ratio >= defect_thresh:
                    candidates.append((wi, gy, gx, defect_ratio))

    print(f"[OOD] {cls}: {len(candidates)} candidates above thresh")
    if not candidates:
        print(f"[OOD] {cls}: zero candidates — abort")
        return 0
    rng.shuffle(candidates)
    if len(candidates) >= per_class:
        pick = candidates[:per_class]
    else:
        # Replacement sampling: cycle through candidates to reach per_class.
        # Some duplicates inevitable when wafer source < target.
        print(f"[OOD] {cls}: replacement sampling ({len(candidates)} < {per_class})")
        pick = [candidates[i % len(candidates)] for i in range(per_class)]

    saved = 0
    for idx, (wi, gy, gx) in enumerate((c[0], c[1], c[2]) for c in pick):
        im = Image.open(wafers[wi])
        arr = np.array(im, dtype=np.uint8)
        y0, x0 = gy * CHIP, gx * CHIP
        tile = arr[y0:y0+CHIP, x0:x0+CHIP]
        out = Image.fromarray(tile, mode="P")
        out.putpalette(palette_bytes)
        out_path = dst_dir / f"{cls}_{idx:04d}.png"
        out.save(out_path, optimize=True)
        saved += 1

    print(f"[OOD] {cls}: saved {saved} → {dst_dir}")
    return saved


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-class", type=int, default=200)
    ap.add_argument("--defect-thresh", type=float, default=0.03,
                    help="min defect pixel ratio (default 0.03 = 3%)")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    rng = random.Random(args.seed)
    total = 0
    for cls in OOD_CLASSES:
        total += extract_class(cls, args.per_class, args.defect_thresh, rng)
    print(f"[OOD] TOTAL {total} chips across {len(OOD_CLASSES)} classes")
    print(f"[OUT] {DST_ROOT}")


if __name__ == "__main__":
    main()
