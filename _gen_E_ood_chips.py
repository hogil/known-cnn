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


def extract_class(cls: str, per_class: int, defect_thresh: float, rng: random.Random,
                  src_root: Path = None, dst_root: Path = None):
    # Resolve src/dst roots explicitly. Caller passes via param OR sets module attrs.
    sr = src_root if src_root is not None else SRC_ROOT
    dr = dst_root if dst_root is not None else DST_ROOT
    src_dir = sr / cls
    dst_dir = dr / cls
    print(f"[OOD] {cls}: src={src_dir}  dst={dst_dir}")
    dst_dir.mkdir(parents=True, exist_ok=True)

    # Incremental: count existing PNGs, generate only delta
    existing = sorted(dst_dir.glob("*.png"))
    n_have = len(existing)
    if n_have >= per_class:
        print(f"[OOD] {cls}: already has {n_have} ≥ {per_class}, skip")
        return n_have
    if n_have > 0:
        print(f"[OOD] {cls}: have {n_have}, need {per_class - n_have} more (incremental)")
    target_delta = per_class - n_have
    idx_start = n_have  # continue numbering from existing

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
    # 사용 가능한 만큼 저장 (target_delta 보다 적으면 적은 대로)
    pick = candidates[:target_delta]

    saved = 0
    for k, (wi, gy, gx) in enumerate((c[0], c[1], c[2]) for c in pick):
        im = Image.open(wafers[wi])
        arr = np.array(im, dtype=np.uint8)
        y0, x0 = gy * CHIP, gx * CHIP
        tile = arr[y0:y0+CHIP, x0:x0+CHIP]
        out = Image.fromarray(tile, mode="P")
        out.putpalette(palette_bytes)
        idx = idx_start + k  # continue from existing
        out_path = dst_dir / f"{cls}_{idx:04d}.png"
        out.save(out_path, optimize=True)
        saved += 1

    total = n_have + saved
    print(f"[OOD] {cls}: saved {saved} new → {dst_dir} (total now {total}/{per_class})")
    return total


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
