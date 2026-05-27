#!/usr/bin/env python3
"""Extract 200x200 OOD chips from D: wafer-canvas (unknown/<OOD_class>/*.png).

For each OOD class (CenterDonut/CrossScratch/DiagonalSmear/Starburst):
- Read wafer 6400x6400 palette PNGs
- Crop 32x32 grid of 200x200 chips per wafer
- Filter chips with sufficient defect signal (non-baseline pixel ratio > threshold)
- Random sample N chips per class
- Save to E:\\data\\images\\chip_multilabel_v15direct\\<OOD_class>\\<class>_<idx:04d>.png

Output palette preserved (8-bit colormap).

260520 — parallel generate_direct_class via multiprocessing.Pool
(default workers = os.cpu_count()).
"""
from __future__ import annotations
import argparse
import os
import random
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
from PIL import Image

CHIP = 200
WAFER = 6400
GRID = WAFER // CHIP   # 32
# 260520 — module-level defaults removed; callers pass src_root / dst_root explicitly.
# (mega_matrix.gen_data passes dst_root = DATA_ROOT/eval_nN. extract_class() optional path
# only used in standalone main() — kept as None to force CLI arg.)
SRC_ROOT: Path | None = None
DST_ROOT: Path | None = None
OOD_CLASSES = ("CenterDonut", "CrossScratch", "DiagonalSmear", "Starburst")


def _palette():
    try:
        from dist_apply._sample_gen import PALETTE
        return PALETTE
    except Exception:
        pal = [
            255, 255, 255, 155, 155, 155, 0, 150, 25, 0, 0, 255,
            217, 29, 255, 255, 255, 0, 255, 0, 0, 0, 0, 0,
            220, 238, 255, 0, 0, 1, 190, 190, 190, 255, 153, 0,
        ]
        while len(pal) < 96:
            pal.append(0)
        return pal


def _grid():
    yy = np.arange(CHIP, dtype=np.float32)[:, None]
    xx = np.arange(CHIP, dtype=np.float32)[None, :]
    return yy, xx


def _line_alpha(yy, xx, cy, cx, angle, sigma, half_len, peak):
    cos_a = float(np.cos(angle))
    sin_a = float(np.sin(angle))
    d_perp = cos_a * (yy - cy) - sin_a * (xx - cx)
    d_along = sin_a * (yy - cy) + cos_a * (xx - cx)
    core = np.exp(-(d_perp * d_perp) / (sigma * sigma)).astype(np.float32)
    taper = np.exp(-np.maximum(np.abs(d_along) - half_len, 0.0) ** 2 / ((sigma * 8.0) ** 2))
    return peak * core * taper.astype(np.float32)


def _alpha_for_chip(cls: str, rng: np.random.Generator):
    yy, xx = _grid()
    cy = CHIP / 2.0 + rng.uniform(-12, 12)
    cx = CHIP / 2.0 + rng.uniform(-12, 12)
    if cls == "CenterDonut":
        r = np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2)
        r0 = rng.uniform(16, 34)
        sigma = rng.uniform(2.5, 5.0)
        return rng.uniform(0.55, 0.90) * np.exp(-((r - r0) ** 2) / (sigma * sigma))
    if cls == "CrossScratch":
        angle = rng.uniform(-0.15, 0.15)
        sigma = rng.uniform(2.5, 5.5)
        half_len = rng.uniform(55, 95)
        a1 = _line_alpha(yy, xx, cy, cx, angle, sigma, half_len, rng.uniform(0.55, 0.90))
        a2 = _line_alpha(yy, xx, cy, cx, angle + np.pi / 2.0, sigma, half_len, rng.uniform(0.55, 0.90))
        return np.maximum(a1, a2)
    if cls == "DiagonalSmear":
        angle = np.deg2rad(rng.uniform(35, 55))
        return _line_alpha(yy, xx, cy, cx, angle, rng.uniform(4.0, 9.0),
                           rng.uniform(70, 120), rng.uniform(0.45, 0.85))
    if cls == "Starburst":
        dy = yy - cy
        dx = xx - cx
        r = np.sqrt(dy * dy + dx * dx)
        theta = np.arctan2(dy, dx)
        alpha = np.exp(-(r * r) / (rng.uniform(7, 13) ** 2)) * rng.uniform(0.65, 0.95)
        n_rays = int(rng.integers(8, 15))
        th0 = rng.uniform(0, 2 * np.pi)
        for i in range(n_rays):
            ray_angle = th0 + 2 * np.pi * i / n_rays + rng.uniform(-0.04, 0.04)
            dth = (theta - ray_angle + np.pi) % (2 * np.pi) - np.pi
            d_perp = r * np.sin(dth)
            forward = np.cos(dth) > 0
            ray = np.exp(-(d_perp * d_perp) / (rng.uniform(2.0, 4.0) ** 2))
            ray *= (r < rng.uniform(70, 105)) & forward
            alpha = np.maximum(alpha, ray * rng.uniform(0.35, 0.80))
        return alpha.astype(np.float32)
    raise ValueError(f"unknown OOD class: {cls}")


def _render_direct_chip(cls: str, rng: np.random.Generator):
    base_u = rng.random((CHIP, CHIP))
    arr = np.where(base_u < 0.83, 0, np.where(base_u < 0.98, 1, 2)).astype(np.uint8)
    alpha = np.clip(_alpha_for_chip(cls, rng), 0.0, 1.0)
    hit = rng.random((CHIP, CHIP)) < alpha
    grade_u = rng.random((CHIP, CHIP))
    defect = np.where(grade_u < 0.55, 2,
                      np.where(grade_u < 0.82, 3,
                               np.where(grade_u < 0.95, 4, 5))).astype(np.uint8)
    arr = np.where(hit, defect, arr).astype(np.uint8)
    return arr


def _gen_one_ood_chip(args):
    """Worker: render one OOD chip + save palette PNG.

    args = (cls, idx, seed, pal_bytes, dst_dir_str)
    Returns 1 on success, 0 if exists.
    """
    cls, idx, seed, pal, dst_dir_str = args
    dst_dir = Path(dst_dir_str)
    out_path = dst_dir / f"{cls}_{idx:05d}.png"
    if out_path.exists():
        return 0
    nrng = np.random.default_rng(seed)
    arr = _render_direct_chip(cls, nrng)
    out = Image.fromarray(arr, mode="P")
    out.putpalette(pal)
    out.save(out_path, optimize=False, compress_level=1)
    return 1


def generate_direct_class(cls: str, per_class: int, rng: random.Random,
                          dst_root: Path = None, n_workers: int = None):
    """OOD chip synth — 260527 delegated to the current-version synth
    (sota_h100.synth.iter_ood_chips: wafer-pattern chip-crop puzzle, no full wafer).
    One pattern realization yields many chips; loop seeds until per_class."""
    from sota_h100 import synth
    dr = dst_root if dst_root is not None else DST_ROOT
    dst_dir = dr / cls
    dst_dir.mkdir(parents=True, exist_ok=True)
    n_have = len(sorted(dst_dir.glob("*.png")))
    if n_have >= per_class:
        print(f"[OOD] {cls}: already {n_have} >= {per_class}, skip")
        return n_have
    if cls not in synth.OOD_CLASSES:
        print(f"[OOD] {cls}: not an OOD class in synth ({synth.OOD_CLASSES}); skip")
        return n_have
    print(f"[OOD] {cls}: synth.iter_ood_chips {n_have}/{per_class}", flush=True)
    idx = n_have
    wafer_i = 0
    max_wafer = per_class * 5 + 50
    while idx < per_class and wafer_i < max_wafer:
        seed = rng.randrange(2**31)
        for gy, gx, img in synth.iter_ood_chips(cls, seed):
            if idx >= per_class:
                break
            img.save(dst_dir / f"{cls}_{idx:04d}.png", optimize=False, compress_level=1)
            idx += 1
        wafer_i += 1
        if wafer_i % 20 == 0 or idx >= per_class:
            print(f"[OOD] {cls}: {idx}/{per_class} ({wafer_i} patterns)", flush=True)
    print(f"[OOD] {cls}: done {idx}/{per_class}", flush=True)
    return idx


# Legacy _render_direct_chip-based variant — kept for back-compat only.
# Should NOT be called by new code; superseded by generate_direct_class above (260521).
def generate_direct_class_legacy_chip_local(cls: str, per_class: int, rng: random.Random,
                          dst_root: Path = None, n_workers: int = None):
    dr = dst_root if dst_root is not None else DST_ROOT
    dst_dir = dr / cls
    dst_dir.mkdir(parents=True, exist_ok=True)
    existing = sorted(dst_dir.glob("*.png"))
    n_have = len(existing)
    if n_have >= per_class:
        print(f"[OOD] {cls}: direct already has {n_have} >= {per_class}, skip")
        return n_have

    pal = _palette()
    if n_workers is None:
        n_workers = max(1, (os.cpu_count() or 4) - 1)
    todo = list(range(n_have, per_class))
    print(f"[OOD] {cls}: direct chip generation {n_have}/{per_class} "
          f"(workers={n_workers}, todo={len(todo)})", flush=True)

    # Pre-draw seeds in main process so reproducibility holds across worker counts.
    tasks = [(cls, idx, rng.randrange(2**31 - 1), pal, str(dst_dir)) for idx in todo]

    if n_workers <= 1 or len(tasks) < 4:
        # Serial fallback (avoid spawn overhead for tiny jobs)
        done = n_have
        step = max(1, per_class // 10)
        for t in tasks:
            _gen_one_ood_chip(t)
            done += 1
            if done == per_class or done % step == 0:
                print(f"[OOD] {cls}: direct {done}/{per_class}", flush=True)
        return per_class

    done = n_have
    step = max(1, per_class // 10)
    with ProcessPoolExecutor(max_workers=n_workers) as exe:
        for _ in exe.map(_gen_one_ood_chip, tasks, chunksize=max(1, len(tasks) // (n_workers * 4))):
            done += 1
            if done == per_class or done % step == 0:
                print(f"[OOD] {cls}: direct {done}/{per_class}", flush=True)
    return per_class


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
        print(f"[OOD] {cls}: no wafer source; generating direct chips")
        return generate_direct_class(cls, per_class, rng, dr)
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
    ap.add_argument("--src-root", type=Path, required=True,
                    help="wafer-canvas source: <src>/<OOD_class>/*.png (6400x6400 palette)")
    ap.add_argument("--dst-root", type=Path, required=True,
                    help="output: <dst>/<OOD_class>/<class>_NNNN.png (200x200)")
    args = ap.parse_args()

    rng = random.Random(args.seed)
    total = 0
    for cls in OOD_CLASSES:
        total += extract_class(cls, args.per_class, args.defect_thresh, rng,
                               src_root=args.src_root, dst_root=args.dst_root)
    print(f"[OOD] TOTAL {total} chips across {len(OOD_CLASSES)} classes")
    print(f"[OUT] {args.dst_root}")


if __name__ == "__main__":
    main()
