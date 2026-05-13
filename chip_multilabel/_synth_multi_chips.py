#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Direct multi-defect chip synthesis — v15direct (260508).

핵심: classification_chips/ (single class) 합성과 동일한 alpha-modulation 방식으로
multi-class chip 을 한번에 합성. min-blend 와 달리 픽셀 평균 없음 — 각 defect 의
alpha 마스크 영역 그대로 보존, 한 픽셀에 여러 alpha 가 fire 하면 max alpha 의 grade 사용.

기존 _synth_chips_only.render_chip(obj=str) → 본 모듈 render_multi_chip(objs=List[str]) 확장.
Train data 와 동일 분포 (alpha 함수, intensity tier, 2-stage / 3-zone grade sampling).

Usage:
    python -m chip_multilabel._synth_multi_chips --per-class 5 --smoke   # preview only
    python -m chip_multilabel._synth_multi_chips --per-class 200          # full

Output:
    $WM811K_ROOT/chip_multilabel_v15direct/ (default: <project>/data/wm-811k)
        bank_boundary/                          (4 single, NEW direct)
        fork/
        scratch/
        scratch_rot/
        bank_boundary+fork/                     (6 2-combo, NEW direct multi)
        bank_boundary+scratch/
        bank_boundary+scratch_rot/
        fork+scratch/
        fork+scratch_rot/
        scratch+scratch_rot/
        bank_boundary+fork+scratch/             (4 3-combo, NEW direct multi)
        bank_boundary+fork+scratch_rot/
        bank_boundary+scratch+scratch_rot/
        fork+scratch+scratch_rot/
        CenterDonut/                            (4 OOD single, COPY from pre_v5)
        CrossScratch/
        DiagonalSmear/
        Starburst/
        bank_boundary+fork+ood_CenterDonut/     (4 OOD overlay, COPY from pre_v5)
        fork+scratch+ood_DiagonalSmear/
        fork+scratch_rot+ood_CrossScratch/
        scratch+scratch_rot+ood_Starburst/
        Normal/                                 (1 special, COPY from pre_v5)
        Invalid/                                (1 special, COPY from pre_v5)
        manifest.csv
        _preview/<class_key>.png                (16-grid 4x4 preview per class)
"""
from __future__ import annotations

import argparse
import csv
import os
import random
import shutil
import sys
import time
from pathlib import Path
from typing import List, Optional

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from dist_apply import _sample_gen as sg

CHIP = sg.CHIP                                                              # 200
DEFECT_CLASSES = ('bank_boundary', 'fork', 'scratch', 'scratch_rot')
DATA_ROOT = Path(os.environ.get("WM811K_ROOT", str(Path(__file__).resolve().parents[1] / "data" / "wm-811k")))
PRE_V5_ROOT = DATA_ROOT / "chip_multilabel_pre_v5_260507"  # COPY source for OOD/Normal/Invalid
DEFAULT_OUT = DATA_ROOT / "chip_multilabel_v15direct"

OOD_SINGLE_CLASSES = ('CenterDonut', 'CrossScratch', 'DiagonalSmear', 'Starburst')
OOD_OVERLAY_CLASSES = (
    'bank_boundary+fork+ood_CenterDonut',
    'fork+scratch+ood_DiagonalSmear',
    'fork+scratch_rot+ood_CrossScratch',
    'scratch+scratch_rot+ood_Starburst',
)
SPECIAL_CLASSES = ('Normal', 'Invalid')

SINGLE_DEFECT_KEYS = list(DEFECT_CLASSES)
COMBO_2_KEYS = [
    'bank_boundary+fork', 'bank_boundary+scratch', 'bank_boundary+scratch_rot',
    'fork+scratch', 'fork+scratch_rot', 'scratch+scratch_rot',
]
COMBO_3_KEYS = [
    'bank_boundary+fork+scratch', 'bank_boundary+fork+scratch_rot',
    'bank_boundary+scratch+scratch_rot', 'fork+scratch+scratch_rot',
]
DIRECT_SYNTH_KEYS = SINGLE_DEFECT_KEYS + COMBO_2_KEYS + COMBO_3_KEYS  # 14 keys
COPY_KEYS = list(OOD_SINGLE_CLASSES) + list(OOD_OVERLAY_CLASSES) + list(SPECIAL_CLASSES)  # 10 keys
ALL_KEYS = DIRECT_SYNTH_KEYS + COPY_KEYS                                       # 24 keys


def _make_palette() -> bytes:
    return sg.PALETTE


def _compute_alpha_grade_for_obj(obj: str, rng, intensity_tier: str):
    """For a defect class, compute (alpha_map, grade_map) at chip scale.

    grade_map already mixes baseline (where alpha low) with defect (where alpha high)
    according to obj-specific sampling rule (2-stage for fork/sc/sr, 3-zone for bb).
    """
    alpha_scale = sg.INTENSITY_ALPHA_SCALE[intensity_tier]
    alpha = sg.ALPHA_FNS[obj](rng) * alpha_scale
    cum_obj = np.cumsum(sg.shifted_object_dist(obj, intensity_tier))

    if obj in ('fork', 'scratch', 'scratch_rot'):
        # 2-stage sampling — matches _sample_gen render @line 954 / _synth_chips_only render_chip
        u_base = rng.random((CHIP, CHIP))
        grades_base = np.searchsorted(sg.CUM_BASE, u_base).astype(np.uint8)
        u1 = rng.random((CHIP, CHIP))
        is_defect = u1 < alpha
        if obj == 'fork':
            lo_t2, hi_t2 = 0.53, 0.90
        else:
            lo_t2, hi_t2 = 0.60, 0.91
        t2 = np.clip((alpha - lo_t2) / (hi_t2 - lo_t2), 0.0, 1.0).astype(np.float32)
        p_2 = (t2 * t2 * (3.0 - 2.0 * t2)).astype(np.float32)
        u2 = rng.random((CHIP, CHIP))
        is_2 = u2 < p_2
        u3 = rng.random((CHIP, CHIP))
        defect_other = np.where(u3 < 0.95, np.uint8(1),
                        np.where(u3 < 0.99, np.uint8(3), np.uint8(4)))
        defect_grade = np.where(is_2, np.uint8(2), defect_other)
        grade = np.where(is_defect, defect_grade, grades_base).astype(np.uint8)
    elif obj == 'bank_boundary':
        # 3-way zone mix (bg ↔ edge ↔ center)
        t_low = np.clip(alpha / 0.45, 0.0, 1.0).astype(np.float32)
        t_high = np.clip((alpha - 0.45) / 0.55, 0.0, 1.0).astype(np.float32)
        s_low = (t_low * t_low * (3.0 - 2.0 * t_low)).astype(np.float32)
        s_high = (t_high * t_high * (3.0 - 2.0 * t_high)).astype(np.float32)
        mask_low = (alpha < 0.45).astype(np.float32)
        mask_high = 1.0 - mask_low
        w_bg = (mask_low * (1.0 - s_low)).astype(np.float32)
        w_edge = (mask_low * s_low + mask_high * (1.0 - s_high)).astype(np.float32)
        w_center = (mask_high * s_high).astype(np.float32)
        cum_mixed = (w_bg[..., None] * sg.CUM_DEFECT_BG[None, None, :] +
                     w_edge[..., None] * sg.CUM_EDGE[None, None, :] +
                     w_center[..., None] * cum_obj[None, None, :])
        uu = rng.random((CHIP, CHIP))
        grade = (uu[..., None] < cum_mixed).argmax(axis=-1).astype(np.uint8)
    else:
        raise ValueError(f"unsupported defect obj: {obj}")
    return alpha.astype(np.float32), grade


def render_multi_chip(
    objs: List[str],
    rng: np.random.Generator,
    intensity_tier: Optional[str] = None,
    bin_id: Optional[int] = None,
    add_border: bool = True,
) -> Image.Image:
    """Render single 200x200 chip with multiple defects via per-pixel max alpha.

    objs: list of defect class names (subset of DEFECT_CLASSES).
          len == 1 => single defect (equivalent to render_chip).
          len >= 2 => multi-defect: per-pixel argmax(alpha) selects which obj's grade is used.
    intensity_tier: 'strong'/'mid'/'weak' (None → random per pick_intensity_tier).
    bin_id: int defect bin (None → random from kind=00C bin pool).
    """
    if not objs:
        raise ValueError("objs must be non-empty")
    for o in objs:
        if o not in DEFECT_CLASSES:
            raise ValueError(f"unsupported obj '{o}'; use {DEFECT_CLASSES}")

    if intensity_tier is None:
        intensity_tier = sg.pick_intensity_tier(rng)
    if bin_id is None:
        bin_id = int(rng.choice(sg.DEFECT_BIN_POOL['00C'], p=sg.DEFECT_BIN_WEIGHTS))

    # baseline canvas (clean baseline grade map; will be overwritten where any obj has high alpha)
    baseline_tier = sg.pick_baseline_tier(rng)
    cum_base = sg.CUM_BASELINE_TIERS[baseline_tier]
    u = rng.random((CHIP, CHIP))
    canvas = np.searchsorted(cum_base, u).astype(np.uint8)

    # compute alpha + grade pattern per obj (each grade map already contains baseline where alpha low)
    alphas, grades = [], []
    for obj in objs:
        a, g = _compute_alpha_grade_for_obj(obj, rng, intensity_tier)
        alphas.append(a)
        grades.append(g)

    if len(objs) == 1:
        canvas = grades[0]
    else:
        # per-pixel argmax over all obj alphas, take that obj's grade.
        # baseline already encoded in each obj's `grade` where its own alpha was low,
        # so for combined pixel: pick the obj-with-highest-alpha at that pixel.
        all_alpha = np.stack(alphas, axis=0)                                # (n, H, W)
        all_grade = np.stack(grades, axis=0)                                # (n, H, W)
        max_idx = all_alpha.argmax(axis=0)                                  # (H, W)
        canvas = np.take_along_axis(all_grade, max_idx[None], axis=0)[0].astype(np.uint8)

    # 2px border (main defect = first obj)
    if add_border:
        border_color = sg.BIN_TO_BORDER_IDX.get(bin_id, sg.KEY_TO_INDEX.get('border_etc', 25))
        canvas[:2, :] = border_color
        canvas[-2:, :] = border_color
        canvas[:, :2] = border_color
        canvas[:, -2:] = border_color

    img = Image.frombytes('P', (CHIP, CHIP), canvas.tobytes())
    img.putpalette(_make_palette())
    return img


def _gen_synth_class(class_key: str, n: int, out_dir: Path, rng_master: np.random.Generator) -> int:
    """Generate n direct-synth chips for class_key (single or 2-combo or 3-combo)."""
    objs = class_key.split('+')
    out_dir.mkdir(parents=True, exist_ok=True)
    made = 0
    for i in range(n):
        seed_i = int(rng_master.integers(0, 2**31 - 1))
        rng_i = np.random.default_rng(seed_i)
        prefix = ''.join(rng_i.choice(list('ABCDEFGHIJKLMNOPQRSTUVWXYZ'), size=3))
        kind = '00P' if rng_i.random() < 0.5 else '00C'
        w_idx = int(rng_i.integers(1, 25))
        TD = str(rng_i.choice(['EE', 'PT', 'PE']))
        LT = str(rng_i.choice(['NORMAL', 'PWQ', 'ENGINEER']))
        x_abs = int(rng_i.integers(0, 32))
        y_abs = int(rng_i.integers(0, 32))
        bin_id = int(rng_i.choice(sg.DEFECT_BIN_POOL[kind], p=sg.DEFECT_BIN_WEIGHTS))

        img = render_multi_chip(objs, rng_i, bin_id=bin_id)
        base = (f"{prefix}{rng_i.integers(100,999):03d}_{kind}_{w_idx:02d}"
                f"_20260508_010000_{TD}_{LT}_X{x_abs}_Y{y_abs}_B{bin_id}")
        img.save(out_dir / f"{base}.png", optimize=False, compress_level=1)
        made += 1
    return made


def _copy_class(class_key: str, n: int, out_dir: Path, src_root: Path) -> int:
    """Copy first n PNGs from src_root/<class_key>/ to out_dir."""
    src_dir = src_root / class_key
    if not src_dir.is_dir():
        print(f"[copy] WARN: source not found {src_dir}", flush=True)
        return 0
    out_dir.mkdir(parents=True, exist_ok=True)
    src_files = sorted(src_dir.glob('*.png'))
    take = src_files[:n]
    for p in take:
        shutil.copy2(p, out_dir / p.name)
    return len(take)


def _make_preview(class_dir: Path, preview_path: Path, n_per_row: int = 4) -> None:
    """Build a NxN grid preview from first N*N PNGs (or smaller square if fewer files)."""
    files = sorted(class_dir.glob('*.png'))
    if not files:
        return
    # Adaptive grid size — pick largest perfect square <= len(files), max 4x4.
    n = min(n_per_row, int(len(files) ** 0.5))
    if n < 1:
        n = 1
    files = files[:n * n]
    chips = []
    for f in files:
        with Image.open(f) as im:
            chips.append(np.array(im.convert('RGB')))
    rows = []
    for r in range(n):
        row = np.concatenate(chips[r*n:(r+1)*n], axis=1)
        rows.append(row)
    grid = np.concatenate(rows, axis=0)
    preview_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(grid).save(preview_path)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--per-class', type=int, default=200)
    ap.add_argument('--out', type=str, default=str(DEFAULT_OUT))
    ap.add_argument('--seed', type=int, default=20260508)
    ap.add_argument('--smoke', action='store_true',
                    help='smoke mode: n=5/key for direct synth + preview only (skip copy step).')
    ap.add_argument('--keys', nargs='*', default=None,
                    help='subset of class_keys to generate (default: all 24)')
    ap.add_argument('--clean-first', action='store_true',
                    help='delete existing chip_multilabel_v15direct/<class>/*.png before generation')
    args = ap.parse_args()

    out_root = Path(args.out)
    out_root.mkdir(parents=True, exist_ok=True)
    preview_root = out_root / '_preview'
    preview_root.mkdir(parents=True, exist_ok=True)

    keys = args.keys if args.keys else (DIRECT_SYNTH_KEYS + COPY_KEYS)
    if args.smoke:
        per_class = 5
        # smoke: only direct-synth subset, copy step skipped (no point copying for smoke)
        keys = [k for k in keys if k in DIRECT_SYNTH_KEYS]
        keys += [k for k in (OOD_SINGLE_CLASSES + OOD_OVERLAY_CLASSES + SPECIAL_CLASSES) if k in keys]
    else:
        per_class = args.per_class

    rng_master = np.random.default_rng(args.seed)
    t0 = time.time()
    total = 0
    summary_rows = []

    for k in keys:
        out_dir = out_root / k
        if args.clean_first and out_dir.exists():
            for p in out_dir.glob('*.png'):
                p.unlink()
        if k in DIRECT_SYNTH_KEYS:
            made = _gen_synth_class(k, per_class, out_dir, rng_master)
            kind = 'direct_synth'
        elif k in (OOD_SINGLE_CLASSES + OOD_OVERLAY_CLASSES + SPECIAL_CLASSES):
            if args.smoke:
                made = _copy_class(k, 5, out_dir, PRE_V5_ROOT)
            else:
                made = _copy_class(k, per_class, out_dir, PRE_V5_ROOT)
            kind = 'copy_from_pre_v5'
        else:
            print(f"[skip] unknown key {k}")
            continue
        _make_preview(out_dir, preview_root / f"{k}.png")
        print(f"[{kind}] {k}: {made}", flush=True)
        summary_rows.append({'class_key': k, 'kind': kind, 'n': made})
        total += made

    # manifest.csv
    manifest = out_root / 'manifest.csv'
    with manifest.open('w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=['class_key', 'kind', 'n'])
        w.writeheader()
        for row in summary_rows:
            w.writerow(row)

    elapsed = time.time() - t0
    print(f"\n[OK] {total} chips ({len(keys)} keys) in {elapsed:.1f}s "
          f"({total / max(0.1, elapsed):.1f} img/s)")
    print(f"Output: {out_root}")
    print(f"Preview: {preview_root}")


if __name__ == '__main__':
    sys.exit(main() or 0)
