#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Chip-only synthesizer — 200x200 chip 만 합성 (wafer 안 만듦).

v19 적용된 alpha_fork / alpha_scratch / alpha_scratch_rot / alpha_bank_boundary +
intensity tier (strong/mid/weak) + grade shift + 2px border + 8-color palette PNG 출력.

Usage:
    python -m dist_apply._synth_chips_only --per-class 200 --out data/wm-811k/classification_chips
    python _synth_chips_only.py --per-class 50 --classes fork scratch scratch_rot

wafer 합성 안 함. 빠르고 가벼움 (chip 200x200 만).
"""
from __future__ import annotations

import argparse
import sys
import os
import time
import random
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

try:
    from . import _sample_gen as sg
except ImportError:
    import _sample_gen as sg

PROJ_ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = Path(os.environ.get("WM811K_ROOT", str(PROJ_ROOT / "data" / "wm-811k"))).resolve()
DEFAULT_OUT = DATA_ROOT / "classification_chips"


def _make_palette() -> bytes:
    """Reuse 8-color palette from _sample_gen palette."""
    return sg.PALETTE


def _try_font(size):
    for path in ["C:/Windows/Fonts/arial.ttf", "C:/Windows/Fonts/calibri.ttf",
                 "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"]:
        if os.path.exists(path):
            try: return ImageFont.truetype(path, size)
            except Exception: pass
    return ImageFont.load_default()


def render_chip(obj: str, rng: np.random.Generator,
                intensity_tier: str = None,
                bin_id: int = None,
                add_border: bool = True,
                add_bin_text: bool = True) -> np.ndarray:
    """Render single 200x200 chip canvas with v19 alpha modulation.

    obj: 'fork'/'scratch'/'scratch_rot'/'bank_boundary'/'invalid_main'
    intensity_tier: 'strong'/'mid'/'weak' (None → random per pick_intensity_tier)
    bin_id: defect bin (200-299). None → random from kind=00C bin pool.
    """
    # 260527: chip rendering delegated to the current-version synth (sota_h100.synth).
    # bank_boundary / fork / scratch / scratch_rot -> render_single_chip;
    # invalid_main -> render_invalid_chip. (Legacy alpha path retired here.)
    from sota_h100 import synth
    if obj == 'invalid_main':
        return synth.render_invalid_chip(rng)
    return synth.render_single_chip(obj, rng, intensity_tier=intensity_tier,
                                    bin_id=bin_id, add_border=add_border)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--per-class', type=int, default=200)
    ap.add_argument('--out', type=str, default=str(DEFAULT_OUT))
    ap.add_argument('--classes', nargs='*',
                    default=['bank_boundary', 'fork', 'scratch', 'scratch_rot', 'invalid_main'])
    ap.add_argument('--seed', type=int, default=20260506)
    ap.add_argument('--clean-first', action='store_true',
                    help='delete existing classification_chips/<class>/*.png before generation')
    ap.add_argument('--prefix-pool', type=str, default=None,
                    help='ASCII prefix file (one prefix per line). Default: random 3-letter')
    args = ap.parse_args()

    out_root = Path(args.out)
    out_root.mkdir(parents=True, exist_ok=True)

    rng_master = np.random.default_rng(args.seed)
    t0 = time.time()
    total_made = 0

    for cls in args.classes:
        cls_dir = out_root / cls
        cls_dir.mkdir(parents=True, exist_ok=True)

        if args.clean_first:
            cnt = 0
            for p in cls_dir.glob('*.png'):
                p.unlink()
                cnt += 1
            print(f"[clean] {cls}: removed {cnt} old PNGs", flush=True)

        for i in range(args.per_class):
            seed_i = int(rng_master.integers(0, 2**31 - 1))
            rng_i = np.random.default_rng(seed_i)
            prefix = ''.join(rng_i.choice(list('ABCDEFGHIJKLMNOPQRSTUVWXYZ'), size=3))
            kind = '00P' if rng_i.random() < 0.5 else '00C'
            w_idx = int(rng_i.integers(1, 25))
            yld = float(rng_i.uniform(85, 99))
            sys_n = int(rng_i.integers(1, 30))
            TD = str(rng_i.choice(['EE', 'PT', 'PE']))
            LT = str(rng_i.choice(['NORMAL', 'PWQ', 'ENGINEER']))
            x_abs = int(rng_i.integers(0, 32))
            y_abs = int(rng_i.integers(0, 32))
            bin_id = int(rng_i.choice(sg.DEFECT_BIN_POOL[kind], p=sg.DEFECT_BIN_WEIGHTS))

            img = render_chip(cls, rng_i, bin_id=bin_id)

            base = (f"{prefix}{rng_i.integers(100,999):03d}_{kind}_{w_idx:02d}"
                    f"_20260501_010000_{TD}_{LT}_X{x_abs}_Y{y_abs}_B{bin_id}")
            out_path = cls_dir / f"{base}.png"
            img.save(out_path, optimize=False, compress_level=1)
            total_made += 1

        print(f"[gen] {cls}: +{args.per_class}  (total in dir: {len(list(cls_dir.glob('*.png')))})", flush=True)

    elapsed = time.time() - t0
    print(f"\n[OK] {total_made} chips ({len(args.classes)} classes × {args.per_class}) "
          f"in {elapsed:.1f}s ({total_made / max(0.1, elapsed):.1f} img/s)")
    print(f"Output: {out_root}")
    return 0


if __name__ == '__main__':
    sys.exit(main())
