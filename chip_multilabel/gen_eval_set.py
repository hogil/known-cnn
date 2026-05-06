"""Generate synthetic multi-label chip eval set (11 class).

Layout:
    <out_root>/
        bank_boundary/                         # 4 single-defect classes
        fork/
        scratch/
        scratch_rot/
        bank_boundary+fork/                    # 5 combo classes
        bank_boundary+scratch/
        bank_boundary+scratch_rot/
        fork+scratch/
        fork+scratch_rot/
        Normal/                                # synthesized
        Invalid/                               # synthesized (white + orange border)
        manifest.csv
        _preview/<class_key>.png               # 16-grid preview (4x4)
        _rejected/<reason>/                    # sanity check failures
"""
from __future__ import annotations

import argparse
import csv
import random
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from .constants import (COMBO_KEYS, DEFAULT_CLASSIFICATION_CHIPS, SINGLE_KEYS)

CHIP_SIZE = 200
ORANGE_RGB = (240, 160, 0)
GREY_PALETTE_GRADE_1_RGB = (155, 155, 155)


@dataclass
class GenStats:
    accepted: Dict[str, int]
    rejected: Dict[str, int]


def _load_chip_rgb(path: Path) -> np.ndarray:
    with Image.open(path) as im:
        arr = np.array(im.convert("RGB"))
    if arr.shape != (CHIP_SIZE, CHIP_SIZE, 3):
        arr = np.array(Image.fromarray(arr).resize((CHIP_SIZE, CHIP_SIZE), Image.BILINEAR))
    return arr


def _whiteness(arr: np.ndarray) -> float:
    diff = np.abs(arr.astype(np.int16) - 255).max(axis=-1)
    return float((diff <= 10).mean())


def _defect_pixel_ratio(arr: np.ndarray) -> float:
    """Approx: fraction of pixels not white and not grey-grade-1."""
    diff_white = np.abs(arr.astype(np.int16) - 255).max(axis=-1)
    diff_grey = np.abs(arr.astype(np.int16) - 155).max(axis=-1)
    not_white = diff_white > 10
    not_grey = diff_grey > 10
    return float((not_white & not_grey).mean())


def _min_blend(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Pixel-wise min in RGB. White (255,255,255) is upper bound — any defect color
    is darker in at least one channel, so MIN preserves defects from either chip.
    Where both chips have a defect at the same pixel, MIN gives a mixed darker color
    (acceptable: 'still a defect, not white')."""
    return np.minimum(a, b)


def _make_normal_chip(rng: np.random.Generator) -> np.ndarray:
    """Diversified BASELINE-like Normal chip (260506 patch — 5 noise/variation sources):

    1. Wider grey ratio range 5-30% (was 10-20%)
    2. Per-pixel grey color noise ±15 (was fixed 155,155,155)
    3. White subtle pixel noise ±10 (was perfect 255)
    4. Sprinkle count 0-1.5% with color mix {green, light-green, grey} (was 0-0.5% green only)
    5. Subtle brightness gradient 30% chance, ±15 along x or y (new)

    Sanity guarantee: _whiteness ≥ 0.70 (Normal definition preserved).
    """
    # 3. White base with subtle pixel noise (250-255 range — keep mostly white)
    arr = np.full((CHIP_SIZE, CHIP_SIZE, 3), 255, dtype=np.int16)
    arr += rng.integers(-5, 1, size=arr.shape, dtype=np.int16)
    arr = np.clip(arr, 0, 255)

    flat = arr.reshape(-1, 3)
    n = flat.shape[0]

    # 1. Wider grey ratio (5-22% — keep whiteness ≥ 0.70 sanity)
    n_grey = int(n * float(rng.uniform(0.05, 0.22)))
    if n_grey > 0:
        grey_idx = rng.choice(n, size=n_grey, replace=False)
        # 2. Per-pixel grey color noise ±15
        grey_noise = rng.integers(-15, 16, size=(n_grey, 3), dtype=np.int16)
        flat[grey_idx] = np.clip(np.array(GREY_PALETTE_GRADE_1_RGB, dtype=np.int16) + grey_noise, 0, 255)

    # 4. Sprinkle count 0-1.5% with color mix
    n_minor = int(n * float(rng.uniform(0.0, 0.015)))
    if n_minor > 0:
        minor_idx = rng.choice(n, size=n_minor, replace=False)
        sprinkle_palette = np.array([
            [0, 150, 25],     # original green
            [50, 180, 100],   # light green
            [130, 130, 130],  # darker grey
        ], dtype=np.int16)
        # Each sprinkle picks a random color from palette
        choices = rng.integers(0, len(sprinkle_palette), size=n_minor)
        flat[minor_idx] = sprinkle_palette[choices]

    arr = arr.astype(np.uint8)

    # 5. Subtle brightness gradient (20% chance, ±8 amplitude — keep whiteness)
    if rng.random() < 0.2:
        amplitude = float(rng.uniform(-8, 8))
        if rng.random() < 0.5:
            grad = np.linspace(0, amplitude, CHIP_SIZE).astype(np.int16)  # left→right
            arr = np.clip(arr.astype(np.int16) + grad[None, :, None], 0, 255).astype(np.uint8)
        else:
            grad = np.linspace(0, amplitude, CHIP_SIZE).astype(np.int16)  # top→bottom
            arr = np.clip(arr.astype(np.int16) + grad[:, None, None], 0, 255).astype(np.uint8)

    return arr


def _make_invalid_chip(rng: np.random.Generator) -> np.ndarray:
    """White interior + 2px orange border + optional 'B<num>' bin text near top-left."""
    arr = np.full((CHIP_SIZE, CHIP_SIZE, 3), 255, dtype=np.uint8)
    arr[:2, :, :] = ORANGE_RGB
    arr[-2:, :, :] = ORANGE_RGB
    arr[:, :2, :] = ORANGE_RGB
    arr[:, -2:, :] = ORANGE_RGB
    if bool(rng.integers(0, 2)):
        bin_num = int(rng.integers(200, 300))
        try:
            font = ImageFont.load_default()
        except Exception:
            font = None
        im = Image.fromarray(arr)
        draw = ImageDraw.Draw(im)
        draw.text((10, 10), f"B{bin_num}", fill=(0, 0, 0), font=font)
        arr = np.array(im)
    return arr


def _sanity_check(class_key: str, arr: np.ndarray, base1: np.ndarray | None,
                  base2: np.ndarray | None) -> Tuple[bool, str]:
    if class_key == "Normal":
        if _whiteness(arr) < 0.70:
            return False, "normal_low_white"
        return True, ""
    if class_key == "Invalid":
        if _whiteness(arr) < 0.80:
            return False, "invalid_low_white"
        # check orange border presence
        from .decision_tree import detect_invalid
        is_inv, _ = detect_invalid(arr, white_ratio_thresh=0.80)
        if not is_inv:
            return False, "invalid_no_border"
        return True, ""
    if "+" in class_key:
        if base1 is None or base2 is None:
            return False, "combo_missing_base"
        d_blend = _defect_pixel_ratio(arr)
        d1 = _defect_pixel_ratio(base1)
        d2 = _defect_pixel_ratio(base2)
        if d_blend < max(d1, d2) - 0.01:
            return False, "combo_defect_loss"
        return True, ""
    if class_key in SINGLE_KEYS:
        if _defect_pixel_ratio(arr) < 0.001:
            return False, "single_no_defect"
        return True, ""
    return False, "unknown_class_key"


def _save_chip_rgb(arr: np.ndarray, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(arr).save(path)


def _build_preview(class_key: str, chips_dir: Path, out_path: Path, n: int = 16) -> None:
    files = sorted(chips_dir.glob("*.png"))[:n]
    if not files:
        return
    cell = 200
    cols = int(np.ceil(np.sqrt(n)))
    rows = int(np.ceil(n / cols))
    canvas = np.full((rows * cell, cols * cell, 3), 255, dtype=np.uint8)
    for i, f in enumerate(files):
        r, c = i // cols, i % cols
        with Image.open(f) as im:
            arr = np.array(im.convert("RGB"))
            if arr.shape[:2] != (cell, cell):
                arr = np.array(Image.fromarray(arr).resize((cell, cell), Image.BILINEAR))
        canvas[r * cell:(r + 1) * cell, c * cell:(c + 1) * cell] = arr
    out_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(canvas).save(out_path)


def _defect_strength(arr_rgb: np.ndarray) -> float:
    """Approx defect strength = pixel ratio NOT white AND NOT grey (i.e. grade 2+)."""
    diff_white = np.abs(arr_rgb.astype(np.int16) - 255).max(axis=-1)
    diff_grey = np.abs(arr_rgb.astype(np.int16) - 155).max(axis=-1)
    not_white = diff_white > 10
    not_grey = diff_grey > 10
    return float((not_white & not_grey).mean())


def generate(out_root: Path, classification_chips_root: Path,
             per_defect: int, per_normal: int, per_invalid: int,
             seed: int, source_strength_pct: float = 100.0) -> GenStats:
    """Class-specific N (260506 user directive — defect/normal/invalid different counts).

    per_defect:  applied to each of 10 defect classes (4 single + 6 combo)
    per_normal:  Normal class count (real-env multi-defect prevalence reflection)
    per_invalid: Invalid class count
    source_strength_pct: filter source chips to top-N% by defect_pixel_ratio.
        100.0 = use all chips (default). 50.0 = only top 50% strongest defect chips.

    manifest.csv now includes `defect_pixel_ratio` column for runtime strength filtering
    (memory rule feedback_no_subset_archive_folders.md — single SoT folder + runtime sampling).
    """
    rng = np.random.default_rng(seed)
    out_root.mkdir(parents=True, exist_ok=True)
    accepted: Dict[str, int] = {}
    rejected: Dict[str, int] = {}
    manifest_rows: List[Dict] = []

    src_chips: Dict[str, List[Path]] = {}
    for cls in SINGLE_KEYS:
        d = classification_chips_root / cls
        files = sorted(d.glob("*.png"))
        if not files:
            raise RuntimeError(f"no source chips at {d}")
        if source_strength_pct < 100.0:
            scored = [(f, _defect_strength(_load_chip_rgb(f))) for f in files]
            scored.sort(key=lambda t: -t[1])
            keep_n = max(1, int(len(scored) * source_strength_pct / 100.0))
            files = [f for f, _ in scored[:keep_n]]
            print(f"[gen] {cls}: kept top {keep_n}/{len(scored)} chips (top {source_strength_pct:.0f}%)")
        src_chips[cls] = files

    def _alloc(class_key: str) -> Path:
        d = out_root / class_key
        d.mkdir(parents=True, exist_ok=True)
        return d

    def _record(class_key: str, arr: np.ndarray, base1_path: str, base2_path: str,
                gen_method: str) -> bool:
        ok, reason = _sanity_check(
            class_key, arr,
            _load_chip_rgb(Path(base1_path)) if base1_path else None,
            _load_chip_rgb(Path(base2_path)) if base2_path else None,
        )
        if not ok:
            rej_dir = out_root / "_rejected" / reason
            rej_dir.mkdir(parents=True, exist_ok=True)
            idx = rejected.get(class_key, 0)
            _save_chip_rgb(arr, rej_dir / f"{class_key}_{idx:04d}.png")
            rejected[class_key] = idx + 1
            return False
        idx = accepted.get(class_key, 0)
        cdir = _alloc(class_key)
        chip_name = f"{class_key}_{idx:04d}.png"
        chip_path = cdir / chip_name
        _save_chip_rgb(arr, chip_path)
        manifest_rows.append({
            "chip_path": str(chip_path),
            "class_key": class_key,
            "defect_pixel_ratio": _defect_pixel_ratio(arr),
            "base1_path": base1_path,
            "base2_path": base2_path,
            "gen_method": gen_method,
        })
        accepted[class_key] = idx + 1
        return True

    # 1) single defects: resample with replacement (per_defect)
    for cls in SINGLE_KEYS:
        n_made = 0
        attempts = 0
        while n_made < per_defect and attempts < per_defect * 3:
            attempts += 1
            f = src_chips[cls][int(rng.integers(0, len(src_chips[cls])))]
            arr = _load_chip_rgb(f)
            if _record(cls, arr, str(f), "", "single_resample"):
                n_made += 1

    # 2) combos: min-blend (per_defect)
    for combo in COMBO_KEYS:
        a, b = combo.split("+")
        n_made = 0
        attempts = 0
        while n_made < per_defect and attempts < per_defect * 3:
            attempts += 1
            fa = src_chips[a][int(rng.integers(0, len(src_chips[a])))]
            fb = src_chips[b][int(rng.integers(0, len(src_chips[b])))]
            arr_a = _load_chip_rgb(fa)
            arr_b = _load_chip_rgb(fb)
            blended = _min_blend(arr_a, arr_b)
            if _record(combo, blended, str(fa), str(fb), "min_blend"):
                n_made += 1

    # 3) normal (per_normal — typically larger to reflect real-env Normal prevalence)
    n_made = 0
    while n_made < per_normal:
        arr = _make_normal_chip(rng)
        if _record("Normal", arr, "", "", "synth_baseline"):
            n_made += 1

    # 4) invalid (per_invalid)
    n_made = 0
    while n_made < per_invalid:
        arr = _make_invalid_chip(rng)
        if _record("Invalid", arr, "", "", "synth_invalid_white_border"):
            n_made += 1

    # manifest + previews
    with open(out_root / "manifest.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["chip_path", "class_key", "defect_pixel_ratio",
                                          "base1_path", "base2_path", "gen_method"])
        w.writeheader()
        w.writerows(manifest_rows)

    preview_dir = out_root / "_preview"
    for class_key in list(accepted.keys()):
        _build_preview(class_key, out_root / class_key, preview_dir / f"{class_key}.png")

    return GenStats(accepted=accepted, rejected=rejected)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-root", required=True)
    # 260506: class-specific N (memory rule feedback_class_size_random.md — random 50~200+, no uniform 200)
    ap.add_argument("--per-defect", type=int, default=50,
                    help="N per defect class (4 single + 6 combo). default 50.")
    ap.add_argument("--per-normal", type=int, default=200,
                    help="N for Normal class (real-env prevalence reflection). default 200.")
    ap.add_argument("--per-invalid", type=int, default=50,
                    help="N for Invalid class. default 50.")
    # Backward compat: --per-class overrides ALL three when given
    ap.add_argument("--per-class", type=int, default=None,
                    help="DEPRECATED: if set, uniform N for all 12 classes (overrides --per-defect/normal/invalid).")
    ap.add_argument("--classification-chips-root", default=DEFAULT_CLASSIFICATION_CHIPS)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--source-strength-pct", type=float, default=100.0,
                    help="filter source chips to top-N%% by defect strength (100=all, 50=top half)")
    ap.add_argument("--clear", action="store_true",
                    help="DELETE existing out_root/ before generating (DANGEROUS — disabled by default)")
    args = ap.parse_args()

    if args.per_class is not None:
        print(f"[WARN] --per-class={args.per_class} overrides per-defect/normal/invalid (legacy uniform mode)")
        per_defect = per_normal = per_invalid = args.per_class
    else:
        per_defect, per_normal, per_invalid = args.per_defect, args.per_normal, args.per_invalid

    out_root = Path(args.out_root)
    if args.clear and out_root.exists():
        print(f"[WARN] removing existing {out_root}")
        shutil.rmtree(out_root)

    print(f"[gen] target: defect={per_defect}/class × 10 + Normal={per_normal} + Invalid={per_invalid} "
          f"= {per_defect*10 + per_normal + per_invalid} chips")
    stats = generate(out_root, Path(args.classification_chips_root),
                     per_defect=per_defect, per_normal=per_normal, per_invalid=per_invalid,
                     seed=args.seed, source_strength_pct=args.source_strength_pct)
    total_acc = sum(stats.accepted.values())
    total_rej = sum(stats.rejected.values())
    print(f"\n[gen] accepted total: {total_acc}")
    for k, v in sorted(stats.accepted.items()):
        print(f"  {k}: {v}")
    if total_rej > 0:
        print(f"\n[gen] rejected total: {total_rej}")
        for k, v in sorted(stats.rejected.items()):
            print(f"  {k}: {v}")
    print(f"\n[gen] out: {out_root}")


if __name__ == "__main__":
    main()
