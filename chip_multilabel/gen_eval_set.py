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


def _pink_noise_field(rng: np.random.Generator, size: int, exponent: float = 1.5) -> np.ndarray:
    """Generate 2D pink (1/f^β) noise field via FFT.

    Natural sensor-like spatial variation (no grid artifacts).
    - β=1.0: pink noise (mild correlation)
    - β=1.5: balanced (default — looks like real sensor drift)
    - β=2.0: red/Brownian noise (very smooth, large blobs)
    """
    white = rng.standard_normal((size, size))
    freqs = np.fft.fftfreq(size)
    fx, fy = np.meshgrid(freqs, freqs)
    freq_mag = np.sqrt(fx ** 2 + fy ** 2)
    freq_mag[0, 0] = 1.0  # avoid div by 0
    spectrum_filter = 1.0 / (freq_mag ** exponent)
    spectrum_filter[0, 0] = 0.0  # remove DC
    filtered = np.real(np.fft.ifft2(np.fft.fft2(white) * spectrum_filter))
    # normalize to [0, 1]
    f_min, f_max = float(filtered.min()), float(filtered.max())
    if f_max - f_min < 1e-9:
        return np.full_like(filtered, 0.5)
    return (filtered - f_min) / (f_max - f_min)


def _make_normal_chip(rng: np.random.Generator) -> Image.Image:
    """Palette-aligned Normal chip — natural sensor-like spatial noise.

    260507 redesign (palette PNG, RGB sprinkle 폐기):
    - 자연스러운 sensor-like 변동: 1/f^β pink noise field (FFT 기반).
      lithography 변동 / sensor drift / hot spots 같은 physical 현상 모방.
      grid 아티팩트 없음.
    - per-chip overall noise level (Beta(2, 8) scale) + spatial pink-noise field
      (β=1.5) modulation.
    - per-pixel grade 0 (white) with prob (1 - p_local), else grade 1/2.
    - grade 2 (green) ratio 도 별개 pink-noise field 로 spatial 변동.
    - ★ palette grade 0/1/2 만 사용 (RGB 자유 색 영구 금지).
    """
    import sys
    from pathlib import Path as _P
    sys.path.insert(0, str(_P(__file__).resolve().parents[1] / "dist_apply"))
    import _sample_gen as sg

    # Per-chip overall scale — some chips cleaner, some hazier
    overall_scale = float(rng.beta(2, 8))   # mean ~0.20, range ~0.02-0.6

    # Natural spatial p_noise field (1/f^1.5 pink noise — sensor-like)
    p_field_raw = _pink_noise_field(rng, CHIP_SIZE, exponent=1.5)
    # scale: pixel-wise probability = overall_scale * (0.4 + 1.2 * p_field_raw)
    # → range ~0.4*scale (cleaner regions) ~ 1.6*scale (noisier)
    p_field = np.clip(overall_scale * (0.4 + 1.2 * p_field_raw), 0.0, 0.85)

    # Natural grade-2 (green) probability field — different pink noise instance
    g2_field_raw = _pink_noise_field(rng, CHIP_SIZE, exponent=2.0)  # smoother
    g2_max = float(rng.uniform(0.02, 0.15))   # per-chip max grade-2 ratio
    g2_field = np.clip(g2_max * (0.2 + g2_field_raw), 0.0, 0.30)

    # Per-pixel sampling
    u = rng.random((CHIP_SIZE, CHIP_SIZE))
    is_noise = u < p_field

    u2 = rng.random((CHIP_SIZE, CHIP_SIZE))
    is_grade2 = u2 < g2_field
    noise_grade = np.where(is_grade2, 2, 1).astype(np.uint8)
    grades = np.where(is_noise, noise_grade, 0).astype(np.uint8)

    img = Image.frombytes('P', (CHIP_SIZE, CHIP_SIZE), grades.tobytes())
    img.putpalette(sg.PALETTE)
    return img


def _make_invalid_chip(rng: np.random.Generator) -> Image.Image:
    """Palette-aligned Invalid chip — grade 0 white interior + 2px orange (palette idx 11) + black text (idx 9).

    260507 redesign (palette PNG):
    - 2px orange border = palette index 11 (border_inv, RGB 255,153,0)
    - Text fill = palette index 9 (text, near-black)
    - White interior = grade 0 (palette index 0)
    - return PIL Image mode='P'
    """
    import sys
    from pathlib import Path as _P
    sys.path.insert(0, str(_P(__file__).resolve().parents[1] / "dist_apply"))
    import _sample_gen as sg

    grades = np.zeros((CHIP_SIZE, CHIP_SIZE), dtype=np.uint8)  # all white (grade 0)
    BORDER_IDX = sg.KEY_TO_INDEX.get('border_inv', 11)
    TEXT_IDX = sg.KEY_TO_INDEX.get('text', 9)
    grades[:2, :] = BORDER_IDX
    grades[-2:, :] = BORDER_IDX
    grades[:, :2] = BORDER_IDX
    grades[:, -2:] = BORDER_IDX

    img = Image.frombytes('P', (CHIP_SIZE, CHIP_SIZE), grades.tobytes())
    img.putpalette(sg.PALETTE)

    # Centered large text (palette index for fill — PIL converts to mode='P' index)
    bin_num = int(rng.integers(200, 300))
    font = None
    for fp in ["C:/Windows/Fonts/arial.ttf", "C:/Windows/Fonts/calibri.ttf",
               "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"]:
        try:
            from os.path import exists
            if exists(fp):
                font = ImageFont.truetype(fp, 64)
                break
        except Exception:
            pass
    if font is None:
        font = ImageFont.load_default()
    draw = ImageDraw.Draw(img)
    text = f"B{bin_num}"
    try:
        bbox = draw.textbbox((0, 0), text, font=font)
        tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
        tx = CHIP_SIZE // 2 - tw // 2 - bbox[0]
        ty = CHIP_SIZE // 2 - th // 2 - bbox[1]
    except Exception:
        tw, th = 120, 50
        tx, ty = CHIP_SIZE // 2 - tw // 2, CHIP_SIZE // 2 - th // 2
    draw.text((tx, ty), text, fill=int(TEXT_IDX), font=font)
    return img


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


def _save_chip_rgb(arr, path: Path) -> None:
    """Accept either numpy.ndarray (RGB) or PIL.Image.Image (any mode).

    260507: palette PNG (mode='P') 통과 위해 PIL Image 도 직접 저장.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(arr, np.ndarray):
        Image.fromarray(arr).save(path)
    else:
        # PIL.Image.Image — palette PNG 그대로 저장
        arr.save(path)


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

    def _record(class_key: str, arr_or_img, base1_path: str, base2_path: str,
                gen_method: str) -> bool:
        # Accept either numpy RGB array OR PIL Image (palette PNG).
        # Save the source object as-is (preserves palette mode), but compute
        # sanity & defect_pixel_ratio on the RGB conversion.
        if isinstance(arr_or_img, Image.Image):
            save_obj = arr_or_img
            arr_rgb = np.array(arr_or_img.convert("RGB"))
        else:
            save_obj = arr_or_img
            arr_rgb = arr_or_img
        ok, reason = _sanity_check(
            class_key, arr_rgb,
            _load_chip_rgb(Path(base1_path)) if base1_path else None,
            _load_chip_rgb(Path(base2_path)) if base2_path else None,
        )
        if not ok:
            rej_dir = out_root / "_rejected" / reason
            rej_dir.mkdir(parents=True, exist_ok=True)
            idx = rejected.get(class_key, 0)
            _save_chip_rgb(save_obj, rej_dir / f"{class_key}_{idx:04d}.png")
            rejected[class_key] = idx + 1
            return False
        idx = accepted.get(class_key, 0)
        cdir = _alloc(class_key)
        chip_name = f"{class_key}_{idx:04d}.png"
        chip_path = cdir / chip_name
        _save_chip_rgb(save_obj, chip_path)
        manifest_rows.append({
            "chip_path": str(chip_path),
            "class_key": class_key,
            "defect_pixel_ratio": _defect_pixel_ratio(arr_rgb),
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
