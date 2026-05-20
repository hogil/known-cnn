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

260520 - per-class generation parallelized via ProcessPoolExecutor
(default workers = os.cpu_count()). Worker = chip generation (CPU-bound
numpy + PIL ops). Main thread = sanity check + save + manifest append
(needs shared accepted/rejected counter state).
"""
from __future__ import annotations

import argparse
import csv
import os
import random
import shutil
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from .constants import (COMBO_KEYS, DEFAULT_CLASSIFICATION_CHIPS, SINGLE_KEYS,
                        TRIPLE_COMBO_KEYS)

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


def _min_blend_n(arrs: List[np.ndarray]) -> np.ndarray:
    """N-way pixel-wise min (260508). Generalizes 2-class min-blend to 3+ sources
    for multi-class combo synthesis. Same semantic as _min_blend extended."""
    if len(arrs) < 2:
        raise ValueError(f"need >=2 sources for blend, got {len(arrs)}")
    return np.minimum.reduce(arrs).astype(np.uint8)


def _make_normal_chip(rng: np.random.Generator) -> Image.Image:
    """Palette-aligned Normal chip — per-chip Beta(2, 10) noise probability.

    260507 redesign (palette PNG, RGB sprinkle 폐기):
    - per-chip p_noise ~ Beta(2, 10), mean ~0.17, range ~0.02-0.50
    - per-pixel grade 0 (white) with prob (1-p_noise), else grade 1 (grey) 95% / grade 2 (green dot) 5%
    - ★ palette grade 0/1/2 만 사용 (RGB 자유 색 영구 금지)
    - return PIL Image mode='P' with palette (chip 결함 generator 와 동일 logic)
    """
    from dist_apply import _sample_gen as sg

    p_noise = float(rng.beta(2, 10))   # per-chip random noise probability
    u = rng.random((CHIP_SIZE, CHIP_SIZE))
    is_noise = u < p_noise
    u2 = rng.random((CHIP_SIZE, CHIP_SIZE))
    # noise 안에서 grade 1 (정상 sprinkle) 95%, grade 2 (가끔 dot) 5%
    noise_grade = np.where(u2 < 0.95, 1, 2).astype(np.uint8)
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
    from dist_apply import _sample_gen as sg

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


def _sanity_check(class_key: str, arr: np.ndarray,
                  bases: List[np.ndarray] | None = None) -> Tuple[bool, str]:
    """260508: bases is now a list (was base1, base2). Supports N-way combo (3-class)."""
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
        if not bases or len(bases) < 2:
            return False, "combo_missing_base"
        d_blend = _defect_pixel_ratio(arr)
        d_max = max(_defect_pixel_ratio(b) for b in bases)
        if d_blend < d_max - 0.01:
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


# === Module-level workers for ProcessPoolExecutor (260520) ============
# Each generates ONE raw chip candidate. Main thread does sanity_check +
# save + manifest append (which requires shared accepted/rejected state).
def _worker_make_single(args):
    src_path, _seed = args
    return _load_chip_rgb(Path(src_path))


def _worker_make_combo2(args):
    a_path, b_path, _seed = args
    return _min_blend(_load_chip_rgb(Path(a_path)), _load_chip_rgb(Path(b_path)))


def _worker_make_combo3(args):
    a_path, b_path, c_path, _seed = args
    return _min_blend_n([
        _load_chip_rgb(Path(a_path)),
        _load_chip_rgb(Path(b_path)),
        _load_chip_rgb(Path(c_path)),
    ])


def _worker_make_normal(seed):
    rng = np.random.default_rng(seed)
    return _make_normal_chip(rng)


def _worker_make_invalid(seed):
    rng = np.random.default_rng(seed)
    return _make_invalid_chip(rng)
# =====================================================================


def generate(out_root: Path, classification_chips_root: Path,
             per_defect: int, per_normal: int, per_invalid: int,
             seed: int, source_strength_pct: float = 100.0,
             include_triples: bool = False,
             n_workers: int = None) -> GenStats:
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

    if n_workers is None:
        n_workers = max(1, (os.cpu_count() or 4) - 1)
    print(f"[gen] CPU workers = {n_workers}", flush=True)

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
                gen_method: str, base3_path: str = "") -> bool:
        # Accept either numpy RGB array OR PIL Image (palette PNG).
        # Save the source object as-is (preserves palette mode), but compute
        # sanity & defect_pixel_ratio on the RGB conversion.
        # 260508: optional base3_path for 3-class combo.
        if isinstance(arr_or_img, Image.Image):
            save_obj = arr_or_img
            arr_rgb = np.array(arr_or_img.convert("RGB"))
        else:
            save_obj = arr_or_img
            arr_rgb = arr_or_img
        bases: List[np.ndarray] = []
        for bp in (base1_path, base2_path, base3_path):
            if bp:
                bases.append(_load_chip_rgb(Path(bp)))
        ok, reason = _sanity_check(class_key, arr_rgb, bases if bases else None)
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
            "base3_path": base3_path,
            "gen_method": gen_method,
        })
        accepted[class_key] = idx + 1
        return True

    def _progress(kind: str, class_key: str, made: int, target: int, attempts: int) -> None:
        if target <= 0:
            return
        step = max(1, target // 10)
        if made == 1 or made == target or made % step == 0:
            pct = 100.0 * made / target
            print(f"[gen] {kind} {class_key}: {made}/{target} ({pct:.0f}%) attempts={attempts}",
                  flush=True)

    # Parallel chip-generation helper. Workers produce chip arrays/images;
    # main thread does _record (sanity + save + manifest, needs shared state).
    # Overshoot factor 1.5x to absorb rejection without re-dispatch round-trips.
    def _run_class_parallel(kind, class_key, target, worker_fn, build_arg,
                            base1_of=None, base2_of=None, base3_of=None,
                            gen_method="generated"):
        if target <= 0:
            return
        print(f"[gen] {kind} {class_key}: start target={target}", flush=True)
        overshoot_factor = 1.5 if kind in ("single", "combo2", "combo3") else 1.05
        n_tasks = int(target * overshoot_factor) + 4
        tasks = [build_arg(rng) for _ in range(n_tasks)]
        n_made = 0
        attempts = 0
        with ProcessPoolExecutor(max_workers=n_workers) as exe:
            try:
                chunk = max(1, n_tasks // (n_workers * 4))
                for i, chip in enumerate(exe.map(worker_fn, tasks, chunksize=chunk)):
                    attempts = i + 1
                    base1 = base1_of(tasks[i]) if base1_of else ""
                    base2 = base2_of(tasks[i]) if base2_of else ""
                    base3 = base3_of(tasks[i]) if base3_of else ""
                    if _record(class_key, chip, base1, base2, gen_method,
                               base3_path=base3):
                        n_made += 1
                        _progress(kind, class_key, n_made, target, attempts)
                        if n_made >= target:
                            break
            finally:
                pass
        if n_made < target:
            # Rare: rejection rate too high — fall back to sequential top-up
            print(f"[gen] {kind} {class_key}: top-up sequential "
                  f"{n_made}/{target} attempts={attempts}", flush=True)
            while n_made < target and attempts < target * 4:
                attempts += 1
                arg = build_arg(rng)
                chip = worker_fn(arg)
                base1 = base1_of(arg) if base1_of else ""
                base2 = base2_of(arg) if base2_of else ""
                base3 = base3_of(arg) if base3_of else ""
                if _record(class_key, chip, base1, base2, gen_method,
                           base3_path=base3):
                    n_made += 1
                    _progress(kind, class_key, n_made, target, attempts)

    # 1) single defects (parallel CPU)
    for cls in SINGLE_KEYS:
        def _build_single(r, cls=cls):
            return (str(src_chips[cls][int(r.integers(0, len(src_chips[cls])))]),
                    int(r.integers(0, 2**31 - 1)))
        _run_class_parallel("single", cls, per_defect, _worker_make_single,
                            _build_single,
                            base1_of=lambda t: t[0],
                            gen_method="single_resample")

    # 2) 2-combos (parallel CPU)
    for combo in COMBO_KEYS:
        a, b = combo.split("+")
        def _build_c2(r, a=a, b=b):
            return (str(src_chips[a][int(r.integers(0, len(src_chips[a])))]),
                    str(src_chips[b][int(r.integers(0, len(src_chips[b])))]),
                    int(r.integers(0, 2**31 - 1)))
        _run_class_parallel("combo2", combo, per_defect, _worker_make_combo2,
                            _build_c2,
                            base1_of=lambda t: t[0],
                            base2_of=lambda t: t[1],
                            gen_method="min_blend")

    # 2b) 3-combos (260508): parallel CPU N-way min-blend
    if include_triples:
        for combo in TRIPLE_COMBO_KEYS:
            a, b, c = combo.split("+")
            def _build_c3(r, a=a, b=b, c=c):
                return (str(src_chips[a][int(r.integers(0, len(src_chips[a])))]),
                        str(src_chips[b][int(r.integers(0, len(src_chips[b])))]),
                        str(src_chips[c][int(r.integers(0, len(src_chips[c])))]),
                        int(r.integers(0, 2**31 - 1)))
            _run_class_parallel("combo3", combo, per_defect, _worker_make_combo3,
                                _build_c3,
                                base1_of=lambda t: t[0],
                                base2_of=lambda t: t[1],
                                base3_of=lambda t: t[2],
                                gen_method="min_blend_n3")

    # 3) Normal (parallel CPU synth, no rejection sampling so overshoot ~1.05)
    def _build_norm(r):
        return int(r.integers(0, 2**31 - 1))
    _run_class_parallel("normal", "Normal", per_normal, _worker_make_normal,
                        _build_norm, gen_method="synth_baseline")

    # 4) Invalid (parallel CPU synth)
    def _build_inv(r):
        return int(r.integers(0, 2**31 - 1))
    _run_class_parallel("invalid", "Invalid", per_invalid, _worker_make_invalid,
                        _build_inv, gen_method="synth_invalid_white_border")

    # manifest + previews
    with open(out_root / "manifest.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["chip_path", "class_key", "defect_pixel_ratio",
                                          "base1_path", "base2_path", "base3_path",
                                          "gen_method"])
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
    ap.add_argument("--include-triples", action="store_true",
                    help="260508: also generate 4 3-combo classes (TRIPLE_COMBO_KEYS) → 14 class.")
    ap.add_argument("--workers", type=int, default=None,
                    help="CPU process workers (default os.cpu_count()-1). "
                         "Set 1 to disable multiprocessing.")
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

    n_combo = 10 if args.include_triples else 10
    n_def_classes = len(SINGLE_KEYS) + len(COMBO_KEYS) + (len(TRIPLE_COMBO_KEYS) if args.include_triples else 0)
    print(f"[gen] target: defect={per_defect}/class × {n_def_classes} + Normal={per_normal} + Invalid={per_invalid} "
          f"= {per_defect*n_def_classes + per_normal + per_invalid} chips "
          f"(triples={'on' if args.include_triples else 'off'})")
    stats = generate(out_root, Path(args.classification_chips_root),
                     per_defect=per_defect, per_normal=per_normal, per_invalid=per_invalid,
                     seed=args.seed, source_strength_pct=args.source_strength_pct,
                     include_triples=args.include_triples,
                     n_workers=args.workers)
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
