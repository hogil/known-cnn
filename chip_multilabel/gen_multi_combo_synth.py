"""Multi-class chip combo bulk synth (260508).

Master folder D:/project/data/wm-811k/chip_multilabel_synth/ 에 10 combo
× N chip pre-baked. 학습/평가 양쪽에서 sampling 가능. defect-only — Normal/Invalid 제외.

10 combo:
- 6 × 2-class (COMBO_KEYS)
- 4 × 3-class (TRIPLE_COMBO_KEYS)

각 combo:
- N source chip (top-source-strength-pct%) random pick × N times
- N-way pixel min-blend (np.minimum.reduce)
- sanity: defect_pixel_ratio(blended) >= max(d_i) - 0.01

Layout:
    <out_root>/
        manifest.csv
        bank_boundary+fork/                  N chip
        ...
        fork+scratch+scratch_rot/            N chip
        _preview/<combo>.png                 16-grid preview
        _rejected/<reason>/                  sanity 위반
"""
from __future__ import annotations

import argparse
import csv
import shutil
from pathlib import Path
from typing import Dict, List

import numpy as np

from .constants import (COMBO_KEYS, DEFAULT_CLASSIFICATION_CHIPS, SINGLE_KEYS,
                        TRIPLE_COMBO_KEYS)
from .gen_eval_set import (GenStats, _build_preview, _defect_pixel_ratio,
                           _defect_strength, _load_chip_rgb, _min_blend,
                           _min_blend_n, _sanity_check, _save_chip_rgb)


def generate(out_root: Path, classification_chips_root: Path,
             per_combo: int, seed: int,
             source_strength_pct: float = 50.0,
             include_triples: bool = True) -> GenStats:
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

    def _record(class_key: str, arr_rgb: np.ndarray, base_paths: List[str],
                gen_method: str) -> bool:
        bases = [_load_chip_rgb(Path(p)) for p in base_paths]
        ok, reason = _sanity_check(class_key, arr_rgb, bases)
        if not ok:
            rej_dir = out_root / "_rejected" / reason
            rej_dir.mkdir(parents=True, exist_ok=True)
            idx = rejected.get(class_key, 0)
            _save_chip_rgb(arr_rgb, rej_dir / f"{class_key}_{idx:04d}.png")
            rejected[class_key] = idx + 1
            return False
        idx = accepted.get(class_key, 0)
        cdir = out_root / class_key
        cdir.mkdir(parents=True, exist_ok=True)
        chip_name = f"{class_key}_{idx:04d}.png"
        chip_path = cdir / chip_name
        _save_chip_rgb(arr_rgb, chip_path)
        row = {
            "chip_path": str(chip_path),
            "class_key": class_key,
            "defect_pixel_ratio": _defect_pixel_ratio(arr_rgb),
            "base1_path": base_paths[0] if len(base_paths) > 0 else "",
            "base2_path": base_paths[1] if len(base_paths) > 1 else "",
            "base3_path": base_paths[2] if len(base_paths) > 2 else "",
            "gen_method": gen_method,
        }
        manifest_rows.append(row)
        accepted[class_key] = idx + 1
        return True

    # 1) 2-combos: min-blend
    for combo in COMBO_KEYS:
        a, b = combo.split("+")
        n_made = 0
        attempts = 0
        while n_made < per_combo and attempts < per_combo * 5:
            attempts += 1
            fa = src_chips[a][int(rng.integers(0, len(src_chips[a])))]
            fb = src_chips[b][int(rng.integers(0, len(src_chips[b])))]
            arr_a = _load_chip_rgb(fa)
            arr_b = _load_chip_rgb(fb)
            blended = _min_blend(arr_a, arr_b)
            if _record(combo, blended, [str(fa), str(fb)], "min_blend"):
                n_made += 1

    # 2) 3-combos: N-way min-blend (260508)
    if include_triples:
        for combo in TRIPLE_COMBO_KEYS:
            a, b, c = combo.split("+")
            n_made = 0
            attempts = 0
            while n_made < per_combo and attempts < per_combo * 5:
                attempts += 1
                fa = src_chips[a][int(rng.integers(0, len(src_chips[a])))]
                fb = src_chips[b][int(rng.integers(0, len(src_chips[b])))]
                fc = src_chips[c][int(rng.integers(0, len(src_chips[c])))]
                arr_a = _load_chip_rgb(fa)
                arr_b = _load_chip_rgb(fb)
                arr_c = _load_chip_rgb(fc)
                blended = _min_blend_n([arr_a, arr_b, arr_c])
                if _record(combo, blended, [str(fa), str(fb), str(fc)], "min_blend_n3"):
                    n_made += 1

    # manifest + preview
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
    ap.add_argument("--out-root", required=True,
                    help="e.g. D:/project/data/wm-811k/chip_multilabel_synth")
    ap.add_argument("--per-combo", type=int, default=100,
                    help="N chip per combo class. default 100.")
    ap.add_argument("--classification-chips-root", default=DEFAULT_CLASSIFICATION_CHIPS)
    ap.add_argument("--source-strength-pct", type=float, default=50.0,
                    help="Filter source chips to top-N%% by defect strength. "
                         "default 50 (combo 합성은 strong defect source 권장).")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--include-triples", action="store_true", default=True,
                    help="3-combo (TRIPLE_COMBO_KEYS) 포함 (default on).")
    ap.add_argument("--no-triples", dest="include_triples", action="store_false",
                    help="2-combo only (6 class).")
    ap.add_argument("--clear", action="store_true",
                    help="DELETE existing out_root/ before generating.")
    args = ap.parse_args()

    out_root = Path(args.out_root)
    if args.clear and out_root.exists():
        print(f"[WARN] removing existing {out_root}")
        shutil.rmtree(out_root)

    n_combo_classes = len(COMBO_KEYS) + (len(TRIPLE_COMBO_KEYS) if args.include_triples else 0)
    print(f"[gen] target: {args.per_combo}/combo × {n_combo_classes} = "
          f"{args.per_combo * n_combo_classes} chips "
          f"(triples={'on' if args.include_triples else 'off'}, "
          f"strength={args.source_strength_pct:.0f}%)")
    stats = generate(out_root, Path(args.classification_chips_root),
                     per_combo=args.per_combo,
                     seed=args.seed,
                     source_strength_pct=args.source_strength_pct,
                     include_triples=args.include_triples)
    total_acc = sum(stats.accepted.values())
    total_rej = sum(stats.rejected.values())
    print(f"\n[gen] accepted: {total_acc}")
    for k, v in sorted(stats.accepted.items()):
        print(f"  {k}: {v}")
    if total_rej > 0:
        print(f"\n[gen] rejected: {total_rej}")
        for k, v in sorted(stats.rejected.items()):
            print(f"  {k}: {v}")
    print(f"\n[gen] out: {out_root}")
    print(f"[OUT] {out_root}")


if __name__ == "__main__":
    main()
