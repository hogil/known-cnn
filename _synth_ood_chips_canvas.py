"""Generate OOD chip via in-memory wafer-canvas crop (no temp wafer file).

Fork of dist_apply._sample_canvas_gen.render_canvas — runs full wafer canvas
synthesis in memory, then extracts defect chips (b >= 200) directly without
writing any wafer PNG/JSON to disk.

Run:  python _synth_ood_chips_canvas.py --target 2000
"""
import argparse, os, sys, time
from pathlib import Path
import numpy as np
from PIL import Image

THIS = Path(__file__).resolve().parent
sys.path.insert(0, str(THIS))

from dist_apply._sample_canvas_gen import (
    SIZE, GRID, CHIP, PALETTE,
    _canvas_wafer_inside_mask, pick_baseline_tier, pick_intensity_tier,
    CUM_BASELINE_TIERS, INTENSITY_ALPHA_SCALE, ALPHA_FN, _build_peak_cum,
    DEFECT_BIN_POOL, DEFECT_BIN_WEIGHTS, KEY_TO_INDEX, BIN_TO_BORDER_IDX,
    IDX_BG,
)

OOD_CLASSES = ("CenterDonut", "CrossScratch", "DiagonalSmear", "Starburst")
OUT_ROOTS = [
    Path("E:/data/images/chip_multilabel_v15direct"),         # master
    Path("E:/data/images/chip_multilabel_v15direct_n2000"),   # eval set
]
PALE_BLUE_IDX = 8


def render_canvas_in_memory(class_name, seed):
    """Run full wafer-canvas synthesis in memory.
    Returns (canvas np.uint8 array SIZE x SIZE, chip_meta dict)."""
    rng = np.random.default_rng(seed)
    inside = _canvas_wafer_inside_mask()
    inside_pix = np.repeat(np.repeat(inside, CHIP, axis=0), CHIP, axis=1)
    baseline_tier = pick_baseline_tier(rng)
    intensity_tier = pick_intensity_tier(rng)
    canvas_tau = float(rng.choice([0.20, 0.30, 0.40], p=[0.25, 0.50, 0.25]))
    cum_base_use = CUM_BASELINE_TIERS[baseline_tier].astype(np.float32)
    alpha = ALPHA_FN[class_name](rng) * INTENSITY_ALPHA_SCALE[intensity_tier]

    def _bilinear_field(rng, h, w, lo, hi):
        coarse = rng.uniform(lo, hi, size=(h, w)).astype(np.float32)
        yi = np.linspace(0, h - 1, SIZE, dtype=np.float32)
        xi = np.linspace(0, w - 1, SIZE, dtype=np.float32)
        yi0 = np.clip(np.floor(yi).astype(np.int32), 0, h - 2)
        xi0 = np.clip(np.floor(xi).astype(np.int32), 0, w - 2)
        fy = yi - yi0; fx = xi - xi0
        return (
            (1 - fy)[:, None] * ((1 - fx)[None, :] * coarse[yi0][:, xi0] +
                                 fx[None, :]       * coarse[yi0][:, xi0 + 1]) +
            fy[:, None]       * ((1 - fx)[None, :] * coarse[yi0 + 1][:, xi0] +
                                 fx[None, :]       * coarse[yi0 + 1][:, xi0 + 1])
        ).astype(np.float32)

    field_medium = _bilinear_field(rng, 32, 32, 0.50, 1.50)
    field_high = _bilinear_field(rng, 128, 128, 0.65, 1.40)
    alpha *= field_medium * field_high
    fine_noise = rng.normal(0, 0.04, (SIZE, SIZE)).astype(np.float32)
    alpha = np.clip(alpha + fine_noise, 0, 1).astype(np.float32)
    del field_medium, field_high, fine_noise
    alpha *= inside_pix.astype(np.float32)

    u = rng.random((SIZE, SIZE))
    canvas = np.searchsorted(cum_base_use, u).astype(np.uint8); del u
    canvas[~inside_pix] = IDX_BG

    cum_peak = _build_peak_cum(class_name)
    for y0 in range(0, SIZE, 800):
        y1 = min(y0 + 800, SIZE)
        a_c = alpha[y0:y1, :]
        if a_c.max() < 0.01: continue
        cum_mixed = ((1 - a_c)[..., None] * cum_base_use[None, None, :] +
                     a_c[..., None] * cum_peak[None, None, :])
        uu = rng.random((y1-y0, SIZE)).astype(np.float32)
        grades = (uu[..., None] < cum_mixed).argmax(axis=-1).astype(np.uint8)
        mask = a_c > 0.01
        canvas[y0:y1][mask] = grades[mask]

    chip_meta = {}
    kind = '00P' if rng.random() < 0.5 else '00C'
    bin_pool = DEFECT_BIN_POOL[kind]
    for gy in range(GRID):
        for gx in range(GRID):
            if not inside[gy, gx]: continue
            y0, x0 = gy*CHIP, gx*CHIP
            chip_alpha = alpha[y0:y0+CHIP, x0:x0+CHIP]
            chip_alpha_mean = float(chip_alpha.mean())
            chip_alpha_max = float(chip_alpha.max())
            if chip_alpha_max < canvas_tau: continue
            score = max(chip_alpha_mean * 3.0, (chip_alpha_max - 0.5) * 1.5)
            p_def = min(score, 1.0)
            if rng.random() > p_def: continue
            b = int(rng.choice(bin_pool, p=DEFECT_BIN_WEIGHTS))
            chip_meta[(gy, gx)] = {'kind': 'defect', 'obj': None, 'bin': b, 'inside': True}
    del alpha

    IDX_BORDER_NORMAL = KEY_TO_INDEX["border"]
    for gy in range(GRID):
        for gx in range(GRID):
            if not inside[gy, gx]: continue
            y0, x0 = gy*CHIP, gx*CHIP; y1, x1 = y0+CHIP, x0+CHIP
            meta = chip_meta.get((gy, gx))
            if meta is None:
                bw, c = 1, IDX_BORDER_NORMAL
            else:
                bw, c = 2, BIN_TO_BORDER_IDX.get(meta['bin'], KEY_TO_INDEX["border_etc"])
            canvas[y0:y0+bw, x0:x1] = c; canvas[y1-bw:y1, x0:x1] = c
            canvas[y0:y1, x0:x0+bw] = c; canvas[y0:y1, x1-bw:x1] = c

    return canvas, chip_meta


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", type=int, default=2000)
    ap.add_argument("--classes", nargs="+", default=OOD_CLASSES)
    ap.add_argument("--max-wafer", type=int, default=1000)
    args = ap.parse_args()
    print(f"target={args.target} chip/cls, classes={args.classes}, in-memory only (no temp wafer files)")
    t0 = time.time()
    for cls in args.classes:
        out_dirs = [r / cls for r in OUT_ROOTS]
        for d in out_dirs: d.mkdir(parents=True, exist_ok=True)
        # per-out-dir independent idx (master may be at 1804 while n2000 at 200)
        start_idxs = [len(list(d.glob("*.png"))) for d in out_dirs]
        max_needed = max(args.target - s for s in start_idxs)
        print(f"\n=== {cls}: existing={start_idxs}, need_max={max_needed} ===")
        if max_needed <= 0:
            print(f"  all out_dirs at target, skip")
            continue
        cls_seed = abs(hash(cls)) % (2**31)
        wafer_i = 0
        while any(s < args.target for s in start_idxs) and wafer_i < args.max_wafer:
            seed = cls_seed + wafer_i * 1000
            try:
                canvas, chip_meta = render_canvas_in_memory(cls, seed)
                added = 0
                for (gy, gx), meta in chip_meta.items():
                    if all(s >= args.target for s in start_idxs): break
                    if int(meta['bin']) < 200: continue
                    y0, x0 = gy*CHIP, gx*CHIP
                    arr = canvas[y0:y0+CHIP, x0:x0+CHIP]
                    if arr.std() < 0.01 and int(arr.flat[0]) == PALE_BLUE_IDX:
                        continue
                    chip_img = Image.frombytes('P', (CHIP, CHIP), arr.tobytes())
                    chip_img.putpalette(PALETTE)
                    # save into every out_dir that still needs more
                    for i, (d, s) in enumerate(zip(out_dirs, start_idxs)):
                        if s >= args.target: continue
                        chip_img.save(d / f"{cls}_{s:04d}.png", optimize=False, compress_level=1)
                        start_idxs[i] = s + 1
                    added += 1
                if wafer_i % 10 == 0 or all(s >= args.target for s in start_idxs):
                    print(f"  wafer {wafer_i+1:>3d}: +{added:>2d}, totals {start_idxs}", flush=True)
            except Exception as e:
                print(f"  wafer {wafer_i+1}: ERROR {type(e).__name__}: {e}", flush=True)
            wafer_i += 1
        print(f"  [{cls}] DONE  totals {start_idxs}, {wafer_i} wafer, {time.time()-t0:.1f}s elapsed")
    print(f"\n[OK] elapsed={time.time()-t0:.1f}s - no temp wafer files created")


if __name__ == "__main__":
    sys.exit(main() or 0)
