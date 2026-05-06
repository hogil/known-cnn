"""Generate paired chips via min-blend of two source class chips.

Used for B4 (260507) — fork+scratch_rot supervised paired training.
Output folder name format: '<a>+<b>' (alphabetic order). The trainer's
_collect_samples_with_multihot() detects '+' in folder name and emits
multi-hot label [a=1, b=1].

Example:
    python -m chip_multilabel.gen_pair_chips \
        --src-a fork --src-b scratch_rot --n 200 --seed 42

Writes to:
    D:/project/data/wm-811k/classification_chips/fork+scratch_rot/*.png
"""
from __future__ import annotations

import argparse
import random
from pathlib import Path

import numpy as np
from PIL import Image


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", default="D:/project/data/wm-811k/classification_chips")
    ap.add_argument("--src-a", required=True,
                    help="source class A folder name (e.g. 'fork')")
    ap.add_argument("--src-b", required=True,
                    help="source class B folder name (e.g. 'scratch_rot')")
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    root = Path(args.data_root)
    src_a = root / args.src_a
    src_b = root / args.src_b
    assert src_a.exists(), f"src-a missing: {src_a}"
    assert src_b.exists(), f"src-b missing: {src_b}"

    # Canonical alphabetic order in folder name (matches sort_label_pair convention)
    lo, hi = sorted([args.src_a, args.src_b])
    out_name = f"{lo}+{hi}"
    out = root / out_name
    out.mkdir(parents=True, exist_ok=True)

    # Clear existing pngs (idempotent regen)
    for f in out.glob("*.png"):
        f.unlink()

    rng = random.Random(args.seed)
    files_a = sorted(src_a.glob("*.png"))
    files_b = sorted(src_b.glob("*.png"))
    assert files_a and files_b, f"empty source: a={len(files_a)} b={len(files_b)}"

    for i in range(args.n):
        a_img = np.array(Image.open(rng.choice(files_a)).convert("RGB"))
        b_img = np.array(Image.open(rng.choice(files_b)).convert("RGB"))
        # Resize b to a if shapes differ (defensive — same source pipeline so usually identical)
        if b_img.shape != a_img.shape:
            b_img = np.array(Image.fromarray(b_img).resize(
                (a_img.shape[1], a_img.shape[0]), Image.BILINEAR))
        blend = np.minimum(a_img, b_img)
        Image.fromarray(blend).save(out / f"{lo}_{hi}_{i:04d}.png")

    print(f"[gen-pair] wrote {args.n} chips to {out}")


if __name__ == "__main__":
    main()
