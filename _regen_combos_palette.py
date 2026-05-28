"""Regenerate ONLY the 2-combo eval folders as palette PNG (mode 'P').

The combo render was the sole RGB-saving path (per-channel min fabricated off-palette
colors). chip_synth.render_combo_chip now blends in grade-index space -> mode 'P'.
Combo appearance uses COMBO_BG_RANGE (fixed), independent of SINGLE_BG, so combos are
identical across the clean/h100/lownoise eval sets. single/OOD/Normal/Invalid are
already palette-P and are left untouched.
"""
import argparse
from pathlib import Path

import numpy as np
from PIL import Image

import chip_synth as synth

COUNTS = {"sota_clean": 400, "sota_h100": 2000, "sota_lownoise": 400}


def regen(ds: str, n: int) -> None:
    root = Path(f"data/images/{ds}/eval_set")
    if not root.exists():
        print(f"  [skip] {root} missing")
        return
    for key in synth.COMBO_2:
        d = root / key
        if not d.exists():
            continue
        old = list(d.glob("*.png"))
        for p in old:
            p.unlink()
        master = np.random.default_rng(abs(hash((ds, key))) % (2 ** 31))
        made = 0
        for i in range(n):
            seed = int(master.integers(0, 2 ** 31 - 1))
            img = synth.render_combo_chip(key, np.random.default_rng(seed))
            assert img.mode == "P", f"{key} not palette!"
            img.save(d / f"{key}_{i:05d}_s{seed}.png", optimize=False, compress_level=1)
            made += 1
        # verify a sample
        sample = next(d.glob("*.png"))
        m = Image.open(sample).mode
        print(f"  {ds}/{key:24s} regen {made} (deleted {len(old)})  sample_mode={m}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", default="sota_clean,sota_h100,sota_lownoise")
    args = ap.parse_args()
    for ds in args.datasets.split(","):
        ds = ds.strip()
        print(f"=== {ds} (combo n={COUNTS.get(ds, 400)}) ===")
        regen(ds, COUNTS.get(ds, 400))
    print("REGEN DONE")


if __name__ == "__main__":
    main()
