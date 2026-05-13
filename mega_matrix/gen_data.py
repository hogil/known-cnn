"""
Mega matrix data generation.

Generates train + eval subsets for the matrix sweep.

Train: deterministic subsets {50, 100, 200} from classification_chips/
       (auto-synth more if master pool < 200 per class)

Eval:  3 sizes {200, 2000, 20000} per defect class. Uses gen_eval_set.py
       to synthesize each (full single + 2-combo + 3-combo + Normal + Invalid + OOD).
"""
import subprocess
import sys
import os
import glob
from pathlib import Path

PROJ_ROOT = Path(__file__).parent.parent
OUT_BASE = PROJ_ROOT / "outputs" / "_mega_matrix"
OUT_BASE.mkdir(parents=True, exist_ok=True)
DATA_ROOT = Path(os.environ.get("WM811K_ROOT", str(PROJ_ROOT / "data" / "wm-811k"))).resolve()

TRAIN_SIZES = [50, 100, 200]
EVAL_SIZES = [200, 2000, 20000]
TRAIN_CLASSES = ["bank_boundary", "fork", "scratch", "scratch_rot"]
MASTER_TRAIN = Path(os.environ.get("CLASSIFICATION_CHIPS_ROOT", str(DATA_ROOT / "classification_chips"))).resolve()


def log(msg):
    print(f"[gen_data] {msg}", flush=True)


def ensure_master_pool(need=200):
    """Make sure classification_chips/ has ≥ need chips per class."""
    counts = []
    for c in TRAIN_CLASSES:
        d = MASTER_TRAIN / c
        n = len(list(d.glob("*.png"))) if d.exists() else 0
        counts.append(n)
        log(f"master {c}: {n} chips")
    min_avail = min(counts) if counts else 0
    if min_avail < need:
        log(f"synthesizing single chips (min={min_avail} < need={need})")
        cmd = [
            sys.executable, "-u", "-m", "dist_apply._synth_chips_only",
            "--per-class", str(need),
            "--out", str(MASTER_TRAIN),
        ]
        subprocess.run(cmd, check=False, cwd=str(PROJ_ROOT))
    else:
        log(f"master pool OK (min={min_avail} ≥ need={need})")


def make_train_subsets():
    for tn in TRAIN_SIZES:
        out_dir = OUT_BASE / f"train_n{tn}"
        if out_dir.exists() and all((out_dir / c).exists() for c in TRAIN_CLASSES):
            existing = len(list(out_dir.glob("*/*.png")))
            log(f"train_n{tn} exists ({existing} files), skip")
            continue
        out_dir.mkdir(parents=True, exist_ok=True)
        for c in TRAIN_CLASSES:
            (out_dir / c).mkdir(exist_ok=True)
            files = sorted((MASTER_TRAIN / c).glob("*.png"))
            for f in files[:tn]:
                tgt = out_dir / c / f.name
                if not tgt.exists():
                    try:
                        tgt.symlink_to(f)
                    except (OSError, NotImplementedError):
                        # Windows fallback — copy
                        import shutil
                        shutil.copy2(f, tgt)
        log(f"train_n{tn} ready ({tn}/class × 4 = {tn*4} chips)")


def make_eval_sets():
    """Generate eval sets using gen_eval_set.py."""
    for en in EVAL_SIZES:
        master_eval_dir = DATA_ROOT / f"chip_multilabel_mega_eval_n{en}"
        local_eval_dir = OUT_BASE / f"eval_n{en}"

        if local_eval_dir.exists() and len(list(local_eval_dir.glob("*/*.png"))) > 0:
            n = len(list(local_eval_dir.glob("*/*.png")))
            log(f"eval_n{en} local exists ({n} chips), skip")
            continue

        if not master_eval_dir.exists() or len(list(master_eval_dir.glob("*/*.png"))) == 0:
            log(f"generating eval_n{en} via gen_eval_set...")
            cmd = [
                sys.executable, "-u", "-X", "utf8", "-m", "chip_multilabel.gen_eval_set",
                "--out-root", str(master_eval_dir),
                "--per-defect", str(en),
                "--per-normal", str(en),
                "--per-invalid", str(max(en // 4, 10)),
                "--include-triples",
                "--classification-chips-root", str(MASTER_TRAIN),
                "--seed", "42",
            ]
            try:
                subprocess.run(cmd, check=False, cwd=str(PROJ_ROOT))
            except Exception as e:
                log(f"gen_eval_set n={en} FAIL: {e}")

        if not master_eval_dir.exists():
            log(f"WARN eval_n{en} not generated, skipping local copy")
            continue

        # Symlink master → local subset folder
        local_eval_dir.mkdir(parents=True, exist_ok=True)
        for cdir in sorted(master_eval_dir.iterdir()):
            if not cdir.is_dir() or cdir.name.startswith("_"):
                continue
            (local_eval_dir / cdir.name).mkdir(exist_ok=True)
            for f in cdir.glob("*.png"):
                tgt = local_eval_dir / cdir.name / f.name
                if not tgt.exists():
                    try:
                        tgt.symlink_to(f)
                    except (OSError, NotImplementedError):
                        import shutil
                        shutil.copy2(f, tgt)
        n = len(list(local_eval_dir.glob("*/*.png")))
        log(f"eval_n{en} prepared ({n} chips)")


def main():
    log(f"project root: {PROJ_ROOT}")
    log(f"data root: {DATA_ROOT}")
    log(f"classification chips: {MASTER_TRAIN}")

    log("=== STAGE 1A: ensure master train pool ===")
    ensure_master_pool(need=200)

    log("=== STAGE 1B: make train subsets ===")
    make_train_subsets()

    log("=== STAGE 1C: generate eval sets ===")
    make_eval_sets()

    log("=== STAGE 1 DONE ===")
    # Summary
    for tn in TRAIN_SIZES:
        d = OUT_BASE / f"train_n{tn}"
        n = len(list(d.glob("*/*.png"))) if d.exists() else 0
        print(f"  train_n{tn}: {n} chips")
    for en in EVAL_SIZES:
        d = OUT_BASE / f"eval_n{en}"
        n = len(list(d.glob("*/*.png"))) if d.exists() else 0
        print(f"  eval_n{en}:  {n} chips")


if __name__ == "__main__":
    main()
