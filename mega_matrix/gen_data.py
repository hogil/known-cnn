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


def _resolve_data_root() -> Path:
    env = os.environ.get("WM811K_ROOT")
    if env:
        return Path(env).resolve()
    default = (PROJ_ROOT / "data" / "wm-811k").resolve()
    # Fallback: if default missing classification_chips AND E:/data/images exists, use E:.
    if not (default / "classification_chips").exists():
        e = Path("E:/data/images")
        if (e / "classification_chips").exists():
            return e.resolve()
    return default


DATA_ROOT = _resolve_data_root()

def _sizes(env, default):
    v = os.environ.get(env)
    return [int(x) for x in v.split(",") if x.strip()] if v else default


TRAIN_SIZES = _sizes("MEGA_TRAIN_SIZES", [50, 100, 200])
EVAL_SIZES = _sizes("MEGA_EVAL_SIZES", [200, 2000, 20000])
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
        subprocess.run(cmd, check=True, cwd=str(PROJ_ROOT))
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

        # Strict local skip: 16 expected classes must all exist with PNGs
        expected_classes = (
            ["bank_boundary", "fork", "scratch", "scratch_rot"]  # 4 single
            + ["bank_boundary+fork", "bank_boundary+scratch", "bank_boundary+scratch_rot",
               "fork+scratch", "fork+scratch_rot", "scratch+scratch_rot"]  # 6 2-combo
            + ["Normal", "Invalid"]
            + ["CenterDonut", "CrossScratch", "DiagonalSmear", "Starburst"]  # 4 OOD
        )
        missing_classes = [c for c in expected_classes
                           if not (local_eval_dir / c).exists()
                           or len(list((local_eval_dir / c).glob("*.png"))) == 0]
        if local_eval_dir.exists() and not missing_classes:
            n = len(list(local_eval_dir.glob("*/*.png")))
            log(f"eval_n{en} local complete ({n} chips, all 16 classes), skip")
            continue
        if missing_classes:
            log(f"eval_n{en} local missing classes: {missing_classes} — proceeding to gen/symlink")

        # Skip gen_eval_set only if 12 base classes (4 single + 6 2-combo + Normal + Invalid) all exist.
        # OOD checked separately below.
        base_classes = [c for c in expected_classes if c not in
                        ("CenterDonut", "CrossScratch", "DiagonalSmear", "Starburst")]
        master_base_done = (master_eval_dir.exists() and all(
            (master_eval_dir / c).exists()
            and len(list((master_eval_dir / c).glob("*.png"))) > 0
            for c in base_classes))
        if master_base_done:
            log(f"master eval_n{en} base 12 classes exist, skip gen_eval_set")
        else:
            missing_base = [c for c in base_classes
                            if not (master_eval_dir / c).exists()
                            or len(list((master_eval_dir / c).glob("*.png"))) == 0]
            log(f"generating eval_n{en} via gen_eval_set (missing base: {missing_base})...")
            cmd = [
                sys.executable, "-u", "-X", "utf8", "-m", "chip_multilabel.gen_eval_set",
                "--out-root", str(master_eval_dir),
                "--per-defect", str(en),
                "--per-normal", str(en),
                "--per-invalid", str(en),  # match defect scale (was en // 4, 260514 fix)
                # NO --include-triples: absolute rule 260512 — positive = single + 2-combo only.
                # 3-combo would inflate eval but is excluded from bit_F1 anyway by aggregator.
                "--classification-chips-root", str(MASTER_TRAIN),
                "--seed", "42",
            ]
            subprocess.run(cmd, check=True, cwd=str(PROJ_ROOT))

        # OOD wafer-pattern chips (absolute rule 260512 — FAR group e):
        # gen_eval_set doesn't synth OOD. Extract from wafer canvas if source dir exists.
        # Skip individual OOD class if already has PNGs.
        ood_classes = ("CenterDonut", "CrossScratch", "DiagonalSmear", "Starburst")
        unknown_src = DATA_ROOT / "unknown"
        ood_n = en
        if unknown_src.exists():
            try:
                sys.path.insert(0, str(PROJ_ROOT))
                import importlib.util
                spec = importlib.util.spec_from_file_location(
                    "_gen_E_ood_chips", str(PROJ_ROOT / "_gen_E_ood_chips.py"))
                if spec and spec.loader:
                    mod = importlib.util.module_from_spec(spec)
                    spec.loader.exec_module(mod)
                    mod.SRC_ROOT = unknown_src
                    mod.DST_ROOT = master_eval_dir
                    import random as _r
                    rng = _r.Random(42)
                    for cls in ood_classes:
                        ood_cdir = master_eval_dir / cls
                        if ood_cdir.exists() and len(list(ood_cdir.glob("*.png"))) > 0:
                            log(f"OOD {cls} exists ({len(list(ood_cdir.glob('*.png')))} chips), skip")
                            continue
                        try:
                            mod.extract_class(cls, ood_n, 0.03, rng)
                        except Exception as e:
                            log(f"WARN OOD extract {cls}: {type(e).__name__}: {e}")
            except Exception as e:
                log(f"WARN OOD module load failed: {type(e).__name__}: {e}")
        else:
            log(f"WARN OOD source {unknown_src} missing — skipping OOD class extraction")

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
