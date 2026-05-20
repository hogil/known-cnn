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
OOD_CLASSES = ("CenterDonut", "CrossScratch", "DiagonalSmear", "Starburst")
MASTER_TRAIN = Path(os.environ.get("CLASSIFICATION_CHIPS_ROOT", str(DATA_ROOT / "classification_chips"))).resolve()


def log(msg):
    print(f"[gen_data] {msg}", flush=True)


def count_pngs(path: Path) -> int:
    return len(list(path.glob("*.png"))) if path.exists() else 0


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def ensure_master_pool(need=200):
    """Make sure classification_chips/ has ≥ need chips per class."""
    deficits = []
    for c in TRAIN_CLASSES:
        d = ensure_dir(MASTER_TRAIN / c)
        n = count_pngs(d)
        deficits.append(max(0, need - n))
        log(f"master {c}: {n} chips")
    max_deficit = max(deficits) if deficits else 0
    if max_deficit > 0:
        log(f"synthesizing single chips to fill deficit (max_deficit={max_deficit})")
        cmd = [
            sys.executable, "-u", "-m", "dist_apply._synth_chips_only",
            "--per-class", str(max_deficit),
            "--out", str(MASTER_TRAIN),
        ]
        subprocess.run(cmd, check=True, cwd=str(PROJ_ROOT))
    else:
        log(f"master pool OK (all classes ≥ {need})")


def make_train_subsets():
    for tn in TRAIN_SIZES:
        out_dir = OUT_BASE / f"train_n{tn}"
        if out_dir.exists() and all(count_pngs(out_dir / c) >= tn for c in TRAIN_CLASSES):
            existing = len(list(out_dir.glob("*/*.png")))
            log(f"train_n{tn} complete ({existing} files, all classes ≥ {tn}), skip")
            continue
        out_dir.mkdir(parents=True, exist_ok=True)
        for c in TRAIN_CLASSES:
            (out_dir / c).mkdir(exist_ok=True)
            n_have = count_pngs(out_dir / c)
            if n_have < tn:
                log(f"train_n{tn}/{c}: {n_have}/{tn} - filling")
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


def fill_ood_eval_class(mod, cls: str, target: int, rng, dst_root: Path):
    dst_dir = ensure_dir(dst_root / cls)
    if not hasattr(mod, "generate_direct_class"):
        raise RuntimeError("mega_matrix.ood_chips missing generate_direct_class")
    n_have = count_pngs(dst_dir)
    if n_have < target:
        log(f"OOD {cls}: direct chip generation {n_have}/{target}")
    mod.generate_direct_class(cls, target, rng, dst_root)


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
            + list(OOD_CLASSES)
        )
        missing_classes = [c for c in expected_classes
                           if not (local_eval_dir / c).exists()
                           or len(list((local_eval_dir / c).glob("*.png"))) < en]
        if local_eval_dir.exists() and not missing_classes:
            n = len(list(local_eval_dir.glob("*/*.png")))
            log(f"eval_n{en} local complete ({n} chips, all 16 classes ≥ {en}), skip")
            continue
        if missing_classes:
            log(f"eval_n{en} local missing classes: {missing_classes} - proceeding to gen/symlink")

        # Skip gen_eval_set only if 12 base classes (4 single + 6 2-combo + Normal + Invalid) all have ≥ en chips.
        # If any class < en, gen_eval_set is monolithic and re-runs all 12 (limitation accepted).
        base_classes = [c for c in expected_classes if c not in OOD_CLASSES]
        master_base_done = (master_eval_dir.exists() and all(
            (master_eval_dir / c).exists()
            and len(list((master_eval_dir / c).glob("*.png"))) >= en
            for c in base_classes))
        if master_base_done:
            log(f"master eval_n{en} base 12 classes all ≥ {en} chips, skip gen_eval_set")
        else:
            partial_base = []
            for c in base_classes:
                cd = master_eval_dir / c
                cnt = len(list(cd.glob("*.png"))) if cd.exists() else 0
                if cnt < en:
                    partial_base.append(f"{c}({cnt}/{en})")
            log(f"generating eval_n{en} via gen_eval_set (partial base: {partial_base})...")
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
        # gen_eval_set doesn't synth OOD, so generate 200x200 OOD chips directly.
        ood_n = en
        try:
            # Relocated 260520: previously top-level _gen_E_ood_chips.py (since
            # removed in top-level cleanup d12fb9c); now lives inside the pipeline.
            from mega_matrix import ood_chips as mod  # type: ignore
            if True:
                import random as _r
                rng = _r.Random(42)
                log(f"OOD direct dst_root={master_eval_dir}")
                for cls in OOD_CLASSES:
                    ood_cdir = master_eval_dir / cls
                    n_have = count_pngs(ood_cdir)
                    if n_have >= ood_n:
                        log(f"OOD {cls} already {n_have} ≥ {ood_n}, skip")
                        continue
                    fill_ood_eval_class(mod, cls, ood_n, rng, master_eval_dir)
        except Exception as e:
            raise RuntimeError(f"OOD direct generation failed: {type(e).__name__}: {e}") from e

        incomplete_ood = []
        for cls in OOD_CLASSES:
            cdir = master_eval_dir / cls
            n_have = count_pngs(cdir)
            if n_have < ood_n:
                incomplete_ood.append(f"{cls}({n_have}/{ood_n})")
        if incomplete_ood:
            raise RuntimeError(
                f"OOD eval classes missing after extraction under {master_eval_dir}: {incomplete_ood}"
            )

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
