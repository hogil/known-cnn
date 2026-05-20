"""
Mega matrix — Stage 5: pseudo-label retrain

Process:
1. Find best primary model (highest bit_F1 from 18-cell matrix)
2. Apply on `chip_multilabel_v15direct_n1000/` single-defect pool (1000/class)
3. Filter: max_prob > THR AND prediction = folder label (high confidence)
4. Add filtered chips to training set
5. Retrain with --val-criterion margin_max
6. Evaluate on eval_n2000
"""
import json
import glob
import ast
import subprocess
import sys
import os
from pathlib import Path
from datetime import datetime

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("HF_DATASETS_OFFLINE", "1")
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")

import numpy as np
import pandas as pd
import torch
from PIL import Image
import timm
import torchvision.transforms as T

PROJ_ROOT = Path(__file__).parent.parent
OUT_BASE = PROJ_ROOT / "outputs" / "_mega_matrix"          # shared data root (train_n*, eval_n*)
DATA_ROOT = Path(os.environ.get("IMAGES_ROOT", str(PROJ_ROOT / "data" / "images"))).resolve()
# Per-backbone model namespace (set by run.sh / run_ddp.sh / run_all.sh via env)
# Falls back to OUT_BASE for legacy single-backbone use
MODEL_BASE = Path(os.environ.get("MEGA_MODEL_BASE", str(OUT_BASE)))
BACKBONE = os.environ.get("MEGA_BACKBONE", "convnextv2_base.fcmae_ft_in22k_in1k_384")
IMG_SIZE = int(os.environ.get("MEGA_IMG_SIZE", "384"))
PSEUDO_POOL_SRC = Path(os.environ.get("PSEUDO_POOL_SRC", str(DATA_ROOT / "chip_multilabel_v15direct_n1000")))
PSEUDO_TRAIN_DIR = MODEL_BASE / "pseudo_train"
TRAIN_CLASSES = ["bank_boundary", "fork", "scratch", "scratch_rot"]
CONF_THRESHOLD = 0.85   # max_prob >= 0.85 for pseudo-label inclusion


def find_model_roots(prefix):
    cands = list(MODEL_BASE.glob(f"*_{prefix}")) + list(MODEL_BASE.glob(prefix))
    return sorted([p for p in cands if p.is_dir()], key=lambda p: p.name, reverse=True)


def find_run_dirs(model_root):
    runs = [p for p in model_root.iterdir() if p.is_dir() and (p / "best_model.pth").exists()]
    return sorted(runs, key=lambda p: p.name, reverse=True)


def log(msg):
    print(f"[pseudo] {msg}", flush=True)


def find_best_primary_model():
    """Scan 6 trained models, return one with highest bit_F1 on eval_n2000."""
    from sklearn.metrics import f1_score
    BITS = TRAIN_CLASSES
    best = (None, None, -1.0, None)  # (TN, SEL, bit_F1, run_path)
    for tn in [50, 100, 200]:
        for sel in ["f1", "margin_max"]:
            model_roots = find_model_roots(f"model_train{tn}_{sel}")
            if not model_roots:
                continue
            runs = find_run_dirs(model_roots[0])
            if not runs:
                continue
            run = runs[0]
            eval_dirs = list(run.glob("eval_2000/stage1_*"))
            if not eval_dirs:
                continue
            df = pd.read_parquet(eval_dirs[0] / "preds_chip.parquet")
            sub = df[df["cell_id"] == "T0__I13"].copy()
            if len(sub) == 0:
                continue
            sub["true_labels"] = sub["true_labels"].apply(lambda s: ast.literal_eval(s) if isinstance(s, str) else list(s))
            sub["n_true"] = sub["true_labels"].apply(len)
            pos = sub[
                ~sub["class_key"].str.contains("Normal|Invalid|ood_|CenterDonut|CrossScratch|DiagonalSmear|Starburst", regex=True)
                & sub["n_true"].isin([1, 2])
            ]
            if len(pos) == 0:
                continue
            thr_path = eval_dirs[0] / "thresholds.json"
            if not thr_path.exists():
                continue
            t = json.load(open(thr_path))["T0__I13"]["thresholds"]
            f1s = []
            for b in BITS:
                gt = pos["true_labels"].apply(lambda l: int(b in l)).values
                pred = (pos[f"prob_{b}"].values >= t[b]).astype(int)
                f1s.append(f1_score(gt, pred, zero_division=0))
            bit_F1 = float(np.mean(f1s))
            log(f"  train{tn}_{sel} bit_F1={bit_F1:.4f}")
            if bit_F1 > best[2]:
                best = (tn, sel, bit_F1, run)
    return best


def predict_pool(model_ckpt, pool_dir, img_size=384):
    """Run model on all chips in pool_dir/<class>/. Return DataFrame."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cd = torch.load(model_ckpt, map_location=device, weights_only=False)
    model = timm.create_model(cd["backbone"], pretrained=False, num_classes=len(TRAIN_CLASSES))
    model.load_state_dict(cd["model"])
    model.to(device).eval()
    tf = T.Compose([
        T.Resize((img_size, img_size), interpolation=T.InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ])
    rows = []
    with torch.no_grad():
        for c in TRAIN_CLASSES:
            d = pool_dir / c
            if not d.exists():
                continue
            files = sorted(d.glob("*.png"))
            log(f"  predict {c}: {len(files)} chips")
            for i in range(0, len(files), 32):
                batch = files[i:i + 32]
                imgs = []
                for f in batch:
                    try:
                        img = Image.open(f).convert("RGB")
                        imgs.append(tf(img))
                    except Exception:
                        imgs.append(torch.zeros(3, img_size, img_size))
                x = torch.stack(imgs).to(device)
                with torch.amp.autocast(device_type=device.type, enabled=device.type == "cuda"):
                    logits = model(x)
                probs = torch.sigmoid(logits.float()).cpu().numpy()
                for j, f in enumerate(batch):
                    rows.append({
                        "chip_path": str(f),
                        "folder_class": c,
                        **{f"prob_{b}": float(probs[j, k]) for k, b in enumerate(TRAIN_CLASSES)},
                    })
    return pd.DataFrame(rows)


def filter_high_conf(df, thr=0.85):
    """Keep chips where max_prob > thr AND argmax_class == folder_class."""
    df["max_prob"] = df[[f"prob_{b}" for b in TRAIN_CLASSES]].max(axis=1)
    df["pred_class"] = df[[f"prob_{b}" for b in TRAIN_CLASSES]].values.argmax(axis=1)
    df["pred_class_name"] = df["pred_class"].apply(lambda i: TRAIN_CLASSES[i])
    df["match"] = df["folder_class"] == df["pred_class_name"]
    filt = df[(df["max_prob"] >= thr) & df["match"]].copy()
    log(f"  total {len(df)} → filtered {len(filt)} (thr={thr}, accuracy match)")
    return filt


def build_pseudo_train_dir(filt_df, base_train="train_n200"):
    """Combine base train + pseudo-labeled chips into pseudo_train/."""
    if PSEUDO_TRAIN_DIR.exists() and any(PSEUDO_TRAIN_DIR.iterdir()):
        n = len(list(PSEUDO_TRAIN_DIR.glob("*/*.png")))
        log(f"  pseudo_train exists ({n} chips), skip rebuild")
        return PSEUDO_TRAIN_DIR
    PSEUDO_TRAIN_DIR.mkdir(parents=True, exist_ok=True)
    base = OUT_BASE / base_train
    for c in TRAIN_CLASSES:
        (PSEUDO_TRAIN_DIR / c).mkdir(exist_ok=True)
        # Original base
        for f in (base / c).glob("*.png"):
            tgt = PSEUDO_TRAIN_DIR / c / ("orig_" + f.name)
            if not tgt.exists():
                try:
                    tgt.symlink_to(f)
                except (OSError, NotImplementedError):
                    import shutil; shutil.copy2(f, tgt)
        # Pseudo-labeled
        sub = filt_df[filt_df["pred_class_name"] == c]
        for _, row in sub.iterrows():
            src = Path(row["chip_path"])
            tgt = PSEUDO_TRAIN_DIR / c / ("pseudo_" + src.name)
            if not tgt.exists():
                try:
                    tgt.symlink_to(src)
                except (OSError, NotImplementedError):
                    import shutil; shutil.copy2(src, tgt)
    # Stats
    for c in TRAIN_CLASSES:
        n = len(list((PSEUDO_TRAIN_DIR / c).glob("*.png")))
        log(f"  pseudo_train/{c}: {n} chips")
    return PSEUDO_TRAIN_DIR


def retrain_and_eval(pseudo_train_dir, sel="margin_max"):
    """Train new model on pseudo_train + evaluate."""
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_root = MODEL_BASE / f"{stamp}_model_pseudo_{sel}"
    if out_root.exists():
        log(f"  pseudo model {sel} exists, skip retrain")
    else:
        log(f"  retrain on pseudo_train (sel={sel}, backbone={BACKBONE})")
        # Offline weights passthrough (closed-network server)
        weights_dir = PROJ_ROOT / "mega_matrix" / "weights"
        offline_wpath = weights_dir / f"{BACKBONE}.pth"
        if not offline_wpath.exists():
            log(f"  ERROR missing offline weights for {BACKBONE} under {weights_dir}")
            log("  closed-network mode forbids HF/timm download; skip pseudo retrain")
            return
        cmd = [
            sys.executable, "-u", "-m", "chip_multilabel._train_chip_variant",
            "--variant", "T7", "--ls", "0.30", "--epochs", "10",
            "--batch", "2", "--accum", "8", "--seed", "1",
            "--lr", "1e-4", "--no-normal",
            "--val-criterion", sel,
            "--save-every-epoch",
            "--data-root", str(pseudo_train_dir),
            "--cutmix-mode", "complement", "--cutmix-pair", "masked",
            "--cutmix-pair-fill", "corner",
            "--cutmix-p", "0.25", "--cutmix-grid-dim", "16", "--cutmix-n-groups", "2",
            "--cutmix-complete-label-scale", "0.5",
            "--backbone-timm", BACKBONE,
            "--img-size", str(IMG_SIZE),
            "--out-root", str(out_root),
            "--tag", f"pseudo_{sel}",
            "--backbone-timm-weights", str(offline_wpath),
        ]
        log(f"  using offline weights {offline_wpath.name}")
        subprocess.run(cmd, check=False, cwd=str(PROJ_ROOT))
        # Cleanup
        runs = find_run_dirs(out_root)
        if runs:
            for ep_pth in runs[0].glob("epoch_*.pth"):
                ep_pth.unlink()
    # Eval on eval_n2000
    runs = find_run_dirs(out_root)
    if not runs:
        log(f"  no run dir, skip eval")
        return
    run = runs[0]
    for en in [200, 2000, 20000]:
        eval_set = OUT_BASE / f"eval_n{en}"
        if not eval_set.exists():
            continue
        eval_out = run / f"eval_{en}"
        if eval_out.exists():
            continue
        log(f"  pseudo eval n={en}")
        cmd = [
            sys.executable, "-u", "-m", "chip_multilabel.run_stage1",
            "--model", str(run / "best_model.pth"),
            "--eval-set", str(eval_set),
            "--out-root", str(eval_out),
            "--variants", "I3,I7,I10,I13",
            "--n-per-class", "99999",
            "--strength-min", "0.0", "--strength-max", "1.0",
            "--seed", "42",
        ]
        subprocess.run(cmd, check=False, cwd=str(PROJ_ROOT))


def main():
    log("=== STAGE 5: pseudo-label retrain ===")
    log(f"data root: {DATA_ROOT}")
    log(f"pseudo pool: {PSEUDO_POOL_SRC}")
    log("Step 1: find best primary model from 18-cell matrix")
    tn, sel, bit_F1, run_path = find_best_primary_model()
    if run_path is None:
        log("ERROR: no primary model found")
        return
    log(f"  best: train{tn}_{sel} bit_F1={bit_F1:.4f} @ {run_path}")
    best_ckpt = run_path / "best_model.pth"

    log("Step 2: predict on pseudo-label pool")
    if not PSEUDO_POOL_SRC.exists():
        log(f"ERROR: pool {PSEUDO_POOL_SRC} missing")
        return
    pred_df = predict_pool(best_ckpt, PSEUDO_POOL_SRC, img_size=IMG_SIZE)
    pred_df.to_csv(MODEL_BASE / "pseudo_preds.csv", index=False)

    log("Step 3: filter high-confidence pseudo-labels")
    filt = filter_high_conf(pred_df, thr=CONF_THRESHOLD)
    filt.to_csv(MODEL_BASE / "pseudo_preds_filtered.csv", index=False)

    log("Step 4: build pseudo_train directory (base train + pseudo-labeled)")
    pseudo_train = build_pseudo_train_dir(filt, base_train=f"train_n{tn}")

    log("Step 5: retrain on pseudo_train (val_margin selection)")
    retrain_and_eval(pseudo_train, sel="margin_max")
    log("Step 5b: also retrain with val_f1 for comparison")
    retrain_and_eval(pseudo_train, sel="f1")

    log("=== STAGE 5 DONE - see model_pseudo_margin_max + model_pseudo_f1 ===")


if __name__ == "__main__":
    main()
