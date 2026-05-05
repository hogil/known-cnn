#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Stage A: per-wafer-class KDE (chip positions) + GMM (count vectors) standalone.

obj_id 32x32 .npy 만 사용 (R failbit 채널 안 봄).
33+ wafer class 각각의 통계 prior 만으로 분류 가능한지 sanity check.

KDE: 각 class 의 chip 위치 (r, c) 분포 학습. score = mean log_density per chip.
GMM: 각 class 의 5-D count vector (chip object 별 chip 수) 분포 학습. score = log_lik.

inference: 33 (kde + gmm) score → argmax = predicted class.

사용:
    python _chipgrid_kde_gmm.py --n-per-class 100 --seed 42
    python _chipgrid_kde_gmm.py --n-per-class 220 --bandwidth 1.5 --n-components 3
"""
from __future__ import annotations
import argparse, json, pickle, time
from pathlib import Path
from datetime import datetime
import numpy as np
from sklearn.neighbors import KernelDensity
from sklearn.mixture import GaussianMixture
from sklearn.metrics import f1_score, accuracy_score, precision_recall_fscore_support

# Reuse data loading from cnn_eval_chipgrid.py (no behavior change)
import sys
sys.path.insert(0, str(Path(__file__).parent))
from cnn_eval_chipgrid import (
    DEFAULT_DATA_DIR, DEFAULT_OBJ_ID_DIR, EXCLUDE_CLASSES, GRID_SIZE,
    build_npy_lookup, collect_samples, split_samples, _load_active_classes, _load_obj_static,
)


# === Feature extraction ===
def extract_obj_id_arrays(samples, npy_lookup, log_fn=print):
    """Returns list of (obj_id_arr 32×32 uint8, label_int)."""
    out = []
    n_missing = 0
    t0 = time.time()
    for png_path, label in samples:
        arr = _load_obj_static(png_path, npy_lookup)
        if arr is None:
            arr = np.zeros((GRID_SIZE, GRID_SIZE), dtype=np.uint8)
            n_missing += 1
        out.append((arr, label))
    log_fn(f"[extract] {len(out)} obj_id arrays in {time.time()-t0:.1f}s  (missing: {n_missing})")
    return out


def positions_from_obj_id(arr):
    """Non-zero chip positions as (n_chips, 2) [r, c]."""
    rs, cs = np.where(arr > 0)
    return np.column_stack([rs, cs]).astype(np.float32)


def count_vector_from_obj_id(arr, n_obj=5):
    """5-D count vector: counts of obj_id 1..N."""
    return np.array([(arr == k).sum() for k in range(1, n_obj + 1)], dtype=np.float32)


# === Per-class KDE / GMM training ===
def train_kde_per_class(samples_arr, n_classes, bandwidth=1.0, log_fn=print):
    """KDE on chip positions per wafer class. None if class has < 2 chip positions."""
    kde_dict = {}
    n_chips_per = {}
    for cls_idx in range(n_classes):
        positions_list = []
        for arr, lbl in samples_arr:
            if lbl != cls_idx:
                continue
            pos = positions_from_obj_id(arr)
            if len(pos) > 0:
                positions_list.append(pos)
        if not positions_list:
            kde_dict[cls_idx] = None
            n_chips_per[cls_idx] = 0
            continue
        positions = np.vstack(positions_list)
        n_chips_per[cls_idx] = len(positions)
        if len(positions) < 2:
            kde_dict[cls_idx] = None
            continue
        try:
            kde = KernelDensity(bandwidth=bandwidth, kernel="gaussian")
            kde.fit(positions)
            kde_dict[cls_idx] = kde
        except Exception as e:
            log_fn(f"  [KDE] class {cls_idx} fit failed: {e}")
            kde_dict[cls_idx] = None
    log_fn(f"[KDE] trained {sum(1 for v in kde_dict.values() if v is not None)}/{n_classes} classes")
    return kde_dict, n_chips_per


def train_gmm_per_class(samples_arr, n_classes, n_components=2, log_fn=print):
    """GMM on 5-D count vector per wafer class."""
    gmm_dict = {}
    n_wafers_per = {}
    for cls_idx in range(n_classes):
        vecs = [count_vector_from_obj_id(arr) for arr, lbl in samples_arr if lbl == cls_idx]
        n_wafers_per[cls_idx] = len(vecs)
        if not vecs:
            gmm_dict[cls_idx] = None
            continue
        X = np.stack(vecs)
        n_actual = max(1, min(n_components, len(X)))
        try:
            gmm = GaussianMixture(n_components=n_actual, covariance_type="full",
                                  reg_covar=1e-3, random_state=42, max_iter=200)
            gmm.fit(X)
            gmm_dict[cls_idx] = gmm
        except Exception as e:
            log_fn(f"  [GMM] class {cls_idx} fit failed: {e}")
            gmm_dict[cls_idx] = None
    log_fn(f"[GMM] trained {sum(1 for v in gmm_dict.values() if v is not None)}/{n_classes} classes")
    return gmm_dict, n_wafers_per


# === Inference ===
def compute_log_lik_one(arr, kde_dict, gmm_dict, n_classes):
    """For one wafer obj_id_arr, return (kde_log_lik 33-D, gmm_log_lik 33-D)."""
    pos = positions_from_obj_id(arr)
    cnt = count_vector_from_obj_id(arr).reshape(1, -1)
    kde_log = np.full(n_classes, -1e9, dtype=np.float32)
    gmm_log = np.full(n_classes, -1e9, dtype=np.float32)
    for cls_idx in range(n_classes):
        kde = kde_dict.get(cls_idx)
        gmm = gmm_dict.get(cls_idx)
        if kde is not None and len(pos) > 0:
            kde_log[cls_idx] = float(kde.score_samples(pos).mean())  # mean log density per chip
        if gmm is not None:
            gmm_log[cls_idx] = float(gmm.score_samples(cnt)[0])
    return kde_log, gmm_log


def compute_log_lik_batch(arrs, kde_dict, gmm_dict, n_classes, log_fn=print):
    """Vectorized eval — one wafer at a time, but batch return."""
    t0 = time.time()
    n = len(arrs)
    kde_mat = np.zeros((n, n_classes), dtype=np.float32)
    gmm_mat = np.zeros((n, n_classes), dtype=np.float32)
    for i, (arr, _) in enumerate(arrs):
        kde_log, gmm_log = compute_log_lik_one(arr, kde_dict, gmm_dict, n_classes)
        kde_mat[i] = kde_log
        gmm_mat[i] = gmm_log
    log_fn(f"[infer] {n} samples in {time.time()-t0:.1f}s")
    return kde_mat, gmm_mat


def evaluate(arrs, kde_dict, gmm_dict, n_classes, classes, name, log_fn=print):
    kde_mat, gmm_mat = compute_log_lik_batch(arrs, kde_dict, gmm_dict, n_classes, log_fn=log_fn)
    score = kde_mat + gmm_mat
    y_pred = score.argmax(axis=1)
    y_true = np.array([lbl for _, lbl in arrs])
    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
    log_fn(f"{name}: acc={acc*100:.2f}%  f1={f1*100:.2f}%")
    # also measure KDE-only and GMM-only
    pred_kde = kde_mat.argmax(axis=1)
    pred_gmm = gmm_mat.argmax(axis=1)
    f1_kde = f1_score(y_true, pred_kde, average="macro", zero_division=0)
    f1_gmm = f1_score(y_true, pred_gmm, average="macro", zero_division=0)
    log_fn(f"{name} ablation: KDE-only f1={f1_kde*100:.2f}%, GMM-only f1={f1_gmm*100:.2f}%")
    return {
        "acc": float(acc), "f1": float(f1),
        "f1_kde_only": float(f1_kde), "f1_gmm_only": float(f1_gmm),
        "y_true": y_true.tolist(), "y_pred": y_pred.tolist(),
    }


def per_class_report(y_true, y_pred, classes):
    p, r, f, sup = precision_recall_fscore_support(
        y_true, y_pred, labels=range(len(classes)), zero_division=0)
    rows = []
    for i, c in enumerate(classes):
        rows.append({"class": c, "f1": float(f[i]), "p": float(p[i]),
                     "r": float(r[i]), "sup": int(sup[i])})
    return rows


# === Main ===
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--n-per-class", type=int, default=100)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--bandwidth", type=float, default=1.0)
    p.add_argument("--n-components", type=int, default=2)
    p.add_argument("--data-dir", default=DEFAULT_DATA_DIR)
    p.add_argument("--obj-id-dir", default=DEFAULT_OBJ_ID_DIR)
    p.add_argument("--active-classes-yaml", default=None,
                   help="YAML with classes: list. Limits Stage A to those classes.")
    p.add_argument("--allow-missing-active-classes", action="store_true",
                   help="Drop active classes missing from data-dir instead of failing.")
    p.add_argument("--log-root", default="logs_chipgrid_kde_gmm")
    p.add_argument("--model-tag", default=None)
    args = p.parse_args()

    ts = datetime.now().strftime("%y%m%d_%H%M%S")
    tag = args.model_tag or f"kdegmm_n{args.n_per_class}_bw{args.bandwidth}_k{args.n_components}"
    out_dir = Path(args.log_root) / f"{tag}_{ts}"
    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = out_dir / "run.log"

    def log(msg):
        line = f"{datetime.now().strftime('%H:%M:%S')}  {msg}"
        print(line, flush=True)
        with open(log_path, "a", encoding="utf-8") as fp:
            fp.write(line + "\n")

    log(f"=== KDE + GMM standalone (Stage A) ===")
    log(f"args: {vars(args)}")

    # Data
    active_classes = _load_active_classes(args.active_classes_yaml)
    if active_classes:
        log(f"[active-classes] {len(active_classes)} classes from {args.active_classes_yaml}")
    samples, classes = collect_samples(
        args.data_dir,
        args.n_per_class,
        active_classes=active_classes,
        allow_missing_active_classes=args.allow_missing_active_classes,
    )
    n_classes = len(classes)
    train_s, val_s, test_s = split_samples(samples, seed=args.seed)
    npy_lookup = build_npy_lookup(args.obj_id_dir)
    log(f"classes ({n_classes}): {classes}")
    log(f"split: train={len(train_s)} val={len(val_s)} test={len(test_s)}")

    train_arr = extract_obj_id_arrays(train_s, npy_lookup, log_fn=log)
    val_arr = extract_obj_id_arrays(val_s, npy_lookup, log_fn=log)
    test_arr = extract_obj_id_arrays(test_s, npy_lookup, log_fn=log)

    # Train
    kde_dict, kde_npc = train_kde_per_class(train_arr, n_classes, bandwidth=args.bandwidth, log_fn=log)
    gmm_dict, gmm_npc = train_gmm_per_class(train_arr, n_classes, n_components=args.n_components, log_fn=log)

    # Eval
    val_res = evaluate(val_arr, kde_dict, gmm_dict, n_classes, classes, "VAL", log_fn=log)
    test_res = evaluate(test_arr, kde_dict, gmm_dict, n_classes, classes, "TEST", log_fn=log)

    # Per-class
    val_per_class = per_class_report(val_res["y_true"], val_res["y_pred"], classes)
    test_per_class = per_class_report(test_res["y_true"], test_res["y_pred"], classes)
    log("VAL weak class (F1 < 0.95):")
    for r in sorted(val_per_class, key=lambda x: x["f1"]):
        if r["f1"] < 0.95:
            log(f"  {r['class']:<32}  F1={r['f1']:.3f}  Sup={r['sup']}")

    # Save
    with open(out_dir / "kde.pkl", "wb") as f:
        pickle.dump({"kde_dict": kde_dict, "classes": classes,
                     "bandwidth": args.bandwidth, "n_chips_per_class": kde_npc}, f)
    with open(out_dir / "gmm.pkl", "wb") as f:
        pickle.dump({"gmm_dict": gmm_dict, "classes": classes,
                     "n_components": args.n_components, "n_wafers_per_class": gmm_npc}, f)
    summary = {
        "val_acc": val_res["acc"], "val_f1": val_res["f1"],
        "val_f1_kde_only": val_res["f1_kde_only"], "val_f1_gmm_only": val_res["f1_gmm_only"],
        "test_acc": test_res["acc"], "test_f1": test_res["f1"],
        "test_f1_kde_only": test_res["f1_kde_only"], "test_f1_gmm_only": test_res["f1_gmm_only"],
        "val_per_class": val_per_class,
        "test_per_class": test_per_class,
        "n_train": len(train_arr), "n_val": len(val_arr), "n_test": len(test_arr),
        "n_classes": n_classes, "classes": classes,
        "bandwidth": args.bandwidth, "n_components": args.n_components,
        "n_per_class": args.n_per_class, "seed": args.seed,
        "run_ts": ts, "tag": tag,
    }
    with open(out_dir / "eval_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    log(f"saved → {out_dir}")
    log(f"=== DONE ===")
    log(f"VAL  f1={val_res['f1']*100:.2f}%  (kde={val_res['f1_kde_only']*100:.2f}, gmm={val_res['f1_gmm_only']*100:.2f})")
    log(f"TEST f1={test_res['f1']*100:.2f}%  (kde={test_res['f1_kde_only']*100:.2f}, gmm={test_res['f1_gmm_only']*100:.2f})")


if __name__ == "__main__":
    main()
