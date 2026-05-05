#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Class-conditional GMM feature options for chipgrid wafers.

This is the Stage-A feature extractor for the class-30/GMM hybrid plan.
It compares four generative features without changing the CNN trainers:

alpha: GMM over all non-zero chip positions.
beta: GMM over target-object chip positions for object-bearing classes.
gamma: wafer-level summary vector.
delta: object-wise binary-map moments.

All GMMs are fit on the train split only, then scored on train/val/test.
"""
from __future__ import annotations

import argparse
import json
import time
from datetime import datetime
from pathlib import Path

import numpy as np
from sklearn.metrics import accuracy_score, f1_score, precision_recall_fscore_support
from sklearn.mixture import GaussianMixture

import sys
sys.path.insert(0, str(Path(__file__).parent))
from cnn_eval_chipgrid import (
    CHIP_OBJ_IDS,
    DEFAULT_DATA_DIR,
    DEFAULT_OBJ_ID_DIR,
    GRID_SIZE,
    _load_active_classes,
    _load_obj_static,
    build_npy_lookup,
    collect_samples,
    split_samples,
)


OBJ_NAME_TO_ID = {name: idx for idx, name in CHIP_OBJ_IDS.items() if idx > 0}
OBJ_NAMES = [CHIP_OBJ_IDS[i] for i in range(1, 6)]
OPTIONS = ("alpha", "beta", "gamma", "delta")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--n-per-class", type=int, default=100)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--options", default="alpha,beta,gamma,delta",
                   help="comma list from alpha,beta,gamma,delta")
    p.add_argument("--n-components", type=int, default=4)
    p.add_argument("--max-iter", type=int, default=200)
    p.add_argument("--reg-covar", type=float, default=1e-3)
    p.add_argument("--data-dir", default=DEFAULT_DATA_DIR)
    p.add_argument("--obj-id-dir", default=DEFAULT_OBJ_ID_DIR)
    p.add_argument("--active-classes-yaml", default=None,
                   help="YAML with classes: list. Limits data to those classes.")
    p.add_argument("--allow-missing-active-classes", action="store_true",
                   help="Drop active classes missing from data-dir instead of failing.")
    p.add_argument("--log-root", default="results_gmm_options")
    p.add_argument("--model-tag", default=None)
    p.add_argument("--save-features", action="store_true",
                   help="save train/val/test score matrices as npz")
    return p.parse_args()


def class_target_obj_id(class_name):
    for obj_name, obj_id in sorted(OBJ_NAME_TO_ID.items(), key=lambda kv: len(kv[0]), reverse=True):
        if class_name.endswith("_" + obj_name):
            return obj_id
    return None


def extract_obj_id_arrays(samples, npy_lookup, log_fn=print):
    out = []
    missing = 0
    t0 = time.time()
    for png_path, label in samples:
        arr = _load_obj_static(png_path, npy_lookup)
        if arr is None:
            arr = np.zeros((GRID_SIZE, GRID_SIZE), dtype=np.uint8)
            missing += 1
        out.append((arr, label))
    log_fn(f"[extract] {len(out)} arrays in {time.time()-t0:.1f}s (missing={missing})")
    return out


def positions_from_arr(arr, obj_id=None):
    if obj_id is None:
        mask = arr > 0
    else:
        mask = arr == obj_id
    ys, xs = np.where(mask)
    if len(ys) == 0:
        return np.zeros((0, 2), dtype=np.float32)
    pos = np.column_stack([ys, xs]).astype(np.float32)
    return pos / float(GRID_SIZE - 1)


def entropy_from_counts(counts):
    total = float(counts.sum())
    if total <= 0:
        return 0.0
    p = counts.astype(np.float64) / total
    p = p[p > 0]
    return float(-(p * np.log(p)).sum() / np.log(len(counts)))


def summary_feature(arr):
    pos = positions_from_arr(arr, None)
    counts = np.array([(arr == k).sum() for k in range(1, 6)], dtype=np.float32)
    n_chip = float(counts.sum())
    if len(pos) == 0:
        mean_y = mean_x = std_y = std_x = 0.0
    else:
        mean_y, mean_x = pos.mean(axis=0)
        std_y, std_x = pos.std(axis=0)
    dominant = float(np.argmax(counts) + 1) / 5.0 if n_chip > 0 else 0.0
    return np.array([
        n_chip / float(GRID_SIZE * GRID_SIZE),
        mean_y,
        mean_x,
        std_y,
        std_x,
        dominant,
        entropy_from_counts(counts),
    ], dtype=np.float32)


def _safe_skew(values):
    if len(values) < 3:
        return 0.0
    std = float(values.std())
    if std < 1e-6:
        return 0.0
    z = (values - float(values.mean())) / std
    return float(np.clip(np.mean(z ** 3), -5.0, 5.0) / 5.0)


def object_moment_feature(arr):
    feats = []
    for obj_id in range(1, 6):
        pos = positions_from_arr(arr, obj_id)
        count = len(pos)
        if count == 0:
            feats.extend([0.0] * 7)
            continue
        ys = pos[:, 0]
        xs = pos[:, 1]
        feats.extend([
            count / float(GRID_SIZE * GRID_SIZE),
            float(ys.mean()),
            float(xs.mean()),
            float(ys.std()),
            float(xs.std()),
            _safe_skew(ys),
            _safe_skew(xs),
        ])
    return np.array(feats, dtype=np.float32)


def fit_gmm(X, n_components, reg_covar, max_iter, seed):
    if X is None or len(X) < 2:
        return None
    n_actual = max(1, min(n_components, len(X)))
    model = GaussianMixture(
        n_components=n_actual,
        covariance_type="full",
        reg_covar=reg_covar,
        random_state=seed,
        max_iter=max_iter,
    )
    model.fit(X)
    return model


def train_position_gmms(train_arr, classes, mode, args, log_fn=print):
    gmms = {}
    target_obj_ids = [class_target_obj_id(c) for c in classes]
    for cls_idx, cls_name in enumerate(classes):
        obj_id = None if mode == "alpha" else target_obj_ids[cls_idx]
        pos_list = []
        for arr, label in train_arr:
            if label != cls_idx:
                continue
            pos = positions_from_arr(arr, obj_id)
            if len(pos) > 0:
                pos_list.append(pos)
        X = np.vstack(pos_list) if pos_list else None
        try:
            gmms[cls_idx] = fit_gmm(X, args.n_components, args.reg_covar, args.max_iter, args.seed)
        except Exception as exc:
            log_fn(f"[{mode}] fit failed class={cls_name}: {exc}")
            gmms[cls_idx] = None
    log_fn(f"[{mode}] trained {sum(v is not None for v in gmms.values())}/{len(classes)} position GMMs")
    return gmms, target_obj_ids


def train_vector_gmms(train_arr, classes, mode, args, log_fn=print):
    gmms = {}
    feature_fn = summary_feature if mode == "gamma" else object_moment_feature
    for cls_idx, cls_name in enumerate(classes):
        vecs = [feature_fn(arr) for arr, label in train_arr if label == cls_idx]
        if not vecs:
            gmms[cls_idx] = None
            continue
        X = np.stack(vecs)
        try:
            gmms[cls_idx] = fit_gmm(X, args.n_components, args.reg_covar, args.max_iter, args.seed)
        except Exception as exc:
            log_fn(f"[{mode}] fit failed class={cls_name}: {exc}")
            gmms[cls_idx] = None
    log_fn(f"[{mode}] trained {sum(v is not None for v in gmms.values())}/{len(classes)} vector GMMs")
    return gmms


def score_position_option(arrs, gmms, target_obj_ids, mode, n_classes):
    scores = np.full((len(arrs), n_classes), -1e9, dtype=np.float32)
    for i, (arr, _) in enumerate(arrs):
        for cls_idx in range(n_classes):
            gmm = gmms.get(cls_idx)
            if gmm is None:
                continue
            obj_id = None if mode == "alpha" else target_obj_ids[cls_idx]
            pos = positions_from_arr(arr, obj_id)
            if len(pos) == 0:
                continue
            scores[i, cls_idx] = float(gmm.score_samples(pos).mean())
    return scores


def score_vector_option(arrs, gmms, mode, n_classes):
    scores = np.full((len(arrs), n_classes), -1e9, dtype=np.float32)
    feature_fn = summary_feature if mode == "gamma" else object_moment_feature
    X = np.stack([feature_fn(arr) for arr, _ in arrs])
    for cls_idx in range(n_classes):
        gmm = gmms.get(cls_idx)
        if gmm is None:
            continue
        scores[:, cls_idx] = gmm.score_samples(X).astype(np.float32)
    return scores


def eval_scores(scores, arrs, classes, split_name, option, log_fn=print):
    y_true = np.array([label for _, label in arrs], dtype=np.int64)
    y_pred = scores.argmax(axis=1)
    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
    p, r, _, _ = precision_recall_fscore_support(y_true, y_pred, average="macro", zero_division=0)
    log_fn(f"[{option}] {split_name}: acc={acc*100:.2f}% f1={f1*100:.2f}% p={p*100:.2f}% r={r*100:.2f}%")
    per_p, per_r, per_f, sup = precision_recall_fscore_support(
        y_true, y_pred, labels=range(len(classes)), zero_division=0)
    weak = []
    for i, cls_name in enumerate(classes):
        if per_f[i] < 0.95:
            weak.append({
                "class": cls_name,
                "p": float(per_p[i]),
                "r": float(per_r[i]),
                "f1": float(per_f[i]),
                "support": int(sup[i]),
            })
    return {
        "acc": float(acc),
        "f1": float(f1),
        "p": float(p),
        "r": float(r),
        "weak": weak,
    }


def main():
    args = parse_args()
    options = [o.strip() for o in args.options.split(",") if o.strip()]
    bad = [o for o in options if o not in OPTIONS]
    if bad:
        raise ValueError(f"unknown options: {bad}")

    ts = datetime.now().strftime("%y%m%d_%H%M%S")
    tag = args.model_tag or f"gmmopts_n{args.n_per_class}_k{args.n_components}"
    out_dir = Path(args.log_root) / f"{tag}_{ts}"
    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = out_dir / "run.log"

    def log(msg):
        line = f"{datetime.now().strftime('%H:%M:%S')}  {msg}"
        print(line, flush=True)
        with open(log_path, "a", encoding="utf-8") as fp:
            fp.write(line + "\n")

    log("=== chipgrid GMM options ===")
    log(f"args: {vars(args)}")
    active_classes = _load_active_classes(args.active_classes_yaml)
    if active_classes:
        log(f"[active-classes] {len(active_classes)} classes from {args.active_classes_yaml}")
    samples, classes = collect_samples(
        args.data_dir,
        args.n_per_class,
        active_classes=active_classes,
        allow_missing_active_classes=args.allow_missing_active_classes,
    )
    train_s, val_s, test_s = split_samples(samples, seed=args.seed)
    npy_lookup = build_npy_lookup(args.obj_id_dir)
    log(f"classes ({len(classes)}): {classes}")
    log(f"split: train={len(train_s)} val={len(val_s)} test={len(test_s)}")
    log(f"obj_id npy indexed: {len(npy_lookup)}")

    train_arr = extract_obj_id_arrays(train_s, npy_lookup, log_fn=log)
    val_arr = extract_obj_id_arrays(val_s, npy_lookup, log_fn=log)
    test_arr = extract_obj_id_arrays(test_s, npy_lookup, log_fn=log)

    summary = {
        "args": vars(args),
        "classes": classes,
        "splits": {"train": len(train_s), "val": len(val_s), "test": len(test_s)},
        "options": {},
    }
    n_classes = len(classes)

    for option in options:
        log(f"--- option {option} ---")
        t0 = time.time()
        if option in ("alpha", "beta"):
            gmms, target_obj_ids = train_position_gmms(train_arr, classes, option, args, log_fn=log)
            train_scores = score_position_option(train_arr, gmms, target_obj_ids, option, n_classes)
            val_scores = score_position_option(val_arr, gmms, target_obj_ids, option, n_classes)
            test_scores = score_position_option(test_arr, gmms, target_obj_ids, option, n_classes)
        else:
            gmms = train_vector_gmms(train_arr, classes, option, args, log_fn=log)
            train_scores = score_vector_option(train_arr, gmms, option, n_classes)
            val_scores = score_vector_option(val_arr, gmms, option, n_classes)
            test_scores = score_vector_option(test_arr, gmms, option, n_classes)

        option_summary = {
            "train": eval_scores(train_scores, train_arr, classes, "TRAIN", option, log_fn=log),
            "val": eval_scores(val_scores, val_arr, classes, "VAL", option, log_fn=log),
            "test": eval_scores(test_scores, test_arr, classes, "TEST", option, log_fn=log),
            "seconds": float(time.time() - t0),
        }
        summary["options"][option] = option_summary
        if args.save_features:
            np.savez_compressed(
                out_dir / f"{option}_features.npz",
                train_scores=train_scores,
                val_scores=val_scores,
                test_scores=test_scores,
                y_train=np.array([label for _, label in train_arr], dtype=np.int64),
                y_val=np.array([label for _, label in val_arr], dtype=np.int64),
                y_test=np.array([label for _, label in test_arr], dtype=np.int64),
                classes=np.array(classes),
            )
            log(f"[{option}] saved features")

    with open(out_dir / "summary.json", "w", encoding="utf-8") as fp:
        json.dump(summary, fp, ensure_ascii=False, indent=2)
    log(f"saved summary: {out_dir / 'summary.json'}")


if __name__ == "__main__":
    main()
