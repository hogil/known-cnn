"""Eval set loader: walks chip_multilabel/ master folder and emits typed records.

260506 — single SoT folder + runtime sampling (memory rule feedback_no_subset_archive_folders.md):
the master `chip_multilabel/` folder always holds all generated chips; runtime CLI flags
(--n-per-class, --strength-min/max, --include-classes, --seed) decide which subset
to evaluate without ever creating subset folders on disk.
"""
from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

from .constants import (ALL_CLASS_KEYS, COMBO_KEYS, KEY_TO_LABELS, OOD_OVERLAY_KEYS,
                        SINGLE_KEYS, TRAIN_CLASSES, WAFER_PATTERN_KEYS)

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


@dataclass
class EvalRecord:
    chip_path: str
    class_key: str
    true_multihot: np.ndarray  # (4,) {0,1}
    is_invalid_gt: bool
    is_normal_gt: bool

    @property
    def true_active(self) -> List[str]:
        return [c for c, v in zip(TRAIN_CLASSES, self.true_multihot) if v == 1]


def _multihot_for_key(class_key: str) -> Tuple[np.ndarray, bool, bool]:
    """Return (multihot 4-d, is_invalid, is_normal) for a class_key in ALL_CLASS_KEYS."""
    if class_key == "Invalid":
        return np.zeros(len(TRAIN_CLASSES), dtype=np.int64), True, False
    if class_key == "Normal":
        return np.zeros(len(TRAIN_CLASSES), dtype=np.int64), False, True
    if class_key in WAFER_PATTERN_KEYS:
        # 260506 — 5 wafer-pattern OOD classes (CenterCircle/CenterDonut/CrescentArc/RingDots/Row).
        # NOT in training. Multihot all-zero (no chip-level obj match). Diagnostic class for
        # measuring model's OOD behavior on chips with wafer-level defect pattern visible.
        return np.zeros(len(TRAIN_CLASSES), dtype=np.int64), False, False
    if class_key in SINGLE_KEYS:
        m = np.zeros(len(TRAIN_CLASSES), dtype=np.int64)
        m[TRAIN_CLASSES.index(class_key)] = 1
        return m, False, False
    if class_key in COMBO_KEYS:
        labels = class_key.split("+")
        m = np.zeros(len(TRAIN_CLASSES), dtype=np.int64)
        for c in labels:
            m[TRAIN_CLASSES.index(c)] = 1
        return m, False, False
    if class_key in OOD_OVERLAY_KEYS:
        # 260507 — 4 OOD overlay classes (2 trained + 1 OOD). GT bits = 2 trained 만 active.
        # format: '<a>+<b>+ood_<OOD>' → split before '+ood_' → 2 trained labels.
        trained_part = class_key.split("+ood_")[0]
        labels = trained_part.split("+")
        m = np.zeros(len(TRAIN_CLASSES), dtype=np.int64)
        for c in labels:
            m[TRAIN_CLASSES.index(c)] = 1
        return m, False, False
    raise KeyError(f"unknown class_key: {class_key}")


def discover_records(eval_root: str | Path) -> List[EvalRecord]:
    """Walk eval_root subfolders for all PNGs (legacy path, no manifest required)."""
    root = Path(eval_root)
    if not root.exists():
        raise FileNotFoundError(f"eval root not found: {root}")
    out: List[EvalRecord] = []
    for class_key in ALL_CLASS_KEYS:
        cdir = root / class_key
        if not cdir.exists():
            continue
        m, is_inv, is_norm = _multihot_for_key(class_key)
        for png in sorted(cdir.glob("*.png")):
            out.append(EvalRecord(
                chip_path=str(png),
                class_key=class_key,
                true_multihot=m,
                is_invalid_gt=is_inv,
                is_normal_gt=is_norm,
            ))
    return out


def _read_manifest(eval_root: Path) -> List[Dict[str, str]]:
    p = eval_root / "manifest.csv"
    if not p.exists():
        raise FileNotFoundError(f"manifest.csv not found at {p}. Regenerate via gen_eval_set.py.")
    rows: List[Dict[str, str]] = []
    with open(p, "r", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            rows.append(row)
    return rows


def discover_records_runtime(
    eval_root: str | Path,
    n_per_class: Optional[int] = None,
    strength_min: float = 0.0,
    strength_max: float = 1.0,
    include_classes: Optional[List[str]] = None,
    seed: int = 42,
) -> List[EvalRecord]:
    """Manifest-driven runtime sampler (260506 — replaces subset folders).

    n_per_class:    cap per class_key (None = take all)
    strength_min/max: filter manifest rows by `defect_pixel_ratio` (range 0.0~1.0).
                    Note: Normal/Invalid rows are NOT filtered by strength (they have
                    no meaningful defect ratio). Range applies to 10 defect classes only.
    include_classes: subset of class_keys to keep (None = all 12 ALL_CLASS_KEYS).
    seed:           sampling reproducibility (rng for per-class subsample).
    """
    root = Path(eval_root)
    rows = _read_manifest(root)
    rng = np.random.default_rng(seed)

    if include_classes is None:
        kept_classes = set(ALL_CLASS_KEYS)
    else:
        kept_classes = set(include_classes)

    by_class: Dict[str, List[Dict[str, str]]] = {}
    for row in rows:
        ck = row["class_key"]
        if ck not in kept_classes:
            continue
        # Strength filter applies to 10 defect classes only; Normal/Invalid pass through
        if ck not in ("Normal", "Invalid"):
            try:
                r = float(row.get("defect_pixel_ratio", "0") or "0")
            except ValueError:
                r = 0.0
            if not (strength_min <= r <= strength_max):
                continue
        by_class.setdefault(ck, []).append(row)

    out: List[EvalRecord] = []
    for ck, rs in by_class.items():
        m, is_inv, is_norm = _multihot_for_key(ck)
        if n_per_class is not None and len(rs) > n_per_class:
            idx = rng.choice(len(rs), size=n_per_class, replace=False)
            rs = [rs[i] for i in sorted(idx)]
        for row in rs:
            out.append(EvalRecord(
                chip_path=row["chip_path"],
                class_key=ck,
                true_multihot=m,
                is_invalid_gt=is_inv,
                is_normal_gt=is_norm,
            ))
    return out


class ChipEvalDataset(Dataset):
    """Returns (img_tensor 3xHxW float32, idx int)."""
    def __init__(self, records: List[EvalRecord], img_size: int = 384):
        self.records = records
        self.tf = transforms.Compose([
            transforms.Resize((img_size, img_size), interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ])

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        rec = self.records[idx]
        img = Image.open(rec.chip_path).convert("RGB")
        x = self.tf(img)
        return x, idx


def stratified_val_eval_split(records: List[EvalRecord], val_ratio: float = 0.2,
                              seed: int = 42) -> Tuple[np.ndarray, np.ndarray]:
    """Stratify by class_key. Returns (val_idx, eval_idx) into records list."""
    rng = np.random.default_rng(seed)
    by_cls: dict[str, List[int]] = {}
    for i, r in enumerate(records):
        by_cls.setdefault(r.class_key, []).append(i)
    val_idx: List[int] = []
    eval_idx: List[int] = []
    for cls, idxs in by_cls.items():
        idxs_arr = np.array(idxs)
        rng.shuffle(idxs_arr)
        n_val = max(1, int(round(len(idxs_arr) * val_ratio)))
        n_val = min(n_val, len(idxs_arr) - 1) if len(idxs_arr) > 1 else 0
        val_idx.extend(idxs_arr[:n_val].tolist())
        eval_idx.extend(idxs_arr[n_val:].tolist())
    return np.array(sorted(val_idx)), np.array(sorted(eval_idx))
