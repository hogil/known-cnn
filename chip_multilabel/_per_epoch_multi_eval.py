"""Per-epoch multi-label eval — class별 F1 + class별 FAR.

Loaded once at training start, then called after each epoch.
Computes:
  - per-class bit_F1 for 10 positive classes (4 single + 6 2-combo)
  - per-class FAR for negative classes (Normal / Invalid / 4 OOD wafer-pattern)
  - aggregate bit_F1 (positive macro), Total FAR

Absolute rule 260512:
  - positive = 4 single + 6 2-combo (NO 3-combo)
  - bit_F1 = per-bit (4 defect bit) macro F1 over positive chips
  - per-class F1 = F1 restricted to chips of that class_key
"""
from __future__ import annotations
import csv
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as T


SINGLES = ("bank_boundary", "fork", "scratch", "scratch_rot")
COMBOS_2 = (
    "bank_boundary+fork", "bank_boundary+scratch", "bank_boundary+scratch_rot",
    "fork+scratch", "fork+scratch_rot", "scratch+scratch_rot",
)
POSITIVE = SINGLES + COMBOS_2
NEG_NI = ("Normal", "Invalid")
NEG_OOD = ("CenterDonut", "CrossScratch", "DiagonalSmear", "Starburst")
BITS = SINGLES  # 4 defect bits


def _class_key_to_bits(ck: str) -> np.ndarray:
    h = np.zeros(len(BITS), dtype=np.int8)
    parts = ck.split("+")
    for i, b in enumerate(BITS):
        if b in parts:
            h[i] = 1
    return h


class _MultiEvalDataset(Dataset):
    def __init__(self, paths: List[str], img_size: int):
        self.paths = paths
        self.tf = T.Compose([
            T.Resize((img_size, img_size), interpolation=T.InterpolationMode.BICUBIC),
            T.ToTensor(),
        ])

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        img = Image.open(self.paths[idx]).convert("RGB")
        return self.tf(img), idx


def load_multi_val(eval_root: str, n_per_class: int = 50, seed: int = 42,
                   img_size: int = 384) -> Dict:
    """Sample n_per_class chips per class_key (positive + negative)."""
    root = Path(eval_root)
    manifest = root / "manifest.csv"
    if not manifest.exists():
        raise FileNotFoundError(f"manifest.csv missing at {manifest}")

    rng = np.random.default_rng(seed)
    rows_by_class: Dict[str, List[str]] = {}
    with open(manifest, "r", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            ck = row.get("class_key", "")
            cp = row.get("chip_path", "")
            if not (ck and cp and Path(cp).exists()):
                continue
            rows_by_class.setdefault(ck, []).append(cp)

    all_classes = list(POSITIVE) + list(NEG_NI) + list(NEG_OOD)
    paths, class_keys = [], []
    for ck in all_classes:
        rs = rows_by_class.get(ck, [])
        if not rs:
            continue
        if len(rs) > n_per_class:
            idx = rng.choice(len(rs), size=n_per_class, replace=False)
            rs = [rs[int(i)] for i in sorted(idx)]
        paths.extend(rs)
        class_keys.extend([ck] * len(rs))

    return {
        "paths": paths,
        "class_keys": class_keys,
        "img_size": img_size,
    }


def evaluate(model: torch.nn.Module, val_cache: Dict, device: str,
             threshold: float = 0.5, batch_size: int = 32, num_workers: int = 0) -> Dict:
    """Forward pass + compute per-class F1 / FAR. Returns dict for printing/logging."""
    ds = _MultiEvalDataset(val_cache["paths"], val_cache["img_size"])
    dl = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=num_workers,
                    pin_memory=True)
    model.eval()
    n = len(ds)
    probs = np.zeros((n, len(BITS)), dtype=np.float32)
    with torch.no_grad():
        for x, idx in dl:
            x = x.to(device, non_blocking=True)
            logits = model(x)
            p = torch.sigmoid(logits).float().cpu().numpy()
            for k, i in enumerate(idx.numpy()):
                probs[int(i)] = p[k]
    pred = (probs >= threshold).astype(np.int8)
    class_keys = np.array(val_cache["class_keys"])

    # Build gt multi-hot for positive chips
    pos_mask = np.isin(class_keys, list(POSITIVE))
    neg_mask = np.isin(class_keys, list(NEG_NI) + list(NEG_OOD))
    gt = np.stack([_class_key_to_bits(ck) for ck in class_keys])

    # bit_F1 (positive macro over 4 bits)
    def _f1(g, p):
        tp = int(((g == 1) & (p == 1)).sum())
        fp = int(((g == 0) & (p == 1)).sum())
        fn = int(((g == 1) & (p == 0)).sum())
        if tp + fp == 0 or tp + fn == 0:
            return 0.0
        prec = tp / (tp + fp)
        rec = tp / (tp + fn)
        return 0.0 if (prec + rec) == 0 else 2 * prec * rec / (prec + rec)

    bit_f1_per_bit = []
    for i in range(len(BITS)):
        if pos_mask.sum() > 0:
            bit_f1_per_bit.append(_f1(gt[pos_mask, i], pred[pos_mask, i]))
        else:
            bit_f1_per_bit.append(0.0)
    bit_F1 = float(np.mean(bit_f1_per_bit)) if bit_f1_per_bit else 0.0

    # Per-class F1 (positive classes only) — per-bit macro restricted to that class_key chips
    per_class_f1: Dict[str, float] = {}
    for ck in POSITIVE:
        m = class_keys == ck
        if m.sum() == 0:
            per_class_f1[ck] = float("nan")
            continue
        f1s = []
        for i in range(len(BITS)):
            f1s.append(_f1(gt[m, i], pred[m, i]))
        per_class_f1[ck] = float(np.mean(f1s))

    # Per-class FAR (negative classes — FP if any bit predicted)
    per_class_far: Dict[str, float] = {}
    for ck in list(NEG_NI) + list(NEG_OOD):
        m = class_keys == ck
        if m.sum() == 0:
            per_class_far[ck] = float("nan")
            continue
        fp = int(pred[m].any(axis=1).sum())
        per_class_far[ck] = 100.0 * fp / m.sum()

    # Aggregate FAR groups
    total_far = ni_far = ood_far = 0.0
    if neg_mask.sum() > 0:
        total_far = 100.0 * int(pred[neg_mask].any(axis=1).sum()) / neg_mask.sum()
    ni_m = np.isin(class_keys, list(NEG_NI))
    if ni_m.sum() > 0:
        ni_far = 100.0 * int(pred[ni_m].any(axis=1).sum()) / ni_m.sum()
    ood_m = np.isin(class_keys, list(NEG_OOD))
    if ood_m.sum() > 0:
        ood_far = 100.0 * int(pred[ood_m].any(axis=1).sum()) / ood_m.sum()

    return {
        "bit_F1": bit_F1,
        "total_far": total_far,
        "ni_far": ni_far,
        "ood_far": ood_far,
        "per_class_f1": per_class_f1,
        "per_class_far": per_class_far,
        "per_bit_f1": dict(zip(BITS, bit_f1_per_bit)),
        "n_pos": int(pos_mask.sum()),
        "n_neg": int(neg_mask.sum()),
    }


def format_compact(m: Dict) -> str:
    """One-line compact summary for [ep NN] printout."""
    pcf = m["per_class_f1"]
    pcfar = m["per_class_far"]
    # 10 positive classes short codes
    short = {"bank_boundary": "bb", "fork": "fk", "scratch": "sc", "scratch_rot": "sr",
             "bank_boundary+fork": "bb+fk", "bank_boundary+scratch": "bb+sc",
             "bank_boundary+scratch_rot": "bb+sr", "fork+scratch": "fk+sc",
             "fork+scratch_rot": "fk+sr", "scratch+scratch_rot": "sc+sr",
             "CenterDonut": "CD", "CrossScratch": "CS", "DiagonalSmear": "DS", "Starburst": "ST"}
    pos_str = " ".join(f"{short[c]}={pcf[c]:.3f}" for c in POSITIVE if c in pcf)
    far_str = " ".join(f"{short[c]}={pcfar[c]:.1f}" for c in (NEG_NI + NEG_OOD) if c in pcfar)
    return (f"bit_F1={m['bit_F1']:.4f} FAR={m['total_far']:.2f}% "
            f"NI={m['ni_far']:.2f}% OOD={m['ood_far']:.2f}%\n"
            f"           pos_F1: {pos_str}\n"
            f"           neg_FAR(%): {far_str}")
