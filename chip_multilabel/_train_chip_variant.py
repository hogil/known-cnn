# -*- coding: utf-8 -*-
"""Train one Stage 2 chip 4-class CNN variant.

Variants:
- T1: CE + label_smoothing 0.1
- T4: ASL (gamma_pos=1, gamma_neg=4, clip=0.05) on one-hot multi-hot target
- T5: BCE on one-hot multi-hot target
- T6: BCE warmup 5ep -> ASL

Init backbone state_dict from existing chip5_round4_v14 best_model.pth (TAPT).
Drops invalid_main / particle_blast / scratch_21deg from training set.
"""
from __future__ import annotations

import sys
try:
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
except Exception:
    pass

import argparse
import json
import os
import shutil
import time
from datetime import datetime
from pathlib import Path
from typing import List, Tuple

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("HF_DATASETS_OFFLINE", "1")
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")

import numpy as np
import timm
import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

try:
    # Use tqdm.std explicitly to avoid tqdm.auto's IPython/notebook detection
    # (which can stall on Windows + headless env when probing ipykernel).
    from tqdm.std import tqdm as _tqdm
    _HAS_TQDM = True
except Exception:
    _HAS_TQDM = False

def _is_main_rank():
    try:
        import torch.distributed as _d
        return (not _d.is_initialized()) or _d.get_rank() == 0
    except Exception:
        return True

def _maybe_tqdm(iterable, **kw):
    if _HAS_TQDM and _is_main_rank():
        return _tqdm(iterable, **kw)
    return iterable


def _atomic_torch_save(payload, path: Path) -> None:
    """Write torch checkpoint without exposing a partial final file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    try:
        torch.save(payload, tmp)
        os.replace(tmp, path)
    except Exception as e:
        try:
            usage = shutil.disk_usage(path.parent)
            free_gb = usage.free / (1024 ** 3)
            total_gb = usage.total / (1024 ** 3)
            print(f"[save] FAIL path={path} tmp={tmp} "
                  f"disk_free={free_gb:.1f}GB/{total_gb:.1f}GB "
                  f"error={type(e).__name__}: {e}", flush=True)
        except Exception:
            print(f"[save] FAIL path={path} tmp={tmp} "
                  f"error={type(e).__name__}: {e}", flush=True)
        raise
    finally:
        try:
            if tmp.exists():
                tmp.unlink()
        except OSError:
            pass


from .constants import DEFAULT_BACKBONE_CKPT, DEFAULT_CLASSIFICATION_CHIPS, TRAIN_CLASSES, TRAIN_VARIANTS
from . import losses as _losses

build_loss = getattr(_losses, "build_loss", None)
if build_loss is None:
    _losses_file = getattr(_losses, "__file__", "<unknown>")
    _available = [name for name in (
        "AsymmetricLoss", "BCEMultiHot", "BCEThenASL", "CEWithSmoothing",
        "CESoftLabel", "FocalLoss", "SigmoidFocalLoss", "build_loss",
    ) if hasattr(_losses, name)]
    raise ImportError(
        f"chip_multilabel.losses has no build_loss. Loaded losses module from "
        f"{_losses_file}; available symbols={_available}. Make sure the server "
        "pulled the latest repo and PYTHONPATH points to this checkout."
    )

VARIANT_TO_LOSS = {
    "T0": "ce_ls01",  # iter 12 (260506) — pure CE baseline alias (use with --ls 0.0 + --cutmix-p 0)
    "T1": "ce_ls01",
    "T3": "focal",
    "T4": "asl",
    "T5": "bce",
    "T6": "bce_then_asl",
    "T7": "bce_ls",  # CutMix-friendly BCE + label smoothing
    "T8": "ce_soft_ls",  # CE + LS + CutMix via soft KL target
    "T9": "sigmoid_focal",  # iter 12 (260506) — RetinaNet sigmoid focal for multi-label
}

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


class ChipFolderDataset(Dataset):
    def __init__(self, samples: List[Tuple[Path, int]], img_size: int, train: bool,
                 teacher_prob_map: dict | None = None):
        self.samples = samples
        # 260509 — KD support. Maps str(path) -> np.ndarray(4,) teacher sigmoid probs.
        # If None or path missing, yields zero-vector (loop skips KD term via flag check).
        self.teacher_prob_map = teacher_prob_map or {}
        if train:
            # NOTE: Rotation + Flip 영구 제거 (사용자 directive 260505) —
            # scratch ↔ scratch_rot 두 class 가 회전/반사 으로 구분되므로 rotation
            # 또는 flip augmentation 이 두 class 를 혼동시킴 (예: scratch flipped vertical
            # → scratch_rot-like). Affine (translate/scale, no rotation no flip) 만 사용.
            self.tf = transforms.Compose([
                transforms.Resize((img_size, img_size), interpolation=transforms.InterpolationMode.BILINEAR),
                transforms.RandomAffine(
                    degrees=0,  # NO rotation
                    translate=(0.05, 0.05),  # ±5% translate
                    scale=(0.95, 1.05),  # ±5% scale
                    fill=255,  # white background
                ),
                transforms.ToTensor(),
                transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
            ])
        else:
            self.tf = transforms.Compose([
                transforms.Resize((img_size, img_size), interpolation=transforms.InterpolationMode.BILINEAR),
                transforms.ToTensor(),
                transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
            ])

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, i: int):
        # 260508: samples is List[Tuple[Path, int, np.ndarray(4,)]] — y_int + multihot
        # 260509: dataset returns 4-tuple (x, y, mh, teacher_prob).  teacher_prob is
        # zero-vector when no map / chip_path missing — loop checks args.kd_teacher_probs flag.
        rec = self.samples[i]
        if len(rec) == 3:
            p, y, mh = rec
            img = Image.open(p).convert("RGB")
            mh_t = torch.from_numpy(mh.astype(np.float32))
        else:
            p, y = rec  # backward compat (shouldn't happen post-260508)
            img = Image.open(p).convert("RGB")
            mh_t = torch.zeros(len(TRAIN_CLASSES), dtype=torch.float32)
        tp_arr = self.teacher_prob_map.get(str(p))
        if tp_arr is None:
            tp_t = torch.zeros(len(TRAIN_CLASSES), dtype=torch.float32)
        else:
            tp_t = torch.from_numpy(tp_arr.astype(np.float32))
        return self.tf(img), y, mh_t, tp_t


NORMAL_SENTINEL = -1  # 260506 — Normal class y label (multi-hot target = all zeros)
MULTI_COMBO_SENTINEL = -2  # 260508 — multi-combo (2/3-class) chip; multi-hot target from folder name


def _multihot_from_combo_name(name: str) -> np.ndarray:
    """'fork+scratch' → [0,1,1,0]. Order = TRAIN_CLASSES."""
    mh = np.zeros(len(TRAIN_CLASSES), dtype=np.uint8)
    for tok in name.split("+"):
        if tok in TRAIN_CLASSES:
            mh[TRAIN_CLASSES.index(tok)] = 1
        else:
            raise ValueError(f"unknown class token '{tok}' in combo name '{name}'")
    return mh


def collect_samples(root: Path, include_normal: bool = True,
                    multi_combo_root: Path | None = None,
                    multi_combo_n_per_class: int | None = None,
                    max_per_class_defect: int | None = None,
                    rng_seed: int = 42) -> List[Tuple[Path, int, np.ndarray]]:
    """Collect train samples.

    260508: returns List of (path, y_int, multihot 4-d).
    - single class: y = TRAIN_CLASSES.index(cname), multihot = one-hot
    - Normal: y = NORMAL_SENTINEL (-1), multihot = [0,0,0,0]
    - multi-combo (NEW): y = MULTI_COMBO_SENTINEL (-2), multihot = parsed from folder name

    260512: max_per_class_defect — limit defect class chip count (low-data ablation).
    """
    out: List[Tuple[Path, int, np.ndarray]] = []
    rng_def = np.random.default_rng(rng_seed)
    for ci, cname in enumerate(TRAIN_CLASSES):
        d = root / cname
        if not d.exists():
            raise FileNotFoundError(f"missing class dir: {d}")
        mh = np.zeros(len(TRAIN_CLASSES), dtype=np.uint8)
        mh[ci] = 1
        files = sorted(d.glob("*.png"))
        if max_per_class_defect is not None and len(files) > max_per_class_defect:
            idx = rng_def.choice(len(files), size=max_per_class_defect, replace=False)
            files = [files[i] for i in sorted(idx)]
        for png in files:
            out.append((png, ci, mh.copy()))
    if include_normal:
        nd = root / "Normal"
        if nd.exists():
            n_normal = 0
            mh_zero = np.zeros(len(TRAIN_CLASSES), dtype=np.uint8)
            for png in sorted(nd.glob("*.png")):
                out.append((png, NORMAL_SENTINEL, mh_zero.copy()))
                n_normal += 1
            if n_normal > 0:
                print(f"[data] including {n_normal} Normal chips (y=-1 sentinel, multi-hot target [0,0,0,0])")
    # 260508 — multi-combo master folder (10 combo × N chip)
    if multi_combo_root is not None:
        if not multi_combo_root.exists():
            raise FileNotFoundError(f"multi-combo root not found: {multi_combo_root}")
        rng = np.random.default_rng(rng_seed)
        n_total = 0
        for cdir in sorted(multi_combo_root.iterdir()):
            if not cdir.is_dir() or cdir.name.startswith("_"):
                continue
            if "+" not in cdir.name:
                continue   # not a combo folder
            try:
                mh = _multihot_from_combo_name(cdir.name)
            except ValueError as e:
                print(f"[data] WARN skipping {cdir.name}: {e}")
                continue
            files = sorted(cdir.glob("*.png"))
            if multi_combo_n_per_class is not None and len(files) > multi_combo_n_per_class:
                idx = rng.choice(len(files), size=multi_combo_n_per_class, replace=False)
                files = [files[i] for i in sorted(idx)]
            for png in files:
                out.append((png, MULTI_COMBO_SENTINEL, mh.copy()))
                n_total += 1
        print(f"[data] including {n_total} multi-combo chips from {multi_combo_root} "
              f"(y=-2 sentinel, multi-hot from folder name)")
    return out


def stratified_split(samples: List[Tuple[Path, int, np.ndarray]],
                     val_ratio: float = 0.2, seed: int = 42):
    """Stratify by combined (y_int, multi-hot tuple) — multi-combo classes split per-combo."""
    by = {}
    for rec in samples:
        p = rec[0]
        y = rec[1]
        mh = rec[2] if len(rec) > 2 else np.zeros(len(TRAIN_CLASSES), dtype=np.uint8)
        # stratify key = y for single + Normal, mh tuple for multi-combo
        key = y if y >= -1 else tuple(mh.tolist())
        by.setdefault(key, []).append((p, y, mh))
    rng = np.random.default_rng(seed)
    train, val = [], []
    for key, recs in by.items():
        idx = list(range(len(recs)))
        rng.shuffle(idx)
        n_val = max(1, int(round(len(recs) * val_ratio)))
        for i in idx[:n_val]:
            val.append(recs[i])
        for i in idx[n_val:]:
            train.append(recs[i])
    return train, val


class ModelEMA:
    """EMA with dynamic decay warmup. Ported from D:/project/anomaly-detection/train.py:214-253.

    decay_t = min(target_decay, (1 + step) / (10 + step))
    """
    def __init__(self, model: torch.nn.Module, decay: float = 0.95):
        import copy
        self.target_decay = float(decay)
        self.module = copy.deepcopy(model).eval()
        for p in self.module.parameters():
            p.requires_grad_(False)
        self.num_updates = 0

    @torch.no_grad()
    def update(self, model: torch.nn.Module) -> None:
        self.num_updates += 1
        decay_t = min(self.target_decay, (1.0 + self.num_updates) / (10.0 + self.num_updates))
        _src = model.module if hasattr(model, "module") else model
        for ema_v, v in zip(self.module.state_dict().values(), _src.state_dict().values()):
            if ema_v.dtype.is_floating_point:
                ema_v.mul_(decay_t).add_(v.detach().to(ema_v.dtype), alpha=1.0 - decay_t)
            else:
                ema_v.copy_(v)


def _load_local_timm_weights(model: torch.nn.Module, wpath: Path) -> Tuple[int, int, int]:
    """Load a local timm backbone state_dict without any HF/timm network path."""
    if wpath.suffix != ".pth":
        raise ValueError("backbone_timm_weights must be a .pth file. Run mega_matrix/download.py first.")
    sd = torch.load(str(wpath), map_location="cpu", weights_only=False)
    if isinstance(sd, dict):
        for key in ("state_dict", "model", "model_state_dict"):
            if key in sd and hasattr(sd[key], "keys"):
                sd = sd[key]
                break
    if not hasattr(sd, "keys"):
        raise ValueError(f"local weight file is not a state_dict: {wpath}")

    msd = model.state_dict()
    compat = {}
    for k, v in sd.items():
        kk = k
        for prefix in ("module.", "model."):
            if kk.startswith(prefix):
                kk = kk[len(prefix):]
        if kk in msd and hasattr(v, "shape") and msd[kk].shape == v.shape:
            compat[kk] = v
    if not compat:
        raise ValueError(f"no compatible keys loaded from {wpath}")
    model.load_state_dict(compat, strict=False)
    return len(compat), len(sd) - len(compat), len(sd)


def build_model(num_classes: int, init_ckpt: Path,
                drop_path_rate: float = 0.0,
                backbone_timm: str = None,
                img_size_override: int = None,
                backbone_timm_weights: str = None) -> Tuple[torch.nn.Module, str, int]:
    # Mode A: timm backbone from an explicit local weight file only.
    if backbone_timm:
        img_size = int(img_size_override) if img_size_override else 224
        if not backbone_timm_weights:
            raise RuntimeError(
                "--backbone-timm requires --backbone-timm-weights. "
                "HF/timm auto-download is forbidden in closed-network mode."
            )
        wpath = Path(backbone_timm_weights)
        if not wpath.is_file():
            raise FileNotFoundError(f"--backbone-timm-weights not found: {wpath}")
        model = timm.create_model(backbone_timm, pretrained=False,
                                  num_classes=num_classes,
                                  drop_path_rate=drop_path_rate)
        loaded, skipped, total = _load_local_timm_weights(model, wpath)
        print(f"[init] backbone={backbone_timm} loaded local weights {wpath} "
              f"{loaded}/{total} keys (skipped={skipped}), img_size={img_size}")
        return model, backbone_timm, img_size
    # Mode B: local ckpt (default ConvNeXtV2-base path)
    ckpt = torch.load(init_ckpt, map_location="cpu", weights_only=False)
    backbone = ckpt["backbone"]
    img_size = int(ckpt["img_size"])
    model = timm.create_model(backbone, pretrained=False, num_classes=num_classes,
                              drop_path_rate=drop_path_rate)
    sd = ckpt["model"]
    msd = model.state_dict()
    compat = {k: v for k, v in sd.items() if k in msd and msd[k].shape == v.shape}
    skipped = len(sd) - len(compat)
    model.load_state_dict(compat, strict=False)
    print(f"[init] backbone={backbone} loaded {len(compat)}/{len(sd)} keys (skipped={skipped})")
    return model, backbone, img_size


@torch.no_grad()
def evaluate(model, loader, device, num_classes: int):
    """Returns dict {val_acc, val_f1, val_auroc, val_loss} for selection criterion options.
    260512 (user): val_acc on 4-way argmax has only ~2 reachable values (160/163, 161/163)
    on tiny val split → coin-flip best selection. Switch to val_f1 macro (per-bit F1 average,
    continuous spectrum) or val_auroc macro (threshold-independent) for proper selection.
    """
    import numpy as _np
    from sklearn.metrics import f1_score as _f1_score, roc_auc_score as _roc_auc_score
    model.eval()
    correct = 0
    total = 0
    correct_normal = 0
    total_normal = 0
    use_amp = device.type == "cuda"
    NORMAL_MAX_PROB = 0.5
    all_y_int = []   # for legacy val_acc compat
    all_mh = []      # multi-hot GT (defect: one-hot, Normal: zeros)
    all_probs = []   # sigmoid logits per-bit
    for batch in _maybe_tqdm(loader, total=len(loader), desc="eval", leave=False, dynamic_ncols=True):
        if len(batch) == 4:
            x, y, mh, _tp = batch
            mh = mh.to(device, non_blocking=True)
        elif len(batch) == 3:
            x, y, mh = batch
            mh = mh.to(device, non_blocking=True)
        else:
            x, y = batch
            mh = None
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        with torch.amp.autocast(device_type=device.type, enabled=use_amp):
            logits = model(x)
        probs = torch.sigmoid(logits)
        defect_mask = (y >= 0)
        if defect_mask.any():
            pred = logits[defect_mask].argmax(dim=1)
            correct += int((pred == y[defect_mask]).sum())
            total += int(defect_mask.sum())
        if mh is not None:
            multi_mask = (y == MULTI_COMBO_SENTINEL)
            if multi_mask.any():
                probs_m = probs[multi_mask]
                pred_bin = (probs_m >= 0.5).float()
                exact = (pred_bin == mh[multi_mask]).all(dim=1)
                correct += int(exact.sum())
                total += int(multi_mask.sum())
        normal_mask = (y == NORMAL_SENTINEL)
        if normal_mask.any():
            probs_n = probs[normal_mask]
            max_p = probs_n.max(dim=1).values
            correct_normal += int((max_p < NORMAL_MAX_PROB).sum())
            total_normal += int(normal_mask.sum())
        # Accumulate per-bit prob + multi-hot GT for F1/AUROC
        # GT mh: provided in 3/4-tuple. For defect single-label, mh is one-hot.
        if mh is not None:
            all_mh.append(mh.detach().cpu().numpy())
            all_probs.append(probs.detach().cpu().numpy())
        all_y_int.append(y.detach().cpu().numpy())
    # val_acc (legacy)
    defect_acc = correct / max(total, 1)
    normal_acc = correct_normal / max(total_normal, 1) if total_normal > 0 else None
    if normal_acc is not None:
        val_acc = (correct + correct_normal) / max(total + total_normal, 1)
    else:
        val_acc = defect_acc
    # val_f1 macro + val_auroc macro (per-bit, defect samples only — Normal y=-1 has zero mh)
    val_f1 = float('nan')
    val_auroc = float('nan')
    if all_mh:
        mh_all = _np.concatenate(all_mh, axis=0)   # (N, num_classes)
        probs_all = _np.concatenate(all_probs, axis=0)
        # mask: defect samples only (mh has at least one 1)
        defect_idx = mh_all.sum(axis=1) > 0
        if defect_idx.sum() > 0:
            mh_d = mh_all[defect_idx]
            probs_d = probs_all[defect_idx]
            pred_d = (probs_d >= 0.5).astype(_np.int32)
            try:
                val_f1 = float(_f1_score(mh_d, pred_d, average='macro', zero_division=0))
            except Exception:
                val_f1 = float('nan')
            try:
                # AUROC: needs at least 1 positive + 1 negative per bit; defect-only may be all-positive
                # → use ALL samples (defect + Normal). Normal mh=zero gives negatives.
                val_auroc = float(_roc_auc_score(mh_all, probs_all, average='macro'))
            except Exception:
                val_auroc = float('nan')
    # 260512 night — analyst opus metric ideas (saturation-robust)
    val_bce = float('nan'); val_brier = float('nan')
    val_margin = float('nan'); val_f1_best_tau = float('nan')
    if all_mh and defect_idx.sum() > 0:
        eps = 1e-6
        p_clip = _np.clip(probs_d, eps, 1.0 - eps)
        # A1: val BCE loss (Cole 2021 SPML — continuous, anti-saturation)
        val_bce = float(-(mh_d * _np.log(p_clip) + (1 - mh_d) * _np.log(1 - p_clip)).mean())
        # A3: Brier score (proper scoring rule)
        val_brier = float(((p_clip - mh_d) ** 2).mean())
        # A4: confidence margin (positive prob - max negative prob)
        try:
            pos_score = (probs_d * mh_d).sum(axis=1) / _np.maximum(mh_d.sum(axis=1), 1)
            # max negative bit prob per sample
            neg_mask = (1 - mh_d).astype(bool)
            neg_score = _np.where(neg_mask, probs_d, -_np.inf).max(axis=1)
            val_margin = float((pos_score - neg_score).mean())
        except Exception:
            val_margin = float('nan')
        # A6: best-τ F1 sweep [0.3, 0.7] (decouples threshold artifact)
        try:
            f1_at_tau = []
            for t in _np.linspace(0.3, 0.7, 9):
                pred_t = (probs_d >= t).astype(_np.int32)
                f1_at_tau.append(_f1_score(mh_d, pred_t, average='macro', zero_division=0))
            val_f1_best_tau = float(max(f1_at_tau))
        except Exception:
            val_f1_best_tau = float('nan')
    return {"val_acc": float(val_acc), "val_f1": val_f1, "val_auroc": val_auroc,
            "val_bce": val_bce, "val_brier": val_brier,
            "val_margin": val_margin, "val_f1_best_tau": val_f1_best_tau}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", required=True, choices=list(VARIANT_TO_LOSS.keys()))
    ap.add_argument("--ls", type=float, default=None,
                    help="label smoothing (T1 only). default 0.1 if not set.")
    ap.add_argument("--tag", type=str, default="",
                    help="optional tag suffix for out_dir name")
    ap.add_argument("--data-root", default=DEFAULT_CLASSIFICATION_CHIPS)
    ap.add_argument("--init-ckpt", default=DEFAULT_BACKBONE_CKPT)
    ap.add_argument("--backbone-timm", type=str, default=None,
                    help="If set, use timm.create_model with explicit local "
                         "--backbone-timm-weights instead of --init-ckpt. "
                         "HF/timm auto-download is forbidden.")
    ap.add_argument("--backbone-timm-weights", type=str, default=None,
                    help="Required local .pth weight file for "
                         "--backbone-timm. Copy mega_matrix/weights/ to the server first, "
                         "then pass --backbone-timm-weights mega_matrix/weights/<file>.")
    ap.add_argument("--img-size", type=int, default=None,
                    help="Override img_size for --backbone-timm mode (default 224).")
    ap.add_argument("--out-root", default="outputs/logs_chip_multilabel")
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--best-from-epoch", type=int, default=0,
                    help="Only consider epochs >= this for best_model.pth selection. "
                         "Default 0 = save best from any epoch. Use 5 to skip ep1-4 "
                         "(rationale: backbone with quick val_acc plateau often picks "
                         "early under-trained ckpt; multi-label perf usually peaks later).")
    ap.add_argument("--save-every-epoch", action="store_true",
                    help="Save epoch_NN_model.pth at every epoch end (in addition to "
                         "best_model.pth). Used for per-epoch eval / multi-label selection. "
                         "Disk cost: ~350 MB × epochs (cleanup recommended after eval).")
    ap.add_argument("--multi-val-set", type=str, default=None,
                    help="Optional path to multi-label eval folder (with manifest.csv) "
                         "for per-epoch class-level F1 + FAR reporting. e.g. "
                         "E:/data/images/chip_multilabel_v15direct_n2000")
    ap.add_argument("--multi-val-n-per-class", type=int, default=50,
                    help="Per-class chip cap for --multi-val-set (default 50 = fast).")
    ap.add_argument("--val-criterion",
                    choices=["acc", "f1", "auroc", "bce_min", "brier_min",
                             "margin_max", "f1_best_tau"],
                    default="acc",
                    help="Which val metric drives best_model.pth selection. "
                         "260512 user: val_acc broken (2 values). "
                         "f1/auroc: existing patches. "
                         "260512 night analyst (Cole 2021): bce_min, brier_min, margin_max, "
                         "f1_best_tau — saturation-robust proper scoring rules. "
                         "_min variants pick smallest value (sign-flipped in selection logic).")
    ap.add_argument("--freeze-backbone", action="store_true",
                    help="Linear probe mode: freeze all backbone params, train only "
                         "classifier head. Tests pretrained representation quality on chip "
                         "domain. Standard SSL paper baseline (FCMAE / DINOv3 / Swin V1 all "
                         "report linear probe). 260512 user directive.")
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--accum", type=int, default=4)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--num-workers", type=int, default=0)
    ap.add_argument("--device", default=None)
    ap.add_argument("--gpu-mem-fraction", type=float, default=None,
                    help="Cap process GPU memory usage as fraction of total VRAM (e.g. 0.4 = 40%%). "
                         "Local-machine rule (260514): set 0.4 to leave room for other work.")
    ap.add_argument("--grad-checkpointing", action="store_true",
                    help="Enable gradient checkpointing (timm set_grad_checkpointing). "
                         "Reduces activation memory ~30%% at ~20%% slower step. "
                         "Use when batch>=2 needed for CutMix but cap 0.40 OOMs (260515).")
    # Phase F additions (anomaly-detection BKM ported)
    ap.add_argument("--warmup-epochs", type=int, default=0,
                    help="LinearLR warmup epochs (0 = no warmup, current default)")
    ap.add_argument("--warmup-start-factor", type=float, default=0.05,
                    help="LinearLR warmup start_factor (anomaly-detection BKM uses 0.05)")
    ap.add_argument("--lr-eta-min", type=float, default=0.0,
                    help="CosineAnnealing eta_min (anomaly-detection BKM uses 1e-6)")
    ap.add_argument("--lr-backbone", type=float, default=None,
                    help="separate backbone LR (None = use --lr for everything)")
    ap.add_argument("--lr-head", type=float, default=None,
                    help="separate head LR (None = use --lr for everything)")
    ap.add_argument("--ema-decay", type=float, default=0.0,
                    help="EMA target decay (0 = off; anomaly-detection BKM uses 0.95)")
    ap.add_argument("--grad-clip", type=float, default=1.0,
                    help="gradient clip max_norm (anomaly-detection BKM 0.5)")
    ap.add_argument("--drop-path-rate", type=float, default=0.0,
                    help="stochastic depth drop_path_rate (anomaly-detection BKM 0.05)")
    ap.add_argument("--no-normal", action="store_true",
                    help="Skip classification_chips/Normal/ folder. 4-class defect only "
                         "(traditional baseline / ablation). Default: include Normal as y=-1 sentinel.")
    # 260508 — multi-combo training data (iter 17B)
    ap.add_argument("--multi-combo-root", type=str, default="",
                    help="Path to multi-combo master folder (e.g. "
                         "data/wm-811k/chip_multilabel_synth). 10 combo subdirs (2+3-class). "
                         "Empty = off. Adds multi-hot supervision for compositional generalization (iter 17B).")
    ap.add_argument("--multi-combo-n-per-class", type=int, default=50,
                    help="N chip per multi-combo class (10 combo). default 50.")
    ap.add_argument("--cutmix-pair-bias", type=str, default="",
                    help="Bias CutMix pair sampling. Format 'C1,C2:K' — fork↔scratch pair "
                         "K× more likely than uniform. e.g., 'fork,scratch:2' → ~67%% of CutMix "
                         "events become fork↔scratch (vs uniform ~17%%).")
    ap.add_argument("--cutmix-p", type=float, default=0.0,
                    help="probability of applying CutMix on a batch (T7 only). 0 = off.")
    ap.add_argument("--cutmix-rect", type=float, default=0.5,
                    help="rectangle area fraction for CutMix paste (default 0.5 = half-image).")
    # iter 12 (260506) — scattered CutMix + soft proportional label
    ap.add_argument("--cutmix-grid-dim", type=int, default=8,
                    help="Spatial grid dimension for complement/grid_complete cutmix "
                         "(GRID x GRID cells, each cell H/GRID x W/GRID px). "
                         "n_groups partitions the GRID*GRID cells. Sweep this to vary "
                         "per-cell pixel size at fixed n_groups (paper §6 NEW axis). "
                         "Default 8 (= legacy 8x8 = 48x48 px cells at img_size=384). "
                         "GRID must be >= n_groups. Suggested sweep: {4, 6, 8, 12, 16}.")
    ap.add_argument("--cutmix-mode", type=str, default="single",
                    choices=["single", "scattered", "grid", "grid_sparse",
                             "grid_complete", "complement", "mixup",
                             "bisect_h", "bisect_v", "bisect_rand"],
                    help="CutMix mode. 'single' (default) = original random rectangle "
                         "(OR multi-hot label or soft λ for CE-soft). "
                         "'scattered' = N small patches + soft proportional label "
                         "label_B = ratio × discount × alpha (iter 12, paper-quality). "
                         "'grid' = 8×8 binary mask (50%% prob per cell) — fine-grained "
                         "spatial mixing, area-proportional soft label (260507). "
                         "'grid_sparse' (260508) = K random cells from 8×8 grid "
                         "(grid + scatter mix). soft proportional label. "
                         "'grid_complete' (260508) = deterministic 50/50 split — "
                         "32 cells of 64 = chip A, other 32 = chip B. "
                         "FULL chip cover, no white space, label exactly [0.5, 0.5]. "
                         "Fixes the 'paste rect maybe whitespace, not defect' problem.")
    ap.add_argument("--cutmix-grid-k", type=int, default=8,
                    help="Number of cells to flip for grid_sparse mode (default 8 of 64). "
                         "Scattered patches but grid-aligned. Used only when "
                         "--cutmix-mode=grid_sparse.")
    ap.add_argument("--cutmix-complete-label-scale", type=float, default=0.5,
                    help="Label scale for grid_complete / complement mode (260508). "
                         "0.5 = soft proportional (area-based), "
                         "0.75 = intermediate, "
                         "1.0 = hard (both defects fully present, justified by complete cover). "
                         "Used only when --cutmix-mode=grid_complete OR complement + multi_hot target.")
    ap.add_argument("--cutmix-n-groups", type=int, default=2,
                    help="N groups for complement mode (260508). 2/3/4. "
                         "1 pair (chip A + B) → N mix chips + N mask chips (paired) = 2N samples. "
                         "All cells of A and B distributed across N mix chips (no info loss). "
                         "Used only when --cutmix-mode=complement.")
    ap.add_argument("--cutmix-label-area-prop", action="store_true",
                    help="Use area-proportional label for complement mode: "
                         "A=a_frac*scale, B=b_frac*scale (vs default symmetric A=B=scale). "
                         "a_frac=1/n_groups, b_frac=(n_groups-1)/n_groups. paper §6.13 ablation.")
    ap.add_argument("--cutmix-ab-labels", type=str, default="",
                    help="Explicit (A_label, B_label) for complement mode. Format 'A,B' "
                         "e.g., '1.0,0.5' (A hard, B half). Overrides --cutmix-complete-label-scale "
                         "and --cutmix-label-area-prop. paper §6.16 asymmetric axis.")
    ap.add_argument("--cutmix-other-label", type=float, default=0.0,
                    help="Label value for off-class bits in mix chip (260512). Default 0 (hard zero). "
                         "Non-zero = label smoothing for 'neutral' bits — chip mix 의 A/B class 가 아닌 "
                         "2 bits 에 적용. e.g., 0.1 → 'soft uncertain' for off-class.")
    ap.add_argument("--cutmix-n-patches", type=int, default=5,
                    help="Number of small patches for scattered mode (default 5). "
                         "Used only when --cutmix-mode=scattered.")
    ap.add_argument("--cutmix-total-ratio", type=float, default=0.3,
                    help="Total paste area fraction for scattered mode. 0.3 = 30%% of image. "
                         "Sweepable axis #1. Used only when --cutmix-mode=scattered.")
    ap.add_argument("--cutmix-discount", type=float, default=0.7,
                    help="Soft label discount factor (accounts for non-defect pixels in paste). "
                         "Fixed at 0.7 per user directive 260506. label_B = ratio × discount × α.")
    ap.add_argument("--cutmix-alpha", type=float, default=1.0,
                    help="Additional soft label scale for scattered mode. Sweepable axis #2. "
                         "label_B = total_ratio × discount × alpha. e.g., r=0.3, α=1.0 → 0.21.")
    # 260507 — grid mode cell-flip probability (sweepable)
    ap.add_argument("--cutmix-grid-prob", type=float, default=0.5,
                    help="Per-cell flip probability for grid CutMix mode (default 0.5). "
                         "0.5 ≈ 50%% area mixed (32/64 cells), 0.25 ≈ 25%% (16/64), "
                         "0.125 ≈ 12.5%% (8/64). Used only when --cutmix-mode=grid.")
    # 260508 — paired CutMix: 동일 rect 영역을 mask 한 paired sample 도 forward.
    # disentangle "mask location prior" from "actual content signal" (counterfactual augmentation).
    ap.add_argument("--cutmix-pair", type=str, default="none", choices=["none", "masked", "masked2"],
                    help="If 'masked', for each CutMix sample also produce a paired sample with "
                         "same rect masked (chip B 무관, fill=corner mean). Both forwarded; "
                         "loss = loss_mix + w * loss_masked. Forces model to read content "
                         "rather than rely on mask-location prior. Single mode only. "
                         "If 'masked2' (260515), also produce a SECOND paired sample with the same "
                         "rect masked on chip B (the cutmix paste source) → label = B only. "
                         "Total 3 forwards per cutmix: mix + maskA + maskB. Trains model to be "
                         "rect-position invariant for both A and B chips. Single mode only.")
    ap.add_argument("--cutmix-pair-loss-w", type=float, default=1.0,
                    help="Loss weight for the masked pair forward. default 1.0 (equal weight).")
    # 260515 — mask sample target override (decouples mask target from main LS smoothing)
    ap.add_argument("--cutmix-mask-pos-target", type=float, default=None,
                    help="Override positive-bit target for mask A / mask B paired samples "
                         "(default: use main loss LS / pos-target). Recommended: 0.7 to weaken "
                         "mask-sample positive pull, reducing OOD over-activation.")
    ap.add_argument("--cutmix-mask-neg-target", type=float, default=None,
                    help="Override negative-bit target for mask paired samples (default: use main).")
    # 260515 — raw target for mix sample (bypass LS / pos/neg smoothing, use mix_t as-is)
    ap.add_argument("--cutmix-mix-raw-target", action="store_true",
                    help="When set, cutmix MIX sample uses a raw BCE loss (no LS, no pos/neg "
                         "rewriting). The mix_t built by --cutmix-ab-labels + --cutmix-other-label "
                         "passes through unchanged. Used for asymmetric main/sub experiments where "
                         "explicit target values (e.g., A=0.9, B=0.7, C/D=0.15) must be preserved.")
    ap.add_argument("--cutmix-pair-fill", type=str, default="corner",
                    choices=["corner", "white", "noise"],
                    help="Fill for the masked rect. 'corner' = chip's own top-left 8×8 mean "
                         "(natural background match). 'white' = palette grade-0 hardcoded "
                         "(post-norm ~2.25). 'noise' = gaussian random.")
    ap.add_argument("--pos-target", type=float, default=None,
                    help="260513: independent positive bit target (e.g., 0.9). "
                         "If set with --neg-target, overrides --ls. "
                         "Decouples symmetric LS into independent per-class targets.")
    ap.add_argument("--neg-target", type=float, default=None,
                    help="260513: independent negative bit target (e.g., 0.1). "
                         "Pairs with --pos-target.")
    ap.add_argument("--pos-targets-asy", type=str, default=None,
                    help="260513: ASYMMETRIC per-bit pos targets, comma-separated 4 values "
                         "in TRAIN_CLASSES order (bank_boundary, fork, scratch, scratch_rot). "
                         "e.g. '1.0,0.9,0.95,0.9'. Overrides --pos-target.")
    ap.add_argument("--neg-targets-asy", type=str, default=None,
                    help="260513: ASYMMETRIC per-bit neg targets, comma-separated 4 values "
                         "in TRAIN_CLASSES order. e.g. '0.05,0.2,0.1,0.15'. Overrides --neg-target.")
    ap.add_argument("--bce-temperature", type=float, default=1.0,
                    help="260513: training-time temperature for BCE. logit/T applied before sigmoid. "
                         "T<1 sharpen (boundary 양극화), T>1 soften (under-confident).")
    ap.add_argument("--asl-gpos", type=float, default=1.0,
                    help="ASL gamma_pos (T4 only). default 1.0 (Ridnik 2021).")
    ap.add_argument("--asl-gneg", type=float, default=4.0,
                    help="ASL gamma_neg (T4 only). default 4.0 (Ridnik 2021).")
    ap.add_argument("--asl-clip", type=float, default=0.05,
                    help="ASL clip (T4 only). default 0.05 (Ridnik 2021).")
    ap.add_argument("--pos-weight", type=str, default=None,
                    help="BCE pos_weight per-class. Format 'IDX:W,IDX:W' or 'NAME:W,NAME:W'. "
                         "e.g., '1:2.0' (fork=2x) or 'fork:2.0,scratch_rot:1.5'. Only used by "
                         "T5/T7 (BCE-based variants). B+1 260507 — fork+sr 2-combo recall fix.")
    # 260509 — Knowledge Distillation (iter32) flags
    ap.add_argument("--kd-teacher-probs", type=str, default="",
                    help="Path to teacher probs parquet (chip_path → 4-bit prob). "
                         "If set, enables KD loss (Yang 2023 multi-label per-class binary KL).")
    ap.add_argument("--kd-alpha", type=float, default=0.5,
                    help="KD loss mix weight: L = alpha*L_hard + (1-alpha)*L_KD. (Hinton 2015)")
    ap.add_argument("--kd-temperature", type=float, default=4.0,
                    help="KD temperature T (default 4.0). L_KD scaled by T^2.")
    ap.add_argument("--kd-skip-on-cutmix", action="store_true",
                    help="Skip KD term when CutMix is applied to a sample. v1 simpler default.")
    args = ap.parse_args()

    # ------------------------------------------------------------------
    # DDP init (torchrun env://) - per-rank batch = args.batch (NOT divided),
    # effective global batch = args.batch * world_size.
    # On single-GPU / non-torchrun, all dist.* are no-ops.
    # ------------------------------------------------------------------
    import torch.distributed as dist
    from torch.nn.parallel import DistributedDataParallel as DDP
    from torch.utils.data.distributed import DistributedSampler
    _ws = int(os.environ.get("WORLD_SIZE", "1"))
    _ddp = _ws > 1
    if _ddp:
        dist.init_process_group("nccl", init_method="env://")
        _local_rank = int(os.environ["LOCAL_RANK"])
        torch.cuda.set_device(_local_rank)
        device = torch.device("cuda", _local_rank)
        _rank = dist.get_rank()
        if _rank != 0:
            # Silence non-main rank to avoid duplicated stdout
            import builtins
            builtins.print = lambda *a, **k: None
        print(f"[ddp] init: rank={_rank}/{_ws} local_rank={_local_rank} "
              f"per_rank_batch={args.batch} effective_batch={args.batch * _ws}")
    else:
        _local_rank = 0
        _rank = 0
        device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))

    # Cap GPU memory share (260514 — leave 60%+ for other processes in this project)
    if device.type == "cuda" and args.gpu_mem_fraction is not None and 0.0 < args.gpu_mem_fraction < 1.0:
        try:
            torch.cuda.set_per_process_memory_fraction(args.gpu_mem_fraction, device.index or 0)
            print(f"[init] gpu_mem_fraction={args.gpu_mem_fraction} (cap ≈{args.gpu_mem_fraction*100:.0f}%% of total VRAM)")
        except Exception as e:
            print(f"[init] WARN set_per_process_memory_fraction failed: {e}")

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # 260509 — load teacher probs for KD (iter32). Empty dict if --kd-teacher-probs not set.
    teacher_prob_map: dict = {}
    if args.kd_teacher_probs:
        import pandas as _pd
        df_kd = _pd.read_parquet(args.kd_teacher_probs)
        prob_cols = [f"teacher_prob_{c}" for c in TRAIN_CLASSES]
        for _, row in df_kd.iterrows():
            teacher_prob_map[str(row["chip_path"])] = np.array(
                [float(row[col]) for col in prob_cols], dtype=np.float32
            )
        print(f"[kd] loaded {len(teacher_prob_map)} teacher probs from "
              f"{args.kd_teacher_probs}  alpha={args.kd_alpha} T={args.kd_temperature} "
              f"skip_on_cutmix={args.kd_skip_on_cutmix}")

    ts = os.environ.get("TRAIN_RUN_STAMP") or datetime.now().strftime("%Y%m%d_%H%M%S")
    name = args.variant if not args.tag else f"{args.variant}_{args.tag}"
    out_dir = Path(args.out_root) / f"{ts}_{name}"
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[train {args.variant}] device={device}  out={out_dir}")
    multi_combo_root = Path(args.multi_combo_root) if args.multi_combo_root else None
    samples = collect_samples(Path(args.data_root), include_normal=not args.no_normal,
                              multi_combo_root=multi_combo_root,
                              multi_combo_n_per_class=args.multi_combo_n_per_class,
                              rng_seed=args.seed)
    train_samples, val_samples = stratified_split(samples, val_ratio=0.2, seed=args.seed)
    print(f"[train] data: train={len(train_samples)} val={len(val_samples)} classes={TRAIN_CLASSES}")

    model, backbone, img_size = build_model(num_classes=len(TRAIN_CLASSES),
                                             backbone_timm=args.backbone_timm,
                                             img_size_override=args.img_size,
                                             backbone_timm_weights=args.backbone_timm_weights,
                                            init_ckpt=Path(args.init_ckpt),
                                            drop_path_rate=args.drop_path_rate)
    model = model.to(device)
    # 260515 — gradient checkpointing for pair-masked CutMix + batch>=2 within cap=0.40.
    # Solves batch=1 silent-cutmix-no-op bug while staying inside GPU memory budget.
    if args.grad_checkpointing:
        if hasattr(model, "set_grad_checkpointing"):
            try:
                model.set_grad_checkpointing(enable=True)
                print(f"[init] gradient checkpointing ENABLED via timm set_grad_checkpointing")
            except Exception as e:
                print(f"[init] WARN set_grad_checkpointing failed: {e}")
        else:
            print(f"[init] WARN model has no set_grad_checkpointing — --grad-checkpointing ignored")
    if _ddp:
        model = DDP(model, device_ids=[_local_rank], output_device=_local_rank,
                    find_unused_parameters=args.freeze_backbone)
    # 260512 — --freeze-backbone: linear probe mode (classifier-only training)
    if args.freeze_backbone:
        # Freeze everything
        n_total = 0
        n_trainable = 0
        for p in model.parameters():
            p.requires_grad = False
            n_total += p.numel()
        # Unfreeze classifier head only
        try:
            head = model.get_classifier()
        except Exception:
            head = getattr(model, 'head', None) or getattr(model, 'fc', None) or getattr(model, 'classifier', None)
        if head is None:
            raise RuntimeError("Could not locate classifier head; cannot --freeze-backbone")
        for p in head.parameters():
            p.requires_grad = True
            n_trainable += p.numel()
        print(f"[init] FROZEN backbone, classifier-only training: "
              f"{n_trainable:,} trainable / {n_total:,} total params "
              f"({100.0*n_trainable/max(n_total,1):.2f}%)")

    ema = ModelEMA(model, decay=args.ema_decay) if args.ema_decay > 0 else None
    if ema is not None:
        print(f"[train] EMA enabled, target decay = {args.ema_decay}")

    train_ds = ChipFolderDataset(train_samples, img_size, train=True,
                                 teacher_prob_map=teacher_prob_map)
    val_ds = ChipFolderDataset(val_samples, img_size, train=False,
                               teacher_prob_map=teacher_prob_map)
    train_sampler = (DistributedSampler(train_ds, shuffle=True, seed=args.seed)
                     if _ddp else None)
    _dl_kw = dict(num_workers=args.num_workers,
                  pin_memory=(device.type == "cuda"),
                  persistent_workers=(args.num_workers > 0),
                  prefetch_factor=4 if args.num_workers > 0 else None)
    train_loader = DataLoader(train_ds, batch_size=args.batch,
                              sampler=train_sampler,
                              shuffle=(train_sampler is None),
                              drop_last=_ddp,
                              **_dl_kw)
    val_loader = DataLoader(val_ds, batch_size=args.batch, shuffle=False, **_dl_kw)

    loss_name = VARIANT_TO_LOSS[args.variant]
    build_kw = {}
    if args.ls is not None:
        build_kw["ls"] = float(args.ls)
    # 260513 absolute rule: pos_target / neg_target independent (override LS)
    if args.pos_target is not None:
        build_kw["pos_target"] = float(args.pos_target)
    if args.neg_target is not None:
        build_kw["neg_target"] = float(args.neg_target)
    # 260513 asymmetric per-bit targets (4 values in TRAIN_CLASSES order)
    if args.pos_targets_asy is not None:
        parts = [float(x) for x in args.pos_targets_asy.split(',')]
        assert len(parts) == len(TRAIN_CLASSES), f"--pos-targets-asy needs {len(TRAIN_CLASSES)} values"
        build_kw["pos_target_per_bit"] = parts
    if args.neg_targets_asy is not None:
        parts = [float(x) for x in args.neg_targets_asy.split(',')]
        assert len(parts) == len(TRAIN_CLASSES), f"--neg-targets-asy needs {len(TRAIN_CLASSES)} values"
        build_kw["neg_target_per_bit"] = parts
    # 260513 training-time temperature (logit / T before BCE)
    if args.bce_temperature != 1.0:
        build_kw["bce_temperature"] = float(args.bce_temperature)
    if args.variant in ("T4", "T6"):
        build_kw["gamma_pos"] = float(args.asl_gpos)
        build_kw["gamma_neg"] = float(args.asl_gneg)
        build_kw["clip"] = float(args.asl_clip)
    # B+1 260507 — BCE pos_weight (T5/T7 only, otherwise unused).
    pos_weight_tensor: torch.Tensor | None = None
    if args.pos_weight:
        weights = [1.0] * len(TRAIN_CLASSES)
        for entry in args.pos_weight.split(","):
            entry = entry.strip()
            if not entry:
                continue
            key_str, w_str = entry.split(":")
            key_str = key_str.strip()
            try:
                idx = int(key_str)
            except ValueError:
                idx = TRAIN_CLASSES.index(key_str)
            if not (0 <= idx < len(TRAIN_CLASSES)):
                raise ValueError(f"pos-weight index out of range: {idx}")
            weights[idx] = float(w_str)
        pos_weight_tensor = torch.tensor(weights, dtype=torch.float32)
        print(f"[train] BCE pos_weight = {dict(zip(TRAIN_CLASSES, weights))}")
        if args.variant in ("T5", "T7"):
            build_kw["pos_weight"] = pos_weight_tensor
        else:
            print(f"[train] WARN: --pos-weight ignored (variant={args.variant} is not BCE-based)")
    if build_kw:
        loss_fn, target_kind = build_loss(loss_name, **build_kw)
        loss_name = f"{loss_name}({build_kw})"
    else:
        loss_fn, target_kind = build_loss(loss_name)
    loss_fn = loss_fn.to(device)
    print(f"[train] loss={loss_name} target_kind={target_kind}")
    # 260515 — raw loss for cutmix mix sample (bypass LS / pos/neg, preserve mix_t)
    loss_fn_mix_raw = None
    if args.cutmix_mix_raw_target:
        # build_loss with no smoothing kwargs → plain BCE
        loss_fn_mix_raw, _ = build_loss(loss_name.split("(")[0])
        loss_fn_mix_raw = loss_fn_mix_raw.to(device)
        print(f"[train] cutmix MIX uses raw BCE (no LS/pos/neg smoothing)")
    # 260515 — alternate loss for mask paired samples (override pos/neg target)
    loss_fn_mask = None
    if args.cutmix_mask_pos_target is not None or args.cutmix_mask_neg_target is not None:
        mask_kw = dict(build_kw)
        # remove ls / pos_target_per_bit / neg_target_per_bit so explicit pos/neg take precedence
        mask_kw.pop("ls", None)
        mask_kw.pop("pos_target_per_bit", None)
        mask_kw.pop("neg_target_per_bit", None)
        if args.cutmix_mask_pos_target is not None:
            mask_kw["pos_target"] = float(args.cutmix_mask_pos_target)
        if args.cutmix_mask_neg_target is not None:
            mask_kw["neg_target"] = float(args.cutmix_mask_neg_target)
        loss_fn_mask, _ = build_loss(loss_name.split("(")[0], **mask_kw)
        loss_fn_mask = loss_fn_mask.to(device)
        print(f"[train] mask paired loss override: pos={args.cutmix_mask_pos_target} neg={args.cutmix_mask_neg_target}")
    # 260515 — guard against silent-no-op CutMix when batch=1.
    # Bug: torch.randperm(1)=[0] → diff_class_mask all False → cutmix never applied,
    # config flags printed but ignored. Result: bit-identical weights across configs.
    if args.cutmix_p > 0 and args.batch < 2:
        raise ValueError(
            f"--cutmix-p={args.cutmix_p} requires --batch >= 2 (got {args.batch}). "
            "CutMix needs intra-batch pair to mix. batch=1 silently degenerates to no-op. "
            "Use --batch 2 --accum 16 (effective batch 32) + --grad-checkpointing if OOM."
        )
    if args.cutmix_p > 0:
        if args.cutmix_mode == "scattered":
            soft_b = float(args.cutmix_total_ratio) * float(args.cutmix_discount) \
                     * float(args.cutmix_alpha)
            print(f"[train] CutMix enabled mode=scattered p={args.cutmix_p} "
                  f"n_patches={args.cutmix_n_patches} total_ratio={args.cutmix_total_ratio} "
                  f"discount={args.cutmix_discount} alpha={args.cutmix_alpha} "
                  f"-> soft label_B={soft_b:.4f}")
        elif args.cutmix_mode == "grid":
            print(f"[train] CutMix enabled mode=grid p={args.cutmix_p} "
                  f"GRID=8x8 grid_prob={args.cutmix_grid_prob} "
                  f"-> expected ratio≈{args.cutmix_grid_prob}")
        elif args.cutmix_mode == "complement":
            print(f"[train] CutMix enabled mode=complement p={args.cutmix_p} "
                  f"n_groups={args.cutmix_n_groups} pair={args.cutmix_pair} "
                  f"label_scale={args.cutmix_complete_label_scale}")
        elif args.cutmix_mode in ("grid_complete", "grid_sparse"):
            print(f"[train] CutMix enabled mode={args.cutmix_mode} p={args.cutmix_p} "
                  f"label_scale={args.cutmix_complete_label_scale}")
        elif args.cutmix_mode in ("bisect_h", "bisect_v", "bisect_rand"):
            # 260513 — half-image bisect cutmix (paper §6.X NEW axis, analyst opus)
            print(f"[train] CutMix enabled mode={args.cutmix_mode} p={args.cutmix_p} "
                  f"pair={args.cutmix_pair} label_scale={args.cutmix_complete_label_scale} "
                  f"ab_labels={args.cutmix_ab_labels or 'symmetric'}")
        else:
            print(f"[train] CutMix enabled mode=single p={args.cutmix_p} "
                  f"rect={args.cutmix_rect}")

    if args.lr_backbone is not None and args.lr_head is not None:
        backbone_params = [p for n, p in model.named_parameters() if "head" not in n]
        head_params = [p for n, p in model.named_parameters() if "head" in n]
        optim = torch.optim.AdamW(
            [{"params": backbone_params, "lr": args.lr_backbone},
             {"params": head_params, "lr": args.lr_head}],
            weight_decay=0.05,
        )
        print(f"[train] two-LR group: backbone={args.lr_backbone} head={args.lr_head}")
    else:
        optim = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.05)
        print(f"[train] single LR: {args.lr}")

    if args.warmup_epochs > 0 and args.warmup_epochs < args.epochs:
        warmup_sched = torch.optim.lr_scheduler.LinearLR(
            optim, start_factor=args.warmup_start_factor, total_iters=args.warmup_epochs,
        )
        cosine_sched = torch.optim.lr_scheduler.CosineAnnealingLR(
            optim, T_max=max(1, args.epochs - args.warmup_epochs), eta_min=args.lr_eta_min,
        )
        sched = torch.optim.lr_scheduler.SequentialLR(
            optim, schedulers=[warmup_sched, cosine_sched], milestones=[args.warmup_epochs],
        )
        print(f"[train] scheduler: warmup({args.warmup_epochs}ep, start_factor={args.warmup_start_factor}) -> cosine(eta_min={args.lr_eta_min})")
    else:
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(
            optim, T_max=args.epochs, eta_min=args.lr_eta_min,
        )
        print(f"[train] scheduler: cosine(eta_min={args.lr_eta_min}), no warmup")

    scaler = torch.amp.GradScaler() if device.type == "cuda" else None

    _mv_cache = None
    if args.multi_val_set:
        try:
            from ._per_epoch_multi_eval import load_multi_val as _mv_load
            _mv_cache = _mv_load(args.multi_val_set,
                                 n_per_class=args.multi_val_n_per_class,
                                 seed=args.seed, img_size=img_size)
            print(f"[before epoch 1] train N={len(train_samples)} "
                  f"trainval N={len(val_samples)} "
                  f"eval N={len(_mv_cache['paths'])} "
                  f"eval_set={args.multi_val_set} "
                  f"eval_cap={args.multi_val_n_per_class}/class")
        except Exception as e:
            print(f"[before epoch 1] train N={len(train_samples)} "
                  f"trainval N={len(val_samples)} eval N=LOAD_FAIL "
                  f"eval_set={args.multi_val_set} "
                  f"error={type(e).__name__}: {e}")
            raise
    else:
        print(f"[before epoch 1] train N={len(train_samples)} "
              f"trainval N={len(val_samples)} eval N=0 eval_set=None")

    history = []
    best_val_acc = float("-inf")
    best_epoch = -1
    t_total = time.time()
    for ep in range(1, args.epochs + 1):
        if train_sampler is not None:
            train_sampler.set_epoch(ep)
        model.train()
        if hasattr(loss_fn, "set_epoch"):
            loss_fn.set_epoch(ep - 1)
        running = 0.0
        nb = 0
        optim.zero_grad()
        _train_iter = _maybe_tqdm(train_loader, total=len(train_loader),
                                  desc=f"ep {ep:02d} train", leave=False, dynamic_ncols=True)
        for step, batch in enumerate(_train_iter):
            # 260508 — dataset returns 3-tuple (x, y, mh).  260509 — 4-tuple adds teacher_prob.
            if len(batch) == 4:
                x, y, mh, teacher_prob = batch
                teacher_prob = teacher_prob.to(device, non_blocking=True)
            else:
                x, y, mh = batch
                teacher_prob = None
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            mh = mh.to(device, non_blocking=True)
            # 260508 paired CutMix: x_masked + tgt_masked tracked for dual forward
            x_masked = None
            tgt_masked = None
            # 260515 masked2: optional second mask (B chip with rect masked → label=B)
            x_masked2 = None
            tgt_masked2 = None
            # 260509 KD: track whether cutmix actually applied (for --kd-skip-on-cutmix).
            # Set True inside CutMix branches when valid_mask.any() triggers.
            cutmix_applied = False
            if target_kind in ("multi_hot", "soft_multihot"):
                # 260508 — initialize tgt from multi-hot directly (handles single + Normal + multi-combo).
                # Single class (y >= 0): mh is one-hot → tgt[i, y[i]] = 1
                # Normal (y == -1): mh is [0,0,0,0]
                # Multi-combo (y == -2): mh has 2-3 bits set
                tgt = mh.float().clone()
                # CutMix: with prob cutmix_p, paste rectangle from another sample
                # with different class. Target rules differ by kind:
                #   multi_hot:    BCE — soft proportional (260508 fix): A=lam, B=1-lam (lam=A area ratio).
                #                       previously: hard OR (1.0 both bits) — overcounts (50% paste → 100% A AND 100% B).
                #                       fix: paste 50% of B → A bit=0.5, B bit=0.5. paste 25% of B → A=0.75, B=0.25.
                #                       model learns "bit confidence ∝ defect area" not "fire on any glimpse".
                #   soft_multihot: CE soft — λ*A + (1-λ)*B (sums to 1, λ from area)
                if args.cutmix_p > 0 and float(np.random.rand()) < args.cutmix_p:
                    bs = y.size(0)
                    perm = torch.randperm(bs, device=device)
                    # 260506 F: cutmix-pair-bias — oversample specific pairs.
                    # Format: "C1,C2:K" → after perm, with prob K/(K+1), force perm of C1→C2 pair.
                    # E.g., "fork,scratch:2" → fork↔scratch pair occurs 2× more often than uniform.
                    if args.cutmix_pair_bias:
                        spec = args.cutmix_pair_bias  # "C1,C2:K"
                        try:
                            cls_str, k_str = spec.split(":")
                            ca, cb = cls_str.split(",")
                            ya = TRAIN_CLASSES.index(ca.strip())
                            yb = TRAIN_CLASSES.index(cb.strip())
                            k_bias = int(k_str)
                        except Exception:
                            ya = yb = -1
                            k_bias = 0
                        if k_bias > 0:
                            # for each row, with prob k/(k+1), find a partner of opposite biased class
                            for bi in range(bs):
                                if float(np.random.rand()) >= k_bias / (k_bias + 1):
                                    continue
                                if int(y[bi].item()) == ya:
                                    cands = (y == yb).nonzero(as_tuple=True)[0]
                                elif int(y[bi].item()) == yb:
                                    cands = (y == ya).nonzero(as_tuple=True)[0]
                                else:
                                    continue
                                if len(cands) > 0:
                                    perm[bi] = cands[int(np.random.randint(0, len(cands)))]
                    diff_class_mask = (y[perm] != y)
                    # 260506: scratch+scratch_rot pair re-allowed per user directive
                    # 260506 C: skip CutMix when EITHER member is Normal (y=-1) — Normal mosaic
                    # would corrupt defect signal. Only defect↔defect pairs.
                    both_defect = (y >= 0) & (y[perm] >= 0)
                    valid_mask = diff_class_mask & both_defect
                    if valid_mask.any():
                        cutmix_applied = True   # 260509 — KD skip flag

                        H, W = x.size(-2), x.size(-1)
                        # 260508 paired CutMix: capture pre-mix copies + collect rects (all 3 modes)
                        # 260515 masked2 also enables do_pair (super-set of masked)
                        do_pair = (args.cutmix_pair in ("masked", "masked2"))
                        do_pair2 = (args.cutmix_pair == "masked2")
                        if do_pair:
                            x_pre = x.clone()
                            tgt_pre = tgt.clone()
                            masked_rects = [[] for _ in range(bs)]   # bi → [(y0,y1,x0,x1), ...]

                        if args.cutmix_mode == "mixup":
                            # 260509 — Mixup (Zhang 2018): pixel α-blend + linear label combination.
                            # NOTE: chip palette discrete grade {0..7} → blend creates mid-values
                            # (e.g., 0.7*grade1 + 0.3*grade5 = grade2.2 non-palette). paper-grade
                            # ablation only — expected to underperform vs FCM-PM region paste.
                            alpha_mp = float(args.cutmix_alpha) if args.cutmix_alpha > 0 else 0.2
                            lam = float(np.random.beta(alpha_mp, alpha_mp))
                            x_new = lam * x + (1.0 - lam) * x[perm]
                            x = x_new
                            for bi in range(bs):
                                if not bool(valid_mask[bi].item()):
                                    continue
                                a_class = int(y[bi].item())
                                b_class = int(y[perm[bi]].item())
                                if target_kind == "soft_multihot":
                                    tgt[bi, a_class] = lam
                                    tgt[bi, b_class] = max(tgt[bi, b_class].item(), 1.0 - lam)
                                else:
                                    tgt[bi, a_class] = max(tgt[bi, a_class].item(), lam)
                                    tgt[bi, b_class] = max(tgt[bi, b_class].item(), 1.0 - lam)
                            do_pair = False  # mixup has no pair mask concept
                        elif args.cutmix_mode == "scattered":
                            n_patches = max(1, int(args.cutmix_n_patches))
                            total_ratio = max(1e-4, float(args.cutmix_total_ratio))
                            patch_area_each = (total_ratio * H * W) / n_patches
                            patch_side = int(round(patch_area_each ** 0.5))
                            patch_side = max(1, min(patch_side, H - 1))
                            cy_list = [int(np.random.randint(0, H - patch_side + 1))
                                       for _ in range(n_patches)]
                            cx_list = [int(np.random.randint(0, W - patch_side + 1))
                                       for _ in range(n_patches)]
                            actual_total_area = float(n_patches * patch_side * patch_side)
                            actual_ratio = min(1.0, actual_total_area / float(H * W))
                            soft_label_b = actual_ratio * float(args.cutmix_discount) \
                                           * float(args.cutmix_alpha)
                            for bi in range(bs):
                                if not bool(valid_mask[bi].item()):
                                    continue
                                for k in range(n_patches):
                                    cy_k = cy_list[k]
                                    cx_k = cx_list[k]
                                    x[bi, :, cy_k:cy_k + patch_side,
                                      cx_k:cx_k + patch_side] = \
                                        x[perm[bi], :, cy_k:cy_k + patch_side,
                                          cx_k:cx_k + patch_side]
                                    if do_pair:
                                        masked_rects[bi].append(
                                            (cy_k, cy_k + patch_side, cx_k, cx_k + patch_side))
                                a_class = int(y[bi].item())
                                b_class = int(y[perm[bi]].item())
                                if target_kind == "soft_multihot":
                                    lam_b = min(soft_label_b, 0.5)
                                    tgt[bi, a_class] = 1.0 - lam_b
                                    tgt[bi, b_class] = lam_b
                                else:
                                    # 260508 fix — soft proportional for BCE multi_hot too
                                    tgt[bi, a_class] = 1.0 - actual_ratio
                                    tgt[bi, b_class] = actual_ratio
                        elif args.cutmix_mode == "grid":
                            GRID = 8
                            cell_h = H // GRID
                            cell_w = W // GRID
                            grid_prob = float(args.cutmix_grid_prob)
                            for bi in range(bs):
                                if not bool(valid_mask[bi].item()):
                                    continue
                                actual_total = 0
                                for gi in range(GRID):
                                    for gj in range(GRID):
                                        if float(np.random.rand()) < grid_prob:
                                            y0 = gi * cell_h
                                            y1 = (gi + 1) * cell_h if gi < GRID - 1 else H
                                            x0 = gj * cell_w
                                            x1 = (gj + 1) * cell_w if gj < GRID - 1 else W
                                            x[bi, :, y0:y1, x0:x1] = \
                                                x[perm[bi], :, y0:y1, x0:x1]
                                            actual_total += (y1 - y0) * (x1 - x0)
                                            if do_pair:
                                                masked_rects[bi].append((y0, y1, x0, x1))
                                actual_ratio = float(actual_total) / float(H * W)
                                a_class = int(y[bi].item())
                                b_class = int(y[perm[bi]].item())
                                if target_kind == "soft_multihot":
                                    lam_b = min(actual_ratio, 0.5)
                                    tgt[bi, a_class] = 1.0 - lam_b
                                    tgt[bi, b_class] = lam_b
                                else:
                                    # 260508 fix — soft proportional for BCE multi_hot too
                                    tgt[bi, a_class] = 1.0 - actual_ratio
                                    tgt[bi, b_class] = actual_ratio
                        elif args.cutmix_mode == "complement":
                            # 260508 — Complement CutMix (★ user-designed)
                            # 1 pair (A,B) → N mix chips + N mask chips = 2N samples per pair.
                            # Each mix_i = chip A의 group_i cells + chip B의 (other groups) cells.
                            # All cells of A and B distributed across N mix chips → no information loss.
                            # mask_i = mix_i with B-cells filled with corner_mean of A → label [A only].
                            # n_groups = 2/3/4. label_scale = 0.5/0.75/1.0 etc.
                            # NOTE: bypasses standard pair logic (do_pair from outer scope) —
                            # complement has its own paired branch built-in via mask chips.
                            GRID = max(2, int(args.cutmix_grid_dim))
                            n_cells = GRID * GRID
                            n_groups = max(2, min(int(args.cutmix_n_groups), n_cells))
                            cells_per_group = n_cells // n_groups
                            label_scale = float(args.cutmix_complete_label_scale)
                            cell_h = H // GRID
                            cell_w = W // GRID
                            include_mask = (args.cutmix_pair == "masked")
                            corner_per_chip = x[:, :, :8, :8].mean(dim=(2, 3), keepdim=True)
                            new_x: List[torch.Tensor] = []
                            new_tgt: List[torch.Tensor] = []
                            for bi in range(bs):
                                if not bool(valid_mask[bi].item()):
                                    new_x.append(x[bi:bi+1])
                                    new_tgt.append(tgt[bi:bi+1])
                                    continue
                                a_cls = int(y[bi].item())
                                b_cls = int(y[perm[bi]].item())
                                chip_a = x[bi]
                                chip_b = x[perm[bi]]
                                cor_a = corner_per_chip[bi]
                                # partition 64 cells into n_groups (random, leftover → last group)
                                perm_cells = np.random.permutation(n_cells)
                                groups = [perm_cells[i*cells_per_group:(i+1)*cells_per_group].tolist()
                                          for i in range(n_groups)]
                                leftover = perm_cells[cells_per_group * n_groups:].tolist()
                                if leftover:
                                    groups[-1].extend(leftover)
                                for i in range(n_groups):
                                    # mix_i: B base, A's group_i cells overwritten
                                    mix = chip_b.clone()
                                    for ci in groups[i]:
                                        gi, gj = int(ci) // GRID, int(ci) % GRID
                                        y0 = gi * cell_h
                                        y1 = (gi + 1) * cell_h if gi < GRID - 1 else H
                                        x0 = gj * cell_w
                                        x1 = (gj + 1) * cell_w if gj < GRID - 1 else W
                                        mix[:, y0:y1, x0:x1] = chip_a[:, y0:y1, x0:x1]
                                    new_x.append(mix.unsqueeze(0))
                                    # 260512: other-label LS support. Initialize with cutmix_other_label
                                    # for all bits, then overwrite A/B class bits below.
                                    other_lbl = float(args.cutmix_other_label)
                                    mix_t = torch.full((len(TRAIN_CLASSES),), other_lbl, device=device)
                                    if args.cutmix_ab_labels:
                                        # paper §6.16: explicit asymmetric (A_label, B_label)
                                        a_lbl, b_lbl = [float(v) for v in args.cutmix_ab_labels.split(',')]
                                        mix_t[a_cls] = a_lbl
                                        mix_t[b_cls] = b_lbl
                                    elif args.cutmix_label_area_prop:
                                        # paper §6.13: area-proportional label
                                        # a_frac = 1/n_groups (A occupies group_i only)
                                        # b_frac = (n_groups-1)/n_groups (B occupies other groups)
                                        a_frac = 1.0 / float(n_groups)
                                        b_frac = (float(n_groups) - 1.0) / float(n_groups)
                                        mix_t[a_cls] = a_frac * label_scale
                                        mix_t[b_cls] = b_frac * label_scale
                                    else:
                                        mix_t[a_cls] = label_scale
                                        mix_t[b_cls] = label_scale
                                    new_tgt.append(mix_t.unsqueeze(0))
                                    if include_mask:
                                        # mask_i: mix_i with B-cells (other groups) filled corner_a → A only
                                        mask = mix.clone()
                                        for j in range(n_groups):
                                            if j == i:
                                                continue
                                            for ci in groups[j]:
                                                gi, gj = int(ci) // GRID, int(ci) % GRID
                                                y0 = gi * cell_h
                                                y1 = (gi + 1) * cell_h if gi < GRID - 1 else H
                                                x0 = gj * cell_w
                                                x1 = (gj + 1) * cell_w if gj < GRID - 1 else W
                                                mask[:, y0:y1, x0:x1] = cor_a
                                        new_x.append(mask.unsqueeze(0))
                                        mask_t = torch.zeros(len(TRAIN_CLASSES), device=device)
                                        if args.cutmix_ab_labels:
                                            a_lbl, _ = [float(v) for v in args.cutmix_ab_labels.split(',')]
                                            mask_t[a_cls] = a_lbl
                                        else:
                                            mask_t[a_cls] = label_scale
                                        new_tgt.append(mask_t.unsqueeze(0))
                            # rebuild x and tgt from collected samples
                            x = torch.cat(new_x, dim=0)
                            tgt = torch.cat(new_tgt, dim=0)
                            do_pair = False  # complement has its own paired branch (mask chips)
                        elif args.cutmix_mode in ("bisect_h", "bisect_v", "bisect_rand"):
                            # 260513 — half-image bisect cutmix (paper §6.X NEW axis).
                            # Single boundary, A=one half, B=other half. 50/50 area.
                            # Tests whether coherent contiguous regions match chip multi-label
                            # manifold better than 8x8 grid scatter (complement g=2).
                            # Pair counterfactual (--cutmix-pair=masked) supported.
                            include_mask = (args.cutmix_pair == "masked")
                            corner_per_chip = x[:, :, :8, :8].mean(dim=(2, 3), keepdim=True)
                            label_scale = float(args.cutmix_complete_label_scale)
                            new_x: List[torch.Tensor] = []
                            new_tgt: List[torch.Tensor] = []
                            for bi in range(bs):
                                if not bool(valid_mask[bi].item()):
                                    new_x.append(x[bi:bi+1])
                                    new_tgt.append(tgt[bi:bi+1])
                                    continue
                                a_cls = int(y[bi].item())
                                b_cls = int(y[perm[bi]].item())
                                chip_a = x[bi]
                                chip_b = x[perm[bi]]
                                cor_a = corner_per_chip[bi]
                                # decide orientation
                                if args.cutmix_mode == "bisect_h":
                                    orient = "h"
                                elif args.cutmix_mode == "bisect_v":
                                    orient = "v"
                                else:
                                    orient = "h" if float(np.random.rand()) < 0.5 else "v"
                                # build mix chip: A on one half, B on the other
                                mix = chip_b.clone()
                                if orient == "h":
                                    mix[:, : H // 2, :] = chip_a[:, : H // 2, :]
                                else:
                                    mix[:, :, : W // 2] = chip_a[:, :, : W // 2]
                                new_x.append(mix.unsqueeze(0))
                                # label rule
                                other_lbl = float(args.cutmix_other_label) if hasattr(args, 'cutmix_other_label') else 0.0
                                mix_t = torch.full((len(TRAIN_CLASSES),), other_lbl, device=device)
                                if args.cutmix_ab_labels:
                                    a_lbl, b_lbl = [float(v) for v in args.cutmix_ab_labels.split(',')]
                                    mix_t[a_cls] = a_lbl
                                    mix_t[b_cls] = b_lbl
                                else:
                                    mix_t[a_cls] = label_scale
                                    mix_t[b_cls] = label_scale
                                new_tgt.append(mix_t.unsqueeze(0))
                                if include_mask:
                                    # paired counterfactual: B-half filled with A's corner -> A only
                                    mask = mix.clone()
                                    if orient == "h":
                                        mask[:, H // 2:, :] = cor_a
                                    else:
                                        mask[:, :, W // 2:] = cor_a
                                    new_x.append(mask.unsqueeze(0))
                                    mask_t = torch.zeros(len(TRAIN_CLASSES), device=device)
                                    if args.cutmix_ab_labels:
                                        a_lbl, _ = [float(v) for v in args.cutmix_ab_labels.split(',')]
                                        mask_t[a_cls] = a_lbl
                                    else:
                                        mask_t[a_cls] = label_scale
                                    new_tgt.append(mask_t.unsqueeze(0))
                            x = torch.cat(new_x, dim=0)
                            tgt = torch.cat(new_tgt, dim=0)
                            do_pair = False  # bisect has its own paired branch (mask chips)
                        elif args.cutmix_mode == "grid_complete":
                            # 260508 — deterministic 50/50 split, full chip cover
                            # GRID*GRID cells = half chip A + half chip B (no whitespace, both defects fully present)
                            GRID = max(2, int(args.cutmix_grid_dim))
                            n_cells = GRID * GRID
                            half = n_cells // 2
                            cell_h = H // GRID
                            cell_w = W // GRID
                            label_scale = float(args.cutmix_complete_label_scale)
                            for bi in range(bs):
                                if not bool(valid_mask[bi].item()):
                                    continue
                                b_cells = np.random.choice(n_cells, size=half, replace=False)
                                for ci in b_cells:
                                    gi = int(ci) // GRID
                                    gj = int(ci) % GRID
                                    y0 = gi * cell_h
                                    y1 = (gi + 1) * cell_h if gi < GRID - 1 else H
                                    x0 = gj * cell_w
                                    x1 = (gj + 1) * cell_w if gj < GRID - 1 else W
                                    x[bi, :, y0:y1, x0:x1] = \
                                        x[perm[bi], :, y0:y1, x0:x1]
                                    if do_pair:
                                        masked_rects[bi].append((y0, y1, x0, x1))
                                a_class = int(y[bi].item())
                                b_class = int(y[perm[bi]].item())
                                if target_kind == "soft_multihot":
                                    # CE-soft (sums to 1) — fixed 0.5/0.5
                                    tgt[bi, a_class] = 0.5
                                    tgt[bi, b_class] = 0.5
                                else:
                                    # multi_hot (BCE) — configurable scale
                                    # justified by complete cover: both defects' full structure visible
                                    tgt[bi, a_class] = label_scale
                                    tgt[bi, b_class] = label_scale
                        elif args.cutmix_mode == "grid_sparse":
                            # 260508 grid + scatter mix: K random cells from 8×8 grid
                            GRID = 8
                            cell_h = H // GRID
                            cell_w = W // GRID
                            k_cells = max(1, min(int(args.cutmix_grid_k), GRID * GRID))
                            for bi in range(bs):
                                if not bool(valid_mask[bi].item()):
                                    continue
                                # pick K random unique (gi, gj) pairs from 8×8 grid
                                cell_idx = np.random.choice(GRID * GRID, size=k_cells, replace=False)
                                actual_total = 0
                                for ci in cell_idx:
                                    gi = int(ci) // GRID
                                    gj = int(ci) % GRID
                                    y0 = gi * cell_h
                                    y1 = (gi + 1) * cell_h if gi < GRID - 1 else H
                                    x0 = gj * cell_w
                                    x1 = (gj + 1) * cell_w if gj < GRID - 1 else W
                                    x[bi, :, y0:y1, x0:x1] = \
                                        x[perm[bi], :, y0:y1, x0:x1]
                                    actual_total += (y1 - y0) * (x1 - x0)
                                    if do_pair:
                                        masked_rects[bi].append((y0, y1, x0, x1))
                                actual_ratio = float(actual_total) / float(H * W)
                                a_class = int(y[bi].item())
                                b_class = int(y[perm[bi]].item())
                                if target_kind == "soft_multihot":
                                    lam_b = min(actual_ratio, 0.5)
                                    tgt[bi, a_class] = 1.0 - lam_b
                                    tgt[bi, b_class] = lam_b
                                else:
                                    # 260508 fix — soft proportional for BCE multi_hot too
                                    tgt[bi, a_class] = 1.0 - actual_ratio
                                    tgt[bi, b_class] = actual_ratio
                        else:
                            # single mode — soft proportional CutMix (260508 fix)
                            side = int(round(float(args.cutmix_rect) ** 0.5 * H))
                            side = max(1, min(side, H - 1))
                            cy = int(np.random.randint(0, H - side + 1))
                            cx = int(np.random.randint(0, W - side + 1))
                            lam = 1.0 - float(side * side) / float(H * W)
                            for bi in range(bs):
                                if not bool(valid_mask[bi].item()):
                                    continue
                                x[bi, :, cy:cy + side, cx:cx + side] = \
                                    x[perm[bi], :, cy:cy + side, cx:cx + side]
                                if do_pair:
                                    masked_rects[bi].append((cy, cy + side, cx, cx + side))
                                a = int(y[bi].item())
                                b = int(y[perm[bi]].item())
                                if target_kind == "soft_multihot":
                                    tgt[bi, a] = lam
                                    tgt[bi, b] = 1.0 - lam
                                else:
                                    # 260508 fix — soft proportional for BCE multi_hot too
                                    # paste B 영역 ratio = 1 - lam (lam = A 의 면적 비율)
                                    tgt[bi, a] = lam
                                    tgt[bi, b] = 1.0 - lam

                        # 260508 paired: build x_masked from x_pre with same rects masked (all 3 modes)
                        # 260515 masked2: ALSO build x_masked2 from x_pre[perm] (B chips) with same rects masked → label=B
                        if do_pair and any(len(r) > 0 for r in masked_rects):
                            pair_fill = args.cutmix_pair_fill
                            # Capture B's pre-mix images BEFORE x_masked mutates x_pre
                            if do_pair2:
                                x_b_pre = x_pre[perm].clone()
                                if pair_fill == "corner":
                                    corner_per_chip2 = x_b_pre[:, :, :8, :8].mean(dim=(2, 3), keepdim=True)
                            x_masked_buf = x_pre   # already cloned
                            if pair_fill == "corner":
                                corner_per_chip = x_pre[:, :, :8, :8].mean(dim=(2, 3), keepdim=True)
                            elif pair_fill == "white":
                                white_norm = torch.tensor(
                                    [(1.0 - 0.485) / 0.229, (1.0 - 0.456) / 0.224, (1.0 - 0.406) / 0.225],
                                    device=device).view(3, 1, 1)
                            for bi in range(bs):
                                for (y0r, y1r, x0r, x1r) in masked_rects[bi]:
                                    hh, ww = y1r - y0r, x1r - x0r
                                    if pair_fill == "noise":
                                        x_masked_buf[bi, :, y0r:y1r, x0r:x1r] = \
                                            torch.randn(3, hh, ww, device=device) * 0.3
                                    elif pair_fill == "white":
                                        x_masked_buf[bi, :, y0r:y1r, x0r:x1r] = \
                                            white_norm.expand(3, hh, ww)
                                    else:   # corner
                                        x_masked_buf[bi, :, y0r:y1r, x0r:x1r] = corner_per_chip[bi]
                            x_masked = x_masked_buf
                            tgt_masked = tgt_pre
                            # 260515 — apply same rect mask to B's pre-mix copies
                            if do_pair2:
                                for bi in range(bs):
                                    for (y0r, y1r, x0r, x1r) in masked_rects[bi]:
                                        hh, ww = y1r - y0r, x1r - x0r
                                        if pair_fill == "noise":
                                            x_b_pre[bi, :, y0r:y1r, x0r:x1r] = \
                                                torch.randn(3, hh, ww, device=device) * 0.3
                                        elif pair_fill == "white":
                                            x_b_pre[bi, :, y0r:y1r, x0r:x1r] = \
                                                white_norm.expand(3, hh, ww)
                                        else:  # corner
                                            x_b_pre[bi, :, y0r:y1r, x0r:x1r] = corner_per_chip2[bi]
                                x_masked2 = x_b_pre
                                tgt_masked2 = tgt_pre[perm]
                tgt_used = tgt
            else:
                tgt_used = y
            with torch.amp.autocast(device_type=device.type, enabled=device.type == "cuda"):
                logits = model(x)
                # 260515 — if cutmix-mix-raw-target set AND we just did a cutmix forward, use raw loss
                _lfn_main = loss_fn_mix_raw if (loss_fn_mix_raw is not None and cutmix_applied) else loss_fn
                loss_main = _lfn_main(logits, tgt_used)
                # 260508 paired CutMix: dual forward — masked counterpart
                if x_masked is not None and tgt_masked is not None:
                    logits_masked = model(x_masked)
                    # 260515 — use loss_fn_mask if set (mask-specific pos/neg target), else loss_fn
                    _lfn = loss_fn_mask if loss_fn_mask is not None else loss_fn
                    loss_masked_pair = _lfn(logits_masked, tgt_masked)
                    loss_pre_kd = loss_main + float(args.cutmix_pair_loss_w) * loss_masked_pair
                else:
                    loss_pre_kd = loss_main
                # 260515 masked2: third forward — B chip with same rect masked → label=B
                if x_masked2 is not None and tgt_masked2 is not None:
                    logits_masked2 = model(x_masked2)
                    _lfn = loss_fn_mask if loss_fn_mask is not None else loss_fn
                    loss_masked2_pair = _lfn(logits_masked2, tgt_masked2)
                    loss_pre_kd = loss_pre_kd + float(args.cutmix_pair_loss_w) * loss_masked2_pair
                # 260509 — KD distillation (Hinton 2015 + Yang 2023 multi-label per-class binary KL).
                # Skip when: KD disabled, no teacher_prob available, batch shape mismatch
                # (e.g., complement mode rebuilt x), or cutmix-skip flag triggered.
                use_kd = (
                    bool(args.kd_teacher_probs)
                    and teacher_prob is not None
                    and teacher_prob.size(0) == logits.size(0)
                    and not (args.kd_skip_on_cutmix and cutmix_applied)
                )
                if use_kd:
                    T_kd = float(args.kd_temperature)
                    eps = 1e-6
                    p_T = teacher_prob.clamp(eps, 1.0 - eps).float()
                    # teacher logit z_T = log(p / (1-p)); soft-target prob with T scaling
                    z_T = torch.log(p_T) - torch.log1p(-p_T)
                    # 260509 patch — use binary_cross_entropy_with_logits (autocast safe).
                    # Apply temperature scaling at logit level: student logit / T → BCE_with_logits target = p_T_soft
                    student_logits_scaled = logits.float() / T_kd
                    p_T_soft = torch.sigmoid(z_T / T_kd)
                    # per-class binary CE via logits (autocast-safe equivalent of Yang 2023 KD)
                    l_kd = torch.nn.functional.binary_cross_entropy_with_logits(
                        student_logits_scaled, p_T_soft, reduction="mean"
                    ) * (T_kd * T_kd)
                    loss_combined = args.kd_alpha * loss_pre_kd + (1.0 - args.kd_alpha) * l_kd
                else:
                    loss_combined = loss_pre_kd
                loss = loss_combined / args.accum
            if scaler is not None:
                scaler.scale(loss).backward()
            else:
                loss.backward()
            running += float(loss.item()) * args.accum
            nb += 1
            if _HAS_TQDM and hasattr(_train_iter, "set_postfix"):
                _train_iter.set_postfix(loss=f"{running/max(1, nb):.4f}", refresh=False)
            if (step + 1) % args.accum == 0 or (step + 1) == len(train_loader):
                if scaler is not None:
                    scaler.unscale_(optim)
                    if args.grad_clip > 0:
                        torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
                    scaler.step(optim)
                    scaler.update()
                else:
                    if args.grad_clip > 0:
                        torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
                    optim.step()
                optim.zero_grad()
                if ema is not None:
                    ema.update(model)
        sched.step()
        eval_model = ema.module if ema is not None else model
        val_metrics = evaluate(eval_model, val_loader, device, num_classes=len(TRAIN_CLASSES))
        val_acc = val_metrics["val_acc"]
        val_f1 = val_metrics["val_f1"]
        val_auroc = val_metrics["val_auroc"]
        val_bce = val_metrics.get("val_bce", float('nan'))
        val_brier = val_metrics.get("val_brier", float('nan'))
        val_margin = val_metrics.get("val_margin", float('nan'))
        val_f1_best_tau = val_metrics.get("val_f1_best_tau", float('nan'))
        # 260512: pick criterion for best selection (default acc backward-compat,
        # f1/auroc/bce_min/brier_min/margin_max/f1_best_tau for multi-label selection).
        def _safe(v, fb):
            return v if v == v else fb  # NaN fallback
        c = args.val_criterion
        if c == "f1":          val_for_best = _safe(val_f1, val_acc)
        elif c == "auroc":     val_for_best = _safe(val_auroc, val_acc)
        elif c == "bce_min":   val_for_best = -_safe(val_bce, -val_acc)   # negate for max
        elif c == "brier_min": val_for_best = -_safe(val_brier, -val_acc)
        elif c == "margin_max":val_for_best = _safe(val_margin, val_acc)
        elif c == "f1_best_tau":val_for_best = _safe(val_f1_best_tau, val_acc)
        else:                  val_for_best = val_acc
        active = getattr(loss_fn, "last_active", loss_name)
        avg_loss = running / max(nb, 1)
        history.append({"epoch": ep, "train_loss": avg_loss, "val_acc": val_acc,
                        "val_f1": val_f1, "val_auroc": val_auroc,
                        "val_bce": val_bce, "val_brier": val_brier,
                        "val_margin": val_margin, "val_f1_best_tau": val_f1_best_tau,
                        "lr": optim.param_groups[0]["lr"], "loss_active": active})
        # 260520 - clearer label: "trainval" = 10% split of train_n{TN} (small),
        # NOT the eval set. The [ep NN/eval @ ...] line below is the proper bit_F1/FAR signal.
        print(f"[ep {ep:02d}/trainval N={len(val_samples)}] "
              f"loss={avg_loss:.4f} v_acc={val_acc:.4f} "
              f"v_f1={val_f1:.4f} v_auroc={val_auroc:.4f} "
              f"v_bce={val_bce:.4f} v_brier={val_brier:.4f} "
              f"v_margin={val_margin:.4f} v_f1bτ={val_f1_best_tau:.4f} "
              f"sel={c}({val_for_best:.4f}) loss_active={active}")
        # Optional: per-epoch class-level F1 + FAR on a multi-label val set
        if args.multi_val_set:
            try:
                from ._per_epoch_multi_eval import evaluate as _mv_evaluate
                from ._per_epoch_multi_eval import format_compact as _mv_format
                eval_model = ema.module if ema is not None else (
                    model.module if hasattr(model, "module") else model)
                m = _mv_evaluate(eval_model, _mv_cache, device=str(device),
                                 threshold=0.5, batch_size=32, num_workers=0)
                # 260520 - explicit label "eval @ <path> N=<count>" so log is unambiguous
                _mv_path = args.multi_val_set
                _mv_n = len(_mv_cache.get('paths', [])) if isinstance(_mv_cache, dict) else 0
                print(f"[ep {ep:02d}/eval @ {_mv_path} N={_mv_n}] {_mv_format(m)}")
                history[-1]["multi_val"] = {
                    "bit_F1": m["bit_F1"], "total_far": m["total_far"],
                    "ni_far": m["ni_far"], "ood_far": m["ood_far"],
                    "per_class_f1": m["per_class_f1"],
                    "per_class_exact": m.get("per_class_exact", {}),
                    "per_class_far": m["per_class_far"],
                    "per_bit_f1": m["per_bit_f1"],
                }
            except Exception as e:
                print(f"[multi-val] FAIL ep{ep}: {type(e).__name__}: {e}")
        save_state_dict = None
        prev_best_ep = best_epoch
        if val_for_best > best_val_acc and ep >= args.best_from_epoch:
            best_val_acc = val_for_best
            best_epoch = ep
            if _is_main_rank():
                save_state_dict = (ema.module.state_dict() if ema is not None
                                    else (model.module.state_dict() if hasattr(model, "module") else model.state_dict()))
                _atomic_torch_save({
                    "model": save_state_dict,
                    "classes": list(TRAIN_CLASSES),
                    "img_size": img_size,
                    "backbone": backbone,
                    "val_acc": float(val_acc),
                    "epoch": ep,
                    "variant": args.variant,
                    "loss_name": loss_name,
                    "ema_decay": args.ema_decay,
                    "warmup_epochs": args.warmup_epochs,
                    "grad_clip": args.grad_clip,
                    "drop_path_rate": args.drop_path_rate,
                }, out_dir / "best_model.pth")
                # 260521 - explicit visibility: which epoch was selected, by which criterion
                print(f"[train] UPDATE best_model.pth: ep{ep:02d} "
                      f"val_{c}={val_for_best:.4f} (prev=ep{prev_best_ep:02d})", flush=True)
        # 260512 — --save-every-epoch: per-epoch ckpt for multi-label selection bug fix
        if args.save_every_epoch and _is_main_rank():
            if save_state_dict is None:
                save_state_dict = (ema.module.state_dict() if ema is not None
                                else (model.module.state_dict() if hasattr(model, "module") else model.state_dict()))
            _atomic_torch_save({
                "model": save_state_dict,
                "classes": list(TRAIN_CLASSES),
                "img_size": img_size,
                "backbone": backbone,
                "val_acc": float(val_acc),
                "epoch": ep,
                "variant": args.variant,
                "loss_name": loss_name,
            }, out_dir / f"epoch_{ep:02d}_model.pth")
        if _ddp:
            dist.barrier()

    elapsed = time.time() - t_total
    print(f"[train] DONE  best_val_acc={best_val_acc:.4f} @ ep{best_epoch}  elapsed={elapsed:.1f}s")

    # 260506 — also save final-epoch model. val_acc (4-way single-class) often saturates
    # at ep1 for multi-hot training (e.g., sc+sr CutMix), causing best_model.pth to be
    # under-trained. final_epoch_model.pth captures end-of-training weights for comparison.
    if _is_main_rank():
        final_state = (ema.module.state_dict() if ema is not None
                       else (model.module.state_dict() if hasattr(model, "module") else model.state_dict()))
        _atomic_torch_save({
            "model": final_state,
            "classes": list(TRAIN_CLASSES),
            "img_size": img_size,
            "backbone": backbone,
            "val_acc": float(val_acc),
            "epoch": args.epochs,
            "variant": args.variant,
            "loss_name": loss_name,
            "ema_decay": args.ema_decay,
            "warmup_epochs": args.warmup_epochs,
            "grad_clip": args.grad_clip,
            "drop_path_rate": args.drop_path_rate,
        }, out_dir / "final_epoch_model.pth")
        print(f"[train] saved final_epoch_model.pth (ep{args.epochs}, val_acc={val_acc:.4f})")

        with open(out_dir / "history.json", "w", encoding="utf-8") as f:
            json.dump(history, f, indent=2)
        with open(out_dir / "train_summary.json", "w", encoding="utf-8") as f:
            json.dump({
                "variant": args.variant,
                "loss_name": loss_name,
                "best_val_acc": best_val_acc,
                "best_epoch": best_epoch,
                "final_val_acc": float(val_acc),
                "final_epoch": args.epochs,
                "epochs": args.epochs,
                "elapsed_sec": elapsed,
                "n_train": len(train_samples),
                "n_val": len(val_samples),
                "ts": ts,
                "out_dir": str(out_dir),
                "no_normal": bool(args.no_normal),
                "ls": (None if args.ls is None else float(args.ls)),
                "cutmix_p": float(args.cutmix_p),
                "cutmix_mode": str(args.cutmix_mode),
                "cutmix_rect": float(args.cutmix_rect),
                "cutmix_n_patches": int(args.cutmix_n_patches),
                "cutmix_total_ratio": float(args.cutmix_total_ratio),
                "cutmix_discount": float(args.cutmix_discount),
                "cutmix_alpha": float(args.cutmix_alpha),
                "pos_weight": (None if args.pos_weight is None else str(args.pos_weight)),
            }, f, indent=2)
    if _ddp:
        dist.barrier()


if __name__ == "__main__":
    main()
