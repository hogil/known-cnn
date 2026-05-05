#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""cnn_eval_chipgrid.py — chip-grid 32x32 native resolution wafer 분류 평가용 standalone.

목적: obj_id encoding 변종 별 빠른 sweep. 기존 cnn_train_*.py 와 분리.

입력:
- wafer PNG (D:/project/data/wm-811k/unknown/<class>/*.png)
  6400x6400 palette idx → AvgPool 32x32 → /31 normalize (R failbit channel)
- obj_id .npy (D:/project/data/wm-811k/obj_id_maps/<subdir>/*.npy)
  32x32 uint8 (0=none, 1..5 chip object id) → variant 별 encoding

모델: 작은 CNN (~1.2M params), 32x32 native input

사용 예:
    python cnn_eval_chipgrid.py --variant V0
    python cnn_eval_chipgrid.py --variant V1 --obj-norm 5
    python cnn_eval_chipgrid.py --variant V2 --target-id 3
    python cnn_eval_chipgrid.py --variant V3
"""
from __future__ import annotations
import argparse, json, random, time
from pathlib import Path
from datetime import datetime
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from sklearn.metrics import f1_score, accuracy_score, precision_recall_fscore_support


# === CONFIG ===
DEFAULT_DATA_DIR    = "D:/project/data/wm-811k/unknown"
DEFAULT_OBJ_ID_DIR  = "D:/project/data/wm-811k/obj_id_maps"
DEFAULT_OBJ_PROB_DIR = "D:/project/data/wm-811k/obj_prob_maps"
DEFAULT_LOG_ROOT    = "logs_chipgrid"
EXCLUDE_CLASSES     = {"Normal", "classification", "classification_chips"}
CHIP_OBJ_IDS        = {0: "none", 1: "bank_boundary", 2: "invalid_main",
                       3: "fork", 4: "scratch", 5: "scratch_rot"}                       # round 26: particle_blast→fork, scratch_21deg→scratch_rot
CHIP_OBJ_LABELS     = tuple(CHIP_OBJ_IDS[i] for i in range(1, 6))
GRID_SIZE           = 32
PALETTE_MAX         = 31


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--variant", type=str, default="V0",
                   choices=["V0", "V1", "V2", "V3", "V4", "V5", "V6"])
    p.add_argument("--target-id", type=int, default=3,
                   help="V2 only: chip object id 1..5 (3=fork default; round 26 spec)")
    p.add_argument("--obj-norm", type=float, default=5.0,
                   help="V1 only: divisor for obj_id integer normalization")
    p.add_argument("--n-per-class", type=int, default=30)
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--batch", type=int, default=32)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--chip-noise", type=float, default=0.0,
                   help="probability of replacing each non-zero obj_id chip with random 1..5 "
                        "(simulates chip CNN inference errors during training)")
    p.add_argument("--chip-noise-eval", action="store_true",
                   help="apply chip-noise also to val/test (simulates production-time chip CNN errors)")
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=0.05)
    p.add_argument("--label-smoothing", type=float, default=0.02)
    p.add_argument("--ema-decay", type=float, default=0.95)
    p.add_argument("--ema-warmup", type=int, default=3)
    p.add_argument("--patience", type=int, default=7)
    p.add_argument("--no-r-channel", action="store_true",
                   help="R 채널 (palette idx mean) 제외하고 obj_id channel(s) 만 입력. "
                        "ablation: V3/V4 의 R 채널 효과 측정 (V3 - R = obj-only)")
    p.add_argument("--aux-heads", type=str, default="none", choices=["none", "factorized"],
                   help="factorized: full-class head에 distribution/object auxiliary heads 추가")
    p.add_argument("--dist-loss-weight", type=float, default=0.0,
                   help="distribution auxiliary CE loss weight")
    p.add_argument("--obj-loss-weight", type=float, default=0.0,
                   help="object auxiliary CE loss weight; object 없는 class는 ignore")
    p.add_argument("--hard-contrastive-weight", type=float, default=0.0,
                   help="same-distribution object swap 억제를 위한 supervised contrastive loss weight")
    p.add_argument("--hard-contrastive-temp", type=float, default=0.2,
                   help="supervised contrastive temperature")
    p.add_argument("--hard-contrastive-scope", type=str, default="edge",
                   choices=["edge", "object", "all"],
                   help="contrastive 대상: edge=Edge-Top/Bottom object class, object=object-bearing class, all=all class")
    p.add_argument("--active-classes-yaml", type=str, default=None,
                   help="YAML with classes: list - 학습 시 그 class 만 사용. "
                        "default: 모든 class (data-dir 의 폴더명 기반)")
    p.add_argument("--allow-missing-active-classes", action="store_true",
                   help="active class YAML 에 없는 data-dir 폴더가 있어도 drop 하고 계속 진행")
    p.add_argument("--data-dir", type=str, default=DEFAULT_DATA_DIR)
    p.add_argument("--obj-id-dir", type=str, default=DEFAULT_OBJ_ID_DIR)
    p.add_argument("--obj-prob-dir", type=str, default=DEFAULT_OBJ_PROB_DIR,
                   help="V4 softmax probability map root")
    p.add_argument("--log-root", type=str, default=DEFAULT_LOG_ROOT)
    p.add_argument("--model-tag", type=str, default=None)
    return p.parse_args()


# === obj_id encoding ===
def encode_v1(arr, divisor=5.0):
    return (arr.astype(np.float32) / divisor)[None, :, :]


def encode_v2(arr, target_id=3):
    return (arr == target_id).astype(np.float32)[None, :, :]


def encode_v3(arr):
    out = np.zeros((5, *arr.shape), dtype=np.float32)
    for i in range(1, 6):
        out[i - 1] = (arr == i).astype(np.float32)
    return out


def get_obj_channels(arr, args, prob_arr=None):
    v = args.variant
    if v == "V0": return None
    if v == "V1": return encode_v1(arr, divisor=args.obj_norm)
    if v == "V2": return encode_v2(arr, target_id=args.target_id)
    if v == "V3": return encode_v3(arr)
    if v == "V4":
        if prob_arr is None:
            raise FileNotFoundError(
                "V4 requires softmax probability maps. Build them with "
                "_build_obj_id_maps.py --save-prob-maps and pass --obj-prob-dir."
            )
        return prob_arr.astype(np.float32)
    raise NotImplementedError(f"Variant {v} is not implemented in this script.")


def channels_for_variant(args):
    return {"V0": 0, "V1": 1, "V2": 1, "V3": 5, "V4": 5}.get(args.variant, 0)


# === Dataset ===
def build_npy_lookup(obj_id_dir):
    """Flat basename (uppercase) → npy path."""
    m = {}
    for p in Path(obj_id_dir).rglob("*.npy"):
        if p.name.startswith("_"):
            continue
        m[p.stem.upper()] = p
    return m


def _load_active_classes(yaml_path):
    """YAML 의 classes: list 반환. None 또는 파일 없으면 None."""
    if not yaml_path: return None
    try:
        import yaml
        with open(yaml_path, encoding="utf-8") as f:
            d = yaml.safe_load(f)
        return list(d.get("classes", []))
    except Exception as e:
        print(f"[!] active-classes-yaml load fail ({yaml_path}): {e} - fallback all classes")
        return None


def collect_samples(data_dir, n_per_class, active_classes=None, allow_missing_active_classes=False):
    """active_classes 명시 시 그 class 만 (순서 보존). 없으면 폴더명 알파벳 순 (기존 동작)."""
    all_dirs = sorted([d.name for d in Path(data_dir).iterdir()
                       if d.is_dir() and d.name not in EXCLUDE_CLASSES])
    if active_classes is not None:
        # active 의 class 중 실제 폴더 있는 것만 (yaml 의 list 순서 보존)
        all_dir_set = set(all_dirs)
        classes = [c for c in active_classes if c in all_dir_set]
        missing = [c for c in active_classes if c not in all_dir_set]
        if missing:
            msg = f"active classes not found in {data_dir}: {missing}"
            if not allow_missing_active_classes:
                raise ValueError(msg)
            print(f"[!] {msg}")
    else:
        classes = all_dirs
    cls_to_idx = {c: i for i, c in enumerate(classes)}
    samples = []
    for c in classes:
        files = sorted([str(p) for p in (Path(data_dir) / c).glob("*.png")])
        for f in files[:n_per_class]:
            samples.append((f, cls_to_idx[c]))
    return samples, classes


def split_samples(samples, seed, ratios=(0.8, 0.1, 0.1)):
    rng = random.Random(seed)
    indices = list(range(len(samples)))
    rng.shuffle(indices)
    n = len(indices)
    n_train = int(n * ratios[0])
    n_val = int(n * ratios[1])
    return ([samples[i] for i in indices[:n_train]],
            [samples[i] for i in indices[n_train:n_train + n_val]],
            [samples[i] for i in indices[n_train + n_val:]])


def _load_r_static(png_path):
    img = Image.open(png_path)
    if img.mode != "P":
        img = img.convert("P")
    arr = np.asarray(img, dtype=np.uint8)
    h, w = arr.shape
    cell_h = h // GRID_SIZE
    cell_w = w // GRID_SIZE
    a = arr[:cell_h * GRID_SIZE, :cell_w * GRID_SIZE]
    a = a.reshape(GRID_SIZE, cell_h, GRID_SIZE, cell_w)
    pooled = a.mean(axis=(1, 3))
    return (pooled / PALETTE_MAX).astype(np.float32)[None, :, :]


def _load_obj_static(png_path, npy_lookup):
    basename = Path(png_path).stem.upper()
    npy_path = npy_lookup.get(basename)
    if npy_path is None:
        return None
    arr = np.load(npy_path)
    if arr.shape != (GRID_SIZE, GRID_SIZE):
        pil = Image.fromarray(arr.astype(np.int32), mode="I").resize(
            (GRID_SIZE, GRID_SIZE), Image.NEAREST)
        arr = np.asarray(pil, dtype=np.uint8)
    return arr.astype(np.uint8)


def _load_obj_prob_static(png_path, prob_lookup, n_obj=5):
    basename = Path(png_path).stem.upper()
    npy_path = prob_lookup.get(basename)
    if npy_path is None:
        return None
    arr = np.load(npy_path).astype(np.float32)
    if arr.ndim != 3:
        raise ValueError(f"prob map must be 3D [C,H,W] or [H,W,C]: {npy_path} shape={arr.shape}")
    if arr.shape[0] != n_obj and arr.shape[-1] == n_obj:
        arr = np.transpose(arr, (2, 0, 1))
    if arr.shape[0] != n_obj:
        raise ValueError(f"prob map channel count mismatch: {npy_path} shape={arr.shape} expected C={n_obj}")
    if arr.shape[1:] != (GRID_SIZE, GRID_SIZE):
        t = torch.from_numpy(arr).unsqueeze(0)
        arr = F.interpolate(t, size=(GRID_SIZE, GRID_SIZE), mode="bilinear", align_corners=False).squeeze(0).numpy()
    return np.clip(arr, 0.0, 1.0).astype(np.float32)


def apply_chip_noise(obj_arr, prob, rng):
    """Randomly replace non-zero chips with random class id 1..5 (simulates chip CNN errors)."""
    if prob <= 0:
        return obj_arr
    out = obj_arr.copy()
    mask_nonzero = out > 0
    rand_swap = rng.random(out.shape) < prob
    swap_mask = mask_nonzero & rand_swap
    n_swap = swap_mask.sum()
    if n_swap > 0:
        random_ids = rng.integers(1, 6, size=int(n_swap), dtype=np.uint8)
        out[swap_mask] = random_ids
    return out


class ChipGridDataset(Dataset):
    """Loads R + obj_id once at __init__ (memory cache) — wafer PNG read 6400x6400 is slow."""

    def __init__(self, samples, npy_lookup, args, preload=True, log_fn=None,
                 prob_lookup=None,
                 noise_prob=0.0, noise_seed=None):
        self.samples = samples
        self.args = args
        self.preload = preload
        self.prob_lookup = prob_lookup or {}
        self.noise_prob = noise_prob
        # per-dataset rng so train/val/test each get own deterministic stream
        self._rng = np.random.default_rng(noise_seed if noise_seed is not None else args.seed)
        self._cache_r = None
        self._cache_obj = None
        self._cache_prob = None
        if preload:
            t0 = time.time()
            r_list = []
            obj_list = []
            prob_list = []
            n_missing = 0
            n_missing_prob = 0
            for png_path, _ in samples:
                r_list.append(_load_r_static(png_path))
                obj = _load_obj_static(png_path, npy_lookup)
                if obj is None:
                    obj = np.zeros((GRID_SIZE, GRID_SIZE), dtype=np.uint8)
                    n_missing += 1
                obj_list.append(obj)
                if args.variant == "V4":
                    prob = _load_obj_prob_static(png_path, self.prob_lookup, n_obj=5)
                    if prob is None:
                        n_missing_prob += 1
                    else:
                        prob_list.append(prob)
            self._cache_r = np.stack(r_list, axis=0)
            self._cache_obj = np.stack(obj_list, axis=0)
            if args.variant == "V4":
                if n_missing_prob:
                    raise FileNotFoundError(
                        f"V4 missing probability maps for {n_missing_prob}/{len(samples)} samples. "
                        f"Run _build_obj_id_maps.py --save-prob-maps or set --obj-prob-dir."
                    )
                self._cache_prob = np.stack(prob_list, axis=0)
            dt = time.time() - t0
            msg = f"[cache] loaded {len(samples)} samples in {dt:.1f}s  (missing obj_id: {n_missing}"
            if args.variant == "V4":
                msg += f", missing prob: {n_missing_prob}"
            msg += ")"
            if log_fn: log_fn(msg)
            else: print(msg, flush=True)
        else:
            self.npy_lookup = npy_lookup

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        if self.preload:
            r = self._cache_r[idx]
            obj_arr = self._cache_obj[idx]
            prob_arr = self._cache_prob[idx] if self._cache_prob is not None else None
        else:
            png_path, _ = self.samples[idx]
            r = _load_r_static(png_path)
            obj_arr = _load_obj_static(png_path, self.npy_lookup)
            if obj_arr is None:
                obj_arr = np.zeros((GRID_SIZE, GRID_SIZE), dtype=np.uint8)
            prob_arr = _load_obj_prob_static(png_path, self.prob_lookup, n_obj=5) if self.args.variant == "V4" else None
        if self.noise_prob > 0:
            obj_arr = apply_chip_noise(obj_arr, self.noise_prob, self._rng)
        obj_ch = get_obj_channels(obj_arr, self.args, prob_arr=prob_arr)
        if getattr(self.args, "no_r_channel", False):
            # ablation: R 채널 제외, obj 채널만 (obj_ch must not be None)
            if obj_ch is None:
                raise ValueError("--no-r-channel requires variant with obj channels (V1/V2/V3/V4)")
            x = obj_ch
        else:
            x = r if obj_ch is None else np.concatenate([r, obj_ch], axis=0)
        label = self.samples[idx][1]
        return torch.from_numpy(x).float(), torch.tensor(label, dtype=torch.long)


# === Model ===
class ChipGridCNN(nn.Module):
    def __init__(self, in_ch, n_classes, n_dist=0, n_obj=0):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(in_ch, 64, 3, padding=1), nn.BatchNorm2d(64), nn.GELU(),
            nn.Conv2d(64, 64, 3, padding=1), nn.BatchNorm2d(64), nn.GELU(),
            nn.Conv2d(64, 128, 3, padding=1, stride=2), nn.BatchNorm2d(128), nn.GELU(),
            nn.Conv2d(128, 128, 3, padding=1), nn.BatchNorm2d(128), nn.GELU(),
            nn.Conv2d(128, 256, 3, padding=1, stride=2), nn.BatchNorm2d(256), nn.GELU(),
            nn.Conv2d(256, 256, 3, padding=1), nn.BatchNorm2d(256), nn.GELU(),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Dropout(0.1),
        )
        self.classifier = nn.Linear(256, n_classes)
        self.dist_head = nn.Linear(256, n_dist) if n_dist > 0 else None
        self.obj_head = nn.Linear(256, n_obj) if n_obj > 0 else None

    def forward(self, x, return_aux=False):
        feat = self.encoder(x)
        logits = self.classifier(feat)
        if not return_aux:
            return logits
        out = {"logits": logits, "features": feat}
        if self.dist_head is not None:
            out["dist_logits"] = self.dist_head(feat)
        if self.obj_head is not None:
            out["obj_logits"] = self.obj_head(feat)
        return out


def parse_class_parts(class_name):
    if "_" not in class_name:
        return class_name, None
    dist, obj = class_name.split("_", 1)
    if obj not in CHIP_OBJ_LABELS:
        return dist, None
    return dist, obj


def build_factor_targets(classes):
    parts = [parse_class_parts(c) for c in classes]
    dist_classes = sorted({d for d, _ in parts})
    dist_to_idx = {d: i for i, d in enumerate(dist_classes)}
    obj_to_idx = {o: i for i, o in enumerate(CHIP_OBJ_LABELS)}
    label_to_dist = torch.tensor([dist_to_idx[d] for d, _ in parts], dtype=torch.long)
    label_to_obj = torch.tensor([obj_to_idx[o] if o is not None else -100 for _, o in parts], dtype=torch.long)
    return dist_classes, list(CHIP_OBJ_LABELS), label_to_dist, label_to_obj


def hard_contrastive_mask(classes, scope):
    mask = []
    for c in classes:
        dist, obj = parse_class_parts(c)
        if scope == "all":
            mask.append(True)
        elif scope == "object":
            mask.append(obj is not None)
        elif scope == "edge":
            mask.append(dist in {"Edge-Top", "Edge-Bottom"} and obj is not None)
        else:
            mask.append(False)
    return torch.tensor(mask, dtype=torch.bool)


def supervised_contrastive_loss(features, labels, temperature=0.2):
    if features.shape[0] < 2:
        return features.new_zeros(())
    features = F.normalize(features, dim=1)
    logits = torch.matmul(features, features.T) / max(float(temperature), 1e-6)
    logits = logits - logits.max(dim=1, keepdim=True).values.detach()
    self_mask = torch.eye(labels.shape[0], device=labels.device, dtype=torch.bool)
    pos_mask = labels[:, None].eq(labels[None, :]) & ~self_mask
    valid = pos_mask.sum(dim=1) > 0
    if not bool(valid.any()):
        return features.new_zeros(())
    exp_logits = torch.exp(logits) * (~self_mask).float()
    log_prob = logits - torch.log(exp_logits.sum(dim=1, keepdim=True).clamp_min(1e-12))
    mean_log_prob_pos = (pos_mask.float() * log_prob).sum(dim=1) / pos_mask.sum(dim=1).clamp_min(1)
    return -mean_log_prob_pos[valid].mean()


def compute_dist_obj_metrics(y_true, y_pred, classes):
    parts = [parse_class_parts(c) for c in classes]
    true_dist = np.array([parts[i][0] for i in y_true])
    pred_dist = np.array([parts[i][0] for i in y_pred])
    true_obj = np.array([parts[i][1] or "" for i in y_true])
    pred_obj = np.array([parts[i][1] or "" for i in y_pred])
    has_obj = true_obj != ""
    dist_ok = true_dist == pred_dist
    metrics = {
        "distribution_acc": float((true_dist == pred_dist).mean()) if len(y_true) else 0.0,
        "object_acc_all": float((true_obj[has_obj] == pred_obj[has_obj]).mean()) if has_obj.any() else 0.0,
        "object_acc_intra_dist": float((true_obj[has_obj & dist_ok] == pred_obj[has_obj & dist_ok]).mean())
        if (has_obj & dist_ok).any() else 0.0,
    }
    edge = has_obj & np.isin(true_dist, ["Edge-Top", "Edge-Bottom"])
    metrics["edge_object_acc"] = float((true_obj[edge] == pred_obj[edge]).mean()) if edge.any() else 0.0
    return metrics


# === EMA ===
class EMA:
    def __init__(self, model, decay=0.95, warmup_steps=0):
        self.decay = decay
        self.warmup_steps = warmup_steps
        self.step = 0
        self.shadow = {n: p.detach().clone() for n, p in model.named_parameters() if p.requires_grad}

    def update(self, model):
        self.step += 1
        if self.step < self.warmup_steps:
            return
        d = self.decay
        for n, p in model.named_parameters():
            if p.requires_grad:
                self.shadow[n].mul_(d).add_(p.detach(), alpha=1 - d)

    def apply_to(self, model):
        backup = {}
        for n, p in model.named_parameters():
            if n in self.shadow:
                backup[n] = p.detach().clone()
                p.data.copy_(self.shadow[n])
        return backup

    def restore(self, model, backup):
        for n, p in model.named_parameters():
            if n in backup:
                p.data.copy_(backup[n])


# === Eval ===
def evaluate(model, loader, device):
    model.eval()
    all_p, all_t = [], []
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device, non_blocking=True)
            out = model(x)
            logits = out["logits"] if isinstance(out, dict) else out
            all_p.append(logits.argmax(1).cpu().numpy())
            all_t.append(y.numpy())
    p = np.concatenate(all_p)
    t = np.concatenate(all_t)
    return accuracy_score(t, p), f1_score(t, p, average="macro", zero_division=0), t, p


def per_class_report(y_true, y_pred, classes):
    p, r, f, sup = precision_recall_fscore_support(
        y_true, y_pred, labels=range(len(classes)), zero_division=0)
    lines = [f"{'Class':<32}  {'F1':>6}  {'P':>6}  {'R':>6}  {'Sup':>5}", "-" * 70]
    for i, c in enumerate(classes):
        lines.append(f"{c:<32}  {f[i]:6.3f}  {p[i]:6.3f}  {r[i]:6.3f}  {int(sup[i]):5d}")
    lines.append("-" * 70)
    lines.append(f"{'macro avg':<32}  {f.mean():6.3f}  {p.mean():6.3f}  {r.mean():6.3f}  {int(sup.sum()):5d}")
    return "\n".join(lines)


# === Main ===
def main():
    args = parse_args()
    if args.variant == "V4" and args.chip_noise > 0:
        raise ValueError("V4 uses probability maps; --chip-noise is only defined for hard obj_id variants V1/V2/V3.")

    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(args.seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    ts = datetime.now().strftime("%y%m%d_%H%M%S")
    tag = args.model_tag or args.variant.lower()
    run_dir = Path(args.log_root) / f"{tag}_{ts}_running"
    run_dir.mkdir(parents=True, exist_ok=True)

    log_path = run_dir / "run.log"

    def log(msg):
        line = f"{datetime.now().strftime('%H:%M:%S')}  {msg}"
        print(line, flush=True)
        with open(log_path, "a", encoding="utf-8") as fp:
            fp.write(line + "\n")

    log("===== chipgrid eval start =====")
    log(f"variant={args.variant} target_id={args.target_id} obj_norm={args.obj_norm} "
        f"seed={args.seed} n_per_class={args.n_per_class} device={device}")
    log(f"aux_heads={args.aux_heads} dist_w={args.dist_loss_weight} obj_w={args.obj_loss_weight} "
        f"hard_contrastive_w={args.hard_contrastive_weight} scope={args.hard_contrastive_scope}")
    log(f"run_dir={run_dir}")

    active_classes = _load_active_classes(getattr(args, "active_classes_yaml", None))
    if active_classes:
        log(f"[active-classes] {len(active_classes)} classes from {args.active_classes_yaml}")
    samples, classes = collect_samples(
        args.data_dir,
        args.n_per_class,
        active_classes=active_classes,
        allow_missing_active_classes=args.allow_missing_active_classes,
    )
    n_classes = len(classes)
    log(f"classes ({n_classes}): {classes}")
    log(f"total samples: {len(samples)}")
    dist_classes, obj_classes, label_to_dist_cpu, label_to_obj_cpu = build_factor_targets(classes)
    hard_label_mask_cpu = hard_contrastive_mask(classes, args.hard_contrastive_scope)
    aux_enabled = (
        args.aux_heads == "factorized"
        or args.dist_loss_weight > 0
        or args.obj_loss_weight > 0
        or args.hard_contrastive_weight > 0
    )
    log(f"factor targets: dist={dist_classes} obj={obj_classes} aux_enabled={aux_enabled}")

    train_s, val_s, test_s = split_samples(samples, seed=args.seed)
    log(f"split: train={len(train_s)} val={len(val_s)} test={len(test_s)}")

    npy_lookup = build_npy_lookup(args.obj_id_dir)
    log(f"obj_id npy indexed: {len(npy_lookup)}")
    prob_lookup = {}
    if args.variant == "V4":
        prob_lookup = build_npy_lookup(args.obj_prob_dir)
        log(f"obj_prob npy indexed: {len(prob_lookup)}")

    eval_noise = args.chip_noise if args.chip_noise_eval else 0.0
    train_ds = ChipGridDataset(train_s, npy_lookup, args, preload=True, log_fn=log,
                               prob_lookup=prob_lookup,
                               noise_prob=args.chip_noise, noise_seed=args.seed)
    val_ds = ChipGridDataset(val_s, npy_lookup, args, preload=True, log_fn=log,
                             prob_lookup=prob_lookup,
                             noise_prob=eval_noise, noise_seed=args.seed + 1)
    test_ds = ChipGridDataset(test_s, npy_lookup, args, preload=True, log_fn=log,
                              prob_lookup=prob_lookup,
                              noise_prob=eval_noise, noise_seed=args.seed + 2)
    if args.chip_noise > 0:
        log(f"[chip-noise] train={args.chip_noise:.2f}  eval={eval_noise:.2f}")

    train_ld = DataLoader(train_ds, batch_size=args.batch, shuffle=True,
                          num_workers=0, pin_memory=True)
    val_ld = DataLoader(val_ds, batch_size=args.batch, shuffle=False, num_workers=0)
    test_ld = DataLoader(test_ds, batch_size=args.batch, shuffle=False, num_workers=0)

    obj_ch_count = channels_for_variant(args)
    in_ch = obj_ch_count if args.no_r_channel else (1 + obj_ch_count)
    model = ChipGridCNN(
        in_ch,
        n_classes,
        n_dist=len(dist_classes) if aux_enabled else 0,
        n_obj=len(obj_classes) if aux_enabled else 0,
    ).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    log(f"model: in_ch={in_ch} out={n_classes} params={n_params:,}")

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    crit = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)
    dist_crit = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)
    obj_crit = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing, ignore_index=-100)
    label_to_dist = label_to_dist_cpu.to(device)
    label_to_obj = label_to_obj_cpu.to(device)
    hard_label_mask = hard_label_mask_cpu.to(device)

    steps_per_epoch = max(1, len(train_ld))
    ema = EMA(model, decay=args.ema_decay, warmup_steps=steps_per_epoch * args.ema_warmup)

    hp = vars(args).copy()
    hp.update({"in_ch": in_ch, "n_classes": n_classes, "classes": classes,
               "dist_classes": dist_classes, "obj_classes": obj_classes,
               "n_params": n_params, "device": device,
               "n_train": len(train_s), "n_val": len(val_s), "n_test": len(test_s),
               "run_ts": ts, "torch_version": torch.__version__})
    with open(run_dir / "hparams.json", "w", encoding="utf-8") as fp:
        json.dump(hp, fp, indent=2, default=str)

    history = []
    best_val_f1 = -1.0
    best_state = {}
    best_updates = []
    epochs_no_improve = 0

    for ep in range(1, args.epochs + 1):
        model.train()
        t0 = time.time()
        ep_loss = ep_correct = ep_total = 0
        for x, y in train_ld:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            out = model(x, return_aux=aux_enabled)
            if isinstance(out, dict):
                logits = out["logits"]
            else:
                logits = out
            loss = crit(logits, y)
            if aux_enabled and args.dist_loss_weight > 0:
                loss = loss + args.dist_loss_weight * dist_crit(out["dist_logits"], label_to_dist[y])
            if aux_enabled and args.obj_loss_weight > 0:
                obj_targets = label_to_obj[y]
                if bool((obj_targets != -100).any()):
                    loss = loss + args.obj_loss_weight * obj_crit(out["obj_logits"], obj_targets)
            if aux_enabled and args.hard_contrastive_weight > 0:
                selected = hard_label_mask[y]
                if bool(selected.sum() >= 2):
                    loss = loss + args.hard_contrastive_weight * supervised_contrastive_loss(
                        out["features"][selected],
                        y[selected],
                        temperature=args.hard_contrastive_temp,
                    )
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
            opt.step()
            ema.update(model)
            ep_loss += loss.item() * y.size(0)
            ep_correct += (logits.argmax(1) == y).sum().item()
            ep_total += y.size(0)
        tr_loss = ep_loss / max(1, ep_total)
        tr_acc = ep_correct / max(1, ep_total)

        backup = ema.apply_to(model)
        val_acc, val_f1, y_v_t, y_v_p = evaluate(model, val_ld, device)
        val_do = compute_dist_obj_metrics(y_v_t, y_v_p, classes)
        is_best = val_f1 > best_val_f1
        if is_best:
            test_acc, test_f1, y_t_t, y_t_p = evaluate(model, test_ld, device)
            test_do = compute_dist_obj_metrics(y_t_t, y_t_p, classes)
            torch.save({"model_state": model.state_dict(), "classes": classes, "in_ch": in_ch,
                        "val_f1": val_f1, "val_acc": val_acc,
                        "test_f1": test_f1, "test_acc": test_acc,
                        "val_dist_obj_metrics": val_do,
                        "test_dist_obj_metrics": test_do,
                        "epoch": ep, "args": vars(args)},
                       run_dir / "best_model.pth")
        ema.restore(model, backup)

        log(f"[Ep {ep}/{args.epochs}] tr_loss={tr_loss:.4f} tr_acc={tr_acc*100:.2f}% | "
            f"val acc={val_acc*100:.2f}% f1={val_f1*100:.2f}% "
            f"intra_obj={val_do['object_acc_intra_dist']*100:.2f}% "
            f"edge_obj={val_do['edge_object_acc']*100:.2f}% | dt={time.time()-t0:.1f}s")

        history.append({"epoch": ep, "tr_loss": tr_loss, "tr_acc": tr_acc,
                        "val_acc": val_acc, "val_f1": val_f1,
                        "val_dist_obj_metrics": val_do})

        if is_best:
            best_val_f1 = val_f1
            best_state = {"epoch": ep, "val_f1": val_f1, "val_acc": val_acc,
                          "test_f1": test_f1, "test_acc": test_acc,
                          "val_dist_obj_metrics": val_do,
                          "test_dist_obj_metrics": test_do,
                          "y_v_t": y_v_t, "y_v_p": y_v_p,
                          "y_t_t": y_t_t, "y_t_p": y_t_p}
            best_updates.append({"epoch": ep, "val_f1": val_f1, "val_acc": val_acc,
                                  "test_f1": test_f1, "test_acc": test_acc,
                                  "val_intra_obj": val_do["object_acc_intra_dist"],
                                  "test_intra_obj": test_do["object_acc_intra_dist"],
                                  "val_edge_obj": val_do["edge_object_acc"],
                                  "test_edge_obj": test_do["edge_object_acc"]})
            log(f"  [best] val acc={val_acc*100:.2f}% f1={val_f1*100:.2f}% | "
                f"test acc={test_acc*100:.2f}% f1={test_f1*100:.2f}% "
                f"intra_obj={test_do['object_acc_intra_dist']*100:.2f}% "
                f"edge_obj={test_do['edge_object_acc']*100:.2f}%")
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1

        with open(run_dir / "history.json", "w", encoding="utf-8") as fp:
            json.dump(history, fp, indent=2)

        if epochs_no_improve >= args.patience:
            log(f"early stop at epoch {ep} (patience={args.patience})")
            break

    # write best_history.txt
    bs = best_state
    lines = [
        "=" * 90,
        f"** BEST OVERALL  |  variant={args.variant}  |  epoch {bs['epoch']}  |  val F1 = {bs['val_f1']:.4f}",
        "=" * 90,
        f"TEST  acc= {bs['test_acc']*100:.2f}%   f1= {bs['test_f1']*100:.2f}%",
        f"VAL   acc= {bs['val_acc']*100:.2f}%   f1= {bs['val_f1']*100:.2f}%",
        f"TEST  intra_obj= {bs['test_dist_obj_metrics']['object_acc_intra_dist']*100:.2f}%   "
        f"edge_obj= {bs['test_dist_obj_metrics']['edge_object_acc']*100:.2f}%",
        f"VAL   intra_obj= {bs['val_dist_obj_metrics']['object_acc_intra_dist']*100:.2f}%   "
        f"edge_obj= {bs['val_dist_obj_metrics']['edge_object_acc']*100:.2f}%",
        "",
        "=" * 90,
        "[1] FINAL BEST per-class (TEST)",
        "=" * 90,
        per_class_report(bs["y_t_t"], bs["y_t_p"], classes),
        "",
        "[1b] FINAL BEST per-class (VAL)",
        "-" * 90,
        per_class_report(bs["y_v_t"], bs["y_v_p"], classes),
        "",
        "=" * 90,
        "[2] BEST UPDATES SUMMARY",
        "=" * 90,
        f"{'ep':>4}  {'val_f1':>8}  {'val_acc':>8}  {'test_f1':>8}  {'test_acc':>8}  "
        f"{'val_intra':>9}  {'test_intra':>10}  {'val_edge':>8}  {'test_edge':>9}",
    ]
    for u in best_updates:
        lines.append(f"{u['epoch']:>4}  {u['val_f1']*100:>7.2f}%  {u['val_acc']*100:>7.2f}%  "
                     f"{u['test_f1']*100:>7.2f}%  {u['test_acc']*100:>7.2f}%  "
                     f"{u['val_intra_obj']*100:>8.2f}%  {u['test_intra_obj']*100:>9.2f}%  "
                     f"{u['val_edge_obj']*100:>7.2f}%  {u['test_edge_obj']*100:>8.2f}%")
    with open(run_dir / "best_history.txt", "w", encoding="utf-8") as fp:
        fp.write("\n".join(lines))

    final_name = run_dir.parent / f"{tag}_{ts}_{bs['test_f1']:.2f}_{bs['val_f1']:.2f}"
    log(f"renaming run dir to: {final_name.name}")
    try:
        run_dir.rename(final_name)
        print(f"run dir renamed OK: {final_name.name}", flush=True)
    except Exception as e:
        print(f"run dir rename failed: {e}", flush=True)

    print(f"\n=== DONE ===\nvariant={args.variant}  BEST val_f1={bs['val_f1']:.4f}  "
          f"test_f1={bs['test_f1']:.4f}  epoch={bs['epoch']}")


if __name__ == "__main__":
    main()
