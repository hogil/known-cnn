#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ObjID mid-fusion ablation training for wafer 33/34-class classification.

This entry point keeps the existing wafer/compound trainers unchanged and
writes to logs_objid_ablation by default. It supports:
  - R-only baseline
  - current input-compound baseline (R + scalar obj_id G channel)
  - mid-fusion ObjID embedding / one-hot / binary adapters at S2/S3
  - HIST-BAND late/head fusion
"""
from __future__ import annotations

import argparse
import json
import logging
import random
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

try:
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
except Exception:
    pass

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
from PIL import Image
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    precision_recall_fscore_support,
)
from torch.utils.data import DataLoader, Dataset, Subset, WeightedRandomSampler
from torchvision import transforms
from torchvision.transforms import functional as TF

import timm
from _resource_guard import ResourceMonitor, assess_start, format_assessment
from cnn_train_compound import (
    BACKBONE,
    CFG as COMPOUND_CFG,
    DEFAULT_DATA_DIR,
    DEFAULT_OBJ_ID_DIR,
    EXCLUDE_CLASSES,
    PALETTE_IDX_NORM,
    EMA,
    FilteredImageFolder,
    FocalLoss,
    apply_subset,
    build_model,
    compute_class_weights,
    load_subset_config,
    rename_run_dir,
    save_confusion_matrix,
    save_confusion_matrix_combined,
    save_curves_png,
    save_pred_samples,
    save_wrong_tree,
    setup_logger,
    stratified_split,
    update_overall_best,
    write_best_history,
)


DEFAULT_LOG_ROOT = "logs_objid_ablation"
RUN_TS = datetime.now().strftime("%y%m%d_%H%M%S")

EDGE_OBJECT_CLASSES = [                                                                  # round 26: particle_blast→fork, scratch_21deg→scratch_rot
    "Edge-Top_bank_boundary",
    "Edge-Top_fork",
    "Edge-Top_scratch",
    "Edge-Top_scratch_rot",
    "Edge-Bottom_bank_boundary",
    "Edge-Bottom_fork",
    "Edge-Bottom_scratch",
    "Edge-Bottom_scratch_rot",
]

FALLBACK_OBJ_ID_TO_LABEL = [                                                             # round 26
    "none",
    "bank_boundary",
    "invalid_main",
    "fork",
    "scratch",
    "scratch_rot",
]


@dataclass(frozen=True)
class CandidateConfig:
    name: str
    input_mode: str = "r-only"          # r-only | input-compound
    fusion_encoding: Optional[str] = None  # embedding | onehot | binary
    fusion_stage: Optional[int] = None     # timm stages index: 1 => 48x48, 2 => 24x24
    embedding_dim: int = 8
    binary_label: Optional[str] = None
    use_hist: bool = False


def _candidate_table() -> Dict[str, CandidateConfig]:
    configs = [
        CandidateConfig("R-only"),
        CandidateConfig("input-compound-current", input_mode="input-compound"),
        CandidateConfig("E8-S2", fusion_encoding="embedding", fusion_stage=1),
        CandidateConfig("E8-S3", fusion_encoding="embedding", fusion_stage=2),
        CandidateConfig("OH-S2", fusion_encoding="onehot", fusion_stage=1),
        CandidateConfig("OH-S3", fusion_encoding="onehot", fusion_stage=2),
        CandidateConfig("BIN-bank_boundary-S3", fusion_encoding="binary", fusion_stage=2, binary_label="bank_boundary"),
        CandidateConfig("BIN-fork-S3", fusion_encoding="binary", fusion_stage=2, binary_label="fork"),                    # round 26
        CandidateConfig("BIN-scratch-S3", fusion_encoding="binary", fusion_stage=2, binary_label="scratch"),
        CandidateConfig("BIN-scratch_rot-S3", fusion_encoding="binary", fusion_stage=2, binary_label="scratch_rot"),      # round 26
        CandidateConfig("BIN-invalid_main-S3", fusion_encoding="binary", fusion_stage=2, binary_label="invalid_main"),
        CandidateConfig("HIST-BAND", use_hist=True),
        CandidateConfig("OH-S3-HIST", fusion_encoding="onehot", fusion_stage=2, use_hist=True),
    ]
    out: Dict[str, CandidateConfig] = {}
    for cfg in configs:
        out[cfg.name.lower()] = cfg
        out[cfg.name.replace("-", "_").lower()] = cfg
    out["ronly"] = configs[0]
    out["r_only"] = configs[0]
    out["input_compound"] = configs[1]
    out["compound"] = configs[1]
    return out


def resolve_candidate(name: str) -> CandidateConfig:
    table = _candidate_table()
    key = name.strip().lower()
    if key not in table:
        valid = sorted({cfg.name for cfg in table.values()})
        raise ValueError(f"unknown candidate {name!r}; valid: {', '.join(valid)}")
    return table[key]


def read_obj_meta(obj_id_dir: str) -> Tuple[int, List[str]]:
    meta_path = Path(obj_id_dir) / "_meta.json"
    if meta_path.exists():
        with meta_path.open("r", encoding="utf-8") as f:
            meta = json.load(f)
        n_chip_objects = int(meta.get("n_chip_objects", 5))
        obj_id_to_label = list(meta.get("obj_id_to_label") or [])
        if len(obj_id_to_label) != n_chip_objects + 1:
            chip_classes = list(meta.get("chip_classes") or [])
            obj_id_to_label = ["none"] + chip_classes
        if len(obj_id_to_label) != n_chip_objects + 1:
            obj_id_to_label = FALLBACK_OBJ_ID_TO_LABEL[: n_chip_objects + 1]
        return n_chip_objects, obj_id_to_label
    return 5, FALLBACK_OBJ_ID_TO_LABEL.copy()


def _yaml_safe(v):
    if v is None or isinstance(v, (bool, int, float)):
        return v
    if type(v) is str:
        return v
    if isinstance(v, str):
        return str(v)
    if isinstance(v, (list, tuple)):
        return [_yaml_safe(x) for x in v]
    if isinstance(v, dict):
        return {str(k): _yaml_safe(x) for k, x in v.items()}
    return str(v)


class ObjIdFusionImageFolder(FilteredImageFolder):
    """ImageFolder that returns R-only image input plus raw ObjID side inputs."""

    _IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32).view(3, 1, 1)
    _IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32).view(3, 1, 1)

    def __init__(
        self,
        root: str,
        obj_id_root: str,
        img_size: int,
        train_aug: bool,
        n_chip_objects: int,
        obj_id_to_label: List[str],
        input_mode: str,
    ):
        super().__init__(root)
        self.obj_id_root = Path(obj_id_root)
        self.img_size = int(img_size)
        self.train_aug = bool(train_aug)
        self.n_chip_objects = int(n_chip_objects)
        self.obj_id_to_label = list(obj_id_to_label)
        self.input_mode = str(input_mode)
        self._missing_obj_id_warned = False
        self._npy_map: Dict[str, Path] = {}
        self._default_grid_shape: Optional[Tuple[int, int]] = None

        if self.obj_id_root.exists():
            for p in self.obj_id_root.rglob("*.npy"):
                if p.name == "_meta.npy":
                    continue
                self._npy_map[p.stem] = p

        if self._npy_map:
            first = next(iter(self._npy_map.values()))
            try:
                arr = np.load(first)
                if arr.ndim == 2:
                    self._default_grid_shape = (int(arr.shape[0]), int(arr.shape[1]))
            except Exception:
                self._default_grid_shape = None

        if self._default_grid_shape is None:
            self._default_grid_shape = (32, 32)
            print("[ObjIdFusionImageFolder] no obj_id shape discovered; fallback grid=(32,32)", flush=True)

        print(
            f"[ObjIdFusionImageFolder] obj_id npy lookup: {len(self._npy_map)} files indexed under {self.obj_id_root}",
            flush=True,
        )

    def _load_r_channel(self, png_path: str) -> torch.Tensor:
        img = Image.open(png_path)
        if img.mode != "P":
            img = img.convert("P")
        idx = np.asarray(img, dtype=np.uint8)
        idx_pil = Image.fromarray(idx)
        idx_resized = idx_pil.resize((self.img_size, self.img_size), Image.BICUBIC)
        r = torch.from_numpy(np.asarray(idx_resized, dtype=np.float32) / float(PALETTE_IDX_NORM))
        return r.clamp_(0.0, 1.0).unsqueeze(0)

    def _load_obj_id(self, png_path: str) -> torch.Tensor:
        basename = Path(png_path).stem
        obj_path = self._npy_map.get(basename)
        if obj_path is not None and obj_path.exists():
            arr = np.load(obj_path).astype(np.int64)
        else:
            arr = np.zeros(self._default_grid_shape, dtype=np.int64)
            if not self._missing_obj_id_warned:
                print(
                    f"[ObjIdFusionImageFolder] missing obj_id for basename={basename!r}; using zeros (warn once)",
                    flush=True,
                )
                self._missing_obj_id_warned = True
        arr = np.clip(arr, 0, self.n_chip_objects)
        return torch.from_numpy(arr).long()

    def _obj_scalar_to_image(self, obj_id: torch.Tensor) -> torch.Tensor:
        arr = obj_id.cpu().numpy().astype(np.uint8)
        obj_pil = Image.fromarray(arr)
        obj_resized = obj_pil.resize((self.img_size, self.img_size), Image.BICUBIC)
        g = torch.from_numpy(np.asarray(obj_resized, dtype=np.float32) / float(max(1, self.n_chip_objects)))
        return g.clamp_(0.0, 1.0).unsqueeze(0)

    def _sample_aug_params(self) -> Optional[Tuple[float, float, float, float]]:
        if not self.train_aug:
            return None
        angle = float(torch.empty(1).uniform_(-15.0, 15.0).item())
        sx = float(torch.empty(1).uniform_(-0.03, 0.03).item())
        sy = float(torch.empty(1).uniform_(-0.03, 0.03).item())
        scale = float(torch.empty(1).uniform_(0.97, 1.03).item())
        return angle, sx, sy, scale

    def _augment_image(self, x: torch.Tensor, params: Optional[Tuple[float, float, float, float]]) -> torch.Tensor:
        if params is None:
            return x
        angle, sx, sy, scale = params
        x = TF.rotate(x, angle, interpolation=transforms.InterpolationMode.BILINEAR, fill=0)
        tx = int(round(sx * self.img_size))
        ty = int(round(sy * self.img_size))
        x = TF.affine(
            x,
            angle=0.0,
            translate=(tx, ty),
            scale=scale,
            shear=[0.0, 0.0],
            interpolation=transforms.InterpolationMode.BILINEAR,
            fill=0,
        )
        noise = torch.randn_like(x[0:1]) * 0.01
        x = torch.cat([(x[0:1] + noise).clamp_(0.0, 1.0), x[1:]], dim=0)
        return x

    def _augment_obj_id(self, obj_id: torch.Tensor, params: Optional[Tuple[float, float, float, float]]) -> torch.Tensor:
        if params is None:
            return obj_id
        angle, sx, sy, scale = params
        h, w = int(obj_id.shape[-2]), int(obj_id.shape[-1])
        x = obj_id.float().unsqueeze(0)
        x = TF.rotate(x, angle, interpolation=transforms.InterpolationMode.NEAREST, fill=0)
        tx = int(round(sx * w))
        ty = int(round(sy * h))
        x = TF.affine(
            x,
            angle=0.0,
            translate=(tx, ty),
            scale=scale,
            shear=[0.0, 0.0],
            interpolation=transforms.InterpolationMode.NEAREST,
            fill=0,
        )
        return x.squeeze(0).round().clamp_(0, self.n_chip_objects).long()

    def _hist_band(self, obj_id: torch.Tensor) -> torch.Tensor:
        h = int(obj_id.shape[-2])
        band = max(1, int(round(h * 0.25)))
        regions = [
            obj_id[:band, :],
            obj_id[h - band :, :],
            obj_id,
        ]
        vals: List[float] = []
        for region in regions:
            denom = float(max(1, region.numel()))
            for k in range(1, self.n_chip_objects + 1):
                vals.append(float((region == k).sum().item()) / denom)
        for region in regions:
            denom = float(max(1, region.numel()))
            vals.append(float((region > 0).sum().item()) / denom)
        return torch.tensor(vals, dtype=torch.float32)

    def _normalize(self, x: torch.Tensor) -> torch.Tensor:
        return (x - self._IMAGENET_MEAN) / self._IMAGENET_STD

    def __getitem__(self, index):
        png_path, label = self.samples[index]
        obj_id = self._load_obj_id(png_path)
        r = self._load_r_channel(png_path)

        if self.input_mode == "input-compound":
            g = self._obj_scalar_to_image(obj_id)
            image = torch.cat([r, g, torch.zeros_like(r)], dim=0)
        else:
            image = torch.cat([r, torch.zeros_like(r), torch.zeros_like(r)], dim=0)

        params = self._sample_aug_params()
        image = self._augment_image(image, params)
        obj_id = self._augment_obj_id(obj_id, params)
        hist = self._hist_band(obj_id)
        image = self._normalize(image)

        return {"image": image, "obj_id": obj_id, "hist": hist}, label


class ObjIdPathSubset(Dataset):
    def __init__(self, base: ObjIdFusionImageFolder, indices: List[int]):
        self.base = base
        self.indices = list(indices)

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, i):
        idx = self.indices[i]
        payload, label = self.base[idx]
        path = self.base.samples[idx][0]
        return payload, label, path


class ResidualObjAdapter(nn.Module):
    def __init__(self, feature_channels: int, obj_channels: int):
        super().__init__()
        self.conv1 = nn.Conv2d(feature_channels + obj_channels, feature_channels, kernel_size=1)
        self.act = nn.GELU()
        self.conv2 = nn.Conv2d(feature_channels, feature_channels, kernel_size=1)
        self.alpha = nn.Parameter(torch.zeros(()))
        nn.init.zeros_(self.conv2.weight)
        nn.init.zeros_(self.conv2.bias)

    def forward(self, x: torch.Tensor, obj_feat: torch.Tensor) -> torch.Tensor:
        delta = self.conv2(self.act(self.conv1(torch.cat([x, obj_feat], dim=1))))
        return x + (1.0 + self.alpha) * delta


class ObjIdMidFusionConvNeXt(nn.Module):
    def __init__(
        self,
        backbone: nn.Module,
        candidate: CandidateConfig,
        n_chip_objects: int,
        obj_label_to_id: Dict[str, int],
        hist_dim: int,
    ):
        super().__init__()
        self.backbone = backbone
        self.candidate = candidate
        self.n_chip_objects = int(n_chip_objects)
        self.hist_dim = int(hist_dim) if candidate.use_hist else 0
        self.fusion_stage = candidate.fusion_stage
        self.fusion_encoding = candidate.fusion_encoding
        self.binary_obj_id = 0

        self.obj_embedding: Optional[nn.Embedding] = None
        self.fusion_adapter: Optional[ResidualObjAdapter] = None

        if self.fusion_encoding is not None:
            if self.fusion_stage not in (1, 2):
                raise ValueError("fusion_stage must be 1 (S2/48x48) or 2 (S3/24x24)")
            feature_channels = self._stage_channels(self.fusion_stage)
            if self.fusion_encoding == "embedding":
                self.obj_embedding = nn.Embedding(self.n_chip_objects + 1, candidate.embedding_dim, padding_idx=0)
                obj_channels = candidate.embedding_dim
            elif self.fusion_encoding == "onehot":
                obj_channels = self.n_chip_objects + 1  # object masks + defect mask
            elif self.fusion_encoding == "binary":
                if not candidate.binary_label:
                    raise ValueError("binary candidate requires binary_label")
                if candidate.binary_label not in obj_label_to_id:
                    raise ValueError(f"binary obj label missing from meta: {candidate.binary_label}")
                self.binary_obj_id = int(obj_label_to_id[candidate.binary_label])
                obj_channels = 1
            else:
                raise ValueError(f"unknown fusion encoding: {self.fusion_encoding}")
            self.fusion_adapter = ResidualObjAdapter(feature_channels, obj_channels)

        self.hist_fc: Optional[nn.Linear] = None
        if self.hist_dim > 0:
            num_features = int(getattr(self.backbone, "num_features"))
            num_classes = int(self.backbone.head.fc.out_features)
            self.hist_fc = nn.Linear(num_features + self.hist_dim, num_classes)
            self.sync_hist_fc_from_backbone_head()

    def _stage_channels(self, stage_idx: int) -> int:
        info = getattr(self.backbone, "feature_info", None)
        if isinstance(info, list) and stage_idx < len(info):
            return int(info[stage_idx]["num_chs"])
        with torch.no_grad():
            x = torch.zeros(1, 3, 384, 384)
            x = self.backbone.stem(x)
            for i, stage in enumerate(self.backbone.stages):
                x = stage(x)
                if i == stage_idx:
                    return int(x.shape[1])
        raise ValueError(f"could not infer feature channels for stage {stage_idx}")

    def sync_hist_fc_from_backbone_head(self) -> None:
        if self.hist_fc is None:
            return
        src = getattr(self.backbone.head, "fc", None)
        if src is None:
            return
        with torch.no_grad():
            self.hist_fc.weight.zero_()
            self.hist_fc.bias.zero_()
            n = min(self.hist_fc.weight.shape[1], src.weight.shape[1])
            self.hist_fc.weight[:, :n].copy_(src.weight[:, :n])
            if src.bias is not None:
                self.hist_fc.bias.copy_(src.bias)

    def _resize_obj_feat(self, feat: torch.Tensor, target_hw: Tuple[int, int]) -> torch.Tensor:
        if self.fusion_stage == 1:
            return F.interpolate(feat, size=target_hw, mode="bilinear", align_corners=False)
        if self.fusion_stage == 2:
            return F.interpolate(feat, size=target_hw, mode="area")
        raise ValueError(f"unsupported fusion_stage={self.fusion_stage}")

    def _encode_obj(self, obj_id: torch.Tensor, target_hw: Tuple[int, int]) -> torch.Tensor:
        obj_id = obj_id.long().clamp(0, self.n_chip_objects)
        if self.fusion_encoding == "embedding":
            assert self.obj_embedding is not None
            emb = self.obj_embedding(obj_id).permute(0, 3, 1, 2).contiguous()
            return self._resize_obj_feat(emb, target_hw)
        if self.fusion_encoding == "onehot":
            oh = F.one_hot(obj_id, num_classes=self.n_chip_objects + 1).permute(0, 3, 1, 2).float()
            obj_masks = oh[:, 1:, :, :]
            defect_mask = (obj_id > 0).float().unsqueeze(1)
            return self._resize_obj_feat(torch.cat([obj_masks, defect_mask], dim=1), target_hw)
        if self.fusion_encoding == "binary":
            mask = (obj_id == self.binary_obj_id).float().unsqueeze(1)
            return self._resize_obj_feat(mask, target_hw)
        raise ValueError("no obj encoding configured")

    def _pre_logits(self, x: torch.Tensor) -> torch.Tensor:
        head = self.backbone.head
        x = head.global_pool(x)
        x = head.norm(x)
        x = head.flatten(x)
        x = head.pre_logits(x)
        x = head.drop(x)
        return x

    def forward(self, payload: Dict[str, torch.Tensor]) -> torch.Tensor:
        x = self.backbone.stem(payload["image"])
        obj_id = payload.get("obj_id")
        for i, stage in enumerate(self.backbone.stages):
            x = stage(x)
            if self.fusion_adapter is not None and i == self.fusion_stage:
                if obj_id is None:
                    raise ValueError("obj_id tensor is required for mid-fusion candidates")
                obj_feat = self._encode_obj(obj_id, (int(x.shape[-2]), int(x.shape[-1])))
                x = self.fusion_adapter(x, obj_feat)

        if self.hist_fc is not None:
            hist = payload.get("hist")
            if hist is None:
                raise ValueError("hist tensor is required for HIST candidates")
            pre = self._pre_logits(x)
            hist = hist.to(dtype=pre.dtype)
            return self.hist_fc(torch.cat([pre, hist], dim=1))
        return self.backbone.head(x)


def move_payload(payload: Dict[str, torch.Tensor], device: torch.device, channels_last: bool) -> Dict[str, torch.Tensor]:
    out: Dict[str, torch.Tensor] = {}
    for k, v in payload.items():
        if k == "image":
            if channels_last:
                out[k] = v.to(device, non_blocking=True, memory_format=torch.channels_last)
            else:
                out[k] = v.to(device, non_blocking=True)
        else:
            out[k] = v.to(device, non_blocking=True)
    return out


def compute_edge_metrics(labels: np.ndarray, preds: np.ndarray, classes: List[str]) -> Dict[str, object]:
    edge_idx = [classes.index(c) for c in EDGE_OBJECT_CLASSES if c in classes]
    if not edge_idx:
        return {
            "edge_object_f1": float("nan"),
            "edge_object_swap_errors": 0,
            "edge_object_support": 0,
            "edge_object_classes": [],
        }
    _, _, f1, _ = precision_recall_fscore_support(
        labels,
        preds,
        labels=edge_idx,
        average="macro",
        zero_division=0,
    )
    cm = confusion_matrix(labels, preds, labels=edge_idx)
    swaps = int(cm.sum() - np.trace(cm))
    support = int(np.isin(labels, edge_idx).sum())
    return {
        "edge_object_f1": float(f1),
        "edge_object_swap_errors": swaps,
        "edge_object_support": support,
        "edge_object_classes": [classes[i] for i in edge_idx],
    }


def train_one_epoch_objid(
    model,
    loader,
    opt,
    scaler,
    scheduler,
    device,
    lg,
    ep,
    total_ep,
    criterion,
    ema: Optional[EMA],
    grad_clip: float,
):
    model.train()
    losses = []
    n_correct = 0
    n_total = 0
    tick = max(1, len(loader) // 10)
    use_amp = device.type == "cuda"
    amp_dtype = torch.bfloat16 if use_amp and torch.cuda.is_bf16_supported() else torch.float16
    channels_last = device.type == "cuda"
    for it, (payload, lbls) in enumerate(loader, 1):
        payload = move_payload(payload, device, channels_last)
        lbls = lbls.to(device, non_blocking=True)
        opt.zero_grad(set_to_none=True)
        with torch.amp.autocast("cuda", enabled=use_amp, dtype=amp_dtype):
            logits = model(payload)
            loss = criterion(logits, lbls)
        scaler.scale(loss).backward()
        if grad_clip and grad_clip > 0:
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        scaler.step(opt)
        scaler.update()
        if scheduler is not None:
            scheduler.step()
        if ema is not None:
            ema.update(model)
        losses.append(float(loss.detach().cpu()))
        with torch.no_grad():
            preds = logits.argmax(1)
        n_correct += int((preds == lbls).sum().item())
        n_total += int(lbls.numel())
        if it % tick == 0 or it == len(loader):
            lg.info(
                f"  Ep {ep}/{total_ep} | {100 * it / len(loader):5.1f}% | "
                f"loss={np.mean(losses):.4f} acc={100 * n_correct / max(1, n_total):.2f}%"
            )
    return float(np.mean(losses)), n_correct / max(1, n_total)


@torch.no_grad()
def evaluate_objid(model, loader, device, classes, criterion=None, return_paths: bool = False):
    model.eval()
    all_p, all_l, all_conf, all_paths = [], [], [], []
    all_loss: List[float] = []
    channels_last = device.type == "cuda"
    for batch in loader:
        if return_paths and len(batch) == 3:
            payload, lbls, paths = batch
        else:
            payload, lbls = batch[0], batch[1]
            paths = None
        payload = move_payload(payload, device, channels_last)
        lbls_dev = lbls.to(device, non_blocking=True)
        logits = model(payload)
        if criterion is not None:
            loss = criterion(logits, lbls_dev)
            all_loss.append(float(loss.detach().cpu()) * int(lbls.numel()))
        probs = F.softmax(logits, dim=1)
        confs, preds = probs.max(dim=1)
        all_p.append(preds.cpu().numpy())
        all_l.append(lbls.numpy())
        all_conf.append(confs.cpu().numpy())
        if paths is not None:
            all_paths.extend(list(paths))

    preds_np = np.concatenate(all_p)
    labels_np = np.concatenate(all_l)
    confs_np = np.concatenate(all_conf)
    acc = accuracy_score(labels_np, preds_np)
    p, r, f1, _ = precision_recall_fscore_support(labels_np, preds_np, average="macro", zero_division=0)
    res = {
        "acc": float(acc),
        "macro_p": float(p),
        "macro_r": float(r),
        "macro_f1": float(f1),
        "preds": preds_np.tolist(),
        "labels": labels_np.tolist(),
        "confs": confs_np.tolist(),
        "classes": classes,
    }
    if all_loss:
        res["val_loss"] = float(sum(all_loss) / max(1, len(labels_np)))
    if return_paths:
        res["paths"] = all_paths
    res.update(compute_edge_metrics(labels_np, preds_np, classes))
    return res


def metric_brief(res: Optional[dict]) -> Optional[dict]:
    if res is None:
        return None
    return {
        "acc": float(res["acc"]),
        "macro_f1": float(res["macro_f1"]),
        "macro_p": float(res["macro_p"]),
        "macro_r": float(res["macro_r"]),
        "edge_object_f1": float(res.get("edge_object_f1", float("nan"))),
        "edge_object_swap_errors": int(res.get("edge_object_swap_errors", 0)),
        "edge_object_support": int(res.get("edge_object_support", 0)),
    }


def load_init_from(model: ObjIdMidFusionConvNeXt, path: str, num_classes: int, lg: logging.Logger) -> None:
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    sd_full = ckpt.get("model", ckpt)
    model_sd = model.state_dict()
    compat = {}
    for k, v in sd_full.items():
        candidate_keys = [k]
        if not k.startswith("backbone."):
            candidate_keys.append("backbone." + k)
        for mk in candidate_keys:
            if mk in model_sd and model_sd[mk].shape == v.shape:
                compat[mk] = v
                break
    missing, unexpected = model.load_state_dict(compat, strict=False)
    model.sync_hist_fc_from_backbone_head()
    init_classes = ckpt.get("classes")
    lg.info(f"[init-from] {path}")
    lg.info(
        f"  loaded {len(compat)} keys "
        f"(init_classes={len(init_classes) if init_classes else '?'}, current_classes={num_classes})"
    )
    lg.info(f"  missing={len(missing)} unexpected={len(unexpected)}")
    adapter_missing = [k for k in missing if "fusion_adapter" in k or "obj_embedding" in k or "hist_fc" in k]
    if adapter_missing:
        lg.info(f"  new ablation params initialized: {len(adapter_missing)}")


def build_argparser() -> argparse.ArgumentParser:
    candidates = sorted({cfg.name for cfg in _candidate_table().values()})
    p = argparse.ArgumentParser()
    p.add_argument("--candidate", default="E8-S3", help="one of: " + ", ".join(candidates))
    p.add_argument("--data-dir", default=DEFAULT_DATA_DIR)
    p.add_argument("--obj-id-dir", default=DEFAULT_OBJ_ID_DIR)
    p.add_argument("--init-from", default=None)
    p.add_argument("--log-root", default=DEFAULT_LOG_ROOT)
    p.add_argument("--model-tag", default="objid_midfusion")
    p.add_argument("--epochs", type=int, default=COMPOUND_CFG["epochs"])
    p.add_argument("--batch", type=int, default=COMPOUND_CFG["batch"])
    p.add_argument("--img-size", type=int, default=COMPOUND_CFG["img_size"])
    p.add_argument("--workers", type=int, default=COMPOUND_CFG["num_workers"])
    p.add_argument("--seed", type=int, default=COMPOUND_CFG["seed"])
    p.add_argument("--split-ratios", nargs=3, type=float, default=None, metavar=("TRAIN", "VAL", "TEST"))
    p.add_argument("--train-val-only", action="store_true")
    p.add_argument("--subset-config", default=None, help="YAML: classes:{name:N, default:N}")
    p.add_argument("--subset-n-per-class", type=int, default=None, help="shortcut for classes.default=N")
    p.add_argument("--lr-backbone", type=float, default=COMPOUND_CFG["lr_backbone"])
    p.add_argument("--lr-head", type=float, default=COMPOUND_CFG["lr_head"])
    p.add_argument("--weight-decay", type=float, default=COMPOUND_CFG["weight_decay"])
    p.add_argument("--warmup-epochs", type=int, default=COMPOUND_CFG["warmup_epochs"])
    p.add_argument("--patience", type=int, default=COMPOUND_CFG["early_stop_patience"])
    p.add_argument("--loss", choices=["ce", "focal"], default=COMPOUND_CFG["loss"])
    p.add_argument("--focal-gamma", type=float, default=COMPOUND_CFG["focal_gamma"])
    p.add_argument("--class-weight", choices=["none", "inverse", "effective"], default=COMPOUND_CFG["class_weight"])
    p.add_argument("--effective-beta", type=float, default=COMPOUND_CFG["effective_beta"])
    p.add_argument("--label-smoothing", type=float, default=COMPOUND_CFG["label_smoothing"])
    p.add_argument("--ema", action="store_true", default=COMPOUND_CFG["ema"])
    p.add_argument("--no-ema", action="store_false", dest="ema")
    p.add_argument("--ema-decay", type=float, default=COMPOUND_CFG["ema_decay"])
    p.add_argument("--ema-warmup", type=int, default=COMPOUND_CFG["ema_warmup"])
    p.add_argument("--stochastic-depth", type=float, default=COMPOUND_CFG["stochastic_depth"])
    p.add_argument("--grad-clip", type=float, default=COMPOUND_CFG["grad_clip"])
    p.add_argument("--weighted-sampler", action="store_true", default=COMPOUND_CFG["weighted_sampler"])
    p.add_argument("--val-loss-guard", type=float, default=COMPOUND_CFG["val_loss_guard"])
    p.add_argument("--val-smooth-window", type=int, default=COMPOUND_CFG["val_smooth_window"])
    p.add_argument("--save-pred-samples", action="store_true", default=COMPOUND_CFG["save_pred_samples"])
    p.add_argument("--update-overall", action="store_true", help="update <log-root>/overall when improved; default off")
    p.add_argument("--allow-compound-log-root", action="store_true", help="allow writing under logs_compound; default refuses")
    p.add_argument("--dry-run-batch", action="store_true", help="build data/model, run one forward batch, then exit")
    p.add_argument("--require-gpu", action="store_true")
    p.add_argument("--ram-limit", type=float, default=80.0)
    p.add_argument("--gpu-mem-limit", type=float, default=90.0)
    p.add_argument("--monitor-interval", type=float, default=30.0)
    return p


def main():
    p = build_argparser()
    args = p.parse_args()
    if args.train_val_only and args.split_ratios is not None:
        p.error("--train-val-only and --split-ratios cannot be used together")
    if args.subset_config and args.subset_n_per_class is not None:
        p.error("--subset-config and --subset-n-per-class cannot be used together")
    log_root_resolved = Path(args.log_root).resolve()
    compound_root = (Path.cwd() / "logs_compound").resolve()
    if not args.allow_compound_log_root and (
        log_root_resolved == compound_root or compound_root in log_root_resolved.parents
    ):
        p.error("cnn_train_midfusion_objid.py refuses logs_compound by default; use logs_objid_ablation")

    candidate = resolve_candidate(args.candidate)
    split_ratios = (0.8, 0.2, 0.0) if args.train_val_only else (
        tuple(args.split_ratios) if args.split_ratios is not None else COMPOUND_CFG["split_ratios"]
    )

    a = assess_start(ram_limit=args.ram_limit, gpu_mem_limit=args.gpu_mem_limit, require_gpu=args.require_gpu)
    print(format_assessment(a), flush=True)
    if not a["ok"]:
        print("[guard] start blocked. Free RAM/GPU resources and retry.", file=sys.stderr)
        sys.exit(2)

    if args.weighted_sampler and args.class_weight != "none":
        print("[!] --weighted-sampler given; forcing --class-weight=none", file=sys.stderr)
        args.class_weight = "none"

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.benchmark = True

    device = torch.device(a["device"])
    out_root = Path(args.log_root)
    out_root.mkdir(parents=True, exist_ok=True)
    tag = f"{args.model_tag}_{candidate.name}"
    out_dir = out_root / f"{tag}_{RUN_TS}_running"
    lg = setup_logger(out_dir)
    lg.info("===== ObjID mid-fusion ablation train start =====")
    lg.info(f"candidate={candidate}")
    lg.info(f"device={device} backbone={BACKBONE} img={args.img_size} batch={args.batch} epochs={args.epochs}")
    lg.info(f"log_root={args.log_root} update_overall={args.update_overall}")
    lg.info(f"data_dir={args.data_dir}, exclude={EXCLUDE_CLASSES}")

    n_chip_objects, obj_id_to_label = read_obj_meta(args.obj_id_dir)
    obj_label_to_id = {label: idx for idx, label in enumerate(obj_id_to_label)}
    if candidate.binary_label and candidate.binary_label not in obj_label_to_id:
        raise RuntimeError(f"binary object {candidate.binary_label!r} not found in {obj_id_to_label}")
    hist_dim = 3 * n_chip_objects + 3
    lg.info(f"[obj_id meta] n_chip_objects={n_chip_objects}, obj_id_to_label={obj_id_to_label}, hist_dim={hist_dim}")

    full_eval = ObjIdFusionImageFolder(
        args.data_dir,
        args.obj_id_dir,
        args.img_size,
        train_aug=False,
        n_chip_objects=n_chip_objects,
        obj_id_to_label=obj_id_to_label,
        input_mode=candidate.input_mode,
    )
    full_train = ObjIdFusionImageFolder(
        args.data_dir,
        args.obj_id_dir,
        args.img_size,
        train_aug=True,
        n_chip_objects=n_chip_objects,
        obj_id_to_label=obj_id_to_label,
        input_mode=candidate.input_mode,
    )

    classes = full_eval.classes
    num_classes = len(classes)
    lg.info(f"[ImageFolder class order] {classes}")
    lg.info(f"Classes ({num_classes}): {classes}")
    lg.info(f"Total samples (pre-subset): {len(full_eval)}")

    subset_dict = {"default": int(args.subset_n_per_class)} if args.subset_n_per_class is not None else load_subset_config(args.subset_config)
    if subset_dict:
        lg.info(f"Subset config: {subset_dict}")
        full_train.samples = apply_subset(full_train.samples, classes, subset_dict, args.seed)
        full_eval.samples = apply_subset(full_eval.samples, classes, subset_dict, args.seed)
        full_train.targets = [s[1] for s in full_train.samples]
        full_eval.targets = [s[1] for s in full_eval.samples]
        lg.info(f"Total samples (post-subset): {len(full_eval)}")
        cnt = np.bincount(full_eval.targets, minlength=num_classes)
        for ci, c in enumerate(classes):
            lg.info(f"  - {c}: {int(cnt[ci])}")

    targets = full_eval.targets
    tr_idx, va_idx, te_idx = stratified_split(targets, split_ratios, args.seed)
    has_test = len(te_idx) > 0
    lg.info(f"Split ratios: train={split_ratios[0]} val={split_ratios[1]} test={split_ratios[2]}")
    lg.info(f"Split sizes: train={len(tr_idx)} val={len(va_idx)} test={len(te_idx)}")

    train_set = Subset(full_train, tr_idx)
    val_set = Subset(full_eval, va_idx)

    train_sampler = None
    shuffle_train = True
    if args.weighted_sampler:
        tr_targets = [targets[i] for i in tr_idx]
        cnt = np.bincount(tr_targets, minlength=num_classes).astype(np.float64)
        cnt = np.maximum(cnt, 1.0)
        sample_w = np.array([1.0 / cnt[t] for t in tr_targets])
        train_sampler = WeightedRandomSampler(
            weights=torch.tensor(sample_w, dtype=torch.float64),
            num_samples=len(tr_targets),
            replacement=True,
        )
        shuffle_train = False
        lg.info("WeightedRandomSampler enabled.")

    train_ld = DataLoader(
        train_set,
        batch_size=args.batch,
        shuffle=shuffle_train,
        sampler=train_sampler,
        num_workers=args.workers,
        pin_memory=True,
        persistent_workers=args.workers > 0,
    )
    val_ld = DataLoader(
        val_set,
        batch_size=args.batch,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=True,
        persistent_workers=args.workers > 0,
    )
    test_ld = None
    if has_test:
        test_set_p = ObjIdPathSubset(full_eval, te_idx)
        test_ld = DataLoader(
            test_set_p,
            batch_size=args.batch,
            shuffle=False,
            num_workers=args.workers,
            pin_memory=True,
            persistent_workers=args.workers > 0,
        )

    cw = compute_class_weights([targets[i] for i in tr_idx], num_classes, args.class_weight, args.effective_beta) if args.class_weight != "none" else None
    if cw is not None:
        lg.info(f"class_weight ({args.class_weight}) min={float(cw.min()):.3f} max={float(cw.max()):.3f}")
    if args.loss == "focal":
        criterion = FocalLoss(gamma=args.focal_gamma, weight=cw, label_smoothing=args.label_smoothing)
    else:
        criterion = nn.CrossEntropyLoss(weight=cw, label_smoothing=args.label_smoothing)
    criterion = criterion.to(device)

    backbone = build_model(num_classes, drop_path_rate=args.stochastic_depth, in_chans=3)
    model = ObjIdMidFusionConvNeXt(backbone, candidate, n_chip_objects, obj_label_to_id, hist_dim)
    if args.init_from:
        load_init_from(model, args.init_from, num_classes, lg)
    model = model.to(device)
    if device.type == "cuda":
        model = model.to(memory_format=torch.channels_last)

    head_params, backbone_params = [], []
    for name, param in model.named_parameters():
        if (
            name.startswith("backbone.head.")
            or name.startswith("hist_fc.")
            or name.startswith("fusion_adapter.")
            or name.startswith("obj_embedding.")
        ):
            head_params.append(param)
        else:
            backbone_params.append(param)
    opt = torch.optim.AdamW(
        [
            {"params": backbone_params, "lr": args.lr_backbone},
            {"params": head_params, "lr": args.lr_head},
        ],
        weight_decay=args.weight_decay,
    )

    steps_per_epoch = max(1, len(train_ld))
    total_steps = args.epochs * steps_per_epoch
    warmup_steps = args.warmup_epochs * steps_per_epoch
    sched_warmup = torch.optim.lr_scheduler.LinearLR(opt, start_factor=0.1, end_factor=1.0, total_iters=max(1, warmup_steps))
    sched_cos = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(1, total_steps - warmup_steps))
    scheduler = torch.optim.lr_scheduler.SequentialLR(opt, schedulers=[sched_warmup, sched_cos], milestones=[warmup_steps])
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    ema = EMA(model, decay=args.ema_decay, warmup_steps=args.ema_warmup * steps_per_epoch) if args.ema else None
    if ema is not None:
        lg.info(f"EMA enabled (decay={args.ema_decay}, warmup_steps={args.ema_warmup * steps_per_epoch})")

    hparams_dict = {
        **COMPOUND_CFG,
        **vars(args),
        "candidate_resolved": candidate.__dict__,
        "backbone": BACKBONE,
        "num_classes": num_classes,
        "classes": classes,
        "n_chip_objects": n_chip_objects,
        "obj_id_to_label": obj_id_to_label,
        "hist_dim": hist_dim,
        "run_ts": RUN_TS,
        "device": str(device),
        "split_ratios_used": split_ratios,
        "torch_version": str(torch.__version__),
        "timm_version": str(getattr(timm, "__version__", "?")),
    }
    with open(out_dir / "hparams.yaml", "w", encoding="utf-8") as f:
        yaml.safe_dump({k: _yaml_safe(v) for k, v in hparams_dict.items()}, f, sort_keys=False, allow_unicode=True)
    with open(out_dir / "hparams.txt", "w", encoding="utf-8") as f:
        for k, v in hparams_dict.items():
            f.write(f"{k:<24}: {v}\n")

    if args.dry_run_batch:
        payload, lbls = next(iter(train_ld))
        payload = move_payload(payload, device, channels_last=device.type == "cuda")
        with torch.no_grad():
            logits = model(payload)
        lg.info(
            f"[dry-run] image={tuple(payload['image'].shape)} obj_id={tuple(payload['obj_id'].shape)} "
            f"hist={tuple(payload['hist'].shape)} logits={tuple(logits.shape)} labels={tuple(lbls.shape)}"
        )
        lg.info(f"[Done] dry-run outputs: {out_dir.resolve()}")
        return

    history = []
    best_score = -1.0
    best_ep = 0
    no_improve = 0
    best_val_loss = float("inf")
    val_f1_window: List[float] = []
    best_snapshots: List[dict] = []
    best_metric_summary: Optional[dict] = None

    monitor = ResourceMonitor(ram_limit=args.ram_limit, interval_sec=args.monitor_interval, logger=lg)
    monitor.start()
    aborted_reason: Optional[str] = None

    for ep in range(1, args.epochs + 1):
        if monitor.should_abort():
            aborted_reason = monitor.abort_reason
            lg.info(f"  [guard] aborting before epoch {ep}: {aborted_reason}")
            break
        tr_loss, tr_acc = train_one_epoch_objid(
            model,
            train_ld,
            opt,
            scaler,
            scheduler,
            device,
            lg,
            ep,
            args.epochs,
            criterion,
            ema,
            args.grad_clip,
        )
        if ema is not None:
            backup = ema.apply_to(model)
            va = evaluate_objid(model, val_ld, device, classes, criterion=criterion)
            ema.restore(model, backup)
        else:
            va = evaluate_objid(model, val_ld, device, classes, criterion=criterion)

        rec = {
            "epoch": ep,
            "train_loss": float(tr_loss),
            "train_acc": float(tr_acc),
            "val_loss": va.get("val_loss", float("nan")),
            "val_acc": va["acc"],
            "val_macro_f1": va["macro_f1"],
            "val_macro_p": va["macro_p"],
            "val_macro_r": va["macro_r"],
            "val_edge_object_f1": va.get("edge_object_f1"),
            "val_edge_object_swap_errors": va.get("edge_object_swap_errors"),
            "val_edge_object_support": va.get("edge_object_support"),
        }
        history.append(rec)
        lg.info(
            f"[Ep {ep}] tr_loss={tr_loss:.4f} | val_loss={rec['val_loss']:.4f} "
            f"acc={100 * va['acc']:.2f}% f1={100 * va['macro_f1']:.2f}% "
            f"edge_f1={100 * va.get('edge_object_f1', float('nan')):.2f}% "
            f"edge_swaps={va.get('edge_object_swap_errors', 0)}"
        )
        save_curves_png(history, out_dir / "curves.png")
        with open(out_dir / "history.json", "w", encoding="utf-8") as hf:
            json.dump({"history": history, "best_epoch": best_ep, "best_smoothed_val_f1": best_score}, hf, indent=2)

        val_f1_window.append(va["macro_f1"])
        if len(val_f1_window) > args.val_smooth_window:
            val_f1_window.pop(0)
        smooth_f1 = float(np.median(val_f1_window))
        guard_block = (
            rec["val_loss"] is not None
            and not np.isnan(rec["val_loss"])
            and best_val_loss < float("inf")
            and rec["val_loss"] > args.val_loss_guard * best_val_loss
        )

        if smooth_f1 > best_score and not guard_block:
            best_score = smooth_f1
            best_ep = ep
            no_improve = 0
            best_val_loss = min(best_val_loss, rec["val_loss"]) if not np.isnan(rec["val_loss"]) else best_val_loss

            best_val_ld = DataLoader(ObjIdPathSubset(full_eval, va_idx), batch_size=args.batch, shuffle=False, num_workers=0, pin_memory=True)
            if ema is not None:
                backup_t = ema.apply_to(model)
                val_res = evaluate_objid(model, best_val_ld, device, classes, return_paths=True)
                test_res = evaluate_objid(model, test_ld, device, classes, return_paths=True) if test_ld is not None else None
                ema.restore(model, backup_t)
            else:
                val_res = evaluate_objid(model, best_val_ld, device, classes, return_paths=True)
                test_res = evaluate_objid(model, test_ld, device, classes, return_paths=True) if test_ld is not None else None
            del best_val_ld

            if test_res is not None:
                lg.info(
                    f"  best | VAL f1={100 * val_res['macro_f1']:.2f}% edge_f1={100 * val_res.get('edge_object_f1', float('nan')):.2f}% "
                    f"swaps={val_res.get('edge_object_swap_errors', 0)} | "
                    f"TEST f1={100 * test_res['macro_f1']:.2f}% edge_f1={100 * test_res.get('edge_object_f1', float('nan')):.2f}% "
                    f"swaps={test_res.get('edge_object_swap_errors', 0)}"
                )
            else:
                lg.info(
                    f"  best | VAL f1={100 * val_res['macro_f1']:.2f}% edge_f1={100 * val_res.get('edge_object_f1', float('nan')):.2f}% "
                    f"swaps={val_res.get('edge_object_swap_errors', 0)}"
                )

            ckpt = {
                "epoch": ep,
                "model": model.state_dict(),
                "classes": classes,
                "img_size": args.img_size,
                "backbone": BACKBONE,
                "candidate": candidate.__dict__,
                "obj_id_to_label": obj_id_to_label,
                "n_chip_objects": int(n_chip_objects),
                "hist_dim": int(hist_dim),
                "val_macro_f1": float(val_res["macro_f1"]),
                "val_acc": float(val_res["acc"]),
                "val_macro_r": float(val_res["macro_r"]),
                "val_edge_object_f1": float(val_res.get("edge_object_f1", float("nan"))),
                "val_edge_object_swap_errors": int(val_res.get("edge_object_swap_errors", 0)),
                "smoothed_val_f1": float(smooth_f1),
            }
            if test_res is not None:
                ckpt.update(
                    {
                        "test_macro_f1": float(test_res["macro_f1"]),
                        "test_macro_r": float(test_res["macro_r"]),
                        "test_acc": float(test_res["acc"]),
                        "test_edge_object_f1": float(test_res.get("edge_object_f1", float("nan"))),
                        "test_edge_object_swap_errors": int(test_res.get("edge_object_swap_errors", 0)),
                    }
                )
            if ema is not None:
                ckpt["ema_state"] = ema.state_dict()
            torch.save(ckpt, out_dir / "best_model.pth")

            n_wrong_val = save_wrong_tree(val_res, out_dir / "wrong" / "val")
            if test_res is not None:
                save_confusion_matrix_combined(val_res, test_res, out_dir / "best_confusion_matrix.png")
                n_wrong = save_wrong_tree(test_res, out_dir / "wrong" / "test")
                lg.info(f"  wrong saved: test={n_wrong} val={n_wrong_val} -> wrong/{{test,val}}/<true>/<pred>/")
            else:
                save_confusion_matrix(val_res, out_dir / "best_confusion_matrix.png")
                lg.info(f"  wrong saved: val={n_wrong_val} -> wrong/val/<true>/<pred>/")

            if test_res is not None:
                best_test_res = test_res
            best_val_res = val_res
            best_snapshots.append(
                {
                    "epoch": ep,
                    "smooth_f1": float(smooth_f1),
                    "train_loss": float(tr_loss),
                    "val_res": val_res,
                    "test_res": test_res,
                }
            )
            write_best_history(best_snapshots, out_dir / "best_history.txt")
            best_metric_summary = {
                "candidate": candidate.name,
                "epoch": ep,
                "smoothed_val_f1": float(smooth_f1),
                "val": metric_brief(val_res),
                "test": metric_brief(test_res),
                "edge_object_classes": val_res.get("edge_object_classes", []),
            }
            with open(out_dir / "best_ablation_metrics.json", "w", encoding="utf-8") as f:
                json.dump(best_metric_summary, f, indent=2, ensure_ascii=False)
        else:
            no_improve += 1
            if guard_block:
                lg.info(f"  [guard] val_loss={rec['val_loss']:.4f} > {args.val_loss_guard}x{best_val_loss:.4f}; best NOT updated.")
            if no_improve >= args.patience:
                lg.info(f"  early stop at ep {ep} (no improve for {args.patience}).")
                break

    monitor.stop()

    with open(out_dir / "history.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "history": history,
                "best_epoch": best_ep,
                "best_smoothed_val_f1": best_score,
                "best_ablation_metrics": best_metric_summary,
                "aborted": aborted_reason,
            },
            f,
            indent=2,
        )
    save_curves_png(history, out_dir / "curves.png")

    if args.save_pred_samples and "best_test_res" in locals():
        save_pred_samples(best_test_res, out_dir / "predictions", max_per_bucket=20)
        lg.info("  pred samples saved -> predictions/")
    elif args.save_pred_samples and "best_val_res" in locals():
        save_pred_samples(best_val_res, out_dir / "predictions_val", max_per_bucket=20)
        lg.info("  pred samples saved -> predictions_val/")

    final_test_f1 = best_test_res["macro_f1"] if "best_test_res" in locals() else None
    final_val_f1 = best_val_res["macro_f1"] if "best_val_res" in locals() else 0.0
    final_dir = rename_run_dir(out_dir, tag, RUN_TS, final_test_f1, final_val_f1)
    if aborted_reason:
        aborted_dir = final_dir.with_name(final_dir.name + "_ABORTED")
        try:
            final_dir.rename(aborted_dir)
            final_dir = aborted_dir
        except Exception as e:
            lg.info(f"  [guard] rename to ABORTED failed: {e}")
        lg.info(f"[Aborted] reason: {aborted_reason}")
    lg.info(f"[Metric source] {'test' if 'best_test_res' in locals() else 'val'}")
    lg.info(f"[Done] outputs: {final_dir.resolve()}")

    if args.update_overall and not aborted_reason:
        try:
            update_overall_best(out_root, final_dir, float(final_val_f1), logger=lg)
        except Exception as e:
            lg.info(f"[overall] update failed: {e}")
    elif not args.update_overall:
        lg.info("[overall] skipped (use --update-overall to enable within log-root)")
    lg.info("===== END =====")


if __name__ == "__main__":
    main()
