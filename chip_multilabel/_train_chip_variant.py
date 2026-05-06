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
import time
from datetime import datetime
from pathlib import Path
from typing import List, Tuple

import numpy as np
import timm
import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from .constants import DEFAULT_BACKBONE_CKPT, TRAIN_CLASSES, TRAIN_VARIANTS
from .losses import BCEThenASL, build_loss

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
    def __init__(self, samples: List[Tuple[Path, int]], img_size: int, train: bool):
        self.samples = samples
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
        p, y = self.samples[i]
        img = Image.open(p).convert("RGB")
        return self.tf(img), y


NORMAL_SENTINEL = -1  # 260506 — Normal class y label (multi-hot target = all zeros)


def collect_samples(root: Path, include_normal: bool = True) -> List[Tuple[Path, int]]:
    """Collect train samples. y >= 0 = TRAIN_CLASSES index. y = NORMAL_SENTINEL = Normal (no defect).

    Normal/ folder is OPTIONAL — if present and include_normal=True, all chips load with y=-1
    so multi-hot target becomes [0,0,0,0] (BCE pulls all sigmoids down → "Normal" representation).
    """
    out: List[Tuple[Path, int]] = []
    for ci, cname in enumerate(TRAIN_CLASSES):
        d = root / cname
        if not d.exists():
            raise FileNotFoundError(f"missing class dir: {d}")
        for png in sorted(d.glob("*.png")):
            out.append((png, ci))
    if include_normal:
        nd = root / "Normal"
        if nd.exists():
            n_normal = 0
            for png in sorted(nd.glob("*.png")):
                out.append((png, NORMAL_SENTINEL))
                n_normal += 1
            if n_normal > 0:
                print(f"[data] including {n_normal} Normal chips (y=-1 sentinel, multi-hot target [0,0,0,0])")
    return out


def stratified_split(samples: List[Tuple[Path, int]], val_ratio: float = 0.2, seed: int = 42):
    by = {}
    for p, y in samples:
        by.setdefault(y, []).append(p)
    rng = np.random.default_rng(seed)
    train, val = [], []
    for y, paths in by.items():
        idx = list(range(len(paths)))
        rng.shuffle(idx)
        n_val = max(1, int(round(len(paths) * val_ratio)))
        for i in idx[:n_val]:
            val.append((paths[i], y))
        for i in idx[n_val:]:
            train.append((paths[i], y))
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
        for ema_v, v in zip(self.module.state_dict().values(), model.state_dict().values()):
            if ema_v.dtype.is_floating_point:
                ema_v.mul_(decay_t).add_(v.detach().to(ema_v.dtype), alpha=1.0 - decay_t)
            else:
                ema_v.copy_(v)


def build_model(num_classes: int, init_ckpt: Path,
                drop_path_rate: float = 0.0) -> Tuple[torch.nn.Module, str, int]:
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
    """val_acc on defect samples only (y >= 0). Normal sentinel (y=-1) skipped because
    4-way argmax can't represent 'no defect' — Normal evaluated separately via max-prob check.
    """
    model.eval()
    correct = 0
    total = 0
    correct_normal = 0
    total_normal = 0
    use_amp = device.type == "cuda"
    NORMAL_MAX_PROB = 0.5  # Normal correct if max sigmoid < 0.5
    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        with torch.amp.autocast(device_type=device.type, enabled=use_amp):
            logits = model(x)
        # defect: argmax accuracy
        defect_mask = (y >= 0)
        if defect_mask.any():
            pred = logits[defect_mask].argmax(dim=1)
            correct += int((pred == y[defect_mask]).sum())
            total += int(defect_mask.sum())
        # Normal: max sigmoid < threshold = correct
        normal_mask = (y < 0)
        if normal_mask.any():
            probs = torch.sigmoid(logits[normal_mask])
            max_p = probs.max(dim=1).values
            correct_normal += int((max_p < NORMAL_MAX_PROB).sum())
            total_normal += int(normal_mask.sum())
    defect_acc = correct / max(total, 1)
    normal_acc = correct_normal / max(total_normal, 1) if total_normal > 0 else None
    if normal_acc is not None:
        # Combined: weighted avg of defect_acc and normal_acc
        return (correct + correct_normal) / max(total + total_normal, 1)
    return defect_acc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", required=True, choices=list(VARIANT_TO_LOSS.keys()))
    ap.add_argument("--ls", type=float, default=None,
                    help="label smoothing (T1 only). default 0.1 if not set.")
    ap.add_argument("--tag", type=str, default="",
                    help="optional tag suffix for out_dir name")
    ap.add_argument("--data-root", default="D:/project/data/wm-811k/classification_chips")
    ap.add_argument("--init-ckpt", default=DEFAULT_BACKBONE_CKPT)
    ap.add_argument("--out-root", default="outputs/logs_chip_multilabel")
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--accum", type=int, default=4)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--num-workers", type=int, default=0)
    ap.add_argument("--device", default=None)
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
    ap.add_argument("--cutmix-pair-bias", type=str, default="",
                    help="Bias CutMix pair sampling. Format 'C1,C2:K' — fork↔scratch pair "
                         "K× more likely than uniform. e.g., 'fork,scratch:2' → ~67%% of CutMix "
                         "events become fork↔scratch (vs uniform ~17%%).")
    ap.add_argument("--cutmix-p", type=float, default=0.0,
                    help="probability of applying CutMix on a batch (T7 only). 0 = off.")
    ap.add_argument("--cutmix-rect", type=float, default=0.5,
                    help="rectangle area fraction for CutMix paste (default 0.5 = half-image).")
    # iter 12 (260506) — scattered CutMix + soft proportional label
    ap.add_argument("--cutmix-mode", type=str, default="single",
                    choices=["single", "scattered"],
                    help="CutMix mode. 'single' (default) = original random rectangle "
                         "(OR multi-hot label or soft λ for CE-soft). "
                         "'scattered' = N small patches + soft proportional label "
                         "label_B = ratio × discount × alpha (iter 12, paper-quality).")
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
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))

    ts = datetime.now().strftime("%y%m%d_%H%M%S")
    name = args.variant if not args.tag else f"{args.variant}_{args.tag}"
    out_dir = Path(args.out_root) / f"{name}_{ts}"
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[train {args.variant}] device={device}  out={out_dir}")
    samples = collect_samples(Path(args.data_root), include_normal=not args.no_normal)
    train_samples, val_samples = stratified_split(samples, val_ratio=0.2, seed=args.seed)
    print(f"[train] data: train={len(train_samples)} val={len(val_samples)} classes={TRAIN_CLASSES}")

    model, backbone, img_size = build_model(num_classes=len(TRAIN_CLASSES),
                                            init_ckpt=Path(args.init_ckpt),
                                            drop_path_rate=args.drop_path_rate)
    model = model.to(device)

    ema = ModelEMA(model, decay=args.ema_decay) if args.ema_decay > 0 else None
    if ema is not None:
        print(f"[train] EMA enabled, target decay = {args.ema_decay}")

    train_ds = ChipFolderDataset(train_samples, img_size, train=True)
    val_ds = ChipFolderDataset(val_samples, img_size, train=False)
    train_loader = DataLoader(train_ds, batch_size=args.batch, shuffle=True,
                              num_workers=args.num_workers, drop_last=False)
    val_loader = DataLoader(val_ds, batch_size=args.batch, shuffle=False,
                            num_workers=args.num_workers)

    loss_name = VARIANT_TO_LOSS[args.variant]
    build_kw = {}
    if args.ls is not None:
        build_kw["ls"] = float(args.ls)
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
    if args.cutmix_p > 0:
        if args.cutmix_mode == "scattered":
            soft_b = float(args.cutmix_total_ratio) * float(args.cutmix_discount) \
                     * float(args.cutmix_alpha)
            print(f"[train] CutMix enabled mode=scattered p={args.cutmix_p} "
                  f"n_patches={args.cutmix_n_patches} total_ratio={args.cutmix_total_ratio} "
                  f"discount={args.cutmix_discount} alpha={args.cutmix_alpha} "
                  f"-> soft label_B={soft_b:.4f}")
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

    history = []
    best_val_acc = 0.0
    best_epoch = -1
    t_total = time.time()
    for ep in range(1, args.epochs + 1):
        model.train()
        if isinstance(loss_fn, BCEThenASL):
            loss_fn.set_epoch(ep - 1)
        running = 0.0
        nb = 0
        optim.zero_grad()
        for step, (x, y) in enumerate(train_loader):
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            if target_kind in ("multi_hot", "soft_multihot"):
                tgt = torch.zeros(y.size(0), len(TRAIN_CLASSES), device=device)
                # Normal sentinel y=-1 → tgt stays all zeros (multi-hot Normal). Defect y>=0 → one-hot.
                defect_mask = (y >= 0)
                if defect_mask.any():
                    di = defect_mask.nonzero(as_tuple=True)[0]
                    tgt[di, y[di]] = 1.0
                # CutMix: with prob cutmix_p, paste rectangle from another sample
                # with different class. Target rules differ by kind:
                #   multi_hot: BCE — OR of two one-hots (sum=2 for mixed; Normal contributes zero row)
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
                        H, W = x.size(-2), x.size(-1)
                        if args.cutmix_mode == "scattered":
                            # iter 12 (260506) — scattered patches + soft proportional label.
                            # label_B = ratio × discount × alpha (continuous, BCE-friendly).
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
                                a_class = int(y[bi].item())
                                b_class = int(y[perm[bi]].item())
                                if target_kind == "soft_multihot":
                                    # CE soft: cap at 0.5 to keep A dominant
                                    lam_b = min(soft_label_b, 0.5)
                                    tgt[bi, a_class] = 1.0 - lam_b
                                    tgt[bi, b_class] = lam_b
                                else:
                                    # BCE multi-hot: soft proportional (NOT OR)
                                    # tgt[bi, a_class] stays 1.0; B gets continuous soft label
                                    tgt[bi, b_class] = soft_label_b
                        else:
                            # single mode — original CutMix (OR for BCE, λ for CE-soft)
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
                                if target_kind == "soft_multihot":
                                    a = int(y[bi].item())
                                    b = int(y[perm[bi]].item())
                                    tgt[bi, a] = lam
                                    tgt[bi, b] = 1.0 - lam
                                else:
                                    tgt[bi, int(y[perm[bi]].item())] = 1.0
                tgt_used = tgt
            else:
                tgt_used = y
            with torch.amp.autocast(device_type=device.type, enabled=device.type == "cuda"):
                logits = model(x)
                loss = loss_fn(logits, tgt_used) / args.accum
            if scaler is not None:
                scaler.scale(loss).backward()
            else:
                loss.backward()
            running += float(loss.item()) * args.accum
            nb += 1
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
        val_acc = evaluate(eval_model, val_loader, device, num_classes=len(TRAIN_CLASSES))
        active = getattr(loss_fn, "last_active", loss_name)
        avg_loss = running / max(nb, 1)
        history.append({"epoch": ep, "train_loss": avg_loss, "val_acc": val_acc,
                        "lr": optim.param_groups[0]["lr"], "loss_active": active})
        print(f"[ep {ep:02d}] loss={avg_loss:.4f} val_acc={val_acc:.4f} loss_active={active}")
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch = ep
            save_state = (ema.module.state_dict() if ema is not None else model.state_dict())
            torch.save({
                "model": save_state,
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

    elapsed = time.time() - t_total
    print(f"[train] DONE  best_val_acc={best_val_acc:.4f} @ ep{best_epoch}  elapsed={elapsed:.1f}s")

    # 260506 — also save final-epoch model. val_acc (4-way single-class) often saturates
    # at ep1 for multi-hot training (e.g., sc+sr CutMix), causing best_model.pth to be
    # under-trained. final_epoch_model.pth captures end-of-training weights for comparison.
    final_state = (ema.module.state_dict() if ema is not None else model.state_dict())
    torch.save({
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


if __name__ == "__main__":
    main()
