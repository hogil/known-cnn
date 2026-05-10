# -*- coding: utf-8 -*-
"""Pre-compute teacher probs for KD distillation (iter32, 260509).

For each chip in classification_chips/<class>/*.png, run forward pass through
each of N teacher checkpoints (4-class trained variants), average probabilities,
write parquet (chip_path -> 4 prob columns).

Used by `_train_chip_variant.py --kd-teacher-probs` flag (Hinton 2015 KD,
Yang 2023 multi-label per-class binary KL adaptation).

Run once before iter32 student training:
    python -m chip_multilabel._kd_make_teacher_probs
"""
from __future__ import annotations

import sys
try:
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
except Exception:
    pass

import argparse
import time
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torchvision import transforms

from .constants import TRAIN_CLASSES
from .model_io import load_chip_backbone

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

# 14 ensemble member runs (iter21/22/24/26 — current 17-bag minus 3 not-yet-promoted).
BAG_RUNS: List[str] = [
    "outputs/iter21E_19C_repeat",
    "outputs/iter22A_seed7",
    "outputs/iter22B_seed42",
    "outputs/iter22D_LS030",
    "outputs/iter24_LS030_seed7",
    "outputs/iter24_LS030_seed42",
    "outputs/iter21F_19E_repeat",
    "outputs/iter21H_19I_repeat",
    "outputs/iter22G_droppath005",
    "outputs/iter26B_g3_LS050",
    "outputs/iter26D_g4_LS040",
    "outputs/iter26F_g2_LS100_white",
    "outputs/iter26G_g2_LS100_noise",
    "outputs/iter26H_g3_LS067_white",
]

TRAIN_CHIPS_DIR = "D:/project/data/wm-811k/classification_chips"
DEFAULT_OUT = "outputs/_teacher_probs_14bag.parquet"

# Folders to include in teacher-prob computation. Per spec: "classification_chips/<class>/*.png"
# = all chip dirs (4 single + invalid_main; if Normal exists also include).
INCLUDE_DIRS = ("bank_boundary", "fork", "scratch", "scratch_rot",
                "invalid_main", "Normal")


def _resolve_ckpt(run_dir: str) -> Path:
    """Find best_model.pth under outputs/<iter>/T*/best_model.pth."""
    cands = sorted(Path(run_dir).glob("T*/best_model.pth"))
    if not cands:
        raise FileNotFoundError(f"no T*/best_model.pth under {run_dir}")
    return cands[0]


def _list_chips(root: Path) -> List[Path]:
    chips: List[Path] = []
    for sub in INCLUDE_DIRS:
        d = root / sub
        if not d.is_dir():
            continue
        chips.extend(sorted(d.glob("*.png")))
    return chips


@torch.no_grad()
def _forward_all(model: torch.nn.Module, chips: List[Path], img_size: int,
                 keep_indices: List[int], device: torch.device,
                 batch_size: int = 16) -> np.ndarray:
    """Forward all chips deterministically (no aug, no shuffle), return (N, 4) sigmoid probs."""
    tf = transforms.Compose([
        transforms.Resize((img_size, img_size),
                          interpolation=transforms.InterpolationMode.BILINEAR),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])
    model.eval()
    use_amp = device.type == "cuda"
    out = np.zeros((len(chips), len(TRAIN_CLASSES)), dtype=np.float32)
    for i0 in range(0, len(chips), batch_size):
        batch_paths = chips[i0:i0 + batch_size]
        imgs = []
        for p in batch_paths:
            with Image.open(p) as im:
                imgs.append(tf(im.convert("RGB")))
        x = torch.stack(imgs).to(device, non_blocking=True)
        with torch.amp.autocast(device_type=device.type, enabled=use_amp):
            logits = model(x)            # (B, num_classes_full)
        logits_4 = logits[:, keep_indices]   # (B, 4)
        probs = torch.sigmoid(logits_4).float().cpu().numpy()
        out[i0:i0 + len(batch_paths)] = probs
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", default=TRAIN_CHIPS_DIR)
    ap.add_argument("--out", default=DEFAULT_OUT)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--device", default=None)
    ap.add_argument("--bag-runs", default=None,
                    help="optional comma-separated override of BAG_RUNS list")
    args = ap.parse_args()

    device = torch.device(args.device or
                          ("cuda" if torch.cuda.is_available() else "cpu"))
    bag_runs = args.bag_runs.split(",") if args.bag_runs else BAG_RUNS

    chips = _list_chips(Path(args.data_root))
    if not chips:
        raise SystemExit(f"no chips under {args.data_root}")
    print(f"[kd] {len(chips)} chips found from {args.data_root}")
    print(f"[kd] {len(bag_runs)} teacher members")

    all_probs: List[np.ndarray] = []
    t_total = time.time()
    for ri, run_dir in enumerate(bag_runs):
        ckpt_path = _resolve_ckpt(run_dir.strip())
        t0 = time.time()
        model, meta, keep_indices = load_chip_backbone(ckpt_path, device)
        probs = _forward_all(model, chips, meta["img_size"],
                             keep_indices, device, batch_size=args.batch_size)
        all_probs.append(probs)
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()
        print(f"[kd] {ri + 1}/{len(bag_runs)}  {run_dir}  "
              f"img_size={meta['img_size']}  {time.time() - t0:.1f}s")

    avg = np.mean(np.stack(all_probs, axis=0), axis=0)   # (N, 4)
    print(f"[kd] avg probs shape={avg.shape}  "
          f"min={avg.min():.4f}  max={avg.max():.4f}  mean={avg.mean():.4f}")

    df = pd.DataFrame({"chip_path": [str(p) for p in chips]})
    for ci, cname in enumerate(TRAIN_CLASSES):
        df[f"teacher_prob_{cname}"] = avg[:, ci]

    out_p = Path(args.out)
    out_p.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out_p, index=False)
    print(f"[kd] DONE  wrote {len(df)} rows -> {out_p}  "
          f"total elapsed={time.time() - t_total:.1f}s")


if __name__ == "__main__":
    main()
