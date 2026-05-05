"""Multi-label wafer 학습 entry (Stage 4 Phase B/C).

cnn_train.py / cnn_train_compound.py 의 single-label 학습 (CE + argmax) 와 별도 —
multi-hot target + sigmoid + (BCE/ASL/AdaGC) loss 로 학습.

지원 dataset 모드:
- single-positive (default, --data-dir D:/project/data/wm-811k/unknown):
  ImageFolder 의 single-label 을 33-bit one-hot 으로 변환. SPML setting (Stage 4 Phase B AdaGC 적합).
- multi-hot (--data-dir D:/project/data/wm-811k/unknown_multi):
  positions JSON 의 multi_labels 에서 다중 distribution × object 를 33-bit multi-hot 으로 expand.

지원 loss (--loss):
- bce / asl / m2 ... m7 (multi_label_losses.build_loss factory 참조)
- adagc — Stage 4 Phase B (teacher EMA 자동 setup)
- adagc_asl (M4) — AdaGC + ASL hybrid
- bce_warmup_asl (M5) — BCE warmup → ASL
- focal_asl (M6) — Focal positive + ASL negative
- adagc_ls (M7) — AdaGC + label_smoothing

출력:
- logs_multilabel/<tag>_<TS>_<f1>/best_model.pth
- best_history.txt, history.json, hparams.yaml, run.log
- logs_multilabel/overall/ best mirror

사용 예:
   # M2 (ASL single SOTA)
   python cnn_train_multilabel.py --loss asl --epochs 30 --batch 8 --workers 0 \
      --data-dir D:/project/data/wm-811k/unknown_multi --model-tag M2_asl

   # M4 (AdaGC + ASL hybrid)
   python cnn_train_multilabel.py --loss adagc_asl --epochs 30 --batch 8 --workers 0 \
      --data-dir D:/project/data/wm-811k/unknown --model-tag M4_adagc_asl

   # M5 (BCE warmup → ASL)
   python cnn_train_multilabel.py --loss bce_warmup_asl --epochs 30 --batch 8 \
      --warmup-epochs 5 --data-dir D:/project/data/wm-811k/unknown_multi --model-tag M5
"""
from __future__ import annotations
import argparse, glob, json, os, random, shutil, sys, time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import transforms
from torchvision.datasets import ImageFolder
from PIL import Image
import timm
from sklearn.metrics import f1_score, average_precision_score

from multi_label_losses import (
    build_loss, AdaGCLoss, MixLoss, ChainedLoss,
    init_teacher, update_teacher_ema,
)

if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8")

DISTRIBUTIONS = ["Center", "Donut", "Edge-Ring", "Edge-Bottom", "Edge-Top", "Full", "Thick-Edge"]
OBJECTS = ["bank_boundary", "fork", "scratch", "scratch_rot", "invalid_main"]            # round 26: particle_blast→fork, scratch_21deg→scratch_rot
DEFAULT_BACKBONE = "convnextv2_base.fcmae_ft_in22k_in1k_384"
DEFAULT_LOG_ROOT = "logs_multilabel"


# ---------------------------- Dataset ----------------------------

class MultiHotImageFolder(Dataset):
    """ImageFolder 기반 — single-positive 또는 multi-hot.

    mode='single_positive': folder name = single class. target = one-hot (B, C).
    mode='multihot_json': sibling positions JSON 의 multi_labels 에서 multi-hot expand.
    """
    def __init__(self, image_root: str, classes: List[str], transform=None,
                 mode: str = "single_positive", positions_root: Optional[str] = None):
        self.image_root = Path(image_root)
        self.classes = classes
        self.cls_to_idx = {c: i for i, c in enumerate(classes)}
        self.transform = transform
        self.mode = mode
        self.positions_root = Path(positions_root) if positions_root else None
        self.samples = self._collect()

    def _collect(self) -> List[Tuple[str, np.ndarray]]:
        samples = []
        if self.mode == "single_positive":
            for cls in sorted(os.listdir(self.image_root)):
                cls_dir = self.image_root / cls
                if not cls_dir.is_dir(): continue
                if cls not in self.cls_to_idx: continue
                idx = self.cls_to_idx[cls]
                for fn in sorted(os.listdir(cls_dir)):
                    if not fn.lower().endswith((".png", ".jpg", ".jpeg")): continue
                    target = np.zeros(len(self.classes), dtype=np.float32)
                    target[idx] = 1.0
                    samples.append((str(cls_dir / fn), target))
        elif self.mode == "multihot_json":
            # flat directory + JSON sibling
            for fn in sorted(os.listdir(self.image_root)):
                if not fn.lower().endswith(".png"): continue
                base = Path(fn).stem
                json_path = (self.positions_root / f"{base}.json"
                              if self.positions_root else self.image_root / f"{base}.json")
                if not json_path.exists():
                    continue
                try:
                    j = json.loads(json_path.read_text(encoding="utf-8"))
                except Exception:
                    continue
                ml = j.get("multi_labels")
                if not ml: continue
                target = np.zeros(len(self.classes), dtype=np.float32)
                for d in ml.get("distributions", []):
                    for o in ml.get("objects", []):
                        cand = f"{d}_{o}"
                        if cand in self.cls_to_idx:
                            target[self.cls_to_idx[cand]] = 1.0
                if target.sum() == 0:
                    continue
                samples.append((str(self.image_root / fn), target))
        else:
            raise ValueError(f"unknown mode: {self.mode}")
        return samples

    def __len__(self): return len(self.samples)

    def __getitem__(self, i):
        path, target = self.samples[i]
        with Image.open(path) as im:
            x = self.transform(im.convert("RGB")) if self.transform else im.convert("RGB")
        return x, torch.from_numpy(target)


def build_transforms(img_size: int):
    norm = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    train_tfm = transforms.Compose([
        transforms.Resize((img_size, img_size), interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.RandomAffine(degrees=15, translate=(0.03, 0.03), scale=(0.97, 1.03)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    eval_tfm = transforms.Compose([
        transforms.Resize((img_size, img_size), interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.ToTensor(), norm,
    ])
    return train_tfm, eval_tfm


# ---------------------------- Train / Eval ----------------------------

def train_one_epoch(model, teacher, loader, opt, criterion, device, ep,
                     teacher_alpha: float = 0.99, needs_teacher: bool = False):
    model.train()
    losses = []
    for it, (x, y) in enumerate(loader, 1):
        x = x.to(device, non_blocking=True); y = y.to(device, non_blocking=True)
        opt.zero_grad(set_to_none=True)
        logits = model(x)
        if needs_teacher:
            with torch.no_grad():
                teacher_logits = teacher(x)
            if isinstance(criterion, MixLoss):
                loss = criterion(logits, y, teacher_logits=teacher_logits)
            else:
                loss = criterion(logits, teacher_logits, y)
        else:
            loss = criterion(logits, y)
        loss.backward()
        opt.step()
        losses.append(float(loss))
        # update teacher EMA after each step
        if teacher is not None:
            update_teacher_ema(model, teacher, teacher_alpha)
    return float(np.mean(losses)) if losses else 0.0


@torch.no_grad()
def evaluate(model, loader, device, classes, threshold: float = 0.5):
    """Multi-label evaluation — sigmoid + threshold."""
    model.eval()
    all_logits, all_y = [], []
    for x, y in loader:
        x = x.to(device, non_blocking=True)
        logits = model(x)
        all_logits.append(logits.cpu().numpy())
        all_y.append(y.numpy())
    L = np.concatenate(all_logits, axis=0)
    Y = np.concatenate(all_y, axis=0).astype(int)
    P = 1.0 / (1.0 + np.exp(-L))
    pred = (P > threshold).astype(int)
    macro_f1 = f1_score(Y, pred, average="macro", zero_division=0)
    micro_f1 = f1_score(Y, pred, average="micro", zero_division=0)
    try:
        mAP = average_precision_score(Y, P, average="macro")
    except Exception:
        mAP = 0.0
    return {"macro_f1": float(macro_f1), "micro_f1": float(micro_f1),
            "mAP": float(mAP), "n": len(Y)}


# ---------------------------- Main ----------------------------

def setup_logger(out_dir: Path):
    import logging
    out_dir.mkdir(parents=True, exist_ok=True)
    lg = logging.getLogger(f"ml_{out_dir.name}")
    lg.handlers.clear(); lg.setLevel(logging.INFO)
    fh = logging.FileHandler(out_dir / "run.log", encoding="utf-8")
    fh.setFormatter(logging.Formatter("%(asctime)s %(message)s"))
    lg.addHandler(fh)
    sh = logging.StreamHandler(sys.stderr)
    sh.setFormatter(logging.Formatter("%(message)s"))
    lg.addHandler(sh)
    return lg


def update_overall_best(log_root: Path, out_dir: Path, val_f1: float):
    """Mirror best run to log_root/overall/."""
    overall = log_root / "overall"
    meta_path = overall / "_overall_meta.json"
    prev_f1 = -1.0
    if meta_path.exists():
        try:
            prev = json.loads(meta_path.read_text())
            prev_f1 = float(prev.get("val_f1", -1.0))
        except Exception:
            pass
    if val_f1 > prev_f1:
        if overall.exists():
            shutil.rmtree(overall)
        shutil.copytree(out_dir, overall)
        meta = {"val_f1": float(val_f1), "source_run": str(out_dir.name),
                 "updated_at": datetime.now().isoformat()}
        meta_path.write_text(json.dumps(meta, indent=2))
        print(f"[*] overall updated: {prev_f1:.4f} → {val_f1:.4f}", file=sys.stderr)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", default="D:/project/data/wm-811k/unknown_multi")
    p.add_argument("--positions-root", default="D:/project/data/positions/unknown_multi")
    p.add_argument("--mode", choices=["single_positive", "multihot_json", "auto"],
                   default="auto",
                   help="auto: detect by data-dir name (unknown_multi → multihot, unknown → single_positive)")
    p.add_argument("--classes-from", default=None,
                   help="checkpoint path to inherit class list from (default: derive from unknown/ folder names)")
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch", type=int, default=8)
    p.add_argument("--workers", type=int, default=0)
    p.add_argument("--img-size", type=int, default=384)
    p.add_argument("--lr-backbone", type=float, default=1e-5)
    p.add_argument("--lr-head", type=float, default=1e-4)
    p.add_argument("--loss", default="asl",
                   help="bce / asl / m2..m7 / adagc / adagc_asl (M4) / bce_warmup_asl (M5) / focal_asl (M6) / adagc_ls (M7)")
    p.add_argument("--gamma-pos", type=float, default=1.0)
    p.add_argument("--gamma-neg", type=float, default=4.0)
    p.add_argument("--clip", type=float, default=0.05)
    p.add_argument("--label-smoothing", type=float, default=0.05)
    p.add_argument("--lambda-gc", type=float, default=0.5)
    p.add_argument("--alpha-pred", type=float, default=0.95)
    p.add_argument("--lambda-warmup-epochs", type=int, default=3)
    p.add_argument("--warmup-epochs", type=int, default=5,
                   help="(BCE warmup → ASL) phase boundary epoch")
    p.add_argument("--teacher-alpha", type=float, default=0.99)
    p.add_argument("--threshold", type=float, default=0.5)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--model-tag", default=None)
    p.add_argument("--log-root", default=DEFAULT_LOG_ROOT)
    p.add_argument("--init-from", default=None,
                   help="best_model.pth from cnn_train.py (TAPT init)")
    p.add_argument("--val-ratio", type=float, default=0.2)
    p.add_argument("--device", default=None)
    args = p.parse_args()

    # seed
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    # mode
    if args.mode == "auto":
        args.mode = "multihot_json" if "unknown_multi" in args.data_dir.replace("\\", "/") else "single_positive"
    print(f"[*] mode: {args.mode}", file=sys.stderr)

    # resolve classes
    if args.classes_from:
        ckpt = torch.load(args.classes_from, map_location="cpu", weights_only=False)
        classes = ckpt.get("classes")
        if not classes:
            raise SystemExit(f"--classes-from has no 'classes': {args.classes_from}")
    else:
        # derive from unknown/ folder
        unknown_root = Path("D:/project/data/wm-811k/unknown")
        if unknown_root.exists():
            classes = sorted([d for d in os.listdir(unknown_root)
                              if (unknown_root / d).is_dir() and d != "Normal"])
        else:
            raise SystemExit("cannot derive classes — pass --classes-from or place D:/project/data/wm-811k/unknown")
    print(f"[*] {len(classes)} classes: {classes[:5]}...", file=sys.stderr)

    device = torch.device(args.device) if args.device else torch.device(
        "cuda" if torch.cuda.is_available() else "cpu")
    print(f"[*] device: {device}", file=sys.stderr)

    # tag
    ts = datetime.now().strftime("%y%m%d_%H%M%S")
    tag = args.model_tag or args.loss
    out_dir = Path(args.log_root) / f"{tag}_{ts}_running"
    lg = setup_logger(out_dir)
    lg.info(f"===== multi-label train start =====")
    lg.info(f"loss={args.loss}  data={args.data_dir}  mode={args.mode}")
    lg.info(f"epochs={args.epochs} batch={args.batch} img={args.img_size}")

    # dataset
    train_tfm, eval_tfm = build_transforms(args.img_size)
    full_train = MultiHotImageFolder(args.data_dir, classes, transform=train_tfm,
                                      mode=args.mode, positions_root=args.positions_root)
    full_eval = MultiHotImageFolder(args.data_dir, classes, transform=eval_tfm,
                                     mode=args.mode, positions_root=args.positions_root)
    if len(full_train) == 0:
        raise SystemExit(f"no samples in {args.data_dir}")
    lg.info(f"total samples: {len(full_train)}")

    n_val = int(len(full_train) * args.val_ratio)
    rng_perm = np.random.default_rng(args.seed).permutation(len(full_train))
    val_idx = rng_perm[:n_val].tolist()
    tr_idx = rng_perm[n_val:].tolist()
    tr_set = Subset(full_train, tr_idx); va_set = Subset(full_eval, val_idx)
    lg.info(f"train={len(tr_set)}  val={len(va_set)}")

    train_ld = DataLoader(tr_set, batch_size=args.batch, shuffle=True,
                          num_workers=args.workers, pin_memory=(device.type == "cuda"))
    val_ld = DataLoader(va_set, batch_size=args.batch, shuffle=False,
                        num_workers=args.workers, pin_memory=(device.type == "cuda"))

    # model
    model = timm.create_model(DEFAULT_BACKBONE, pretrained=False, num_classes=len(classes))
    if args.init_from:
        ckpt = torch.load(args.init_from, map_location=device, weights_only=False)
        state = ckpt.get("model") or ckpt.get("state_dict") or ckpt
        model.load_state_dict(state, strict=False)
        lg.info(f"init from {args.init_from}")
    model = model.to(device)

    # loss
    needs_teacher = args.loss.lower() in ("adagc", "adagc_asl", "m4", "adagc_ls", "m7")
    criterion = build_loss(args.loss, num_classes=len(classes),
                            gamma_pos=args.gamma_pos, gamma_neg=args.gamma_neg,
                            clip=args.clip, label_smoothing=args.label_smoothing,
                            lambda_gc=args.lambda_gc, alpha_pred=args.alpha_pred,
                            lambda_warmup_epochs=args.lambda_warmup_epochs,
                            warmup_epochs=args.warmup_epochs)
    if isinstance(criterion, AdaGCLoss):
        # plain AdaGC, signature (logits, teacher_logits, target)
        pass
    criterion = criterion.to(device) if hasattr(criterion, "to") else criterion
    lg.info(f"loss class: {type(criterion).__name__}, needs_teacher={needs_teacher}")

    # teacher (for AdaGC variants)
    teacher = init_teacher(model).to(device) if needs_teacher else None

    # optimizer (separate lr for head)
    head_params = list(model.get_classifier().parameters())
    head_param_ids = {id(p) for p in head_params}
    backbone_params = [p for p in model.parameters() if id(p) not in head_param_ids]
    opt = torch.optim.AdamW([
        {"params": backbone_params, "lr": args.lr_backbone},
        {"params": head_params, "lr": args.lr_head},
    ], weight_decay=0.05)

    # train loop
    best_f1 = -1.0
    history = []
    for ep in range(1, args.epochs + 1):
        t0 = time.time()
        train_loss = train_one_epoch(model, teacher, train_ld, opt, criterion, device, ep,
                                       teacher_alpha=args.teacher_alpha,
                                       needs_teacher=needs_teacher)
        val = evaluate(model, val_ld, device, classes, threshold=args.threshold)
        elapsed = time.time() - t0
        if hasattr(criterion, "step_epoch"):
            criterion.step_epoch()
        history.append({"epoch": ep, "train_loss": train_loss,
                        "val_macro_f1": val["macro_f1"], "val_micro_f1": val["micro_f1"],
                        "val_mAP": val["mAP"], "elapsed_sec": elapsed})
        lg.info(f"Ep {ep:3d}/{args.epochs} | loss={train_loss:.4f}  "
                f"val_macro_f1={val['macro_f1']:.4f}  micro_f1={val['micro_f1']:.4f}  "
                f"mAP={val['mAP']:.4f} | {elapsed:.1f}s")

        # save current history
        (out_dir / "history.json").write_text(json.dumps(history, indent=2))

        # save best
        if val["macro_f1"] > best_f1:
            best_f1 = val["macro_f1"]
            ckpt = {
                "model": model.state_dict(),
                "classes": classes,
                "img_size": args.img_size,
                "backbone": DEFAULT_BACKBONE,
                "loss": args.loss,
                "best_val_macro_f1": best_f1,
                "best_val_micro_f1": val["micro_f1"],
                "best_val_mAP": val["mAP"],
                "epoch": ep,
            }
            torch.save(ckpt, out_dir / "best_model.pth")

    # close logger handlers first (Windows file lock)
    for h in list(lg.handlers):
        try:
            h.close()
            lg.removeHandler(h)
        except Exception:
            pass

    # rename out_dir with final F1
    final_dir = out_dir.parent / out_dir.name.replace(
        "_running", f"_f{best_f1:.2f}".replace(".", ""))
    try:
        if final_dir.exists():
            shutil.rmtree(final_dir)
        out_dir.rename(final_dir)
    except Exception as e:
        print(f"[!] rename failed (Windows file lock?): {e}", file=sys.stderr)
        final_dir = out_dir                                                              # fallback to running name

    # best_history.txt
    (final_dir / "best_history.txt").write_text(
        f"BEST OVERALL\n  val_macro_f1: {best_f1:.4f}\n"
        f"  loss: {args.loss}\n"
        f"  epochs: {args.epochs}\n"
        f"  total samples (train/val): {len(tr_set)} / {len(va_set)}\n",
        encoding="utf-8")
    (final_dir / "hparams.yaml").write_text(json.dumps(vars(args), indent=2))

    # update overall best
    update_overall_best(Path(args.log_root), final_dir, best_f1)

    lg.info(f"\n[OK] training done. best_val_macro_f1={best_f1:.4f}")
    lg.info(f"     output: {final_dir}")
    print(f"\n[OK] best val_macro_f1 = {best_f1:.4f}")
    print(f"     output: {final_dir}")


if __name__ == "__main__":
    main()
