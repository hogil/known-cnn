"""obj_id only — wafer 33-class classifier.

Step 1 of Two-stream Pre-training plan. obj_id_map (32×32 uint8) 만 입력으로
받아 wafer pattern 분류. 작은 4-layer CNN + Embedding (categorical 의미 보존).

목적: "obj_id 정보 만으로 wafer 33-class 어느 정도 분류 가능한가?" 측정.
결과 → Step 2 (Mid-level Fusion) 의 obj backbone capacity 결정.

학습 데이터:
- label: D:/project/data/wm-811k/unknown/<class>/*.png (folder 명 = wafer class)
- input: obj_id .npy (D:/project/data/wm-811k/obj_id_maps 안 모든 .npy, flat basename lookup)

산출:
- logs_objid_ablation/<tag>_<TS>_f<f1>/best_model.pth
- history.json, run.log, hparams.yaml
- logs_objid_ablation/overall/ best mirror

사용:
    # Smoke (n=20/class subset, 3 epoch)
    python cnn_train_objonly.py --subset-config experiments/phase_smoke_n20.yaml \\
        --epochs 3 --batch 32 --workers 0 --model-tag objonly_smoke --train-val-only

    # Full
    python cnn_train_objonly.py --epochs 30 --batch 32 --workers 0 \\
        --model-tag objonly_full --train-val-only
"""
from __future__ import annotations
import argparse, json, os, random, shutil, sys, time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, Subset
from PIL import Image
import yaml
from sklearn.metrics import f1_score, accuracy_score, precision_recall_fscore_support

if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8")

DEFAULT_DATA_DIR = "D:/project/data/wm-811k/unknown"
DEFAULT_OBJ_ID_DIR = "D:/project/data/wm-811k/obj_id_maps"
DEFAULT_LOG_ROOT = "logs_objid_ablation"
EXCLUDE_CLASSES = {"Normal", "classification", "classification_chips"}                    # match cnn_train.py 정책 + chip data 폴더 제외


# ---------------------------- Dataset ----------------------------

class ObjOnlyDataset(Dataset):
    """ImageFolder 패턴 + flat basename → obj_id .npy lookup.

    PNG path 는 label 추출용으로만 쓰고, 실제 입력은 obj_id .npy 의 (32, 32) uint8.
    """
    def __init__(self, png_root: str, obj_id_root: str, classes: List[str],
                 train_aug: bool, n_chip_objects: int):
        self.png_root = Path(png_root)
        self.obj_id_root = Path(obj_id_root)
        self.classes = list(classes)
        self.cls_to_idx = {c: i for i, c in enumerate(self.classes)}
        self.train_aug = bool(train_aug)
        self.n_chip_objects = int(n_chip_objects)
        # flat basename → npy_path lookup
        self._npy_map: Dict[str, Path] = {}
        if self.obj_id_root.exists():
            for p in self.obj_id_root.rglob("*.npy"):
                self._npy_map[p.stem] = p
        print(f"[ObjOnlyDataset] {len(self._npy_map)} npy indexed under {self.obj_id_root}", flush=True)
        # collect samples (png_path, label) — only those with matching .npy
        self.samples: List[Tuple[str, int]] = []
        n_no_npy = 0
        for cls in self.classes:
            cls_dir = self.png_root / cls
            if not cls_dir.is_dir(): continue
            idx = self.cls_to_idx[cls]
            for fn in sorted(os.listdir(cls_dir)):
                if not fn.lower().endswith(".png"): continue
                base = fn[:-4]
                if base in self._npy_map:
                    self.samples.append((str(cls_dir / fn), idx))
                else:
                    n_no_npy += 1
        if n_no_npy:
            print(f"[ObjOnlyDataset] {n_no_npy} PNG without matching .npy (skipped)", flush=True)
        # ImageFolder-compat targets list
        self.targets = [t for _, t in self.samples]

    def __len__(self): return len(self.samples)

    def _augment(self, x: torch.Tensor) -> torch.Tensor:
        """±15° rotation only (도메인 정책: HFlip/VFlip 금지). Categorical preserve.

        Rotation 시 NEAREST interpolation (categorical 보존).
        """
        from torchvision.transforms import functional as TF, InterpolationMode
        angle = float(torch.empty(1).uniform_(-15.0, 15.0).item())
        x = TF.rotate(x, angle, interpolation=InterpolationMode.NEAREST, fill=0)
        return x

    def __getitem__(self, i):
        png_path, label = self.samples[i]
        base = Path(png_path).stem
        obj_id = np.load(self._npy_map[base]).astype(np.int64)                            # (32, 32) int64
        x = torch.from_numpy(obj_id).unsqueeze(0)                                         # (1, 32, 32) long
        if self.train_aug:
            x = self._augment(x)
        return x, label


# ---------------------------- Model ----------------------------

class ObjOnlyCNN(nn.Module):
    """4-layer CNN with categorical embedding, 32x32 input → 33-class.

    obj_id (B, 1, 32, 32) long → embed (B, D, 32, 32) → conv → GAP → head.
    ~0.5M parameters.
    """
    def __init__(self, num_classes: int = 33, n_chip_objects: int = 5,
                 embed_dim: int = 16, drop_path: float = 0.1):
        super().__init__()
        self.num_classes = num_classes
        self.n_chip_objects = n_chip_objects
        self.embed_dim = embed_dim
        # Embedding (n_chip_objects + 1 because obj_id 0 = none + 1..N obj)
        self.embed = nn.Embedding(n_chip_objects + 1, embed_dim)
        # 32×32 → 4 conv stages (stride 2 each, but first 2 stay at 32×32)
        self.conv = nn.Sequential(
            nn.Conv2d(embed_dim, 32, 3, padding=1),
            nn.BatchNorm2d(32), nn.GELU(),                                                # 32×32×32
            nn.Conv2d(32, 64, 3, padding=1, stride=2),
            nn.BatchNorm2d(64), nn.GELU(),                                                # 16×16×64
            nn.Conv2d(64, 128, 3, padding=1, stride=2),
            nn.BatchNorm2d(128), nn.GELU(),                                               # 8×8×128
            nn.Conv2d(128, 256, 3, padding=1, stride=2),
            nn.BatchNorm2d(256), nn.GELU(),                                               # 4×4×256
        )
        self.drop = nn.Dropout(drop_path)
        self.head = nn.Linear(256, num_classes)
        self.feature_dim = 256

    def forward_features(self, obj_id_long: torch.Tensor) -> torch.Tensor:
        """(B, 1, 32, 32) long → (B, 256, 4, 4)."""
        if obj_id_long.dim() == 4 and obj_id_long.size(1) == 1:
            obj_id_long = obj_id_long.squeeze(1)                                          # (B, 32, 32)
        emb = self.embed(obj_id_long.long())                                              # (B, 32, 32, D)
        emb = emb.permute(0, 3, 1, 2).contiguous()                                        # (B, D, 32, 32)
        return self.conv(emb)                                                             # (B, 256, 4, 4)

    def forward(self, obj_id_long: torch.Tensor) -> torch.Tensor:
        feat = self.forward_features(obj_id_long)
        feat = F.adaptive_avg_pool2d(feat, 1).flatten(1)                                  # (B, 256)
        feat = self.drop(feat)
        return self.head(feat)


# ---------------------------- Train / Eval ----------------------------

@torch.no_grad()
def evaluate(model, loader, device, classes) -> Dict:
    model.eval()
    all_p, all_l = [], []
    losses = []
    crit = nn.CrossEntropyLoss()
    for x, y in loader:
        x = x.to(device, non_blocking=True); y = y.to(device, non_blocking=True)
        logits = model(x)
        losses.append(float(crit(logits, y)))
        preds = logits.argmax(1).cpu().numpy()
        all_p.append(preds); all_l.append(y.cpu().numpy())
    P = np.concatenate(all_p); L = np.concatenate(all_l)
    acc = accuracy_score(L, P)
    p, r, f1, _ = precision_recall_fscore_support(L, P, average="macro", zero_division=0)
    return {"acc": float(acc), "macro_f1": float(f1), "macro_p": float(p),
            "macro_r": float(r), "loss": float(np.mean(losses)) if losses else 0.0}


def stratified_split(targets: List[int], ratios: Tuple[float, float, float], seed: int):
    rng = np.random.default_rng(seed)
    targets = np.array(targets)
    tr, va, te = [], [], []
    for c in np.unique(targets):
        idx = np.where(targets == c)[0]
        rng.shuffle(idx)
        n = len(idx)
        n_tr = int(n * ratios[0])
        n_va = int(n * ratios[1])
        tr.extend(idx[:n_tr].tolist())
        va.extend(idx[n_tr:n_tr + n_va].tolist())
        te.extend(idx[n_tr + n_va:].tolist())
    return tr, va, te


def apply_subset(samples, classes, subset_dict, seed):
    rng = np.random.default_rng(seed)
    by_cls = {c: [] for c in classes}
    for s in samples:
        by_cls[classes[s[1]]].append(s)
    out = []
    default_n = subset_dict.get("default")
    for c, lst in by_cls.items():
        n = subset_dict.get(c, default_n)
        if n is None: out.extend(lst)
        else:
            idx = rng.permutation(len(lst))[:n]
            out.extend([lst[i] for i in idx])
    return out


def setup_logger(out_dir: Path):
    import logging
    out_dir.mkdir(parents=True, exist_ok=True)
    lg = logging.getLogger(f"objonly_{out_dir.name}")
    lg.handlers.clear(); lg.setLevel(logging.INFO)
    fh = logging.FileHandler(out_dir / "run.log", encoding="utf-8")
    fh.setFormatter(logging.Formatter("%(asctime)s %(message)s"))
    lg.addHandler(fh)
    sh = logging.StreamHandler(sys.stderr)
    sh.setFormatter(logging.Formatter("%(message)s"))
    lg.addHandler(sh)
    return lg


def update_overall_best(log_root: Path, out_dir: Path, val_f1: float):
    overall = log_root / "overall"
    meta_path = overall / "_overall_meta.json"
    prev_f1 = -1.0
    if meta_path.exists():
        try:
            prev_f1 = float(json.loads(meta_path.read_text()).get("val_f1", -1.0))
        except Exception: pass
    if val_f1 > prev_f1:
        if overall.exists():
            shutil.rmtree(overall)
        shutil.copytree(out_dir, overall)
        meta_path.write_text(json.dumps({
            "val_f1": float(val_f1),
            "source_run": str(out_dir.name),
            "updated_at": datetime.now().isoformat(),
        }, indent=2))
        print(f"[overall] {prev_f1:.4f} → {val_f1:.4f}", file=sys.stderr)


# ---------------------------- Main ----------------------------

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", default=DEFAULT_DATA_DIR)
    p.add_argument("--obj-id-dir", default=DEFAULT_OBJ_ID_DIR)
    p.add_argument("--log-root", default=DEFAULT_LOG_ROOT)
    p.add_argument("--model-tag", default="objonly")
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch", type=int, default=32)
    p.add_argument("--workers", type=int, default=0)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--wd", type=float, default=0.05)
    p.add_argument("--embed-dim", type=int, default=16)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--subset-config", default=None)
    p.add_argument("--active-classes-yaml", default=None,
                   help="YAML with classes: list — 학습 시 그 class 만 사용. "
                        "default: 모든 class (data-dir 의 폴더명 기반)")
    p.add_argument("--train-val-only", action="store_true",
                   help="0.8/0.2 split, no test")
    p.add_argument("--device", default=None)
    args = p.parse_args()

    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.benchmark = True

    device = torch.device(args.device) if args.device else torch.device(
        "cuda" if torch.cuda.is_available() else "cpu")
    ts = datetime.now().strftime("%y%m%d_%H%M%S")
    out_dir = Path(args.log_root) / f"{args.model_tag}_{ts}_running"
    lg = setup_logger(out_dir)
    lg.info(f"===== obj-only train start =====")
    lg.info(f"device={device}  data={args.data_dir}  obj_id={args.obj_id_dir}")
    lg.info(f"epochs={args.epochs} batch={args.batch} lr={args.lr}")

    # Read n_chip_objects from obj_id_maps/_meta.json
    meta_path = Path(args.obj_id_dir) / "_meta.json"
    if meta_path.exists():
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        n_chip_objects = int(meta["n_chip_objects"])
        chip_classes = meta.get("chip_classes", [])
        lg.info(f"[meta] n_chip_objects={n_chip_objects} chip_classes={chip_classes}")
    else:
        n_chip_objects = 5
        lg.info(f"[meta] {meta_path} not found, fallback n_chip_objects=5")

    # Resolve wafer classes from data_dir folder names
    all_dirs = sorted([d for d in os.listdir(args.data_dir)
                       if (Path(args.data_dir) / d).is_dir() and d not in EXCLUDE_CLASSES])
    if args.active_classes_yaml:
        try:
            import yaml
            with open(args.active_classes_yaml, encoding="utf-8") as f:
                yc = yaml.safe_load(f)
            active = list(yc.get("classes", []))
            classes = [c for c in active if c in set(all_dirs)]
            missing = set(active) - set(all_dirs)
            if missing:
                lg.info(f"[active-classes] missing in data_dir: {sorted(missing)}")
            lg.info(f"[active-classes] {len(classes)} classes from {args.active_classes_yaml}")
        except Exception as e:
            lg.info(f"[active-classes] load fail: {e} — fallback all classes")
            classes = all_dirs
    else:
        classes = all_dirs
    num_classes = len(classes)
    lg.info(f"[classes] {num_classes}: {classes}")

    # Datasets
    train_ds = ObjOnlyDataset(args.data_dir, args.obj_id_dir, classes,
                                train_aug=True, n_chip_objects=n_chip_objects)
    eval_ds = ObjOnlyDataset(args.data_dir, args.obj_id_dir, classes,
                               train_aug=False, n_chip_objects=n_chip_objects)
    if len(train_ds) == 0:
        raise SystemExit(f"no samples found — check {args.data_dir} + {args.obj_id_dir}")

    # Subset
    if args.subset_config:
        with open(args.subset_config, encoding="utf-8") as f:
            yc = yaml.safe_load(f)
        sd = yc.get("classes", yc) if isinstance(yc, dict) else {}
        train_ds.samples = apply_subset(train_ds.samples, classes, sd, args.seed)
        eval_ds.samples = apply_subset(eval_ds.samples, classes, sd, args.seed)
        train_ds.targets = [t for _, t in train_ds.samples]
        eval_ds.targets = [t for _, t in eval_ds.samples]
        lg.info(f"[subset] applied — total={len(train_ds)}")

    lg.info(f"[total] {len(train_ds)} samples")

    # Split
    ratios = (0.8, 0.2, 0.0) if args.train_val_only else (0.7, 0.15, 0.15)
    tr_idx, va_idx, te_idx = stratified_split(eval_ds.targets, ratios, args.seed)
    tr_set = Subset(train_ds, tr_idx)
    va_set = Subset(eval_ds, va_idx)
    te_set = Subset(eval_ds, te_idx) if te_idx else None
    lg.info(f"[split] train={len(tr_set)} val={len(va_set)} test={len(te_idx)}")

    train_ld = DataLoader(tr_set, batch_size=args.batch, shuffle=True,
                          num_workers=args.workers, pin_memory=(device.type == "cuda"))
    val_ld = DataLoader(va_set, batch_size=args.batch, shuffle=False,
                        num_workers=args.workers, pin_memory=(device.type == "cuda"))
    test_ld = DataLoader(te_set, batch_size=args.batch, shuffle=False,
                          num_workers=args.workers, pin_memory=(device.type == "cuda")) if te_set else None

    # Model
    model = ObjOnlyCNN(num_classes=num_classes, n_chip_objects=n_chip_objects,
                        embed_dim=args.embed_dim).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    lg.info(f"[model] ObjOnlyCNN n_params={n_params/1e6:.2f}M (embed_dim={args.embed_dim})")

    # Optimizer + scheduler
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs * len(train_ld))
    crit = nn.CrossEntropyLoss(label_smoothing=0.05)

    # Train loop
    best_f1 = -1.0
    history = []
    for ep in range(1, args.epochs + 1):
        t0 = time.time()
        model.train()
        ep_losses = []; n_correct = 0; n_total = 0
        for x, y in train_ld:
            x = x.to(device, non_blocking=True); y = y.to(device, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            logits = model(x)
            loss = crit(logits, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            ep_losses.append(float(loss))
            n_correct += int((logits.argmax(1) == y).sum())
            n_total += y.size(0)
        train_loss = float(np.mean(ep_losses)) if ep_losses else 0.0
        train_acc = n_correct / max(n_total, 1)

        val = evaluate(model, val_ld, device, classes)
        elapsed = time.time() - t0
        history.append({
            "epoch": ep, "train_loss": train_loss, "train_acc": train_acc,
            "val_loss": val["loss"], "val_acc": val["acc"],
            "val_macro_f1": val["macro_f1"], "val_macro_p": val["macro_p"],
            "val_macro_r": val["macro_r"], "elapsed_sec": elapsed,
        })
        lg.info(f"Ep {ep:3d}/{args.epochs} | train_loss={train_loss:.4f} acc={100*train_acc:.2f}% "
                f"| val_loss={val['loss']:.4f} acc={100*val['acc']:.2f}% f1={100*val['macro_f1']:.2f}% "
                f"| {elapsed:.1f}s")

        # Save history
        (out_dir / "history.json").write_text(json.dumps(history, indent=2))

        # Save best
        if val["macro_f1"] > best_f1:
            best_f1 = val["macro_f1"]
            ckpt = {
                "epoch": ep, "model": model.state_dict(),
                "classes": classes, "num_classes": num_classes,
                "n_chip_objects": n_chip_objects,
                "embed_dim": args.embed_dim,
                "feature_dim": model.feature_dim,
                "arch": "ObjOnlyCNN",
                "val_macro_f1": float(best_f1),
                "val_acc": float(val["acc"]),
            }
            torch.save(ckpt, out_dir / "best_model.pth")

    # Final test
    if test_ld:
        # Load best for test eval
        ck = torch.load(out_dir / "best_model.pth", map_location=device, weights_only=False)
        model.load_state_dict(ck["model"])
        test_res = evaluate(model, test_ld, device, classes)
        lg.info(f"[TEST] acc={100*test_res['acc']:.2f}% f1={100*test_res['macro_f1']:.2f}%")
        ck["test_macro_f1"] = float(test_res["macro_f1"])
        ck["test_acc"] = float(test_res["acc"])
        torch.save(ck, out_dir / "best_model.pth")

    # Hparams + summary
    (out_dir / "hparams.yaml").write_text(json.dumps(vars(args), indent=2))
    (out_dir / "best_history.txt").write_text(
        f"BEST OVERALL\n  val_macro_f1: {best_f1:.4f}\n"
        f"  arch: ObjOnlyCNN  embed_dim={args.embed_dim}  n_params={n_params/1e6:.2f}M\n"
        f"  data: {args.data_dir}\n  obj_id: {args.obj_id_dir}\n"
        f"  total samples: {len(train_ds)}\n"
        f"  train/val/test: {len(tr_set)}/{len(va_set)}/{len(te_idx)}\n"
        f"  epochs: {args.epochs} batch: {args.batch} lr: {args.lr}\n",
        encoding="utf-8")

    # Close logger handlers (Windows lock)
    for h in list(lg.handlers):
        try: h.close(); lg.removeHandler(h)
        except Exception: pass

    # Rename out_dir
    final_dir = out_dir.parent / out_dir.name.replace(
        "_running", f"_f{best_f1:.2f}".replace(".", ""))
    try:
        if final_dir.exists(): shutil.rmtree(final_dir)
        out_dir.rename(final_dir)
    except Exception as e:
        print(f"[!] rename failed (Windows file lock?): {e}", file=sys.stderr)
        final_dir = out_dir

    update_overall_best(Path(args.log_root), final_dir, best_f1)
    print(f"\n[OK] best val_macro_f1 = {best_f1:.4f}\n     output: {final_dir}")


if __name__ == "__main__":
    main()
