#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Stage B — ChipGridCNN + KDE/GMM late fusion (joint train).

V3 (chip-grid 32 native, one-hot 5ch) + Stage A 의 KDE/GMM auxiliary feature concat → small MLP head.

아키텍처:
    입력 (32×32×6, R + 5 obj binary)
      ↓ ChipGridCNN body (cnn_eval_chipgrid 의 ChipGridCNN.feat 까지)
      ↓ penultimate 256-D
                                     + KDE log-lik (n_classes-D)
                                     + GMM log-lik (n_classes-D)
      ↓ Concat → (256 + 2*n_classes)-D
      ↓ BN1d + Linear(., 128) + GELU + Dropout
      ↓ Linear(128, n_classes)

KDE/GMM 은 train split 에서 사전 학습 (Stage A 의 함수 재사용), val/test 에 적용 (data leakage 0).

사용:
    python cnn_train_chipgrid_fusion.py --n-per-class 100 --epochs 30 --seed 42 --model-tag fusion_seed42
"""
from __future__ import annotations
import argparse, json, random, time
from pathlib import Path
from datetime import datetime
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import f1_score, accuracy_score, precision_recall_fscore_support

# Reuse from cnn_eval_chipgrid.py (data loading + V3 architecture)
import sys
sys.path.insert(0, str(Path(__file__).parent))
from cnn_eval_chipgrid import (
    DEFAULT_DATA_DIR, DEFAULT_OBJ_ID_DIR, GRID_SIZE, PALETTE_MAX,
    build_npy_lookup, collect_samples, split_samples,
    _load_active_classes, _load_r_static, _load_obj_static, encode_v3, EMA, per_class_report,
)
from _chipgrid_kde_gmm import (
    extract_obj_id_arrays, train_kde_per_class, train_gmm_per_class,
    compute_log_lik_one,
)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--n-per-class", type=int, default=100)
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch", type=int, default=32)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=0.05)
    p.add_argument("--label-smoothing", type=float, default=0.02)
    p.add_argument("--ema-decay", type=float, default=0.95)
    p.add_argument("--ema-warmup", type=int, default=3)
    p.add_argument("--patience", type=int, default=7)
    p.add_argument("--bandwidth", type=float, default=1.0, help="KDE bandwidth")
    p.add_argument("--n-components", type=int, default=2, help="GMM n_components")
    p.add_argument("--mlp-hidden", type=int, default=128)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--image-branch", default="r-plus-v3",
                   choices=["r-only", "v3", "r-plus-v3"],
                   help="image branch input: r-only, v3 object one-hot only, or existing r-plus-v3")
    p.add_argument("--data-dir", default=DEFAULT_DATA_DIR)
    p.add_argument("--obj-id-dir", default=DEFAULT_OBJ_ID_DIR)
    p.add_argument("--active-classes-yaml", default=None,
                   help="YAML with classes: list. Limits fusion training to those classes.")
    p.add_argument("--allow-missing-active-classes", action="store_true",
                   help="Drop active classes missing from data-dir instead of failing.")
    p.add_argument("--log-root", default="logs_chipgrid_fusion")
    p.add_argument("--model-tag", default=None)
    return p.parse_args()


# === Dataset (preload R + obj_id + KDE/GMM features) ===
class FusionDataset(Dataset):
    """Returns ((x_img Cx32x32, x_aux 2*n_classes), label).

    x_img: selected image branch (`r-only`, `v3`, or `r-plus-v3`).
    x_aux: concat of (kde_log_lik 33-D, gmm_log_lik 33-D) precomputed.
    """
    def __init__(self, samples, npy_lookup, kde_dict, gmm_dict, n_classes,
                 image_branch="r-plus-v3", log_fn=None):
        self.samples = samples
        self.n_classes = n_classes
        self.image_branch = image_branch
        # preload all
        t0 = time.time()
        r_list, obj_list, aux_list = [], [], []
        n_missing = 0
        for png_path, _ in samples:
            r = _load_r_static(png_path)
            obj = _load_obj_static(png_path, npy_lookup)
            if obj is None:
                obj = np.zeros((GRID_SIZE, GRID_SIZE), dtype=np.uint8)
                n_missing += 1
            r_list.append(r)
            obj_list.append(obj)
            kde_log, gmm_log = compute_log_lik_one(obj, kde_dict, gmm_dict, n_classes)
            aux_list.append(np.concatenate([kde_log, gmm_log], dtype=np.float32))
        self._cache_r = np.stack(r_list, axis=0)             # (N, 1, 32, 32)
        self._cache_obj = np.stack(obj_list, axis=0)         # (N, 32, 32)
        self._cache_aux = np.stack(aux_list, axis=0)         # (N, 2*n_classes)
        msg = (f"[fusion-cache] loaded {len(samples)} samples in {time.time()-t0:.1f}s "
               f"(missing obj: {n_missing}, aux shape: {self._cache_aux.shape})")
        if log_fn: log_fn(msg)
        else: print(msg, flush=True)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        r = self._cache_r[idx]
        obj_arr = self._cache_obj[idx]
        obj_ch = encode_v3(obj_arr)                          # (5, 32, 32)
        if self.image_branch == "r-only":
            x_img = r                                        # (1, 32, 32)
        elif self.image_branch == "v3":
            x_img = obj_ch                                   # (5, 32, 32)
        else:
            x_img = np.concatenate([r, obj_ch], axis=0)      # (6, 32, 32)
        x_aux = self._cache_aux[idx]                         # (2*n_classes,)
        label = self.samples[idx][1]
        return (torch.from_numpy(x_img).float(),
                torch.from_numpy(x_aux).float(),
                torch.tensor(label, dtype=torch.long))


# === Model ===
class FusionModel(nn.Module):
    """ChipGridCNN body + concat KDE/GMM aux + small MLP head."""
    def __init__(self, in_ch_img: int, n_classes: int, aux_dim: int,
                 mlp_hidden: int = 128, dropout: float = 0.1):
        super().__init__()
        # body: same as ChipGridCNN.feat but stop before classifier
        self.body = nn.Sequential(
            nn.Conv2d(in_ch_img, 64, 3, padding=1), nn.BatchNorm2d(64), nn.GELU(),
            nn.Conv2d(64, 64, 3, padding=1), nn.BatchNorm2d(64), nn.GELU(),
            nn.Conv2d(64, 128, 3, padding=1, stride=2), nn.BatchNorm2d(128), nn.GELU(),
            nn.Conv2d(128, 128, 3, padding=1), nn.BatchNorm2d(128), nn.GELU(),
            nn.Conv2d(128, 256, 3, padding=1, stride=2), nn.BatchNorm2d(256), nn.GELU(),
            nn.Conv2d(256, 256, 3, padding=1), nn.BatchNorm2d(256), nn.GELU(),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
        )
        self.body_dim = 256
        # auxiliary normalization (BN1d on KDE/GMM log-lik 으로 scale 정규화)
        self.aux_bn = nn.BatchNorm1d(aux_dim)
        # fusion head
        fused_dim = self.body_dim + aux_dim
        self.head = nn.Sequential(
            nn.BatchNorm1d(fused_dim),
            nn.Linear(fused_dim, mlp_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden, n_classes),
        )

    def forward(self, x_img, x_aux):
        feat = self.body(x_img)                              # (B, 256)
        aux = self.aux_bn(x_aux)                             # (B, aux_dim)
        fused = torch.cat([feat, aux], dim=1)                # (B, 256 + aux_dim)
        return self.head(fused)


# === Eval ===
@torch.no_grad()
def evaluate_fusion(model, loader, device):
    model.eval()
    all_p, all_t = [], []
    for x_img, x_aux, y in loader:
        x_img = x_img.to(device, non_blocking=True)
        x_aux = x_aux.to(device, non_blocking=True)
        logits = model(x_img, x_aux)
        all_p.append(logits.argmax(1).cpu().numpy())
        all_t.append(y.numpy())
    p = np.concatenate(all_p)
    t = np.concatenate(all_t)
    return accuracy_score(t, p), f1_score(t, p, average="macro", zero_division=0), t, p


# === Main ===
def main():
    args = parse_args()
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    ts = datetime.now().strftime("%y%m%d_%H%M%S")
    tag = args.model_tag or f"fusion_n{args.n_per_class}_seed{args.seed}"
    run_dir = Path(args.log_root) / f"{tag}_{ts}_running"
    run_dir.mkdir(parents=True, exist_ok=True)
    log_path = run_dir / "run.log"

    def log(msg):
        line = f"{datetime.now().strftime('%H:%M:%S')}  {msg}"
        print(line, flush=True)
        with open(log_path, "a", encoding="utf-8") as fp:
            fp.write(line + "\n")

    log(f"===== fusion train start =====")
    log(f"args: {vars(args)}")
    log(f"run_dir={run_dir}")

    # Data
    active_classes = _load_active_classes(args.active_classes_yaml)
    if active_classes:
        log(f"[active-classes] {len(active_classes)} classes from {args.active_classes_yaml}")
    samples, classes = collect_samples(
        args.data_dir,
        args.n_per_class,
        active_classes=active_classes,
        allow_missing_active_classes=args.allow_missing_active_classes,
    )
    n_classes = len(classes)
    train_s, val_s, test_s = split_samples(samples, seed=args.seed)
    npy_lookup = build_npy_lookup(args.obj_id_dir)
    log(f"classes ({n_classes}): {classes}")
    log(f"split: train={len(train_s)} val={len(val_s)} test={len(test_s)}")

    # Stage A — KDE/GMM train on train split only (no leakage)
    log("Stage A: train KDE + GMM on train split...")
    train_arr = extract_obj_id_arrays(train_s, npy_lookup, log_fn=log)
    kde_dict, kde_npc = train_kde_per_class(train_arr, n_classes,
                                             bandwidth=args.bandwidth, log_fn=log)
    gmm_dict, gmm_npc = train_gmm_per_class(train_arr, n_classes,
                                             n_components=args.n_components, log_fn=log)

    # Stage B — fusion training
    train_ds = FusionDataset(train_s, npy_lookup, kde_dict, gmm_dict, n_classes,
                             image_branch=args.image_branch, log_fn=log)
    val_ds = FusionDataset(val_s, npy_lookup, kde_dict, gmm_dict, n_classes,
                           image_branch=args.image_branch, log_fn=log)
    test_ds = FusionDataset(test_s, npy_lookup, kde_dict, gmm_dict, n_classes,
                            image_branch=args.image_branch, log_fn=log)

    train_ld = DataLoader(train_ds, batch_size=args.batch, shuffle=True,
                          num_workers=0, pin_memory=True)
    val_ld = DataLoader(val_ds, batch_size=args.batch, shuffle=False, num_workers=0)
    test_ld = DataLoader(test_ds, batch_size=args.batch, shuffle=False, num_workers=0)

    in_ch_img = {"r-only": 1, "v3": 5, "r-plus-v3": 6}[args.image_branch]
    aux_dim = 2 * n_classes
    model = FusionModel(in_ch_img=in_ch_img, n_classes=n_classes, aux_dim=aux_dim,
                        mlp_hidden=args.mlp_hidden, dropout=args.dropout).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    log(f"model: FusionModel  image_branch={args.image_branch}  in_ch_img={in_ch_img}  aux_dim={aux_dim}  "
        f"hidden={args.mlp_hidden}  params={n_params:,}")

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    crit = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)

    steps_per_epoch = max(1, len(train_ld))
    ema = EMA(model, decay=args.ema_decay, warmup_steps=steps_per_epoch * args.ema_warmup)

    # save hparams
    hp = vars(args).copy()
    hp.update({"in_ch_img": in_ch_img, "aux_dim": aux_dim, "n_classes": n_classes,
               "image_branch": args.image_branch,
               "classes": classes, "n_params": n_params, "device": device,
               "n_train": len(train_s), "n_val": len(val_s), "n_test": len(test_s),
               "run_ts": ts, "torch_version": torch.__version__,
               "kde_n_chips_per_class": kde_npc, "gmm_n_wafers_per_class": gmm_npc})
    with open(run_dir / "hparams.json", "w", encoding="utf-8") as fp:
        json.dump(hp, fp, indent=2, default=str)

    history = []
    best_val_f1 = -1.0
    best_state = {}
    best_updates = []
    no_improve = 0

    for ep in range(1, args.epochs + 1):
        model.train()
        t0 = time.time()
        ep_loss = ep_correct = ep_total = 0
        for x_img, x_aux, y in train_ld:
            x_img = x_img.to(device, non_blocking=True)
            x_aux = x_aux.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            logits = model(x_img, x_aux)
            loss = crit(logits, y)
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
        val_acc, val_f1, y_v_t, y_v_p = evaluate_fusion(model, val_ld, device)
        is_best = val_f1 > best_val_f1
        if is_best:
            test_acc, test_f1, y_t_t, y_t_p = evaluate_fusion(model, test_ld, device)
            torch.save({"model_state": model.state_dict(), "classes": classes,
                        "in_ch_img": in_ch_img, "aux_dim": aux_dim,
                        "val_f1": val_f1, "val_acc": val_acc,
                        "test_f1": test_f1, "test_acc": test_acc,
                        "epoch": ep, "args": vars(args)},
                       run_dir / "best_model.pth")
        ema.restore(model, backup)

        log(f"[Ep {ep}/{args.epochs}] tr_loss={tr_loss:.4f} tr_acc={tr_acc*100:.2f}% | "
            f"val acc={val_acc*100:.2f}% f1={val_f1*100:.2f}% | dt={time.time()-t0:.1f}s")
        history.append({"epoch": ep, "tr_loss": tr_loss, "tr_acc": tr_acc,
                        "val_acc": val_acc, "val_f1": val_f1})

        if is_best:
            best_val_f1 = val_f1
            best_state = {"epoch": ep, "val_f1": val_f1, "val_acc": val_acc,
                          "test_f1": test_f1, "test_acc": test_acc,
                          "y_v_t": y_v_t, "y_v_p": y_v_p,
                          "y_t_t": y_t_t, "y_t_p": y_t_p}
            best_updates.append({"epoch": ep, "val_f1": val_f1, "val_acc": val_acc,
                                  "test_f1": test_f1, "test_acc": test_acc})
            log(f"  [best] val acc={val_acc*100:.2f}% f1={val_f1*100:.2f}% | "
                f"test acc={test_acc*100:.2f}% f1={test_f1*100:.2f}%")
            no_improve = 0
        else:
            no_improve += 1

        with open(run_dir / "history.json", "w", encoding="utf-8") as fp:
            json.dump(history, fp, indent=2)

        if no_improve >= args.patience:
            log(f"early stop at epoch {ep} (patience={args.patience})")
            break

    # write best_history.txt
    bs = best_state
    lines = [
        "=" * 90,
        f"** BEST OVERALL  |  fusion (V3+KDE+GMM)  |  epoch {bs['epoch']}  |  val F1 = {bs['val_f1']:.4f}",
        "=" * 90,
        f"TEST  acc= {bs['test_acc']*100:.2f}%   f1= {bs['test_f1']*100:.2f}%",
        f"VAL   acc= {bs['val_acc']*100:.2f}%   f1= {bs['val_f1']*100:.2f}%",
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
        f"{'ep':>4}  {'val_f1':>8}  {'val_acc':>8}  {'test_f1':>8}  {'test_acc':>8}",
    ]
    for u in best_updates:
        lines.append(f"{u['epoch']:>4}  {u['val_f1']*100:>7.2f}%  {u['val_acc']*100:>7.2f}%  "
                     f"{u['test_f1']*100:>7.2f}%  {u['test_acc']*100:>7.2f}%")
    with open(run_dir / "best_history.txt", "w", encoding="utf-8") as fp:
        fp.write("\n".join(lines))

    final_name = run_dir.parent / f"{tag}_{ts}_{bs['test_f1']:.2f}_{bs['val_f1']:.2f}"
    log(f"renaming run dir to: {final_name.name}")
    try:
        run_dir.rename(final_name)
        print(f"run dir renamed OK: {final_name.name}", flush=True)
    except Exception as e:
        print(f"run dir rename failed: {e}", flush=True)

    print(f"\n=== DONE ===\n"
          f"fusion BEST val_f1={bs['val_f1']:.4f}  test_f1={bs['test_f1']:.4f}  epoch={bs['epoch']}")


if __name__ == "__main__":
    main()
