"""Operator-match on REAL public SVHN house numbers.

The public counterpart to the private chip overlay-vs-FCM-PM crossover, on REAL
photographed data (not author-constructed).

  SINGLE-LABEL TRAIN pool = real SVHN single-digit crops (format-1 bboxes).
  REAL MULTI-LABEL EVAL   = real SVHN full house numbers with exactly k digits
                            (format-1), multi-hot over the 10 digit classes =
                            WHICH digits appear. A REAL partition-law test set.

Arms (all share the same real single-digit pool, differ only in the combination
operator used to synthesize two-label training canvases):

  single_only : singles only (floor)
  partition   : tile two digits into disjoint horizontal cells (matched to the
                real side-by-side house-number layout)  == FCM-PM analogue
  overlay     : max-union two digits in one cell (mismatched)  == superposition
  cutmix      : random-rectangle paste (content-blind)
  mixup       : 0.5/0.5 blend (content-blind)
  oracle      : trained on REAL multi-digit house numbers (upper reference)

HYPOTHESIS: partition (matched to the real house-number partition law) beats
overlay AND cutmix/mixup on the REAL multi-digit eval, and recovers the oracle
-- reproducing the chip crossover on REAL PUBLIC data.

Grayscale (a house-number digit is legible in grayscale) so the SmallCNN
grayscale pipeline is reused directly; --backbone resnet18 adds capacity.
"""
import os
import csv as csvmod
import argparse
import time

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

from .datasets.multimnist import all_pairs
from .datasets.svhn_format1 import load_single_digit_pool, load_multidigit
from .datasets import svhn_operator as ops
from .metrics import bit_f1, far, compute_map, pos_neg_prob

ARMS = ["single_only", "partition", "overlay", "cutmix", "mixup", "oracle"]
MATCHED = "partition"
CONTENT_BLIND = ["overlay", "cutmix", "mixup"]     # matched must dominate these
FIELDS = ["dataset", "arm", "matched", "backbone", "seed", "bit_f1", "mAP",
          "far", "normal_far", "pos_prob", "neg_prob"]


# --------------------------------------------------------------------------- #
# models / train
# --------------------------------------------------------------------------- #
def _build_model(backbone, num_classes=10):
    if backbone == "smallcnn":
        from .models.small_cnn import SmallCNN
        return SmallCNN(num_classes=num_classes, in_ch=1)
    if backbone == "resnet18":
        from torchvision.models import resnet18
        m = resnet18(weights=None, num_classes=num_classes)
        # 1-channel stem, lighter downsampling for small 32x64 canvases
        m.conv1 = nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1, bias=False)
        m.maxpool = nn.Identity()
        return m
    raise ValueError(f"unknown backbone {backbone}")


def train_model(trX, trY, backbone, epochs, bs, lr, device, seed):
    torch.manual_seed(seed)
    model = _build_model(backbone, num_classes=trY.shape[1]).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    lossf = nn.BCEWithLogitsLoss()
    ds = TensorDataset(torch.from_numpy(trX), torch.from_numpy(trY))
    loader = DataLoader(ds, batch_size=bs, shuffle=True)
    for _ in range(epochs):
        model.train()
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            loss = lossf(model(xb), yb)
            loss.backward()
            opt.step()
    return model


def _predict(model, X, bs, device):
    model.eval()
    P = []
    with torch.no_grad():
        for (xb,) in DataLoader(TensorDataset(torch.from_numpy(X)),
                                batch_size=bs):
            P.append(torch.sigmoid(model(xb.to(device))).cpu().numpy())
    return np.concatenate(P)


# --------------------------------------------------------------------------- #
# data prep (cached)
# --------------------------------------------------------------------------- #
def prepare_data(train_dir, test_dir, digit_size, grid, digit_counts,
                 per_class_cap, max_train_images, max_test_images, cache):
    if cache and os.path.isfile(cache):
        z = np.load(cache, allow_pickle=True)
        print(f"[cache] loaded {cache}", flush=True)
        return (z["pool_imgs"], z["pool_labels"], z["oracle_imgs"],
                z["oracle_Y"], z["eval_imgs"], z["eval_Y"],
                z["meta"].item())
    t0 = time.time()
    pool_imgs, pool_labels = load_single_digit_pool(
        train_dir, digit_size=digit_size, max_images=max_train_images,
        per_class_cap=per_class_cap, seed=0)
    print(f"[prep] single-digit pool {pool_imgs.shape} "
          f"({time.time()-t0:.0f}s)", flush=True)
    canvas_w = digit_size * grid
    oracle_imgs, oracle_Y, om = load_multidigit(
        train_dir, canvas_w=canvas_w, canvas_h=digit_size,
        digit_counts=digit_counts, require_distinct=True,
        max_images=max_train_images)
    print(f"[prep] oracle real multi-digit {oracle_imgs.shape} "
          f"{om} ({time.time()-t0:.0f}s)", flush=True)
    eval_imgs, eval_Y, em = load_multidigit(
        test_dir, canvas_w=canvas_w, canvas_h=digit_size,
        digit_counts=digit_counts, require_distinct=True,
        max_images=max_test_images)
    print(f"[prep] REAL eval multi-digit {eval_imgs.shape} "
          f"{em} ({time.time()-t0:.0f}s)", flush=True)
    meta = dict(pool=list(pool_imgs.shape), oracle=om, eval=em,
                digit_counts=list(digit_counts))
    if cache:
        os.makedirs(os.path.dirname(cache), exist_ok=True)
        np.savez_compressed(cache, pool_imgs=pool_imgs, pool_labels=pool_labels,
                            oracle_imgs=oracle_imgs, oracle_Y=oracle_Y,
                            eval_imgs=eval_imgs, eval_Y=eval_Y,
                            meta=np.array(meta, dtype=object))
        print(f"[cache] saved {cache}", flush=True)
    return (pool_imgs, pool_labels, oracle_imgs, oracle_Y, eval_imgs, eval_Y,
            meta)


def build_train_for_arm(arm, pool_imgs, pool_labels, oracle_imgs, oracle_Y,
                        pairs, per_class_single, n_train, seed, digit_size,
                        grid, n_classes=10):
    spX, spY = ops.build_singles(pool_imgs, pool_labels, per_class_single, seed,
                                 digit_size=digit_size, grid=grid,
                                 n_classes=n_classes)
    if arm == "single_only":
        return spX, spY
    if arm == "oracle":
        rng = np.random.default_rng(seed)
        k = min(n_train, oracle_imgs.shape[0])
        idx = rng.choice(oracle_imgs.shape[0], size=k, replace=False)
        return (np.concatenate([oracle_imgs[idx], spX]),
                np.concatenate([oracle_Y[idx], spY]))
    if arm in ("cutmix", "mixup"):
        cX, cY = ops.build_multi_baseline(pool_imgs, pool_labels, n_train, seed,
                                          pairs, arm, digit_size=digit_size,
                                          grid=grid, n_classes=n_classes)
        return np.concatenate([cX, spX]), np.concatenate([cY, spY])
    law = "partition" if arm == "partition" else "superposition"
    cX, cY = ops.build_multi(pool_imgs, pool_labels, n_train, seed, pairs, law,
                             digit_size=digit_size, grid=grid,
                             n_classes=n_classes)
    return np.concatenate([cX, spX]), np.concatenate([cY, spY])


# --------------------------------------------------------------------------- #
# run
# --------------------------------------------------------------------------- #
def run(data, arms, seeds, per_class_single, n_train, n_normal, epochs, bs, lr,
        device, digit_size, grid, backbone, out_csv, dataset="svhn"):
    pool_imgs, pool_labels, oracle_imgs, oracle_Y, eval_imgs, eval_Y, meta = data
    pairs = all_pairs(10)
    nX, _ = ops.build_normal(n_normal, digit_size=digit_size, grid=grid)
    rows = []
    for arm in arms:
        for seed in seeds:
            trX, trY = build_train_for_arm(
                arm, pool_imgs, pool_labels, oracle_imgs, oracle_Y, pairs,
                per_class_single, n_train, seed, digit_size, grid)
            model = train_model(trX, trY, backbone, epochs, bs, lr, device, seed)
            P = _predict(model, eval_imgs, bs, device)
            nP = _predict(model, nX, bs, device)
            pos, neg = pos_neg_prob(P, eval_Y)
            r = {
                "dataset": dataset, "arm": arm, "matched": MATCHED,
                "backbone": backbone, "seed": seed,
                "bit_f1": round(float(bit_f1(P, eval_Y)), 4),
                "mAP": round(float(compute_map(P, eval_Y)), 4),
                "far": round(float(far(P, eval_Y)), 4),
                "normal_far": round(float((nP >= 0.5).any(1).mean()), 4),
                "pos_prob": round(float(pos), 4),
                "neg_prob": round(float(neg), 4),
            }
            rows.append(r)
            mark = "MATCH" if arm == MATCHED else ""
            print(f"[svhn][{backbone}] {arm:12s} s{seed} "
                  f"bitF1={r['bit_f1']:.4f} mAP={r['mAP']:.4f} "
                  f"FAR={r['far']:.4f} nFAR={r['normal_far']:.4f} "
                  f"pos={r['pos_prob']:.4f} neg={r['neg_prob']:.4f} {mark}",
                  flush=True)
    if out_csv:
        os.makedirs(os.path.dirname(out_csv), exist_ok=True)
        with open(out_csv, "w", newline="") as f:
            w = csvmod.DictWriter(f, fieldnames=FIELDS)
            w.writeheader(); w.writerows(rows)
    return rows


def _agg(rows, arm, key):
    vals = [r[key] for r in rows if r["arm"] == arm]
    return (float(np.mean(vals)), float(np.std(vals))) if vals else (float("nan"),
                                                                     float("nan"))


def print_table(rows, backbone):
    print(f"\n=== OPERATOR-MATCH ON REAL SVHN HOUSE NUMBERS "
          f"(backbone={backbone}, mean +/- std over seeds) ===")
    print("| Arm         | matched | bit_F1 (mean+/-std) | mAP (mean+/-std)   "
          "| FAR    | nFAR   |")
    print("|-------------|---------|---------------------|--------------------"
          "|--------|--------|")
    for arm in ARMS:
        if not any(r["arm"] == arm for r in rows):
            continue
        bm, bs_ = _agg(rows, arm, "bit_f1")
        mm, ms = _agg(rows, arm, "mAP")
        fm, _ = _agg(rows, arm, "far")
        nm, _ = _agg(rows, arm, "normal_far")
        mk = "yes" if arm == MATCHED else ""
        print(f"| {arm:11s} | {mk:7s} | {bm:.4f} +/- {bs_:.4f}  "
              f"| {mm:.4f} +/- {ms:.4f} | {fm:.4f} | {nm:.4f} |")


def print_verdict(rows, tol=0.02):
    print("\n=== VERDICT (REAL SVHN partition-law eval) ===")
    bm, bmstd = _agg(rows, MATCHED, "bit_f1")
    boc, _ = _agg(rows, "oracle", "bit_f1")
    recovers_oracle = bm >= (boc - tol)
    beaten_by, overlaps = [], []
    for base in CONTENT_BLIND:
        if not any(r["arm"] == base for r in rows):
            continue
        bb, bbstd = _agg(rows, base, "bit_f1")
        clean_win = (bm - bmstd) > (bb + bbstd)
        print(f"  {MATCHED} {bm:.4f}+/-{bmstd:.4f}  vs  {base:8s} "
              f"{bb:.4f}+/-{bbstd:.4f}  -> clean_win={clean_win}")
        if bb >= bm:
            beaten_by.append(base)
        if not clean_win:
            overlaps.append(base)
    print(f"  oracle bit_F1={boc:.4f} -> recovers_oracle={recovers_oracle} "
          f"(partition {bm:.4f} >= oracle-{tol})")
    win = (len(beaten_by) == 0) and (len(overlaps) == 0) and recovers_oracle
    if win:
        verdict = "WIN: operator-match partition reproduces on REAL SVHN"
    elif beaten_by:
        verdict = (f"NO-WIN: partition tied/beaten by {beaten_by} on REAL SVHN")
    elif overlaps:
        verdict = (f"TIE: partition best-mean but std-band overlaps {overlaps}")
    else:
        verdict = "NO-WIN: partition does not recover oracle"
    print(f"\nPLAIN VERDICT: {verdict}")
    print(f"  partition_wins_all_content_blind={len(beaten_by)==0 and len(overlaps)==0}"
          f"  recovers_oracle={recovers_oracle}  FLIP_REPRODUCES={win}")
    return win


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arms", nargs="+", default=ARMS)
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    ap.add_argument("--per-class-single", type=int, default=300)
    ap.add_argument("--n-train", type=int, default=3000)
    ap.add_argument("--n-normal", type=int, default=1000)
    ap.add_argument("--epochs", type=int, default=25)
    ap.add_argument("--bs", type=int, default=128)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--backbone", default="smallcnn",
                    choices=["smallcnn", "resnet18"])
    ap.add_argument("--digit-size", type=int, default=32)
    ap.add_argument("--grid", type=int, default=2)
    ap.add_argument("--digit-counts", nargs="+", type=int, default=[2])
    ap.add_argument("--per-class-cap", type=int, default=3000)
    ap.add_argument("--max-train-images", type=int, default=None)
    ap.add_argument("--max-test-images", type=int, default=None)
    ap.add_argument("--root", default="E:/data/torchvision/SVHN_format1")
    ap.add_argument("--cache", default=None)
    ap.add_argument("--out-csv",
                    default="outputs/multilabel_synth/operator_match_svhn.csv")
    args = ap.parse_args()

    if args.device == "cuda" and not torch.cuda.is_available():
        print("[warn] cuda unavailable -> cpu", flush=True)
        args.device = "cpu"

    train_dir = os.path.join(args.root, "train")
    test_dir = os.path.join(args.root, "test")
    data = prepare_data(train_dir, test_dir, args.digit_size, args.grid,
                        tuple(args.digit_counts), args.per_class_cap,
                        args.max_train_images, args.max_test_images, args.cache)
    print(f"[meta] {data[-1]}", flush=True)
    rows = run(data, args.arms, args.seeds, args.per_class_single, args.n_train,
               args.n_normal, args.epochs, args.bs, args.lr, args.device,
               args.digit_size, args.grid, args.backbone, args.out_csv)
    print_table(rows, args.backbone)
    print_verdict(rows)
    print(f"\n[OUT] {os.path.abspath(args.out_csv)}")


if __name__ == "__main__":
    main()
