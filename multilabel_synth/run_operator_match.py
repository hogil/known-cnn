"""Public operator-match demonstration on MNIST.

Two multi-label domains built from the same public source (MNIST), differing in
ONE variable: the domain's true combination law (SUPERPOSITION vs PARTITION).

For each domain we train the shared SmallCNN under four content-blind arms and
evaluate on the domain's held-out real multi-label test set:

  arm 'overlay'     : combos synthesized by max-union in one cell (superposition op)
  arm 'partition'   : combos synthesized by placing each digit in a distinct cell
                      (FCM-PM analogue)
  arm 'single_only' : no combos, single-digit pool only (floor)
  arm 'oracle'      : combos drawn under the domain's TRUE law (upper reference)

Operator-match law prediction:
  PARTITION domain      -> 'partition' (matched) should win
  SUPERPOSITION domain  -> 'overlay'   (matched) should win

That flip is the operator-match law made publicly verifiable.

Same backbone / recipe as the existing MultiMNIST harness (train.train_model:
SmallCNN, Adam 1e-3, BCEWithLogits). Metrics from multilabel_synth.metrics.
"""
import os
import csv as csvmod
import argparse

import numpy as np

from .datasets.multimnist import load_mnist, all_pairs
from .datasets.multimnist_operator import build_singles, build_multi, build_normal
from .train import train_model, evaluate
from .metrics import bit_f1, far, compute_map, pos_neg_prob

REGIMES = ["partition", "superposition"]
ARMS = ["overlay", "partition", "single_only", "oracle"]
MATCHED = {"partition": "partition", "superposition": "overlay"}
FIELDS = ["dataset", "regime", "arm", "matched", "seed", "bit_f1", "mAP", "far",
          "normal_far", "pos_prob", "neg_prob"]


def _predict(model, X, bs, device):
    import torch
    from torch.utils.data import TensorDataset, DataLoader
    model.eval()
    P = []
    with torch.no_grad():
        for (xb,) in DataLoader(TensorDataset(torch.from_numpy(X)),
                                batch_size=bs):
            P.append(torch.sigmoid(model(xb.to(device))).cpu().numpy())
    return np.concatenate(P)


def build_train_for_arm(arm, regime, tr_imgs, tr_labels, pairs,
                        per_class_single, n_train, seed, grid, cell,
                        n_classes=10):
    spX, spY = build_singles(tr_imgs, tr_labels, per_class_single, seed,
                             grid=grid, cell=cell, n_classes=n_classes)
    if arm == "single_only":
        return spX, spY
    if arm == "overlay":
        law = "superposition"
    elif arm == "partition":
        law = "partition"
    elif arm == "oracle":
        law = regime           # the domain's true law
    else:
        raise ValueError(arm)
    cX, cY = build_multi(tr_imgs, tr_labels, n_train, seed, pairs, law,
                         grid=grid, cell=cell, n_classes=n_classes)
    return np.concatenate([cX, spX]), np.concatenate([cY, spY])


def run(tr_imgs, tr_labels, te_imgs, te_labels, arms, seeds,
        per_class_single, n_train, n_test, n_normal,
        epochs, bs, device, grid, cell, out_csv,
        n_classes=10, dataset="mnist"):
    pairs = all_pairs(n_classes)
    rows = []
    for regime in REGIMES:
        # fixed held-out real multi-label test + blank-normal probe (test split)
        teX, teY = build_multi(te_imgs, te_labels, n_test, seed=999,
                               allowed_pairs=pairs, law=regime,
                               grid=grid, cell=cell, n_classes=n_classes)
        nX, _ = build_normal(n_normal, grid=grid, cell=cell, n_classes=n_classes)
        for arm in arms:
            for seed in seeds:
                trX, trY = build_train_for_arm(
                    arm, regime, tr_imgs, tr_labels, pairs,
                    per_class_single, n_train, seed, grid, cell, n_classes)
                model = train_model(trX, trY, epochs=epochs, bs=bs,
                                    device=device, seed=seed)
                P = _predict(model, teX, bs, device)
                nP = _predict(model, nX, bs, device)
                pos, neg = pos_neg_prob(P, teY)
                nfar = float((nP >= 0.5).any(1).mean())
                r = {
                    "dataset": dataset,
                    "regime": regime, "arm": arm, "matched": MATCHED[regime],
                    "seed": seed,
                    "bit_f1": round(float(bit_f1(P, teY)), 4),
                    "mAP": round(float(compute_map(P, teY)), 4),
                    "far": round(float(far(P, teY)), 4),
                    "normal_far": round(nfar, 4),
                    "pos_prob": round(float(pos), 4),
                    "neg_prob": round(float(neg), 4),
                }
                rows.append(r)
                mark = "MATCH" if arm == MATCHED[regime] else ""
                print(f"[{dataset:14s}][{regime:13s}] {arm:12s} s{seed} "
                      f"bitF1={r['bit_f1']:.4f} mAP={r['mAP']:.4f} "
                      f"FAR={r['far']:.4f} nFAR={r['normal_far']:.4f} "
                      f"pos={r['pos_prob']:.4f} neg={r['neg_prob']:.4f} {mark}",
                      flush=True)
    if out_csv:
        os.makedirs(os.path.dirname(out_csv), exist_ok=True)
        with open(out_csv, "w", newline="") as f:
            w = csvmod.DictWriter(f, fieldnames=FIELDS)
            w.writeheader()
            w.writerows(rows)
    return rows


def _agg(rows, regime, arm, key, dataset=None):
    vals = [r[key] for r in rows if r["regime"] == regime and r["arm"] == arm
            and (dataset is None or r.get("dataset") == dataset)]
    return float(np.mean(vals)), float(np.std(vals))


def print_table(rows, dataset=None):
    title = dataset if dataset else "PUBLIC MNIST"
    print(f"\n=== OPERATOR-MATCH ON {title.upper()} "
          "(mean +/- std over seeds) ===")
    header = ("| Regime        | Arm         | matched | bit_F1 (mean+/-std) "
              "| mAP (mean+/-std)   | FAR    | nFAR   |")
    dash = ("|---------------|-------------|---------|--------------------"
            "|--------------------|--------|--------|")
    print(header)
    print(dash)
    for regime in REGIMES:
        for arm in ARMS:
            bm, bs_ = _agg(rows, regime, arm, "bit_f1", dataset)
            mm, ms = _agg(rows, regime, arm, "mAP", dataset)
            fm, _ = _agg(rows, regime, arm, "far", dataset)
            nm, _ = _agg(rows, regime, arm, "normal_far", dataset)
            mk = "yes" if arm == MATCHED[regime] else ""
            print(f"| {regime:13s} | {arm:11s} | {mk:7s} "
                  f"| {bm:.4f} +/- {bs_:.4f} | {mm:.4f} +/- {ms:.4f} "
                  f"| {fm:.4f} | {nm:.4f} |")


def flip_holds(rows, dataset=None):
    """Flip verdict: in BOTH regimes the matched operator beats the mismatched
    with no mean+/-std overlap AND recovers the oracle (matched >= oracle-tol).

    Returns (flip_ok, details dict per regime).
    """
    tol = 0.02
    details = {}
    flip_ok = True
    for regime in REGIMES:
        matched = MATCHED[regime]
        other = "overlay" if matched == "partition" else "partition"
        bm, bmstd = _agg(rows, regime, matched, "bit_f1", dataset)
        bo, bostd = _agg(rows, regime, other, "bit_f1", dataset)
        boc, _ = _agg(rows, regime, "oracle", "bit_f1", dataset)
        no_overlap = (bm - bmstd) > (bo + bostd)
        recovers_oracle = bm >= (boc - tol)
        win = (bm > bo) and no_overlap and recovers_oracle
        flip_ok = flip_ok and win
        details[regime] = dict(matched=matched, other=other, bm=bm, bmstd=bmstd,
                               bo=bo, bostd=bostd, oracle=boc,
                               no_overlap=no_overlap,
                               recovers_oracle=recovers_oracle, win=win)
    return flip_ok, details


def print_verdict(rows, dataset=None):
    print("\n=== VERDICT ===")
    flip_ok, details = flip_holds(rows, dataset)
    for regime in REGIMES:
        d = details[regime]
        mm, _ = _agg(rows, regime, d["matched"], "mAP", dataset)
        mo, _ = _agg(rows, regime, d["other"], "mAP", dataset)
        print(f"[{regime}] matched={d['matched']} bit_F1={d['bm']:.4f}+/-"
              f"{d['bmstd']:.4f} vs {d['other']} bit_F1={d['bo']:.4f}+/-"
              f"{d['bostd']:.4f} | oracle={d['oracle']:.4f} | mAP {mm:.4f} vs "
              f"{mo:.4f} -> no_overlap={d['no_overlap']} "
              f"recovers_oracle={d['recovers_oracle']} "
              f"-> matched {'WINS' if d['win'] else 'does NOT win'}")
    print(f"\nFLIP {'APPEARS' if flip_ok else 'does NOT cleanly appear'}: "
          f"matched operator best (no std overlap + recovers oracle) in each "
          f"regime = {flip_ok}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arms", nargs="+", default=ARMS)
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    ap.add_argument("--per-class-single", type=int, default=200)
    ap.add_argument("--n-train", type=int, default=3000)
    ap.add_argument("--n-test", type=int, default=2000)
    ap.add_argument("--n-normal", type=int, default=1000)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--bs", type=int, default=64)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--grid", type=int, default=2)
    ap.add_argument("--cell", type=int, default=28)
    ap.add_argument("--mnist-root", default="E:/data/torchvision")
    ap.add_argument("--out-csv",
                    default="outputs/multilabel_synth/operator_match_mnist.csv")
    args = ap.parse_args()

    tr_imgs, tr_labels = load_mnist(args.mnist_root, train=True)
    te_imgs, te_labels = load_mnist(args.mnist_root, train=False)
    rows = run(tr_imgs, tr_labels, te_imgs, te_labels, args.arms, args.seeds,
               args.per_class_single, args.n_train, args.n_test, args.n_normal,
               args.epochs, args.bs, args.device, args.grid, args.cell,
               args.out_csv, n_classes=10, dataset="mnist")
    print_table(rows)
    print_verdict(rows)
    print(f"\n[OUT] {os.path.abspath(args.out_csv)}")


if __name__ == "__main__":
    main()
