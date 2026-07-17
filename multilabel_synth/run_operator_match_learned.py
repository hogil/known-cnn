"""Learnable operator-match: can the combination OPERATOR be LEARNED per domain
instead of hand-selected?

The operator-match control (run_operator_match.py) hand-picks one of two KNOWN
operators per domain (overlay for superposition, partition for the partition
regime). A reviewer's objection: that is "a criterion for choosing between two
known operators, no algorithmic novelty". This script tests whether the operator
can instead be LEARNED from a small real multi-label signal WITHOUT the domain
ever being labeled partition/superposition -- converting "choose the operator"
into "learn the operator".

The synthesis combine of two single-source tiles is parameterized as a learnable
convex blend of the two candidate primitives (multimnist_blend.build_blend_*):

    x(g) = g * overlay(a, b) + (1 - g) * partition(a, b)          g in [0, 1]

g = 1 -> pure superposition operator ; g = 0 -> pure partition operator. Two
estimators of the learned g are run, neither told the domain:

  (1) GRADIENT (soft-gate, Option A): g = sigmoid(rho), rho a single learnable
      scalar. One-step-lookahead hypergradient: a virtual SGD step trains the
      classifier on x(g) synthetic combos, and rho is moved to reduce the loss of
      that virtually-updated classifier on a held-out REAL multi-label batch.
      End-to-end; g is genuinely learned, initialized neutral (g=0.5).
  (2) PROFILE (val-selected continuous g): train a fresh classifier at each g on
      a grid, pick the g maximizing real held-out multi-label bit_F1. The whole
      landscape is recorded (does it peak at the matched endpoint?).

Learning uses only a small REAL multi-label pool synthesized from the TRAIN split
under the domain's true law (the "real signal available in the domain"); final
comparison uses the TEST split, so no instance leaks. QUESTION answered per
regime: does learned g -> 1 on superposition and -> 0 on partition (recover the
operator-match prediction), and does the learned-operator arm match/beat the
hand-selected matched arm vs the mismatched arm?
"""
import os
import csv as csvmod
import argparse

import numpy as np
import torch
import torch.nn as nn
from torch.func import functional_call

from .datasets.multimnist import load_torchvision, all_pairs
from .datasets.multimnist_operator import build_multi
from .datasets.multimnist_blend import build_blend_pairs, build_singles
from .train import train_model
from .metrics import bit_f1, far, compute_map, pos_neg_prob
from .run_operator_match import build_train_for_arm, MATCHED, MISMATCHED, REGIMES

FIELDS = ["dataset", "regime", "arm", "matched_op", "seed", "g", "split",
          "bit_f1", "mAP", "far", "pos_prob", "neg_prob"]

DEFAULT_DATASETS = [
    ("mnist",        "MNIST",        None),
    ("fashionmnist", "FashionMNIST", None),
]


def _predict(model, X, bs, device):
    from torch.utils.data import TensorDataset, DataLoader
    model.eval()
    P = []
    with torch.no_grad():
        for (xb,) in DataLoader(TensorDataset(torch.from_numpy(X)),
                                batch_size=bs):
            P.append(torch.sigmoid(model(xb.to(device))).cpu().numpy())
    return np.concatenate(P)


def learn_g_gradient(O, Q, Y, valX, valY, K, device, seed,
                     meta_steps=400, bs=64, val_bs=128, inner_lr=0.1,
                     rho_lr=0.05, clf_lr=1e-3, in_ch=1, rho_init=0.0):
    """Learn the scalar gate g=sigmoid(rho) end-to-end by one-step-lookahead
    hypergradient. Returns (converged_g, trajectory).

    converged_g = mean of the last 20% of the g trajectory (noise-robust).
    """
    from .models.small_cnn import SmallCNN
    torch.manual_seed(seed)
    Ot = torch.from_numpy(O).to(device)
    Qt = torch.from_numpy(Q).to(device)
    Yt = torch.from_numpy(Y).to(device)
    vX = torch.from_numpy(valX).to(device)
    vY = torch.from_numpy(valY).to(device)
    n, nval = Ot.shape[0], vX.shape[0]
    clf = SmallCNN(num_classes=K, in_ch=in_ch).to(device)
    rho = torch.tensor(float(rho_init), device=device, requires_grad=True)
    rho_opt = torch.optim.Adam([rho], lr=rho_lr)
    clf_opt = torch.optim.Adam(clf.parameters(), lr=clf_lr)
    lossf = nn.BCEWithLogitsLoss()
    names = [nm for nm, _ in clf.named_parameters()]
    rng = np.random.default_rng(seed + 12345)
    traj = []
    for step in range(meta_steps):
        idx = rng.integers(0, n, bs)
        Ob, Qb, yb = Ot[idx], Qt[idx], Yt[idx]
        g = torch.sigmoid(rho)
        xb = g * Ob + (1 - g) * Qb
        # inner: one virtual SGD step on the classifier (keeps graph in g)
        inner_loss = lossf(clf(xb), yb)
        grads = torch.autograd.grad(inner_loss, clf.parameters(),
                                    create_graph=True)
        fast = {nm: p - inner_lr * gr
                for nm, p, gr in zip(names, clf.parameters(), grads)}
        # outer: real held-out multi-label loss of the virtually-updated net
        vidx = rng.integers(0, nval, min(val_bs, nval))
        vlogits = functional_call(clf, fast, (vX[vidx],))
        val_loss = lossf(vlogits, vY[vidx])
        rho_opt.zero_grad()
        clf_opt.zero_grad()
        val_loss.backward()                      # grad flows to rho
        rho_opt.step()
        # real classifier update at the current (detached) g
        g_d = torch.sigmoid(rho).detach()
        xb2 = g_d * Ob + (1 - g_d) * Qb
        clf_opt.zero_grad()
        lossf(clf(xb2), yb).backward()
        clf_opt.step()
        traj.append(float(torch.sigmoid(rho).item()))
    tail = max(1, len(traj) // 5)
    return float(np.mean(traj[-tail:])), traj


def _blend_train_data(imgs, labels, pairs, per_class_single, n_train, seed,
                      grid, cell, n_classes, g):
    """Learned-arm training data: combos synthesized at gate g + single pool.
    Mirrors run_operator_match.build_train_for_arm (combos + singles)."""
    O, Q, Yc = build_blend_pairs(imgs, labels, n_train, seed, pairs,
                                 grid=grid, cell=cell, n_classes=n_classes)
    cX = g * O + (1 - g) * Q
    spX, spY = build_singles(imgs, labels, per_class_single, seed,
                             grid=grid, cell=cell, n_classes=n_classes)
    return (np.concatenate([cX, spX]).astype(np.float32),
            np.concatenate([Yc, spY]).astype(np.float32))


def _eval_row(model, teX, teY, bs, device):
    P = _predict(model, teX, bs, device)
    pos, neg = pos_neg_prob(P, teY)
    return dict(bit_f1=round(float(bit_f1(P, teY)), 4),
                mAP=round(float(compute_map(P, teY)), 4),
                far=round(float(far(P, teY)), 4),
                pos_prob=round(float(pos), 4), neg_prob=round(float(neg), 4))


def run_dataset(label, tr_imgs, tr_labels, te_imgs, te_labels, n_classes,
                seeds, per_class_single, n_train, n_test, n_val,
                epochs, bs, device, grid, cell, g_grid, meta_steps):
    pairs = all_pairs(n_classes)
    rows = []
    for regime in REGIMES:
        law = regime
        matched_op = MATCHED[regime]      # 'overlay' (g=1) or 'partition' (g=0)
        mism_op = MISMATCHED[regime]
        g_matched = 1.0 if matched_op == "overlay" else 0.0
        # real held-out multi-label signal (TRAIN split, true law) for LEARNING
        valX, valY = build_multi(tr_imgs, tr_labels, n_val, seed=777,
                                 allowed_pairs=pairs, law=law,
                                 grid=grid, cell=cell, n_classes=n_classes)
        # final comparison test set (TEST split, true law) -- no instance leak
        teX, teY = build_multi(te_imgs, te_labels, n_test, seed=999,
                               allowed_pairs=pairs, law=law,
                               grid=grid, cell=cell, n_classes=n_classes)
        for seed in seeds:
            # (1) GRADIENT-learned g (soft-gate), init neutral g=0.5
            O, Q, Yc = build_blend_pairs(tr_imgs, tr_labels, n_train, seed,
                                         pairs, grid=grid, cell=cell,
                                         n_classes=n_classes)
            spX, spY = build_singles(tr_imgs, tr_labels, per_class_single, seed,
                                     grid=grid, cell=cell, n_classes=n_classes)
            Opool = np.concatenate([O, spX]).astype(np.float32)
            Qpool = np.concatenate([Q, spX]).astype(np.float32)
            Ypool = np.concatenate([Yc, spY]).astype(np.float32)
            g_grad, traj = learn_g_gradient(Opool, Qpool, Ypool, valX, valY,
                                            n_classes, device, seed,
                                            meta_steps=meta_steps, bs=bs)

            # (2) PROFILE: val bit_F1 across the g grid, argmax = profile g
            prof = {}
            for gv in g_grid:
                trX, trY = _blend_train_data(tr_imgs, tr_labels, pairs,
                                             per_class_single, n_train, seed,
                                             grid, cell, n_classes, gv)
                m = train_model(trX, trY, epochs=epochs, bs=bs,
                                device=device, seed=seed)
                r = _eval_row(m, valX, valY, bs, device)
                prof[gv] = r["bit_f1"]
                rows.append(dict(dataset=label, regime=regime, arm="profile_scan",
                                 matched_op=matched_op, seed=seed, g=round(gv, 4),
                                 split="val", **r))
            g_prof = max(prof, key=prof.get)

            print(f"[{label}][{regime:13s}] s{seed} "
                  f"g_grad={g_grad:.3f} (init0.5) g_prof={g_prof:.2f} "
                  f"matched={matched_op}(g={g_matched:.0f}) "
                  f"| profile bitF1 " +
                  " ".join(f"g{gv:.2f}={prof[gv]:.3f}" for gv in g_grid),
                  flush=True)

            # (3) FINAL EVAL on TEST: learned(grad), learned(profile),
            #     hand-matched, hand-mismatched -- identical recipe
            def _fit_eval(trX, trY):
                m = train_model(trX, trY, epochs=epochs, bs=bs,
                                device=device, seed=seed)
                return _eval_row(m, teX, teY, bs, device)

            lg = _blend_train_data(tr_imgs, tr_labels, pairs, per_class_single,
                                   n_train, seed, grid, cell, n_classes, g_grad)
            lp = _blend_train_data(tr_imgs, tr_labels, pairs, per_class_single,
                                   n_train, seed, grid, cell, n_classes, g_prof)
            mm = build_train_for_arm(matched_op, regime, tr_imgs, tr_labels,
                                     pairs, per_class_single, n_train, seed,
                                     grid, cell, n_classes)
            xm = build_train_for_arm(mism_op, regime, tr_imgs, tr_labels, pairs,
                                     per_class_single, n_train, seed, grid, cell,
                                     n_classes)
            arm_data = [
                ("learned_grad", g_grad, _fit_eval(*lg)),
                ("learned_profile", g_prof, _fit_eval(*lp)),
                ("matched", g_matched, _fit_eval(*mm)),
                ("mismatched", 1.0 - g_matched, _fit_eval(*xm)),
            ]
            for arm, gval, r in arm_data:
                rows.append(dict(dataset=label, regime=regime, arm=arm,
                                 matched_op=matched_op, seed=seed,
                                 g=round(float(gval), 4), split="test", **r))
                print(f"    {arm:16s} g={gval:.3f} "
                      f"bitF1={r['bit_f1']:.4f} mAP={r['mAP']:.4f} "
                      f"FAR={r['far']:.4f} pos={r['pos_prob']:.4f} "
                      f"neg={r['neg_prob']:.4f}", flush=True)
    return rows


def _agg(rows, dataset, regime, arm, key, split="test"):
    vals = [r[key] for r in rows if r["dataset"] == dataset
            and r["regime"] == regime and r["arm"] == arm and r["split"] == split]
    return (float(np.mean(vals)), float(np.std(vals))) if vals else (float("nan"),
                                                                     float("nan"))


def _agg_g(rows, dataset, regime, arm):
    vals = [r["g"] for r in rows if r["dataset"] == dataset
            and r["regime"] == regime and r["arm"] == arm and r["split"] == "test"]
    return (float(np.mean(vals)), float(np.std(vals))) if vals else (float("nan"),
                                                                     float("nan"))


def print_learned_g(rows, datasets):
    print("\n=== LEARNED OPERATOR g PER REGIME (convention: g=1 overlay/"
          "superposition, g=0 partition; matched g in parens) ===")
    head = ("| Dataset         | Regime        | matched op  | matched g "
            "| g_grad (mean+/-std) | g_profile (mean+/-std) | recovers? |")
    dash = ("|-----------------|---------------|-------------|-----------"
            "|---------------------|------------------------|-----------|")
    print(head)
    print(dash)
    for label in datasets:
        for regime in REGIMES:
            matched_op = MATCHED[regime]
            g_m = 1.0 if matched_op == "overlay" else 0.0
            gg, ggs = _agg_g(rows, label, regime, "learned_grad")
            gp, gps = _agg_g(rows, label, regime, "learned_profile")
            # "recovers" = both learned estimators land on the matched side (<0.5
            # when matched g=0, >0.5 when matched g=1)
            if g_m == 1.0:
                rec = (gg > 0.5) and (gp > 0.5)
            else:
                rec = (gg < 0.5) and (gp < 0.5)
            print(f"| {label:15s} | {regime:13s} | {matched_op:11s} "
                  f"| {g_m:9.0f} | {gg:.3f} +/- {ggs:.3f}     "
                  f"| {gp:.3f} +/- {gps:.3f}        | {'yes' if rec else 'NO':9s} |")


def print_eval_table(rows, datasets):
    print("\n=== EVAL bit_F1 (TEST, mean+/-std over seeds): "
          "learned vs hand-matched vs mismatched ===")
    head = ("| Dataset         | Regime        | learned_grad    "
            "| learned_profile | matched (hand)  | mismatched      |")
    dash = ("|-----------------|---------------|-----------------"
            "|-----------------|-----------------|-----------------|")
    print(head)
    print(dash)
    for label in datasets:
        for regime in REGIMES:
            cells = []
            for arm in ("learned_grad", "learned_profile", "matched",
                        "mismatched"):
                m, s = _agg(rows, label, regime, arm, "bit_f1")
                cells.append(f"{m:.4f}+/-{s:.4f}")
            print(f"| {label:15s} | {regime:13s} | {cells[0]:15s} "
                  f"| {cells[1]:15s} | {cells[2]:15s} | {cells[3]:15s} |")


def print_verdict(rows, datasets, tol=0.02):
    print("\n=== VERDICT ===")
    n_recover = 0
    n_regimes = 0
    learned_ge_matched = True
    learned_gt_matched = False
    for label in datasets:
        for regime in REGIMES:
            n_regimes += 1
            matched_op = MATCHED[regime]
            g_m = 1.0 if matched_op == "overlay" else 0.0
            gg, _ = _agg_g(rows, label, regime, "learned_grad")
            gp, _ = _agg_g(rows, label, regime, "learned_profile")
            rec = ((gg > 0.5) and (gp > 0.5)) if g_m == 1.0 \
                else ((gg < 0.5) and (gp < 0.5))
            n_recover += int(rec)
            lg, _ = _agg(rows, label, regime, "learned_grad", "bit_f1")
            lp, _ = _agg(rows, label, regime, "learned_profile", "bit_f1")
            mt, _ = _agg(rows, label, regime, "matched", "bit_f1")
            best_learned = max(lg, lp)
            if best_learned < mt - tol:
                learned_ge_matched = False
            if best_learned > mt + tol:
                learned_gt_matched = True
            print(f"[{label}/{regime}] matched={matched_op}(g={g_m:.0f}) "
                  f"| g_grad={gg:.3f} g_prof={gp:.3f} -> recovers={rec} "
                  f"| bitF1 learned(grad/prof)={lg:.3f}/{lp:.3f} "
                  f"vs matched={mt:.3f}")
    print(f"\nRecovered the matched operator (both estimators land on matched "
          f"side) in {n_recover}/{n_regimes} regime-datasets.")
    if n_recover == n_regimes:
        if learned_gt_matched and learned_ge_matched:
            verdict = ("STRONG: learned operator recovers the matched primitive "
                       "per domain AND matches/beats hand-selection")
        elif learned_ge_matched:
            verdict = ("MODERATE: learned operator recovers the operator-match "
                       "prediction (criterion is learnable) but no gain over "
                       "hand-selection")
        else:
            verdict = ("MODERATE-: learned g recovers the matched side but the "
                       "learned-operator arm underperforms hand-selection")
    elif n_recover >= 1:
        verdict = ("WEAK: learned operator recovers the prediction only in some "
                   "regimes -- finicky")
    else:
        verdict = ("NEGATIVE: learned operator does not recover the matched "
                   "primitive")
    print(f"VERDICT: {verdict}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+", default=None)
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    ap.add_argument("--per-class-single", type=int, default=200)
    ap.add_argument("--n-train", type=int, default=3000)
    ap.add_argument("--n-test", type=int, default=2000)
    ap.add_argument("--n-val", type=int, default=1000)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--meta-steps", type=int, default=400)
    ap.add_argument("--bs", type=int, default=64)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--grid", type=int, default=2)
    ap.add_argument("--cell", type=int, default=28)
    ap.add_argument("--g-grid", type=float, nargs="+",
                    default=[0.0, 0.25, 0.5, 0.75, 1.0])
    ap.add_argument("--root", default="E:/data/torchvision")
    ap.add_argument("--out-csv",
                    default="outputs/multilabel_synth/operator_match_learned.csv")
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()

    if args.smoke:
        args.seeds = [0]
        args.per_class_single = 40
        args.n_train = 400
        args.n_test = 300
        args.n_val = 300
        args.epochs = 4
        args.meta_steps = 40
        args.datasets = ["mnist"]

    datasets = DEFAULT_DATASETS
    if args.datasets:
        want = set(args.datasets)
        datasets = [d for d in DEFAULT_DATASETS if d[0] in want]

    device = args.device if torch.cuda.is_available() else "cpu"
    all_rows, ran = [], []
    for label, tvname, split in datasets:
        try:
            tr_imgs, tr_labels, n_classes = load_torchvision(
                tvname, args.root, train=True, split=split)
            te_imgs, te_labels, _ = load_torchvision(
                tvname, args.root, train=False, split=split)
        except Exception as e:
            print(f"[SKIP] {label} not available: {repr(e)[:200]}", flush=True)
            continue
        print(f"\n########## {label} ({tvname}) n_classes={n_classes} "
              f"train={len(tr_labels)} test={len(te_labels)} device={device} "
              f"##########", flush=True)
        rows = run_dataset(label, tr_imgs, tr_labels, te_imgs, te_labels,
                           n_classes, args.seeds, args.per_class_single,
                           args.n_train, args.n_test, args.n_val, args.epochs,
                           args.bs, device, args.grid, args.cell, args.g_grid,
                           args.meta_steps)
        all_rows.extend(rows)
        ran.append(label)

    if all_rows:
        os.makedirs(os.path.dirname(args.out_csv), exist_ok=True)
        with open(args.out_csv, "w", newline="") as f:
            w = csvmod.DictWriter(f, fieldnames=FIELDS)
            w.writeheader()
            w.writerows(all_rows)

    print_learned_g(all_rows, ran)
    print_eval_table(all_rows, ran)
    print_verdict(all_rows, ran)
    print(f"\n[OUT] {os.path.abspath(args.out_csv)}")


if __name__ == "__main__":
    main()
