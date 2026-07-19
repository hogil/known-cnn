"""Can UNLABELED real multi-label tiles close the provably-irreducible appearance
floor A*(S) that content-blind synthesis cannot? (feasibility probe)

THEORY CONTEXT (docs/mlsynth_paper/THEORY_LOWERBOUND_260718.md): single-label
sources cannot identify the co-occurrence-copula-dependent appearance -- Thm L1
floors the worst-case conditional TV at A*(S) = half the copula-diameter surviving
the operator; on land cover this shows up as a ~0.10 mAP residual between the best
content-blind synthesis (partition_realistic, ~0.832 mAP) and the real-multi oracle
(~0.94). Labeled few-shot (synth + K labeled real multi) needs O(hundreds) of real
tiles to close it (outputs/multilabel_synth/landcover_fewshot_ksweep.csv).

But UNLABELED real multi-label images REVEAL the real appearance distribution
oplus_# Q_real WITHOUT needing labels. This probe asks: does a domain-alignment
loss that pulls the classifier's features on SYNTHETIC COMBOS toward the feature
distribution of UNLABELED real multi-label tiles close the gap -- and more
efficiently (per unit of extra real data) than the LABELED few-shot?

ARM: unlabeled_align
  * classification supervision: ONLY the synthetic labels of the partition_realistic
    base (2000 synth multi + real singles), unchanged. This is the K=0 content-blind
    training set with SYNTHETIC labels.
  * additionally: U REAL multi-label tiles from the oracle TRAIN pool with LABELS
    WITHHELD (images only). They contribute NO labels -- only a Deep CORAL
    alignment loss (feature 1st+2nd-moment matching) that pulls synthetic-combo
    features toward the unlabeled-real-multi feature distribution.
  * NO EVAL LEAK: the U tiles are the first-U of the SAME nested permutation the
    few-shot sweep uses (rng 1000+seed over the oracle pool, disjoint from eval).
    -> unlabeled_align @ U uses EXACTLY the same real tiles as labeled few-shot @ K=U,
    so "same amount of extra real data, labels vs no labels" is apples-to-apples.

Alignment = Deep CORAL (Sun & Saenko ECCV'16) on the resnet18 penultimate 512-d
features: match mean + covariance of source (synthetic combos) and target
(unlabeled real multi). Chosen over MMD (no bandwidth to tune) and adversarial
(no extra unstable head). The unlabeled target also passes through BatchNorm in
train mode (an AdaBN effect) -- a legitimate part of "using the unlabeled images".

REFERENCE POINTS (same held-out real multi eval, resnet18, 3 seeds):
  content-blind floor  = synth_plus_k @ K=0  (few-shot CSV) ~0.832
  oracle               = real_only_k @ K=all (few-shot CSV) ~0.94
  labeled few-shot     = synth_plus_k @ K=U  (few-shot CSV) -- matched-amount rival

HONESTY / early-kill: if unlabeled alignment does NOT beat the content-blind floor,
that is reported plainly -- it bounds the (b) direction (appearance needs LABELED
co-occurrence). No tuning to force a win: coral_lambda is fixed by init loss-
magnitude balance (Sec. calibrate), NOT by eval mAP, and held constant across the
whole sweep.
"""
import os
import csv as csvmod
import argparse

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

from .datasets.dlrsd import prepare_landcover, harvest_mask_templates, DEFAULT_SUBSET
from .synthesis import landcover_ops as ops
from .models.resnet import build_resnet18
from .metrics import bit_f1, compute_map, far, pos_neg_prob
from .run_operator_match_landcover import _norm, _predict
from .run_fewshot_ksweep_landcover import build_base_synth

DEFAULT_US = [0, 50, 100, 250, 500, 1000]
# Fixed by init loss-magnitude balance (calibrate_lambda / --calibrate), NOT by
# eval mAP: measured init BCE=0.743 vs raw CORAL=0.0101 -> lambda~73 for 1x
# balance at start. Rounded to 70 and held constant across all U and seeds
# (pre-registered before reading the U-sweep). The --calibrate path documents it.
DEFAULT_LAMBDA = 70.0
# lambda-robustness grid (U=all, seed 0) written to the CSV so the negative is not
# an artifact of one over-strong weight: shows whether ANY weight helps.
DEFAULT_LAMBDA_GRID = [0.0, 10.0, 30.0, 70.0, 150.0]

FIELDS = ["dataset", "arm", "U", "backbone", "seed", "n_labeled", "n_unlabeled",
          "coral_lambda", "align_method", "eval_map", "eval_bit_f1", "eval_far",
          "eval_pos_prob", "eval_neg_prob"]


def forward_feats(model, x):
    """resnet18 forward returning (penultimate 512-d feats, logits)."""
    m = model
    x = m.conv1(x); x = m.bn1(x); x = m.relu(x); x = m.maxpool(x)
    x = m.layer1(x); x = m.layer2(x); x = m.layer3(x); x = m.layer4(x)
    x = m.avgpool(x)
    feats = torch.flatten(x, 1)
    logits = m.fc(feats)
    return feats, logits


def coral_loss(fs, ft):
    """Deep CORAL: match 1st (mean) + 2nd (covariance) feature moments of source
    (synthetic combos) and target (unlabeled real multi). fs,ft: [n, d]."""
    d = fs.size(1)
    ms = fs.mean(0, keepdim=True)
    mt = ft.mean(0, keepdim=True)
    mean_term = ((ms - mt) ** 2).sum() / d
    fsc = fs - ms
    ftc = ft - mt
    cov_s = (fsc.t() @ fsc) / max(fs.size(0) - 1, 1)
    cov_t = (ftc.t() @ ftc) / max(ft.size(0) - 1, 1)
    cov_term = ((cov_s - cov_t) ** 2).sum() / (4.0 * d * d)
    return cov_term + mean_term


def train_align(baseX, baseY, srcX, tgtX, epochs, bs, lr, device, seed, n_classes,
                pretrained, coral_lambda):
    """Train resnet18 with BCE on the synthetic-labeled base + (if tgtX given) a
    Deep CORAL loss pulling synthetic-combo (srcX) features toward unlabeled real
    (tgtX) features. tgtX contributes NO labels."""
    torch.manual_seed(seed)
    model = build_resnet18(n_classes, pretrained=pretrained).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    bce = nn.BCEWithLogitsLoss()
    base_ds = TensorDataset(torch.from_numpy(_norm(baseX)), torch.from_numpy(baseY))
    loader = DataLoader(base_ds, batch_size=bs, shuffle=True)
    use_align = (tgtX is not None and coral_lambda > 0 and tgtX.shape[0] > 0)
    if use_align:
        src_t = torch.from_numpy(_norm(srcX))
        tgt_t = torch.from_numpy(_norm(tgtX))
        g = torch.Generator().manual_seed(10000 + seed)
    for _ in range(epochs):
        model.train()
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            _, logits = forward_feats(model, xb)
            loss = bce(logits, yb)
            if use_align:
                nb = xb.size(0)
                si = torch.randint(0, src_t.size(0), (nb,), generator=g)
                nt = min(nb, tgt_t.size(0))
                ti = torch.randint(0, tgt_t.size(0), (nt,), generator=g)
                fs, _ = forward_feats(model, src_t[si].to(device))
                ft, _ = forward_feats(model, tgt_t[ti].to(device))
                loss = loss + coral_lambda * coral_loss(fs, ft)
            loss.backward()
            opt.step()
    return model


def calibrate_lambda(baseX, baseY, srcX, tgtX, bs, device, n_classes, target=1.0):
    """Report init BCE vs raw CORAL magnitude so lambda can be fixed by balance,
    NOT by eval mAP. Returns (bce0, coral0, suggested_lambda)."""
    torch.manual_seed(0)
    model = build_resnet18(n_classes, pretrained=True).to(device)
    model.train()
    bce = nn.BCEWithLogitsLoss()
    xb = torch.from_numpy(_norm(baseX[:bs])).to(device)
    yb = torch.from_numpy(baseY[:bs]).to(device)
    with torch.no_grad():
        _, logits = forward_feats(model, xb)
        b0 = float(bce(logits, yb))
        fs, _ = forward_feats(model, torch.from_numpy(_norm(srcX[:bs])).to(device))
        ft, _ = forward_feats(model, torch.from_numpy(_norm(tgtX[:bs])).to(device))
        c0 = float(coral_loss(fs, ft))
    lam = target * b0 / c0 if c0 > 0 else float("nan")
    print(f"[calibrate] init BCE={b0:.4f} rawCORAL={c0:.4f} -> "
          f"lambda for {target}x balance = {lam:.3f}", flush=True)
    return b0, c0, lam


def run_lambda_probe(baseX, baseY, srcX, tgtX_all, lambda_grid, epochs, bs, lr,
                     device, seed, n_classes, evX, eval_Y, U_all, dataset):
    """lambda-robustness at U=all, one seed: does ANY alignment weight beat the
    no-align floor? Rows tagged arm='lambda_probe' (coral_lambda distinguishes)."""
    rows = []
    for lam in lambda_grid:
        model = train_align(baseX, baseY, srcX, tgtX_all, epochs, bs, lr, device,
                            seed, n_classes, True, lam)
        P = _predict(model, evX, bs, device)
        pos, neg = pos_neg_prob(P, eval_Y)
        r = {
            "dataset": dataset, "arm": "lambda_probe", "U": int(U_all),
            "backbone": "resnet18", "seed": int(seed),
            "n_labeled": int(baseX.shape[0]), "n_unlabeled": int(U_all),
            "coral_lambda": float(lam), "align_method": "deep_coral_mean_cov",
            "eval_map": round(float(compute_map(P, eval_Y)), 4),
            "eval_bit_f1": round(float(bit_f1(P, eval_Y)), 4),
            "eval_far": round(float(far(P, eval_Y)), 4),
            "eval_pos_prob": round(float(pos), 4),
            "eval_neg_prob": round(float(neg), 4),
        }
        rows.append(r)
        print(f"[lam-probe] lam={lam:6.1f} s{seed} U={U_all} "
              f"mAP={r['eval_map']:.4f} bitF1={r['eval_bit_f1']:.4f} "
              f"FAR={r['eval_far']:.4f} neg={r['eval_neg_prob']:.4f}", flush=True)
    return rows


def run(data, templates, seeds, Us, per_class_single, n_multi, epochs, bs, lr,
        device, cell, grid, min_frac, feather_sigma, coral_lambda, out_csv,
        dataset="landcover", lambda_grid=None):
    (pool_imgs, pool_labels, oracle_imgs, oracle_Y, eval_imgs, eval_Y,
     names, meta) = data
    n_classes = len(names)
    canvas = cell * grid
    evX = ops.real_to_chw(eval_imgs)
    oX_all = ops.real_to_chw(oracle_imgs)          # real multi-label TRAIN pool
    No = oX_all.shape[0]
    Us = [u for u in Us if u <= No]
    rows = []
    probe_base = None
    for seed in seeds:
        baseX, baseY, spX, spY = build_base_synth(
            pool_imgs, pool_labels, templates, per_class_single, n_multi, seed,
            cell, grid, canvas, n_classes, min_frac, feather_sigma)
        n_singles = spX.shape[0]
        srcX = baseX[:baseX.shape[0] - n_singles]  # synthetic combos only (source)
        perm = np.random.default_rng(1000 + seed).permutation(No)  # SAME as few-shot
        if seed == seeds[0]:
            probe_base = (baseX, baseY, srcX, oX_all[perm[:No]], seed)
        print(f"[seed {seed}] base_synth n={baseX.shape[0]} "
              f"(combos={srcX.shape[0]}+singles={n_singles}), oracle_pool={No}",
              flush=True)
        for U in Us:
            tgtX = None if U == 0 else oX_all[perm[:U]]
            lam = coral_lambda if U > 0 else 0.0
            model = train_align(baseX, baseY, srcX, tgtX, epochs, bs, lr, device,
                                seed, n_classes, True, lam)
            P = _predict(model, evX, bs, device)
            pos, neg = pos_neg_prob(P, eval_Y)
            r = {
                "dataset": dataset, "arm": "unlabeled_align", "U": int(U),
                "backbone": "resnet18", "seed": int(seed),
                "n_labeled": int(baseX.shape[0]), "n_unlabeled": int(U),
                "coral_lambda": float(lam), "align_method": "deep_coral_mean_cov",
                "eval_map": round(float(compute_map(P, eval_Y)), 4),
                "eval_bit_f1": round(float(bit_f1(P, eval_Y)), 4),
                "eval_far": round(float(far(P, eval_Y)), 4),
                "eval_pos_prob": round(float(pos), 4),
                "eval_neg_prob": round(float(neg), 4),
            }
            rows.append(r)
            tag = " (content-blind base, no align)" if U == 0 else ""
            print(f"[unlab-align] U={U:4d} s{seed} n_lab={r['n_labeled']} "
                  f"n_unlab={U} lam={lam:.2f} mAP={r['eval_map']:.4f} "
                  f"bitF1={r['eval_bit_f1']:.4f} FAR={r['eval_far']:.4f} "
                  f"pos={r['eval_pos_prob']:.4f} neg={r['eval_neg_prob']:.4f}{tag}",
                  flush=True)
    if lambda_grid and probe_base is not None:
        bX, bY, sX, tX, pseed = probe_base
        rows += run_lambda_probe(bX, bY, sX, tX, lambda_grid, epochs, bs, lr,
                                 device, pseed, n_classes, evX, eval_Y, No, dataset)
    if out_csv:
        os.makedirs(os.path.dirname(out_csv), exist_ok=True)
        with open(out_csv, "w", newline="") as f:
            w = csvmod.DictWriter(f, fieldnames=FIELDS)
            w.writeheader(); w.writerows(rows)
    return rows, No


def _agg(rows, arm, key, U=None, Kfield="U"):
    vals = [r[key] for r in rows
            if r["arm"] == arm and (U is None or r[Kfield] == U)]
    return (float(np.mean(vals)), float(np.std(vals))) if vals else (float("nan"),
                                                                     float("nan"))


def _load_fewshot(path):
    """Load labeled few-shot CSV -> dict rows for comparison. Returns list of dicts
    with float casts. Empty if file missing."""
    if not os.path.isfile(path):
        return []
    out = []
    with open(path, newline="") as f:
        for r in csvmod.DictReader(f):
            d = dict(r)
            for k in ("K", "seed", "n_train"):
                if k in d:
                    d[k] = int(d[k])
            for k in ("eval_map", "eval_bit_f1", "eval_far"):
                if k in d:
                    d[k] = float(d[k])
            out.append(d)
    return out


def _fs_agg(fs_rows, arm, K, key):
    vals = [r[key] for r in fs_rows if r["arm"] == arm and r["K"] == K]
    return (float(np.mean(vals)), float(np.std(vals))) if vals else (float("nan"),
                                                                     float("nan"))


def print_report(rows, Us, No, fewshot_csv, tol=0.02):
    fs = _load_fewshot(fewshot_csv)
    Us = [u for u in Us if u <= No]
    # reference points from the labeled few-shot sweep (same eval, same recipe)
    floor_m, floor_s = _fs_agg(fs, "synth_plus_k", 0, "eval_map")      # content-blind
    or_m, or_s = _fs_agg(fs, "real_only_k", No, "eval_map")            # oracle
    # our own K=0 (should reproduce the content-blind floor)
    u0_m, u0_s = _agg(rows, "unlabeled_align", "eval_map", U=0)
    if np.isnan(floor_m):
        floor_m, floor_s = u0_m, u0_s
    gap = or_m - floor_m

    print("\n=== UNLABELED-MULTI ALIGNMENT vs LABELED FEW-SHOT  mAP(amount)  "
          "(real DLRSD/UC-Merced land cover, resnet18, mean+/-std over 3 seeds) ===")
    lam_used = next((r["coral_lambda"] for r in rows
                     if r["arm"] == "unlabeled_align" and r["U"] > 0), "NA")
    print(f"    content-blind floor (synth K=0) mAP={floor_m:.4f}+/-{floor_s:.4f}   "
          f"oracle (real multi ALL) mAP={or_m:.4f}+/-{or_s:.4f}   "
          f"appearance gap={gap:+.4f}")
    print(f"    alignment = deep_coral_mean_cov  lambda={lam_used} "
          f"(fixed by init loss balance, not eval)")
    print("| amount | unlab-align mAP    | u-align gapclosed | labeled few-shot mAP | "
          "lab gapclosed | u-align beats floor? | u-align vs labeled |")
    print("|--------|--------------------|-------------------|----------------------|"
          "---------------|----------------------|--------------------|")
    for U in Us:
        um, us = _agg(rows, "unlabeled_align", "eval_map", U=U)
        lm, ls = _fs_agg(fs, "synth_plus_k", U, "eval_map")
        u_closed = (um - floor_m) / gap if abs(gap) > 1e-9 else float("nan")
        l_closed = (lm - floor_m) / gap if abs(gap) > 1e-9 else float("nan")
        beats_floor = (um - us) > (floor_m + floor_s) if U > 0 else False
        # clean band comparison unlab-align vs labeled few-shot at matched amount
        if np.isnan(lm):
            vs = "n/a"
        elif (um - us) > (lm + ls):
            vs = "u-align WINS"
        elif (lm - ls) > (um + us):
            vs = "labeled WINS"
        else:
            vs = "tie"
        amt = f"{U}" + (" base" if U == 0 else "")
        lm_str = f"{lm:.4f} +/- {ls:.4f}" if not np.isnan(lm) else "      n/a       "
        l_cl_str = f"{l_closed*100:6.1f}%" if not np.isnan(lm) else "   n/a"
        print(f"| {amt:6s} | {um:.4f} +/- {us:.4f} | {u_closed*100:6.1f}%          | "
              f"{lm_str} | {l_cl_str}       | {str(beats_floor):5s}                | {vs:18s}|")

    # ---- lambda-robustness (U=all, seed 0): does ANY weight beat the floor? ----
    lp = [r for r in rows if r["arm"] == "lambda_probe"]
    if lp:
        print("\n=== ALIGNMENT-WEIGHT ROBUSTNESS (U=all, seed 0): "
              "does ANY coral_lambda beat the no-align floor? ===")
        u0m, _ = _agg(rows, "unlabeled_align", "eval_map", U=0)
        print(f"    no-align floor: lambda=0 row below is the seed-0 baseline; "
              f"all-seeds no-align mean mAP={u0m:.4f}")
        print("| coral_lambda | eval mAP | eval bitF1 | eval FAR | eval neg_prob |")
        print("|--------------|----------|------------|----------|---------------|")
        for r in sorted(lp, key=lambda x: x["coral_lambda"]):
            print(f"| {r['coral_lambda']:12.1f} | {r['eval_map']:.4f}   | "
                  f"{r['eval_bit_f1']:.4f}     | {r['eval_far']:.4f}   | "
                  f"{r['eval_neg_prob']:.4f}        |")
        best_lp = max(lp, key=lambda x: x["eval_map"])
        print(f"  best lambda by mAP = {best_lp['coral_lambda']} "
              f"(mAP={best_lp['eval_map']:.4f}); higher lambda -> "
              "more alignment pressure.")

    # ---- verdict ----
    print("\n=== VERDICT ===")
    # Q1: does unlabeled alignment beat the content-blind floor (clean band) at any U>0?
    best_U, best_m, best_s = None, -1.0, 0.0
    any_beats = False
    for U in Us:
        if U == 0:
            continue
        um, us = _agg(rows, "unlabeled_align", "eval_map", U=U)
        if (um - us) > (floor_m + floor_s):
            any_beats = True
        if um > best_m:
            best_U, best_m, best_s = U, um, us
    print(f"  Q1 beats content-blind floor: best unlab-align mAP={best_m:.4f}+/-{best_s:.4f} "
          f"at U={best_U} vs floor {floor_m:.4f}+/-{floor_s:.4f} "
          f"-> clean_win_over_floor={any_beats}")
    # Q2: at matched amount, unlab-align vs labeled few-shot
    q2_wins, q2_ties, q2_losses = [], [], []
    for U in Us:
        if U == 0:
            continue
        um, us = _agg(rows, "unlabeled_align", "eval_map", U=U)
        lm, ls = _fs_agg(fs, "synth_plus_k", U, "eval_map")
        if np.isnan(lm):
            continue
        if (um - us) > (lm + ls):
            q2_wins.append(U)
        elif (lm - ls) > (um + us):
            q2_losses.append(U)
        else:
            q2_ties.append(U)
    print(f"  Q2 unlab-align vs labeled few-shot (matched amount): "
          f"u-align cleanly wins @U={q2_wins}, ties @U={q2_ties}, "
          f"labeled cleanly wins @U={q2_losses}")
    # Q3: approaches oracle?
    within = best_m >= (or_m - tol)
    print(f"  Q3 approaches oracle (within {tol} mAP): best unlab-align {best_m:.4f} "
          f"vs oracle {or_m:.4f} -> {within}")

    if not any_beats:
        verdict = ("NEGATIVE: unlabeled-multi CORAL alignment does NOT cleanly beat "
                   "the content-blind floor -> the appearance floor A*(S) needs "
                   "LABELED co-occurrence; unlabeled appearance alone (feature-moment "
                   "matching) is insufficient. Honest bound on the (b) direction, "
                   "consistent with the lower-bound theory.")
    elif q2_wins and not q2_losses:
        verdict = (f"STRONG-POSITIVE: unlabeled alignment beats the floor AND beats "
                   f"labeled few-shot at matched amount @U={q2_wins} -> unlabeled "
                   f"(far cheaper) real multi closes the appearance gap more "
                   f"efficiently than labels.")
    elif any_beats and not q2_wins:
        verdict = ("PARTIAL: unlabeled alignment beats the content-blind floor but is "
                   "NOT more label-efficient than labeled few-shot at matched amount "
                   "(labels still help at least as much per tile).")
    else:
        verdict = ("MIXED: unlabeled alignment beats the floor and is competitive with "
                   "labeled few-shot at some amounts; see per-amount table.")
    print(f"\nPLAIN VERDICT: {verdict}")
    return verdict


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    ap.add_argument("--Us", nargs="+", type=int, default=DEFAULT_US)
    ap.add_argument("--subset", nargs="+", default=DEFAULT_SUBSET)
    ap.add_argument("--per-class-single", type=int, default=140)
    ap.add_argument("--n-multi", type=int, default=2000)
    ap.add_argument("--n-oracle", type=int, default=1000)
    ap.add_argument("--n-eval", type=int, default=400)
    ap.add_argument("--per-class-cap", type=int, default=140)
    ap.add_argument("--purity", type=float, default=0.6)
    ap.add_argument("--min-side", type=int, default=48)
    ap.add_argument("--cell", type=int, default=64)
    ap.add_argument("--grid", type=int, default=2)
    ap.add_argument("--feather-sigma", type=float, default=3.0)
    ap.add_argument("--min-frac", type=float, default=0.01)
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--bs", type=int, default=64)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--coral-lambda", type=float, default=DEFAULT_LAMBDA)
    ap.add_argument("--lambda-grid", nargs="+", type=float,
                    default=DEFAULT_LAMBDA_GRID,
                    help="alignment-weight robustness grid (U=all, seed 0)")
    ap.add_argument("--no-lambda-probe", action="store_true",
                    help="skip the lambda-robustness pass")
    ap.add_argument("--calibrate", action="store_true",
                    help="only report init BCE vs CORAL magnitude then exit")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--base", default="E:/data/dlrsd_extracted/DLRSD")
    ap.add_argument("--cache", default=None)
    ap.add_argument("--fewshot-csv",
                    default="outputs/multilabel_synth/landcover_fewshot_ksweep.csv")
    ap.add_argument("--out-csv",
                    default="outputs/multilabel_synth/landcover_unlabeled_align.csv")
    args = ap.parse_args()

    if args.device == "cuda" and not torch.cuda.is_available():
        print("[warn] cuda unavailable -> cpu", flush=True)
        args.device = "cpu"

    canvas = args.cell * args.grid
    data = prepare_landcover(
        args.base, subset=args.subset, cell=args.cell, canvas=canvas,
        n_eval=args.n_eval, n_oracle=args.n_oracle, per_class_cap=args.per_class_cap,
        min_side=args.min_side, purity=args.purity, seed=0, cache=args.cache)
    print(f"[meta] {data[-1]}", flush=True)
    templates = harvest_mask_templates(
        args.base, subset=args.subset, canvas=canvas, n_eval=args.n_eval,
        min_frac=args.min_frac, seed=0)

    if args.calibrate:
        (pool_imgs, pool_labels, oracle_imgs, oracle_Y, eval_imgs, eval_Y,
         names, meta) = data
        n_classes = len(names)
        baseX, baseY, spX, spY = build_base_synth(
            pool_imgs, pool_labels, templates, args.per_class_single, args.n_multi,
            0, args.cell, args.grid, canvas, n_classes, args.min_frac,
            args.feather_sigma)
        srcX = baseX[:baseX.shape[0] - spX.shape[0]]
        tgtX = ops.real_to_chw(oracle_imgs)
        calibrate_lambda(baseX, baseY, srcX, tgtX, args.bs, args.device, n_classes)
        return

    lambda_grid = None if args.no_lambda_probe else args.lambda_grid
    rows, No = run(data, templates, args.seeds, args.Us, args.per_class_single,
                   args.n_multi, args.epochs, args.bs, args.lr, args.device,
                   args.cell, args.grid, args.min_frac, args.feather_sigma,
                   args.coral_lambda, args.out_csv, lambda_grid=lambda_grid)
    print_report(rows, args.Us, No, args.fewshot_csv)
    print(f"\n[OUT] {os.path.abspath(args.out_csv)}")


if __name__ == "__main__":
    main()
