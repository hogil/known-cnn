"""Few-shot appearance-gap closing on REAL DLRSD/UC-Merced land cover.

THEORY CONTEXT: the paper proves the APPEARANCE component of the sim-to-real gap
is IRREDUCIBLE from single-label sources alone (non-identifiable). The best
content-blind synthesis (partition_realistic: REAL region geometry + area stats +
co-occurrence, but SYNTHETIC appearance from the single-class crop pool) reaches
~0.83 mAP on the real multi-label eval; the real-multi-label oracle reaches ~0.937
-- a ~0.10 residual APPEARANCE gap that geometry-realistic synthesis provably
cannot close from single-label sources.

QUESTION: how FEW real multi-label tiles added to the synthetic base close that
provably-irreducible appearance gap?  This sweeps K and reports the recovery
curve mAP(K), plus a few-shot-ALONE baseline that isolates whether the synthetic
base still contributes on top of the K real tiles.

Arms (SAME resnet18 / recipe / 3 seeds; eval on the SAME held-out real multi test):
  synth_plus_k : partition_realistic base (2000 synth multi + 587 real singles)
                 + K real multi-label tiles drawn from the oracle TRAIN pool.
  real_only_k  : 587 real singles + K real multi-label tiles (NO synthetic multi).
                 Few-shot-ALONE baseline.  synth_plus_k and real_only_k differ by
                 EXACTLY the 2000 synthetic multi tiles -> the difference is the
                 pure contribution of the synthesis.

REFERENCE POINTS (all emitted by the same sweep, no special-casing):
  K=0   synth_plus_k = pure content-blind synthesis (the ~0.83 partial result)
  K=0   real_only_k  = single_only floor (real single-class crops only)
  K=all real_only_k  = ORACLE = real multi-label pool alone (no synthetic) ~0.937

NO EVAL LEAKAGE: the K real tiles come from the oracle TRAIN pool, which
prepare_landcover splits DISJOINTLY from the held-out eval tiles via the seed-0
permutation.  Per seed the K tiles are the first-K of a fixed permutation of the
oracle pool -> nested subsets (K=5 subset of K=10 subset ...), lower cross-K
variance.

HONESTY: if a small K (<=50) closes most of the gap that is the strong result; if
it needs hundreds, that is reported plainly and is itself an honest bound
consistent with the theory (appearance needs real co-occurrence data).  No tuning
to force a small K -- fixed fair recipe, reported either way.
"""
import os
import csv as csvmod
import argparse

import numpy as np
import torch

from .datasets.dlrsd import prepare_landcover, harvest_mask_templates, DEFAULT_SUBSET
from .synthesis import landcover_ops as ops
from .metrics import bit_f1, compute_map, far, pos_neg_prob
from .run_operator_match_landcover import train_model, _predict

DEFAULT_KS = [0, 5, 10, 25, 50, 100, 250, 500, 1000]
FIELDS = ["dataset", "arm", "K", "backbone", "seed", "n_train",
          "eval_map", "eval_bit_f1", "eval_far", "eval_pos_prob", "eval_neg_prob",
          "train_map", "train_pos_prob", "train_neg_prob"]


def build_base_synth(pool_imgs, pool_labels, templates, per_class_single, n_multi,
                     seed, cell, grid, canvas, n_classes, min_frac, feather_sigma):
    """partition_realistic base = 2000 synth multi (real geometry, synth appearance)
    + real single-class crops.  This is the K=0 content-blind synthesis pool."""
    spX, spY = ops.build_singles(pool_imgs, pool_labels, per_class_single, seed,
                                 cell, grid, n_classes)
    rmX, rmY = ops.build_multi_realistic(templates, pool_imgs, pool_labels, n_multi,
                                         seed, canvas, n_classes, min_frac=min_frac,
                                         feather_sigma=feather_sigma)
    baseX = np.concatenate([rmX, spX])
    baseY = np.concatenate([rmY, spY])
    return baseX, baseY, spX, spY


def eval_row(dataset, arm, K, seed, trX, trY, evX, eval_Y, epochs, bs, lr, device,
             n_classes, pretrained):
    model = train_model(trX, trY, epochs, bs, lr, device, seed, n_classes, pretrained)
    P = _predict(model, evX, bs, device)
    tP = _predict(model, trX, bs, device)
    epos, eneg = pos_neg_prob(P, eval_Y)
    tpos, tneg = pos_neg_prob(tP, trY)
    return {
        "dataset": dataset, "arm": arm, "K": int(K), "backbone": "resnet18",
        "seed": int(seed), "n_train": int(trX.shape[0]),
        "eval_map": round(float(compute_map(P, eval_Y)), 4),
        "eval_bit_f1": round(float(bit_f1(P, eval_Y)), 4),
        "eval_far": round(float(far(P, eval_Y)), 4),
        "eval_pos_prob": round(float(epos), 4),
        "eval_neg_prob": round(float(eneg), 4),
        "train_map": round(float(compute_map(tP, trY)), 4),
        "train_pos_prob": round(float(tpos), 4),
        "train_neg_prob": round(float(tneg), 4),
    }


def run(data, templates, seeds, Ks, per_class_single, n_multi, epochs, bs, lr,
        device, cell, grid, min_frac, feather_sigma, out_csv, dataset="landcover"):
    (pool_imgs, pool_labels, oracle_imgs, oracle_Y, eval_imgs, eval_Y,
     names, meta) = data
    n_classes = len(names)
    canvas = cell * grid
    evX = ops.real_to_chw(eval_imgs)
    oX_all = ops.real_to_chw(oracle_imgs)           # real multi-label TRAIN pool
    No = oX_all.shape[0]
    Ks = [k for k in Ks if k <= No]
    rows = []
    for seed in seeds:
        baseX, baseY, spX, spY = build_base_synth(
            pool_imgs, pool_labels, templates, per_class_single, n_multi, seed,
            cell, grid, canvas, n_classes, min_frac, feather_sigma)
        perm = np.random.default_rng(1000 + seed).permutation(No)   # nested subsets
        print(f"[seed {seed}] base_synth n={baseX.shape[0]} (multi+{spX.shape[0]} "
              f"singles), oracle_pool={No}", flush=True)
        for K in Ks:
            sel = perm[:K]
            oX, oY = oX_all[sel], oracle_Y[sel]
            # synth_plus_k = partition_realistic base + K real multi
            if K == 0:
                trX, trY = baseX, baseY
            else:
                trX = np.concatenate([baseX, oX]); trY = np.concatenate([baseY, oY])
            r = eval_row(dataset, "synth_plus_k", K, seed, trX, trY, evX, eval_Y,
                         epochs, bs, lr, device, n_classes, pretrained=True)
            rows.append(r)
            print(f"[synth+K] K={K:4d} s{seed} n={r['n_train']:4d} "
                  f"mAP={r['eval_map']:.4f} bitF1={r['eval_bit_f1']:.4f} "
                  f"FAR={r['eval_far']:.4f} pos={r['eval_pos_prob']:.4f} "
                  f"neg={r['eval_neg_prob']:.4f}", flush=True)
            # real_only_k = singles + K real multi (few-shot ALONE; K=all -> ORACLE)
            if K == 0:
                trX2, trY2 = spX, spY
            else:
                trX2 = np.concatenate([spX, oX]); trY2 = np.concatenate([spY, oY])
            r2 = eval_row(dataset, "real_only_k", K, seed, trX2, trY2, evX, eval_Y,
                          epochs, bs, lr, device, n_classes, pretrained=True)
            rows.append(r2)
            tag = " (ORACLE)" if K == No else (" (floor)" if K == 0 else "")
            print(f"[real_onlyK] K={K:4d} s{seed} n={r2['n_train']:4d} "
                  f"mAP={r2['eval_map']:.4f} bitF1={r2['eval_bit_f1']:.4f} "
                  f"FAR={r2['eval_far']:.4f} pos={r2['eval_pos_prob']:.4f} "
                  f"neg={r2['eval_neg_prob']:.4f}{tag}", flush=True)
    if out_csv:
        os.makedirs(os.path.dirname(out_csv), exist_ok=True)
        with open(out_csv, "w", newline="") as f:
            w = csvmod.DictWriter(f, fieldnames=FIELDS)
            w.writeheader(); w.writerows(rows)
    return rows, No


def _agg(rows, arm, K, key):
    vals = [r[key] for r in rows if r["arm"] == arm and r["K"] == K]
    return (float(np.mean(vals)), float(np.std(vals))) if vals else (float("nan"),
                                                                     float("nan"))


def print_report(rows, Ks, No, tol=0.02):
    Ks = [k for k in Ks if k <= No]
    om, oms = _agg(rows, "real_only_k", No, "eval_map")          # oracle (real all)
    obf, _ = _agg(rows, "real_only_k", No, "eval_bit_f1")
    base_m, _ = _agg(rows, "synth_plus_k", 0, "eval_map")        # K=0 synth base
    floor_m, _ = _agg(rows, "real_only_k", 0, "eval_map")        # single_only floor
    gap = om - base_m

    print("\n=== FEW-SHOT RECOVERY CURVE  mAP(K)  (real DLRSD/UC-Merced land cover, "
          "resnet18, mean+/-std over 3 seeds) ===")
    print(f"    oracle(real multi ALL, no synth) mAP={om:.4f}+/-{oms:.4f}  bitF1={obf:.4f}")
    print(f"    K=0 synth base (partition_realistic) mAP={base_m:.4f}   "
          f"single_only floor mAP={floor_m:.4f}   appearance gap base->oracle={gap:+.4f}")
    print("| K    | synth+K mAP       | synth+K bitF1 | real-only-K mAP   | "
          "synth adv | gap closed | within 0.02? |")
    print("|------|-------------------|---------------|-------------------|"
          "-----------|------------|--------------|")
    smallest_k = None
    for K in Ks:
        sm, sms = _agg(rows, "synth_plus_k", K, "eval_map")
        sbf, _ = _agg(rows, "synth_plus_k", K, "eval_bit_f1")
        rm, rms = _agg(rows, "real_only_k", K, "eval_map")
        adv = sm - rm
        closed = (sm - base_m) / gap if abs(gap) > 1e-9 else float("nan")
        within = sm >= (om - tol)
        if within and smallest_k is None and K > 0:
            smallest_k = K
        tail = " ORACLE" if K == No else (" base/floor" if K == 0 else "")
        print(f"| {K:4d} | {sm:.4f} +/- {sms:.4f} | {sbf:.4f}        | "
              f"{rm:.4f} +/- {rms:.4f} | {adv:+.4f}   | {closed*100:6.1f}%   | "
              f"{str(within):5s}{tail}")

    print("\n=== VERDICT ===")
    print(f"  oracle mAP={om:.4f}; K=0 synth base mAP={base_m:.4f}; "
          f"appearance gap={gap:+.4f} (bit_F1 base->oracle also reported in table)")
    if smallest_k is None:
        print(f"  NO K in {Ks} reaches within {tol} mAP of the oracle "
              f"(synth+K@K={No} = {_agg(rows,'synth_plus_k',No,'eval_map')[0]:.4f}).")
        verdict = ("NEEDS-LARGE-K: even the largest tested K does not bring "
                   "synth+K within 0.02 mAP of the oracle -> few-shot bound is large; "
                   "appearance needs substantial real co-occurrence data (honest "
                   "negative, consistent with the theory).")
    else:
        sm_at, _ = _agg(rows, "synth_plus_k", smallest_k, "eval_map")
        closed_at = (sm_at - base_m) / gap if abs(gap) > 1e-9 else float("nan")
        size = "SMALL" if smallest_k <= 50 else "MODERATE" if smallest_k <= 250 else "LARGE"
        verdict = (f"{size}-K-CLOSES: smallest K within 0.02 mAP of oracle = {smallest_k} "
                   f"(synth+K mAP={sm_at:.4f}, closes {closed_at*100:.0f}% of the "
                   f"appearance gap). "
                   + ("A handful of real multi-label tiles closes the provably-"
                      "irreducible appearance gap." if smallest_k <= 50 else
                      "Needs tens-to-hundreds of real tiles -- moderate few-shot bound."))
    print(f"  SMALLEST K within {tol} mAP of oracle: {smallest_k}")
    # does synthetic base still help at each K?
    advs = [(_agg(rows, "synth_plus_k", K, "eval_map")[0]
             - _agg(rows, "real_only_k", K, "eval_map")[0]) for K in Ks if K < No]
    n_pos = sum(1 for a in advs if a > 0)
    print(f"  synth base advantage over few-shot-alone: positive at {n_pos}/{len(advs)} "
          f"tested K<all; mean adv={np.mean(advs):+.4f}")
    print(f"\nPLAIN VERDICT: {verdict}")
    return verdict


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    ap.add_argument("--Ks", nargs="+", type=int, default=DEFAULT_KS)
    ap.add_argument("--subset", nargs="+", default=DEFAULT_SUBSET)
    ap.add_argument("--per-class-single", type=int, default=140)
    ap.add_argument("--n-multi", type=int, default=2000,
                    help="synthetic partition_realistic multi tiles in the base pool")
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
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--base", default="E:/data/dlrsd_extracted/DLRSD")
    ap.add_argument("--cache", default=None)
    ap.add_argument("--out-csv",
                    default="outputs/multilabel_synth/landcover_fewshot_ksweep.csv")
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
    rows, No = run(data, templates, args.seeds, args.Ks, args.per_class_single,
                   args.n_multi, args.epochs, args.bs, args.lr, args.device,
                   args.cell, args.grid, args.min_frac, args.feather_sigma,
                   args.out_csv)
    print_report(rows, args.Ks, No)
    print(f"\n[OUT] {os.path.abspath(args.out_csv)}")


if __name__ == "__main__":
    main()
