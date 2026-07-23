"""Severstal confirmatory FULL-SYSTEM run (the audit's confirmatory lever).

Unique public industrial domain with a REAL normal class + single-defect sources
+ multi-defect eval, all IMAGE-LEVEL (no pixel/location). This lets the FULL
system run for the first time on public data: operator-match synthesis (best-grid,
proxy-selected test-blind) + val-margin checkpoint + NB-reject + minimal
known-good-normal conformal calibration, evaluated at REAL-normal FAR.

Pre-registration: single 80/20 train/source-val; normal 50 calibration / 50 sealed
FAR test; ALL multi = sealed positive test. Proxy operator + best-grid frozen (hash)
on source-val BEFORE the sealed test. Primary metric: multi-defect bit-F1 at
real-normal 1% FAR (threshold from normal-cal, FAR measured on untouched
normal-test). Reuses the audit-fixed SVHN components (normalization, shared val
bank, equal budget). Class-2 has 195 single sources (just below the 200 guideline;
Severstal ClassId-2 is intrinsically rare -- documented near-miss, threshold 190).
"""
import argparse, hashlib, json, os, collections, csv as csvmod
import numpy as np
import torch, torch.nn as nn

from .datasets import severstal as SEV
from . import run_svhn_full_image_operator_match as SV   # reuse operators + model

ARMS = ["single_only", "partition", "summation", "cutmix", "mixup", "fcm_pm"]
N_CLASSES = SEV.N_CLASSES   # 4
OUTDIR = "outputs/multilabel_synth/severstal_operator_match_v1"


# ---- image-level synthesis from REAL single-defect sources ----
def synth_combos(arm, byc, imgs, n_per_pair, rng, grid=9, n_groups=3):
    """Combine two real single-defect images (different classes) -> 2-defect combo
    via the operator. Returns (X uint8, Y multihot)."""
    classes = sorted(byc)
    pairs = [(a, b) for i, a in enumerate(classes) for b in classes[i + 1:]]
    Xo, Yo = [], []
    for (ca, cb) in pairs:
        for _ in range(n_per_pair):
            ia = byc[ca][int(rng.integers(len(byc[ca])))]
            ib = byc[cb][int(rng.integers(len(byc[cb])))]
            Xo.append(SV.synth_pair(arm, imgs[ia], imgs[ib], rng, grid=grid, n_groups=n_groups))
            y = np.zeros(N_CLASSES, np.float32); y[ca] = 1.0; y[cb] = 1.0; Yo.append(y)
    return np.stack(Xo).astype(np.uint8), np.stack(Yo)


def build_arm(arm, sX, sY, byc, n_per_pair, rng, grid=9, n_groups=3, match_n=None):
    if arm == "single_only":
        if match_n and match_n > len(sX):
            idx = rng.integers(0, len(sX), size=match_n); return sX[idx], sY[idx]
        return sX, sY
    cX, cY = synth_combos(arm, byc, sX, n_per_pair, rng, grid, n_groups)
    return np.concatenate([cX, sX]), np.concatenate([cY, sY])


def evidence_margin(P, Y):
    Y = (Y >= 0.5)
    pos = np.array([P[i, Y[i]].min() if Y[i].any() else np.nan for i in range(len(P))])
    neg = np.array([P[i, ~Y[i]].max() if (~Y[i]).any() else 0.0 for i in range(len(P))])
    return float(np.nanmean(pos - neg))


def bit_f1_at(P, Y, thr):
    pred = (P >= thr).astype(int); Yb = (Y >= 0.5).astype(int)
    tp = (pred & Yb).sum(); fp = (pred & ~Yb).sum(); fn = (~pred & Yb).sum()
    p = tp / (tp + fp + 1e-9); r = tp / (tp + fn + 1e-9); return 2 * p * r / (p + r + 1e-9)


def far_threshold(normal_max, alpha):
    """tau so real-normal sample-FAR = alpha (quantile of per-normal max score)."""
    return float(np.quantile(normal_max, 1.0 - alpha))


def nb_reject(cal_P, cal_Y, nrm_cal_P, test_P, nrm_test_P, alpha):
    """Per-pattern diagonal-Gaussian defectness gate; gate calibrated so alpha of
    normal-cal pass. Returns (test_accept, nrm_test_accept)."""
    groups = collections.defaultdict(list)
    for i, y in enumerate((cal_Y >= 0.5).astype(int)):
        groups[tuple(y)].append(i)
    pats = []
    for k, idx in groups.items():
        if sum(k) == 0 or len(idx) < 5: continue
        Xg = cal_P[idx]; pats.append((Xg.mean(0), Xg.var(0) + 1e-3))
    mu_n, var_n = nrm_cal_P.mean(0), nrm_cal_P.var(0) + 1e-3
    def llr(X):
        lp = np.max([-0.5 * (((X - mu) ** 2 / var) + np.log(2*np.pi*var)).sum(1) for mu, var in pats], axis=0)
        ln = -0.5 * (((X - mu_n) ** 2 / var_n) + np.log(2*np.pi*var_n)).sum(1)
        return lp - ln
    gate = float(np.quantile(llr(nrm_cal_P), alpha))
    return llr(test_P) >= gate, llr(nrm_test_P) >= gate


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="E:/data/severstal")
    ap.add_argument("--backbone", default="convnextv2_tiny",
                    choices=["small", "resnet18", "convnextv2_tiny", "convnext_tiny_dinov3"])
    ap.add_argument("--seeds", nargs="+", type=int, default=[1, 2, 3, 4, 5])
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--n-per-pair", type=int, default=120)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--run-test", action="store_true")
    ap.add_argument("--max-normal", type=int, default=2000, help="cap normals loaded (mem)")
    ap.add_argument("--max-multi", type=int, default=427)
    args = ap.parse_args()
    os.makedirs(OUTDIR, exist_ok=True)
    SV._BACKBONE = args.backbone
    SV.N_CLASSES = N_CLASSES        # reused model factory builds a 4-class head

    sp = SEV.load_split(args.root, seed=0)
    img_dir = sp["img_dir"]
    print(f"backbone={args.backbone} single_train={len(sp['single_train'][0])} "
          f"single_val={len(sp['single_val'][0])} normal_cal={len(sp['normal_cal'][0])} "
          f"normal_test={len(sp['normal_test'][0])} multi_test={len(sp['multi_test'][0])}", flush=True)

    def load(ids, cap=None):
        ids = ids[:cap] if cap else ids
        return SEV.load_images(ids, img_dir)

    # source singles (train) + their class index
    str_ids, str_Y = sp["single_train"]
    sX = load(str_ids); sY = str_Y
    byc = collections.defaultdict(list)
    for i, y in enumerate((sY >= 0.5)): byc[int(np.argmax(y))].append(i)
    sv_ids, sv_Y = sp["single_val"]; vX = load(sv_ids); vY = sv_Y
    byc_v = collections.defaultdict(list)
    for i, y in enumerate((vY >= 0.5)): byc_v[int(np.argmax(y))].append(i)

    # ---- Phase A: proxy operator + best-grid, frozen (source-val only, NO test) ----
    probe = SV.train(sX, sY, args.epochs, 0, args.device,
                     lr=2e-4 if args.backbone in SV._PRETRAINED else 1e-3)
    scores = {}
    for arm in [a for a in ARMS if a != "single_only"]:
        cX, cY = build_arm(arm, vX, vY, byc_v, 20, np.random.default_rng(1))
        n_c = len(cX) - len(vX)
        scores[arm] = evidence_margin(SV.predict(probe, cX[:n_c], args.device), cY[:n_c])
    ranking = sorted(scores, key=scores.get, reverse=True)
    # FCM-PM geometry FIXED at primary g=3, 9x9 (audit: no grid search -- avoids the
    # "searched g4/16 not primary" criticism; g3/9x9 is the pre-declared primary).
    best = {"best_g": 3, "best_grid": 9, "note": "fixed primary (no grid search)"}
    manifest = dict(backbone=args.backbone, arms=ARMS, classes=list(range(N_CLASSES)),
                    proxy_scores=scores, proxy_ranking=ranking, fcm_pm_best_grid=best,
                    predicted_winner=ranking[0], n_per_pair=args.n_per_pair, epochs=args.epochs)
    manifest["hash"] = hashlib.sha256(json.dumps(manifest, sort_keys=True).encode()).hexdigest()[:16]
    with open(os.path.join(OUTDIR, "proxy_manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)
    print("=== PROXY (source-val; NO test) ===")
    for a in ranking: print(f"  {a:12s} margin={scores[a]:+.4f}")
    print(f"best-grid g={best['best_g']} grid={best['best_grid']} | WINNER={ranking[0]} | hash={manifest['hash']}")
    if not args.run_test:
        print(f"[OUT] {os.path.abspath(os.path.join(OUTDIR,'proxy_manifest.json'))} (Phase A only)")
        return

    # ---- Phase B: freeze protocol, then sealed test ONCE ----
    proto = dict(proxy_hash=manifest["hash"], seeds=args.seeds, per_class_min=190,
                 primary="bitF1@real_normal_FAR_1pct", fpr_targets=[0.005, 0.01, 0.05],
                 checkpoint="val_margin_shared_bank", nb_reject=True, conformal=True)
    proto["hash"] = hashlib.sha256(json.dumps(proto, sort_keys=True).encode()).hexdigest()[:16]
    with open(os.path.join(OUTDIR, "test_protocol.json"), "w") as f:
        json.dump(proto, f, indent=2)
    print(f"[protocol frozen] hash={proto['hash']}", flush=True)

    # load sealed sets ONCE (real normals + real multi)
    ncal_X = load(sp["normal_cal"][0], cap=args.max_normal)
    ntest_X = load(sp["normal_test"][0], cap=args.max_normal)
    mtest_X, mtest_Y = sp["multi_test"]; mtest_X = load(mtest_X, cap=args.max_multi); mtest_Y = mtest_Y[:args.max_multi]
    calib_X, calib_Y = build_arm("partition", sX, sY, byc, 30, np.random.default_rng(555))  # for NB patterns
    print(f"[SEALED] normal_cal={len(ncal_X)} normal_test={len(ntest_X)} multi_test={len(mtest_X)}", flush=True)

    # shared operator-balanced val bank (source-val), all arms
    vbx, vby = [], []
    for op in ["partition", "summation", "cutmix", "mixup"]:
        fx, fy = build_arm(op, vX, vY, byc_v, 8, np.random.default_rng(7777))
        n_c = len(fx) - len(vX); vbx.append(fx[:n_c]); vby.append(fy[:n_c])
    vmX = np.concatenate(vbx); vmY = np.concatenate(vby)

    def _gg(arm): return dict(grid=best["best_grid"], n_groups=best["best_g"]) if arm == "fcm_pm" else {}
    synth_total = len({(a, b) for a in range(N_CLASSES) for b in range(a+1, N_CLASSES)}) * args.n_per_pair + len(sX)

    rows = []
    for arm in ARMS:
        for seed in args.seeds:
            rng = np.random.default_rng(1000 + seed)
            trX, trY = build_arm(arm, sX, sY, byc, args.n_per_pair, rng, match_n=synth_total, **_gg(arm))
            pre = args.backbone in SV._PRETRAINED
            lr = 2e-4 if pre else 1e-3
            m = SV.train_with_margin(trX, trY, vmX, vmY, args.epochs, seed, args.device,
                                     lr=lr, neg_target=0.02,
                                     head_only_epochs=(2 if pre else 0),   # audit 2-stage FT
                                     backbone_lr=(2e-5 if pre else None),  # two-LR
                                     warmup_epochs=2)
            Pcal = SV.predict(m, calib_X, args.device)
            Pnc = SV.predict(m, ncal_X, args.device); Pnt = SV.predict(m, ntest_X, args.device)
            Pmt = SV.predict(m, mtest_X, args.device)
            r = {"arm": arm, "seed": seed, "mAP": _map(Pmt, mtest_Y)}
            # primary: bit-F1 @ real-normal FAR targets (tau from normal_cal, FAR on normal_test)
            for a in proto["fpr_targets"]:
                tau = far_threshold(Pnc.max(1), a)
                r[f"F1@FAR{a}"] = bit_f1_at(Pmt, mtest_Y, tau)
                r[f"realFAR@{a}"] = float((Pnt.max(1) >= tau).mean())
            # NB-reject at 1%
            acc_t, acc_n = nb_reject(Pcal, calib_Y, Pnc, Pmt, Pnt, 0.01)
            r["nb_coverage"] = float(acc_t.mean())
            r["nb_far_after"] = float((acc_n & (Pnt.max(1) >= far_threshold(Pnc.max(1), 0.01))).mean())
            rows.append(r)
            print(f"[{arm:11s} s{seed}] mAP={r['mAP']:.4f} F1@FAR1%={r['F1@FAR0.01']:.4f} "
                  f"realFAR={r['realFAR@0.01']:.3f} nb_cov={r['nb_coverage']:.3f}", flush=True)

    with open(os.path.join(OUTDIR, "sealed_test_results.csv"), "w", newline="") as f:
        w = csvmod.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
    _verdict(rows, ranking[0])
    print(f"[OUT] {os.path.abspath(os.path.join(OUTDIR,'sealed_test_results.csv'))}", flush=True)


def _map(P, Y):
    aps = []
    for c in range(N_CLASSES):
        y = (Y[:, c] >= 0.5).astype(int)
        if y.sum() == 0: continue
        o = np.argsort(-P[:, c]); yc = y[o]; tp = np.cumsum(yc)
        prec = tp / (np.arange(len(yc)) + 1); rec = tp / y.sum()
        aps.append(np.sum((rec[1:] - rec[:-1]) * prec[1:]) + rec[0] * prec[0])
    return float(np.mean(aps))


def _verdict(rows, proxy_winner):
    ag = collections.defaultdict(lambda: collections.defaultdict(list))
    for r in rows:
        ag[r["arm"]]["f1"].append(r["F1@FAR0.01"]); ag[r["arm"]]["map"].append(r["mAP"])
    arms = sorted(ag, key=lambda a: -np.mean(ag[a]["f1"]))
    print("\n=== SEVERSTAL sealed | bit-F1 @ real-normal 1% FAR (primary), mean ===")
    for a in arms:
        tag = " <-ours(FCM-PM)" if a == "fcm_pm" else ""
        print(f"  {a:12s} F1@FAR1%={np.mean(ag[a]['f1']):.4f} mAP={np.mean(ag[a]['map']):.4f}{tag}")
    adm = [a for a in arms if a != "oracle"]
    seeds = sorted(set(r["seed"] for r in rows))
    def by(a): return {r["seed"]: r["F1@FAR0.01"] for r in rows if r["arm"] == a}
    fcm = by("fcm_pm")
    print("\nFCM-PM full-system paired vs content-blind (F1@real-FAR1%):")
    for opp in ["single_only", "summation", "cutmix", "mixup"]:
        ov = by(opp); d = [fcm[s] - ov[s] for s in seeds if s in fcm and s in ov]
        w = sum(1 for x in d if x > 0); lo = np.mean(d) - 1.96 * np.std(d, ddof=1) / np.sqrt(len(d))
        print(f"  vs {opp:11s}: d={np.mean(d):+.4f} wins={w}/{len(d)} CI_low={lo:+.4f}{' CI>0' if lo>0 else ''}")


if __name__ == "__main__":
    main()
