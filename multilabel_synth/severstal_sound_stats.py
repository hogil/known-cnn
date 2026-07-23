"""Soundness re-analysis of the Severstal sealed CSV (audit fix, 260724).

Replaces the 1.96*SE normal-approx CI with (a) paired t 95% CI, (b) exact Wilcoxon
signed-rank, (c) seed bootstrap CI, over the SAME seed-matched arm pairs. Also
reports realized-FAR mean/range and the NB-reject null effect. No training.

Honest naming: the Severstal 'fcm_pm' arm is the SIMPLIFIED single grid-complement
(9x9, 1/3 cells swapped) -- reported as grid_complement_g3_9, NOT full FCM-PM
(which would use the complement-set + Pair-Mask view in synthesis/fcmpm_image.py,
not wired into this runner).
"""
import csv, numpy as np
from scipy import stats

CSV = "outputs/multilabel_synth/severstal_operator_match_v2_b1/sealed_test_results.csv"
RENAME = {"fcm_pm": "grid_complement_g3_9"}


def load():
    rows = list(csv.DictReader(open(CSV)))
    for r in rows:
        for k in r:
            if k != "arm":
                try: r[k] = float(r[k])
                except: pass
        r["arm"] = RENAME.get(r["arm"], r["arm"])
    return rows


def by(rows, arm, key):
    return {int(r["seed"]): r[key] for r in rows if r["arm"] == arm}


def paired(rows, a, b, key):
    seeds = sorted(set(int(r["seed"]) for r in rows))
    xa, xb = by(rows, a, key), by(rows, b, key)
    d = np.array([xa[s] - xb[s] for s in seeds if s in xa and s in xb])
    n = len(d); md = d.mean()
    # paired t 95% CI
    if np.allclose(d, d[0]):
        tlo = thi = md; p_t = np.nan
    else:
        se = d.std(ddof=1) / np.sqrt(n); tcrit = stats.t.ppf(0.975, n - 1)
        tlo, thi = md - tcrit * se, md + tcrit * se
        p_t = stats.ttest_rel([xa[s] for s in seeds], [xb[s] for s in seeds]).pvalue
    # exact Wilcoxon signed-rank (n=5 -> exact)
    try:
        w = stats.wilcoxon(d, alternative="greater", zero_method="wilcox", mode="exact")
        p_w = w.pvalue
    except Exception:
        p_w = np.nan
    # seed bootstrap 95% CI of mean diff (10000 resamples, fixed seed for reproducibility)
    rng = np.random.default_rng(0)
    bs = np.array([rng.choice(d, size=n, replace=True).mean() for _ in range(10000)])
    blo, bhi = np.percentile(bs, [2.5, 97.5])
    wins = int((d > 0).sum())
    return dict(mean=md, n=n, wins=wins, t_lo=tlo, t_hi=thi, p_t=p_t,
                p_wilcoxon=p_w, boot_lo=blo, boot_hi=bhi)


def fmt(a, b, key, r):
    sig = "sig" if (r["t_lo"] > 0) else ("bound" if r["mean"] > 0 else "ns")
    return (f"| {a:20s} - {b:12s} | {r['mean']:+.4f} | {r['wins']}/{r['n']} | "
            f"[{r['t_lo']:+.4f},{r['t_hi']:+.4f}] | [{r['boot_lo']:+.4f},{r['boot_hi']:+.4f}] | "
            f"{r['p_wilcoxon']:.3f} | {sig:5s} |")


def main():
    rows = load()
    arms = ["partition", "cutmix", "summation", "mixup", "grid_complement_g3_9", "single_only"]
    for key, lab in [("F1@FAR0.01", "FAR1%"), ("F1@FAR0.05", "FAR5%")]:
        print(f"\n=== bit-F1 @ real-normal {lab} | mean +/- (paired) ===")
        for a in arms:
            v = np.array([r[key] for r in rows if r["arm"] == a])
            rf = np.array([r["realFAR@0.01" if key == "F1@FAR0.01" else "realFAR@0.05"]
                           for r in rows if r["arm"] == a]) if False else None
            print(f"  {a:22s} {v.mean():.4f}  (seeds {v.min():.4f}-{v.max():.4f})")
        print(f"\n  {lab} paired comparisons (paired-t 95% CI, bootstrap 95% CI, exact Wilcoxon 1-sided):")
        print("| comparison                          |  mean d | wins | paired-t 95% CI      | bootstrap 95% CI     | Wilcx | verdict |")
        print("|-------------------------------------|---------|------|----------------------|----------------------|-------|---------|")
        # each synth vs single
        for a in ["partition", "cutmix", "summation", "mixup", "grid_complement_g3_9"]:
            print(fmt(a, "single_only", key, paired(rows, a, "single_only", key)))
        # partition vs each other synth
        for b in ["cutmix", "summation", "mixup", "grid_complement_g3_9"]:
            print(fmt("partition", b, key, paired(rows, "partition", b, key)))
    # realized FAR and NB null effect
    print("\n=== realized FAR @ nominal 1% (calibration honesty) ===")
    for a in arms:
        rf = np.array([r["realFAR@0.01"] for r in rows if r["arm"] == a])
        print(f"  {a:22s} realFAR mean {rf.mean()*100:.2f}%  range {rf.min()*100:.2f}-{rf.max()*100:.2f}%")
    print("\n=== NB-reject effect on Severstal (nb_far_after vs realFAR@0.01) ===")
    same = 0; tot = 0
    for r in rows:
        tot += 1
        if abs(r["nb_far_after"] - r["realFAR@0.01"]) < 1e-9: same += 1
    covs = np.array([r["nb_coverage"] for r in rows])
    print(f"  nb_far_after == realFAR@0.01 in {same}/{tot} rows -> NB adds NO FAR reduction")
    print(f"  nb_coverage range {covs.min():.3f}-{covs.max():.3f} (NB only drops some positive coverage)")


if __name__ == "__main__":
    main()
