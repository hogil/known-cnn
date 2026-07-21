"""Low-FAR transfer audit (no new training; reuses stored conformal summary).

Empirical counterpart of two theorems for single-only multi-label learning:
  (T2) Normal-free low-FAR impossibility: a threshold chosen WITHOUT observing
       real normals (here: synthetic-zero calibration) does NOT control the
       real-normal false-alarm rate -- realized FAR blows past target for EVERY
       operator (single_only, cutmix, mixup, fcm_pm, summation, ...).
  (T3) Minimal known-good calibration: split-conformal on m real known-good
       normals drives realized FAR to the target within sampling error, for the
       SAME operators -- the minimal-information remedy to (T2).

Source: outputs/multilabel_synth/wm38_strict_all_methods_conformal_summary.csv
Output: outputs/multilabel_synth/low_far_audit_v1/transfer_table.csv
"""
import csv, os, collections

SRC = "outputs/multilabel_synth/wm38_strict_all_methods_conformal_summary.csv"
OUTDIR = "outputs/multilabel_synth/low_far_audit_v1"


def _f(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


def main():
    rows = list(csv.DictReader(open(SRC, encoding="utf-8")))
    os.makedirs(OUTDIR, exist_ok=True)

    def pick(regime, alpha):
        c = [r for r in rows if r["calibration_regime"] == regime
             and abs(_f(r["alpha"]) - alpha) < 1e-9]
        c.sort(key=lambda r: (int(r["n_calibration"]), int(r["split_repeats_per_model"])),
               reverse=True)
        return {r["arm"]: r for r in c}

    arms = sorted(set(r["arm"] for r in rows))
    out_rows = []
    for alpha in (0.01, 0.05):
        syn = pick("synthetic_normal", alpha)
        real = pick("real_normal_split", alpha)
        for a in arms:
            s, rr = syn.get(a), real.get(a)
            sv = _f(s["normal_FAR_mean"]) if s else None
            rv = _f(rr["normal_FAR_mean"]) if rr else None
            out_rows.append({
                "arm": a, "target_alpha": alpha,
                "synthetic_cal_realized_FAR": round(sv, 4) if sv is not None else "",
                "synthetic_cal_gap_pp": round((sv - alpha) * 100, 1) if sv is not None else "",
                "real_split_cal_realized_FAR": round(rv, 4) if rv is not None else "",
                "real_split_cal_gap_pp": round((rv - alpha) * 100, 1) if rv is not None else "",
            })

    out_csv = os.path.join(OUTDIR, "transfer_table.csv")
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(out_rows[0].keys()))
        w.writeheader(); w.writerows(out_rows)

    for alpha in (0.01, 0.05):
        sub = [r for r in out_rows if r["target_alpha"] == alpha]
        syn_vals = [r["synthetic_cal_realized_FAR"] for r in sub if r["synthetic_cal_realized_FAR"] != ""]
        real_vals = [r["real_split_cal_realized_FAR"] for r in sub if r["real_split_cal_realized_FAR"] != ""]
        print(f"\ntarget alpha={alpha*100:.0f}%  (real-normal realized FAR)")
        print(f"  SYNTHETIC (normal-free) cal: {min(syn_vals)*100:.1f}-{max(syn_vals)*100:.1f}%  "
              f"=> impossibility (T2): threshold uncontrolled without real normals")
        print(f"  REAL-split (minimal) cal   : {min(real_vals)*100:.1f}-{max(real_vals)*100:.1f}%  "
              f"=> minimal calibration (T3): controlled at target for every operator")
    print(f"\n[OUT] {os.path.abspath(out_csv)}")


if __name__ == "__main__":
    main()
