"""6-method threshold sweep on frozen iter116J eval probs.

Methods:
  M1 F1-max (sklearn precision_recall_curve, baseline)
  M2 Gaussian fit (Bayes optimal decision boundary between pos/neg Gaussians)
  M3 Beta fit (probs in [0,1], Beta more natural than Gaussian)
  M4 KDE crossover (kernel density estimate, find x where neg_pdf == pos_pdf)
  M5 Youden J (max TPR-FPR on ROC)
  M6 Equal Error Rate (TPR=1-FPR, EER point)

For each method: compute bit_F1 + Total FAR on v15direct_n2000 eval.
"""
import sys, json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import beta as beta_dist
from sklearn.metrics import precision_recall_curve, roc_curve

TRAIN_BITS = ['bank_boundary', 'fork', 'scratch', 'scratch_rot']
NEG_KEYS = {'Normal', 'Invalid', 'CenterDonut', 'CrossScratch', 'DiagonalSmear', 'Starburst'}
POS9 = [
    ('bank_boundary',), ('fork',), ('scratch',), ('scratch_rot',),
    ('bank_boundary','fork'), ('bank_boundary','scratch'), ('bank_boundary','scratch_rot'),
    ('fork','scratch'), ('fork','scratch_rot'),
]

EVAL_PATH = Path("outputs/iter116J_eval_no3fp/eval_260526_055757/preds_chip.parquet")


def bits_from_key(k):
    if not isinstance(k, str):
        return set()
    return set(p for p in k.split('+') if p in TRAIN_BITS)


def method_f1_max(p_pos, p_neg):
    p = np.concatenate([p_pos, p_neg])
    y = np.concatenate([np.ones_like(p_pos), np.zeros_like(p_neg)])
    prec, rec, thr = precision_recall_curve(y, p)
    f1 = 2*prec*rec / np.maximum(prec+rec, 1e-12)
    i = int(np.argmax(f1[:-1]))
    return float(thr[i])


def method_gaussian(p_pos, p_neg):
    """Bayes optimal threshold between two Gaussians: solve N(mu_p, sig_p) = N(mu_n, sig_n)."""
    mu_p, sig_p = p_pos.mean(), max(p_pos.std(), 1e-3)
    mu_n, sig_n = p_neg.mean(), max(p_neg.std(), 1e-3)
    # Solve x: -0.5*((x-mu_p)/sig_p)^2 - log(sig_p) = -0.5*((x-mu_n)/sig_n)^2 - log(sig_n)
    a = 1/sig_n**2 - 1/sig_p**2
    b = 2*(mu_p/sig_p**2 - mu_n/sig_n**2)
    c = mu_n**2/sig_n**2 - mu_p**2/sig_p**2 - 2*np.log(sig_p/sig_n)
    if abs(a) < 1e-9:
        if abs(b) < 1e-9:
            return 0.5
        return float(np.clip(-c/b, 0, 1))
    disc = b**2 - 4*a*c
    if disc < 0:
        return 0.5
    x1 = (-b + np.sqrt(disc)) / (2*a)
    x2 = (-b - np.sqrt(disc)) / (2*a)
    cands = [x for x in (x1, x2) if 0 < x < 1]
    if not cands:
        return float((mu_p + mu_n) / 2)
    return float(min(cands, key=lambda x: abs(x - (mu_p+mu_n)/2)))


def method_beta(p_pos, p_neg):
    """Fit Beta to pos and neg; find crossover."""
    eps = 1e-6
    pp = np.clip(p_pos, eps, 1-eps)
    pn = np.clip(p_neg, eps, 1-eps)
    try:
        ap, bp, _, _ = beta_dist.fit(pp, floc=0, fscale=1)
        an, bn, _, _ = beta_dist.fit(pn, floc=0, fscale=1)
    except Exception:
        return method_gaussian(p_pos, p_neg)
    xs = np.linspace(0.01, 0.99, 999)
    pdf_p = beta_dist.pdf(xs, ap, bp)
    pdf_n = beta_dist.pdf(xs, an, bn)
    diff = pdf_p - pdf_n
    # find crossover where diff changes sign from negative to positive
    sign_change = np.where(np.diff(np.sign(diff)) > 0)[0]
    if len(sign_change) == 0:
        return method_gaussian(p_pos, p_neg)
    return float(xs[sign_change[-1]])


def method_kde(p_pos, p_neg):
    """KDE-based crossover."""
    try:
        kp = stats.gaussian_kde(p_pos, bw_method='scott')
        kn = stats.gaussian_kde(p_neg, bw_method='scott')
    except Exception:
        return method_gaussian(p_pos, p_neg)
    xs = np.linspace(0.001, 0.999, 999)
    dp = kp(xs)
    dn = kn(xs)
    diff = dp - dn
    sign_change = np.where(np.diff(np.sign(diff)) > 0)[0]
    if len(sign_change) == 0:
        return method_gaussian(p_pos, p_neg)
    return float(xs[sign_change[-1]])


def method_youden(p_pos, p_neg):
    p = np.concatenate([p_pos, p_neg])
    y = np.concatenate([np.ones_like(p_pos), np.zeros_like(p_neg)])
    fpr, tpr, thr = roc_curve(y, p)
    j = tpr - fpr
    i = int(np.argmax(j))
    return float(thr[i])


def method_eer(p_pos, p_neg):
    p = np.concatenate([p_pos, p_neg])
    y = np.concatenate([np.ones_like(p_pos), np.zeros_like(p_neg)])
    fpr, tpr, thr = roc_curve(y, p)
    fnr = 1 - tpr
    i = int(np.argmin(np.abs(fpr - fnr)))
    return float(thr[i])


METHODS = {
    'M1_F1max':    method_f1_max,
    'M2_Gaussian': method_gaussian,
    'M3_Beta':     method_beta,
    'M4_KDE':      method_kde,
    'M5_Youden':   method_youden,
    'M6_EER':      method_eer,
}


def eval_with_thresholds(df, thr_map):
    """Apply per-bit thresholds, compute bit_F1 (POS9 macro) + FAR splits."""
    pred_bits_list = []
    for _, row in df.iterrows():
        bits = set()
        for b in TRAIN_BITS:
            if row['prob_'+b] >= thr_map[b]:
                bits.add(b)
        pred_bits_list.append(bits)

    # bit-level F1 (POS9 macro across the 9 keys)
    f1s = []
    for key in POS9:
        keyset = set(key)
        tp = fp = fn = 0
        for _, (row, pb) in enumerate(zip(df.itertuples(), pred_bits_list)):
            gt = bits_from_key(row.class_key)
            for b in TRAIN_BITS:
                if b in keyset:
                    pred_has = b in pb
                    gt_has = b in gt and gt == keyset
                    # POS9 F1: TP if class_key matches AND bit predicted
                    # simpler approach: per-bit F1 within this key's chips
        # use simpler per-bit POS9 cell F1
        bit = list(keyset)[0] if len(keyset) == 1 else None  # handle later
        # actually compute exact-match for combo, single-bit for singles
        if len(keyset) == 1:
            b = list(keyset)[0]
            mask = df['class_key'] == key[0]
            n_pos = mask.sum()
            if n_pos == 0:
                continue
            tp = sum(1 for i, m in enumerate(mask) if m and b in pred_bits_list[i])
            fn = n_pos - tp
            fp = sum(1 for i, m in enumerate(mask) if (not m) and b in pred_bits_list[i] and
                     bits_from_key(df.iloc[i]['class_key']) != keyset)
        else:
            keystr = '+'.join(sorted(keyset))
            mask = df['class_key'].apply(lambda k: bits_from_key(k) == keyset).values
            n_pos = mask.sum()
            if n_pos == 0:
                continue
            tp = sum(1 for i, m in enumerate(mask) if m and pred_bits_list[i] == keyset)
            fn = n_pos - tp
            fp = sum(1 for i, m in enumerate(mask) if (not m) and pred_bits_list[i] == keyset)
        prec = tp / max(tp+fp, 1)
        rec  = tp / max(tp+fn, 1)
        f1 = 2*prec*rec / max(prec+rec, 1e-12) if (prec+rec) > 0 else 0.0
        f1s.append(f1)
    bit_f1 = float(np.mean(f1s))

    # FAR: any prediction on negative class chip
    far_groups = {'NI': ['Normal', 'Invalid'],
                  'OOD': ['CenterDonut', 'CrossScratch', 'DiagonalSmear', 'Starburst']}
    far_out = {}
    for g, keys in far_groups.items():
        mask = df['class_key'].isin(keys)
        n = mask.sum()
        if n == 0:
            far_out[g] = 0.0
            continue
        fp = sum(1 for i, m in enumerate(mask) if m and len(pred_bits_list[i]) > 0)
        far_out[g] = 100.0 * fp / n
    neg_mask = df['class_key'].isin(NEG_KEYS)
    n_neg = neg_mask.sum()
    fp_tot = sum(1 for i, m in enumerate(neg_mask) if m and len(pred_bits_list[i]) > 0)
    far_out['Total'] = 100.0 * fp_tot / max(n_neg, 1)
    return bit_f1, far_out


def main():
    df = pd.read_parquet(EVAL_PATH)
    print(f"loaded {len(df)} rows from {EVAL_PATH}")
    # I10 only
    df = df[df['cell_id'] == 'T0__I10'].reset_index(drop=True)
    print(f"I10 rows: {len(df)}, unique class_key: {df['class_key'].nunique()}")

    # gather per-bit pos/neg prob arrays
    bit_data = {}
    for b in TRAIN_BITS:
        is_pos = df['class_key'].apply(lambda k: b in bits_from_key(k)).values
        bit_data[b] = (df.loc[is_pos, 'prob_'+b].values,
                       df.loc[~is_pos, 'prob_'+b].values)

    print()
    print("=== Per-bit thresholds by method ===")
    print(f"{'method':<14} " + " ".join(f"{b:>14}" for b in TRAIN_BITS))
    thr_per_method = {}
    for mname, fn in METHODS.items():
        thrs = {b: fn(*bit_data[b]) for b in TRAIN_BITS}
        thr_per_method[mname] = thrs
        print(f"{mname:<14} " + " ".join(f"{thrs[b]:>14.4f}" for b in TRAIN_BITS))

    print()
    print("=== Eval results ===")
    print(f"{'method':<14} {'bit_F1':>8} {'NI':>8} {'OOD':>8} {'Total':>8}")
    results = []
    for mname, thrs in thr_per_method.items():
        bit_f1, far = eval_with_thresholds(df, thrs)
        print(f"{mname:<14} {bit_f1:>8.4f} {far['NI']:>8.2f} {far['OOD']:>8.2f} {far['Total']:>8.2f}")
        results.append({'method': mname, 'bit_F1': bit_f1, **far, 'thresholds': thrs})

    out_path = Path("_threshold_methods_sweep.json")
    out_path.write_text(json.dumps(results, indent=2))
    print(f"\n[OUT] D:/project/known-cnn/{out_path}")

if __name__ == '__main__':
    main()
