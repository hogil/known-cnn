"""Threshold method sweep — applied AFTER I10 entropy gate (frozen).

Insight: in frozen eval, all 8397 negative chips caught by entropy gate.
Threshold method only affects positive chip recall (bit_F1).
FAR = 0 for ALL methods (negatives don't reach this stage).

Compare 6 methods on positive chips only. Best method = highest bit_F1.
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
    mu_p, sig_p = p_pos.mean(), max(p_pos.std(), 1e-3)
    mu_n, sig_n = p_neg.mean(), max(p_neg.std(), 1e-3)
    a = 1/sig_n**2 - 1/sig_p**2
    b = 2*(mu_p/sig_p**2 - mu_n/sig_n**2)
    c = mu_n**2/sig_n**2 - mu_p**2/sig_p**2 - 2*np.log(sig_p/sig_n)
    if abs(a) < 1e-9:
        return 0.5 if abs(b) < 1e-9 else float(np.clip(-c/b, 0, 1))
    disc = b**2 - 4*a*c
    if disc < 0:
        return 0.5
    cands = [x for x in ((-b+np.sqrt(disc))/(2*a), (-b-np.sqrt(disc))/(2*a)) if 0 < x < 1]
    if not cands:
        return float((mu_p + mu_n) / 2)
    return float(min(cands, key=lambda x: abs(x - (mu_p+mu_n)/2)))


def method_beta(p_pos, p_neg):
    eps = 1e-6
    pp = np.clip(p_pos, eps, 1-eps)
    pn = np.clip(p_neg, eps, 1-eps)
    try:
        ap, bp, _, _ = beta_dist.fit(pp, floc=0, fscale=1)
        an, bn, _, _ = beta_dist.fit(pn, floc=0, fscale=1)
    except Exception:
        return method_gaussian(p_pos, p_neg)
    xs = np.linspace(0.01, 0.99, 999)
    diff = beta_dist.pdf(xs, ap, bp) - beta_dist.pdf(xs, an, bn)
    sc = np.where(np.diff(np.sign(diff)) > 0)[0]
    if len(sc) == 0:
        return method_gaussian(p_pos, p_neg)
    return float(xs[sc[-1]])


def method_kde(p_pos, p_neg):
    try:
        kp = stats.gaussian_kde(p_pos, bw_method='scott')
        kn = stats.gaussian_kde(p_neg, bw_method='scott')
    except Exception:
        return method_gaussian(p_pos, p_neg)
    xs = np.linspace(0.001, 0.999, 999)
    diff = kp(xs) - kn(xs)
    sc = np.where(np.diff(np.sign(diff)) > 0)[0]
    if len(sc) == 0:
        return method_gaussian(p_pos, p_neg)
    return float(xs[sc[-1]])


def method_youden(p_pos, p_neg):
    p = np.concatenate([p_pos, p_neg])
    y = np.concatenate([np.ones_like(p_pos), np.zeros_like(p_neg)])
    fpr, tpr, thr = roc_curve(y, p)
    return float(thr[int(np.argmax(tpr-fpr))])


def method_eer(p_pos, p_neg):
    p = np.concatenate([p_pos, p_neg])
    y = np.concatenate([np.ones_like(p_pos), np.zeros_like(p_neg)])
    fpr, tpr, thr = roc_curve(y, p)
    return float(thr[int(np.argmin(np.abs(fpr - (1-tpr))))])


METHODS = {
    'M1_F1max':    method_f1_max,
    'M2_Gaussian': method_gaussian,
    'M3_Beta':     method_beta,
    'M4_KDE':      method_kde,
    'M5_Youden':   method_youden,
    'M6_EER':      method_eer,
}


def eval_on_positives(df_pos, thr_map):
    """Apply per-bit threshold on positive chips, compute per-bit macro F1 (matches frozen metric)."""
    # Predict bits
    pred_bits = []
    for _, row in df_pos.iterrows():
        b = set()
        for bit in TRAIN_BITS:
            if row['prob_' + bit] >= thr_map[bit]:
                b.add(bit)
        pred_bits.append(b)
    keys = df_pos['class_key'].apply(bits_from_key).tolist()

    # Per-bit macro F1 (frozen's bit_F1 metric)
    per_bit = {}
    for bit in TRAIN_BITS:
        tp = sum(1 for gt, pb in zip(keys, pred_bits) if bit in gt and bit in pb)
        fp = sum(1 for gt, pb in zip(keys, pred_bits) if bit not in gt and bit in pb)
        fn = sum(1 for gt, pb in zip(keys, pred_bits) if bit in gt and bit not in pb)
        prec = tp / max(tp+fp, 1)
        rec = tp / max(tp+fn, 1)
        f1 = 2*prec*rec/max(prec+rec, 1e-12) if (prec+rec) > 0 else 0.0
        per_bit[bit] = round(f1, 4)
    bit_f1 = float(np.mean(list(per_bit.values())))
    return bit_f1, per_bit


def main():
    df = pd.read_parquet(EVAL_PATH)
    df = df[df['cell_id'] == 'T0__I10'].reset_index(drop=True)
    # POSITIVE chips only — entropy gate caught all negatives in frozen
    pos = df[~df['class_key'].isin(NEG_KEYS)].reset_index(drop=True)
    print(f"positive chips for eval: {len(pos)} (of {len(df)} total)")
    print(f"class_keys: {sorted(pos['class_key'].unique())}")
    print()

    # gather per-bit pos/neg prob arrays (within positive subset)
    bit_data = {}
    for b in TRAIN_BITS:
        is_pos = pos['class_key'].apply(lambda k: b in bits_from_key(k)).values
        bit_data[b] = (pos.loc[is_pos, 'prob_'+b].values,
                       pos.loc[~is_pos, 'prob_'+b].values)
        print(f"  {b}: pos n={is_pos.sum()} neg n={(~is_pos).sum()}")

    print()
    print("=== Per-bit thresholds (computed on positive chip subset) ===")
    print(f"{'method':<14} " + " ".join(f"{b:>14}" for b in TRAIN_BITS))
    thr_per_method = {}
    for mname, fn in METHODS.items():
        thrs = {b: fn(*bit_data[b]) for b in TRAIN_BITS}
        thr_per_method[mname] = thrs
        print(f"{mname:<14} " + " ".join(f"{thrs[b]:>14.4f}" for b in TRAIN_BITS))

    # baseline reference: frozen thresholds
    FROZEN = {'bank_boundary': 0.46, 'fork': 0.26, 'scratch': 0.18, 'scratch_rot': 0.26}
    thr_per_method['M0_Frozen'] = FROZEN

    print()
    print("=== Eval results (FAR=0% for all, only bit_F1 matters) ===")
    print(f"{'method':<14} {'bit_F1':>8}  per-key F1")
    results = []
    rows_table = []
    for mname in ['M0_Frozen'] + list(METHODS.keys()):
        thrs = thr_per_method[mname]
        bit_f1, per_key = eval_on_positives(pos, thrs)
        print(f"{mname:<14} {bit_f1:>8.4f}  {per_key}")
        results.append({'method': mname, 'bit_F1': bit_f1, 'thresholds': thrs, 'per_key': per_key})
        rows_table.append((mname, bit_f1, per_key))

    out_path = Path("_threshold_methods_v2.json")
    out_path.write_text(json.dumps(results, indent=2))
    print(f"\n[OUT] D:/project/known-cnn/{out_path}")

    # find best
    best = max(rows_table, key=lambda r: r[1])
    print(f"\n>>> BEST: {best[0]} bit_F1={best[1]:.4f}")

if __name__ == '__main__':
    main()
