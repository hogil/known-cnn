"""Inference-side OOD ensemble: maximize DEFECT-vs-NEG margin (no retraining).

4 OOD scores, each computed from the frozen iter116J 4-bit prob vector:
  1. NB log-lik     — Gaussian Naive Bayes (defect-only fit) max joint log-likelihood
  2. JointEnergy    — published multi-label OOD score [Wang et al. NeurIPS 2021]:
                        E(x) = sum_b log(1+exp(z_b)) = -sum_b log(1-p_b),  z=logit(p)
  3. max_prob       — max per-bit prob (baseline MSP)
  4. neg_entropy    — -softmax_entropy (low entropy = confident = ID)

For each score we report the DEFECT-vs-NEG separation:
  DEFECT min / 5%   vs   NEG max / 95%   →  gap = margin we want to maximize.
Then we pick, per score, the threshold giving FAR=0% and report bit_F1.
Finally an AND-ensemble (accept only if ALL scores pass their FAR=0 threshold)
and the NB+reject reference.
"""
import sys
import numpy as np
import pandas as pd
from sklearn.naive_bayes import GaussianNB

TRAIN_BITS = ['bank_boundary', 'fork', 'scratch', 'scratch_rot']
NEG_GROUP = ['Normal', 'Invalid', 'CenterDonut', 'CrossScratch', 'DiagonalSmear', 'Starburst']
COMBO = ['bank_boundary+fork', 'bank_boundary+scratch', 'bank_boundary+scratch_rot',
         'fork+scratch', 'fork+scratch_rot', 'scratch+scratch_rot']
SINGLE = ['bank_boundary', 'fork', 'scratch', 'scratch_rot']
DEFECT = SINGLE + COMBO


def bits_from_key(k):
    return set(p for p in k.split('+') if p in TRAIN_BITS) if isinstance(k, str) else set()


def bit_f1(pred_keys, ytest):
    pred_bits = [bits_from_key(p) if p != 'UNKNOWN' else set() for p in pred_keys]
    gt_bits = [bits_from_key(t) for t in ytest]
    per = []
    for bit in TRAIN_BITS:
        tp = sum(1 for g, p in zip(gt_bits, pred_bits) if bit in g and bit in p)
        fp = sum(1 for g, p in zip(gt_bits, pred_bits) if bit not in g and bit in p)
        fn = sum(1 for g, p in zip(gt_bits, pred_bits) if bit in g and bit not in p)
        prec = tp / max(tp + fp, 1); rec = tp / max(tp + fn, 1)
        per.append(2 * prec * rec / max(prec + rec, 1e-12) if (prec + rec) > 0 else 0.0)
    return float(np.mean(per))


def far(pred_keys, ytest):
    pred_bits = [bits_from_key(p) if p != 'UNKNOWN' else set() for p in pred_keys]
    neg_mask = np.isin(ytest, NEG_GROUP)
    fp = sum(1 for i, m in enumerate(neg_mask) if m and len(pred_bits[i]) > 0)
    return 100 * fp / max(neg_mask.sum(), 1)


def sep_report(name, score, defect_mask, neg_mask, higher_is_id=True):
    """Print DEFECT-vs-NEG separation. Returns (gap, threshold for FAR=0)."""
    d = score[defect_mask]; n = score[neg_mask]
    if higher_is_id:
        # ID = high score. NEG should be low. threshold = NEG max → accept >= thr
        d_lo, n_hi = d.min(), n.max()
        gap = d_lo - n_hi  # positive = clean separation
        thr = n_hi  # FAR=0 boundary (anything <= NEG max gets some NEG fp at equality, use +eps)
        return name, d_lo, np.percentile(d, 5), n_hi, np.percentile(n, 95), gap, thr
    else:
        d_hi, n_lo = d.max(), n.min()
        gap = n_lo - d_hi
        thr = n_lo
        return name, d_hi, np.percentile(d, 95), n_lo, np.percentile(n, 5), gap, thr


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else \
        'outputs/iter116J_eval_no3fp/eval_260526_055757/preds_chip.parquet'
    df = pd.read_parquet(path)
    i10 = df[df['cell_id'] == 'T0__I10'].reset_index(drop=True)

    rng = np.random.default_rng(42)
    fit_idx, test_idx = [], []
    for c in i10['class_key'].unique():
        idx = i10.index[i10['class_key'] == c].values
        rng.shuffle(idx); h = len(idx) // 2
        fit_idx.extend(idx[:h]); test_idx.extend(idx[h:])
    fit_df, test_df = i10.loc[fit_idx], i10.loc[test_idx].reset_index(drop=True)
    cols = ['prob_' + b for b in TRAIN_BITS]
    Xtest = np.clip(test_df[cols].values, 1e-6, 1 - 1e-6)
    ytest = test_df['class_key'].values

    defect_mask = np.isin(ytest, DEFECT)
    neg_mask = np.isin(ytest, NEG_GROUP)

    # --- score 1: NB log-likelihood (defect-only fit) ---
    fmask = fit_df['class_key'].isin(DEFECT)
    Xfit = np.clip(fit_df.loc[fmask, cols].values, 1e-6, 1 - 1e-6)
    yfit = fit_df.loc[fmask, 'class_key'].values
    nb = GaussianNB(var_smoothing=1e-6).fit(Xfit, yfit)
    classes = list(nb.classes_)
    ll = nb._joint_log_likelihood(Xtest)
    nb_score = ll.max(axis=1)            # higher = more ID
    nb_pred = np.array([classes[i] for i in ll.argmax(axis=1)])

    # --- score 2: JointEnergy = -sum log(1-p)  (higher = more ID) ---
    joint_energy = -np.log(1 - Xtest).sum(axis=1)

    # --- score 3: max_prob (MSP) ---
    max_prob = Xtest.max(axis=1)

    # --- score 4: neg softmax entropy over 4 bits (higher = more ID) ---
    sm = Xtest / Xtest.sum(axis=1, keepdims=True)
    ent = -(sm * np.log(sm + 1e-12)).sum(axis=1)
    neg_entropy = -ent

    scores = {
        'NB log-lik': nb_score,
        'JointEnergy': joint_energy,
        'max_prob': max_prob,
        'neg_entropy': neg_entropy,
    }

    # --- separation table ---
    print(f'{"score":<14} | {"DEF min":>9} | {"DEF 5%":>9} | {"NEG max":>9} | {"NEG 95%":>9} | {"gap":>8} | {"FAR=0 thr":>10}')
    print('-' * 92)
    thr_far0 = {}
    for name, sc in scores.items():
        _, dmin, d5, nmax, n95, gap, thr = sep_report(name, sc, defect_mask, neg_mask, True)
        # FAR=0 threshold = strictly above NEG max
        eps = abs(nmax) * 1e-6 + 1e-9
        thr_far0[name] = nmax + eps
        print(f'{name:<14} | {dmin:>9.3f} | {d5:>9.3f} | {nmax:>9.3f} | {n95:>9.3f} | {gap:>8.3f} | {thr:>10.3f}')

    # --- per-score: bit_F1 at FAR=0 threshold (predict via NB class label) ---
    print()
    print(f'{"approach":<28} | {"bit_F1":>7} | {"FAR%":>6} | {"DEF kept%":>10} | {"NEG rej%":>9}')
    print('-' * 72)

    def eval_accept(accept, label):
        pred = np.where(accept, nb_pred, 'UNKNOWN')
        bf = bit_f1(pred, ytest)
        fa = far(pred, ytest)
        def_kept = 100 * (accept & defect_mask).sum() / max(defect_mask.sum(), 1)
        neg_rej = 100 * ((~accept) & neg_mask).sum() / max(neg_mask.sum(), 1)
        print(f'{label:<28} | {bf:>7.4f} | {fa:>6.2f} | {def_kept:>10.2f} | {neg_rej:>9.2f}')
        return bf, fa

    single_accept = {}
    for name, sc in scores.items():
        acc = sc >= thr_far0[name]
        single_accept[name] = acc
        eval_accept(acc, f'{name} @ FAR=0')

    # --- AND ensemble: accept only if ALL pass FAR=0 thr ---
    and_acc = np.ones(len(ytest), dtype=bool)
    for name in scores:
        and_acc &= single_accept[name]
    eval_accept(and_acc, 'AND-ensemble (all 4)')

    # --- OR ensemble: accept if ANY passes (most permissive) ---
    or_acc = np.zeros(len(ytest), dtype=bool)
    for name in scores:
        or_acc |= single_accept[name]
    eval_accept(or_acc, 'OR-ensemble (any 4)')

    # --- NB+reject reference (best single from prior work) ---
    # already = NB log-lik @ FAR=0 above

    # --- 2-of-best: NB AND JointEnergy (the two with widest gap typically) ---
    eval_accept(single_accept['NB log-lik'] & single_accept['JointEnergy'],
                'NB AND JointEnergy')


if __name__ == '__main__':
    main()
