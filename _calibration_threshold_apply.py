"""
Calibration threshold protocol (paper rigor — no eval label leak).

Protocol:
1. Compute F1-max threshold on chip_multilabel_v15direct_n1000 (calibration set)
2. Apply that threshold to chip_multilabel_v15direct (held-out test set)
3. Report bit_F1 + Total FAR as TRUE test metric (no in-sample leak)

Usage:
    python _calibration_threshold_apply.py
"""
import json
import ast
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

BITS = ['bank_boundary', 'fork', 'scratch', 'scratch_rot']
I13_GATE = 0.55


def compute_calibration_threshold(calib_parquet, method='f1_max'):
    """Compute per-bit threshold from calibration set."""
    df = pd.read_parquet(calib_parquet)
    sub = df[df['cell_id'] == 'T0__I13'].copy()
    sub['true_labels'] = sub['true_labels'].apply(
        lambda s: ast.literal_eval(s) if isinstance(s, str) else list(s)
    )
    sub['n_true'] = sub['true_labels'].apply(len)
    prob_cols = [f'prob_{b}' for b in BITS]
    sub['max_prob'] = sub[prob_cols].max(axis=1)
    sub['i13_gate'] = sub['max_prob'] < I13_GATE

    pos_grp = sub[
        ~sub['class_key'].str.contains('Normal|Invalid|CenterDonut|CrossScratch|DiagonalSmear|Starburst|ood_', regex=True)
        & sub['n_true'].isin([1, 2])
    ].copy()
    pos_grp = pos_grp[~pos_grp['class_key'].str.contains(r'scratch\+scratch_rot', regex=True)]

    thrs = {}
    for b in BITS:
        gt = pos_grp['true_labels'].apply(lambda l: int(b in l)).values
        probs = pos_grp[f'prob_{b}'].values
        if method == 'f1_max':
            best_f1 = 0
            best_thr = 0.5
            for t in np.arange(0.05, 0.95, 0.01):
                pred = ((probs >= t) & (~pos_grp['i13_gate'].values)).astype(int)
                f = f1_score(gt, pred, zero_division=0)
                if f > best_f1:
                    best_f1 = f
                    best_thr = t
            thrs[b] = best_thr
        elif method == 'youden':
            pos_only = probs[gt == 1]
            neg_in_pos_grp = probs[gt == 0]
            neg_other = sub.loc[~sub['true_labels'].apply(lambda l: b in l), f'prob_{b}'].values
            all_neg = np.concatenate([neg_in_pos_grp, neg_other[:len(neg_in_pos_grp)*2]])  # cap
            best_y = -1
            best_thr = 0.5
            for t in np.arange(0.05, 0.95, 0.01):
                tpr = (pos_only >= t).mean() if len(pos_only) else 0
                fpr = (all_neg >= t).mean()
                if tpr - fpr > best_y:
                    best_y = tpr - fpr
                    best_thr = t
            thrs[b] = best_thr
    return thrs


def apply_threshold_to_test(test_parquet, thresholds):
    """Apply pre-computed threshold to test set, compute metric."""
    df = pd.read_parquet(test_parquet)
    sub = df[df['cell_id'] == 'T0__I13'].copy()
    sub['true_labels'] = sub['true_labels'].apply(
        lambda s: ast.literal_eval(s) if isinstance(s, str) else list(s)
    )
    sub['n_true'] = sub['true_labels'].apply(len)
    prob_cols = [f'prob_{b}' for b in BITS]
    sub['max_prob'] = sub[prob_cols].max(axis=1)
    sub['i13_gate'] = sub['max_prob'] < I13_GATE

    pos_grp = sub[
        ~sub['class_key'].str.contains('Normal|Invalid|CenterDonut|CrossScratch|DiagonalSmear|Starburst|ood_', regex=True)
        & sub['n_true'].isin([1, 2])
    ].copy()
    pos_grp = pos_grp[~pos_grp['class_key'].str.contains(r'scratch\+scratch_rot', regex=True)]

    f1s = []
    for b in BITS:
        gt = pos_grp['true_labels'].apply(lambda l: int(b in l)).values
        pred = ((pos_grp[f'prob_{b}'].values >= thresholds[b]) & (~pos_grp['i13_gate'].values)).astype(int)
        f1s.append(f1_score(gt, pred, zero_division=0))
    bit_f1 = float(np.mean(f1s))

    neg_chips = sub[sub['class_key'].str.contains('Normal|Invalid|CenterDonut|CrossScratch|DiagonalSmear|Starburst', regex=True)
                    & ~sub['class_key'].str.contains('ood_', regex=False)].copy()
    if len(neg_chips):
        fp = ((neg_chips['prob_bank_boundary'] >= thresholds['bank_boundary']) |
              (neg_chips['prob_fork'] >= thresholds['fork']) |
              (neg_chips['prob_scratch'] >= thresholds['scratch']) |
              (neg_chips['prob_scratch_rot'] >= thresholds['scratch_rot'])) & ~neg_chips['i13_gate']
        far = float(fp.mean() * 100)
    else:
        far = 0
    return bit_f1, far


def main():
    MODELS = [
        ('iter112', 'outputs/iter112_ep20/T7_*'),
        ('iter116J', 'outputs/iter116J_g3_ls30/T7_*'),
        ('iter125b (g=4 n=2)', 'outputs/iter125_b_g4_n2/T7_*'),
        ('iter125d (g=2 n=5)', 'outputs/iter125_d_g2_n5/T7_*'),
        ('iter125f (g=2 n=6)', 'outputs/iter125_f_g2_n6/T7_*'),
        ('iter126e (g=2 n=8)', 'outputs/iter126_e_g2_n8/T7_*'),
        ('W2_pt100_nt5', 'outputs/W2_pt100_nt5/T7_*'),
        ('W2_pt100_nt30', 'outputs/W2_pt100_nt30/T7_*'),
    ]
    print(f'{"model":<20} | {"method":<8} | thresholds (bb/fk/sc/sr) | {"bit_F1":>7} | {"FAR":>7}')
    print('=' * 100)
    for name, run_glob in MODELS:
        run_dirs = list(Path('.').glob(run_glob))
        if not run_dirs:
            print(f'{name:<20} no run dir')
            continue
        run = run_dirs[0]
        # Calibration parquet (n1000 with calibration suffix)
        calib_p = list(run.glob('eval_v15direct_n1000_calib/stage1_*/preds_chip.parquet'))
        if not calib_p:
            print(f'{name:<20} no calibration n1000 eval (run _run_calibration_eval.sh first)')
            continue
        # Test parquet (existing v15direct eval, or v15direct_n200)
        test_p = (list(run.glob('eval_v15direct/stage1_*/preds_chip.parquet'))
                  + list(run.glob('eval_v15direct_n200/stage1_*/preds_chip.parquet')))
        if not test_p:
            print(f'{name:<20} no test eval')
            continue
        test_parquet = test_p[0]
        calib_parquet = calib_p[0]

        for method in ['f1_max', 'youden']:
            thrs = compute_calibration_threshold(calib_parquet, method=method)
            bf1, far = apply_threshold_to_test(test_parquet, thrs)
            mark = ' *' if (bf1 >= 0.99 and far <= 0.5) else ''
            print(f'{name:<20} | {method:<8} | {thrs["bank_boundary"]:.2f}/{thrs["fork"]:.2f}/{thrs["scratch"]:.2f}/{thrs["scratch_rot"]:.2f} | {bf1:.4f} | {far:>6.2f}%{mark}')
        print()


if __name__ == '__main__':
    main()
