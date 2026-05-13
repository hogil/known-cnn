"""
5-fold cross-validation threshold evaluation (paper rigor).

Replaces in-sample F1-max threshold tuning. For each model:
1. Split positive group + negative group into 5 folds
2. For each fold f:
   - Compute per-bit threshold on the other 4 folds (training)
   - Predict on fold f (test)
3. Aggregate OOF predictions → bit_F1 + FAR
4. Report mean threshold (across folds) and std (consistency)

Usage:
    python _cv5_threshold_eval.py [glob_pattern]
    python _cv5_threshold_eval.py outputs/W2_pt100_nt30/T7_*/eval_v15direct/stage1_*

Methods supported:
- f1_max: argmax F1 per bit (paper convention)
- youden: argmax (TPR - FPR)
- neg_3sigma: neg_avg + 3*sigma (conservative)
"""
import sys
import json
import ast
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score
from sklearn.model_selection import KFold

BITS = ['bank_boundary', 'fork', 'scratch', 'scratch_rot']
I13_GATE = 0.55


def threshold_method(method, gt_train, pos_probs_train, neg_probs_train):
    """Return threshold for one bit, given train fold data."""
    if method == 'f1_max':
        best_f1 = 0
        best_thr = 0.5
        for t in np.arange(0.05, 0.95, 0.01):
            pred = (pos_probs_train >= t).astype(int)
            f = f1_score(gt_train, pred, zero_division=0)
            if f > best_f1:
                best_f1 = f
                best_thr = t
        return best_thr
    elif method == 'youden':
        pos_only = pos_probs_train[gt_train == 1]
        best_y = -1
        best_thr = 0.5
        for t in np.arange(0.05, 0.95, 0.01):
            tpr = (pos_only >= t).mean() if len(pos_only) else 0
            fpr = (neg_probs_train >= t).mean()
            if tpr - fpr > best_y:
                best_y = tpr - fpr
                best_thr = t
        return best_thr
    elif method == 'neg_3sigma':
        return min(0.99, neg_probs_train.mean() + 3 * neg_probs_train.std())


def cv5_eval(parquet_path, methods=('f1_max', 'youden', 'neg_3sigma'), n_folds=5, seed=42):
    df = pd.read_parquet(parquet_path)
    sub = df[df['cell_id'] == 'T0__I13'].copy()
    sub['true_labels'] = sub['true_labels'].apply(
        lambda s: ast.literal_eval(s) if isinstance(s, str) else list(s)
    )
    sub['n_true'] = sub['true_labels'].apply(len)
    prob_cols = [f'prob_{b}' for b in BITS]
    sub['max_prob'] = sub[prob_cols].max(axis=1)
    sub['i13_gate'] = sub['max_prob'] < I13_GATE

    # Positive group: single defect + 2-combo (excluding scratch+scratch_rot same-family)
    pos_grp = sub[
        ~sub['class_key'].str.contains('Normal|Invalid|CenterDonut|CrossScratch|DiagonalSmear|Starburst|ood_', regex=True)
        & sub['n_true'].isin([1, 2])
    ].copy()
    pos_grp = pos_grp[~pos_grp['class_key'].str.contains(r'scratch\+scratch_rot', regex=True)]

    # Negative group: NI + OOD (excluding 3-combo overlays)
    neg_grp = sub[sub['class_key'].str.contains('Normal|Invalid|CenterDonut|CrossScratch|DiagonalSmear|Starburst', regex=True)
                  & ~sub['class_key'].str.contains('ood_', regex=False)].copy()

    pos_grp = pos_grp.reset_index(drop=True)
    neg_grp = neg_grp.reset_index(drop=True)

    kf = KFold(n_splits=n_folds, shuffle=True, random_state=seed)
    pos_splits = list(kf.split(pos_grp))
    neg_splits = list(kf.split(neg_grp))

    results = {}
    for method in methods:
        oof_preds = {b: [] for b in BITS}
        oof_gts = {b: [] for b in BITS}
        oof_neg_fps = []
        per_fold_thrs = {b: [] for b in BITS}

        for fold_idx in range(n_folds):
            pos_tr_idx, pos_te_idx = pos_splits[fold_idx]
            neg_tr_idx, neg_te_idx = neg_splits[fold_idx]
            pos_tr = pos_grp.iloc[pos_tr_idx]
            pos_te = pos_grp.iloc[pos_te_idx]
            neg_tr = neg_grp.iloc[neg_tr_idx]
            neg_te = neg_grp.iloc[neg_te_idx]

            fold_thrs = {}
            for b in BITS:
                gt_tr = pos_tr['true_labels'].apply(lambda l: int(b in l)).values
                pos_probs_tr = pos_tr[f'prob_{b}'].values
                # Negative probs for this bit: from neg_grp AND pos_grp where bit is not positive
                pos_tr_neg_mask = ~pos_tr['true_labels'].apply(lambda l: b in l)
                neg_probs_combined = np.concatenate([
                    neg_tr[f'prob_{b}'].values,
                    pos_tr.loc[pos_tr_neg_mask, f'prob_{b}'].values,
                ])
                # F1-max uses gt_tr + pos_probs_tr (within pos_grp)
                if method == 'f1_max':
                    thr = threshold_method('f1_max', gt_tr, pos_probs_tr, neg_probs_combined)
                else:
                    thr = threshold_method(method, gt_tr, pos_probs_tr, neg_probs_combined)
                fold_thrs[b] = thr
                per_fold_thrs[b].append(thr)

            # Apply threshold to test fold
            for b in BITS:
                gt_te = pos_te['true_labels'].apply(lambda l: int(b in l)).values
                pred_te = ((pos_te[f'prob_{b}'].values >= fold_thrs[b]) & (~pos_te['i13_gate'].values)).astype(int)
                oof_preds[b].extend(pred_te)
                oof_gts[b].extend(gt_te)

            # FAR on neg test fold
            fp_te = ((neg_te['prob_bank_boundary'] >= fold_thrs['bank_boundary']) |
                     (neg_te['prob_fork'] >= fold_thrs['fork']) |
                     (neg_te['prob_scratch'] >= fold_thrs['scratch']) |
                     (neg_te['prob_scratch_rot'] >= fold_thrs['scratch_rot'])) & ~neg_te['i13_gate']
            oof_neg_fps.extend(fp_te.values)

        # Aggregate
        f1s = [f1_score(oof_gts[b], oof_preds[b], zero_division=0) for b in BITS]
        bit_f1_macro = float(np.mean(f1s))
        far = float(np.mean(oof_neg_fps) * 100)
        thr_mean = {b: float(np.mean(per_fold_thrs[b])) for b in BITS}
        thr_std = {b: float(np.std(per_fold_thrs[b])) for b in BITS}
        per_bit_f1 = {b: float(f1s[i]) for i, b in enumerate(BITS)}
        results[method] = dict(
            bit_F1=bit_f1_macro,
            FAR=far,
            per_bit_F1=per_bit_f1,
            threshold_mean=thr_mean,
            threshold_std=thr_std,
        )
    return results


def main():
    glob = sys.argv[1] if len(sys.argv) > 1 else None
    if glob:
        models = [(Path(p).parts[-4] if len(Path(p).parts) >= 4 else Path(p).name, p) for p in sorted(Path('.').glob(glob))]
    else:
        # Default: top candidates
        models = []
        for name, g in [
            ('iter112', 'outputs/iter112_ep20/T7_*/eval_v15direct_n200_best_model/stage1_*/preds_chip.parquet'),
            ('iter116J', 'outputs/iter116J_g3_ls30/T7_*/eval_v15direct_n200/stage1_*/preds_chip.parquet'),
            ('iter124g', 'outputs/iter124_g_g3_n3/T7_*/eval_v15direct/stage1_*/preds_chip.parquet'),
            ('iter125b', 'outputs/iter125_b_g4_n2/T7_*/eval_v15direct/stage1_*/preds_chip.parquet'),
            ('iter125d', 'outputs/iter125_d_g2_n5/T7_*/eval_v15direct/stage1_*/preds_chip.parquet'),
            ('iter125f', 'outputs/iter125_f_g2_n6/T7_*/eval_v15direct/stage1_*/preds_chip.parquet'),
            ('iter126e', 'outputs/iter126_e_g2_n8/T7_*/eval_v15direct/stage1_*/preds_chip.parquet'),
            ('iter126d', 'outputs/iter126_d_g4_n4/T7_*/eval_v15direct/stage1_*/preds_chip.parquet'),
        ]:
            files = list(Path('.').glob(g))
            if files:
                models.append((name, files[0]))

    print(f'5-fold CV out-of-fold (seed=42, n_folds=5)')
    print(f'{"model":<14} | {"method":<12} | {"bit_F1":>7} | {"FAR":>7} | {"bb thr mean(std)":>20} | {"fk":>12} | {"sc":>12} | {"sr":>12}')
    print('=' * 130)
    all_results = []
    for name, p in models:
        try:
            res = cv5_eval(str(p))
            for m, r in res.items():
                tm = r['threshold_mean']
                ts = r['threshold_std']
                mark = ' *' if (r['bit_F1'] >= 0.99 and r['FAR'] <= 0.5) else ''
                print(f'{name:<14} | {m:<12} | {r["bit_F1"]:.4f} | {r["FAR"]:>6.2f}% | {tm["bank_boundary"]:.2f}({ts["bank_boundary"]:.2f})           | {tm["fork"]:.2f}({ts["fork"]:.2f}) | {tm["scratch"]:.2f}({ts["scratch"]:.2f}) | {tm["scratch_rot"]:.2f}({ts["scratch_rot"]:.2f}){mark}')
                all_results.append((name, m, r))
            print()
        except Exception as e:
            print(f'{name:<14} ERR: {e}')

    # Winner by dual-gate
    print()
    print('=== Dual-gate (bit_F1 >= 0.99 AND FAR <= 0.5%) ===')
    winners = [(n, m, r) for n, m, r in all_results if r['bit_F1'] >= 0.99 and r['FAR'] <= 0.5]
    for n, m, r in sorted(winners, key=lambda x: -x[2]['bit_F1']):
        print(f'  * {n:<14} {m:<12} bit_F1={r["bit_F1"]:.4f}  FAR={r["FAR"]:.2f}%')


if __name__ == '__main__':
    main()
