"""
Compute margin (pos_prob_mean - neg_prob_mean) + bit_F1 + Total FAR per cell.

Usage:
    python _compute_margin_table.py outputs/W1_*

Output columns:
    cell | bit_F1 | Total FAR | NI FAR | OOD FAR | pos_mean | neg_mean | margin
"""
import sys
import glob
import json
import ast
import re
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

BITS = ['bank_boundary', 'fork', 'scratch', 'scratch_rot']
I13_GATE = 0.55


def compute(parquet, thr_json):
    df = pd.read_parquet(parquet)
    thr = json.load(open(thr_json))
    sub = df[df['cell_id'] == 'T0__I13'].copy()
    if len(sub) == 0:
        return None
    sub['true_labels'] = sub['true_labels'].apply(
        lambda s: ast.literal_eval(s) if isinstance(s, str) else list(s)
    )
    sub['n_true'] = sub['true_labels'].apply(len)
    t = thr['T0__I13']['thresholds']
    prob_cols = [f'prob_{b}' for b in BITS]
    sub['max_prob'] = sub[prob_cols].max(axis=1)
    sub['i13_gate_normal'] = sub['max_prob'] < I13_GATE

    pos = sub[
        ~sub['class_key'].str.contains('Normal|Invalid|CenterDonut|CrossScratch|DiagonalSmear|Starburst|ood_', regex=True)
        & sub['n_true'].isin([1, 2])
    ].copy()
    pos = pos[~pos['class_key'].str.contains(r'scratch\+scratch_rot|\+scratch_rot\+scratch', regex=True)]

    # Margin: per chip, mean(prob[positive bits]) - max(prob[negative bits])
    pos_means = []
    neg_means = []
    for _, row in pos.iterrows():
        true = row['true_labels']
        pos_probs = [row[f'prob_{b}'] for b in BITS if b in true]
        neg_probs = [row[f'prob_{b}'] for b in BITS if b not in true]
        if pos_probs:
            pos_means.append(np.mean(pos_probs))
        if neg_probs:
            neg_means.append(np.max(neg_probs))
    pos_mean = float(np.mean(pos_means)) if pos_means else 0.0
    neg_mean = float(np.mean(neg_means)) if neg_means else 0.0
    margin = pos_mean - neg_mean

    f1s = []
    for b in BITS:
        gt = pos['true_labels'].apply(lambda l: int(b in l)).values
        pred = ((pos[f'prob_{b}'].values >= t[b]) & (~pos['i13_gate_normal'].values)).astype(int)
        f1s.append(f1_score(gt, pred, zero_division=0))
    bit_f1 = float(np.mean(f1s))

    neg = sub[sub['class_key'].str.contains('Normal|Invalid|CenterDonut|CrossScratch|DiagonalSmear|Starburst', regex=True)].copy()
    neg = neg[~neg['class_key'].str.contains('ood_', regex=False)]
    if len(neg):
        fp = (((neg['prob_bank_boundary'] >= t['bank_boundary']) |
               (neg['prob_fork'] >= t['fork']) |
               (neg['prob_scratch'] >= t['scratch']) |
               (neg['prob_scratch_rot'] >= t['scratch_rot'])) &
              (~neg['i13_gate_normal'])).astype(int)
        total_far = fp.mean() * 100
        ni = neg[neg['class_key'].str.contains('Normal|Invalid')]
        ood = neg[~neg['class_key'].str.contains('Normal|Invalid')]
        ni_fp = (((ni['prob_bank_boundary'] >= t['bank_boundary']) |
                  (ni['prob_fork'] >= t['fork']) |
                  (ni['prob_scratch'] >= t['scratch']) |
                  (ni['prob_scratch_rot'] >= t['scratch_rot'])) &
                 (~ni['i13_gate_normal'])).astype(int) if len(ni) else None
        ood_fp = (((ood['prob_bank_boundary'] >= t['bank_boundary']) |
                   (ood['prob_fork'] >= t['fork']) |
                   (ood['prob_scratch'] >= t['scratch']) |
                   (ood['prob_scratch_rot'] >= t['scratch_rot'])) &
                  (~ood['i13_gate_normal'])).astype(int) if len(ood) else None
        ni_far = ni_fp.mean() * 100 if ni_fp is not None else 0.0
        ood_far = ood_fp.mean() * 100 if ood_fp is not None else 0.0
    else:
        total_far = ni_far = ood_far = 0.0

    return dict(bit_F1=bit_f1, Total_FAR=total_far, NI_FAR=ni_far, OOD_FAR=ood_far,
                pos_mean=pos_mean, neg_mean=neg_mean, margin=margin)


def main():
    pattern = sys.argv[1] if len(sys.argv) > 1 else 'outputs/W1_*'
    rows = []
    # Direct glob OR pattern with /T*/eval_*/stage1_*
    candidates = []
    for run_dir in sorted(glob.glob(pattern)):
        # find stage1 dir
        s1s = list(Path(run_dir).glob('T*/eval_*/stage1_*')) + list(Path(run_dir).glob('T*/eval_*/stage1_*/.'))
        for s1 in s1s:
            p = s1 / 'preds_chip.parquet'
            t = s1 / 'thresholds.json'
            if p.exists() and t.exists():
                candidates.append((run_dir, s1, p, t))
                break  # only first eval dir per run

    print(f'{"cell":<20} {"bit_F1":>7} {"Total":>8} {"NI":>6} {"OOD":>7} {"pos":>6} {"neg":>6} {"margin":>7}')
    print('-' * 80)
    for run_dir, s1, p, t in candidates:
        cell_name = Path(run_dir).name.replace('iter124_', '').replace('iter125_', '').replace('iter126_', '')
        try:
            res = compute(p, t)
            if res is None:
                continue
            rows.append((cell_name, res))
            print(f'{cell_name:<20} {res["bit_F1"]:>7.4f} {res["Total_FAR"]:>7.2f}% {res["NI_FAR"]:>5.2f}% {res["OOD_FAR"]:>6.2f}% '
                  f'{res["pos_mean"]:>6.3f} {res["neg_mean"]:>6.3f} {res["margin"]:>+7.3f}')
        except Exception as e:
            print(f'{cell_name:<20} ERR {e}')

    if rows:
        print()
        print('=== Top 5 by margin ===')
        rows.sort(key=lambda x: -x[1]['margin'])
        for name, r in rows[:5]:
            print(f'  {name:<20} margin={r["margin"]:+.3f}  bit_F1={r["bit_F1"]:.4f}  FAR={r["Total_FAR"]:.2f}%')
        print('=== Top 5 by bit_F1 (FAR <= 1%) ===')
        qual = [(n, r) for (n, r) in rows if r['Total_FAR'] <= 1.0]
        qual.sort(key=lambda x: -x[1]['bit_F1'])
        for name, r in qual[:5]:
            print(f'  {name:<20} bit_F1={r["bit_F1"]:.4f}  FAR={r["Total_FAR"]:.2f}%  margin={r["margin"]:+.3f}')


if __name__ == '__main__':
    main()
