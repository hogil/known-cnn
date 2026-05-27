"""Per-class within-chip pos-bit vs neg-bit prob margin.

For each DEFECT class, a chip has present bits (pos) and absent bits (neg).
We want: mean(pos bit prob) HIGH, mean(max neg bit prob) LOW → margin wide.
This is the "pos prob vs neg prob 분리" the user wants maximized.

Compare frozen iter116J vs N01 (cmp=1.0) to show why cmp=1.0 collapses the margin.
"""
import sys
import numpy as np
import pandas as pd

TRAIN_BITS = ['bank_boundary', 'fork', 'scratch', 'scratch_rot']
NEG_GROUP = ['Normal', 'Invalid', 'CenterDonut', 'CrossScratch', 'DiagonalSmear', 'Starburst']
COMBO = ['bank_boundary+fork', 'bank_boundary+scratch', 'bank_boundary+scratch_rot',
         'fork+scratch', 'fork+scratch_rot', 'scratch+scratch_rot']
SINGLE = ['bank_boundary', 'fork', 'scratch', 'scratch_rot']
DEFECT = SINGLE + COMBO


def bits_from_key(k):
    return set(p for p in k.split('+') if p in TRAIN_BITS) if isinstance(k, str) else set()


def analyze(path, label):
    df = pd.read_parquet(path)
    cell = 'T0__I10' if 'T0__I10' in df['cell_id'].unique() else df['cell_id'].unique()[0]
    d = df[df['cell_id'] == cell].reset_index(drop=True)
    cols = ['prob_' + b for b in TRAIN_BITS]
    P = d[cols].values
    y = d['class_key'].values

    print(f'\n=== {label} ({cell}) ===')
    print(f'{"class":<26} | {"pos avg":>8} | {"pos min":>8} | {"neg avg":>8} | {"neg max":>8} | {"margin":>8}')
    print('-' * 84)
    all_margins = []
    for cls in DEFECT:
        m = y == cls
        if m.sum() == 0:
            continue
        present = bits_from_key(cls)
        pos_idx = [i for i, b in enumerate(TRAIN_BITS) if b in present]
        neg_idx = [i for i, b in enumerate(TRAIN_BITS) if b not in present]
        pos_p = P[m][:, pos_idx]           # (n, |pos|)
        neg_p = P[m][:, neg_idx]           # (n, |neg|)
        pos_avg = pos_p.mean()
        pos_min = pos_p.min(axis=1).mean()    # worst present bit per chip, averaged
        neg_avg = neg_p.mean()
        neg_max = neg_p.max(axis=1).mean()    # worst absent bit per chip, averaged
        margin = pos_min - neg_max            # the gap we want wide
        all_margins.append(margin)
        print(f'{cls:<26} | {pos_avg:>8.3f} | {pos_min:>8.3f} | {neg_avg:>8.3f} | {neg_max:>8.3f} | {margin:>8.3f}')

    # NEG group: all bits are "neg", we want them all LOW
    print('-' * 84)
    for cls in NEG_GROUP:
        m = y == cls
        if m.sum() == 0:
            continue
        neg_max = P[m].max(axis=1).mean()
        neg_avg = P[m].mean()
        print(f'{cls:<26} | {"-":>8} | {"-":>8} | {neg_avg:>8.3f} | {neg_max:>8.3f} | {"(OOD)":>8}')

    print(f'\n  DEFECT mean within-chip margin (pos_min - neg_max) = {np.mean(all_margins):.3f}')


def main():
    frozen = 'outputs/iter116J_eval_no3fp/eval_260526_055757/preds_chip.parquet'
    analyze(frozen, 'frozen iter116J  cmp=0.25')
    # N01 if available
    import glob
    n01 = glob.glob('outputs/iter116J_nb_limit_*N01*/**/preds_chip.parquet', recursive=True)
    n01 += glob.glob('outputs/*N01*/**/preds_chip.parquet', recursive=True)
    if n01:
        analyze(n01[0], 'N01 cmp=1.0')
    else:
        print('\n(N01 preds parquet not found — frozen only)')


if __name__ == '__main__':
    main()
