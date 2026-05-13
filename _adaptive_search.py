"""
Adaptive g=2 search — observe → propose → train → repeat.

Strategy:
1. Read all completed cells (W2 + iterX) from outputs/
2. Compute bit_F1 + FAR + margin for each
3. Find top N cells (Pareto-front on bit_F1 vs FAR)
4. Propose K new cells = neighbors of top + creative variations
5. Dispatch single training (single GPU sequential, ~7 min each)
6. After each cell, re-analyze, propose next

Adaptive heuristics:
- Hill-climb: try (pt±0.05, nt±0.025, n±1) around current best
- Density: if no top cell in (n, pt) area for long, sample mid-axis
- Diversity: 20% random axis exploration
- Asymmetric: try per-bit pt/nt when scalar saturates
- Temperature: try T_train ≠ 1.0 when prob distribution narrow

Usage:
    python _adaptive_search.py --max-cells 30 --budget-hours 4
"""
import argparse
import json
import ast
import subprocess
import sys
import time
import random
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

BITS = ['bank_boundary', 'fork', 'scratch', 'scratch_rot']
EVAL_SET = 'D:/project/data/wm-811k/chip_multilabel_v15direct'
BACKBONE = 'convnextv2_base.fcmae_ft_in22k_in1k_384'
WEIGHTS = f'mega_matrix/weights/{BACKBONE}.pth'


def get_cell_metrics(parquet_path):
    """Compute bit_F1, FAR, margin for a cell."""
    try:
        df = pd.read_parquet(parquet_path)
    except Exception:
        return None
    sub = df[df['cell_id'] == 'T0__I13'].copy()
    sub['true_labels'] = sub['true_labels'].apply(lambda s: ast.literal_eval(s) if isinstance(s, str) else list(s))
    sub['n_true'] = sub['true_labels'].apply(len)
    prob_cols = [f'prob_{b}' for b in BITS]
    sub['max_prob'] = sub[prob_cols].max(axis=1)
    sub['gate'] = sub['max_prob'] < 0.55
    pos_grp = sub[
        ~sub['class_key'].str.contains('Normal|Invalid|CenterDonut|CrossScratch|DiagonalSmear|Starburst|ood_', regex=True)
        & sub['n_true'].isin([1, 2])
    ].copy()
    pos_grp = pos_grp[~pos_grp['class_key'].str.contains(r'scratch\+scratch_rot', regex=True)]
    f1s = []
    pos_means = []
    neg_means = []
    for b in BITS:
        gt = pos_grp['true_labels'].apply(lambda l: int(b in l)).values
        thr_path = parquet_path.parent / 'thresholds.json'
        if not thr_path.exists():
            return None
        thr = json.load(open(thr_path))['T0__I13']['thresholds'][b]
        pred = ((pos_grp[f'prob_{b}'].values >= thr) & (~pos_grp['gate'].values)).astype(int)
        f1s.append(f1_score(gt, pred, zero_division=0))
        pos = pos_grp.loc[pos_grp['true_labels'].apply(lambda l: b in l), f'prob_{b}'].values
        neg = sub.loc[~sub['true_labels'].apply(lambda l: b in l), f'prob_{b}'].values
        if len(pos): pos_means.append(pos.mean())
        if len(neg): neg_means.append(neg.mean())
    neg_chips = sub[sub['class_key'].str.contains('Normal|Invalid|CenterDonut|CrossScratch|DiagonalSmear|Starburst', regex=True)
                    & ~sub['class_key'].str.contains('ood_', regex=False)].copy()
    if len(neg_chips):
        thr_data = json.load(open(parquet_path.parent / 'thresholds.json'))['T0__I13']['thresholds']
        fp = ((neg_chips['prob_bank_boundary'] >= thr_data['bank_boundary']) | (neg_chips['prob_fork'] >= thr_data['fork']) | (neg_chips['prob_scratch'] >= thr_data['scratch']) | (neg_chips['prob_scratch_rot'] >= thr_data['scratch_rot'])) & ~neg_chips['gate']
        far = fp.mean() * 100
    else:
        far = 0
    return dict(bit_F1=float(np.mean(f1s)), FAR=float(far),
                margin=float(np.mean(pos_means) - np.mean(neg_means)) if pos_means and neg_means else 0)


def parse_w2_cell_name(tag):
    """W2_pt100_nt5 -> (n=8, pt=1.0, nt=0.05)."""
    import re
    m = re.match(r'W2_pt(\d+)_nt(\d+)', tag)
    if m:
        return dict(n=8, pt=int(m.group(1))/100, nt=int(m.group(2))/100)
    m = re.match(r'W3_n(\d+)_pt(\d+)_nt(\d+)', tag)
    if m:
        return dict(n=int(m.group(1)), pt=int(m.group(2))/100, nt=int(m.group(3))/100)
    return None


def gather_state():
    """Read all cells, return list of (tag, params, metrics)."""
    state = []
    for run_dir in sorted(Path('outputs').glob('W[23]_*')):
        params = parse_w2_cell_name(run_dir.name)
        if not params: continue
        s1 = list(run_dir.glob('T*/eval_v15direct/stage1_*/preds_chip.parquet'))
        if not s1: continue
        m = get_cell_metrics(s1[0])
        if m:
            state.append((run_dir.name, params, m))
    return state


def pick_top(state, top_k=3, criterion='bit_F1', far_max=1.0):
    """Top cells by bit_F1 with FAR <= far_max."""
    qual = [s for s in state if s[2]['FAR'] <= far_max]
    qual.sort(key=lambda x: -x[2][criterion])
    return qual[:top_k]


def propose_neighbors(state, n_propose=5):
    """Propose neighbor cells around top picks + creative variations."""
    proposals = []
    tops = pick_top(state, top_k=2, far_max=1.0)
    if not tops:
        # bootstrap: try uniform grid
        for n in [2, 4, 6]:
            for pt in [1.0, 0.9]:
                for nt in [0.05, 0.15]:
                    proposals.append(dict(n=n, pt=pt, nt=nt))
        return proposals[:n_propose]

    done_tags = set(s[0] for s in state)
    seen_params = set((s[1]['n'], s[1]['pt'], s[1]['nt']) for s in state)

    for name, p, m in tops:
        # Hill-climb neighbors
        for dn in [-2, -1, 1, 2]:
            new_n = p['n'] + dn
            if 1 <= new_n <= 10 and (new_n, p['pt'], p['nt']) not in seen_params:
                proposals.append(dict(n=new_n, pt=p['pt'], nt=p['nt']))
        for dpt in [-0.025, -0.05, 0.025, 0.05]:
            new_pt = round(p['pt'] + dpt, 3)
            if 0.7 <= new_pt <= 1.0 and (p['n'], new_pt, p['nt']) not in seen_params:
                proposals.append(dict(n=p['n'], pt=new_pt, nt=p['nt']))
        for dnt in [-0.025, 0.025, -0.05, 0.05]:
            new_nt = round(p['nt'] + dnt, 3)
            if 0 <= new_nt <= 0.3 and (p['n'], p['pt'], new_nt) not in seen_params:
                proposals.append(dict(n=p['n'], pt=p['pt'], nt=new_nt))

    # diversity: random axis exploration (20%)
    for _ in range(2):
        n = random.choice([1, 2, 3, 4, 5, 6, 7, 8])
        pt = random.choice([0.7, 0.8, 0.85, 0.9, 0.95, 1.0])
        nt = random.choice([0.05, 0.10, 0.15, 0.20, 0.25])
        if (n, pt, nt) not in seen_params:
            proposals.append(dict(n=n, pt=pt, nt=nt))

    # dedup and sort by margin proxy
    unique = []
    seen_propose = set()
    for p in proposals:
        key = (p['n'], p['pt'], p['nt'])
        if key in seen_propose: continue
        if key in seen_params: continue
        seen_propose.add(key)
        unique.append(p)
    random.shuffle(unique)
    return unique[:n_propose]


def dispatch_cell(params):
    """Train + eval a single cell. Blocking ~7 min."""
    n, pt, nt = params['n'], params['pt'], params['nt']
    tag = f"W3_n{n}_pt{int(pt*100)}_nt{int(nt*100)}"
    out_root = f"outputs/{tag}"
    if Path(out_root).exists():
        runs = list(Path(out_root).glob('T*/best_model.pth'))
        if runs:
            print(f"[skip] {tag} (already done)")
            return tag
        else:
            subprocess.run(['rm', '-rf', out_root], check=False)
    print(f"[adaptive] dispatch {tag} (n={n}, pt={pt}, nt={nt})")
    grid_dim = 2 * n
    cmd = [
        'python', '-X', 'utf8', '-m', 'chip_multilabel._train_chip_variant',
        '--variant', 'T7', '--pos-target', str(pt), '--neg-target', str(nt),
        '--batch', '2', '--accum', '8', '--seed', '1', '--epochs', '10', '--lr', '1e-4',
        '--backbone-timm', BACKBONE, '--img-size', '384',
        '--backbone-timm-weights', WEIGHTS,
        '--no-normal', '--val-criterion', 'margin_max', '--save-every-epoch',
        '--cutmix-mode', 'complement', '--cutmix-pair', 'masked', '--cutmix-pair-fill', 'corner',
        '--cutmix-p', '0.25', '--cutmix-grid-dim', str(grid_dim), '--cutmix-n-groups', '2',
        '--cutmix-complete-label-scale', '0.5',
        '--tag', tag, '--out-root', out_root,
    ]
    train_log = open(f'outputs/_{tag}_train.log', 'w')
    subprocess.run(cmd, stdout=train_log, stderr=subprocess.STDOUT, check=False)
    train_log.close()
    runs = list(Path(out_root).glob('T*/best_model.pth'))
    if not runs:
        print(f"[FAIL] {tag} train failed")
        return None
    # Eval
    run_dir = runs[0].parent
    eval_out = f"{run_dir}/eval_v15direct"
    if not Path(eval_out).exists():
        cmd_eval = [
            'python', '-X', 'utf8', '-m', 'chip_multilabel.run_stage1',
            '--model', str(runs[0]), '--eval-set', EVAL_SET, '--out-root', eval_out,
            '--variants', 'I3,I7,I10,I13', '--n-per-class', '200',
            '--strength-min', '0.0', '--strength-max', '1.0', '--seed', '42',
        ]
        eval_log = open(f'outputs/_{tag}_eval.log', 'w')
        subprocess.run(cmd_eval, stdout=eval_log, stderr=subprocess.STDOUT, check=False)
        eval_log.close()
    # Cleanup epoch ckpts
    for ep in run_dir.glob('epoch_*.pth'):
        ep.unlink()
    return tag


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--max-cells', type=int, default=30, help='max cells to add this run')
    ap.add_argument('--budget-hours', type=float, default=4.0, help='time budget in hours')
    args = ap.parse_args()

    t_start = time.time()
    added = 0
    while added < args.max_cells:
        elapsed = (time.time() - t_start) / 3600
        if elapsed >= args.budget_hours:
            print(f'[adaptive] budget {args.budget_hours} hr reached after {added} cells')
            break

        state = gather_state()
        print(f'[adaptive] state: {len(state)} cells in pool')
        if state:
            tops = pick_top(state, top_k=3, far_max=1.0)
            for n, p, m in tops:
                print(f'  top: {n:<22} bit_F1={m["bit_F1"]:.4f} FAR={m["FAR"]:.2f}% margin={m["margin"]:.3f}')
        proposals = propose_neighbors(state, n_propose=3)
        if not proposals:
            print('[adaptive] no new proposals, exiting')
            break
        for params in proposals:
            if added >= args.max_cells: break
            tag = dispatch_cell(params)
            if tag: added += 1
            elapsed = (time.time() - t_start) / 3600
            if elapsed >= args.budget_hours:
                break

    print(f'[adaptive] done. added {added} cells in {elapsed:.2f} hr')


if __name__ == '__main__':
    main()
