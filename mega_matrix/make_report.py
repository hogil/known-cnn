"""
Generate summary_mega_sweep.md + plots from mega matrix outputs.

Reads outputs/_mega_matrix/model_*/T*/eval_*/stage1_*/preds_chip.parquet
Writes docs/chip-multilabel/manager_report/summary_mega_sweep.md + .png plots
"""
import json
import glob
import ast
import math
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score


BITS = ['bank_boundary', 'fork', 'scratch', 'scratch_rot']
SHORT = {'bank_boundary': 'bb', 'fork': 'fk', 'scratch': 'sc', 'scratch_rot': 'sr'}

OUT_BASE = Path("outputs/_mega_matrix")
REPORT_DIR = Path("docs/chip-multilabel/manager_report")
REPORT_DIR.mkdir(parents=True, exist_ok=True)
FIG_DIR = REPORT_DIR / "figs_mega"
FIG_DIR.mkdir(parents=True, exist_ok=True)

TRAIN_NS = [50, 100, 200]
EVAL_NS = ['200', '2000', 'max']
SELS = ['f1', 'margin_max']


def get_group(row):
    ck = row['class_key']
    n = row['n_true']
    if 'Normal' in ck: return 'Normal'
    if 'Invalid' in ck: return 'Invalid'
    if 'ood_' in ck or ck in ['CenterDonut', 'CrossScratch', 'DiagonalSmear', 'Starburst']:
        return f'OOD_{ck}'
    if n == 1: return 'single'
    if n == 2: return '2-combo'
    return '3-combo'


def metric_one_eval(eval_dir, cell='T0__I13'):
    """Return dict with bit_F1, Total FAR, per-class F1, per-group prob stats."""
    pchip = Path(eval_dir) / 'preds_chip.parquet'
    thr_path = Path(eval_dir) / 'thresholds.json'
    if not pchip.exists() or not thr_path.exists():
        return None
    df = pd.read_parquet(pchip)
    sub = df[df['cell_id'] == cell].copy()
    if len(sub) == 0:
        return None
    sub['true_labels'] = sub['true_labels'].apply(lambda s: ast.literal_eval(s) if isinstance(s, str) else list(s))
    sub['pred_labels'] = sub['pred_labels'].apply(lambda s: ast.literal_eval(s) if isinstance(s, str) else list(s))
    sub['n_true'] = sub['true_labels'].apply(len)
    sub['group'] = sub.apply(get_group, axis=1)

    pos = sub[sub['group'].isin(['single', '2-combo'])]
    neg = sub[sub['group'].apply(lambda g: g in ['Normal', 'Invalid'] or g.startswith('OOD_'))]

    thr = json.load(open(thr_path))[cell]['thresholds']

    # Per-bit F1 (positive group, absolute rule)
    per_bit = {}
    f1_list = []
    for b in BITS:
        if len(pos) == 0:
            per_bit[b] = float('nan')
            continue
        gt = pos['true_labels'].apply(lambda l: int(b in l)).values
        pred = (pos[f'prob_{b}'].values >= thr[b]).astype(int)
        per_bit[b] = float(f1_score(gt, pred, zero_division=0))
        f1_list.append(per_bit[b])
    bit_F1 = float(np.mean(f1_list)) if f1_list else float('nan')

    # Total FAR (negative groups)
    fp = int(neg['pred_labels'].apply(lambda x: len(x) > 0).sum()) if len(neg) else 0
    far = fp / max(len(neg), 1)

    # Per-OOD class FAR
    ood_far = {}
    for grp in sub['group'].unique():
        if grp.startswith('OOD_') or grp in ['Normal', 'Invalid']:
            ssub = sub[sub['group'] == grp]
            fp_g = int(ssub['pred_labels'].apply(lambda x: len(x) > 0).sum())
            ood_far[grp] = (fp_g, len(ssub), fp_g / max(len(ssub), 1))

    # Per-group active/inactive prob stats
    prob_stats = {}
    for grp in sorted(sub['group'].unique()):
        ssub = sub[sub['group'] == grp]
        stat = {'n': len(ssub)}
        for b in BITS:
            active = ssub['true_labels'].apply(lambda l: b in l).values
            active_probs = ssub[f'prob_{b}'].values[active]
            inactive_probs = ssub[f'prob_{b}'].values[~active]
            stat[f'{SHORT[b]}_pos'] = (float(active_probs.mean()), float(active_probs.std())) if len(active_probs) > 0 else None
            stat[f'{SHORT[b]}_neg'] = (float(inactive_probs.mean()), float(inactive_probs.std())) if len(inactive_probs) > 0 else None
        prob_stats[grp] = stat

    return {
        'bit_F1': bit_F1,
        'total_far': far,
        'per_bit_F1': per_bit,
        'ood_far': ood_far,
        'prob_stats': prob_stats,
        'thresholds': thr,
        'n_pos': len(pos),
        'n_neg': len(neg),
    }


def collect_all():
    """Iterate over the 18-cell matrix and collect metrics."""
    rows = []
    for tn in TRAIN_NS:
        for sel in SELS:
            for en in EVAL_NS:
                model_root = OUT_BASE / f"model_train{tn}_{sel}"
                runs = list(model_root.glob("T*/"))
                if not runs:
                    continue
                run = runs[0]
                eval_dirs = list(run.glob(f"eval_{en}/stage1_*"))
                if not eval_dirs:
                    continue
                eval_dir = eval_dirs[0]
                m = metric_one_eval(eval_dir)
                if m is None:
                    continue
                rows.append({
                    'train_n': tn,
                    'selection': sel,
                    'eval_n': en,
                    **m,
                })
    return rows


def write_table_md(rows, fpath):
    out = []
    out.append("# Mega Matrix Sweep — `train {50/100/200}` × `eval {200/2000/MAX}` × `selection {val_f1, val_margin}`\n")
    out.append("Run date: 2026-05-13\n")
    out.append("\n")

    # ====== Section A: Method explanation ======
    out.append("## 1. Selection criteria — `val_f1` vs `val_margin`\n")
    out.append("""
### val_f1 (legacy)
- Per-bit BCE F1, macro-averaged across 4 defect bits
- Threshold = 0.5 fixed
- Saturates on small val (only 3 reachable values: 0.9818 / 0.9847 / 0.9907)
- → coin-flip best_model selection

### val_margin (NEW, paper §3 contribution)
- `margin = mean(prob[positive bits]) - max(prob[negative bits])` per chip, averaged
- Decision boundary sharpness metric (Sharma & Bilen 2024 style)
- Anti-saturation: continuous spectrum, no plateau
- Multi-label friendly: directly rewards `well-separated active vs inactive`

### Why margin > f1 for multi-label selection?
1. **Continuous resolution**: val_f1 has 3 reachable values on small val → ties broken by epoch order. val_margin spans full [0, 1].
2. **Boundary-aware**: large margin = pos and neg well-separated → robust deployment with threshold/gate.
3. **OOD-friendly side effect**: low neg_prob in margin → low max_prob in OOD chips → low FAR.
4. **Empirical**: Spearman ρ vs eval bit_F1 = +0.56 (margin) vs −0.10 (f1) on 35-ckpt audit.

### Side effects
1. Easy chip preference → boundary cases under-learned (mitigated by FCM-PM)
2. Per-class imbalance ignored (use weight=1 average)
3. Conservative selection (ep6 not ep20) — leaves some F1 on table
4. Gate threshold coupling — paper deployment must report gate value
""")

    # ====== Section B: Main results table ======
    out.append("\n## 2. Main results matrix\n")
    out.append("\n**Metric definitions**:\n")
    out.append("- `bit_F1` = absolute rule: macro-F1 over (single + 2-combo) positive bits (threshold per-bit auto-tuned)\n")
    out.append("- `Total FAR` = (Normal + Invalid + OOD) chips with non-empty pred_labels / total negatives\n")
    out.append("- Cell `T0__I13` selected (entropy gate + Invalid heuristic, paper SOTA cell)\n\n")

    out.append("\n| train n | selection | eval n | n_pos | n_neg | bit_F1 | Total FAR | bb F1 | fork F1 | sc F1 | sr F1 |\n")
    out.append("|---|---|---|---|---|---|---|---|---|---|---|\n")
    for r in rows:
        out.append(
            f"| {r['train_n']} | {r['selection']} | {r['eval_n']} | "
            f"{r['n_pos']} | {r['n_neg']} | "
            f"**{r['bit_F1']:.4f}** | **{r['total_far']*100:.2f}%** | "
            f"{r['per_bit_F1']['bank_boundary']:.4f} | "
            f"{r['per_bit_F1']['fork']:.4f} | "
            f"{r['per_bit_F1']['scratch']:.4f} | "
            f"{r['per_bit_F1']['scratch_rot']:.4f} |\n"
        )

    # ====== Section C: val_f1 vs val_margin side-by-side ======
    out.append("\n## 3. `val_f1` vs `val_margin` direct comparison (per train_n, fixed eval=2000)\n\n")
    out.append("\n| train_n | bit_F1 (val_f1) | bit_F1 (val_margin) | Δ | FAR (val_f1) | FAR (val_margin) | Δ FAR |\n")
    out.append("|---|---|---|---|---|---|---|\n")
    for tn in TRAIN_NS:
        r_f1 = next((r for r in rows if r['train_n'] == tn and r['selection'] == 'f1' and r['eval_n'] == '2000'), None)
        r_mg = next((r for r in rows if r['train_n'] == tn and r['selection'] == 'margin_max' and r['eval_n'] == '2000'), None)
        if r_f1 and r_mg:
            d_bit = r_mg['bit_F1'] - r_f1['bit_F1']
            d_far = r_mg['total_far'] - r_f1['total_far']
            out.append(
                f"| {tn} | {r_f1['bit_F1']:.4f} | {r_mg['bit_F1']:.4f} | "
                f"{d_bit:+.4f} | {r_f1['total_far']*100:.2f}% | "
                f"{r_mg['total_far']*100:.2f}% | {d_far*100:+.2f}% |\n"
            )

    # ====== Section D: Per-OOD class FAR breakdown ======
    out.append("\n## 4. Per-class FAR breakdown (val_margin, eval=2000)\n\n")
    out.append("\n| train_n | Normal | Invalid | CenterDonut | CrossScratch | DiagonalSmear | Starburst |\n")
    out.append("|---|---|---|---|---|---|---|\n")
    for tn in TRAIN_NS:
        r = next((r for r in rows if r['train_n'] == tn and r['selection'] == 'margin_max' and r['eval_n'] == '2000'), None)
        if not r:
            continue
        cells = []
        for grp in ['Normal', 'Invalid', 'OOD_CenterDonut', 'OOD_CrossScratch', 'OOD_DiagonalSmear', 'OOD_Starburst']:
            if grp in r['ood_far']:
                fp, n, rate = r['ood_far'][grp]
                cells.append(f"{fp}/{n} ({rate*100:.1f}%)")
            else:
                cells.append("N/A")
        out.append(f"| {tn} | " + " | ".join(cells) + " |\n")

    # ====== Section E: Per-defect class detail ======
    out.append("\n## 5. Per-defect group prob distribution (val_margin, train=200, eval=2000)\n\n")
    r_main = next((r for r in rows if r['train_n'] == 200 and r['selection'] == 'margin_max' and r['eval_n'] == '2000'), None)
    if r_main:
        ps = r_main['prob_stats']
        out.append("| group | n | bb_pos | bb_neg | fk_pos | fk_neg | sc_pos | sc_neg | sr_pos | sr_neg |\n")
        out.append("|---|---|---|---|---|---|---|---|---|---|\n")
        for g in ['single', '2-combo', '3-combo', 'Normal', 'Invalid',
                   'OOD_CenterDonut', 'OOD_CrossScratch', 'OOD_DiagonalSmear', 'OOD_Starburst']:
            if g not in ps:
                continue
            s = ps[g]
            def fmt(key):
                v = s.get(key)
                return f"{v[0]:.2f}±{v[1]:.2f}" if v else "N/A"
            out.append(f"| {g} | {s['n']} | {fmt('bb_pos')} | {fmt('bb_neg')} | "
                       f"{fmt('fk_pos')} | {fmt('fk_neg')} | "
                       f"{fmt('sc_pos')} | {fmt('sc_neg')} | "
                       f"{fmt('sr_pos')} | {fmt('sr_neg')} |\n")

    # ====== Section F: Analysis ======
    out.append("\n## 6. Analysis\n\n")
    out.append("### 6.1 Selection criterion impact\n")
    if rows:
        f1_avg_bit = np.mean([r['bit_F1'] for r in rows if r['selection'] == 'f1'])
        mg_avg_bit = np.mean([r['bit_F1'] for r in rows if r['selection'] == 'margin_max'])
        f1_avg_far = np.mean([r['total_far'] for r in rows if r['selection'] == 'f1'])
        mg_avg_far = np.mean([r['total_far'] for r in rows if r['selection'] == 'margin_max'])
        out.append(f"- avg bit_F1: val_f1={f1_avg_bit:.4f} vs val_margin={mg_avg_bit:.4f} ({(mg_avg_bit - f1_avg_bit):+.4f})\n")
        out.append(f"- avg Total FAR: val_f1={f1_avg_far*100:.2f}% vs val_margin={mg_avg_far*100:.2f}% ({(mg_avg_far - f1_avg_far)*100:+.2f}%)\n")

    out.append("\n### 6.2 Train data scaling\n")
    for sel in SELS:
        out.append(f"\n**{sel}**:\n")
        for tn in TRAIN_NS:
            rs = [r for r in rows if r['train_n'] == tn and r['selection'] == sel]
            if rs:
                avg_bit = np.mean([r['bit_F1'] for r in rs])
                avg_far = np.mean([r['total_far'] for r in rs])
                out.append(f"  - train={tn}: bit_F1={avg_bit:.4f}, FAR={avg_far*100:.2f}%\n")

    out.append("\n### 6.3 Eval size impact (statistical power)\n")
    for en in EVAL_NS:
        rs = [r for r in rows if r['eval_n'] == en]
        if rs:
            avg_bit = np.mean([r['bit_F1'] for r in rs])
            out.append(f"  - eval={en}: avg bit_F1={avg_bit:.4f} (n_pos={rs[0]['n_pos']}, n_neg={rs[0]['n_neg']})\n")

    # ====== Section G: Plots ======
    out.append("\n## 7. Plots\n\n")

    # Plot 1: bit_F1 heatmap (3 train × 3 eval, separate per selection)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for ax, sel in zip(axes, SELS):
        mat = np.zeros((len(TRAIN_NS), len(EVAL_NS)))
        for i, tn in enumerate(TRAIN_NS):
            for j, en in enumerate(EVAL_NS):
                r = next((r for r in rows if r['train_n'] == tn and r['selection'] == sel and r['eval_n'] == en), None)
                mat[i, j] = r['bit_F1'] if r else np.nan
        im = ax.imshow(mat, cmap='viridis', vmin=0.85, vmax=1.0, aspect='auto')
        ax.set_xticks(range(len(EVAL_NS)))
        ax.set_xticklabels([f'eval={e}' for e in EVAL_NS])
        ax.set_yticks(range(len(TRAIN_NS)))
        ax.set_yticklabels([f'train={t}' for t in TRAIN_NS])
        ax.set_title(f'bit_F1 ({sel})')
        for i in range(len(TRAIN_NS)):
            for j in range(len(EVAL_NS)):
                if not np.isnan(mat[i, j]):
                    ax.text(j, i, f'{mat[i, j]:.4f}', ha='center', va='center', color='white', fontsize=9)
        plt.colorbar(im, ax=ax)
    plt.tight_layout()
    plt.savefig(FIG_DIR / 'bit_F1_heatmap.png', dpi=120)
    plt.close()
    out.append("### Plot 1 — bit_F1 heatmap (train × eval)\n\n")
    out.append("![bit_F1 heatmap](figs_mega/bit_F1_heatmap.png)\n\n")

    # Plot 2: Total FAR heatmap
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for ax, sel in zip(axes, SELS):
        mat = np.zeros((len(TRAIN_NS), len(EVAL_NS)))
        for i, tn in enumerate(TRAIN_NS):
            for j, en in enumerate(EVAL_NS):
                r = next((r for r in rows if r['train_n'] == tn and r['selection'] == sel and r['eval_n'] == en), None)
                mat[i, j] = r['total_far'] * 100 if r else np.nan
        im = ax.imshow(mat, cmap='Reds', vmin=0, vmax=5, aspect='auto')
        ax.set_xticks(range(len(EVAL_NS)))
        ax.set_xticklabels([f'eval={e}' for e in EVAL_NS])
        ax.set_yticks(range(len(TRAIN_NS)))
        ax.set_yticklabels([f'train={t}' for t in TRAIN_NS])
        ax.set_title(f'Total FAR % ({sel})')
        for i in range(len(TRAIN_NS)):
            for j in range(len(EVAL_NS)):
                if not np.isnan(mat[i, j]):
                    ax.text(j, i, f'{mat[i, j]:.2f}%', ha='center', va='center', fontsize=9)
        plt.colorbar(im, ax=ax)
    plt.tight_layout()
    plt.savefig(FIG_DIR / 'total_far_heatmap.png', dpi=120)
    plt.close()
    out.append("### Plot 2 — Total FAR heatmap (train × eval)\n\n")
    out.append("![FAR heatmap](figs_mega/total_far_heatmap.png)\n\n")

    # Plot 3: train_n scaling (bit_F1 + FAR per selection)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    for sel in SELS:
        for en in EVAL_NS:
            xs = []
            ys_bit = []
            ys_far = []
            for tn in TRAIN_NS:
                r = next((r for r in rows if r['train_n'] == tn and r['selection'] == sel and r['eval_n'] == en), None)
                if r:
                    xs.append(tn)
                    ys_bit.append(r['bit_F1'])
                    ys_far.append(r['total_far'] * 100)
            if xs:
                lbl = f'{sel} eval={en}'
                ax1.plot(xs, ys_bit, marker='o', label=lbl)
                ax2.plot(xs, ys_far, marker='o', label=lbl)
    ax1.set_xlabel('train_n per class')
    ax1.set_ylabel('bit_F1')
    ax1.set_title('bit_F1 vs train data size')
    ax1.legend(fontsize=8)
    ax1.grid(True, alpha=0.3)
    ax2.set_xlabel('train_n per class')
    ax2.set_ylabel('Total FAR (%)')
    ax2.set_title('Total FAR vs train data size')
    ax2.legend(fontsize=8)
    ax2.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(FIG_DIR / 'scaling_curves.png', dpi=120)
    plt.close()
    out.append("### Plot 3 — Train size scaling curves\n\n")
    out.append("![scaling curves](figs_mega/scaling_curves.png)\n\n")

    # ====== Footer ======
    out.append("\n## 8. Recipe (all cells)\n\n")
    out.append("```\n")
    out.append("backbone: convnextv2_base.fcmae_ft_in22k_in1k_384\n")
    out.append("variant T7 (BCE+LS=0.30)  epochs 10  batch 2 accum 8 seed 1\n")
    out.append("lr 1e-4  --no-normal  --save-every-epoch\n")
    out.append("cutmix-mode complement  pair masked  fill corner\n")
    out.append("cutmix-p 0.25  n-groups 3  complete-label-scale 0.5\n")
    out.append("selection: val_f1 or val_margin (4 selection criterion)\n")
    out.append("```\n")

    with open(fpath, 'w', encoding='utf-8') as f:
        f.writelines(out)
    return out


def main():
    rows = collect_all()
    print(f"Collected {len(rows)} eval results")
    if not rows:
        print("ERROR: no eval results found.")
        return
    fpath = REPORT_DIR / 'summary_mega_sweep.md'
    write_table_md(rows, fpath)
    print(f"Report: {fpath}")
    print(f"Figures: {FIG_DIR}/")


if __name__ == '__main__':
    main()
