"""
Generate summary_mega_sweep.md + plots from mega matrix outputs.

Reads outputs/_mega_matrix/*_model_*/*/eval_*/stage1_*/preds_chip.parquet
Writes docs/chip-multilabel/manager_report/summary_mega_sweep.md + .png plots
"""
import json
import glob
import ast
import math
import os
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from chip_multilabel._bit_metrics import compute_bit_metrics


BITS = ['bank_boundary', 'fork', 'scratch', 'scratch_rot']
SHORT = {'bank_boundary': 'bb', 'fork': 'fk', 'scratch': 'sc', 'scratch_rot': 'sr'}
SEL_LABEL = {'f1': 'val_f1', 'margin_max': 'val_margin'}
SEL_ALIAS = {
    'f1': 'f1',
    'val_f1': 'f1',
    'margin': 'margin_max',
    'val_margin': 'margin_max',
    'margin_max': 'margin_max',
}

# OUT_BASE / MODEL_BASE = the GROUP folder for this run (TS-prefixed)
# Group dir convention: $OUT_BASE/<TS>_<backbone>/
#   summary_mega_sweep.md, figs_mega/ — both written here
#   train{TN}_{SEL}/<inner_run>/best_model.pth + eval_{EN}/stage1_*/
# Backward compat: if MEGA_GROUP_DIR not set, fall back to MEGA_MODEL_BASE or default.
GROUP_DIR = Path(os.environ.get("MEGA_GROUP_DIR",
                                os.environ.get("MEGA_MODEL_BASE",
                                               "outputs/_mega_matrix")))
OUT_BASE = GROUP_DIR    # alias for find_model_root globs
REPORT_DIR = GROUP_DIR
REPORT_DIR.mkdir(parents=True, exist_ok=True)
FIG_DIR = REPORT_DIR / "figs_mega"
FIG_DIR.mkdir(parents=True, exist_ok=True)

# 260521 — sweep axes are env-overridable (run.sh / run_ddp.sh export
# MEGA_TRAIN_SIZES / MEGA_EVAL_SIZES / MEGA_SELS to match the actual run).
def _sizes_env(name, default):
    v = os.environ.get(name)
    if not v:
        return default
    return [x.strip() for x in v.replace(',', ' ').split() if x.strip()]


def _sels_env(name, default):
    out = []
    for raw in _sizes_env(name, default):
        sel = SEL_ALIAS.get(raw, raw)
        if sel not in out:
            out.append(sel)
    return out


TRAIN_NS = [int(x) for x in _sizes_env("MEGA_TRAIN_SIZES", ["50", "100", "200", "400"])]
EVAL_NS = _sizes_env("MEGA_EVAL_SIZES", ["200", "2000", "20000"])
SELS = _sels_env("MEGA_SELS", ["f1", "margin_max"])


def find_model_root(tn, sel):
    # New layout: GROUP/train{tn}_{sel}/   (TS prefix is on GROUP_DIR, not cell)
    # Legacy layouts: GROUP/<TS>_model_train{tn}_{sel}/ , GROUP/model_train{tn}_{sel}/
    cands = (list(OUT_BASE.glob(f"train{tn}_{sel}"))
             + list(OUT_BASE.glob(f"*_model_train{tn}_{sel}"))
             + list(OUT_BASE.glob(f"model_train{tn}_{sel}")))
    return sorted([p for p in cands if p.is_dir()], key=lambda p: p.name, reverse=True)


def find_run_dirs(model_root):
    runs = [p for p in model_root.iterdir() if p.is_dir() and (p / 'best_model.pth').exists()]
    return sorted(runs, key=lambda p: p.name, reverse=True)


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
    if not pchip.exists():
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
    ni_sub = sub[sub['group'].isin(['Normal', 'Invalid'])]
    ood_sub = sub[sub['group'].apply(lambda g: g.startswith('OOD_'))]

    bit_metrics = compute_bit_metrics(sub)
    per_bit = {
        b: float(bit_metrics.get('per_bit_F1_positive', {}).get(b, {}).get('f1', float('nan')))
        for b in BITS
    }
    bit_F1 = float(bit_metrics.get('macro_F1_positive', float('nan')))
    far = float(bit_metrics.get('chip_FAR', 0.0))
    ni_far = float(bit_metrics.get('normal_invalid_chip_FAR', 0.0))
    ood_far_total = float(bit_metrics.get('ood_chip_FAR', 0.0))

    # Per-negative-class FAR (kept for class breakdown table). OOD keys keep the
    # legacy OOD_ prefix used by the existing markdown table.
    ood_far = {}
    for cls, stat in bit_metrics.get('per_class_FAR', {}).items():
        key = f'OOD_{cls}' if cls not in ('Normal', 'Invalid') else cls
        fp_g = int(stat.get('FAR_chip_count', 0))
        n_g = int(stat.get('n_chips', 0))
        if n_g == 0:
            continue
        rate = float(stat.get('chip_FAR', 0.0))
        ood_far[key] = (fp_g, n_g, rate)

    if thr_path.exists():
        thr = json.load(open(thr_path, encoding='utf-8')).get(cell, {}).get('thresholds', {})
    else:
        thr = {}

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
        'ni_far': ni_far,
        'ood_far_total': ood_far_total,
        'n_ni': len(ni_sub),
        'n_ood': len(ood_sub),
        'per_bit_F1': per_bit,
        'per_class_FAR': ood_far,
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
                model_roots = find_model_root(tn, sel)
                if not model_roots:
                    continue
                runs = find_run_dirs(model_roots[0])
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
                    'run_dir': str(run),
                    'eval_dir': str(eval_dir),
                    'cell': 'T0__I13',
                    **m,
                })
    return rows


def expected_result_keys():
    return {(tn, sel, str(en)) for tn in TRAIN_NS for sel in SELS for en in EVAL_NS}


def result_key(row):
    return (int(row['train_n']), str(row['selection']), str(row['eval_n']))


def missing_result_keys(rows):
    found = {result_key(r) for r in rows}
    missing = expected_result_keys() - found
    return sorted(missing, key=lambda x: (x[0], x[1], x[2]))


def missing_required_selections():
    required = {'f1', 'margin_max'}
    return sorted(required - set(SELS))


def expected_plot_names(rows):
    names = [
        'bit_F1_heatmap.png',
        'total_far_heatmap.png',
        'ni_far_heatmap.png',
        'ood_far_heatmap.png',
        'scaling_curves.png',
        'combined_bit_far_by_sel.png',
        'bit_F1_by_sel.png',
        'far_by_sel.png',
    ]
    if paired_selection_rows(rows):
        names.insert(0, 'best_model_eval_by_selection.png')
    return names


def assert_expected_plots(rows):
    missing = []
    for name in expected_plot_names(rows):
        path = FIG_DIR / name
        if not path.exists() or path.stat().st_size <= 0:
            missing.append(path)
    if missing:
        print("[make_report] ERROR: missing plot files:")
        for path in missing:
            print(f"   - {path}")
        raise SystemExit(1)


def sel_label(sel: str) -> str:
    return SEL_LABEL.get(sel, f"val_{sel}")


def paired_selection_rows(rows):
    """Side-by-side val_f1 vs val_margin eval metrics for every condition."""
    out = []
    left = 'f1'
    right = 'margin_max'
    for tn in TRAIN_NS:
        for en in EVAL_NS:
            r_f1 = next((r for r in rows if r['train_n'] == tn
                         and r['selection'] == left and r['eval_n'] == en), None)
            r_mg = next((r for r in rows if r['train_n'] == tn
                         and r['selection'] == right and r['eval_n'] == en), None)
            if not (r_f1 or r_mg):
                continue
            out.append({
                'train_n': tn,
                'eval_n': en,
                'val_f1': r_f1,
                'val_margin': r_mg,
            })
    return out


def format_pct(v) -> str:
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return "NA"
    return f"{float(v) * 100:.2f}%"


def format_float(v) -> str:
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return "NA"
    return f"{float(v):.4f}"


def write_table_md(rows, fpath):
    out = []
    out.append("# Mega Matrix Sweep - best-model eval bit_F1 / FAR\n")
    out.append(f"Generated: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    out.append(f"Sweep axes: train={TRAIN_NS}, eval={EVAL_NS}, selection={[sel_label(s) for s in SELS]}\n")
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

    # ====== Section A2: User-requested final selection table ======
    out.append("\n## 2. Best-model eval performance - val_f1 vs val_margin\n\n")
    out.append("Each row evaluates the trained `best_model.pth` selected by the listed validation criterion. Metrics are from cell `T0__I13` and use eval bit_F1 plus chip-level FAR.\n\n")
    out.append("| train_n | eval_n | val_f1 eval bit_F1 | val_f1 eval FAR | val_f1 NI FAR | val_f1 OOD FAR | val_margin eval bit_F1 | val_margin eval FAR | val_margin NI FAR | val_margin OOD FAR | d bit_F1 | d FAR |\n")
    out.append("|---|---|---|---|---|---|---|---|---|---|---|---|\n")
    for pr in paired_selection_rows(rows):
        r_f1 = pr['val_f1']
        r_mg = pr['val_margin']
        f1_bit = r_f1['bit_F1'] if r_f1 else None
        mg_bit = r_mg['bit_F1'] if r_mg else None
        f1_far = r_f1['total_far'] if r_f1 else None
        mg_far = r_mg['total_far'] if r_mg else None
        d_bit = (mg_bit - f1_bit) if (r_f1 and r_mg) else None
        d_far = (mg_far - f1_far) if (r_f1 and r_mg) else None
        out.append(
            f"| {pr['train_n']} | {pr['eval_n']} | "
            f"{format_float(f1_bit)} | {format_pct(f1_far)} | "
            f"{format_pct(r_f1['ni_far'] if r_f1 else None)} | {format_pct(r_f1['ood_far_total'] if r_f1 else None)} | "
            f"{format_float(mg_bit)} | {format_pct(mg_far)} | "
            f"{format_pct(r_mg['ni_far'] if r_mg else None)} | {format_pct(r_mg['ood_far_total'] if r_mg else None)} | "
            f"{format_float(d_bit)} | {format_pct(d_far)} |\n"
        )

    # ====== Section B: Main results table ======
    out.append("\n## 3. Main results matrix\n")
    out.append("\n**Metric definitions**:\n")
    out.append("- `bit_F1` = absolute rule: macro-F1 over (single + 2-combo) positive bits, using final `pred_labels`\n")
    out.append("- `Total FAR` = (Normal + Invalid + OOD) chips with non-empty defect `pred_labels` / total negatives\n")
    out.append("- Cell `T0__I13` selected (entropy gate + Invalid heuristic, paper SOTA cell)\n\n")

    out.append("\n| train n | selection | eval n | n_pos | n_neg | bit_F1 | Total FAR | NI FAR | OOD FAR | bb F1 | fork F1 | sc F1 | sr F1 |\n")
    out.append("|---|---|---|---|---|---|---|---|---|---|---|---|---|\n")
    for r in rows:
        out.append(
            f"| {r['train_n']} | {sel_label(r['selection'])} | {r['eval_n']} | "
            f"{r['n_pos']} | {r['n_neg']} | "
            f"**{r['bit_F1']:.4f}** | **{r['total_far']*100:.2f}%** | "
            f"{r['ni_far']*100:.2f}% | {r['ood_far_total']*100:.2f}% | "
            f"{r['per_bit_F1']['bank_boundary']:.4f} | "
            f"{r['per_bit_F1']['fork']:.4f} | "
            f"{r['per_bit_F1']['scratch']:.4f} | "
            f"{r['per_bit_F1']['scratch_rot']:.4f} |\n"
        )

    # ====== Section C: val_f1 vs val_margin side-by-side ======
    compare_eval = '2000' if '2000' in EVAL_NS else (EVAL_NS[-1] if EVAL_NS else '')
    out.append(f"\n## 4. `val_f1` vs `val_margin` direct comparison (per train_n, fixed eval={compare_eval})\n\n")
    out.append("\n| train_n | bit_F1 (val_f1) | bit_F1 (val_margin) | Δ | FAR (val_f1) | FAR (val_margin) | Δ FAR |\n")
    out.append("|---|---|---|---|---|---|---|\n")
    for tn in TRAIN_NS:
        r_f1 = next((r for r in rows if r['train_n'] == tn and r['selection'] == 'f1' and r['eval_n'] == compare_eval), None)
        r_mg = next((r for r in rows if r['train_n'] == tn and r['selection'] == 'margin_max' and r['eval_n'] == compare_eval), None)
        if r_f1 and r_mg:
            d_bit = r_mg['bit_F1'] - r_f1['bit_F1']
            d_far = r_mg['total_far'] - r_f1['total_far']
            out.append(
                f"| {tn} | {r_f1['bit_F1']:.4f} | {r_mg['bit_F1']:.4f} | "
                f"{d_bit:+.4f} | {r_f1['total_far']*100:.2f}% | "
                f"{r_mg['total_far']*100:.2f}% | {d_far*100:+.2f}% |\n"
            )

    # ====== Section D: Per-OOD class FAR breakdown ======
    out.append(f"\n## 5. Per-class FAR breakdown (val_margin, eval={compare_eval})\n\n")
    out.append("\n| train_n | Normal | Invalid | CenterDonut | CrossScratch | DiagonalSmear | Starburst |\n")
    out.append("|---|---|---|---|---|---|---|\n")
    for tn in TRAIN_NS:
        r = next((r for r in rows if r['train_n'] == tn and r['selection'] == 'margin_max' and r['eval_n'] == compare_eval), None)
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
    detail_train = 200 if 200 in TRAIN_NS else (TRAIN_NS[-1] if TRAIN_NS else 0)
    out.append(f"\n## 6. Per-defect group prob distribution (val_margin, train={detail_train}, eval={compare_eval})\n\n")
    r_main = next((r for r in rows if r['train_n'] == detail_train and r['selection'] == 'margin_max' and r['eval_n'] == compare_eval), None)
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
    out.append("\n## 7. Analysis\n\n")
    out.append("### 7.1 Selection criterion impact\n")
    if rows:
        f1_avg_bit = np.mean([r['bit_F1'] for r in rows if r['selection'] == 'f1'])
        mg_avg_bit = np.mean([r['bit_F1'] for r in rows if r['selection'] == 'margin_max'])
        f1_avg_far = np.mean([r['total_far'] for r in rows if r['selection'] == 'f1'])
        mg_avg_far = np.mean([r['total_far'] for r in rows if r['selection'] == 'margin_max'])
        out.append(f"- avg bit_F1: val_f1={f1_avg_bit:.4f} vs val_margin={mg_avg_bit:.4f} ({(mg_avg_bit - f1_avg_bit):+.4f})\n")
        out.append(f"- avg Total FAR: val_f1={f1_avg_far*100:.2f}% vs val_margin={mg_avg_far*100:.2f}% ({(mg_avg_far - f1_avg_far)*100:+.2f}%)\n")

    out.append("\n### 7.2 Train data scaling\n")
    for sel in SELS:
        out.append(f"\n**{sel}**:\n")
        for tn in TRAIN_NS:
            rs = [r for r in rows if r['train_n'] == tn and r['selection'] == sel]
            if rs:
                avg_bit = np.mean([r['bit_F1'] for r in rs])
                avg_far = np.mean([r['total_far'] for r in rs])
                out.append(f"  - train={tn}: bit_F1={avg_bit:.4f}, FAR={avg_far*100:.2f}%\n")

    out.append("\n### 7.3 Eval size impact (statistical power)\n")
    for en in EVAL_NS:
        rs = [r for r in rows if r['eval_n'] == en]
        if rs:
            avg_bit = np.mean([r['bit_F1'] for r in rs])
            out.append(f"  - eval={en}: avg bit_F1={avg_bit:.4f} (n_pos={rs[0]['n_pos']}, n_neg={rs[0]['n_neg']})\n")

    # ====== Section G: Plots ======
    out.append("\n## 8. Plots\n\n")

    # Plot 0: explicit final val_f1 vs val_margin best-model eval performance.
    pairs = paired_selection_rows(rows)
    if pairs:
        x = np.arange(len(pairs))
        labels = [f"t{p['train_n']}\ne{p['eval_n']}" for p in pairs]
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(max(10, len(pairs) * 0.75), 8), sharex=True)
        for key, label, color in [('val_f1', 'val_f1 best_model', 'tab:blue'),
                                  ('val_margin', 'val_margin best_model', 'tab:green')]:
            ys = [p[key]['bit_F1'] if p[key] else np.nan for p in pairs]
            ax1.plot(x, ys, marker='o', label=label, color=color)
        for key, label, color in [('val_f1', 'val_f1 best_model', 'tab:red'),
                                  ('val_margin', 'val_margin best_model', 'tab:orange')]:
            ys = [p[key]['total_far'] * 100 if p[key] else np.nan for p in pairs]
            ax2.plot(x, ys, marker='s', label=label, color=color)
        ax1.set_ylabel('eval bit_F1')
        ax1.set_title('Best-model eval bit_F1 by validation selection')
        ax1.grid(True, alpha=0.3)
        ax1.legend(fontsize=9)
        ax2.set_ylabel('eval Total FAR (%)')
        ax2.set_xlabel('condition (train_n / eval_n)')
        ax2.set_title('Best-model eval FAR by validation selection')
        ax2.grid(True, alpha=0.3)
        ax2.legend(fontsize=9)
        ax2.set_xticks(x)
        ax2.set_xticklabels(labels, rotation=0)
        plt.tight_layout()
        plt.savefig(FIG_DIR / 'best_model_eval_by_selection.png', dpi=120)
        plt.close()
        out.append("### Plot 0 - Best-model eval bit_F1/FAR by validation selection\n\n")
        out.append("![best-model eval by selection](figs_mega/best_model_eval_by_selection.png)\n\n")

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

    # Plot 2b: NI FAR heatmap (Normal + Invalid only)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for ax, sel in zip(axes, SELS):
        mat = np.zeros((len(TRAIN_NS), len(EVAL_NS)))
        for i, tn in enumerate(TRAIN_NS):
            for j, en in enumerate(EVAL_NS):
                r = next((r for r in rows if r['train_n'] == tn and r['selection'] == sel and r['eval_n'] == en), None)
                mat[i, j] = r['ni_far'] * 100 if r else np.nan
        im = ax.imshow(mat, cmap='Oranges', vmin=0, vmax=5, aspect='auto')
        ax.set_xticks(range(len(EVAL_NS)))
        ax.set_xticklabels([f'eval={e}' for e in EVAL_NS])
        ax.set_yticks(range(len(TRAIN_NS)))
        ax.set_yticklabels([f'train={t}' for t in TRAIN_NS])
        ax.set_title(f'NI FAR % (Normal+Invalid, {sel})')
        for i in range(len(TRAIN_NS)):
            for j in range(len(EVAL_NS)):
                if not np.isnan(mat[i, j]):
                    ax.text(j, i, f'{mat[i, j]:.2f}%', ha='center', va='center', fontsize=9)
        plt.colorbar(im, ax=ax)
    plt.tight_layout()
    plt.savefig(FIG_DIR / 'ni_far_heatmap.png', dpi=120)
    plt.close()
    out.append("### Plot 2b — NI FAR heatmap (Normal + Invalid only)\n\n")
    out.append("![NI FAR heatmap](figs_mega/ni_far_heatmap.png)\n\n")

    # Plot 2c: OOD FAR heatmap (wafer-pattern OOD only)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for ax, sel in zip(axes, SELS):
        mat = np.zeros((len(TRAIN_NS), len(EVAL_NS)))
        for i, tn in enumerate(TRAIN_NS):
            for j, en in enumerate(EVAL_NS):
                r = next((r for r in rows if r['train_n'] == tn and r['selection'] == sel and r['eval_n'] == en), None)
                mat[i, j] = r['ood_far_total'] * 100 if r else np.nan
        im = ax.imshow(mat, cmap='Purples', vmin=0, vmax=5, aspect='auto')
        ax.set_xticks(range(len(EVAL_NS)))
        ax.set_xticklabels([f'eval={e}' for e in EVAL_NS])
        ax.set_yticks(range(len(TRAIN_NS)))
        ax.set_yticklabels([f'train={t}' for t in TRAIN_NS])
        ax.set_title(f'OOD FAR % (wafer-pattern, {sel})')
        for i in range(len(TRAIN_NS)):
            for j in range(len(EVAL_NS)):
                if not np.isnan(mat[i, j]):
                    ax.text(j, i, f'{mat[i, j]:.2f}%', ha='center', va='center', fontsize=9)
        plt.colorbar(im, ax=ax)
    plt.tight_layout()
    plt.savefig(FIG_DIR / 'ood_far_heatmap.png', dpi=120)
    plt.close()
    out.append("### Plot 2c — OOD FAR heatmap (wafer-pattern OOD)\n\n")
    out.append("![OOD FAR heatmap](figs_mega/ood_far_heatmap.png)\n\n")

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
    out.append("### Plot 3 — Train size scaling curves (all eval_n)\n\n")
    out.append("![scaling curves](figs_mega/scaling_curves.png)\n\n")

    # ====== 260521 — user-requested 3 plots: val_f1 vs val_margin selection ======
    # Plot 4: combined plot — bit_F1 + FAR, both selections (4 legend lines)
    #   x = train_n, dual y-axis, one subplot per eval_n
    fig, axes = plt.subplots(1, len(EVAL_NS), figsize=(6 * len(EVAL_NS), 5), squeeze=False)
    for ax_idx, en in enumerate(EVAL_NS):
        ax = axes[0, ax_idx]
        ax_far = ax.twinx()
        # bit_F1 lines (left axis, solid)
        for sel, color in zip(SELS, ['tab:blue', 'tab:green']):
            xs, ys = [], []
            for tn in TRAIN_NS:
                r = next((r for r in rows if r['train_n'] == tn and r['selection'] == sel and r['eval_n'] == en), None)
                if r:
                    xs.append(tn); ys.append(r['bit_F1'])
            if xs:
                ax.plot(xs, ys, marker='o', color=color, linestyle='-',
                        label=f'bit_F1 (val_{sel})')
        # FAR lines (right axis, dashed)
        for sel, color in zip(SELS, ['tab:red', 'tab:orange']):
            xs, ys = [], []
            for tn in TRAIN_NS:
                r = next((r for r in rows if r['train_n'] == tn and r['selection'] == sel and r['eval_n'] == en), None)
                if r:
                    xs.append(tn); ys.append(r['total_far'] * 100)
            if xs:
                ax_far.plot(xs, ys, marker='s', color=color, linestyle='--',
                            label=f'Total_FAR % (val_{sel})')
        ax.set_xlabel('train_n per class')
        ax.set_ylabel('bit_F1', color='tab:blue')
        ax_far.set_ylabel('Total_FAR (%)', color='tab:red')
        ax.set_title(f'eval_n = {en}')
        ax.grid(True, alpha=0.3)
        # Combined legend (both axes)
        h1, l1 = ax.get_legend_handles_labels()
        h2, l2 = ax_far.get_legend_handles_labels()
        ax.legend(h1 + h2, l1 + l2, fontsize=8, loc='center right')
    plt.suptitle('bit_F1 + Total_FAR by val selection criterion (4 legends)')
    plt.tight_layout()
    plt.savefig(FIG_DIR / 'combined_bit_far_by_sel.png', dpi=120)
    plt.close()
    out.append("### Plot 4 — bit_F1 + Total_FAR by selection (combined 4-legend)\n\n")
    out.append("![combined](figs_mega/combined_bit_far_by_sel.png)\n\n")

    # Plot 5: bit_F1 only — 2 lines per eval_n (val_f1 vs val_margin)
    fig, axes = plt.subplots(1, len(EVAL_NS), figsize=(6 * len(EVAL_NS), 5), squeeze=False)
    for ax_idx, en in enumerate(EVAL_NS):
        ax = axes[0, ax_idx]
        for sel, color in zip(SELS, ['tab:blue', 'tab:green']):
            xs, ys = [], []
            for tn in TRAIN_NS:
                r = next((r for r in rows if r['train_n'] == tn and r['selection'] == sel and r['eval_n'] == en), None)
                if r:
                    xs.append(tn); ys.append(r['bit_F1'])
            if xs:
                ax.plot(xs, ys, marker='o', color=color, label=f'val_{sel}')
        ax.set_xlabel('train_n per class')
        ax.set_ylabel('bit_F1')
        ax.set_title(f'eval_n = {en}')
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=9)
    plt.suptitle('bit_F1 by selection criterion')
    plt.tight_layout()
    plt.savefig(FIG_DIR / 'bit_F1_by_sel.png', dpi=120)
    plt.close()
    out.append("### Plot 5 — bit_F1 by selection (val_f1 vs val_margin)\n\n")
    out.append("![bit_F1 by sel](figs_mega/bit_F1_by_sel.png)\n\n")

    # Plot 6: Total_FAR only — 2 lines per eval_n (val_f1 vs val_margin)
    fig, axes = plt.subplots(1, len(EVAL_NS), figsize=(6 * len(EVAL_NS), 5), squeeze=False)
    for ax_idx, en in enumerate(EVAL_NS):
        ax = axes[0, ax_idx]
        for sel, color in zip(SELS, ['tab:red', 'tab:orange']):
            xs, ys = [], []
            for tn in TRAIN_NS:
                r = next((r for r in rows if r['train_n'] == tn and r['selection'] == sel and r['eval_n'] == en), None)
                if r:
                    xs.append(tn); ys.append(r['total_far'] * 100)
            if xs:
                ax.plot(xs, ys, marker='s', color=color, label=f'val_{sel}')
        ax.set_xlabel('train_n per class')
        ax.set_ylabel('Total_FAR (%)')
        ax.set_title(f'eval_n = {en}')
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=9)
    plt.suptitle('Total_FAR by selection criterion')
    plt.tight_layout()
    plt.savefig(FIG_DIR / 'far_by_sel.png', dpi=120)
    plt.close()
    out.append("### Plot 6 — Total_FAR by selection (val_f1 vs val_margin)\n\n")
    out.append("![FAR by sel](figs_mega/far_by_sel.png)\n\n")

    # ====== Footer ======
    out.append("\n## 9. Recipe (all cells)\n\n")
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


def print_final_selection_summary(rows):
    pairs = paired_selection_rows(rows)
    if not pairs:
        print("[make_report] final selection table: no val_f1/val_margin pairs found")
        return
    print("[make_report] final val_f1 vs val_margin best_model eval performance (cell=T0__I13)")
    print("[make_report] train eval | val_f1 bit_F1/FAR | val_margin bit_F1/FAR | d_bit_F1 d_FAR")
    for p in pairs:
        r_f1 = p['val_f1']
        r_mg = p['val_margin']
        f1_bit = r_f1['bit_F1'] if r_f1 else None
        mg_bit = r_mg['bit_F1'] if r_mg else None
        f1_far = r_f1['total_far'] if r_f1 else None
        mg_far = r_mg['total_far'] if r_mg else None
        d_bit = (mg_bit - f1_bit) if (r_f1 and r_mg) else None
        d_far = (mg_far - f1_far) if (r_f1 and r_mg) else None
        print("[make_report] "
              f"t{p['train_n']} e{p['eval_n']} | "
              f"{format_float(f1_bit)}/{format_pct(f1_far)} | "
              f"{format_float(mg_bit)}/{format_pct(mg_far)} | "
              f"{format_float(d_bit)} {format_pct(d_far)}")


def main():
    print(f"[make_report] GROUP_DIR={GROUP_DIR}")
    print(f"[make_report] sweep axes: TRAIN_NS={TRAIN_NS} EVAL_NS={EVAL_NS} SELS={SELS}")
    print(f"[make_report] expected cells: {len(TRAIN_NS)*len(SELS)} train x {len(EVAL_NS)} eval = "
          f"{len(TRAIN_NS)*len(SELS)*len(EVAL_NS)}")
    missing_sels = missing_required_selections()
    if missing_sels:
        print("[make_report] ERROR: final report requires both val_f1 and val_margin selections.")
        print(f"[make_report] missing selections: {[sel_label(s) for s in missing_sels]}")
        print("[make_report] Refusing f1-only or margin-only summary. Set MEGA_ALLOW_SINGLE_SELECTION=1 only for debugging.")
        if os.environ.get("MEGA_ALLOW_SINGLE_SELECTION", "0") != "1":
            raise SystemExit(1)
        print("[make_report] WARN: MEGA_ALLOW_SINGLE_SELECTION=1, writing single-selection debug summary.")
    rows = collect_all()
    print(f"[make_report] Collected {len(rows)} eval results")
    if not rows:
        # Diagnostic: show what's in GROUP_DIR so user sees why scan failed
        print(f"[make_report] ERROR: no eval results found under {GROUP_DIR}")
        if GROUP_DIR.exists():
            print(f"[make_report] GROUP_DIR contents:")
            for p in sorted(GROUP_DIR.iterdir()):
                print(f"   - {p.name}/")
            print(f"[make_report] looking for: train<TN>_<SEL>/<inner_run>/eval_<EN>/stage1_*/preds_chip.parquet")
        else:
            print(f"[make_report] GROUP_DIR does not exist (run training first).")
        raise SystemExit(1)
    missing = missing_result_keys(rows)
    if missing:
        print("[make_report] ERROR: incomplete eval results; missing cells:")
        for tn, sel, en in missing:
            print(f"   - train{tn}_{sel}/<inner_run>/eval_{en}/stage1_*/preds_chip.parquet")
        print("[make_report] Refusing to write a partial summary. Set MEGA_ALLOW_PARTIAL_REPORT=1 only for debugging.")
        if os.environ.get("MEGA_ALLOW_PARTIAL_REPORT", "0") != "1":
            raise SystemExit(1)
        print("[make_report] WARN: MEGA_ALLOW_PARTIAL_REPORT=1, writing partial summary anyway.")
    fpath = REPORT_DIR / 'summary_mega_sweep.md'
    write_table_md(rows, fpath)
    assert_expected_plots(rows)
    print_final_selection_summary(rows)
    print(f"[make_report] Report: {fpath}")
    print(f"[make_report] Figures: {FIG_DIR}/")


if __name__ == '__main__':
    main()
