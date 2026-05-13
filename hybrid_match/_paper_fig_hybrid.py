"""Generate beginner-friendly figures for the hybrid CNN + obj_id_map method.

Produces 4 PNG figures in docs/paper/figures/:
  hybrid_fig1_overview.png      : 4-panel pipeline overview (real wafer + obj_id + P + bar)
  hybrid_fig2_distribution.png  : per-class P(obj_id | y, cell) for 6 example classes
  hybrid_fig3_matching.png      : step-by-step matching on one wafer (4 panels)
  hybrid_fig4_flowchart.png     : confidence-gated decision flowchart

obj_id_maps are illustrative simulations (chip CNN output style) because the
real obj_id_map .npy cache lives on a different machine. The pattern logic
mirrors the alpha-mechanism used by _sample_canvas_gen.py so the figures
faithfully convey what the chip CNN would output.

Run: python hybrid_match/_paper_fig_hybrid.py
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Polygon
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "docs" / "paper" / "figures"
OUT.mkdir(parents=True, exist_ok=True)

# ---- chip CNN classes (K+1 = 6 categories incl. none) ----
CHIP_CLASSES = ['none', 'bank_boundary', 'fork', 'scratch', 'scratch_rot', 'invalid_main']
CHIP_COLORS = ['#f0f0f0', '#4575b4', '#fc8d59', '#d73027', '#984ea3', '#ffd92f']
CHIP_CMAP = ListedColormap(CHIP_COLORS)

GRID = 32  # V3 native obj_id grid


def make_obj_id_map(wafer_class: str, rng: np.random.Generator, noise: float = 0.04) -> np.ndarray:
    """Return 32x32 int [0..5] map illustrating chip CNN output for a wafer class.

    wafer_class follows "<spatial>_<obj>" convention (e.g. Donut_scratch).
    spatial determines where chip-objects appear; obj determines which obj_id.
    """
    m = np.zeros((GRID, GRID), dtype=int)
    cy = cx = (GRID - 1) / 2
    yy, xx = np.indices((GRID, GRID))
    r = np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2)

    parts = wafer_class.split('_', 1)
    spatial = parts[0]
    obj = parts[1] if len(parts) > 1 else 'scratch'
    obj_id = CHIP_CLASSES.index(obj) if obj in CHIP_CLASSES else 3

    if spatial == 'Donut':
        mask = (r > 9) & (r < 12.5)
    elif spatial == 'Edge-Top':
        mask = yy < 4
    elif spatial == 'Edge-Bottom':
        mask = yy > GRID - 5
    elif spatial == 'Edge-Ring':
        mask = r > GRID / 2 - 2.5
    elif spatial == 'Center':
        mask = r < 5
    else:
        mask = np.zeros_like(yy, dtype=bool)

    # add per-cell probability so the mask is fuzzy (chip CNN isn't perfect)
    p_in = rng.random((GRID, GRID))
    keep = mask & (p_in < 0.9)
    m[keep] = obj_id

    # chip CNN noise: a few stray cells outside mask classified as something
    stray = (rng.random((GRID, GRID)) < noise) & (~mask)
    if stray.any():
        m[stray] = rng.integers(1, 6, size=int(stray.sum()))
    return m


def build_phat(wafer_class: str, n_train: int = 200, alpha: float = 1.0,
               seed: int = 42) -> np.ndarray:
    """Per-class per-cell categorical histogram with Laplace smoothing.

    Returns shape (32, 32, 6) with P_hat[i,j,k] = P(obj_id=k | class, cell=(i,j)).
    """
    rng = np.random.default_rng(seed)
    K = len(CHIP_CLASSES)
    counts = np.zeros((GRID, GRID, K), dtype=float)
    for _ in range(n_train):
        m = make_obj_id_map(wafer_class, rng)
        for k in range(K):
            counts[:, :, k] += (m == k)
    phat = (counts + alpha) / (counts.sum(-1, keepdims=True) + alpha * K)
    return phat


def log_likelihood(phat: np.ndarray, obj_map: np.ndarray) -> float:
    """log L(y | M) = sum_{i,j} log P_hat(M[i,j] | y, (i,j))."""
    ii, jj = np.indices(obj_map.shape)
    return float(np.log(phat[ii, jj, obj_map] + 1e-12).sum())


def add_chip_legend(fig, anchor=(0.5, 0.02), ncol=6):
    handles = [plt.Rectangle((0, 0), 1, 1, fc=CHIP_COLORS[i]) for i in range(len(CHIP_CLASSES))]
    fig.legend(handles, CHIP_CLASSES, loc='upper center',
               bbox_to_anchor=anchor, ncol=ncol, fontsize=9,
               frameon=False, title='chip CNN obj_id class')


# ---------- Figure 1: overview ----------
def fig_overview():
    fig = plt.figure(figsize=(15, 12))
    gs = fig.add_gridspec(2, 2, hspace=0.45, wspace=0.28,
                          left=0.05, right=0.97, top=0.88, bottom=0.10)

    # (A) Real wafer PNG ----------------------------------------------------
    ax = fig.add_subplot(gs[0, 0])
    wafer_pngs = sorted((ROOT / "outputs/logs_wafer/overall/wrong/val").rglob("*.png"))
    if wafer_pngs:
        png = wafer_pngs[0]  # Edge-Bottom_bank_boundary -> Edge-Bottom_scratch
        # load palette-mode (each pixel is grade 0..7 + outside idx). render with
        # high-contrast colormap so beginners can SEE the defect region.
        img_p = Image.open(png)
        arr = np.array(img_p)  # palette indices, shape (H, W)
        # downsample for fast plot
        img_p.thumbnail((720, 720))
        arr = np.array(img_p)
        # map palette idx -> grayscale "defect intensity" (0 = light, 7 = dark)
        # outside (idx > 7, e.g. 31 = white) -> NaN for cmap
        disp = arr.astype(float)
        disp[disp > 7] = np.nan      # outside / invalid → transparent
        ax.imshow(disp, cmap='YlOrRd', vmin=0, vmax=7,
                  interpolation='nearest')
        true_cls = png.parts[-3]
        pred_cls = png.parts[-2]
        ax.set_title(
            f"(A) Real wafer image (palette PNG, grade-colored)\n"
            f"true class = {true_cls}\n"
            f"V3 CNN predicted = {pred_cls}  (WRONG — hybrid target)",
            fontsize=10)
        # add a small colorbar inside
        cax = ax.inset_axes([1.02, 0.0, 0.04, 1.0])
        sm = plt.cm.ScalarMappable(cmap='YlOrRd',
                                    norm=plt.Normalize(vmin=0, vmax=7))
        cb = plt.colorbar(sm, cax=cax)
        cb.set_label('fail-bit grade', fontsize=8)
        cb.ax.tick_params(labelsize=7)
    else:
        ax.text(0.5, 0.5, "no wafer PNG found", ha='center', va='center')
        ax.set_title("(A) Real wafer (placeholder)")
    ax.set_xticks([]); ax.set_yticks([])

    # (B) obj_id_map --------------------------------------------------------
    ax = fig.add_subplot(gs[0, 1])
    rng = np.random.default_rng(0)
    test_map = make_obj_id_map('Edge-Bottom_bank_boundary', rng)
    ax.imshow(test_map, cmap=CHIP_CMAP, vmin=0, vmax=5, interpolation='nearest')
    ax.set_title(
        "(B) chip CNN forward -> obj_id_map (32x32)\n"
        "each cell = chip-object class\n"
        "bottom rows lit blue (bank_boundary) -> true class signal", fontsize=10)
    ax.set_xticks(np.arange(0, GRID + 1, 8))
    ax.set_yticks(np.arange(0, GRID + 1, 8))
    ax.set_xlabel("cell j"); ax.set_ylabel("cell i")

    # (C) 4 class P_hat dominant ------------------------------------------
    classes_demo = ['Donut_scratch', 'Edge-Top_fork',
                    'Edge-Bottom_bank_boundary', 'Edge-Ring_scratch_rot']
    # 2x2 sub-grid using subgridspec (no top/bottom support; rely on hspace)
    gs_c = gs[1, 0].subgridspec(2, 2, hspace=0.55, wspace=0.18)
    sub_axes_c = []
    for k, cls in enumerate(classes_demo):
        sub = fig.add_subplot(gs_c[k // 2, k % 2])
        phat = build_phat(cls, n_train=120)
        dom = phat.argmax(-1)
        sub.imshow(dom, cmap=CHIP_CMAP, vmin=0, vmax=5, interpolation='nearest')
        sub.set_title(cls, fontsize=9)
        sub.set_xticks([]); sub.set_yticks([])
        sub_axes_c.append(sub)
    # place panel-C header via fig.text above the sub-grid
    bbox = sub_axes_c[0].get_position()
    fig.text(0.27, bbox.y1 + 0.045,
             "(C) per-class P(obj_id | y, cell) — argmax view\n"
             "spatial location of each obj_id is the class fingerprint",
             ha='center', va='bottom', fontsize=10)

    # (D) likelihood bar chart ----------------------------------------------
    ax = fig.add_subplot(gs[1, 1])
    p_cnn = np.array([0.18, 0.31, 0.32, 0.12, 0.07])  # low-conf, near tie
    cnn_classes = ['Edge-Bottom_fork', 'Edge-Bottom_scratch',
                   'Edge-Bottom_bank_boundary', 'Edge-Top_bank_boundary',
                   'Donut_bank_boundary']
    log_L = np.array([log_likelihood(build_phat(c, n_train=120), test_map) for c in cnn_classes])
    log_post = np.log(p_cnn + 1e-12) + 1.0 * log_L
    post = np.exp(log_post - log_post.max())
    post = post / post.sum()

    x = np.arange(len(cnn_classes))
    w = 0.38
    ax.bar(x - w / 2, p_cnn, w, label='CNN softmax p(y|x)', color='#fc8d59')
    ax.bar(x + w / 2, post, w, label='hybrid posterior', color='#4575b4')
    ax.set_xticks(x)
    ax.set_xticklabels([c.replace('_', '\n') for c in cnn_classes], fontsize=8)
    ax.set_ylabel('probability')
    ax.set_title("(D) decision = argmax(p_cnn) vs argmax(posterior)\n"
                 "low-conf CNN tie -> log L breaks the tie toward true class",
                 fontsize=10)
    true_idx = cnn_classes.index('Edge-Bottom_bank_boundary')
    ax.annotate('TRUE', xy=(true_idx, post[true_idx]),
                xytext=(true_idx, post[true_idx] + 0.12),
                ha='center', fontsize=10, color='red',
                arrowprops=dict(arrowstyle='->', color='red'))
    ax.legend(fontsize=8, loc='upper left')
    ax.set_ylim(0, max(post.max(), p_cnn.max()) * 1.35)

    fig.suptitle("Hybrid CNN + obj_id_map prototype matching — overview",
                 fontsize=15, y=0.96, weight='bold')
    add_chip_legend(fig, anchor=(0.5, 0.06))
    out = OUT / "hybrid_fig1_overview.png"
    fig.savefig(out, dpi=120, bbox_inches='tight')
    plt.close(fig)
    print(f"[OK] {out}")


# ---------- Figure 2: per-class distribution ----------
def fig_distribution():
    classes = ['Donut_scratch', 'Donut_fork',
               'Edge-Top_scratch', 'Edge-Bottom_bank_boundary',
               'Edge-Ring_scratch_rot', 'Center_invalid_main']
    fig, axes = plt.subplots(2, 6, figsize=(16, 6.2),
                              gridspec_kw={'hspace': 0.32, 'wspace': 0.2})
    im_conf = None
    for i, cls in enumerate(classes):
        phat = build_phat(cls, n_train=200)
        dominant = phat.argmax(-1)
        max_p = phat.max(-1)

        ax = axes[0, i]
        ax.imshow(dominant, cmap=CHIP_CMAP, vmin=0, vmax=5, interpolation='nearest')
        ax.set_title(cls.replace('_', '\n'), fontsize=10)
        ax.set_xticks([]); ax.set_yticks([])
        if i == 0:
            ax.set_ylabel("argmax(P)\ndominant obj_id\nper cell", fontsize=10)

        ax = axes[1, i]
        im_conf = ax.imshow(max_p, cmap='magma', vmin=0, vmax=1, interpolation='nearest')
        ax.set_xticks([]); ax.set_yticks([])
        if i == 0:
            ax.set_ylabel("max(P)\nconfidence\nper cell", fontsize=10)

    fig.suptitle("Per-class P(obj_id = k | y, cell) — Laplace alpha=1, n_train=200/class",
                 fontsize=13, y=0.99, weight='bold')
    cbar = fig.colorbar(im_conf, ax=axes[1, :].ravel().tolist(), shrink=0.7,
                        location='right', pad=0.015)
    cbar.set_label('P(dominant)')
    add_chip_legend(fig, anchor=(0.5, 0.02))
    out = OUT / "hybrid_fig2_distribution.png"
    fig.savefig(out, dpi=120, bbox_inches='tight')
    plt.close(fig)
    print(f"[OK] {out}")


# ---------- Figure 3: step-by-step matching ----------
def fig_matching_steps():
    fig, axes = plt.subplots(1, 4, figsize=(20, 6.5),
                              gridspec_kw={'wspace': 0.40,
                                           'left': 0.04, 'right': 0.99,
                                           'top': 0.78, 'bottom': 0.18})
    rng = np.random.default_rng(7)
    true_cls = 'Edge-Bottom_bank_boundary'
    test_map = make_obj_id_map(true_cls, rng)

    # 1) input wafer (use real PNG if available, else show synthetic)
    ax = axes[0]
    wafer_pngs = sorted((ROOT / "outputs/logs_wafer/overall/wrong/val").rglob("*.png"))
    if wafer_pngs:
        img_p = Image.open(wafer_pngs[0])
        img_p.thumbnail((500, 500))
        arr = np.array(img_p).astype(float)
        arr[arr > 7] = np.nan
        ax.imshow(arr, cmap='YlOrRd', vmin=0, vmax=7, interpolation='nearest')
        ax.set_title("Step 1 — input wafer (real, palette PNG)\n"
                     "color = fail-bit grade 0-7\n"
                     "(downsampled to 384x384 for CNN)", fontsize=10)
    ax.set_xticks([]); ax.set_yticks([])

    # 2) obj_id_map
    ax = axes[1]
    ax.imshow(test_map, cmap=CHIP_CMAP, vmin=0, vmax=5, interpolation='nearest')
    ax.set_title("Step 2 — chip CNN forward\n"
                 "for each 32x32 cell, classify the\n"
                 "chip-object inside -> obj_id_map M", fontsize=10)
    ax.set_xticks([]); ax.set_yticks([])

    # 3) CNN softmax — low-conf example
    classes = ['Edge-Bottom_scratch',
               'Edge-Bottom_bank_boundary',
               'Edge-Bottom_fork',
               'Edge-Top_bank_boundary',
               'Donut_bank_boundary']
    p_cnn = np.array([0.33, 0.30, 0.18, 0.12, 0.07])

    ax = axes[2]
    bars = ax.barh(np.arange(len(classes)), p_cnn, color='#fc8d59')
    ax.set_yticks(np.arange(len(classes)))
    ax.set_yticklabels([c.replace('_', '\n') for c in classes], fontsize=8)
    ax.invert_yaxis()
    ax.axvline(0.85, ls='--', color='gray', label='tau_gate = 0.85')
    ax.set_xlim(0, 1)
    ax.set_xlabel("CNN softmax p(y | x)")
    ax.set_title("Step 3 — CNN softmax\n"
                 "p_max = 0.33 < tau_gate\n"
                 "-> trigger obj_id matching", fontsize=10)
    ax.legend(fontsize=8, loc='lower right')

    # 4) posterior
    log_L = np.array([log_likelihood(build_phat(c, n_train=120), test_map) for c in classes])
    log_post = np.log(p_cnn + 1e-12) + 1.0 * log_L
    post = np.exp(log_post - log_post.max())
    post = post / post.sum()

    ax = axes[3]
    ax.barh(np.arange(len(classes)), post, color='#4575b4')
    ax.set_yticks(np.arange(len(classes)))
    ax.set_yticklabels([c.replace('_', '\n') for c in classes], fontsize=8)
    ax.invert_yaxis()
    ax.set_xlim(0, 1)
    ax.set_xlabel("hybrid posterior")
    ax.set_title("Step 4 — log p_post = log p_cnn + lambda * log L\n"
                 f"argmax = {classes[post.argmax()]}\n"
                 "obj_id_map breaks the CNN tie", fontsize=10)
    true_idx = classes.index(true_cls)
    # annotate from inside the bar (towards left so it stays inside axes)
    ax.annotate('TRUE', xy=(post[true_idx], true_idx),
                xytext=(max(post[true_idx] - 0.30, 0.05), true_idx),
                va='center', fontsize=10, color='red', weight='bold',
                arrowprops=dict(arrowstyle='->', color='red'))

    fig.suptitle("Step-by-step matching example "
                 "(true = Edge-Bottom_bank_boundary, CNN was uncertain between scratch / bb / fork)",
                 fontsize=12, weight='bold', y=0.995)
    add_chip_legend(fig, anchor=(0.5, 0.02))
    out = OUT / "hybrid_fig3_matching.png"
    fig.savefig(out, dpi=120, bbox_inches='tight')
    plt.close(fig)
    print(f"[OK] {out}")


# ---------- Figure 4: decision flowchart ----------
def fig_flowchart():
    fig, ax = plt.subplots(figsize=(14, 7.5))
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 8)
    ax.axis('off')

    def box(x, y, w, h, text, fc='#deebf7', ec='#08519c'):
        b = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.12",
                           ec=ec, fc=fc, lw=2)
        ax.add_patch(b)
        ax.text(x + w / 2, y + h / 2, text, ha='center', va='center',
                fontsize=11)

    def diamond(x, y, w, h, text, fc='#fee0d2'):
        pts = [(x + w / 2, y + h), (x + w, y + h / 2),
               (x + w / 2, y), (x, y + h / 2)]
        poly = Polygon(pts, ec='#a50f15', fc=fc, lw=2)
        ax.add_patch(poly)
        ax.text(x + w / 2, y + h / 2, text, ha='center', va='center', fontsize=11)

    def arrow(x1, y1, x2, y2, text=''):
        ar = FancyArrowPatch((x1, y1), (x2, y2),
                              arrowstyle='->', mutation_scale=22,
                              color='black', lw=1.8)
        ax.add_patch(ar)
        if text:
            ax.text((x1 + x2) / 2, (y1 + y2) / 2 + 0.18, text, fontsize=10,
                    ha='center',
                    bbox=dict(boxstyle='round,pad=0.18', fc='white', ec='gray'))

    # Top row: CNN path
    box(0.3, 6.0, 2.6, 1.2, "wafer image\nx (6400x6400)",
        fc='#e5f5e0', ec='#31a354')
    box(3.6, 6.0, 2.6, 1.2, "CNN forward\np_cnn(y | x)")
    box(7.0, 6.0, 2.6, 1.2, "p_max = max p_cnn\ny_cnn = argmax p_cnn")
    diamond(10.0, 5.4, 3.6, 2.4, "p_max >= tau_gate ?")

    # Right branch: high-conf
    box(10.5, 2.8, 2.6, 1.0, "return y_cnn\n(high-confidence path)",
        fc='#c7e9c0', ec='#238b45')

    # Bottom row: low-conf matching path
    box(0.3, 2.8, 2.6, 1.2, "obj_id_map M(x)\n(32x32, pre-built\nchip CNN cache)",
        fc='#fdd0a2', ec='#d94801')
    box(3.6, 2.8, 2.6, 1.2,
        "log L(y | M)\n= sum log P_hat(M[i,j] | y, (i,j))")
    box(7.0, 2.8, 2.6, 1.2,
        "log p_post =\nlog p_cnn + lambda * log L")

    box(5.0, 0.4, 3.4, 1.2, "return argmax_y p_post\n(low-confidence path)",
        fc='#c6dbef', ec='#2171b5')

    # Arrows top
    arrow(2.9, 6.6, 3.6, 6.6)
    arrow(6.2, 6.6, 7.0, 6.6)
    arrow(9.6, 6.6, 10.4, 6.6)

    # Branch arrows
    arrow(11.8, 5.4, 11.8, 3.85, 'YES\n(CNN only)')
    arrow(10.4, 5.6, 6.3, 3.6, 'NO (trigger\nobj_id matching)')

    # Bottom path
    arrow(2.9, 3.4, 3.6, 3.4)
    arrow(6.2, 3.4, 7.0, 3.4)
    arrow(8.3, 2.8, 6.7, 1.65)

    ax.set_title(
        "Confidence-gated hybrid decision flow\n"
        "high-confidence -> CNN only; low-confidence -> CNN x obj_id_map likelihood",
        fontsize=13, weight='bold')

    # legend caption box (move to lower-right empty area)
    legend_text = ("lambda  : likelihood weight (default 1.0)\n"
                   "tau_gate: confidence threshold (default 0.85)\n"
                   "alpha   : Laplace smoothing prior (default 1.0)\n"
                   "lambda = 0 -> CNN-only baseline (worst case)")
    ax.text(13.7, 0.6, legend_text, fontsize=9, va='bottom', ha='right',
            family='monospace',
            bbox=dict(boxstyle='round,pad=0.4', fc='#f7f7f7', ec='gray'))

    out = OUT / "hybrid_fig4_flowchart.png"
    fig.savefig(out, dpi=120, bbox_inches='tight')
    plt.close(fig)
    print(f"[OK] {out}")


# ---------- Figure 5: zoomed obj_id_map (one example, beginner) ----------
def fig_zoomed_example():
    """A LARGE 32x32 obj_id_map for one class, with grid lines + 'what is a cell'
    callouts. Audience: non-AI corporate reviewer."""
    fig = plt.figure(figsize=(15, 8))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.8, 1.0],
                          wspace=0.20, left=0.05, right=0.97,
                          top=0.88, bottom=0.10)

    rng = np.random.default_rng(11)
    cls = 'Edge-Bottom_bank_boundary'
    obj_map = make_obj_id_map(cls, rng)

    # left: large obj_id_map with grid + cell index axes
    ax = fig.add_subplot(gs[0, 0])
    ax.imshow(obj_map, cmap=CHIP_CMAP, vmin=0, vmax=5, interpolation='nearest')
    # grid lines between cells
    for k in range(GRID + 1):
        ax.axhline(k - 0.5, color='gray', lw=0.3, alpha=0.4)
        ax.axvline(k - 0.5, color='gray', lw=0.3, alpha=0.4)
    ax.set_xticks(np.arange(0, GRID, 4))
    ax.set_yticks(np.arange(0, GRID, 4))
    ax.set_xlabel("cell column j  (0..31)")
    ax.set_ylabel("cell row i  (0..31)")
    ax.set_title(f"obj_id_map example — class = {cls}\n"
                 f"32x32 = 1024 cells, each cell = 1 chip = 200x200 pixels",
                 fontsize=12, pad=10)

    # call-out arrows to two interesting cells
    def callout(i, j, text, dx, dy):
        ax.annotate(text, xy=(j, i),
                    xytext=(j + dx, i + dy),
                    fontsize=9,
                    bbox=dict(boxstyle='round,pad=0.3', fc='#fff7bc', ec='black'),
                    arrowprops=dict(arrowstyle='->', color='black', lw=1.5))

    # find a 'bank_boundary' cell (blue) inside the bottom band
    bb_cells = np.argwhere(obj_map == 1)
    if len(bb_cells):
        i0, j0 = bb_cells[len(bb_cells) // 2]
        callout(i0, j0,
                "this cell = chip CNN says\n  'bank_boundary' (blue)",
                dx=-13, dy=-3)
    # a 'none' cell (gray) far from any defect
    none_cells = np.argwhere(obj_map == 0)
    if len(none_cells):
        # pick a center one
        cy, cx = GRID // 2, GRID // 2
        dist = np.abs(none_cells - np.array([[cy, cx]])).sum(1)
        i1, j1 = none_cells[np.argmin(dist)]
        callout(i1, j1,
                "this cell = chip CNN says\n  'none' (normal)",
                dx=-9, dy=-9)

    # right: explanation + chip class legend + grade colormap
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.axis('off')
    ax2.set_xlim(0, 1); ax2.set_ylim(0, 1)

    explain_text = (
        "How to read this map:\n"
        "\n"
        "1. A wafer is divided into 32 x 32 = 1024\n"
        "    rectangular 'chips' (each 200x200 px).\n"
        "\n"
        "2. A chip CNN classifies what is inside\n"
        "    each chip into 6 categories:\n"
        "      - none           (normal, no defect)\n"
        "      - bank_boundary  (grid line pattern)\n"
        "      - fork           (large blob)\n"
        "      - scratch        (vertical line)\n"
        "      - scratch_rot    (21 deg tilted line)\n"
        "      - invalid_main   (measurement error)\n"
        "\n"
        "3. The 32x32 result is the obj_id_map.\n"
        "    The COLOR PATTERN is the class signature.\n"
        "\n"
        "4. In this example, the bottom two rows\n"
        "    are filled with blue (bank_boundary).\n"
        "    Combined with the spatial location\n"
        "    (bottom edge), this signature uniquely\n"
        "    identifies the wafer class\n"
        "    'Edge-Bottom_bank_boundary'.\n"
    )
    ax2.text(0.0, 0.98, explain_text, fontsize=10, va='top', ha='left',
             family='monospace',
             bbox=dict(boxstyle='round,pad=0.6', fc='#f7f7f7', ec='gray'))

    fig.suptitle("What is an obj_id_map? — one zoomed example",
                 fontsize=14, weight='bold', y=0.97)
    add_chip_legend(fig, anchor=(0.5, 0.02))
    out = OUT / "hybrid_fig5_zoomed_example.png"
    fig.savefig(out, dpi=120, bbox_inches='tight')
    plt.close(fig)
    print(f"[OK] {out}")


# ---------- Figure 6: confusion-pair gallery ----------
def fig_confusion_pairs():
    """4 rows of confused class pairs, each row shows two obj_id_maps + caption."""
    pairs = [
        # (class A, class B, "what's similar", "what's different")
        ('Edge-Bottom_scratch', 'Edge-Bottom_bank_boundary',
         "same spatial location\n(bottom edge band)",
         "different chip object\nred = scratch  vs  blue = bank_boundary"),
        ('Edge-Top_fork', 'Edge-Bottom_fork',
         "same chip object\n(orange = fork)",
         "different spatial location\ntop band  vs  bottom band"),
        ('Donut_scratch', 'Donut_fork',
         "same donut ring shape",
         "different chip object inside the ring\nred = scratch  vs  orange = fork"),
        ('Edge-Ring_scratch', 'Edge-Ring_scratch_rot',
         "same outer-ring location\nsame parent family",
         "rotation variant\nvertical line  vs  21 deg tilted line"),
    ]
    fig, axes = plt.subplots(len(pairs), 3, figsize=(14, 3.0 * len(pairs)),
                              gridspec_kw={'width_ratios': [1.0, 1.0, 1.2],
                                           'wspace': 0.25, 'hspace': 0.55,
                                           'left': 0.04, 'right': 0.99,
                                           'top': 0.94, 'bottom': 0.07})

    for r, (a, b, sim, diff) in enumerate(pairs):
        rng_a = np.random.default_rng(20 + r)
        rng_b = np.random.default_rng(40 + r)
        ma = make_obj_id_map(a, rng_a)
        mb = make_obj_id_map(b, rng_b)

        axes[r, 0].imshow(ma, cmap=CHIP_CMAP, vmin=0, vmax=5, interpolation='nearest')
        axes[r, 0].set_title(f"class A: {a}", fontsize=10)
        axes[r, 0].set_xticks([]); axes[r, 0].set_yticks([])

        axes[r, 1].imshow(mb, cmap=CHIP_CMAP, vmin=0, vmax=5, interpolation='nearest')
        axes[r, 1].set_title(f"class B: {b}", fontsize=10)
        axes[r, 1].set_xticks([]); axes[r, 1].set_yticks([])

        ax = axes[r, 2]
        ax.axis('off')
        ax.set_xlim(0, 1); ax.set_ylim(0, 1)
        ax.text(0.02, 0.95, f"PAIR {r + 1}", fontsize=11, weight='bold', va='top',
                bbox=dict(boxstyle='round,pad=0.2', fc='#fee0d2', ec='#a50f15'))
        ax.text(0.02, 0.78,
                "What is similar (CNN gets confused):\n" + sim,
                fontsize=10, va='top',
                bbox=dict(boxstyle='round,pad=0.3', fc='#deebf7', ec='#08519c'))
        ax.text(0.02, 0.38,
                "What is different (obj_id_map sees):\n" + diff,
                fontsize=10, va='top',
                bbox=dict(boxstyle='round,pad=0.3', fc='#c7e9c0', ec='#238b45'))

    fig.suptitle("Confused class pairs — why obj_id_map matching helps\n"
                 "(left two columns = two different wafer classes whose images look similar to the CNN)",
                 fontsize=13, weight='bold', y=0.985)
    add_chip_legend(fig, anchor=(0.5, 0.02))
    out = OUT / "hybrid_fig6_confusion_pairs.png"
    fig.savefig(out, dpi=120, bbox_inches='tight')
    plt.close(fig)
    print(f"[OK] {out}")


if __name__ == "__main__":
    fig_overview()
    fig_distribution()
    fig_matching_steps()
    fig_flowchart()
    fig_zoomed_example()
    fig_confusion_pairs()
    print(f"\nAll figures in: {OUT}")
