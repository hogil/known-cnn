#!/bin/bash
# iter124 — FCM-PM grid×group ablation (paper §6 NEW axis, clean parameterization)
#
# User insight: GRID should equal n_groups * n where n is the per-group cell
# multiplier. This eliminates leftover cells (current iter116J GRID=8/g=3 has
# 64/3=21+1 leftover - uneven groups). Clean grids guarantee each group has
# exactly n^2 cells.
#
# Baseline (frozen): T7 BCE+LS=0.30, lr 1e-4, ep 10, val_margin, save_every_epoch,
# cutmix-mode=complement, pair=masked, fill=corner, p=0.25, label-scale=0.5.
#
# Cells (only --cutmix-grid-dim and --cutmix-n-groups vary together as g*n):
#
# Group=2 axis (2n x 2n grid):
#   124a: g=2 n=1  GRID=2  (4 cells, 192x192 each, 2 cells/group)
#   124b: g=2 n=2  GRID=4  (16 cells, 96x96, 8 cells/group)
#   124c: g=2 n=3  GRID=6  (36 cells, 64x64, 18 cells/group)
#   124d: g=2 n=4  GRID=8  (64 cells, 48x48, 32 cells/group)  - same cell-size as iter116J
#
# Group=3 axis (3n x 3n grid):
#   124e: g=3 n=1  GRID=3  (9 cells, 128x128, 3 cells/group)
#   124f: g=3 n=2  GRID=6  (36 cells, 64x64, 12 cells/group)
#   124g: g=3 n=3  GRID=9  (81 cells, 43x43, 27 cells/group)
#
# Bisect direction (group=2 special, halves are 2x1 grid not 2x2):
#   124h: bisect_h (top/bottom halves, 384x192 each)
#   124i: bisect_v (left/right halves, 192x384 each)
#
# 9 cells sequential single-GPU, ~80 min total. eval = chip_multilabel_v15direct_n200.

set -e
cd "$(dirname "$0")"

BACKBONE="convnextv2_base.fcmae_ft_in22k_in1k_384"
WEIGHTS="mega_matrix/weights/${BACKBONE}.pth"
EVAL_SET="D:/project/data/wm-811k/chip_multilabel_v15direct_n200"

OUT_BASE="outputs"
SUMMARY="${OUT_BASE}/_iter124_grid_size_sweep_summary.log"
: > "$SUMMARY"
echo "$(date) [iter124] sweep start (clean GRID = n_groups * n)" | tee -a "$SUMMARY"

train_one() {
    local TAG="$1"; local MODE="$2"; local NG="$3"; local N="$4"
    local OUT_ROOT="${OUT_BASE}/iter124_${TAG}"
    if [ -d "$OUT_ROOT" ]; then
        echo "$(date) [iter124] $TAG already exists, skip" | tee -a "$SUMMARY"
        return 0
    fi

    local GRID_FLAG=""
    local NG_FLAG=""
    if [ "$MODE" = "complement" ]; then
        local GRID=$((NG * N))
        GRID_FLAG="--cutmix-grid-dim $GRID"
        NG_FLAG="--cutmix-n-groups $NG"
        echo "$(date) [iter124] TRAIN $TAG (complement g=$NG n=$N GRID=$GRID cells=$((GRID*GRID)))" | tee -a "$SUMMARY"
    else
        echo "$(date) [iter124] TRAIN $TAG (mode=$MODE)" | tee -a "$SUMMARY"
    fi

    python -X utf8 -m chip_multilabel._train_chip_variant \
        --variant T7 --ls 0.30 --batch 2 --accum 8 --seed 1 \
        --epochs 10 --lr 1e-4 \
        --backbone-timm "$BACKBONE" --img-size 384 \
        --backbone-timm-weights "$WEIGHTS" \
        --no-normal --val-criterion margin_max --save-every-epoch \
        --cutmix-mode "$MODE" --cutmix-pair masked --cutmix-pair-fill corner \
        --cutmix-p 0.25 $GRID_FLAG $NG_FLAG --cutmix-complete-label-scale 0.5 \
        --tag "iter124_$TAG" --out-root "$OUT_ROOT" \
        >> "${OUT_BASE}/_iter124_${TAG}_train.log" 2>&1
    local RUN=$(ls -d "$OUT_ROOT"/T*/ 2>/dev/null | head -1)
    [ -n "$RUN" ] && rm -f "$RUN"epoch_*.pth || true
    echo "$(date) [iter124] DONE TRAIN $TAG -> $RUN" | tee -a "$SUMMARY"
}

eval_one() {
    local TAG="$1"
    local OUT_ROOT="${OUT_BASE}/iter124_${TAG}"
    local RUN=$(ls -d "$OUT_ROOT"/T*/ 2>/dev/null | head -1)
    [ -z "$RUN" ] && { echo "[iter124] no run for $TAG"; return 0; }
    local EVAL_OUT="${RUN}eval_v15direct_n200"
    [ -d "$EVAL_OUT" ] && { echo "$(date) [iter124] $TAG eval exists, skip"; return 0; }
    echo "$(date) [iter124] EVAL $TAG" | tee -a "$SUMMARY"
    python -X utf8 -m chip_multilabel.run_stage1 \
        --model "${RUN}best_model.pth" \
        --eval-set "$EVAL_SET" --out-root "$EVAL_OUT" \
        --variants I3,I7,I10,I13 --n-per-class 99999 \
        --strength-min 0.0 --strength-max 1.0 --seed 42 \
        >> "${OUT_BASE}/_iter124_${TAG}_eval.log" 2>&1 || true
}

# === group=2 axis (2n x 2n grid), n ∈ {1,2,3,4} ===
train_one a_g2_n1 complement 2 1; eval_one a_g2_n1
train_one b_g2_n2 complement 2 2; eval_one b_g2_n2
train_one c_g2_n3 complement 2 3; eval_one c_g2_n3
train_one d_g2_n4 complement 2 4; eval_one d_g2_n4

# === group=3 axis (3n x 3n grid), n ∈ {1,2,3} ===
train_one e_g3_n1 complement 3 1; eval_one e_g3_n1
train_one f_g3_n2 complement 3 2; eval_one f_g3_n2
train_one g_g3_n3 complement 3 3; eval_one g_g3_n3

# === bisect direction (group=2 special) ===
train_one h_bisect_h bisect_h 0 0; eval_one h_bisect_h
train_one i_bisect_v bisect_v 0 0; eval_one i_bisect_v

echo "$(date) [iter124] sweep complete" | tee -a "$SUMMARY"
