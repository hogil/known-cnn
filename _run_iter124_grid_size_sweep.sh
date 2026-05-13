#!/bin/bash
# iter124 — FCM-PM TRUE grid cell size ablation (paper §6 NEW axis)
#
# Discovery: GRID is hardcoded at 8 (= 64 cells of 48x48 px at img_size=384).
# n_groups only PARTITIONS those 64 cells. Real cell-pixel-size is GRID-dim.
# Patched trainer with --cutmix-grid-dim flag; this sweep varies GRID at
# FIXED n_groups (3 = iter116J baseline) to isolate cell-size effect.
#
# Baseline (frozen from iter116J): T7 BCE+LS=0.30, lr 1e-4, ep 10,
# val_margin selection, save_every_epoch, cutmix-mode=complement, pair=masked,
# fill=corner, p=0.25, n_groups=3, label-scale=0.5.
#
# Cells (only --cutmix-grid-dim varies):
#   124a: GRID=4    (16 cells, 96x96 px)  - very coarse
#   124b: GRID=6    (36 cells, 64x64 px)  - coarse
#   124c: GRID=8    (64 cells, 48x48 px)  - iter116J baseline (re-run for fair eval)
#   124d: GRID=12   (144 cells, 32x32 px) - fine
#   124e: GRID=16   (256 cells, 24x24 px) - very fine
#
# Plus group=2 bisect direction comparison (independent of GRID axis):
#   124f: bisect_h  (200x100 horizontal halves)
#   124g: bisect_v  (100x200 vertical halves)
#
# 7 cells sequential single-GPU, ~63-70 min total. eval = chip_multilabel_v15direct_n200.

set -e
cd "$(dirname "$0")"

BACKBONE="convnextv2_base.fcmae_ft_in22k_in1k_384"
WEIGHTS="mega_matrix/weights/${BACKBONE}.pth"
EVAL_SET="D:/project/data/wm-811k/chip_multilabel_v15direct_n200"

OUT_BASE="outputs"
SUMMARY="${OUT_BASE}/_iter124_grid_size_sweep_summary.log"
: > "$SUMMARY"
echo "$(date) [iter124] sweep start" | tee -a "$SUMMARY"

train_one() {
    local TAG="$1"; local MODE="$2"; local GRID="$3"
    local OUT_ROOT="${OUT_BASE}/iter124_${TAG}"
    if [ -d "$OUT_ROOT" ]; then
        echo "$(date) [iter124] $TAG already exists, skip" | tee -a "$SUMMARY"
        return 0
    fi
    echo "$(date) [iter124] TRAIN $TAG (mode=$MODE GRID=$GRID)" | tee -a "$SUMMARY"

    # bisect modes ignore GRID/n_groups
    local GRID_FLAG=""
    local NG_FLAG=""
    if [ "$MODE" = "complement" ]; then
        GRID_FLAG="--cutmix-grid-dim $GRID"
        NG_FLAG="--cutmix-n-groups 3"
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

# === 5 GRID-dim cells (fixed n_groups=3) ===
train_one a_grid4  complement 4
eval_one  a_grid4
train_one b_grid6  complement 6
eval_one  b_grid6
train_one c_grid8  complement 8
eval_one  c_grid8
train_one d_grid12 complement 12
eval_one  d_grid12
train_one e_grid16 complement 16
eval_one  e_grid16

# === 2 group=2 bisect direction cells ===
train_one f_bisect_h bisect_h 0
eval_one  f_bisect_h
train_one g_bisect_v bisect_v 0
eval_one  g_bisect_v

echo "$(date) [iter124] sweep complete" | tee -a "$SUMMARY"
