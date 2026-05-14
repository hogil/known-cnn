#!/bin/bash
# Retrain pure-hard 4-bag winner: 24_LS030_seed42 + 26B + 26D + 26H
# All FCM-PM val_margin, complement, ConvNeXtV2-Base FCMAE 384, batch 2 accum 8, ep=10.

set -e
cd "$(dirname "$0")"

export WM811K_ROOT="E:/data/images"
BACKBONE="convnextv2_base.fcmae_ft_in22k_in1k_384"
WEIGHTS="models/${BACKBONE}.pth"
EVAL_SET="E:/data/images/chip_multilabel_v15direct_n2000"
SUMMARY="outputs/_4bag_winner_summary.log"

train_one() {
    local TAG="$1"; local SEED="$2"; local LS="$3"; local GRID="$4"; local NG="$5"; local FILL="${6:-corner}"
    local OUT_ROOT="outputs/${TAG}"
    if [ -d "$OUT_ROOT" ] && find "$OUT_ROOT" -maxdepth 2 -name "best_model.pth" 2>/dev/null | grep -q .; then
        echo "$(date) [4BAG] $TAG already done, skip" | tee -a "$SUMMARY"
        return 0
    fi
    rm -rf "$OUT_ROOT"
    echo "$(date) [4BAG] TRAIN $TAG (seed=$SEED LS=$LS grid=$GRID g=$NG fill=$FILL)" | tee -a "$SUMMARY"
    python -X utf8 -m chip_multilabel._train_chip_variant \
        --variant T7 --ls "$LS" --batch 2 --accum 8 --seed "$SEED" \
        --epochs 10 --lr 1e-4 \
        --backbone-timm "$BACKBONE" --img-size 384 \
        --backbone-timm-weights "$WEIGHTS" \
        --no-normal --val-criterion margin_max --save-every-epoch \
        --multi-val-set "$EVAL_SET" --multi-val-n-per-class 50 \
        --gpu-mem-fraction 0.30 \
        --cutmix-mode complement --cutmix-pair masked --cutmix-pair-fill "$FILL" \
        --cutmix-p 0.25 --cutmix-grid-dim "$GRID" --cutmix-n-groups "$NG" --cutmix-complete-label-scale 0.5 \
        --tag "$TAG" --out-root "$OUT_ROOT" \
        > "outputs/_${TAG}_train.log" 2>&1
    local RUN=$(ls -d "$OUT_ROOT"/*/ 2>/dev/null | head -1)
    [ -n "$RUN" ] && rm -f "$RUN"epoch_*.pth "$RUN"final_epoch_model.pth || true
    echo "$(date) [4BAG] DONE TRAIN $TAG" | tee -a "$SUMMARY"
}

eval_one() {
    local TAG="$1"
    local OUT_ROOT="outputs/${TAG}"
    local RUN=$(ls -d "$OUT_ROOT"/*/ 2>/dev/null | while read d; do [ -f "${d}best_model.pth" ] && echo "$d"; done | head -1)
    [ -z "$RUN" ] && return 0
    local EVAL_OUT="${RUN}eval_n2000_pred"
    if [ -d "$EVAL_OUT" ] && find "$EVAL_OUT" -name "preds_chip.parquet" 2>/dev/null | grep -q .; then
        return 0
    fi
    echo "$(date) [4BAG] EVAL $TAG" | tee -a "$SUMMARY"
    python -X utf8 -m chip_multilabel.run_stage1 \
        --model "${RUN}best_model.pth" \
        --eval-set "$EVAL_SET" --out-root "$EVAL_OUT" \
        --variants I3,I7,I10,I13 --n-per-class 2000 \
        --strength-min 0.0 --strength-max 1.0 --seed 42 \
        >> "outputs/_${TAG}_eval.log" 2>&1
    echo "$(date) [4BAG] DONE EVAL $TAG" | tee -a "$SUMMARY"
}

echo "$(date) [4BAG] start (4 cells: 24_LS030_seed42 + 26B + 26D + 26H)" | tee -a "$SUMMARY"

# Bag 1: 24_LS030_seed42 — g=2 LS=0.30 seed=42 (anchor)
train_one "bag1_24_g2_ls30_s42" 42 0.30 4 2 corner
eval_one  "bag1_24_g2_ls30_s42"

# Bag 2: 26B — g=3 LS=0.50 (best single of iter26)
train_one "bag2_26B_g3_ls50" 42 0.50 6 3 corner
eval_one  "bag2_26B_g3_ls50"

# Bag 3: 26D — g=4 LS=0.40
train_one "bag3_26D_g4_ls40" 42 0.40 8 4 corner
eval_one  "bag3_26D_g4_ls40"

# Bag 4: 26H — g=3 LS=0.67 white-fill (diversity)
train_one "bag4_26H_g3_ls67_white" 42 0.67 6 3 white
eval_one  "bag4_26H_g3_ls67_white"

echo "$(date) [4BAG] ALL DONE" | tee -a "$SUMMARY"
