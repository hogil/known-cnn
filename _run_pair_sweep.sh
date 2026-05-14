#!/bin/bash
# Pair-mask sweep: re-test ladder5 with proper grid/p variations
# User directive (2026-05-15): "조건 바꿔서 다시해봐라 p를 바꾸던가 rect말고 grid 일부선택 + pair로"
# Tests:
#   ladder5b — complement g=4 n=2 + pair masked corner + p=0.25 (proper FCM-PM, seed=1 fresh)
#   ladder5c — single rect + pair masked corner + p=0.50 (cutmix-p sweep on existing setup)
#   ladder5d — complement g=4 n=2 + pair masked corner + p=1.00 (proper grid+pair, max p)

set -e
cd "$(dirname "$0")"

export WM811K_ROOT="E:/data/images"
BACKBONE="convnextv2_base.fcmae_ft_in22k_in1k_384"
WEIGHTS="models/${BACKBONE}.pth"
EVAL_SET="E:/data/images/chip_multilabel_v15direct_n2000"
SUMMARY="outputs/_pair_sweep_summary.log"

train_one() {
    local TAG="$1"; shift
    local OUT_ROOT="outputs/${TAG}"
    if [ -d "$OUT_ROOT" ] && find "$OUT_ROOT" -maxdepth 2 -name "best_model.pth" 2>/dev/null | grep -q .; then
        echo "$(date) [PAIR-SWEEP] $TAG already done, skip" | tee -a "$SUMMARY"
        return 0
    fi
    rm -rf "$OUT_ROOT"
    echo "$(date) [PAIR-SWEEP] TRAIN $TAG ($*)" | tee -a "$SUMMARY"
    python -X utf8 -m chip_multilabel._train_chip_variant \
        --variant T7 --ls 0.30 --batch 2 --accum 8 --seed 1 \
        --epochs 10 --lr 1e-4 \
        --backbone-timm "$BACKBONE" --img-size 384 \
        --backbone-timm-weights "$WEIGHTS" \
        --no-normal --val-criterion margin_max --save-every-epoch \
        --multi-val-set "$EVAL_SET" --multi-val-n-per-class 50 \
        --gpu-mem-fraction 0.40 --grad-checkpointing \
        --tag "$TAG" --out-root "$OUT_ROOT" \
        "$@" \
        > "outputs/_${TAG}_train.log" 2>&1
    local RUN=$(ls -d "$OUT_ROOT"/*/ 2>/dev/null | head -1)
    [ -n "$RUN" ] && rm -f "$RUN"epoch_*.pth "$RUN"final_epoch_model.pth || true
    echo "$(date) [PAIR-SWEEP] DONE TRAIN $TAG" | tee -a "$SUMMARY"
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
    echo "$(date) [PAIR-SWEEP] EVAL $TAG" | tee -a "$SUMMARY"
    python -X utf8 -m chip_multilabel.run_stage1 \
        --model "${RUN}best_model.pth" \
        --eval-set "$EVAL_SET" --out-root "$EVAL_OUT" \
        --variants I3,I7,I10,I13 --n-per-class 2000 \
        --strength-min 0.0 --strength-max 1.0 --seed 42 \
        >> "outputs/_${TAG}_eval.log" 2>&1
    echo "$(date) [PAIR-SWEEP] DONE EVAL $TAG" | tee -a "$SUMMARY"
}

echo "$(date) [PAIR-SWEEP] start (3 new ladder5 variants)" | tee -a "$SUMMARY"

# 5b: complement g=4 n=2 + pair masked corner + p=0.25 (proper FCM-PM, seed=1 fresh)
train_one "ladder5b_complement_g4n2_p25" \
    --cutmix-mode complement --cutmix-pair masked --cutmix-pair-fill corner \
    --cutmix-p 0.25 --cutmix-grid-dim 4 --cutmix-n-groups 2 --cutmix-complete-label-scale 0.5
eval_one  "ladder5b_complement_g4n2_p25"

# 5c: single + pair masked corner + p=0.50 (cutmix-p sweep, single mode)
train_one "ladder5c_single_pair_p50" \
    --cutmix-mode single --cutmix-pair masked --cutmix-pair-fill corner \
    --cutmix-p 0.50
eval_one  "ladder5c_single_pair_p50"

# 5d: complement g=4 n=2 + pair masked corner + p=1.00 (proper grid+pair, always)
train_one "ladder5d_complement_g4n2_p100" \
    --cutmix-mode complement --cutmix-pair masked --cutmix-pair-fill corner \
    --cutmix-p 1.00 --cutmix-grid-dim 4 --cutmix-n-groups 2 --cutmix-complete-label-scale 0.5
eval_one  "ladder5d_complement_g4n2_p100"

# 5e: complement g=4 n=2 + NO pair + p=0.25 (FCM-only, isolates FCM vs pair-mask contribution)
train_one "ladder5e_fcm_only_g4n2_p25" \
    --cutmix-mode complement --cutmix-pair none \
    --cutmix-p 0.25 --cutmix-grid-dim 4 --cutmix-n-groups 2 --cutmix-complete-label-scale 0.5
eval_one  "ladder5e_fcm_only_g4n2_p25"

# 5f: complement g=4 n=2 + NO pair + p=1.00 (FCM-only at max p, p×mode axis fill)
train_one "ladder5f_fcm_only_g4n2_p100" \
    --cutmix-mode complement --cutmix-pair none \
    --cutmix-p 1.00 --cutmix-grid-dim 4 --cutmix-n-groups 2 --cutmix-complete-label-scale 0.5
eval_one  "ladder5f_fcm_only_g4n2_p100"

echo "$(date) [PAIR-SWEEP] ALL DONE" | tee -a "$SUMMARY"
