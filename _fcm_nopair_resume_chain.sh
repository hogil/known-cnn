#!/bin/bash
# Resume FCM nopair sweep — eval s7 (already trained), continue s11/s23/s33/ls20_s1/ls20_s77
set -u
cd "$(dirname "$0")"
LOG="outputs/_fcm_nopair_resume.log"
: > "$LOG"
log() { echo "$(date) $1" | tee -a "$LOG"; }

BACKBONE="convnextv2_base.fcmae_ft_in22k_in1k_384"
WEIGHTS="models/${BACKBONE}.pth"
DATA="E:/data/images/classification_chips"
EVAL="E:/data/images/chip_multilabel_v15direct_n2000"

eval_existing() {
    local TAG="$1"
    local CKPT=$(find "outputs/fcm_nopair_${TAG}" -name best_model.pth 2>/dev/null | head -1)
    if [ -z "$CKPT" ]; then log "[resume] SKIP $TAG (no ckpt)"; return 1; fi
    local RUN=$(dirname "$CKPT")
    # skip if already evaled
    if find "$RUN/eval_n2000_pred" -name preds_chip.parquet 2>/dev/null | head -1 | grep -q parquet; then
        log "[resume] SKIP $TAG (eval done)"; return 0
    fi
    log "[resume] EVAL $TAG"
    timeout 600 python -u -X utf8 -m chip_multilabel.run_stage1 \
        --model "$CKPT" --eval-set "$EVAL" --out-root "$RUN/eval_n2000_pred" \
        --variants I3,I7,I10,I13 --n-per-class 2000 \
        --strength-min 0.0 --strength-max 1.0 --seed 42 --error-cap 0 \
        --batch-size 32 --num-workers 0 \
        2>&1 | tail -n 10 >> "$LOG"
    log "[resume] DONE EVAL $TAG"
}

train_eval() {
    local TAG="$1"; local SEED="$2"; local LS="$3"
    local OUT="outputs/fcm_nopair_${TAG}"
    if [ -f "$(find "$OUT" -name best_model.pth 2>/dev/null | head -1)" ]; then
        log "[resume] SKIP TRAIN $TAG (ckpt exists), eval only"
        eval_existing "$TAG"
        return
    fi
    [ -d "$OUT" ] && rm -rf "$OUT"
    mkdir -p "$OUT"
    log "[resume] TRAIN $TAG seed=$SEED LS=$LS"
    timeout 1500 python -u -X utf8 -m chip_multilabel._train_chip_variant \
        --variant T7 --ls "$LS" --batch 2 --accum 8 --seed "$SEED" --lr 1e-4 --epochs 8 \
        --data-root "$DATA" \
        --backbone-timm "$BACKBONE" --img-size 384 --backbone-timm-weights "$WEIGHTS" \
        --no-normal --save-every-epoch --grad-checkpointing \
        --cutmix-mode complement --cutmix-pair none --cutmix-pair-fill corner \
        --cutmix-p 0.25 --cutmix-n-groups 3 --cutmix-complete-label-scale 0.5 \
        --tag "T7_fcm_$TAG" --out-root "$OUT" \
        2>&1 | tail -n 15 >> "$LOG"
    find "$OUT" -name "epoch_*_model.pth" -delete 2>/dev/null
    find "$OUT" -name "final_epoch_model.pth" -delete 2>/dev/null
    eval_existing "$TAG"
    log "[resume] DONE $TAG"
}

log "[resume] starting"
# 1. Eval s7 (already trained)
eval_existing "ls30_s7"

# 2. LS=0.30 remaining seeds
train_eval "ls30_s77"  77 0.30
train_eval "ls30_s11"  11 0.30
train_eval "ls30_s23"  23 0.30
train_eval "ls30_s33"  33 0.30

# 3. LS=0.20 sweep
train_eval "ls20_s1"    1 0.20
train_eval "ls20_s77"  77 0.20

log "[resume] all done"
