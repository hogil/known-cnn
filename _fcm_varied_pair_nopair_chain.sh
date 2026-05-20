#!/bin/bash
# FCM-PM varied hparam pair vs nopair — matched pairs at same condition
# Goal: paper-grade pair-mask FAR-essential claim across condition variations
# Waits for current sweep (_fcm_nopair_sweep_chain.sh) to complete first
set -u
cd "$(dirname "$0")"
LOG="outputs/_fcm_varied.log"
: > "$LOG"
log() { echo "$(date) $1" | tee -a "$LOG"; }

BACKBONE="convnextv2_base.fcmae_ft_in22k_in1k_384"
WEIGHTS="models/${BACKBONE}.pth"
DATA="E:/data/images/classification_chips"
EVAL="E:/data/images/chip_multilabel_v15direct_n2000"

# Wait for current sweep to finish
log "[varied] waiting for fcm_nopair_sweep to complete"
waited=0
while [ $waited -lt 9000 ]; do
    if grep -q "all 7 trains complete" outputs/_fcm_nopair_sweep.log 2>/dev/null; then
        log "[varied] sweep done, proceeding"
        break
    fi
    sleep 60
    waited=$((waited + 60))
done

train_fcm_eval() {
    local TAG="$1"; local PAIR="$2"; local LS="$3"; local G="$4"; local CLS="$5"; local P="$6"
    local OUT="outputs/fcm_${TAG}"
    [ -d "$OUT" ] && rm -rf "$OUT"
    mkdir -p "$OUT"
    log "[varied] TRAIN $TAG pair=$PAIR LS=$LS g=$G cls=$CLS p=$P seed=42"
    python -u -X utf8 -m chip_multilabel._train_chip_variant \
        --variant T7 --ls "$LS" --batch 2 --accum 8 --seed 42 --lr 1e-4 --epochs 8 \
        --data-root "$DATA" \
        --backbone-timm "$BACKBONE" --img-size 384 --backbone-timm-weights "$WEIGHTS" \
        --no-normal --save-every-epoch --grad-checkpointing \
        --cutmix-mode complement --cutmix-pair "$PAIR" --cutmix-pair-fill corner \
        --cutmix-p "$P" --cutmix-n-groups "$G" --cutmix-complete-label-scale "$CLS" \
        --tag "T7_fcm_$TAG" --out-root "$OUT" \
        2>&1 | tail -n 15 >> "$LOG"
    local CKPT=$(find "$OUT" -name best_model.pth 2>/dev/null | head -1)
    if [ -z "$CKPT" ]; then log "[varied] FAIL $TAG"; return 1; fi
    find "$OUT" -name "epoch_*_model.pth" -delete 2>/dev/null
    find "$OUT" -name "final_epoch_model.pth" -delete 2>/dev/null
    local RUN=$(dirname "$CKPT")
    log "[varied] EVAL $TAG"
    python -u -X utf8 -m chip_multilabel.run_stage1 \
        --model "$CKPT" --eval-set "$EVAL" --out-root "$RUN/eval_n2000_pred" \
        --variants I3,I7,I10,I13 --n-per-class 2000 \
        --strength-min 0.0 --strength-max 1.0 --seed 42 --error-cap 0 \
        --batch-size 32 --num-workers 4 \
        2>&1 | tail -n 8 >> "$LOG"
    log "[varied] DONE $TAG"
}

# 5 conditions x 2 sides (pair vs nopair) = 10 trains, all seed=42
# Condition variation: g, cls, LS (not p — already done in Row 5)

# g variation (cls=0.5 LS=0.30 p=0.25)
train_fcm_eval "pair_g2"     masked  0.30 2 0.5 0.25
train_fcm_eval "nopair_g2"   none    0.30 2 0.5 0.25
train_fcm_eval "pair_g4"     masked  0.30 4 0.5 0.25
train_fcm_eval "nopair_g4"   none    0.30 4 0.5 0.25

# cls variation (g=3 LS=0.30 p=0.25)
train_fcm_eval "pair_cls03"  masked  0.30 3 0.3 0.25
train_fcm_eval "nopair_cls03" none   0.30 3 0.3 0.25
train_fcm_eval "pair_cls07"  masked  0.30 3 0.7 0.25
train_fcm_eval "nopair_cls07" none   0.30 3 0.7 0.25

# LS variation (g=3 cls=0.5 p=0.25)
train_fcm_eval "pair_ls40"   masked  0.40 3 0.5 0.25
train_fcm_eval "nopair_ls40" none    0.40 3 0.5 0.25

log "[varied] all 10 trains complete"
