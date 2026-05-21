#!/bin/bash
# FCM-PM margin-max chain: 5 conditions × pair/nopair = 10 trains
# val_margin best ckpt (best_from_epoch=6) + save_every_epoch + multi_val_set
# Both best+final eval, full stderr captured (no tail truncation)
set -u
cd "$(dirname "$0")"
LOG="outputs/_fcm_margin.log"
: > "$LOG"
log() { echo "$(date) $1" | tee -a "$LOG"; }

BACKBONE="convnextv2_base.fcmae_ft_in22k_in1k_384"
WEIGHTS="models/${BACKBONE}.pth"
DATA="E:/data/images/classification_chips"
EVAL="E:/data/images/chip_multilabel_v15direct_n2000"

train_and_eval() {
    local TAG="$1"; local PAIR="$2"; local LS="$3"; local G="$4"; local CLS="$5"; local SEED="$6"
    local OUT="outputs/fcm_margin_${TAG}"
    [ -d "$OUT" ] && rm -rf "$OUT"
    mkdir -p "$OUT"
    log "[margin] TRAIN $TAG pair=$PAIR LS=$LS g=$G cls=$CLS seed=$SEED"
    # full stderr capture (tee fork, no tail truncation)
    timeout 1800 python -u -X utf8 -m chip_multilabel._train_chip_variant \
        --variant T7 --ls "$LS" --batch 2 --accum 8 --seed "$SEED" --lr 1e-4 --epochs 8 \
        --val-criterion margin_max --best-from-epoch 6 --save-every-epoch \
        --multi-val-set "$EVAL" --multi-val-n-per-class 50 \
        --data-root "$DATA" \
        --backbone-timm "$BACKBONE" --img-size 384 --backbone-timm-weights "$WEIGHTS" \
        --no-normal --grad-checkpointing \
        --cutmix-mode complement --cutmix-pair "$PAIR" --cutmix-pair-fill corner \
        --cutmix-p 0.25 --cutmix-n-groups "$G" --cutmix-complete-label-scale "$CLS" \
        --tag "T7_margin_$TAG" --out-root "$OUT" \
        > "$OUT/train.log" 2>&1
    local RC=$?
    tail -n 40 "$OUT/train.log" >> "$LOG"
    if [ $RC -ne 0 ]; then
        log "[margin] FAIL $TAG (rc=$RC) — see $OUT/train.log"
        return
    fi

    local FINAL=$(find "$OUT" -name "final_epoch_model.pth" 2>/dev/null | head -1)
    local BEST=$(find "$OUT" -name "best_model.pth" 2>/dev/null | head -1)
    local RUN=$(dirname "$BEST")

    # Eval best_model.pth (margin-max best ckpt, from epoch>=6)
    if [ -n "$BEST" ]; then
        log "[margin] EVAL $TAG best_model.pth (margin_max ep>=6)"
        timeout 1800 python -u -X utf8 -m chip_multilabel.run_stage1 \
            --model "$BEST" --eval-set "$EVAL" --out-root "$RUN/eval_n2000_margin_best" \
            --variants I3,I7,I10,I13 --n-per-class 2000 \
            --strength-min 0.0 --strength-max 1.0 --seed 42 --error-cap 0 \
            --batch-size 32 --num-workers 4 \
            > "$RUN/eval_n2000_margin_best.log" 2>&1
        tail -n 20 "$RUN/eval_n2000_margin_best.log" >> "$LOG"
    fi

    # Eval final_epoch_model.pth (ep8 baseline)
    if [ -n "$FINAL" ]; then
        log "[margin] EVAL $TAG final_epoch_model.pth (ep8)"
        timeout 1800 python -u -X utf8 -m chip_multilabel.run_stage1 \
            --model "$FINAL" --eval-set "$EVAL" --out-root "$RUN/eval_n2000_margin_final" \
            --variants I3,I7,I10,I13 --n-per-class 2000 \
            --strength-min 0.0 --strength-max 1.0 --seed 42 --error-cap 0 \
            --batch-size 32 --num-workers 4 \
            > "$RUN/eval_n2000_margin_final.log" 2>&1
        tail -n 20 "$RUN/eval_n2000_margin_final.log" >> "$LOG"
    fi

    # Keep best + final, delete intermediate per-epoch ckpts
    find "$OUT" -name "epoch_*_model.pth" -delete 2>/dev/null
    log "[margin] DONE $TAG"
}

log "[margin] starting val_margin chain — 10 trains (5 cond x pair/nopair)"
log "[margin] spec: --val-criterion margin_max --best-from-epoch 6 --save-every-epoch --multi-val-set"

# 1. g=2 cls=0.5 LS=0.30 (retry — prev FAIL both)
train_and_eval "g2_cls05_ls30_pair"   masked  0.30  2  0.5  42
train_and_eval "g2_cls05_ls30_nopair" none    0.30  2  0.5  42

# 2. g=3 cls=0.3 LS=0.30
train_and_eval "g3_cls03_ls30_pair"   masked  0.30  3  0.3  42
train_and_eval "g3_cls03_ls30_nopair" none    0.30  3  0.3  42

# 3. g=3 cls=0.5 LS=0.40
train_and_eval "g3_cls05_ls40_pair"   masked  0.40  3  0.5  42
train_and_eval "g3_cls05_ls40_nopair" none    0.40  3  0.5  42

# 4. g=3 cls=0.7 LS=0.30
train_and_eval "g3_cls07_ls30_pair"   masked  0.30  3  0.7  42
train_and_eval "g3_cls07_ls30_nopair" none    0.30  3  0.7  42

# 5. g=4 cls=0.5 LS=0.30
train_and_eval "g4_cls05_ls30_pair"   masked  0.30  4  0.5  42
train_and_eval "g4_cls05_ls30_nopair" none    0.30  4  0.5  42

log "[margin] all 10 margin-chain trains done"
