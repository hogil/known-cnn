#!/bin/bash
# 260510 — iter54: non-KD single-model improvements + hyperparameter combinations
#  6 cells, 26B base recipe + various non-KD techniques
set -e
cd /d/project/known-cnn

RUN_LOG=outputs/_iter54_nonKD_recipes.log
V15="D:/project/data/wm-811k/chip_multilabel_v15direct"

echo "$(date) [iter54] start non-KD recipes" > "$RUN_LOG"

train_eval() {
    TAG=$1
    EXTRA="$2"
    OUT_ROOT="outputs/iter54${TAG}"
    echo "$(date) [iter54-${TAG}] $EXTRA" >> "$RUN_LOG"
    set +e
    python -m chip_multilabel._train_chip_variant \
        --variant T7 --ls 0.20 --batch 2 --accum 8 --seed 1 \
        --cutmix-mode complement --cutmix-pair masked --cutmix-pair-fill corner \
        --cutmix-p 0.25 --cutmix-n-groups 3 --cutmix-complete-label-scale 0.50 \
        ${EXTRA} \
        --out-root "$OUT_ROOT" --tag "iter54${TAG}" \
        >> "$RUN_LOG" 2>&1
    RUN=$(ls -d "$OUT_ROOT"/T*/ 2>/dev/null | head -1)
    [ -z "$RUN" ] && return 0
    for SMAX in "1.00 _n200" "0.50 _HARD050"; do
        SVAL=$(echo $SMAX | cut -d' ' -f1)
        SDIR=$(echo $SMAX | cut -d' ' -f2)
        OUT_EVAL="${RUN}eval_v15direct${SDIR}"
        python -m chip_multilabel.run_stage1 --model "${RUN}best_model.pth" \
            --eval-set "$V15" --out-root "$OUT_EVAL" \
            --variants I3,I6,I7,I10 --n-per-class 200 \
            --strength-min 0.0 --strength-max ${SVAL} --seed 42 \
            >> "$RUN_LOG" 2>&1 || true
    done
    echo "$(date) [iter54-${TAG}] DONE" >> "$RUN_LOG"
    set -e
}

# A. EMA teacher (Mean Teacher style) — ema-decay 0.99 (slow EMA update)
train_eval A_EMA099 "--epochs 8 --ema-decay 0.99"

# B. Longer training — epochs 16 (vs 8 default)
train_eval B_epochs16 "--epochs 16"

# C. LR warmup + cosine — warmup 3 epochs
train_eval C_warmup3 "--epochs 8 --warmup-epochs 3 --warmup-start-factor 0.1"

# D. DropPath regularization — 0.1 stochastic depth
train_eval D_droppath01 "--epochs 8 --drop-path-rate 0.1"

# E. Stronger LS at training — ls 0.10 (vs 0.20)
train_eval E_LS010 "--epochs 8 --ls 0.10"

# F. Combined recipe — warmup=2 + drop-path=0.05 + epochs=12
train_eval F_combined "--epochs 12 --warmup-epochs 2 --drop-path-rate 0.05"

echo "$(date) [iter54] DONE 6/6" >> "$RUN_LOG"
