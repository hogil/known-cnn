#!/bin/bash
# 260509 — iter 22: 19C (FCM-PM, paper winner) 3-seed verify + same-mechanism hparam fine-tune sweep.
# 10 trains, dual-eval (v14class + v15direct).
# Base: T7 + LS 0.20 + complement g=2 LS=1.0 masked corner + cutmix-p 0.25, 8 epochs, batch 4 accum 4.
set -e
cd /d/project/known-cnn
LOG=outputs/_iter22_19C_tune.log
echo "$(date) [iter22] 19C tune — 10 trains (3-seed verify + 7 hparam variants)" > "$LOG"

V14="D:/project/data/wm-811k/chip_multilabel_v14class"
V15="D:/project/data/wm-811k/chip_multilabel_v15direct"

# 19C base (FCM-PM): T7 + LS 0.20 + cutmix-p 0.25 complement g=2 LS=1.0 masked corner
BASE19C="--variant T7 --ls 0.20 --cutmix-p 0.25 --cutmix-mode complement --cutmix-n-groups 2 --cutmix-complete-label-scale 1.0 --cutmix-pair masked --cutmix-pair-fill corner"

run_one() {
    TAG=$1
    SEED=$2
    EXTRA="$3"
    echo "$(date) [iter22-${TAG}] seed=${SEED}  ${EXTRA}" >> "$LOG"
    OUT_ROOT="outputs/iter22${TAG}"
    python -m chip_multilabel._train_chip_variant \
        --epochs 8 --batch 4 --accum 4 --seed ${SEED} \
        ${BASE19C} ${EXTRA} \
        --out-root "$OUT_ROOT" \
        --tag "iter22${TAG}_seed${SEED}" \
        >> "$LOG" 2>&1
    RUN=$(ls -d "$OUT_ROOT"/*iter22${TAG}_*/ 2>/dev/null | head -1)
    if [ -z "${RUN}" ]; then
        echo "$(date) [iter22-${TAG}] TRAIN FAILED" >> "$LOG"
        return 1
    fi
    # eval v14class
    python -m chip_multilabel.run_stage1 \
        --model "${RUN}best_model.pth" \
        --eval-set "$V14" \
        --out-root "${RUN}eval_v14class" \
        --variants I3,I6,I7,I10 --n-per-class 50 --strength-min 0.0 --seed 42 \
        >> "$LOG" 2>&1
    # eval v15direct
    python -m chip_multilabel.run_stage1 \
        --model "${RUN}best_model.pth" \
        --eval-set "$V15" \
        --out-root "${RUN}eval_v15direct" \
        --variants I3,I6,I7,I10 --n-per-class 50 --strength-min 0.0 --seed 42 \
        >> "$LOG" 2>&1
    echo "$(date) [iter22-${TAG}] DONE  run=${RUN}" >> "$LOG"
}

# A,B. 3-seed verification (seed 1 already known from 19C baseline)
run_one A_seed7        7   ""
run_one B_seed42       42  ""

# C,D. LS sweep
run_one C_LS010        1   "--ls 0.10"
run_one D_LS030        1   "--ls 0.30"

# E,F. CutMix freq sweep (override --cutmix-p)
run_one E_cutmix015    1   "--cutmix-p 0.15"
run_one F_cutmix040    1   "--cutmix-p 0.40"

# G. drop_path
run_one G_droppath005  1   "--drop-path-rate 0.05"

# H. EMA
run_one H_ema095       1   "--ema-decay 0.95"

# I. warmup
run_one I_warmup2      1   "--warmup-epochs 2"

# J. lr-head ↓
run_one J_lrhead5e5    1   "--lr-head 5e-5"

echo "$(date) [iter22] DONE  10/10" >> "$LOG"
