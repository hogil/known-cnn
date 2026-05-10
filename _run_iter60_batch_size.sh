#!/bin/bash
# 260511 — iter60: effective batch size dimension test
#  paper main = batch=2 accum=8 (effective 16). Untested: 8, 32, 64
set -e
cd /d/project/known-cnn

RUN_LOG=outputs/_iter60_batch_size.log
V15="D:/project/data/wm-811k/chip_multilabel_v15direct"
T="outputs/_teacher_probs_4bag_new_main.parquet"

echo "$(date) [iter60] start" > "$RUN_LOG"

train_eval() {
    TAG=$1
    EXTRA="$2"
    OUT_ROOT="outputs/iter60${TAG}"
    echo "$(date) [iter60-${TAG}] $EXTRA" >> "$RUN_LOG"
    set +e
    python -m chip_multilabel._train_chip_variant \
        --variant T7 --ls 0.20 --epochs 8 --seed 1 \
        --cutmix-mode complement --cutmix-pair masked --cutmix-pair-fill corner \
        --cutmix-p 0.25 --cutmix-n-groups 3 --cutmix-complete-label-scale 0.50 \
        --kd-teacher-probs $T --kd-alpha 0.5 --kd-temperature 4.0 --kd-skip-on-cutmix \
        ${EXTRA} \
        --out-root "$OUT_ROOT" --tag "iter60${TAG}" \
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
    echo "$(date) [iter60-${TAG}] DONE" >> "$RUN_LOG"
    set -e
}

# A. effective batch=8 (smaller than paper main 16)
train_eval A_eff08 "--batch 2 --accum 4"

# B. effective batch=32 (2x paper main)
train_eval B_eff32 "--batch 2 --accum 16"

# C. effective batch=32 alt (different physical batch)
train_eval C_eff32_b4 "--batch 4 --accum 8"

# D. effective batch=64 (4x paper main)
train_eval D_eff64 "--batch 2 --accum 32"

# E. physical batch=4 + accum=4 (eff=16, same as paper but different physical)
train_eval E_eff16_b4 "--batch 4 --accum 4"

# F. physical batch=1 + accum=16 (eff=16, single sample BN)
train_eval F_eff16_b1 "--batch 1 --accum 16"

echo "$(date) [iter60] DONE 6/6" >> "$RUN_LOG"
