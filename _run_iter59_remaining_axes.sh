#!/bin/bash
# 260511 — iter59: remaining hyperparam axes (cutmix-discount, n-patches, grad-clip + finer α)
set -e
cd /d/project/known-cnn

RUN_LOG=outputs/_iter59_remaining_axes.log
V15="D:/project/data/wm-811k/chip_multilabel_v15direct"
T="outputs/_teacher_probs_4bag_new_main.parquet"

echo "$(date) [iter59] start" > "$RUN_LOG"

train_eval() {
    TAG=$1
    EXTRA="$2"
    OUT_ROOT="outputs/iter59${TAG}"
    echo "$(date) [iter59-${TAG}] $EXTRA" >> "$RUN_LOG"
    set +e
    python -m chip_multilabel._train_chip_variant \
        --variant T7 --ls 0.20 --epochs 8 --batch 2 --accum 8 --seed 1 \
        --cutmix-mode complement --cutmix-pair masked --cutmix-pair-fill corner \
        --cutmix-p 0.25 --cutmix-n-groups 3 --cutmix-complete-label-scale 0.50 \
        --kd-teacher-probs $T --kd-temperature 4.0 --kd-skip-on-cutmix \
        ${EXTRA} \
        --out-root "$OUT_ROOT" --tag "iter59${TAG}" \
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
    echo "$(date) [iter59-${TAG}] DONE" >> "$RUN_LOG"
    set -e
}

# A. α=0.45 (between 0.4 FAIL and 0.5 PASS, finer)
train_eval A_alpha045 "--kd-alpha 0.45"

# B. α=0.55 (between 0.5 PASS and 0.7 FAIL — iter51F was 0.55 FAIL but worth retry)
train_eval B_alpha055 "--kd-alpha 0.55"

# C. cutmix-discount 0.5 (vs 0.7 default)
train_eval C_discount05 "--kd-alpha 0.5 --cutmix-discount 0.5"

# D. cutmix-discount 0.9 (vs 0.7)
train_eval D_discount09 "--kd-alpha 0.5 --cutmix-discount 0.9"

# E. cutmix-grid-prob 0.3 (vs 0.5 default)
train_eval E_gridprob03 "--kd-alpha 0.5 --cutmix-grid-prob 0.3"

# F. grad-clip 2.0 (vs 1.0 default)
train_eval F_gradclip2 "--kd-alpha 0.5 --grad-clip 2.0"

echo "$(date) [iter59] DONE 6/6" >> "$RUN_LOG"
