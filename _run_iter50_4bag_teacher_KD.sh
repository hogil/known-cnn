#!/bin/bash
# 260510 — iter50: 4-bag teacher KD distillation
#  Step 1: compute teacher probs from NEW MAIN 4-bag {24_LS030_s42 + 26H + 33A + 37E}
#  Step 2: train student with KD α=0.3 T=4 (paper main KD recipe)
#  Goal: production single 0.99+ (currently 33A=0.9840, gap to 4-bag=0.9964)
set -e
cd /d/project/known-cnn

RUN_LOG=outputs/_iter50_4bag_teacher_KD.log
V15="D:/project/data/wm-811k/chip_multilabel_v15direct"
TEACHER_PARQUET="outputs/_teacher_probs_4bag_new_main.parquet"

echo "$(date) [iter50] start 4-bag teacher KD" > "$RUN_LOG"

# Step 1: Compute teacher probs (4-bag NEW MAIN)
if [ ! -f "$TEACHER_PARQUET" ]; then
    echo "$(date) [iter50] computing 4-bag teacher probs ..." >> "$RUN_LOG"
    python -m chip_multilabel._kd_make_teacher_probs \
        --out "$TEACHER_PARQUET" \
        --bag-runs "outputs/iter24_LS030_seed42,outputs/iter26H_g3_LS067_white,outputs/iter33A_alpha03_T4,outputs/iter37E_g3_1.0_0.5" \
        >> "$RUN_LOG" 2>&1
fi

# Step 2: Train students with various KD α/T configs
train_eval() {
    TAG=$1
    EXTRA="$2"
    OUT_ROOT="outputs/iter50${TAG}"
    echo "$(date) [iter50-${TAG}] $EXTRA" >> "$RUN_LOG"
    set +e
    python -m chip_multilabel._train_chip_variant \
        --variant T7 --ls 0.20 --epochs 8 --batch 2 --accum 8 --seed 1 \
        --cutmix-mode complement --cutmix-pair masked --cutmix-pair-fill corner \
        --cutmix-p 0.25 --cutmix-n-groups 3 --cutmix-complete-label-scale 0.50 \
        --kd-teacher-probs "$TEACHER_PARQUET" --kd-skip-on-cutmix \
        ${EXTRA} \
        --out-root "$OUT_ROOT" --tag "iter50${TAG}_seed1" \
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
    echo "$(date) [iter50-${TAG}] DONE" >> "$RUN_LOG"
    set -e
}

# Sweep α/T against 4-bag teacher
train_eval A_alpha03_T4 "--kd-alpha 0.3 --kd-temperature 4.0"
train_eval B_alpha05_T4 "--kd-alpha 0.5 --kd-temperature 4.0"
train_eval C_alpha07_T4 "--kd-alpha 0.7 --kd-temperature 4.0"
train_eval D_alpha03_T2 "--kd-alpha 0.3 --kd-temperature 2.0"
train_eval E_alpha03_T8 "--kd-alpha 0.3 --kd-temperature 8.0"

echo "$(date) [iter50] DONE 5/5" >> "$RUN_LOG"
