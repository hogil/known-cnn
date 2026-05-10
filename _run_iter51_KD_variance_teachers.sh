#!/bin/bash
# 260510 — iter51: iter50B variance + alternate teacher composition + finer α grid
#  6 cells × 2-strength eval (FULL n=200 + HARD050)
set -e
cd /d/project/known-cnn

RUN_LOG=outputs/_iter51_KD_variance_teachers.log
V15="D:/project/data/wm-811k/chip_multilabel_v15direct"

# Teacher 1: NEW MAIN all-axes 4-bag (existing)
TEACHER_NEW_MAIN="outputs/_teacher_probs_4bag_new_main.parquet"
# Teacher 2: NEW HEADLINE pure-hard 4-bag
TEACHER_PURE_HARD="outputs/_teacher_probs_4bag_pureHard.parquet"
# Teacher 3: iter33 4-bag (paper §5.21)
TEACHER_ITER33="outputs/_teacher_probs_4bag_iter33.parquet"

echo "$(date) [iter51] start KD variance + teachers" > "$RUN_LOG"

# Compute teacher probs if missing
if [ ! -f "$TEACHER_PURE_HARD" ]; then
    echo "$(date) [iter51] compute pure-hard 4-bag teacher" >> "$RUN_LOG"
    python -m chip_multilabel._kd_make_teacher_probs \
        --out "$TEACHER_PURE_HARD" \
        --bag-runs "outputs/iter24_LS030_seed42,outputs/iter26B_g3_LS050,outputs/iter26D_g4_LS040,outputs/iter26H_g3_LS067_white" \
        >> "$RUN_LOG" 2>&1
fi

if [ ! -f "$TEACHER_ITER33" ]; then
    echo "$(date) [iter51] compute iter33 4-bag teacher" >> "$RUN_LOG"
    python -m chip_multilabel._kd_make_teacher_probs \
        --out "$TEACHER_ITER33" \
        --bag-runs "outputs/iter26B_g3_LS050,outputs/iter21F_19E_repeat,outputs/iter21H_19I_repeat,outputs/iter26D_g4_LS040" \
        >> "$RUN_LOG" 2>&1
fi

train_eval() {
    TAG=$1
    EXTRA="$2"
    OUT_ROOT="outputs/iter51${TAG}"
    echo "$(date) [iter51-${TAG}] $EXTRA" >> "$RUN_LOG"
    set +e
    python -m chip_multilabel._train_chip_variant \
        --variant T7 --ls 0.20 --epochs 8 --batch 2 --accum 8 \
        --cutmix-mode complement --cutmix-pair masked --cutmix-pair-fill corner \
        --cutmix-p 0.25 --cutmix-n-groups 3 --cutmix-complete-label-scale 0.50 \
        --kd-skip-on-cutmix \
        ${EXTRA} \
        --out-root "$OUT_ROOT" --tag "iter51${TAG}" \
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
    echo "$(date) [iter51-${TAG}] DONE" >> "$RUN_LOG"
    set -e
}

# A,B: iter50B (α=0.5 T=4 all-axes teacher) variance
train_eval A_50B_seed7 "--seed 7 --kd-teacher-probs $TEACHER_NEW_MAIN --kd-alpha 0.5 --kd-temperature 4.0"
train_eval B_50B_seed42 "--seed 42 --kd-teacher-probs $TEACHER_NEW_MAIN --kd-alpha 0.5 --kd-temperature 4.0"

# C: pure-hard 4-bag teacher (NEW HEADLINE) student
train_eval C_pureHard_teacher "--seed 1 --kd-teacher-probs $TEACHER_PURE_HARD --kd-alpha 0.5 --kd-temperature 4.0"

# D: iter33 4-bag teacher student
train_eval D_iter33_teacher "--seed 1 --kd-teacher-probs $TEACHER_ITER33 --kd-alpha 0.5 --kd-temperature 4.0"

# E: finer α=0.4 (between 0.3 and 0.5) with all-axes teacher
train_eval E_alpha04 "--seed 1 --kd-teacher-probs $TEACHER_NEW_MAIN --kd-alpha 0.4 --kd-temperature 4.0"

# F: finer α=0.55 (between 0.5 and 0.7) with all-axes teacher
train_eval F_alpha055 "--seed 1 --kd-teacher-probs $TEACHER_NEW_MAIN --kd-alpha 0.55 --kd-temperature 4.0"

echo "$(date) [iter51] DONE 6/6" >> "$RUN_LOG"
