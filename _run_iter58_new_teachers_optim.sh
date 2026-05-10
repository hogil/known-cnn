#!/bin/bash
# 260511 — iter58: new teacher composition (pure-asym, pure-KD) + optimization hyperparams
set -e
cd /d/project/known-cnn

RUN_LOG=outputs/_iter58_new_teachers_optim.log
V15="D:/project/data/wm-811k/chip_multilabel_v15direct"

# New teacher: pure-asymmetric (4-bag of asym-only)
T_PUREASYM="outputs/_teacher_probs_4bag_pureAsym.parquet"
# New teacher: pure-KD (4-bag of KD students 33A-D)
T_PUREKD="outputs/_teacher_probs_4bag_pureKD.parquet"
# Existing
T_NEWMAIN="outputs/_teacher_probs_4bag_new_main.parquet"

echo "$(date) [iter58] start" > "$RUN_LOG"

# Compute new teachers if missing
[ ! -f "$T_PUREASYM" ] && python -m chip_multilabel._kd_make_teacher_probs \
    --out "$T_PUREASYM" \
    --bag-runs "outputs/iter37A_g2_1.0_0.5,outputs/iter37D_g2_0.75_1.0,outputs/iter37E_g3_1.0_0.5,outputs/iter37H_g3_0.75_1.0" >> "$RUN_LOG" 2>&1

[ ! -f "$T_PUREKD" ] && python -m chip_multilabel._kd_make_teacher_probs \
    --out "$T_PUREKD" \
    --bag-runs "outputs/iter33A_alpha03_T4,outputs/iter33B_alpha07_T4,outputs/iter33C_alpha05_T2,outputs/iter33D_alpha05_T8" >> "$RUN_LOG" 2>&1

train_eval() {
    TAG=$1
    EXTRA="$2"
    OUT_ROOT="outputs/iter58${TAG}"
    echo "$(date) [iter58-${TAG}] $EXTRA" >> "$RUN_LOG"
    set +e
    python -m chip_multilabel._train_chip_variant \
        --variant T7 --ls 0.20 --epochs 8 --batch 2 --accum 8 --seed 1 \
        --cutmix-mode complement --cutmix-pair masked --cutmix-pair-fill corner \
        --cutmix-p 0.25 --cutmix-n-groups 3 --cutmix-complete-label-scale 0.50 \
        --kd-skip-on-cutmix --kd-temperature 4.0 \
        ${EXTRA} \
        --out-root "$OUT_ROOT" --tag "iter58${TAG}" \
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
    echo "$(date) [iter58-${TAG}] DONE" >> "$RUN_LOG"
    set -e
}

# A. pure-asymmetric 4-bag teacher α=0.5 (NEW teacher composition)
train_eval A_pureAsym_a05 "--kd-teacher-probs $T_PUREASYM --kd-alpha 0.5"

# B. pure-asymmetric 4-bag teacher α=0.3 (smaller α for sharper teacher)
train_eval B_pureAsym_a03 "--kd-teacher-probs $T_PUREASYM --kd-alpha 0.3"

# C. pure-KD 4-bag teacher α=0.5 (KD students as teachers — circular distillation!)
train_eval C_pureKD_a05 "--kd-teacher-probs $T_PUREKD --kd-alpha 0.5"

# D. NEW MAIN teacher α=0.5 + two-LR (backbone vs head)
train_eval D_50B_twoLR "--kd-teacher-probs $T_NEWMAIN --kd-alpha 0.5 --lr-backbone 5e-5 --lr-head 2e-4"

# E. NEW MAIN teacher α=0.5 + warmup-epochs=1 (mild warmup)
train_eval E_50B_warmup1 "--kd-teacher-probs $T_NEWMAIN --kd-alpha 0.5 --warmup-epochs 1"

# F. NEW MAIN teacher α=0.5 + grad-clip=0.5 (vs default 1.0)
train_eval F_50B_gradclip05 "--kd-teacher-probs $T_NEWMAIN --kd-alpha 0.5 --grad-clip 0.5"

echo "$(date) [iter58] DONE 6/6" >> "$RUN_LOG"
