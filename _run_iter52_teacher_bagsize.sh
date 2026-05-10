#!/bin/bash
# 260510 — iter52: KD teacher-bag-size sweep (paper §6.21 curve)
#  Goal: how does teacher bag size affect student bF1 + α sensitivity?
#  6 cells: 2-bag/3-bag/4-bag/5-bag/6-bag/iter27-14-bag teachers, all student α=0.5 T=4
set -e
cd /d/project/known-cnn

RUN_LOG=outputs/_iter52_teacher_bagsize.log
V15="D:/project/data/wm-811k/chip_multilabel_v15direct"

T2="outputs/_teacher_probs_2bag.parquet"
T3="outputs/_teacher_probs_3bag.parquet"
T4="outputs/_teacher_probs_4bag_new_main.parquet"  # existing (NEW MAIN)
T5="outputs/_teacher_probs_5bag.parquet"
T6="outputs/_teacher_probs_6bag.parquet"
T14="outputs/_teacher_probs_14bag.parquet"  # existing (iter27 composition)

echo "$(date) [iter52] start teacher bag-size sweep" > "$RUN_LOG"

# Compute missing teachers
[ ! -f "$T2" ] && python -m chip_multilabel._kd_make_teacher_probs \
    --out "$T2" --bag-runs "outputs/iter37E_g3_1.0_0.5,outputs/iter33A_alpha03_T4" >> "$RUN_LOG" 2>&1
[ ! -f "$T3" ] && python -m chip_multilabel._kd_make_teacher_probs \
    --out "$T3" --bag-runs "outputs/iter37E_g3_1.0_0.5,outputs/iter33A_alpha03_T4,outputs/iter24_LS030_seed42" >> "$RUN_LOG" 2>&1
[ ! -f "$T5" ] && python -m chip_multilabel._kd_make_teacher_probs \
    --out "$T5" --bag-runs "outputs/iter24_LS030_seed42,outputs/iter26B_g3_LS050,outputs/iter26H_g3_LS067_white,outputs/iter33A_alpha03_T4,outputs/iter37E_g3_1.0_0.5" >> "$RUN_LOG" 2>&1
[ ! -f "$T6" ] && python -m chip_multilabel._kd_make_teacher_probs \
    --out "$T6" --bag-runs "outputs/iter24_LS030_seed42,outputs/iter26B_g3_LS050,outputs/iter26D_g4_LS040,outputs/iter26H_g3_LS067_white,outputs/iter33A_alpha03_T4,outputs/iter37E_g3_1.0_0.5" >> "$RUN_LOG" 2>&1

train_eval() {
    TAG=$1
    EXTRA="$2"
    OUT_ROOT="outputs/iter52${TAG}"
    echo "$(date) [iter52-${TAG}] $EXTRA" >> "$RUN_LOG"
    set +e
    python -m chip_multilabel._train_chip_variant \
        --variant T7 --ls 0.20 --epochs 8 --batch 2 --accum 8 --seed 1 \
        --cutmix-mode complement --cutmix-pair masked --cutmix-pair-fill corner \
        --cutmix-p 0.25 --cutmix-n-groups 3 --cutmix-complete-label-scale 0.50 \
        --kd-skip-on-cutmix --kd-alpha 0.5 --kd-temperature 4.0 \
        ${EXTRA} \
        --out-root "$OUT_ROOT" --tag "iter52${TAG}" \
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
    echo "$(date) [iter52-${TAG}] DONE" >> "$RUN_LOG"
    set -e
}

train_eval A_2bag_teacher "--kd-teacher-probs $T2"
train_eval B_3bag_teacher "--kd-teacher-probs $T3"
train_eval C_4bag_teacher "--kd-teacher-probs $T4"
train_eval D_5bag_teacher "--kd-teacher-probs $T5"
train_eval E_6bag_teacher "--kd-teacher-probs $T6"
train_eval F_14bag_teacher "--kd-teacher-probs $T14"

echo "$(date) [iter52] DONE 6/6" >> "$RUN_LOG"
