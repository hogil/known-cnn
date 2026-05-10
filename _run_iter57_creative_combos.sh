#!/bin/bash
# 260511 — iter57: untested creative combinations
#  6 cells: A T9+KD, B T7+KD+drop-path, C T7+KD+ep10, D T7+multi-teacher α=0.3, E pair-loss-w=2.0, F grid-mode KD
set -e
cd /d/project/known-cnn

RUN_LOG=outputs/_iter57_creative_combos.log
V15="D:/project/data/wm-811k/chip_multilabel_v15direct"
T_NEWMAIN="outputs/_teacher_probs_4bag_new_main.parquet"
T_PUREHARD="outputs/_teacher_probs_4bag_pureHard.parquet"
T_MULTI_AC="outputs/_teacher_probs_multi_AC.parquet"

echo "$(date) [iter57] start creative combos" > "$RUN_LOG"

train_eval() {
    TAG=$1
    EXTRA="$2"
    OUT_ROOT="outputs/iter57${TAG}"
    echo "$(date) [iter57-${TAG}] $EXTRA" >> "$RUN_LOG"
    set +e
    python -m chip_multilabel._train_chip_variant \
        --ls 0.20 --epochs 8 --batch 2 --accum 8 --seed 1 \
        --cutmix-pair masked --cutmix-pair-fill corner \
        --cutmix-p 0.25 --cutmix-n-groups 3 --cutmix-complete-label-scale 0.50 \
        ${EXTRA} \
        --out-root "$OUT_ROOT" --tag "iter57${TAG}" \
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
    echo "$(date) [iter57-${TAG}] DONE" >> "$RUN_LOG"
    set -e
}

# A. T9 sigmoid focal + 4-bag KD (focal + KD combine, NEVER tested)
train_eval A_T9_KD "--variant T9 --cutmix-mode complement --kd-teacher-probs $T_NEWMAIN --kd-alpha 0.5 --kd-temperature 4.0 --kd-skip-on-cutmix"

# B. T7 + KD + drop-path 0.05 (light reg + KD)
train_eval B_T7KD_droppath "--variant T7 --cutmix-mode complement --drop-path-rate 0.05 --kd-teacher-probs $T_NEWMAIN --kd-alpha 0.5 --kd-temperature 4.0 --kd-skip-on-cutmix"

# C. T7 + KD + epoch=10 (between 8 and 12)
train_eval C_T7KD_ep10 "--variant T7 --cutmix-mode complement --epochs 10 --kd-teacher-probs $T_NEWMAIN --kd-alpha 0.5 --kd-temperature 4.0 --kd-skip-on-cutmix"

# D. T7 + multi-teacher (NEW MAIN + pure-hard) α=0.3 (vs iter53B α=0.5 FAIL)
train_eval D_T7_multi_AC_a03 "--variant T7 --cutmix-mode complement --kd-teacher-probs $T_MULTI_AC --kd-alpha 0.3 --kd-temperature 4.0 --kd-skip-on-cutmix"

# E. T7 + KD + pair-loss-w=2.0 (heavier pair, vs iter49E without KD = 0.9677)
train_eval E_T7KD_pair_w20 "--variant T7 --cutmix-mode complement --cutmix-pair-loss-w 2.0 --kd-teacher-probs $T_NEWMAIN --kd-alpha 0.5 --kd-temperature 4.0 --kd-skip-on-cutmix"

# F. T7 + KD + grid mode (vs complement)
train_eval F_T7KD_grid "--variant T7 --cutmix-mode grid --kd-teacher-probs $T_NEWMAIN --kd-alpha 0.5 --kd-temperature 4.0 --kd-skip-on-cutmix"

echo "$(date) [iter57] DONE 6/6" >> "$RUN_LOG"
