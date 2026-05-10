#!/bin/bash
# 260510 — iter53: multi-teacher KD fusion + iter33-teacher α tuning
#  6 cells: 3 multi-teacher fusion + 2 iter33 α tuning + 1 control
set -e
cd /d/project/known-cnn

RUN_LOG=outputs/_iter53_multi_teacher.log
V15="D:/project/data/wm-811k/chip_multilabel_v15direct"

T_NEWMAIN="outputs/_teacher_probs_4bag_new_main.parquet"
T_PUREHARD="outputs/_teacher_probs_4bag_pureHard.parquet"
T_ITER33="outputs/_teacher_probs_4bag_iter33.parquet"

T_AB="outputs/_teacher_probs_multi_AB.parquet"   # NEW MAIN ⊕ iter33
T_AC="outputs/_teacher_probs_multi_AC.parquet"   # NEW MAIN ⊕ pure-hard
T_ABC="outputs/_teacher_probs_multi_ABC.parquet" # all 3

echo "$(date) [iter53] start multi-teacher KD" > "$RUN_LOG"

# Build merged teachers
[ ! -f "$T_AB" ] && python _phase51_merge_teachers.py --inputs "$T_NEWMAIN" "$T_ITER33" --out "$T_AB" >> "$RUN_LOG" 2>&1
[ ! -f "$T_AC" ] && python _phase51_merge_teachers.py --inputs "$T_NEWMAIN" "$T_PUREHARD" --out "$T_AC" >> "$RUN_LOG" 2>&1
[ ! -f "$T_ABC" ] && python _phase51_merge_teachers.py --inputs "$T_NEWMAIN" "$T_PUREHARD" "$T_ITER33" --out "$T_ABC" >> "$RUN_LOG" 2>&1

train_eval() {
    TAG=$1
    EXTRA="$2"
    OUT_ROOT="outputs/iter53${TAG}"
    echo "$(date) [iter53-${TAG}] $EXTRA" >> "$RUN_LOG"
    set +e
    python -m chip_multilabel._train_chip_variant \
        --variant T7 --ls 0.20 --epochs 8 --batch 2 --accum 8 --seed 1 \
        --cutmix-mode complement --cutmix-pair masked --cutmix-pair-fill corner \
        --cutmix-p 0.25 --cutmix-n-groups 3 --cutmix-complete-label-scale 0.50 \
        --kd-skip-on-cutmix --kd-temperature 4.0 \
        ${EXTRA} \
        --out-root "$OUT_ROOT" --tag "iter53${TAG}" \
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
    echo "$(date) [iter53-${TAG}] DONE" >> "$RUN_LOG"
    set -e
}

# A: multi-teacher AB (NEW MAIN + iter33), α=0.5
train_eval A_multi_AB "--kd-teacher-probs $T_AB --kd-alpha 0.5"

# B: multi-teacher AC (NEW MAIN + pure-hard), α=0.5
train_eval B_multi_AC "--kd-teacher-probs $T_AC --kd-alpha 0.5"

# C: multi-teacher ABC (all 3 4-bag teachers), α=0.5
train_eval C_multi_ABC "--kd-teacher-probs $T_ABC --kd-alpha 0.5"

# D: iter33 teacher α=0.3 (vs iter51D's α=0.5=0.9790)
train_eval D_iter33_alpha03 "--kd-teacher-probs $T_ITER33 --kd-alpha 0.3"

# E: iter33 teacher α=0.7 (over-mimic test)
train_eval E_iter33_alpha07 "--kd-teacher-probs $T_ITER33 --kd-alpha 0.7"

# F: pure-hard teacher α=0.3 (does smaller α rescue iter51C FAIL?)
train_eval F_pureHard_alpha03 "--kd-teacher-probs $T_PUREHARD --kd-alpha 0.3"

echo "$(date) [iter53] DONE 6/6" >> "$RUN_LOG"
