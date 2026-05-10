#!/bin/bash
# 260511 — iter56: iter50B-recipe combinations + 26B finer cutmix-p
#  6 cells: A 50B+pos_weight, B 50B+epoch12, C 50B+drop-path, D 50B+lr=5e-5, E 26B+p=0.15, F 26B+p=0.35
set -e
cd /d/project/known-cnn

RUN_LOG=outputs/_iter56_recipe_combos.log
V15="D:/project/data/wm-811k/chip_multilabel_v15direct"
T_NEWMAIN="outputs/_teacher_probs_4bag_new_main.parquet"

echo "$(date) [iter56] start recipe combinations" > "$RUN_LOG"

train_eval() {
    TAG=$1
    EXTRA="$2"
    OUT_ROOT="outputs/iter56${TAG}"
    echo "$(date) [iter56-${TAG}] $EXTRA" >> "$RUN_LOG"
    set +e
    python -m chip_multilabel._train_chip_variant \
        --variant T7 --ls 0.20 --batch 2 --accum 8 --seed 1 \
        --cutmix-mode complement --cutmix-pair masked --cutmix-pair-fill corner \
        --cutmix-n-groups 3 --cutmix-complete-label-scale 0.50 \
        ${EXTRA} \
        --out-root "$OUT_ROOT" --tag "iter56${TAG}" \
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
    echo "$(date) [iter56-${TAG}] DONE" >> "$RUN_LOG"
    set -e
}

# A. iter50B-recipe + pos-weight fork=2.0 (boost fork)
train_eval A_50B_pw_fork20 "--epochs 8 --cutmix-p 0.25 --kd-teacher-probs $T_NEWMAIN --kd-alpha 0.5 --kd-temperature 4.0 --kd-skip-on-cutmix --pos-weight fork:2.0"

# B. iter50B-recipe + epoch=12 (longer KD training)
train_eval B_50B_ep12 "--epochs 12 --cutmix-p 0.25 --kd-teacher-probs $T_NEWMAIN --kd-alpha 0.5 --kd-temperature 4.0 --kd-skip-on-cutmix"

# C. iter50B-recipe + drop-path 0.05 (light regularization)
train_eval C_50B_droppath005 "--epochs 8 --cutmix-p 0.25 --drop-path-rate 0.05 --kd-teacher-probs $T_NEWMAIN --kd-alpha 0.5 --kd-temperature 4.0 --kd-skip-on-cutmix"

# D. iter50B-recipe + lr=5e-5 (slower)
train_eval D_50B_lr5e5 "--epochs 8 --cutmix-p 0.25 --lr 5e-5 --kd-teacher-probs $T_NEWMAIN --kd-alpha 0.5 --kd-temperature 4.0 --kd-skip-on-cutmix"

# E. 26B-recipe + cutmix-p=0.15 (rare cutmix)
train_eval E_26B_p015 "--epochs 8 --cutmix-p 0.15"

# F. 26B-recipe + cutmix-p=0.35 (more frequent)
train_eval F_26B_p035 "--epochs 8 --cutmix-p 0.35"

echo "$(date) [iter56] DONE 6/6" >> "$RUN_LOG"
