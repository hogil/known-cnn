#!/bin/bash
# 260510 — iter49: KD + asymmetric multi-seed + new recipe variations
#  6 cells: 33A multi-seed (2) + 37E multi-seed (2) + heavy pair-loss + rect=0.7
#  Each × 2-strength eval (FULL n=200 + HARD050)
set -e
cd /d/project/known-cnn

RUN_LOG=outputs/_iter49_specialist_seeds.log
V15="D:/project/data/wm-811k/chip_multilabel_v15direct"
echo "$(date) [iter49] start specialist multi-seed + recipe variations" > "$RUN_LOG"

train_eval() {
    TAG=$1
    EXTRA="$2"
    OUT_ROOT="outputs/iter49${TAG}"
    echo "$(date) [iter49-${TAG}] $EXTRA" >> "$RUN_LOG"
    set +e
    python -m chip_multilabel._train_chip_variant \
        --variant T7 --ls 0.20 --epochs 8 --batch 2 --accum 8 \
        --cutmix-mode complement --cutmix-pair masked --cutmix-pair-fill corner \
        --cutmix-p 0.25 \
        ${EXTRA} \
        --out-root "$OUT_ROOT" --tag "iter49${TAG}" \
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
    echo "$(date) [iter49-${TAG}] DONE" >> "$RUN_LOG"
    set -e
}

# A,B: 33A KD multi-seed (33A_seed1 paper main = 0.9840)
TEACHER="outputs/_teacher_probs_14bag.parquet"
train_eval A_33A_seed7 "--seed 7 --cutmix-n-groups 3 --cutmix-complete-label-scale 0.50 --kd-teacher-probs $TEACHER --kd-alpha 0.3 --kd-temperature 4.0 --kd-skip-on-cutmix"
train_eval B_33A_seed42 "--seed 42 --cutmix-n-groups 3 --cutmix-complete-label-scale 0.50 --kd-teacher-probs $TEACHER --kd-alpha 0.3 --kd-temperature 4.0 --kd-skip-on-cutmix"

# C,D: 37E asymmetric multi-seed (37E_seed1 paper main = 0.9782)
train_eval C_37E_seed7 "--seed 7 --cutmix-n-groups 3 --cutmix-ab-labels '1.0,0.5'"
train_eval D_37E_seed42 "--seed 42 --cutmix-n-groups 3 --cutmix-ab-labels '1.0,0.5'"

# E: 26H recipe with heavier pair loss (cutmix-pair-loss-w=2.0 default 1.0)
train_eval E_26H_pair_w20 "--seed 1 --cutmix-n-groups 3 --cutmix-complete-label-scale 0.67 --cutmix-pair-fill white --cutmix-pair-loss-w 2.0"

# F: g=3 LS=0.50 + cutmix-rect=0.7 (opposite direction from iter46E rect=0.3)
train_eval F_g3LS050_rect07 "--seed 1 --cutmix-n-groups 3 --cutmix-complete-label-scale 0.50 --cutmix-rect 0.7"

echo "$(date) [iter49] DONE 6/6" >> "$RUN_LOG"
