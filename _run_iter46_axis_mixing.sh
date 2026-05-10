#!/bin/bash
# 260510 — iter46: diverse axis mixing — pair-mask × group-complete × pair-fill × cutmix-rect × cutmix-p
#  Each cell evaluated at 3 strength points (FULL n=200, HARD050, HARD060) for condition mixing
#  6 training cells × 3 eval strengths = 18 inferences
set -e
cd /d/project/known-cnn

RUN_LOG=outputs/_iter46_axis_mixing.log
V15="D:/project/data/wm-811k/chip_multilabel_v15direct"

echo "$(date) [iter46] start diverse axis mixing" > "$RUN_LOG"

train_eval() {
    TAG=$1
    EXTRA="$2"
    OUT_ROOT="outputs/iter46${TAG}"
    echo "$(date) [iter46-${TAG}] train: $EXTRA" >> "$RUN_LOG"
    set +e
    python -m chip_multilabel._train_chip_variant \
        --variant T7 --ls 0.20 --epochs 8 --batch 2 --accum 8 --seed 1 \
        ${EXTRA} \
        --out-root "$OUT_ROOT" --tag "iter46${TAG}_seed1" \
        >> "$RUN_LOG" 2>&1
    RUN=$(ls -d "$OUT_ROOT"/T*/ 2>/dev/null | head -1)
    [ -z "$RUN" ] && return 0
    # Eval at 3 strength points
    for SMAX in "1.00 _n200" "0.50 _HARD050" "0.60 _HARD060"; do
        SVAL=$(echo $SMAX | cut -d' ' -f1)
        SDIR=$(echo $SMAX | cut -d' ' -f2)
        OUT_EVAL="${RUN}eval_v15direct${SDIR}"
        echo "$(date) [iter46-${TAG}-s${SVAL}] eval" >> "$RUN_LOG"
        python -m chip_multilabel.run_stage1 --model "${RUN}best_model.pth" \
            --eval-set "$V15" --out-root "$OUT_EVAL" \
            --variants I3,I6,I7,I10 --n-per-class 200 \
            --strength-min 0.0 --strength-max ${SVAL} --seed 42 \
            >> "$RUN_LOG" 2>&1 || true
    done
    echo "$(date) [iter46-${TAG}] DONE" >> "$RUN_LOG"
    set -e
}

# A. baseline 26B-recipe with pair=none (ablation: pair-mask 효과 측정)
train_eval A_g3LS050_pair_none "--cutmix-p 0.25 --cutmix-mode complement --cutmix-n-groups 3 --cutmix-complete-label-scale 0.50 --cutmix-pair none"

# B. baseline 26B-recipe with cutmix-mode "single" (no complement, ablation: complement 효과 측정)
train_eval B_g3LS050_mode_single "--cutmix-p 0.25 --cutmix-mode single --cutmix-n-groups 3 --cutmix-complete-label-scale 0.50 --cutmix-pair masked --cutmix-pair-fill corner"

# C. g=3 LS=0.30 with pair-fill=noise (low-LS + noise-fill: new combo)
train_eval C_g3LS030_pair_noise "--cutmix-p 0.25 --cutmix-mode complement --cutmix-n-groups 3 --cutmix-complete-label-scale 0.30 --cutmix-pair masked --cutmix-pair-fill noise"

# D. g=4 LS=0.40 with cutmix-p=0.40 (more frequent cutmix)
train_eval D_g4LS040_p040 "--cutmix-p 0.40 --cutmix-mode complement --cutmix-n-groups 4 --cutmix-complete-label-scale 0.40 --cutmix-pair masked --cutmix-pair-fill corner"

# E. g=3 LS=0.50 with cutmix-rect 0.3 (rectangular bias change)
train_eval E_g3LS050_rect03 "--cutmix-p 0.25 --cutmix-mode complement --cutmix-n-groups 3 --cutmix-complete-label-scale 0.50 --cutmix-rect 0.3 --cutmix-pair masked --cutmix-pair-fill corner"

# F. g=2 LS=0.30 with pair=none + p=0.40 (multi-axis mix)
train_eval F_g2LS030_pair_none_p040 "--cutmix-p 0.40 --cutmix-mode complement --cutmix-n-groups 2 --cutmix-complete-label-scale 0.30 --cutmix-pair none"

echo "$(date) [iter46] DONE 6/6" >> "$RUN_LOG"
