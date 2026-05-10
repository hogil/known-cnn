#!/bin/bash
# 260509 — iter 21: clean baseline retrain (NO v15direct supplement, classification_chips ONLY)
# 8 models for paper-grade comparison. eval on v14class + v15direct (no leak).
set -e
cd /d/project/known-cnn
LOG=outputs/_iter21_clean_baseline.log
echo "$(date) [iter21] clean baseline (classification_chips only) — 8 trains" > "$LOG"

V14="D:/project/data/wm-811k/chip_multilabel_v14class"
V15="D:/project/data/wm-811k/chip_multilabel_v15direct"

run_one() {
    TAG=$1
    EXTRA="$2"
    BATCH=${3:-4}
    echo "$(date) [iter21-${TAG}] batch=${BATCH}  ${EXTRA}" >> "$LOG"
    OUT_ROOT="outputs/iter21${TAG}"
    python -m chip_multilabel._train_chip_variant \
        --epochs 8 --batch ${BATCH} --accum 4 --seed 1 \
        ${EXTRA} \
        --out-root "$OUT_ROOT" \
        --tag "iter21${TAG}_seed1" \
        >> "$LOG" 2>&1
    RUN=$(ls -d "$OUT_ROOT"/*iter21${TAG}_*/ 2>/dev/null | head -1)
    if [ -z "${RUN}" ]; then
        echo "$(date) [iter21-${TAG}] TRAIN FAILED" >> "$LOG"
        return 1
    fi
    # eval on v14class
    python -m chip_multilabel.run_stage1 \
        --model "${RUN}best_model.pth" \
        --eval-set "$V14" \
        --out-root "${RUN}eval_v14class" \
        --variants I3,I6,I7,I10 --n-per-class 50 --strength-min 0.0 --seed 42 \
        >> "$LOG" 2>&1
    # eval on v15direct
    python -m chip_multilabel.run_stage1 \
        --model "${RUN}best_model.pth" \
        --eval-set "$V15" \
        --out-root "${RUN}eval_v15direct" \
        --variants I3,I6,I7,I10 --n-per-class 50 --strength-min 0.0 --seed 42 \
        >> "$LOG" 2>&1
}

# A. 12-T5 reproduction: T5 BCE + no-Normal + no-CutMix (paper old baseline)
run_one A_12T5_reprod   "--variant T5 --no-normal"

# B. T7N pure: T7 + Normal + no-CutMix (Normal-only contribution)
run_one B_T7N_pure      "--variant T7 --ls 0.20"

# C. T7N + standard CutMix (vanilla CutMix paper Yun 2019)
run_one C_T7N_cutmix    "--variant T7 --ls 0.20 --cutmix-p 0.25 --cutmix-rect 0.5"

# D. 18F1 grid_complete LS=0.5
run_one D_18F1_repeat   "--variant T7 --ls 0.20 --cutmix-p 0.25 --cutmix-mode grid_complete --cutmix-complete-label-scale 0.5"

# E. 19C complement g=2 LS=1.0 masked (★ FCM-PM)
run_one E_19C_repeat    "--variant T7 --ls 0.20 --cutmix-p 0.25 --cutmix-mode complement --cutmix-n-groups 2 --cutmix-complete-label-scale 1.0 --cutmix-pair masked --cutmix-pair-fill corner"

# F. 19E complement g=3 LS=0.67 masked
run_one F_19E_repeat    "--variant T7 --ls 0.20 --cutmix-p 0.25 --cutmix-mode complement --cutmix-n-groups 3 --cutmix-complete-label-scale 0.67 --cutmix-pair masked --cutmix-pair-fill corner"

# G. 19G complement g=4 LS=0.25 masked
run_one G_19G_repeat    "--variant T7 --ls 0.20 --cutmix-p 0.25 --cutmix-mode complement --cutmix-n-groups 4 --cutmix-complete-label-scale 0.25 --cutmix-pair masked --cutmix-pair-fill corner" 2

# H. 19I complement g=4 LS=0.75 masked
run_one H_19I_repeat    "--variant T7 --ls 0.20 --cutmix-p 0.25 --cutmix-mode complement --cutmix-n-groups 4 --cutmix-complete-label-scale 0.75 --cutmix-pair masked --cutmix-pair-fill corner" 2

echo "$(date) [iter21] DONE" >> "$LOG"
