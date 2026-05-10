#!/bin/bash
# 260509 — iter 20: v15direct multi-class supplement retraining (paper-grade)
# 6 models = 5 winners + 1 ablation, all retrained with --multi-combo-root v15direct
# Same single-class data (classification_chips) + 14 multi-class folders × 50 chip = 700 added training samples

set -e
cd /d/project/known-cnn
LOG=outputs/_iter20_v15supplement.log
echo "$(date) [iter20] v15direct multi-class supplement — 6 trains" > "$LOG"

V15="D:/project/data/wm-811k/chip_multilabel_v15direct"

run_one() {
    TAG=$1
    EXTRA="$2"
    echo "$(date) [iter20-${TAG}] ${EXTRA}" >> "$LOG"
    OUT_ROOT="outputs/iter20${TAG}"
    python -m chip_multilabel._train_chip_variant \
        --variant T7 --ls 0.20 \
        --epochs 8 --batch 4 --accum 4 --seed 1 \
        --multi-combo-root "$V15" --multi-combo-n-per-class 50 \
        ${EXTRA} \
        --out-root "$OUT_ROOT" \
        --tag "T7_iter20${TAG}_v15suppl_seed1" \
        >> "$LOG" 2>&1
    RUN=$(ls -d "$OUT_ROOT"/T7_T7_iter20${TAG}_*/ 2>/dev/null | head -1)
    if [ -z "${RUN}" ]; then
        echo "$(date) [iter20-${TAG}] TRAIN FAILED" >> "$LOG"
        return 1
    fi
    # eval on v14class (min-blend, paper old eval)
    echo "$(date) [iter20-${TAG}] eval v14class" >> "$LOG"
    python -m chip_multilabel.run_stage1 \
        --model "${RUN}best_model.pth" \
        --eval-set D:/project/data/wm-811k/chip_multilabel_v14class \
        --out-root "${RUN}eval_v14class" \
        --variants I3,I6,I7,I10 --n-per-class 50 --strength-min 0.0 --seed 42 \
        >> "$LOG" 2>&1
    # eval on v15direct (NEW direct synth)
    echo "$(date) [iter20-${TAG}] eval v15direct" >> "$LOG"
    python -m chip_multilabel.run_stage1 \
        --model "${RUN}best_model.pth" \
        --eval-set "$V15" \
        --out-root "${RUN}eval_v15direct" \
        --variants I3,I6,I7,I10 --n-per-class 50 --strength-min 0.0 --seed 42 \
        >> "$LOG" 2>&1
}

# 1. T7N + v15 only (no CutMix) — ablation: isolate v15direct contribution
run_one A_T7N_v15only "--cutmix-p 0.0"

# 2. 18F1 spec + v15direct
run_one B_18F1_v15 "--cutmix-p 0.25 --cutmix-mode grid_complete --cutmix-complete-label-scale 0.5 --cutmix-rect 0.5"

# 3. 19C spec + v15direct (★ FCM-PM with v15)
run_one C_19C_v15 "--cutmix-p 0.25 --cutmix-mode complement --cutmix-n-groups 2 --cutmix-complete-label-scale 1.0 --cutmix-pair masked --cutmix-pair-fill corner"

# 4. 19E spec + v15direct
run_one D_19E_v15 "--cutmix-p 0.25 --cutmix-mode complement --cutmix-n-groups 3 --cutmix-complete-label-scale 0.67 --cutmix-pair masked --cutmix-pair-fill corner"

# 5. 19G spec + v15direct (batch=2 group=4)
run_one E_19G_v15 "--cutmix-p 0.25 --cutmix-mode complement --cutmix-n-groups 4 --cutmix-complete-label-scale 0.25 --cutmix-pair masked --cutmix-pair-fill corner"

# 6. 19I spec + v15direct (batch=2 group=4)
run_one F_19I_v15 "--cutmix-p 0.25 --cutmix-mode complement --cutmix-n-groups 4 --cutmix-complete-label-scale 0.75 --cutmix-pair masked --cutmix-pair-fill corner"

echo "$(date) [iter20] DONE" >> "$LOG"
