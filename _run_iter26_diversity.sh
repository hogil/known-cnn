#!/bin/bash
# 260509 — iter26: FCM-PM diversity sweep (gap fill + pair_fill variants)
# 9 trains + dual-eval. 모든 학습 = classification_chips/ ONLY (NO multi-combo-root).
set -e
cd /d/project/known-cnn
LOG=outputs/_iter26_diversity.log
echo "$(date) [iter26] FCM-PM diversity sweep — 9 trains" > "$LOG"

V14="D:/project/data/wm-811k/chip_multilabel_v14class"
V15="D:/project/data/wm-811k/chip_multilabel_v15direct"

run_one() {
    TAG=$1
    EXTRA="$2"
    BATCH=${3:-4}
    OUT_ROOT="outputs/iter26${TAG}"
    echo "$(date) [iter26-${TAG}] batch=${BATCH}  ${EXTRA}" >> "$LOG"
    python -m chip_multilabel._train_chip_variant \
        --variant T7 --ls 0.20 \
        --epochs 8 --batch ${BATCH} --accum 4 --seed 1 \
        --cutmix-p 0.25 --cutmix-mode complement \
        ${EXTRA} \
        --out-root "$OUT_ROOT" \
        --tag "iter26${TAG}_seed1" \
        >> "$LOG" 2>&1
    RUN=$(ls -d "$OUT_ROOT"/*iter26${TAG}_*/ 2>/dev/null | head -1)
    if [ -z "$RUN" ]; then echo "$(date) [iter26-${TAG}] FAILED" >> "$LOG"; return 1; fi
    python -m chip_multilabel.run_stage1 --model "${RUN}best_model.pth" --eval-set "$V14" \
        --out-root "${RUN}eval_v14class" --variants I3,I6,I7,I10 \
        --n-per-class 50 --strength-min 0.0 --seed 42 >> "$LOG" 2>&1
    python -m chip_multilabel.run_stage1 --model "${RUN}best_model.pth" --eval-set "$V15" \
        --out-root "${RUN}eval_v15direct" --variants I3,I6,I7,I10 \
        --n-per-class 50 --strength-min 0.0 --seed 42 >> "$LOG" 2>&1
    echo "$(date) [iter26-${TAG}] DONE" >> "$LOG"
}

# (g, LS) gap fill 5
run_one A_g2_LS085 "--cutmix-n-groups 2 --cutmix-complete-label-scale 0.85 --cutmix-pair masked --cutmix-pair-fill corner"
run_one B_g3_LS050 "--cutmix-n-groups 3 --cutmix-complete-label-scale 0.50 --cutmix-pair masked --cutmix-pair-fill corner"
run_one C_g3_LS083 "--cutmix-n-groups 3 --cutmix-complete-label-scale 0.83 --cutmix-pair masked --cutmix-pair-fill corner"
run_one D_g4_LS040 "--cutmix-n-groups 4 --cutmix-complete-label-scale 0.40 --cutmix-pair masked --cutmix-pair-fill corner" 2
run_one E_g4_LS060 "--cutmix-n-groups 4 --cutmix-complete-label-scale 0.60 --cutmix-pair masked --cutmix-pair-fill corner" 2

# pair_fill variants (white / noise) of top dual-pass configs
run_one F_g2_LS100_white "--cutmix-n-groups 2 --cutmix-complete-label-scale 1.0  --cutmix-pair masked --cutmix-pair-fill white"
run_one G_g2_LS100_noise "--cutmix-n-groups 2 --cutmix-complete-label-scale 1.0  --cutmix-pair masked --cutmix-pair-fill noise"
run_one H_g3_LS067_white "--cutmix-n-groups 3 --cutmix-complete-label-scale 0.67 --cutmix-pair masked --cutmix-pair-fill white"
run_one I_g4_LS075_white "--cutmix-n-groups 4 --cutmix-complete-label-scale 0.75 --cutmix-pair masked --cutmix-pair-fill white" 2

echo "$(date) [iter26] DONE 9/9" >> "$LOG"
