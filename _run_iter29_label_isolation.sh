#!/bin/bash
# 260509 — iter29: label-rule × spatial-rule isolation matrix (paper §5 ablation).
# Chains AFTER iter28-ext done.
set -e
cd /d/project/known-cnn
WAIT_LOG=outputs/_iter28_mixup_extended.log
RUN_LOG=outputs/_iter29_label_isolation.log

echo "$(date) [iter29] waiting for iter28-ext to finish ..." > "$RUN_LOG"
until grep -q "\[iter28-ext\] DONE" "$WAIT_LOG" 2>/dev/null; do
    sleep 30
done
echo "$(date) [iter29] iter28-ext done — starting label×spatial ablation" >> "$RUN_LOG"

V14="D:/project/data/wm-811k/chip_multilabel_v14class"
V15="D:/project/data/wm-811k/chip_multilabel_v15direct"

run_one() {
    TAG=$1
    EXTRA="$2"
    OUT_ROOT="outputs/iter29${TAG}"
    echo "$(date) [iter29-${TAG}] ${EXTRA}" >> "$RUN_LOG"
    python -m chip_multilabel._train_chip_variant \
        --variant T7 --ls 0.20 --epochs 8 --batch 4 --accum 4 --seed 1 \
        --cutmix-p 0.25 ${EXTRA} \
        --out-root "$OUT_ROOT" \
        --tag "iter29${TAG}_seed1" \
        >> "$RUN_LOG" 2>&1
    RUN=$(ls -d "$OUT_ROOT"/T*/ | head -1)
    if [ -z "$RUN" ]; then echo "$(date) [iter29-${TAG}] FAILED" >> "$RUN_LOG"; return 1; fi
    python -m chip_multilabel.run_stage1 --model "${RUN}best_model.pth" --eval-set "$V14" \
        --out-root "${RUN}eval_v14class" --variants I3,I6,I7,I10 \
        --n-per-class 50 --strength-min 0.0 --seed 42 >> "$RUN_LOG" 2>&1
    python -m chip_multilabel.run_stage1 --model "${RUN}best_model.pth" --eval-set "$V15" \
        --out-root "${RUN}eval_v15direct" --variants I3,I6,I7,I10 \
        --n-per-class 50 --strength-min 0.0 --seed 42 >> "$RUN_LOG" 2>&1
    echo "$(date) [iter29-${TAG}] DONE" >> "$RUN_LOG"
}

# Cell A1 of ablation matrix:
#   spatial axis: {std box-cut, full-cover (grid_complete), full-cover+pair (FCM-PM)}
#   label axis: {λ-mix soft, hard both [A=1,B=1]}
#
# Already done (paper):
#   - 21C: std box-cut + λ-mix (baseline broken)
#   - 21D / 18F1: full-cover grid_complete + LS=0.5 (soft label)
#   - 21E / 19C: full-cover + pair mask + hard label (FCM-PM ★)
#
# Missing (NEW iter29):
#   A. std box-cut + hard label both: isolate "label rule" effect on broken Yun 2019
#   B. complement (FCM-PM) + soft label λ=0.5: isolate "label rule" effect on FCM-PM
#   C. grid_complete + hard label (LS=1.0): full-cover + hard isolation (vs 18F1 LS=0.5)

run_one A_stdcutmix_hardlabel "--cutmix-mode single --cutmix-rect 0.5 --variant T7 --ls 0.20"
# Note: "single" mode 의 hard label 변환 — trainer code 에서 single mode 가 multi_hot target 일 때 OR 처리.
# 이미 multi_hot 이라 추가 flag 없이 동작 (trainer 가 OR multi-hot 적용).

run_one B_complement_g2_LS050 "--cutmix-mode complement --cutmix-n-groups 2 --cutmix-complete-label-scale 0.5 --cutmix-pair masked --cutmix-pair-fill corner"

run_one C_gridcomplete_LS100 "--cutmix-mode grid_complete --cutmix-complete-label-scale 1.0"

echo "$(date) [iter29] DONE 3/3" >> "$RUN_LOG"
