#!/bin/bash
# 260509 — iter30: FCM-PM variant sweep (g 확장 + 26B 3-seed mini-ensemble + LS variations)
# Chains AFTER iter29 done.
set -e
cd /d/project/known-cnn
WAIT_LOG=outputs/_iter29_label_isolation.log
RUN_LOG=outputs/_iter30_fcmpm_variants.log

echo "$(date) [iter30] waiting for iter29 to finish ..." > "$RUN_LOG"
until grep -q "\[iter29\] DONE" "$WAIT_LOG" 2>/dev/null; do
    sleep 30
done
echo "$(date) [iter30] iter29 done — starting FCM-PM variant sweep" >> "$RUN_LOG"

V14="D:/project/data/wm-811k/chip_multilabel_v14class"
V15="D:/project/data/wm-811k/chip_multilabel_v15direct"

run_one() {
    TAG=$1
    EXTRA="$2"
    BATCH=${3:-4}
    OUT_ROOT="outputs/iter30${TAG}"
    echo "$(date) [iter30-${TAG}] batch=${BATCH}  ${EXTRA}" >> "$RUN_LOG"
    python -m chip_multilabel._train_chip_variant \
        --variant T7 --ls 0.20 --epochs 8 --batch ${BATCH} --accum 4 \
        --cutmix-p 0.25 --cutmix-mode complement --cutmix-pair masked --cutmix-pair-fill corner \
        ${EXTRA} \
        --out-root "$OUT_ROOT" \
        --tag "iter30${TAG}_seed1" \
        >> "$RUN_LOG" 2>&1
    RUN=$(ls -d "$OUT_ROOT"/T*/ | head -1)
    if [ -z "$RUN" ]; then echo "$(date) [iter30-${TAG}] FAILED" >> "$RUN_LOG"; return 1; fi
    python -m chip_multilabel.run_stage1 --model "${RUN}best_model.pth" --eval-set "$V14" \
        --out-root "${RUN}eval_v14class" --variants I3,I6,I7,I10 \
        --n-per-class 50 --strength-min 0.0 --seed 42 >> "$RUN_LOG" 2>&1
    python -m chip_multilabel.run_stage1 --model "${RUN}best_model.pth" --eval-set "$V15" \
        --out-root "${RUN}eval_v15direct" --variants I3,I6,I7,I10 \
        --n-per-class 50 --strength-min 0.0 --seed 42 >> "$RUN_LOG" 2>&1
    echo "$(date) [iter30-${TAG}] DONE" >> "$RUN_LOG"
}

# A. g=5 LS=0.20 (★ NEW — extending g grid beyond 4, requires patch)
run_one A_g5_LS020 "--cutmix-n-groups 5 --cutmix-complete-label-scale 0.20 --seed 1" 2

# B. g=5 LS=0.50 (mid)
run_one B_g5_LS050 "--cutmix-n-groups 5 --cutmix-complete-label-scale 0.50 --seed 1" 2

# C. g=6 LS=0.30
run_one C_g6_LS030 "--cutmix-n-groups 6 --cutmix-complete-label-scale 0.30 --seed 1" 2

# D. g=2 LS=0.50 (직접 LS sweep within g=2)
run_one D_g2_LS050 "--cutmix-n-groups 2 --cutmix-complete-label-scale 0.50 --seed 1" 4

# E. 26B family seed=7 (26B 의 g=3 LS=0.50 의 3-seed verify)
run_one E_g3_LS050_seed7 "--cutmix-n-groups 3 --cutmix-complete-label-scale 0.50 --seed 7" 4

# F. 26B family seed=42
run_one F_g3_LS050_seed42 "--cutmix-n-groups 3 --cutmix-complete-label-scale 0.50 --seed 42" 4

echo "$(date) [iter30] DONE 6/6" >> "$RUN_LOG"
