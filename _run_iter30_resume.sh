#!/bin/bash
# 260509 — iter30 RESUME (A eval + B-F trains)
set -e
cd /d/project/known-cnn
LOG=outputs/_iter30_fcmpm_variants_resume.log
echo "$(date) [iter30-resume] starting" > "$LOG"

V14="D:/project/data/wm-811k/chip_multilabel_v14class"
V15="D:/project/data/wm-811k/chip_multilabel_v15direct"

# A. eval only (already trained)
RUN_A="outputs/iter30A_g5_LS020/T7_iter30A_g5_LS020_seed1_260509_184223"
echo "$(date) [iter30-A] eval recovery" >> "$LOG"
python -m chip_multilabel.run_stage1 --model "${RUN_A}/best_model.pth" --eval-set "$V14" \
    --out-root "${RUN_A}/eval_v14class" --variants I3,I6,I7,I10 \
    --n-per-class 50 --strength-min 0.0 --seed 42 >> "$LOG" 2>&1
python -m chip_multilabel.run_stage1 --model "${RUN_A}/best_model.pth" --eval-set "$V15" \
    --out-root "${RUN_A}/eval_v15direct" --variants I3,I6,I7,I10 \
    --n-per-class 50 --strength-min 0.0 --seed 42 >> "$LOG" 2>&1

run_one() {
    TAG=$1
    EXTRA="$2"
    BATCH=${3:-4}
    OUT_ROOT="outputs/iter30${TAG}"
    echo "$(date) [iter30-${TAG}] batch=${BATCH}  ${EXTRA}" >> "$LOG"
    python -m chip_multilabel._train_chip_variant \
        --variant T7 --ls 0.20 --epochs 8 --batch ${BATCH} --accum 4 \
        --cutmix-p 0.25 --cutmix-mode complement --cutmix-pair masked --cutmix-pair-fill corner \
        ${EXTRA} \
        --out-root "$OUT_ROOT" \
        --tag "iter30${TAG}_seed1" \
        >> "$LOG" 2>&1
    RUN=$(ls -d "$OUT_ROOT"/T*/ 2>/dev/null | head -1)
    if [ -z "$RUN" ]; then echo "$(date) [iter30-${TAG}] FAILED" >> "$LOG"; return 0; fi
    python -m chip_multilabel.run_stage1 --model "${RUN}best_model.pth" --eval-set "$V14" \
        --out-root "${RUN}eval_v14class" --variants I3,I6,I7,I10 \
        --n-per-class 50 --strength-min 0.0 --seed 42 >> "$LOG" 2>&1 || true
    python -m chip_multilabel.run_stage1 --model "${RUN}best_model.pth" --eval-set "$V15" \
        --out-root "${RUN}eval_v15direct" --variants I3,I6,I7,I10 \
        --n-per-class 50 --strength-min 0.0 --seed 42 >> "$LOG" 2>&1 || true
    echo "$(date) [iter30-${TAG}] DONE" >> "$LOG"
}

# remove set -e for runs (continue on failure)
set +e

run_one B_g5_LS050 "--cutmix-n-groups 5 --cutmix-complete-label-scale 0.50 --seed 1" 2
run_one C_g6_LS030 "--cutmix-n-groups 6 --cutmix-complete-label-scale 0.30 --seed 1" 2
run_one D_g2_LS050 "--cutmix-n-groups 2 --cutmix-complete-label-scale 0.50 --seed 1" 4
run_one E_g3_LS050_seed7 "--cutmix-n-groups 3 --cutmix-complete-label-scale 0.50 --seed 7" 4
run_one F_g3_LS050_seed42 "--cutmix-n-groups 3 --cutmix-complete-label-scale 0.50 --seed 42" 4

echo "$(date) [iter30-resume] DONE" >> "$LOG"
