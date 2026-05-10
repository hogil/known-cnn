#!/bin/bash
# 260509 — iter24: 22D_LS030 의 3-seed verify (seed 7, 42 + LS=0.30)
# Chains AFTER iter23 completes. Critical — LS=0.30 가 stable winner 인지 확인.
set -e
cd /d/project/known-cnn
WAIT_LOG=outputs/_iter23_pos_weight.log
RUN_LOG=outputs/_iter24_LS030_3seed.log

echo "$(date) [iter24-after-iter23] waiting for iter23 to finish ..." > "$RUN_LOG"
until grep -q "\[iter23-after-iter22\] iter23 DONE" "$WAIT_LOG" 2>/dev/null; do
    sleep 30
done
echo "$(date) [iter24-after-iter23] iter23 done — starting LS=0.30 3-seed verify" >> "$RUN_LOG"

V14="D:/project/data/wm-811k/chip_multilabel_v14class"
V15="D:/project/data/wm-811k/chip_multilabel_v15direct"

run_one() {
    SEED=$1
    OUT_ROOT="outputs/iter24_LS030_seed${SEED}"
    echo "$(date) [iter24-LS030_s${SEED}] LS=0.30 seed=${SEED}" >> "$RUN_LOG"
    python -m chip_multilabel._train_chip_variant \
        --variant T7 --ls 0.30 --epochs 8 --batch 4 --accum 4 --seed ${SEED} \
        --cutmix-p 0.25 --cutmix-mode complement --cutmix-n-groups 2 \
        --cutmix-complete-label-scale 1.0 --cutmix-pair masked --cutmix-pair-fill corner \
        --out-root "$OUT_ROOT" \
        --tag "iter24_LS030_seed${SEED}" \
        >> "$RUN_LOG" 2>&1
    RUN=$(ls -d "$OUT_ROOT"/*iter24_LS030_seed${SEED}_*/ 2>/dev/null | head -1)
    if [ -z "$RUN" ]; then return 1; fi
    python -m chip_multilabel.run_stage1 --model "${RUN}best_model.pth" --eval-set "$V14" \
        --out-root "${RUN}eval_v14class" --variants I3,I6,I7,I10 \
        --n-per-class 50 --strength-min 0.0 --seed 42 >> "$RUN_LOG" 2>&1
    python -m chip_multilabel.run_stage1 --model "${RUN}best_model.pth" --eval-set "$V15" \
        --out-root "${RUN}eval_v15direct" --variants I3,I6,I7,I10 \
        --n-per-class 50 --strength-min 0.0 --seed 42 >> "$RUN_LOG" 2>&1
}

run_one 7
run_one 42

echo "$(date) [iter24-after-iter23] iter24 LS=0.30 3-seed DONE" >> "$RUN_LOG"
