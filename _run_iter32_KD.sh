#!/bin/bash
# 260509 — iter32: Knowledge Distillation (KD) student training
# Teacher = 14-bag ensemble avg sigmoid probs (pre-computed parquet).
# Student = 26B base recipe (T7 LS=0.20 cutmix=0.25 complement g=3 LS=0.50 paired=masked).
# KD α=0.5 T=4.0 (Hinton 2015 default), skip-on-cutmix v1.
# Goal: distill ensemble-level performance into single model — close 0.97 → 0.99 gap.
# Chains AFTER iter31 (26B regularization sweep).
set -e
cd /d/project/known-cnn

WAIT_LOG=outputs/_iter31_26B_tune.log
RUN_LOG=outputs/_iter32_KD.log

V14="D:/project/data/wm-811k/chip_multilabel_v14class"
V15="D:/project/data/wm-811k/chip_multilabel_v15direct"
TEACHER_PARQUET="outputs/_teacher_probs_14bag.parquet"

echo "$(date) [iter32-KD] waiting for iter31 to finish ..." > "$RUN_LOG"
until grep -q "\[iter31\] DONE" "$WAIT_LOG" 2>/dev/null; do
    sleep 30
done
echo "$(date) [iter32-KD] iter31 done — start" >> "$RUN_LOG"

# Step 1: pre-compute teacher probs (skip if parquet already exists)
if [ -f "$TEACHER_PARQUET" ]; then
    echo "$(date) [iter32-KD] teacher parquet already exists at $TEACHER_PARQUET (reuse)" >> "$RUN_LOG"
else
    echo "$(date) [iter32-KD] computing teacher probs (14-bag avg) ..." >> "$RUN_LOG"
    python -m chip_multilabel._kd_make_teacher_probs \
        --out "$TEACHER_PARQUET" \
        >> "$RUN_LOG" 2>&1
fi

# Step 2: train student with KD (26B base recipe + KD α=0.5 T=4)
TAG=iter32A_KD_alpha05_T4
OUT_ROOT="outputs/${TAG}"
echo "$(date) [iter32-KD] training student ${TAG} ..." >> "$RUN_LOG"
set +e
python -m chip_multilabel._train_chip_variant \
    --variant T7 --ls 0.20 --epochs 8 --batch 4 --accum 4 --seed 1 \
    --cutmix-p 0.25 --cutmix-mode complement --cutmix-n-groups 3 \
    --cutmix-complete-label-scale 0.50 --cutmix-pair masked --cutmix-pair-fill corner \
    --kd-teacher-probs "$TEACHER_PARQUET" \
    --kd-alpha 0.5 --kd-temperature 4.0 --kd-skip-on-cutmix \
    --out-root "$OUT_ROOT" \
    --tag "${TAG}_seed1" \
    >> "$RUN_LOG" 2>&1
TRAIN_STATUS=$?
set -e

if [ $TRAIN_STATUS -ne 0 ]; then
    echo "$(date) [iter32-KD] TRAIN FAILED (exit=$TRAIN_STATUS)" >> "$RUN_LOG"
    exit 1
fi

RUN=$(ls -d "$OUT_ROOT"/T*/ 2>/dev/null | head -1)
if [ -z "$RUN" ]; then
    echo "$(date) [iter32-KD] no T* dir under $OUT_ROOT — aborting" >> "$RUN_LOG"
    exit 1
fi

# Step 3: dual eval (v14 + v15)
echo "$(date) [iter32-KD] eval v14class ..." >> "$RUN_LOG"
python -m chip_multilabel.run_stage1 --model "${RUN}best_model.pth" \
    --eval-set "$V14" \
    --out-root "${RUN}eval_v14class" --variants I3,I6,I7,I10 \
    --n-per-class 50 --strength-min 0.0 --seed 42 >> "$RUN_LOG" 2>&1 || true
echo "$(date) [iter32-KD] eval v15direct ..." >> "$RUN_LOG"
python -m chip_multilabel.run_stage1 --model "${RUN}best_model.pth" \
    --eval-set "$V15" \
    --out-root "${RUN}eval_v15direct" --variants I3,I6,I7,I10 \
    --n-per-class 50 --strength-min 0.0 --seed 42 >> "$RUN_LOG" 2>&1 || true

echo "$(date) [iter32-KD] DONE  run=${RUN}" >> "$RUN_LOG"
