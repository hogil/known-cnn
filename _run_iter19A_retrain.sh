#!/bin/bash
# 260508 — iter19A retrain (epoch-1 crash 회수)
# spec: g=2 label=0.5 pair=masked batch=4 (sweep 12 cells 와 동등)
# 기존 outputs/iter19A_complement_g2_l0.5_pmasked/T7_T7_iter19A_*_140548 폴더 보존,
# 새 timestamp 폴더로 retrain.

set -e
cd /d/project/known-cnn
LOG=outputs/_iter19A_retrain.log
echo "$(date) [iter19A-retrain] g2 label0.5 masked batch=4 (8 epochs full)" > "$LOG"

python -m chip_multilabel._train_chip_variant \
    --variant T7 --ls 0.20 \
    --epochs 8 --batch 4 --accum 4 --seed 1 \
    --cutmix-p 0.25 \
    --cutmix-mode complement --cutmix-n-groups 2 \
    --cutmix-complete-label-scale 0.5 \
    --cutmix-pair masked --cutmix-pair-fill corner \
    --out-root outputs/iter19A_complement_g2_l0.5_pmasked \
    --tag T7_iter19A_complement_g2_l0.5_masked_seed1_retrain \
    >> "$LOG" 2>&1

RUN=$(ls -dt outputs/iter19A_complement_g2_l0.5_pmasked/T7_T7_iter19A_*retrain*/ 2>/dev/null | head -1)
if [ -n "${RUN}" ]; then
    echo "$(date) [iter19A-retrain] eval (4 variants)" >> "$LOG"
    python -m chip_multilabel.run_stage1 \
        --model "${RUN}best_model.pth" \
        --eval-set D:/project/data/wm-811k/chip_multilabel_v14class \
        --out-root "${RUN}eval_seed1" \
        --variants I3,I6,I7,I10 --n-per-class 50 --strength-min 0.0 --seed 42 \
        >> "$LOG" 2>&1
fi

echo "$(date) [iter19A-retrain] DONE" >> "$LOG"
