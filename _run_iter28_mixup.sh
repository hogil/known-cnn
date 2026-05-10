#!/bin/bash
# 260509 — iter28: Mixup ablation (paper completeness)
set -e
cd /d/project/known-cnn
LOG=outputs/_iter28_mixup.log
echo "$(date) [iter28] Mixup ablation — 1 train" > "$LOG"

V14="D:/project/data/wm-811k/chip_multilabel_v14class"
V15="D:/project/data/wm-811k/chip_multilabel_v15direct"

# Mixup with α=0.2 (Zhang 2018 default for image classification)
python -m chip_multilabel._train_chip_variant \
    --variant T7 --ls 0.20 --epochs 8 --batch 4 --accum 4 --seed 1 \
    --cutmix-p 0.25 --cutmix-mode mixup --cutmix-alpha 0.2 \
    --out-root outputs/iter28A_mixup_a02 \
    --tag iter28A_mixup_a02_seed1 \
    >> "$LOG" 2>&1
RUN=$(ls -d outputs/iter28A_mixup_a02/T*/ | head -1)
python -m chip_multilabel.run_stage1 --model "${RUN}best_model.pth" --eval-set "$V14" \
    --out-root "${RUN}eval_v14class" --variants I3,I6,I7,I10 \
    --n-per-class 50 --strength-min 0.0 --seed 42 >> "$LOG" 2>&1
python -m chip_multilabel.run_stage1 --model "${RUN}best_model.pth" --eval-set "$V15" \
    --out-root "${RUN}eval_v15direct" --variants I3,I6,I7,I10 \
    --n-per-class 50 --strength-min 0.0 --seed 42 >> "$LOG" 2>&1

# also Mixup α=1.0 (heavier mixing)
python -m chip_multilabel._train_chip_variant \
    --variant T7 --ls 0.20 --epochs 8 --batch 4 --accum 4 --seed 1 \
    --cutmix-p 0.25 --cutmix-mode mixup --cutmix-alpha 1.0 \
    --out-root outputs/iter28B_mixup_a10 \
    --tag iter28B_mixup_a10_seed1 \
    >> "$LOG" 2>&1
RUN2=$(ls -d outputs/iter28B_mixup_a10/T*/ | head -1)
python -m chip_multilabel.run_stage1 --model "${RUN2}best_model.pth" --eval-set "$V14" \
    --out-root "${RUN2}eval_v14class" --variants I3,I6,I7,I10 \
    --n-per-class 50 --strength-min 0.0 --seed 42 >> "$LOG" 2>&1
python -m chip_multilabel.run_stage1 --model "${RUN2}best_model.pth" --eval-set "$V15" \
    --out-root "${RUN2}eval_v15direct" --variants I3,I6,I7,I10 \
    --n-per-class 50 --strength-min 0.0 --seed 42 >> "$LOG" 2>&1

echo "$(date) [iter28] DONE 2/2" >> "$LOG"
