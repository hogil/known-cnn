#!/bin/bash
# 260509 — iter23: 19C base + fork pos_weight 0.7 (analyst recommended atop iter21E winner)
# Chains AFTER iter22 sweep completes (polls _iter22_19C_tune.log for "DONE").
set -e
cd /d/project/known-cnn
WAIT_LOG=outputs/_iter22_19C_tune.log
RUN_LOG=outputs/_iter23_pos_weight.log

echo "$(date) [iter23-after-iter22] waiting for iter22 sweep to finish ..." > "$RUN_LOG"
until grep -q "\[iter22\] DONE" "$WAIT_LOG" 2>/dev/null; do
    sleep 30
done
echo "$(date) [iter23-after-iter22] iter22 done — starting iter23 fork pos_weight 0.7" >> "$RUN_LOG"

V14="D:/project/data/wm-811k/chip_multilabel_v14class"
V15="D:/project/data/wm-811k/chip_multilabel_v15direct"

# A: fork pos_weight 0.7 (analyst recommendation)
echo "$(date) [iter23-A] 19C + fork:0.7" >> "$RUN_LOG"
python -m chip_multilabel._train_chip_variant \
    --variant T7 --ls 0.20 --epochs 8 --batch 4 --accum 4 --seed 1 \
    --cutmix-p 0.25 --cutmix-mode complement --cutmix-n-groups 2 \
    --cutmix-complete-label-scale 1.0 --cutmix-pair masked --cutmix-pair-fill corner \
    --pos-weight "fork:0.7" \
    --out-root outputs/iter23A_19C_pw_fork07 \
    --tag iter23A_19C_pw_fork07_seed1 \
    >> "$RUN_LOG" 2>&1
RUN_A=$(ls -d outputs/iter23A_19C_pw_fork07/*iter23A_*/ 2>/dev/null | head -1)

# B: fork pos_weight 0.5 (more aggressive)
echo "$(date) [iter23-B] 19C + fork:0.5" >> "$RUN_LOG"
python -m chip_multilabel._train_chip_variant \
    --variant T7 --ls 0.20 --epochs 8 --batch 4 --accum 4 --seed 1 \
    --cutmix-p 0.25 --cutmix-mode complement --cutmix-n-groups 2 \
    --cutmix-complete-label-scale 1.0 --cutmix-pair masked --cutmix-pair-fill corner \
    --pos-weight "fork:0.5" \
    --out-root outputs/iter23B_19C_pw_fork05 \
    --tag iter23B_19C_pw_fork05_seed1 \
    >> "$RUN_LOG" 2>&1
RUN_B=$(ls -d outputs/iter23B_19C_pw_fork05/*iter23B_*/ 2>/dev/null | head -1)

# eval both on v14 + v15
for RUN in "$RUN_A" "$RUN_B"; do
    if [ -z "$RUN" ]; then continue; fi
    python -m chip_multilabel.run_stage1 --model "${RUN}best_model.pth" --eval-set "$V14" \
        --out-root "${RUN}eval_v14class" --variants I3,I6,I7,I10 \
        --n-per-class 50 --strength-min 0.0 --seed 42 >> "$RUN_LOG" 2>&1
    python -m chip_multilabel.run_stage1 --model "${RUN}best_model.pth" --eval-set "$V15" \
        --out-root "${RUN}eval_v15direct" --variants I3,I6,I7,I10 \
        --n-per-class 50 --strength-min 0.0 --seed 42 >> "$RUN_LOG" 2>&1
done

echo "$(date) [iter23-after-iter22] iter23 DONE" >> "$RUN_LOG"
