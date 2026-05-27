#!/usr/bin/env bash
# M1: maximize pos/neg prob separation THROUGH TRAINING (not post-hoc).
# Base = frozen iter116J recipe (cmp=0.25 complement masked, grid8 g3, T7, 24ep).
# Atomic change vs frozen: symmetric LS 0.30 (pos0.85/neg0.15)
#                          -> asymmetric target pos0.95 / neg0.05  (widen margin).
# Masked-pair loss gets the same asymmetric target. Selector = margin_max.
# Eval: run_stage1 I10,I13 on n2000 direct eval set -> bit_F1 + FAR.

set -u
NAME="M1_asy_pos095_neg005"
OUT="outputs/margin_asy_${NAME}"
LOG="_margin_asy_${NAME}.log"

echo "=== [$(date +%H:%M:%S)] TRAIN ${NAME} (frozen recipe + asy pos0.95/neg0.05, 24ep) ==="

SEED=1 CUDA_VISIBLE_DEVICES=0 python -u -m chip_multilabel._train_chip_variant \
    --variant T7 --ls 0.30 --epochs 24 --batch 2 --accum 8 --seed 1 \
    --num-workers 0 --lr 1e-4 --no-normal --val-criterion margin_max \
    --multi-val-set E:/data/images/chip_multilabel_v15direct_n2000 --multi-val-n-per-class 50 \
    --data-root E:/data/images/classification_chips \
    --cutmix-mode complement --cutmix-pair masked --cutmix-pair-fill corner \
    --cutmix-grid-dim 8 --cutmix-n-groups 3 --cutmix-complete-label-scale 0.5 \
    --cutmix-p 0.25 \
    --pos-targets-asy 0.95,0.95,0.95,0.95 --neg-targets-asy 0.05,0.05,0.05,0.05 \
    --cutmix-mask-pos-target 0.95 --cutmix-mask-neg-target 0.05 \
    --grad-checkpointing \
    --backbone-timm convnextv2_base.fcmae_ft_in22k_in1k_384 --img-size 384 \
    --backbone-timm-weights models/convnextv2_base.fcmae_ft_in22k_in1k_384.pth \
    --out-root "$OUT" --tag "$NAME" \
    > "$LOG" 2>&1

if ! grep -q "DONE" "$LOG"; then
    echo "  ${NAME}: TRAIN FAIL (last lines):"
    tail -5 "$LOG"
    exit 1
fi

MODEL=$(find "$OUT" -name "best_model.pth" 2>/dev/null | head -1)
[ -z "$MODEL" ] && { echo "  ${NAME}: no best_model.pth"; exit 1; }
echo "=== [$(date +%H:%M:%S)] EVAL ${NAME} I10,I13 n2000 ==="

CUDA_VISIBLE_DEVICES=0 python -u -m chip_multilabel.run_stage1 \
    --model "$MODEL" --eval-set E:/data/images/chip_multilabel_v15direct_n2000 \
    --out-root "$(dirname $MODEL)/eval" --variants I10,I13 --n-per-class 2000 \
    --batch-size 32 --num-workers 0 --strength-min 0.0 --strength-max 1.0 --seed 42 \
    >> "$LOG" 2>&1

RES=$(grep "BEST cell" "$LOG" | tail -1)
echo "  ${NAME}: $RES"
echo "=== END M1 ==="
