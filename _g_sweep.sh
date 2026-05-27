#!/usr/bin/env bash
# G sweep — cmp=1.0, n_groups=2, grid 2 vs 4
# User request: fcmpm 100% + group 2 + grid 2x2 / 4x4

RECIPES=(
  "G01_cmp1.0_g2_grid2|--cutmix-p 1.0 --cutmix-n-groups 2 --cutmix-grid-dim 2"
  "G02_cmp1.0_g2_grid4|--cutmix-p 1.0 --cutmix-n-groups 2 --cutmix-grid-dim 4"
)

COUNT=0
for entry in "${RECIPES[@]}"; do
    NAME="${entry%%|*}"
    EXTRA="${entry##*|}"
    COUNT=$((COUNT + 1))
    LOG="_g_sweep_${NAME}.log"
    OUT="outputs/iter116J_g_${NAME}"
    stamp=$(date +%H:%M:%S)
    echo "=== [$stamp] G sweep ${COUNT}/${#RECIPES[@]}: ${NAME} (extra: ${EXTRA}) ==="

    SEED=1 CUDA_VISIBLE_DEVICES=0 python -u -m chip_multilabel._train_chip_variant \
        --variant T7 --ls 0.30 --epochs 10 --batch 2 --accum 8 --seed 1 \
        --num-workers 0 --lr 1e-4 --no-normal --val-criterion margin_max \
        --multi-val-set E:/data/images/chip_multilabel_v15direct_n2000 --multi-val-n-per-class 50 \
        --data-root E:/data/images/classification_chips \
        --cutmix-mode complement --cutmix-pair masked --cutmix-pair-fill corner \
        --cutmix-grid-dim 8 --cutmix-n-groups 3 --cutmix-complete-label-scale 0.5 \
        --backbone-timm convnextv2_base.fcmae_ft_in22k_in1k_384 --img-size 384 \
        --backbone-timm-weights models/convnextv2_base.fcmae_ft_in22k_in1k_384.pth \
        --out-root "$OUT" --tag "$NAME" $EXTRA \
        > "$LOG" 2>&1

    if ! grep -q "DONE" "$LOG"; then
        echo "  ${NAME}: TRAIN FAIL"
        continue
    fi

    MODEL=$(find "$OUT" -name "best_model.pth" 2>/dev/null | head -1)
    [ -z "$MODEL" ] && { echo "  ${NAME}: no best_model.pth"; continue; }
    CUDA_VISIBLE_DEVICES=0 python -u -m chip_multilabel.run_stage1 \
        --model "$MODEL" --eval-set E:/data/images/chip_multilabel_v15direct_n2000 \
        --out-root "$(dirname $MODEL)/eval" --variants I10,I13 --n-per-class 2000 \
        --batch-size 32 --num-workers 0 --strength-min 0.0 --strength-max 1.0 --seed 42 \
        >> "$LOG" 2>&1

    RES=$(grep "BEST cell" "$LOG" | tail -1)
    BIT_F1=$(echo "$RES" | sed -n 's/.*eval_bit_F1=\([0-9.]*\).*/\1/p')
    TOT=$(echo "$RES" | sed -n 's/.*Total=\([0-9.]*\).*/\1/p')
    NI=$(echo "$RES" | sed -n 's/.*NI=\([0-9.]*\).*/\1/p')
    OOD=$(echo "$RES" | sed -n 's/.*OOD=\([0-9.]*\).*/\1/p')
    echo "  ${NAME}: bit_F1=${BIT_F1} Tot=${TOT} NI=${NI} OOD=${OOD}"
done

echo "=== G sweep END ==="
