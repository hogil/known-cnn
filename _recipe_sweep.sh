#!/usr/bin/env bash
# Recipe sweep: try different hyperparams from iter116J base
# Goal: bit_F1 >= 0.99 AND Total FAR = 0
# Train: classification_chips (clean), Eval: chip_multilabel_v15direct_n2000

# Recipe format: NAME|extra_args
RECIPES=(
  "R01_no_pair|--cutmix-pair none"
  "R02_n_groups_2|--cutmix-n-groups 2"
  "R03_no_pair_n2|--cutmix-pair none --cutmix-n-groups 2"
  "R04_ls_0.10|--ls 0.10"
  "R05_ls_0.20|--ls 0.20"
  "R06_ls_0.40|--ls 0.40"
  "R07_cutmix_p_0.10|--cutmix-p 0.10"
  "R08_cutmix_p_0.50|--cutmix-p 0.50"
  "R09_epochs_15|--epochs 15"
  "R10_epochs_20|--epochs 20"
  "R11_lr_5e-5|--lr 5e-5"
  "R12_lr_2e-4|--lr 2e-4"
  "R13_val_acc|--val-criterion acc"
  "R14_val_f1|--val-criterion f1"
  "R15_drop_path|--drop-path-rate 0.1"
  "R16_warmup|--warmup-epochs 2"
  "R17_grid_dim_4|--cutmix-grid-dim 4"
  "R18_grid_dim_16|--cutmix-grid-dim 16"
  "R19_cls_scale_0.3|--cutmix-complete-label-scale 0.3"
  "R20_cls_scale_0.7|--cutmix-complete-label-scale 0.7"
)

COUNT=0
for entry in "${RECIPES[@]}"; do
    NAME="${entry%%|*}"
    EXTRA="${entry##*|}"
    COUNT=$((COUNT + 1))
    LOG="_recipe_${NAME}.log"
    OUT="outputs/iter116J_recipe_${NAME}"
    stamp=$(date +%H:%M:%S)
    echo "=== [$stamp] recipe ${COUNT}/${#RECIPES[@]}: ${NAME} (extra: ${EXTRA}) ==="

    # base args (iter116J seed=1) + recipe extra (override)
    SEED=1 CUDA_VISIBLE_DEVICES=0 python -u -m chip_multilabel._train_chip_variant \
        --variant T7 --ls 0.30 --epochs 10 --batch 2 --accum 8 --seed 1 \
        --num-workers 0 --lr 1e-4 --no-normal --val-criterion margin_max \
        --multi-val-set E:/data/images/chip_multilabel_v15direct_n2000 --multi-val-n-per-class 50 \
        --data-root E:/data/images/classification_chips \
        --cutmix-mode complement --cutmix-pair masked --cutmix-pair-fill corner \
        --cutmix-p 0.25 --cutmix-grid-dim 8 --cutmix-n-groups 3 --cutmix-complete-label-scale 0.5 \
        --backbone-timm convnextv2_base.fcmae_ft_in22k_in1k_384 --img-size 384 \
        --backbone-timm-weights models/convnextv2_base.fcmae_ft_in22k_in1k_384.pth \
        --out-root "$OUT" --tag "$NAME" $EXTRA \
        > "$LOG" 2>&1

    if ! grep -q "DONE" "$LOG"; then
        echo "  ${NAME}: TRAIN FAIL (check $LOG)"
        continue
    fi

    # find best model + eval
    MODEL=$(find "$OUT" -name "best_model.pth" 2>/dev/null | head -1)
    if [ -z "$MODEL" ]; then
        echo "  ${NAME}: no best_model.pth"
        continue
    fi
    CUDA_VISIBLE_DEVICES=0 python -u -m chip_multilabel.run_stage1 \
        --model "$MODEL" \
        --eval-set E:/data/images/chip_multilabel_v15direct_n2000 \
        --out-root "$(dirname $MODEL)/eval" \
        --variants I10,I13 --n-per-class 2000 \
        --batch-size 32 --num-workers 0 \
        --strength-min 0.0 --strength-max 1.0 --seed 42 \
        >> "$LOG" 2>&1

    RES=$(grep "BEST cell" "$LOG" | tail -1)
    BIT_F1=$(echo "$RES" | sed -n 's/.*eval_bit_F1=\([0-9.]*\).*/\1/p')
    TOT_FAR=$(echo "$RES" | sed -n 's/.*Total=\([0-9.]*\).*/\1/p')
    NI_FAR=$(echo "$RES" | sed -n 's/.*NI=\([0-9.]*\).*/\1/p')
    OOD_FAR=$(echo "$RES" | sed -n 's/.*OOD=\([0-9.]*\).*/\1/p')
    echo "  ${NAME}: bit_F1=${BIT_F1} Tot=${TOT_FAR} NI=${NI_FAR} OOD=${OOD_FAR}"

    SUCCESS=$(python -c "
b = float('${BIT_F1:-0}'); t = float('${TOT_FAR:-100}')
print('YES' if b >= 0.99 and t == 0.0 else 'NO')
" 2>/dev/null)
    if [ "$SUCCESS" = "YES" ]; then
        echo ""
        echo "=== SUCCESS recipe=${NAME} bit_F1=${BIT_F1} Total FAR=${TOT_FAR}% ==="
        exit 0
    fi
done

echo ""
echo "=== END: tried ${COUNT} recipes, no success ==="
exit 1
