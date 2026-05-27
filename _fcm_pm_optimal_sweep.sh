#!/usr/bin/env bash
# FCM-PM masked-LS focused sweep — pair=masked 고정, mask 영역 / asymmetric LS sweep
# Goal: bit_F1 >= 0.99 AND Total FAR = 0
# Base: iter116J recipe (T7, cmp=0.25, pair=masked, g=3, grid=8, cls=0.5)
# Variable: mask region pos/neg target, global LS, per-bit asymmetric

RECIPES=(
  # Mask region pos target (default = use main LS pos)
  "M01_mask_pos_0.7|--cutmix-mask-pos-target 0.7"
  "M02_mask_pos_0.85|--cutmix-mask-pos-target 0.85"
  "M03_mask_pos_0.95|--cutmix-mask-pos-target 0.95"
  # Mask region neg target
  "M04_mask_neg_0.10|--cutmix-mask-neg-target 0.10"
  "M05_mask_neg_0.20|--cutmix-mask-neg-target 0.20"
  # Combined mask pos+neg
  "M06_mask_p0.85_n0.15|--cutmix-mask-pos-target 0.85 --cutmix-mask-neg-target 0.15"
  "M07_mask_p0.7_n0.20|--cutmix-mask-pos-target 0.7 --cutmix-mask-neg-target 0.20"
  # Mask LS + global LS interaction (iter116J ls=0.30 default)
  "M08_mask_p0.85_ls0.20|--cutmix-mask-pos-target 0.85 --ls 0.20"
  "M09_mask_p0.85_ls0.35|--cutmix-mask-pos-target 0.85 --ls 0.35"
  "M10_mask_p0.7_ls0.20|--cutmix-mask-pos-target 0.7 --ls 0.20"
  # Per-bit asymmetric pos (fork+sr weak in frozen → bias higher target for them)
  "M11_asy_pos_1.0_0.85_0.95_0.85|--pos-targets-asy 1.0,0.85,0.95,0.85"
  "M12_asy_pos_1.0_0.8_1.0_0.8|--pos-targets-asy 1.0,0.8,1.0,0.8"
  # Per-bit asymmetric neg (sc fires high for NEG → stricter)
  "M13_asy_neg_0.05_0.20_0.05_0.15|--neg-targets-asy 0.05,0.20,0.05,0.15"
  # Combined asy pos + asy neg
  "M14_asy_both|--pos-targets-asy 1.0,0.85,1.0,0.85 --neg-targets-asy 0.05,0.20,0.05,0.15"
  "M15_asy_pos_mask_neg|--pos-targets-asy 1.0,0.85,1.0,0.85 --cutmix-mask-neg-target 0.15"
)

COUNT=0
for entry in "${RECIPES[@]}"; do
    NAME="${entry%%|*}"
    EXTRA="${entry##*|}"
    COUNT=$((COUNT + 1))
    LOG="_fcm_mask_${NAME}.log"
    OUT="outputs/iter116J_fcm_mask_${NAME}"
    stamp=$(date +%H:%M:%S)
    echo "=== [$stamp] FCM-MASK sweep ${COUNT}/${#RECIPES[@]}: ${NAME} (extra: ${EXTRA}) ==="

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
echo "=== END: tried ${COUNT} FCM-MASK recipes, no success ==="
exit 1
