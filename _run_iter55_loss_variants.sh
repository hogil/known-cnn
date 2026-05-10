#!/bin/bash
# 260511 — iter55: alternate loss functions + LS strength variants
#  6 cells: T3 Focal, T4 ASL, T9 sigmoid focal, T8 CE+soft+LS, T7+ls=0.05, T7+ls=0.30
set -e
cd /d/project/known-cnn

RUN_LOG=outputs/_iter55_loss_variants.log
V15="D:/project/data/wm-811k/chip_multilabel_v15direct"

echo "$(date) [iter55] start loss variants" > "$RUN_LOG"

train_eval() {
    TAG=$1
    EXTRA="$2"
    OUT_ROOT="outputs/iter55${TAG}"
    echo "$(date) [iter55-${TAG}] $EXTRA" >> "$RUN_LOG"
    set +e
    python -m chip_multilabel._train_chip_variant \
        --epochs 8 --batch 2 --accum 8 --seed 1 \
        --cutmix-mode complement --cutmix-pair masked --cutmix-pair-fill corner \
        --cutmix-p 0.25 --cutmix-n-groups 3 --cutmix-complete-label-scale 0.50 \
        ${EXTRA} \
        --out-root "$OUT_ROOT" --tag "iter55${TAG}" \
        >> "$RUN_LOG" 2>&1
    RUN=$(ls -d "$OUT_ROOT"/T*/ 2>/dev/null | head -1)
    [ -z "$RUN" ] && return 0
    for SMAX in "1.00 _n200" "0.50 _HARD050"; do
        SVAL=$(echo $SMAX | cut -d' ' -f1)
        SDIR=$(echo $SMAX | cut -d' ' -f2)
        OUT_EVAL="${RUN}eval_v15direct${SDIR}"
        python -m chip_multilabel.run_stage1 --model "${RUN}best_model.pth" \
            --eval-set "$V15" --out-root "$OUT_EVAL" \
            --variants I3,I6,I7,I10 --n-per-class 200 \
            --strength-min 0.0 --strength-max ${SVAL} --seed 42 \
            >> "$RUN_LOG" 2>&1 || true
    done
    echo "$(date) [iter55-${TAG}] DONE" >> "$RUN_LOG"
    set -e
}

# Loss function alternatives (vs paper main T7=BCE+LS)
train_eval A_T3_Focal "--variant T3 --ls 0.20"
train_eval B_T4_ASL "--variant T4 --ls 0.20"
train_eval C_T9_SigFocal "--variant T9 --ls 0.20"
train_eval D_T8_CEsoft "--variant T8 --ls 0.20"

# LS strength variants (vs paper main ls=0.20)
train_eval E_T7_ls005 "--variant T7 --ls 0.05"
train_eval F_T7_ls030 "--variant T7 --ls 0.30"

echo "$(date) [iter55] DONE 6/6" >> "$RUN_LOG"
