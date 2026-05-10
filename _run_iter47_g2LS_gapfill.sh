#!/bin/bash
# 260510 — iter47: g=2 LS axis untested gap fill + new pair-fill combos
#  paper §6 narrow PASS basin 정밀화 + diversity 후보 추가
#  6 cells × 2-strength eval (FULL n=200 + HARD050)
set -e
cd /d/project/known-cnn

RUN_LOG=outputs/_iter47_g2LS_gapfill.log
V15="D:/project/data/wm-811k/chip_multilabel_v15direct"

echo "$(date) [iter47] start g=2 LS axis gap fill" > "$RUN_LOG"

train_eval() {
    TAG=$1
    EXTRA="$2"
    OUT_ROOT="outputs/iter47${TAG}"
    echo "$(date) [iter47-${TAG}] $EXTRA" >> "$RUN_LOG"
    set +e
    python -m chip_multilabel._train_chip_variant \
        --variant T7 --ls 0.20 --epochs 8 --batch 2 --accum 8 --seed 1 \
        --cutmix-mode complement --cutmix-pair masked --cutmix-pair-fill corner \
        ${EXTRA} \
        --out-root "$OUT_ROOT" --tag "iter47${TAG}_seed1" \
        >> "$RUN_LOG" 2>&1
    RUN=$(ls -d "$OUT_ROOT"/T*/ 2>/dev/null | head -1)
    [ -z "$RUN" ] && return 0
    # Eval at FULL n=200 + HARD050
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
    echo "$(date) [iter47-${TAG}] DONE" >> "$RUN_LOG"
    set -e
}

# A-F: g=2 LS untested points (extremes + gaps)
train_eval A_g2_LS005 "--cutmix-p 0.25 --cutmix-n-groups 2 --cutmix-complete-label-scale 0.05"
train_eval B_g2_LS010 "--cutmix-p 0.25 --cutmix-n-groups 2 --cutmix-complete-label-scale 0.10"
train_eval C_g2_LS015 "--cutmix-p 0.25 --cutmix-n-groups 2 --cutmix-complete-label-scale 0.15"
train_eval D_g2_LS025 "--cutmix-p 0.25 --cutmix-n-groups 2 --cutmix-complete-label-scale 0.25"
train_eval E_g2_LS035 "--cutmix-p 0.25 --cutmix-n-groups 2 --cutmix-complete-label-scale 0.35"
train_eval F_g2_LS050_white "--cutmix-p 0.25 --cutmix-n-groups 2 --cutmix-complete-label-scale 0.50 --cutmix-pair-fill white"

echo "$(date) [iter47] DONE 6/6" >> "$RUN_LOG"
