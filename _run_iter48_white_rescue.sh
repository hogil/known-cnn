#!/bin/bash
# 260510 — iter48: white-fill rescue test (corner-FAIL points retest with white-fill)
#  paper §6 fill-style boundary 정밀화: corner FAIL → white fill 으로 PASS 변경 가능?
#  4 cells × 2-strength eval (FULL n=200 + HARD050)
set -e
cd /d/project/known-cnn

RUN_LOG=outputs/_iter48_white_rescue.log
V15="D:/project/data/wm-811k/chip_multilabel_v15direct"
echo "$(date) [iter48] start white-fill rescue test" > "$RUN_LOG"

train_eval() {
    TAG=$1
    EXTRA="$2"
    OUT_ROOT="outputs/iter48${TAG}"
    echo "$(date) [iter48-${TAG}] $EXTRA" >> "$RUN_LOG"
    set +e
    python -m chip_multilabel._train_chip_variant \
        --variant T7 --ls 0.20 --epochs 8 --batch 2 --accum 8 --seed 1 \
        --cutmix-mode complement --cutmix-pair masked \
        --cutmix-p 0.25 \
        ${EXTRA} \
        --out-root "$OUT_ROOT" --tag "iter48${TAG}_seed1" \
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
    echo "$(date) [iter48-${TAG}] DONE" >> "$RUN_LOG"
    set -e
}

# A. g=3 LS=0.40 + white-fill (corner=FAIL iter40C, does white rescue?)
train_eval A_g3_LS040_white "--cutmix-n-groups 3 --cutmix-complete-label-scale 0.40 --cutmix-pair-fill white"

# B. g=4 LS=0.50 + white-fill (corner=FAIL iter40E, does white rescue?)
train_eval B_g4_LS050_white "--cutmix-n-groups 4 --cutmix-complete-label-scale 0.50 --cutmix-pair-fill white"

# C. g=2 LS=0.45 + white-fill (corner=FAIL iter36B, does white rescue?)
train_eval C_g2_LS045_white "--cutmix-n-groups 2 --cutmix-complete-label-scale 0.45 --cutmix-pair-fill white"

# D. g=2 LS=0.65 + white-fill (corner=FAIL iter36E, does white rescue?)
train_eval D_g2_LS065_white "--cutmix-n-groups 2 --cutmix-complete-label-scale 0.65 --cutmix-pair-fill white"

echo "$(date) [iter48] DONE 4/4" >> "$RUN_LOG"
