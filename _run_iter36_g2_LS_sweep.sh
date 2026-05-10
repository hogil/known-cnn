#!/bin/bash
# 260509 — iter36: g=2 symmetric LS sweep (paper §6 missing LS values)
# Goal: complete g=2 LS axis (0.30/0.50/0.75/0.85/1.00 already; missing 0.4/0.45/0.55/0.6/0.65/0.7/0.8/0.9)
# Chains AFTER iter35 area-prop done.
set -e
cd /d/project/known-cnn

WAIT_LOG=outputs/_iter35_areaprop.log
RUN_LOG=outputs/_iter36_g2_LS_sweep.log

V15="D:/project/data/wm-811k/chip_multilabel_v15direct"

echo "$(date) [iter36] waiting for iter35 to finish ..." > "$RUN_LOG"
until grep -q "\[iter35\] DONE" "$WAIT_LOG" 2>/dev/null; do
    sleep 30
done
echo "$(date) [iter36] iter35 done — start g=2 LS sweep" >> "$RUN_LOG"

run_one() {
    TAG=$1
    LS_VAL=$2
    OUT_ROOT="outputs/iter36${TAG}"
    echo "$(date) [iter36-${TAG}] g=2 LS=${LS_VAL}" >> "$RUN_LOG"
    set +e
    python -m chip_multilabel._train_chip_variant \
        --variant T7 --ls 0.20 --epochs 8 --batch 2 --accum 8 --seed 1 \
        --cutmix-p 0.25 --cutmix-mode complement --cutmix-n-groups 2 \
        --cutmix-complete-label-scale ${LS_VAL} \
        --cutmix-pair masked --cutmix-pair-fill corner \
        --out-root "$OUT_ROOT" --tag "iter36${TAG}_seed1" \
        >> "$RUN_LOG" 2>&1
    RUN=$(ls -d "$OUT_ROOT"/T*/ 2>/dev/null | head -1)
    [ -z "$RUN" ] && return 0
    python -m chip_multilabel.run_stage1 --model "${RUN}best_model.pth" \
        --eval-set "$V15" --out-root "${RUN}eval_v15direct" \
        --variants I3,I6,I7,I10 --n-per-class 50 --strength-min 0.0 --seed 42 \
        >> "$RUN_LOG" 2>&1 || true
    echo "$(date) [iter36-${TAG}] DONE" >> "$RUN_LOG"
    set -e
}

# 8 missing LS values for g=2 symmetric (paper §6 axis completion)
run_one A_LS040 0.40
run_one B_LS045 0.45
run_one C_LS055 0.55
run_one D_LS060 0.60
run_one E_LS065 0.65
run_one F_LS070 0.70
run_one G_LS080 0.80
run_one H_LS090 0.90

echo "$(date) [iter36] DONE 8/8" >> "$RUN_LOG"
