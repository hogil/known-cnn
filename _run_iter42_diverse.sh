#!/bin/bash
# 260510 — iter42: diverse experiments (paper §6 axis fill + NEW HEADLINE variance + recipe variation)
#  6 cells, ~1.5h. Chains AFTER iter41 done.
set -e
cd /d/project/known-cnn

WAIT_LOG=outputs/_iter41_variance.log
RUN_LOG=outputs/_iter42_diverse.log
V15="D:/project/data/wm-811k/chip_multilabel_v15direct"

echo "$(date) [iter42] waiting for iter41 ..." > "$RUN_LOG"
until grep -q "\[iter41\] DONE" "$WAIT_LOG" 2>/dev/null; do sleep 30; done
echo "$(date) [iter42] iter41 done — start" >> "$RUN_LOG"

run_one() {
    TAG=$1
    ARGS="$2"
    OUT_ROOT="outputs/iter42${TAG}"
    echo "$(date) [iter42-${TAG}] $ARGS" >> "$RUN_LOG"
    set +e
    python -m chip_multilabel._train_chip_variant \
        --variant T7 --ls 0.20 --epochs 8 --batch 2 --accum 8 \
        --cutmix-mode complement --cutmix-pair masked --cutmix-pair-fill corner \
        ${ARGS} \
        --out-root "$OUT_ROOT" --tag "iter42${TAG}" \
        >> "$RUN_LOG" 2>&1
    RUN=$(ls -d "$OUT_ROOT"/T*/ 2>/dev/null | head -1)
    [ -z "$RUN" ] && return 0
    python -m chip_multilabel.run_stage1 --model "${RUN}best_model.pth" \
        --eval-set "$V15" --out-root "${RUN}eval_v15direct" \
        --variants I3,I6,I7,I10 --n-per-class 200 --strength-min 0.0 --seed 42 \
        >> "$RUN_LOG" 2>&1 || true
    echo "$(date) [iter42-${TAG}] DONE" >> "$RUN_LOG"
    set -e
}

# A,B: 24_LS030 multi-seed (NEW HEADLINE component variance — does seed=100/200 also give 0.9955 in 4-bag?)
run_one A_24_LS030_seed100 "--seed 100 --cutmix-p 0.25 --cutmix-n-groups 2 --cutmix-complete-label-scale 0.30"
run_one B_24_LS030_seed200 "--seed 200 --cutmix-p 0.25 --cutmix-n-groups 2 --cutmix-complete-label-scale 0.30"

# C,D: paper §6 axis fill (g=3 LS=0.85 + g=4 LS=0.85 untested)
run_one C_g3_LS085 "--seed 1 --cutmix-p 0.25 --cutmix-n-groups 3 --cutmix-complete-label-scale 0.85"
run_one D_g4_LS085 "--seed 1 --cutmix-p 0.25 --cutmix-n-groups 4 --cutmix-complete-label-scale 0.85"

# E: g=4 LS=0.20 (extreme low at g=4)
run_one E_g4_LS020 "--seed 1 --cutmix-p 0.25 --cutmix-n-groups 4 --cutmix-complete-label-scale 0.20"

# F: 26B recipe with aggressive cutmix-p=0.50 (more frequent cutmix)
run_one F_26B_cutmix050 "--seed 1 --cutmix-p 0.50 --cutmix-n-groups 3 --cutmix-complete-label-scale 0.50"

echo "$(date) [iter42] DONE 6/6" >> "$RUN_LOG"
