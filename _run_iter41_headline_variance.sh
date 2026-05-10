#!/bin/bash
# 260510 — iter41: multi-seed variance for NEW HEADLINE 4-bag components
#  Goal: paper §6 robustness — does ensemble hold 0.9992 with seed substitutions?
#  6 cells: 26B/26D/26H × seeds 7,42 (already have seed=1 versions)
set -e
cd /d/project/known-cnn

RUN_LOG=outputs/_iter41_variance.log
V15="D:/project/data/wm-811k/chip_multilabel_v15direct"

echo "$(date) [iter41] start NEW HEADLINE component variance" > "$RUN_LOG"

run_one() {
    TAG=$1
    G=$2
    LS=$3
    SEED=$4
    PAIR_FILL=${5:-corner}
    OUT_ROOT="outputs/iter41${TAG}"
    echo "$(date) [iter41-${TAG}] g=${G} LS=${LS} seed=${SEED} pair_fill=${PAIR_FILL}" >> "$RUN_LOG"
    set +e
    python -m chip_multilabel._train_chip_variant \
        --variant T7 --ls 0.20 --epochs 8 --batch 2 --accum 8 --seed ${SEED} \
        --cutmix-p 0.25 --cutmix-mode complement --cutmix-n-groups ${G} \
        --cutmix-complete-label-scale ${LS} \
        --cutmix-pair masked --cutmix-pair-fill ${PAIR_FILL} \
        --out-root "$OUT_ROOT" --tag "iter41${TAG}_seed${SEED}" \
        >> "$RUN_LOG" 2>&1
    RUN=$(ls -d "$OUT_ROOT"/T*/ 2>/dev/null | head -1)
    [ -z "$RUN" ] && return 0
    python -m chip_multilabel.run_stage1 --model "${RUN}best_model.pth" \
        --eval-set "$V15" --out-root "${RUN}eval_v15direct" \
        --variants I3,I6,I7,I10 --n-per-class 50 --strength-min 0.0 --seed 42 \
        >> "$RUN_LOG" 2>&1 || true
    echo "$(date) [iter41-${TAG}] DONE" >> "$RUN_LOG"
    set -e
}

# 26B (g=3 LS=0.50) seeds 7, 42
run_one A_26B_seed7  3 0.50 7  corner
run_one B_26B_seed42 3 0.50 42 corner

# 26D (g=4 LS=0.40) seeds 7, 42
run_one C_26D_seed7  4 0.40 7  corner
run_one D_26D_seed42 4 0.40 42 corner

# 26H (g=3 LS=0.67 white-fill) seeds 7, 42
run_one E_26H_seed7  3 0.67 7  white
run_one F_26H_seed42 3 0.67 42 white

echo "$(date) [iter41] DONE 6/6" >> "$RUN_LOG"
