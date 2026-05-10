#!/bin/bash
# 260510 — iter40: systematic LS-axis gap-fill (paper §6 strengthening + bag mining)
#  Goal: find new diversity points to potentially supersede NEW HEADLINE 0.9992
#  6 cells: gaps in (g, LS) grid that aren't yet covered
set -e
cd /d/project/known-cnn

RUN_LOG=outputs/_iter40_gapfill_LS.log
V15="D:/project/data/wm-811k/chip_multilabel_v15direct"

echo "$(date) [iter40] start systematic LS-axis gap-fill" > "$RUN_LOG"

run_one() {
    TAG=$1
    G=$2
    LS_VAL=$3
    OUT_ROOT="outputs/iter40${TAG}"
    echo "$(date) [iter40-${TAG}] g=${G} LS=${LS_VAL}" >> "$RUN_LOG"
    set +e
    python -m chip_multilabel._train_chip_variant \
        --variant T7 --ls 0.20 --epochs 8 --batch 2 --accum 8 --seed 1 \
        --cutmix-p 0.25 --cutmix-mode complement --cutmix-n-groups ${G} \
        --cutmix-complete-label-scale ${LS_VAL} \
        --cutmix-pair masked --cutmix-pair-fill corner \
        --out-root "$OUT_ROOT" --tag "iter40${TAG}_seed1" \
        >> "$RUN_LOG" 2>&1
    RUN=$(ls -d "$OUT_ROOT"/T*/ 2>/dev/null | head -1)
    [ -z "$RUN" ] && return 0
    python -m chip_multilabel.run_stage1 --model "${RUN}best_model.pth" \
        --eval-set "$V15" --out-root "${RUN}eval_v15direct" \
        --variants I3,I6,I7,I10 --n-per-class 50 --strength-min 0.0 --seed 42 \
        >> "$RUN_LOG" 2>&1 || true
    echo "$(date) [iter40-${TAG}] DONE" >> "$RUN_LOG"
    set -e
}

# A. g=2 LS=0.20 — paper §6 LS axis fill (between iter22A/B LS=0.20 with seeds, and iter22D LS=0.30 default seed)
run_one A_g2_LS020 2 0.20

# B. g=3 LS=0.30 — gap fill (g=3 has LS=0.50, 0.67; missing 0.30/0.40)
run_one B_g3_LS030 3 0.30

# C. g=3 LS=0.40 — gap fill
run_one C_g3_LS040 3 0.40

# D. g=4 LS=0.30 — gap fill (g=4 has LS=0.40, 0.75; missing 0.30 low edge)
run_one D_g4_LS030 4 0.30

# E. g=4 LS=0.50 — gap fill (between 26D LS=0.40 and 21H LS=0.75)
run_one E_g4_LS050 4 0.50

# F. g=4 LS=0.60 — gap fill
run_one F_g4_LS060 4 0.60

echo "$(date) [iter40] DONE 6/6" >> "$RUN_LOG"
