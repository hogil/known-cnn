#!/bin/bash
# 260510 — iter38: variance check on NEW HEADLINE 37E + gap-fill non-monotonic AB axis
#  - A,B: 37E (g=3 (1.0, 0.5)) seeds 7, 42 — paper claim variance estimate
#  - C-F: g=2 gap-fill between PASS/FAIL boundaries from iter37
set -e
cd /d/project/known-cnn

WAIT_LOG=outputs/_iter37_AB_labels.log
RUN_LOG=outputs/_iter38_variance_gapfill.log

V15="D:/project/data/wm-811k/chip_multilabel_v15direct"

echo "$(date) [iter38] waiting for iter37 to finish ..." > "$RUN_LOG"
until grep -q "\[iter37\] DONE" "$WAIT_LOG" 2>/dev/null; do sleep 30; done
echo "$(date) [iter38] iter37 done — start variance + gap-fill sweep" >> "$RUN_LOG"

run_one() {
    TAG=$1
    G=$2
    AB=$3
    SEED=$4
    OUT_ROOT="outputs/iter38${TAG}"
    echo "$(date) [iter38-${TAG}] g=${G} (A,B)=${AB} seed=${SEED}" >> "$RUN_LOG"
    set +e
    python -m chip_multilabel._train_chip_variant \
        --variant T7 --ls 0.20 --epochs 8 --batch 2 --accum 8 --seed ${SEED} \
        --cutmix-p 0.25 --cutmix-mode complement --cutmix-n-groups ${G} \
        --cutmix-ab-labels "${AB}" \
        --cutmix-pair masked --cutmix-pair-fill corner \
        --out-root "$OUT_ROOT" --tag "iter38${TAG}_seed${SEED}" \
        >> "$RUN_LOG" 2>&1
    RUN=$(ls -d "$OUT_ROOT"/T*/ 2>/dev/null | head -1)
    [ -z "$RUN" ] && return 0
    python -m chip_multilabel.run_stage1 --model "${RUN}best_model.pth" \
        --eval-set "$V15" --out-root "${RUN}eval_v15direct" \
        --variants I3,I6,I7,I10 --n-per-class 50 --strength-min 0.0 --seed 42 \
        >> "$RUN_LOG" 2>&1 || true
    echo "$(date) [iter38-${TAG}] DONE" >> "$RUN_LOG"
    set -e
}

# A, B: 37E seed=7, seed=42 — NEW HEADLINE 4-bag KEY component variance
run_one A_37E_seed7  3 "1.0,0.5" 7
run_one B_37E_seed42 3 "1.0,0.5" 42

# C, D, E, F: g=2 asymmetric gap-fill (paper §6 non-monotonic evidence)
run_one C_g2_1.0_0.6  2 "1.0,0.6"  1   # between 37A (1.0,0.5) PASS & 37B (1.0,0.75) FAIL
run_one D_g2_1.0_0.4  2 "1.0,0.4"  1   # softer extension
run_one E_g2_0.6_1.0  2 "0.6,1.0"  1   # between 37D (0.75,1.0) PASS & 37C (0.5,1.0) FAIL
run_one F_g2_0.4_1.0  2 "0.4,1.0"  1   # softer extension

echo "$(date) [iter38] DONE 6/6" >> "$RUN_LOG"
