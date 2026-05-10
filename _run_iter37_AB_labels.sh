#!/bin/bash
set -e
cd /d/project/known-cnn
WAIT_LOG=outputs/_iter36_g2_LS_sweep.log
RUN_LOG=outputs/_iter37_AB_labels.log

V15="D:/project/data/wm-811k/chip_multilabel_v15direct"

echo "$(date) [iter37] waiting for iter36 to finish ..." > "$RUN_LOG"
until grep -q "\[iter36\] DONE" "$WAIT_LOG" 2>/dev/null; do sleep 30; done
echo "$(date) [iter37] iter36 done — start asymmetric AB label sweep" >> "$RUN_LOG"

run_one() {
    TAG=$1
    G=$2
    AB=$3
    OUT_ROOT="outputs/iter37${TAG}"
    set +e
    python -m chip_multilabel._train_chip_variant \
        --variant T7 --ls 0.20 --epochs 8 --batch 2 --accum 8 --seed 1 \
        --cutmix-p 0.25 --cutmix-mode complement --cutmix-n-groups ${G} \
        --cutmix-ab-labels "${AB}" \
        --cutmix-pair masked --cutmix-pair-fill corner \
        --out-root "$OUT_ROOT" --tag "iter37${TAG}_seed1" \
        >> "$RUN_LOG" 2>&1
    RUN=$(ls -d "$OUT_ROOT"/T*/ 2>/dev/null | head -1)
    [ -z "$RUN" ] && return 0
    python -m chip_multilabel.run_stage1 --model "${RUN}best_model.pth" \
        --eval-set "$V15" --out-root "${RUN}eval_v15direct" \
        --variants I3,I6,I7,I10 --n-per-class 50 --strength-min 0.0 --seed 42 \
        >> "$RUN_LOG" 2>&1 || true
    echo "$(date) [iter37-${TAG}] DONE" >> "$RUN_LOG"
    set -e
}

# g=2 asymmetric (4 cells)
run_one A_g2_1.0_0.5 2 "1.0,0.5"
run_one B_g2_1.0_0.75 2 "1.0,0.75"
run_one C_g2_0.5_1.0 2 "0.5,1.0"
run_one D_g2_0.75_1.0 2 "0.75,1.0"

# g=3 asymmetric (4 cells)
run_one E_g3_1.0_0.5 3 "1.0,0.5"
run_one F_g3_1.0_0.75 3 "1.0,0.75"
run_one G_g3_0.5_1.0 3 "0.5,1.0"
run_one H_g3_0.75_1.0 3 "0.75,1.0"

# g=4 asymmetric (4 cells)
run_one I_g4_1.0_0.5 4 "1.0,0.5"
run_one J_g4_1.0_0.75 4 "1.0,0.75"
run_one K_g4_0.5_1.0 4 "0.5,1.0"
run_one L_g4_0.25_1.0 4 "0.25,1.0"  # area-prop matched

echo "$(date) [iter37] DONE 12/12" >> "$RUN_LOG"
