#!/bin/bash
# 260508 — iter 19 RESUME (B-L only): A finished, eval recovered separately
set -e
cd /d/project/known-cnn
LOG=outputs/_iter19_complement_resume.log
echo "$(date) [iter19-resume] B-L sweep — 11 trains" > "$LOG"

run_one() {
    TAG=$1
    NGRP=$2
    SCALE=$3
    PAIR=$4
    BATCH=$5
    echo "$(date) [iter19-${TAG}] complement g${NGRP} label${SCALE} pair=${PAIR} batch=${BATCH}" >> "$LOG"
    python -m chip_multilabel._train_chip_variant \
        --variant T7 --ls 0.20 \
        --epochs 8 --batch ${BATCH} --accum 4 --seed 1 \
        --cutmix-p 0.25 \
        --cutmix-mode complement --cutmix-n-groups ${NGRP} \
        --cutmix-complete-label-scale ${SCALE} \
        --cutmix-pair ${PAIR} --cutmix-pair-fill corner \
        --out-root outputs/iter19${TAG}_complement_g${NGRP}_l${SCALE}_p${PAIR} \
        --tag T7_iter19${TAG}_complement_g${NGRP}_l${SCALE}_${PAIR}_seed1 \
        >> "$LOG" 2>&1
    RUN=$(ls -d outputs/iter19${TAG}_complement_g${NGRP}_l${SCALE}_p${PAIR}/T7_T7_iter19${TAG}_*/ 2>/dev/null | head -1)
    if [ -n "${RUN}" ]; then
        echo "$(date) [iter19-${TAG}] eval (4 variants)" >> "$LOG"
        python -m chip_multilabel.run_stage1 \
            --model "${RUN}best_model.pth" \
            --eval-set D:/project/data/wm-811k/chip_multilabel_v14class \
            --out-root "${RUN}eval_seed1" \
            --variants I3,I6,I7,I10 --n-per-class 50 --strength-min 0.0 --seed 42 \
            >> "$LOG" 2>&1
    fi
}

# B-L (A skipped — already done)
run_one B 2 0.75 masked 4
run_one C 2 1.0  masked 4
run_one D 3 0.33 masked 4
run_one E 3 0.67 masked 4
run_one F 3 1.0  masked 4
run_one G 4 0.25 masked 2
run_one H 4 0.5  masked 2
run_one I 4 0.75 masked 2
run_one J 4 1.0  masked 2
run_one K 2 1.0  none   4
run_one L 4 1.0  none   4

echo "$(date) [iter19-resume] DONE" >> "$LOG"
