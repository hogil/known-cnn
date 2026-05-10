#!/bin/bash
# 260509 — iter28 extended: Mixup α sweep + 변형. iter28 (28A/B) 끝난 후 자동 trigger.
set -e
cd /d/project/known-cnn
WAIT_LOG=outputs/_iter28_mixup.log
RUN_LOG=outputs/_iter28_mixup_extended.log

echo "$(date) [iter28-ext] waiting for iter28 to finish ..." > "$RUN_LOG"
until grep -q "\[iter28\] DONE" "$WAIT_LOG" 2>/dev/null; do
    sleep 30
done
echo "$(date) [iter28-ext] iter28 done — starting Mixup α sweep + variants" >> "$RUN_LOG"

V14="D:/project/data/wm-811k/chip_multilabel_v14class"
V15="D:/project/data/wm-811k/chip_multilabel_v15direct"

run_one() {
    TAG=$1
    EXTRA="$2"
    OUT_ROOT="outputs/iter28${TAG}"
    echo "$(date) [iter28-${TAG}] ${EXTRA}" >> "$RUN_LOG"
    python -m chip_multilabel._train_chip_variant \
        --variant T7 --ls 0.20 --epochs 8 --batch 4 --accum 4 --seed 1 \
        --cutmix-p 0.25 ${EXTRA} \
        --out-root "$OUT_ROOT" \
        --tag "iter28${TAG}_seed1" \
        >> "$RUN_LOG" 2>&1
    RUN=$(ls -d "$OUT_ROOT"/T*/ | head -1)
    if [ -z "$RUN" ]; then echo "$(date) [iter28-${TAG}] FAILED" >> "$RUN_LOG"; return 1; fi
    python -m chip_multilabel.run_stage1 --model "${RUN}best_model.pth" --eval-set "$V14" \
        --out-root "${RUN}eval_v14class" --variants I3,I6,I7,I10 \
        --n-per-class 50 --strength-min 0.0 --seed 42 >> "$RUN_LOG" 2>&1
    python -m chip_multilabel.run_stage1 --model "${RUN}best_model.pth" --eval-set "$V15" \
        --out-root "${RUN}eval_v15direct" --variants I3,I6,I7,I10 \
        --n-per-class 50 --strength-min 0.0 --seed 42 >> "$RUN_LOG" 2>&1
    echo "$(date) [iter28-${TAG}] DONE" >> "$RUN_LOG"
}

# Mixup α sweep (28A=0.2, 28B=1.0 already done; add 0.1, 0.4, 2.0)
run_one C_mixup_a01  "--cutmix-mode mixup --cutmix-alpha 0.1"
run_one D_mixup_a04  "--cutmix-mode mixup --cutmix-alpha 0.4"
run_one E_mixup_a20  "--cutmix-mode mixup --cutmix-alpha 2.0"

# Higher cutmix-p (Mixup applied more often)
run_one F_mixup_p050 "--cutmix-mode mixup --cutmix-alpha 0.4 --cutmix-p 0.50"

# Combination: Mixup + std CutMix (alternating per batch — both modes can't run same batch,
# so this is approximated by running CutMix std with --cutmix-p 0.50 + Mixup p 0.50,
# but trainer uses one mode per run. So skipping for now — keep clean Mixup sweep)

echo "$(date) [iter28-ext] DONE 4/4" >> "$RUN_LOG"
