#!/bin/bash
# 260509 — iter31: 26B (g=3 LS=0.50, best single 0.9791) regularization tuning
# 목표: single-model v15 0.97 → 0.99 (ensemble 0.9929 격차 축소)
# Chains AFTER iter30-resume done.
set -e
cd /d/project/known-cnn
WAIT_LOG=outputs/_iter30_fcmpm_variants_resume.log
RUN_LOG=outputs/_iter31_26B_tune.log

echo "$(date) [iter31] waiting for iter30-resume to finish ..." > "$RUN_LOG"
until grep -q "\[iter30-resume\] DONE" "$WAIT_LOG" 2>/dev/null; do
    sleep 30
done
echo "$(date) [iter31] iter30 done — starting 26B regularization sweep" >> "$RUN_LOG"

V14="D:/project/data/wm-811k/chip_multilabel_v14class"
V15="D:/project/data/wm-811k/chip_multilabel_v15direct"

run_one() {
    TAG=$1
    EXTRA="$2"
    EPOCHS=${3:-8}
    OUT_ROOT="outputs/iter31${TAG}"
    echo "$(date) [iter31-${TAG}] epochs=${EPOCHS}  ${EXTRA}" >> "$RUN_LOG"
    set +e
    python -m chip_multilabel._train_chip_variant \
        --variant T7 --ls 0.20 --epochs ${EPOCHS} --batch 4 --accum 4 --seed 1 \
        --cutmix-p 0.25 --cutmix-mode complement --cutmix-n-groups 3 \
        --cutmix-complete-label-scale 0.50 --cutmix-pair masked --cutmix-pair-fill corner \
        ${EXTRA} \
        --out-root "$OUT_ROOT" \
        --tag "iter31${TAG}_seed1" \
        >> "$RUN_LOG" 2>&1
    RUN=$(ls -d "$OUT_ROOT"/T*/ 2>/dev/null | head -1)
    if [ -z "$RUN" ]; then echo "$(date) [iter31-${TAG}] FAILED" >> "$RUN_LOG"; return 0; fi
    python -m chip_multilabel.run_stage1 --model "${RUN}best_model.pth" --eval-set "$V14" \
        --out-root "${RUN}eval_v14class" --variants I3,I6,I7,I10 \
        --n-per-class 50 --strength-min 0.0 --seed 42 >> "$RUN_LOG" 2>&1 || true
    python -m chip_multilabel.run_stage1 --model "${RUN}best_model.pth" --eval-set "$V15" \
        --out-root "${RUN}eval_v15direct" --variants I3,I6,I7,I10 \
        --n-per-class 50 --strength-min 0.0 --seed 42 >> "$RUN_LOG" 2>&1 || true
    echo "$(date) [iter31-${TAG}] DONE" >> "$RUN_LOG"
    set -e
}

# A. EMA=0.95
run_one A_ema095 "--ema-decay 0.95"

# B. drop_path=0.05
run_one B_droppath005 "--drop-path-rate 0.05"

# C. drop_path=0.10
run_one C_droppath010 "--drop-path-rate 0.10"

# D. warmup=2
run_one D_warmup2 "--warmup-epochs 2"

# E. EMA + warmup combo (★ 가장 강할 가능성)
run_one E_ema_warmup "--ema-decay 0.95 --warmup-epochs 2"

# F. epochs=16 (longer training)
run_one F_epochs16 "--ema-decay 0.95" 16

# G. lr-head 5e-5 (head LR ↓)
run_one G_lrhead5e5 "--lr-head 5e-5 --ema-decay 0.95"

echo "$(date) [iter31] DONE 7/7" >> "$RUN_LOG"
