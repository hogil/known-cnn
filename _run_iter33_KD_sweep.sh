#!/bin/bash
# 260509 — iter33: KD hparam sweep + creative variants. Chains after iter32.
# v15-only eval (260509 user directive). batch=2 accum=8 (effective 16, GPU friendly).
set -e
cd /d/project/known-cnn

WAIT_LOG=outputs/_iter32_KD.log
RUN_LOG=outputs/_iter33_KD_sweep.log

V15="D:/project/data/wm-811k/chip_multilabel_v15direct"
TEACHER_PARQUET="outputs/_teacher_probs_14bag.parquet"

echo "$(date) [iter33] waiting for iter32 to finish ..." > "$RUN_LOG"
until grep -q "\[iter32-KD\] DONE" "$WAIT_LOG" 2>/dev/null; do
    sleep 30
done
echo "$(date) [iter33] iter32 done — start KD sweep" >> "$RUN_LOG"

run_one() {
    TAG=$1
    EXTRA="$2"
    OUT_ROOT="outputs/iter33${TAG}"
    echo "$(date) [iter33-${TAG}] ${EXTRA}" >> "$RUN_LOG"
    set +e
    python -m chip_multilabel._train_chip_variant \
        --variant T7 --ls 0.20 --epochs 8 --batch 2 --accum 8 --seed 1 \
        --cutmix-p 0.25 --cutmix-mode complement --cutmix-n-groups 3 \
        --cutmix-complete-label-scale 0.50 --cutmix-pair masked --cutmix-pair-fill corner \
        --kd-teacher-probs "$TEACHER_PARQUET" \
        ${EXTRA} \
        --out-root "$OUT_ROOT" \
        --tag "iter33${TAG}_seed1" \
        >> "$RUN_LOG" 2>&1
    RUN=$(ls -d "$OUT_ROOT"/T*/ 2>/dev/null | head -1)
    if [ -z "$RUN" ]; then echo "$(date) [iter33-${TAG}] FAILED" >> "$RUN_LOG"; return 0; fi
    python -m chip_multilabel.run_stage1 --model "${RUN}best_model.pth" \
        --eval-set "$V15" \
        --out-root "${RUN}eval_v15direct" --variants I3,I6,I7,I10 \
        --n-per-class 50 --strength-min 0.0 --seed 42 >> "$RUN_LOG" 2>&1 || true
    echo "$(date) [iter33-${TAG}] DONE" >> "$RUN_LOG"
    set -e
}

# A. α 강조 — KD weight ↑ (more teacher influence)
run_one A_alpha03_T4 "--kd-alpha 0.3 --kd-temperature 4.0 --kd-skip-on-cutmix"

# B. α 약화 — hard label weight ↑
run_one B_alpha07_T4 "--kd-alpha 0.7 --kd-temperature 4.0 --kd-skip-on-cutmix"

# C. T sharper (T=2) — teacher prob 더 sharp
run_one C_alpha05_T2 "--kd-alpha 0.5 --kd-temperature 2.0 --kd-skip-on-cutmix"

# D. T smoother (T=8) — teacher prob 더 부드럽게
run_one D_alpha05_T8 "--kd-alpha 0.5 --kd-temperature 8.0 --kd-skip-on-cutmix"

# E. KD 적용 ON cutmix (skip 끔) — CutMix 동시에 KD 도 적용
run_one E_alpha05_T4_with_cutmix "--kd-alpha 0.5 --kd-temperature 4.0"

# F. KD + EMA combo (creative)
run_one F_alpha05_T4_ema "--kd-alpha 0.5 --kd-temperature 4.0 --kd-skip-on-cutmix --ema-decay 0.95"

# G. KD + epochs=16 (longer training)
run_one G_alpha05_T4_ep16 "--kd-alpha 0.5 --kd-temperature 4.0 --kd-skip-on-cutmix"

echo "$(date) [iter33] DONE 7/7" >> "$RUN_LOG"
