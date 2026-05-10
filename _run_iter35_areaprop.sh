#!/bin/bash
# iter35 — area-proportional FCM-PM (paper §6.13)
# Asymmetric label A=a_frac*scale, B=b_frac*scale (vs symmetric A=B=scale).
# a_frac = 1/n_groups, b_frac = (n_groups-1)/n_groups.
# Chained: waits for iter33 KD sweep to finish first.
set -e
cd /d/project/known-cnn
WAIT_LOG=outputs/_iter33_KD_sweep.log
RUN_LOG=outputs/_iter35_areaprop.log

V15="D:/project/data/wm-811k/chip_multilabel_v15direct"

echo "$(date) [iter35] waiting for iter33 to finish ..." > "$RUN_LOG"
until grep -q "\[iter33\] DONE" "$WAIT_LOG" 2>/dev/null; do sleep 30; done
echo "$(date) [iter35] iter33 done — start area-prop sweep" >> "$RUN_LOG"

run_one() {
    TAG=$1
    EXTRA="$2"
    OUT_ROOT="outputs/iter35${TAG}"
    set +e
    python -m chip_multilabel._train_chip_variant \
        --variant T7 --ls 0.20 --epochs 8 --batch 2 --accum 8 --seed 1 \
        --cutmix-p 0.25 --cutmix-mode complement --cutmix-pair masked --cutmix-pair-fill corner \
        --cutmix-label-area-prop \
        ${EXTRA} \
        --out-root "$OUT_ROOT" --tag "iter35${TAG}_seed1" \
        >> "$RUN_LOG" 2>&1
    RUN=$(ls -d "$OUT_ROOT"/T*/ 2>/dev/null | head -1)
    [ -z "$RUN" ] && return 0
    python -m chip_multilabel.run_stage1 --model "${RUN}best_model.pth" \
        --eval-set "$V15" --out-root "${RUN}eval_v15direct" \
        --variants I3,I6,I7,I10 --n-per-class 50 --strength-min 0.0 --seed 42 \
        >> "$RUN_LOG" 2>&1 || true
    echo "$(date) [iter35-${TAG}] DONE" >> "$RUN_LOG"
    set -e
}

# A. g=3 area-prop pure (A=0.33, B=0.67) — scale=1.0
run_one A_g3_areaprop_s10 "--cutmix-n-groups 3 --cutmix-complete-label-scale 1.0"

# B. g=4 area-prop pure (A=0.25, B=0.75) — scale=1.0
run_one B_g4_areaprop_s10 "--cutmix-n-groups 4 --cutmix-complete-label-scale 1.0"

# C. g=3 area-prop with B=1 cap (A=0.50, B=1.0) — scale=1.5
run_one C_g3_areaprop_s15 "--cutmix-n-groups 3 --cutmix-complete-label-scale 1.5"

# D. g=4 area-prop with B=1 cap (A=0.33, B=1.0) — scale=1.33
run_one D_g4_areaprop_s133 "--cutmix-n-groups 4 --cutmix-complete-label-scale 1.33"

# E. g=3 area-prop conservative (A=0.17, B=0.33) — scale=0.5
run_one E_g3_areaprop_s05 "--cutmix-n-groups 3 --cutmix-complete-label-scale 0.5"

# F. g=4 area-prop conservative (A=0.125, B=0.375) — scale=0.5
run_one F_g4_areaprop_s05 "--cutmix-n-groups 4 --cutmix-complete-label-scale 0.5"

# G. g=2 area-prop sanity (= symmetric, A=B=0.5) — should match LS=0.5
run_one G_g2_areaprop_s10 "--cutmix-n-groups 2 --cutmix-complete-label-scale 1.0"

# H. g=3 with smaller scale (A=0.10, B=0.20) — even more conservative
run_one H_g3_areaprop_s03 "--cutmix-n-groups 3 --cutmix-complete-label-scale 0.3"

echo "$(date) [iter35] DONE 8/8" >> "$RUN_LOG"
