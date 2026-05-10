#!/bin/bash
# 260510 — Phase 28: re-eval NEW HEADLINE components on v15direct_n1000 at n_per_class=500
#  defect 4 single × 500 + 6 combo × 500 = 5000 defect chips (10x current)
#  Normal 200 + Invalid 50 = 250 ni chips (3x current)
#  Total ~5250 vs 800 — much tighter confidence
set -e
cd /d/project/known-cnn

V15N=D:/project/data/wm-811k/chip_multilabel_v15direct_n1000
RUN_LOG=outputs/_phase28_n500_eval.log

echo "$(date) [phase28] start n=500 eval on v15direct_n1000" > "$RUN_LOG"

eval_one() {
    TAG=$1
    MODEL_PATTERN=$2
    MODEL=$(ls $MODEL_PATTERN 2>/dev/null | head -1)
    if [ -z "$MODEL" ]; then echo "$(date) [phase28-${TAG}] MODEL NOT FOUND" >> "$RUN_LOG"; return 0; fi
    OUT_DIR=$(dirname "$MODEL")/eval_v15direct_n1000_n500
    echo "$(date) [phase28-${TAG}] eval $MODEL" >> "$RUN_LOG"
    set +e
    python -m chip_multilabel.run_stage1 \
        --model "$MODEL" \
        --eval-set "$V15N" --out-root "$OUT_DIR" \
        --variants I3,I6,I7,I10 --n-per-class 500 --strength-min 0.0 --seed 42 \
        >> "$RUN_LOG" 2>&1
    echo "$(date) [phase28-${TAG}] DONE" >> "$RUN_LOG"
    set -e
}

# NEW HEADLINE 4-bag components
eval_one "24_30s42" "outputs/iter24_LS030_seed42/T*/best_model.pth"
eval_one "26B" "outputs/iter26B_g3_LS050/T*/best_model.pth"
eval_one "26D" "outputs/iter26D_g4_LS040/T*/best_model.pth"
eval_one "26H" "outputs/iter26H_g3_LS067_white/T*/best_model.pth"
# alternates
eval_one "24_30s7" "outputs/iter24_LS030_seed7/T*/best_model.pth"
eval_one "33D" "outputs/iter33D_alpha05_T8/T*/best_model.pth"
eval_one "33A" "outputs/iter33A_alpha03_T4/T*/best_model.pth"
eval_one "37E" "outputs/iter37E_g3_1.0_0.5/T*/best_model.pth"
eval_one "21H" "outputs/iter21H_19I_repeat/T*/best_model.pth"

echo "$(date) [phase28] DONE" >> "$RUN_LOG"
