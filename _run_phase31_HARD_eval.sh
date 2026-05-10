#!/bin/bash
# 260510 — Phase 31: HARD eval on v15direct with strength filter
#  Only chips with defect_pixel_ratio in [0.0, 0.40] = lower ~25% (HARD chips)
#  Goal: break 0.99+ saturation, reveal 4-bag composition differences
set -e
cd /d/project/known-cnn

V15=D:/project/data/wm-811k/chip_multilabel_v15direct
RUN_LOG=outputs/_phase31_HARD_eval.log

echo "$(date) [phase31] start HARD eval (strength <= 0.40)" > "$RUN_LOG"

eval_one() {
    TAG=$1
    MODEL_PATTERN=$2
    MODEL=$(ls $MODEL_PATTERN 2>/dev/null | head -1)
    if [ -z "$MODEL" ]; then echo "$(date) [phase31-${TAG}] MISSING" >> "$RUN_LOG"; return 0; fi
    OUT_DIR=$(dirname "$MODEL")/eval_v15direct_HARD040
    echo "$(date) [phase31-${TAG}] eval HARD" >> "$RUN_LOG"
    set +e
    python -m chip_multilabel.run_stage1 \
        --model "$MODEL" \
        --eval-set "$V15" --out-root "$OUT_DIR" \
        --variants I3,I6,I7,I10 --n-per-class 200 \
        --strength-min 0.0 --strength-max 0.40 \
        --seed 42 \
        >> "$RUN_LOG" 2>&1
    echo "$(date) [phase31-${TAG}] DONE" >> "$RUN_LOG"
    set -e
}

# All NEW HEADLINE component models
eval_one "24_30s42" "outputs/iter24_LS030_seed42/T*/best_model.pth"
eval_one "26B" "outputs/iter26B_g3_LS050/T*/best_model.pth"
eval_one "26D" "outputs/iter26D_g4_LS040/T*/best_model.pth"
eval_one "26H" "outputs/iter26H_g3_LS067_white/T*/best_model.pth"
eval_one "33A" "outputs/iter33A_alpha03_T4/T*/best_model.pth"
eval_one "33D" "outputs/iter33D_alpha05_T8/T*/best_model.pth"
eval_one "37E" "outputs/iter37E_g3_1.0_0.5/T*/best_model.pth"
eval_one "21H" "outputs/iter21H_19I_repeat/T*/best_model.pth"
eval_one "24_30s7" "outputs/iter24_LS030_seed7/T*/best_model.pth"

echo "$(date) [phase31] DONE" >> "$RUN_LOG"
