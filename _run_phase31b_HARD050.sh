#!/bin/bash
# 260510 — Phase 31b: HARD eval at strength<=0.50 (median, includes bank_boundary chips with min=0.41)
set -e
cd /d/project/known-cnn

V15=D:/project/data/wm-811k/chip_multilabel_v15direct
RUN_LOG=outputs/_phase31b_HARD050_eval.log
echo "$(date) [phase31b] start HARD eval (strength <= 0.50, includes bb)" > "$RUN_LOG"

eval_one() {
    TAG=$1; MODEL_PATTERN=$2
    MODEL=$(ls $MODEL_PATTERN 2>/dev/null | head -1)
    if [ -z "$MODEL" ]; then echo "$(date) [phase31b-${TAG}] MISSING" >> "$RUN_LOG"; return 0; fi
    OUT_DIR=$(dirname "$MODEL")/eval_v15direct_HARD050
    set +e
    python -m chip_multilabel.run_stage1 --model "$MODEL" \
        --eval-set "$V15" --out-root "$OUT_DIR" \
        --variants I3,I6,I7,I10 --n-per-class 200 \
        --strength-min 0.0 --strength-max 0.50 --seed 42 \
        >> "$RUN_LOG" 2>&1
    echo "$(date) [phase31b-${TAG}] DONE" >> "$RUN_LOG"
    set -e
}

eval_one "24_30s42" "outputs/iter24_LS030_seed42/T*/best_model.pth"
eval_one "26B" "outputs/iter26B_g3_LS050/T*/best_model.pth"
eval_one "26D" "outputs/iter26D_g4_LS040/T*/best_model.pth"
eval_one "26H" "outputs/iter26H_g3_LS067_white/T*/best_model.pth"
eval_one "33A" "outputs/iter33A_alpha03_T4/T*/best_model.pth"
eval_one "33D" "outputs/iter33D_alpha05_T8/T*/best_model.pth"
eval_one "37E" "outputs/iter37E_g3_1.0_0.5/T*/best_model.pth"
eval_one "21H" "outputs/iter21H_19I_repeat/T*/best_model.pth"
eval_one "24_30s7" "outputs/iter24_LS030_seed7/T*/best_model.pth"
echo "$(date) [phase31b] DONE" >> "$RUN_LOG"
