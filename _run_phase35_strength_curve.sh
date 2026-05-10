#!/bin/bash
# 260510 — Phase 35: strength-curve eval at strength_max in {0.45, 0.55, 0.60}
#  (already have 0.40 [HARD040] and 0.50 [HARD050] from Phase 31a/b, 1.00 = FULL n=200 from Phase 27)
#  Goal: paper §5 figure showing how 4-bag composition winner shifts with eval difficulty
set -e
cd /d/project/known-cnn

V15=D:/project/data/wm-811k/chip_multilabel_v15direct
RUN_LOG=outputs/_phase35_strength_curve.log
echo "$(date) [phase35] start strength curve sweep" > "$RUN_LOG"

eval_one() {
    TAG=$1
    MODEL_PATTERN=$2
    SMAX=$3
    MODEL=$(ls $MODEL_PATTERN 2>/dev/null | head -1)
    if [ -z "$MODEL" ]; then echo "$(date) [phase35-${TAG}-s${SMAX}] MISSING" >> "$RUN_LOG"; return 0; fi
    TAG_S=$(echo $SMAX | tr -d '.')
    OUT_DIR=$(dirname "$MODEL")/eval_v15direct_HARD${TAG_S}
    set +e
    python -m chip_multilabel.run_stage1 --model "$MODEL" \
        --eval-set "$V15" --out-root "$OUT_DIR" \
        --variants I3,I6,I7,I10 --n-per-class 200 \
        --strength-min 0.0 --strength-max ${SMAX} --seed 42 \
        >> "$RUN_LOG" 2>&1
    echo "$(date) [phase35-${TAG}-s${SMAX}] DONE" >> "$RUN_LOG"
    set -e
}

# 9 models × 3 strength points (0.45, 0.55, 0.60)
for SMAX in 0.45 0.55 0.60; do
    eval_one "24_30s42" "outputs/iter24_LS030_seed42/T*/best_model.pth" "$SMAX"
    eval_one "26B" "outputs/iter26B_g3_LS050/T*/best_model.pth" "$SMAX"
    eval_one "26D" "outputs/iter26D_g4_LS040/T*/best_model.pth" "$SMAX"
    eval_one "26H" "outputs/iter26H_g3_LS067_white/T*/best_model.pth" "$SMAX"
    eval_one "33A" "outputs/iter33A_alpha03_T4/T*/best_model.pth" "$SMAX"
    eval_one "33D" "outputs/iter33D_alpha05_T8/T*/best_model.pth" "$SMAX"
    eval_one "37E" "outputs/iter37E_g3_1.0_0.5/T*/best_model.pth" "$SMAX"
    eval_one "21H" "outputs/iter21H_19I_repeat/T*/best_model.pth" "$SMAX"
    eval_one "24_30s7" "outputs/iter24_LS030_seed7/T*/best_model.pth" "$SMAX"
done

echo "$(date) [phase35] DONE" >> "$RUN_LOG"
