#!/bin/bash
# Calibration eval — run inference on chip_multilabel_v15direct_n1000 (held-out calibration set)
# for top SOTA model candidates. Use n1000 probs to determine F1-max threshold,
# then apply that threshold to v15direct (test) preds for unbiased final metric.
#
# This implements proper paper rigor: threshold determined WITHOUT looking at eval labels.

set -e
cd "$(dirname "$0")"

CALIB_SET="D:/project/data/wm-811k/chip_multilabel_v15direct_n1000"

# Top model candidates (best_model.pth glob)
MODELS=(
    "outputs/iter112_ep20/T7_*/best_model.pth"
    "outputs/iter116J_g3_ls30/T7_*/best_model.pth"
    "outputs/iter126_e_g2_n8/T7_*/best_model.pth"
    "outputs/iter125_f_g2_n6/T7_*/best_model.pth"
    "outputs/iter125_b_g4_n2/T7_*/best_model.pth"
    "outputs/iter125_d_g2_n5/T7_*/best_model.pth"
    "outputs/W2_pt100_nt30/T7_*/best_model.pth"
    "outputs/W2_pt100_nt5/T7_*/best_model.pth"
)

LOG="outputs/_calibration_eval.log"
: > "$LOG"
echo "$(date) [calib] start calibration eval on n1000 for top models" | tee -a "$LOG"

for pat in "${MODELS[@]}"; do
    ckpt=$(ls $pat 2>/dev/null | head -1)
    [ -z "$ckpt" ] && { echo "[calib] no ckpt for $pat"; continue; }
    run_dir=$(dirname "$ckpt")
    eval_out="${run_dir}/eval_v15direct_n1000_calib"
    if [ -d "$eval_out" ] && [ -n "$(ls -A "$eval_out" 2>/dev/null)" ]; then
        echo "[calib] $eval_out exists, skip" | tee -a "$LOG"
        continue
    fi
    echo "$(date) [calib] EVAL $ckpt -> $eval_out" | tee -a "$LOG"
    python -X utf8 -m chip_multilabel.run_stage1 \
        --model "$ckpt" \
        --eval-set "$CALIB_SET" \
        --out-root "$eval_out" \
        --variants I3,I7,I10,I13 \
        --n-per-class 99999 \
        --strength-min 0.0 --strength-max 1.0 \
        --seed 42 \
        >> "${LOG}" 2>&1 || true
done

echo "$(date) [calib] all calibration evals complete" | tee -a "$LOG"
echo "Next: python _calibration_threshold_apply.py"
