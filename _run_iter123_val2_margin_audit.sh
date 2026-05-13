#!/bin/bash
# iter123 — val 2-combo bit-margin audit (no training)
# Source rule: audited checkpoints must be trained from 4 single defect classes only.
set -e
cd /d/project/known-cnn

if [ "${1:-}" != "--run-full-audit" ]; then
    echo "Refusing to run full audit by default. Use PowerShell smoke:"
    echo "  ./_run_iter123_val2_margin_audit.ps1 -Smoke"
    echo "Full audit requires explicit approval and:"
    echo "  bash _run_iter123_val2_margin_audit.sh --run-full-audit"
    exit 2
fi
shift

RUN_LOG=outputs/_iter123_val2_margin_audit.log
MODELS=outputs/_iter123_val2_margin_models.txt
V15="D:/project/data/wm-811k/chip_multilabel_v15direct"

echo "$(date) [iter123] start val2 margin audit" > "$RUN_LOG"
mkdir -p outputs
: > "$MODELS"

add_model() {
    TAG=$1
    CKPT=$2
    if [ -f "$CKPT" ]; then
        echo "${TAG}=${CKPT}" >> "$MODELS"
        echo "$(date) [iter123] add ${TAG}" >> "$RUN_LOG"
    else
        echo "$(date) [iter123] skip missing ${TAG}: ${CKPT}" >> "$RUN_LOG"
    fi
}

# Known reference points. Extend this list as new cells finish.
add_model iter112_best_ep06 "outputs/iter112_ep20/T7_iter112_ep20_260512_214618/best_model.pth"
add_model iter112_final_ep20 "outputs/iter112_ep20/T7_iter112_ep20_260512_214618/final_epoch_model.pth"
add_model iter120A_baseline "outputs/iter120A_baseline/T7_iter120A_baseline_260513_032245/best_model.pth"
add_model iter120B_dp005 "outputs/iter120B_dp005/T7_iter120B_dp005_260513_033255/best_model.pth"
add_model iter120C_dp010 "outputs/iter120C_dp010/T7_iter120C_dp010_260513_034134/best_model.pth"
add_model iter120D_p015 "outputs/iter120D_p015/T7_iter120D_p015_260513_034929/best_model.pth"
add_model iter121A_p040 "outputs/iter121A_p040/T7_iter121A_p040_260513_055923/best_model.pth"
add_model iter121B_p060 "outputs/iter121B_p060/T7_iter121B_p060_260513_061051/best_model.pth"
add_model iter121C_cls10 "outputs/iter121C_cls10/T7_iter121C_cls10_260513_062131/best_model.pth"
add_model iter121D_ab1008 "outputs/iter121D_ab1008/T7_iter121D_ab1008_260513_063315/best_model.pth"
add_model iter121E_ab1010 "outputs/iter121E_ab1010/T7_iter121E_ab1010_260513_064451/best_model.pth"
add_model iter121F_ep15 "outputs/iter121F_ep15/T7_iter121F_ep15_260513_075003/best_model.pth"
add_model iter121G_ep20 "outputs/iter121G_ep20/T7_iter121G_ep20_260513_080328/best_model.pth"

if [ ! -s "$MODELS" ]; then
    echo "$(date) [iter123] FAIL no checkpoints found" >> "$RUN_LOG"
    exit 1
fi

python -X utf8 _run_iter123_val2_margin_audit.py \
    --models-file "$MODELS" \
    --eval-set "$V15" \
    --n-per-class 200 \
    --strength-min 0.0 --strength-max 1.0 \
    --seed 42 --val-ratio 0.2 \
    --q-low 0.05 --q-high 0.95 \
    --alphas 0.50,0.65,0.80 \
    --out-prefix outputs/_iter123_val2_margin \
    >> "$RUN_LOG" 2>&1

echo "$(date) [iter123] DONE" >> "$RUN_LOG"
