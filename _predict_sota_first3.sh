#!/bin/bash
# Phase A: predict 3 SOTA single models on eval_n2000.
# After this, user reviews table → decide if "rest of models" worth running.

set -e
cd "$(dirname "$0")"

EVAL_SET="E:/data/images/chip_multilabel_v15direct_n2000"
SUMMARY="outputs/_predict_sota_first3.log"

if [ ! -d "$EVAL_SET" ]; then
    echo "[predict] ERROR eval set not found: $EVAL_SET — run _gen_eval_n2000.sh first." | tee -a "$SUMMARY"
    exit 1
fi

# 3 SOTA single-model candidates
MODELS=(
    "outputs/iter116J_g3_ls30/T7_iter116J_g3_ls30_260513_010015/best_model.pth"
    "outputs/iter116F_g4_ls30/T7_iter116F_g4_ls30_260513_002653/best_model.pth"
    "outputs/iter112_ep20/T7_iter112_ep20_260512_214618/best_model.pth"
)

I=0
for M in "${MODELS[@]}"; do
    I=$((I+1))
    RUN_DIR=$(dirname "$M")
    CELL=$(basename "$(dirname "$RUN_DIR")")
    EVAL_OUT="${RUN_DIR}/eval_n2000_pred"
    if find "$EVAL_OUT" -name "preds_chip.parquet" 2>/dev/null | grep -q .; then
        echo "$(date) [predict $I/3] $CELL already done, skip" | tee -a "$SUMMARY"
        continue
    fi
    echo "$(date) [predict $I/3] $CELL" | tee -a "$SUMMARY"
    python -X utf8 -m chip_multilabel.run_stage1 \
        --model "$M" --eval-set "$EVAL_SET" --out-root "$EVAL_OUT" \
        --variants I3,I7,I10,I13 \
        --n-per-class 2000 --strength-min 0.0 --strength-max 1.0 --seed 42 \
        >> "${RUN_DIR}/eval_n2000_pred.log" 2>&1
    echo "$(date) [predict $I/3] DONE $CELL" | tee -a "$SUMMARY"
done

echo "$(date) [predict] ALL 3 DONE" | tee -a "$SUMMARY"
