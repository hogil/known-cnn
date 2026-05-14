#!/bin/bash
# Predict all candidate SOTA models on eval_n2000 (2000 per defect class).
# Skips cells whose eval_n2000_pred/ already has preds_chip.parquet.

set -e
cd "$(dirname "$0")"

EVAL_SET="E:/data/images/chip_multilabel_v15direct_n2000"
SUMMARY="outputs/_predict_n2000_summary.log"

if [ ! -d "$EVAL_SET" ]; then
    echo "[predict] ERROR eval set not found: $EVAL_SET" | tee -a "$SUMMARY"
    echo "[predict] Run _gen_eval_n2000.sh first." | tee -a "$SUMMARY"
    exit 1
fi

N_EVAL=$(find "$EVAL_SET" -name "*.png" 2>/dev/null | wc -l)
echo "$(date) [predict] start — eval_set=$EVAL_SET ($N_EVAL chips)" | tee -a "$SUMMARY"

# Find all candidate models (exclude W1/W2 sweep cells + smoke)
MODELS=$(find outputs -maxdepth 3 -name "best_model.pth" 2>/dev/null | grep -vE "W1_|W2_|smoke" | sort)
N_MODELS=$(echo "$MODELS" | wc -l)
echo "$(date) [predict] $N_MODELS candidate models" | tee -a "$SUMMARY"

I=0
for M in $MODELS; do
    I=$((I+1))
    RUN_DIR=$(dirname "$M")
    EVAL_OUT="${RUN_DIR}/eval_n2000_pred"
    CELL=$(basename "$(dirname "$RUN_DIR")")   # parent of inner T7_*
    if [ -d "$EVAL_OUT" ] && find "$EVAL_OUT" -name "preds_chip.parquet" -print -quit 2>/dev/null | grep -q .; then
        echo "$(date) [predict $I/$N_MODELS] $CELL already done, skip" | tee -a "$SUMMARY"
        continue
    fi
    echo "$(date) [predict $I/$N_MODELS] $CELL" | tee -a "$SUMMARY"
    python -X utf8 -m chip_multilabel.run_stage1 \
        --model "$M" \
        --eval-set "$EVAL_SET" \
        --out-root "$EVAL_OUT" \
        --variants I3,I7,I10,I13 \
        --n-per-class 2000 \
        --strength-min 0.0 --strength-max 1.0 \
        --seed 42 \
        >> "${RUN_DIR}/eval_n2000_pred.log" 2>&1 || echo "  FAIL $CELL" | tee -a "$SUMMARY"
done

echo "$(date) [predict] ALL DONE" | tee -a "$SUMMARY"
echo "[OUT] $SUMMARY"
