#!/bin/bash
# Generate fresh chips at E:\data\images using existing v19 generators.
# Step 1: training chips (classification_chips) — 5 class × 200 each
# Step 2: eval set (chip_multilabel_v15direct) — 12 class single+combo+Normal+Invalid
#
# Output mirrors D:/project/data/wm-811k/ structure under E:/data/images/

set -e
cd "$(dirname "$0")"

OUT_BASE="E:/data/images"
TRAIN_OUT="${OUT_BASE}/classification_chips"
EVAL_OUT="${OUT_BASE}/chip_multilabel_v15direct"
LOG="${OUT_BASE}/_gen.log"

mkdir -p "$OUT_BASE"
echo "$(date) [GEN] start" | tee -a "$LOG"

# --- Step 1: training chips ---
echo "$(date) [GEN] STEP 1 training chips per-class=200 (5 class, ~few min)" | tee -a "$LOG"
python -X utf8 -m dist_apply._synth_chips_only \
    --per-class 200 \
    --out "$TRAIN_OUT" \
    --seed 42 \
    >> "$LOG" 2>&1
echo "$(date) [GEN] STEP 1 DONE" | tee -a "$LOG"

# --- Step 2: eval set ---
echo "$(date) [GEN] STEP 2 eval set per-defect=200 per-normal=200 per-invalid=50" | tee -a "$LOG"
python -X utf8 -m chip_multilabel.gen_eval_set \
    --out-root "$EVAL_OUT" \
    --per-defect 200 \
    --per-normal 200 \
    --per-invalid 50 \
    --classification-chips-root "$TRAIN_OUT" \
    --seed 42 \
    >> "$LOG" 2>&1
echo "$(date) [GEN] STEP 2 DONE" | tee -a "$LOG"

echo "$(date) [GEN] ALL DONE" | tee -a "$LOG"
echo "[OUT] $TRAIN_OUT" | tee -a "$LOG"
echo "[OUT] $EVAL_OUT" | tee -a "$LOG"
