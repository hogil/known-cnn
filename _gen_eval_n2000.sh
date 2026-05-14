#!/bin/bash
# Generate eval_n2000 set at E:/data/images/chip_multilabel_v15direct_n2000/
# 16 class × 2000 per defect/Normal + Invalid 500 + 4 OOD extracted from D: wafer canvas.

set -e
cd "$(dirname "$0")"

export WM811K_ROOT="E:/data/images"
OUT="E:/data/images/chip_multilabel_v15direct_n2000"
LOG="E:/data/images/_gen_n2000.log"

echo "$(date) [GEN n2000] STEP 1 — gen_eval_set per-defect=2000 per-normal=2000 per-invalid=500" | tee -a "$LOG"
python -X utf8 -m chip_multilabel.gen_eval_set \
    --out-root "$OUT" \
    --per-defect 2000 \
    --per-normal 2000 \
    --per-invalid 500 \
    --classification-chips-root "E:/data/images/classification_chips" \
    --seed 42 \
    >> "$LOG" 2>&1
echo "$(date) [GEN n2000] STEP 1 DONE" | tee -a "$LOG"

echo "$(date) [GEN n2000] STEP 2 — OOD chips (4 class × 2000)" | tee -a "$LOG"
python -X utf8 -c "
import sys
sys.path.insert(0, '.')
from _gen_E_ood_chips import extract_class
from pathlib import Path
import random

# override DST_ROOT to new eval dir
import _gen_E_ood_chips as g
g.DST_ROOT = Path('$OUT')
rng = random.Random(42)
total = 0
for cls in g.OOD_CLASSES:
    total += extract_class(cls, 2000, 0.03, rng)
print(f'[OOD] TOTAL {total} OOD chips')
" >> "$LOG" 2>&1
echo "$(date) [GEN n2000] STEP 2 DONE" | tee -a "$LOG"

echo "$(date) [GEN n2000] ALL DONE" | tee -a "$LOG"
echo "[OUT] $OUT"
