#!/bin/bash
# Wave 1 — g=2 LS×n Full Matrix (LS 1.0 → 0.0 descending order)
#
# User directive 260513: "ls 는 1부터 0.9 0.8 씩 내려가게"
# Matrix: g=2 fixed, n ∈ {1..8}, LS ∈ {1.0, 0.9, 0.8, ..., 0.0} (11 values, descending)
#
# Skip already-done:
#   LS=0.30: n=1 (124a), n=2 (124b), n=3 (124c), n=4 (124d),
#            n=5 (125d), n=6 (125f), n=8 (126e)  -- 7 cells skip
#   LS=0.00: n=1 (W1_n1_ls0), n=2 (W1_n2_ls0), n=3 (W1_n3_ls0) -- 3 cells skip
#
# Total: 88 - 10 = 78 cells, ~9.1 hr sequential
#
# Cell naming: W1_n{N}_ls{LS×10}
#   - LS=1.0 -> "ls10"
#   - LS=0.5 -> "ls5"
#   - LS=0.0 -> "ls0"

set -e
cd "$(dirname "$0")"

export WM811K_ROOT="E:/data/images"

BACKBONE="convnextv2_base.fcmae_ft_in22k_in1k_384"
WEIGHTS="models/${BACKBONE}.pth"
EVAL_SET="E:/data/images/chip_multilabel_v15direct"

OUT_BASE="outputs"
SUMMARY="${OUT_BASE}/_W1_g2_matrix_summary.log"
echo "$(date) [W1] g=2 LS×n matrix sweep RESUME (LS 1.0→0.0, 78 cells, ~9.1 hr)" >> "$SUMMARY"

train_one() {
    local N="$1"; local LS_INT="$2"   # LS_INT = LS*10
    local LS=$(python -c "print($LS_INT / 10.0)")
    local TAG="W1_n${N}_ls${LS_INT}"
    local OUT_ROOT="${OUT_BASE}/${TAG}"
    if [ -d "$OUT_ROOT" ]; then
        local DONE=$(find "$OUT_ROOT" -maxdepth 2 -name "best_model.pth" 2>/dev/null | wc -l)
        if [ "$DONE" -ge 1 ]; then
            echo "$(date) [W1] $TAG already done, skip" | tee -a "$SUMMARY"
            return 0
        else
            rm -rf "$OUT_ROOT"
        fi
    fi
    local GRID=$((2 * N))
    echo "$(date) [W1] TRAIN $TAG (g=2 n=$N GRID=$GRID LS=$LS)" | tee -a "$SUMMARY"

    python -X utf8 -m chip_multilabel._train_chip_variant \
        --variant T7 --ls $LS --batch 2 --accum 8 --seed 1 \
        --epochs 10 --lr 1e-4 \
        --backbone-timm "$BACKBONE" --img-size 384 \
        --backbone-timm-weights "$WEIGHTS" \
        --no-normal --val-criterion margin_max --save-every-epoch \
        --cutmix-mode complement --cutmix-pair masked --cutmix-pair-fill corner \
        --cutmix-p 0.25 --cutmix-grid-dim $GRID --cutmix-n-groups 2 \
        --cutmix-complete-label-scale 0.5 \
        --tag "$TAG" --out-root "$OUT_ROOT" \
        >> "${OUT_BASE}/_${TAG}_train.log" 2>&1
    # Trainer's run dir is timestamp-prefixed (e.g. 20260514_xxx_T7_W1_n1_ls10), not just T*
    local RUN=$(ls -d "$OUT_ROOT"/*/ 2>/dev/null | head -1)
    [ -n "$RUN" ] && rm -f "$RUN"epoch_*.pth "$RUN"final_epoch_model.pth || true
    echo "$(date) [W1] DONE TRAIN $TAG" | tee -a "$SUMMARY"
}

eval_one() {
    local N="$1"; local LS_INT="$2"
    local TAG="W1_n${N}_ls${LS_INT}"
    local OUT_ROOT="${OUT_BASE}/${TAG}"
    local RUN=$(ls -d "$OUT_ROOT"/T*/ 2>/dev/null | head -1)
    [ -z "$RUN" ] && return 0
    local EVAL_OUT="${RUN}eval_v15direct"
    if [ -d "$EVAL_OUT" ] && [ -n "$(ls -A "$EVAL_OUT" 2>/dev/null)" ]; then
        return 0
    fi
    echo "$(date) [W1] EVAL $TAG" | tee -a "$SUMMARY"
    python -X utf8 -m chip_multilabel.run_stage1 \
        --model "${RUN}best_model.pth" \
        --eval-set "$EVAL_SET" --out-root "$EVAL_OUT" \
        --variants I3,I7,I10,I13 --n-per-class 200 \
        --strength-min 0.0 --strength-max 1.0 --seed 42 \
        >> "${OUT_BASE}/_${TAG}_eval.log" 2>&1 || true
}

# === Phases dispatched LS=1.0 → 0.0 (descending, user requested) ===
# LS=1.0 (10): all 8 cells NEW
# LS=0.9 (9):  all 8 cells NEW
# LS=0.8 (8):  all 8 cells NEW
# LS=0.7 (7):  all 8 cells NEW
# LS=0.6 (6):  all 8 cells NEW
# LS=0.5 (5):  all 8 cells NEW
# LS=0.4 (4):  all 8 cells NEW
# LS=0.3 (3):  ONLY n=7 NEW (rest done)
# LS=0.2 (2):  all 8 cells NEW
# LS=0.1 (1):  all 8 cells NEW
# LS=0.0 (0):  n=4..8 NEW (n=1..3 done from initial run)

for LS in 10 9 8 7 6 5 4 3 2 1 0; do
    echo "$(date) [W1] === LS=0.${LS} phase ===" | tee -a "$SUMMARY"
    for N in 1 2 3 4 5 6 7 8; do
        train_one $N $LS
        eval_one $N $LS
    done
done

echo "$(date) [W1] === g=2 LS×n matrix sweep COMPLETE ===" | tee -a "$SUMMARY"
