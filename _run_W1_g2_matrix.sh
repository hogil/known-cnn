#!/bin/bash
# Wave 1 — g=2 LS×n Full Matrix (41 new cells, ~4.8 hr)
#
# Matrix: g=2 fixed, n ∈ {1..8}, LS ∈ {0.0, 0.1, 0.2, 0.3, 0.4, 0.5}
# Skip already-done at LS=0.30: n=1 (124a), n=2 (124b), n=3 (124c),
#   n=4 (124d), n=5 (125d), n=6 (125f), n=8 (126e)
# Only n=7 at LS=0.30 is new.
#
# Total: 48 - 7 = 41 cells. Phase a/b/c/e/f = 8 cells each. Phase d = 1 cell.
#
# Cell naming: W1_n{N}_ls{LSx10}    e.g. W1_n3_ls20 = n=3, LS=0.20

set -e
cd "$(dirname "$0")"

BACKBONE="convnextv2_base.fcmae_ft_in22k_in1k_384"
WEIGHTS="mega_matrix/weights/${BACKBONE}.pth"
EVAL_SET="D:/project/data/wm-811k/chip_multilabel_v15direct"

OUT_BASE="outputs"
SUMMARY="${OUT_BASE}/_W1_g2_matrix_summary.log"
: > "$SUMMARY"
echo "$(date) [W1] g=2 LS×n matrix sweep start (41 cells, ~4.8 hr)" | tee -a "$SUMMARY"

train_one() {
    local N="$1"; local LS_INT="$2"   # LS_INT is LS*10, e.g. 30 for 0.30
    local LS=$(python -c "print($LS_INT / 10.0)")
    local TAG="W1_n${N}_ls${LS_INT}"
    local OUT_ROOT="${OUT_BASE}/${TAG}"
    if [ -d "$OUT_ROOT" ]; then
        echo "$(date) [W1] $TAG already exists, skip" | tee -a "$SUMMARY"
        return 0
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
    local RUN=$(ls -d "$OUT_ROOT"/T*/ 2>/dev/null | head -1)
    [ -n "$RUN" ] && rm -f "$RUN"epoch_*.pth || true
    echo "$(date) [W1] DONE TRAIN $TAG" | tee -a "$SUMMARY"
}

eval_one() {
    local N="$1"; local LS_INT="$2"
    local TAG="W1_n${N}_ls${LS_INT}"
    local OUT_ROOT="${OUT_BASE}/${TAG}"
    local RUN=$(ls -d "$OUT_ROOT"/T*/ 2>/dev/null | head -1)
    [ -z "$RUN" ] && return 0
    local EVAL_OUT="${RUN}eval_v15direct"
    [ -d "$EVAL_OUT" ] && [ -n "$(ls -A "$EVAL_OUT" 2>/dev/null)" ] && return 0
    echo "$(date) [W1] EVAL $TAG" | tee -a "$SUMMARY"
    python -X utf8 -m chip_multilabel.run_stage1 \
        --model "${RUN}best_model.pth" \
        --eval-set "$EVAL_SET" --out-root "$EVAL_OUT" \
        --variants I3,I7,I10,I13 --n-per-class 200 \
        --strength-min 0.0 --strength-max 1.0 --seed 42 \
        >> "${OUT_BASE}/_${TAG}_eval.log" 2>&1 || true
}

# === Phase a: LS=0.0, n=1..8 (8 cells) ===
echo "$(date) [W1] === Phase a: LS=0.0 ===" | tee -a "$SUMMARY"
for N in 1 2 3 4 5 6 7 8; do
    train_one $N 0
    eval_one $N 0
done

# === Phase b: LS=0.1, n=1..8 (8 cells) ===
echo "$(date) [W1] === Phase b: LS=0.1 ===" | tee -a "$SUMMARY"
for N in 1 2 3 4 5 6 7 8; do
    train_one $N 1
    eval_one $N 1
done

# === Phase c: LS=0.2, n=1..8 (8 cells) ===
echo "$(date) [W1] === Phase c: LS=0.2 ===" | tee -a "$SUMMARY"
for N in 1 2 3 4 5 6 7 8; do
    train_one $N 2
    eval_one $N 2
done

# === Phase d: LS=0.3, n=7 only (1 cell, others done in iter124/125/126) ===
echo "$(date) [W1] === Phase d: LS=0.3, n=7 (rest already done) ===" | tee -a "$SUMMARY"
train_one 7 3
eval_one 7 3

# === Phase e: LS=0.4, n=1..8 (8 cells) ===
echo "$(date) [W1] === Phase e: LS=0.4 ===" | tee -a "$SUMMARY"
for N in 1 2 3 4 5 6 7 8; do
    train_one $N 4
    eval_one $N 4
done

# === Phase f: LS=0.5, n=1..8 (8 cells) ===
echo "$(date) [W1] === Phase f: LS=0.5 ===" | tee -a "$SUMMARY"
for N in 1 2 3 4 5 6 7 8; do
    train_one $N 5
    eval_one $N 5
done

echo "$(date) [W1] === g=2 matrix sweep complete (41 cells) ===" | tee -a "$SUMMARY"
