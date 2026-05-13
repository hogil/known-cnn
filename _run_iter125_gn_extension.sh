#!/bin/bash
# iter125 — FCM-PM (g, n) parameterization extension (6 cells, ~50 min)
#
# Extends iter124 (g={2,3} × n={1,2,3,4 / 1,2,3}) with:
#   Phase A: g=4 axis NEW   - 125a (g=4 n=1), 125b (g=4 n=2), 125c (g=4 n=3)
#   Phase B: high-n         - 125d (g=2 n=5), 125e (g=3 n=4)
#   Phase C: triple-matched - 125f (g=2 n=6) [pairs with 125c, 125e at GRID=12]
#
# Same frozen baseline as iter124:
#   T7 BCE+LS=0.30, lr 1e-4, ep 10, seed=1, val_margin selection,
#   cutmix-mode=complement, pair=masked, fill=corner, p=0.25, label-scale=0.5.

set -e
cd "$(dirname "$0")"

BACKBONE="convnextv2_base.fcmae_ft_in22k_in1k_384"
WEIGHTS="mega_matrix/weights/${BACKBONE}.pth"
EVAL_SET="D:/project/data/wm-811k/chip_multilabel_v15direct"

OUT_BASE="outputs"
SUMMARY="${OUT_BASE}/_iter125_gn_extension_summary.log"
: > "$SUMMARY"
echo "$(date) [iter125] (g, n) extension sweep start" | tee -a "$SUMMARY"

train_one() {
    local TAG="$1"; local NG="$2"; local N="$3"
    local OUT_ROOT="${OUT_BASE}/iter125_${TAG}"
    if [ -d "$OUT_ROOT" ]; then
        echo "$(date) [iter125] $TAG already exists, skip" | tee -a "$SUMMARY"
        return 0
    fi
    local GRID=$((NG * N))
    echo "$(date) [iter125] TRAIN $TAG (g=$NG n=$N GRID=$GRID cells=$((GRID*GRID)) per_group=$((N*N)))" | tee -a "$SUMMARY"

    python -X utf8 -m chip_multilabel._train_chip_variant \
        --variant T7 --ls 0.30 --batch 2 --accum 8 --seed 1 \
        --epochs 10 --lr 1e-4 \
        --backbone-timm "$BACKBONE" --img-size 384 \
        --backbone-timm-weights "$WEIGHTS" \
        --no-normal --val-criterion margin_max --save-every-epoch \
        --cutmix-mode complement --cutmix-pair masked --cutmix-pair-fill corner \
        --cutmix-p 0.25 --cutmix-grid-dim $GRID --cutmix-n-groups $NG \
        --cutmix-complete-label-scale 0.5 \
        --tag "iter125_$TAG" --out-root "$OUT_ROOT" \
        >> "${OUT_BASE}/_iter125_${TAG}_train.log" 2>&1
    local RUN=$(ls -d "$OUT_ROOT"/T*/ 2>/dev/null | head -1)
    [ -n "$RUN" ] && rm -f "$RUN"epoch_*.pth || true
    echo "$(date) [iter125] DONE TRAIN $TAG -> $RUN" | tee -a "$SUMMARY"
}

eval_one() {
    local TAG="$1"
    local OUT_ROOT="${OUT_BASE}/iter125_${TAG}"
    local RUN=$(ls -d "$OUT_ROOT"/T*/ 2>/dev/null | head -1)
    [ -z "$RUN" ] && { echo "[iter125] no run for $TAG"; return 0; }
    local EVAL_OUT="${RUN}eval_v15direct"
    [ -d "$EVAL_OUT" ] && [ -n "$(ls -A "$EVAL_OUT" 2>/dev/null)" ] && { echo "$(date) [iter125] $TAG eval exists, skip"; return 0; }
    echo "$(date) [iter125] EVAL $TAG" | tee -a "$SUMMARY"
    python -X utf8 -m chip_multilabel.run_stage1 \
        --model "${RUN}best_model.pth" \
        --eval-set "$EVAL_SET" --out-root "$EVAL_OUT" \
        --variants I3,I7,I10,I13 --n-per-class 200 \
        --strength-min 0.0 --strength-max 1.0 --seed 42 \
        >> "${OUT_BASE}/_iter125_${TAG}_eval.log" 2>&1 || true
}

# === Phase A — g=4 axis NEW ===
train_one a_g4_n1 4 1; eval_one a_g4_n1
train_one b_g4_n2 4 2; eval_one b_g4_n2
train_one c_g4_n3 4 3; eval_one c_g4_n3

# === Phase B — high-n extension ===
train_one d_g2_n5 2 5; eval_one d_g2_n5
train_one e_g3_n4 3 4; eval_one e_g3_n4

# === Phase C — triple-matched GRID=12 completion ===
train_one f_g2_n6 2 6; eval_one f_g2_n6

echo "$(date) [iter125] sweep complete" | tee -a "$SUMMARY"
