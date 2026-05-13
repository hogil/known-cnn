#!/bin/bash
# iter126 — FCM-PM (g, n) MORE diversity (5 cells, ~40 min)
#
# Adds: g=5 axis NEW + g=6 axis (single cell for GRID=6 quadruple-match)
#       + g=4 n=4 (g=n diagonal) + g=2 n=8 (high-n extreme)
#
# After iter124 (9 cells: g=2 n=1-4, g=3 n=1-3, bisect_h/v)
# After iter125 (6 cells: g=4 n=1-3, g=2 n=5/6, g=3 n=4)
# iter126 (this) — extreme corners + multi-match GRIDs for paper figure
#
# Multi-match GRIDs (paper §5.47 KEY figure):
#   GRID=6  quadruple: 124c (g=2 n=3) + 124f (g=3 n=2) + 126c (g=6 n=1) NEW
#   GRID=10 dual:      125d (g=2 n=5) + 126b (g=5 n=2) NEW
#   GRID=12 triple:    125c (g=4 n=3) + 125e (g=3 n=4) + 125f (g=2 n=6) — done
#   GRID=16 dual:      126d (g=4 n=4) NEW + 126e (g=2 n=8) NEW
#
# Same frozen baseline as iter124/125:
#   T7 BCE+LS=0.30, lr 1e-4, ep 10, seed=1, val_margin, save_every_epoch,
#   cutmix-mode=complement, pair=masked, fill=corner, p=0.25, label-scale=0.5.

set -e
cd "$(dirname "$0")"

BACKBONE="convnextv2_base.fcmae_ft_in22k_in1k_384"
WEIGHTS="mega_matrix/weights/${BACKBONE}.pth"
EVAL_SET="D:/project/data/wm-811k/chip_multilabel_v15direct"

OUT_BASE="outputs"
SUMMARY="${OUT_BASE}/_iter126_more_diversity_summary.log"
: > "$SUMMARY"
echo "$(date) [iter126] (g, n) more-diversity sweep start" | tee -a "$SUMMARY"

train_one() {
    local TAG="$1"; local NG="$2"; local N="$3"
    local OUT_ROOT="${OUT_BASE}/iter126_${TAG}"
    if [ -d "$OUT_ROOT" ]; then
        echo "$(date) [iter126] $TAG already exists, skip" | tee -a "$SUMMARY"
        return 0
    fi
    local GRID=$((NG * N))
    echo "$(date) [iter126] TRAIN $TAG (g=$NG n=$N GRID=$GRID cells=$((GRID*GRID)) per_group=$((N*N)))" | tee -a "$SUMMARY"

    python -X utf8 -m chip_multilabel._train_chip_variant \
        --variant T7 --ls 0.30 --batch 2 --accum 8 --seed 1 \
        --epochs 10 --lr 1e-4 \
        --backbone-timm "$BACKBONE" --img-size 384 \
        --backbone-timm-weights "$WEIGHTS" \
        --no-normal --val-criterion margin_max --save-every-epoch \
        --cutmix-mode complement --cutmix-pair masked --cutmix-pair-fill corner \
        --cutmix-p 0.25 --cutmix-grid-dim $GRID --cutmix-n-groups $NG \
        --cutmix-complete-label-scale 0.5 \
        --tag "iter126_$TAG" --out-root "$OUT_ROOT" \
        >> "${OUT_BASE}/_iter126_${TAG}_train.log" 2>&1
    local RUN=$(ls -d "$OUT_ROOT"/T*/ 2>/dev/null | head -1)
    [ -n "$RUN" ] && rm -f "$RUN"epoch_*.pth || true
    echo "$(date) [iter126] DONE TRAIN $TAG -> $RUN" | tee -a "$SUMMARY"
}

eval_one() {
    local TAG="$1"
    local OUT_ROOT="${OUT_BASE}/iter126_${TAG}"
    local RUN=$(ls -d "$OUT_ROOT"/T*/ 2>/dev/null | head -1)
    [ -z "$RUN" ] && { echo "[iter126] no run for $TAG"; return 0; }
    local EVAL_OUT="${RUN}eval_v15direct"
    [ -d "$EVAL_OUT" ] && [ -n "$(ls -A "$EVAL_OUT" 2>/dev/null)" ] && { echo "$(date) [iter126] $TAG eval exists, skip"; return 0; }
    echo "$(date) [iter126] EVAL $TAG" | tee -a "$SUMMARY"
    python -X utf8 -m chip_multilabel.run_stage1 \
        --model "${RUN}best_model.pth" \
        --eval-set "$EVAL_SET" --out-root "$EVAL_OUT" \
        --variants I3,I7,I10,I13 --n-per-class 200 \
        --strength-min 0.0 --strength-max 1.0 --seed 42 \
        >> "${OUT_BASE}/_iter126_${TAG}_eval.log" 2>&1 || true
}

# === Phase A — g=5 axis NEW (2 cells) ===
train_one a_g5_n1 5 1; eval_one a_g5_n1
train_one b_g5_n2 5 2; eval_one b_g5_n2

# === Phase B — GRID=6 quadruple-match (g=6 n=1, paper figure 3rd point) ===
train_one c_g6_n1 6 1; eval_one c_g6_n1

# === Phase C — GRID=16 dual-match (g=n diagonal + high-n extreme) ===
train_one d_g4_n4 4 4; eval_one d_g4_n4
train_one e_g2_n8 2 8; eval_one e_g2_n8

echo "$(date) [iter126] sweep complete" | tee -a "$SUMMARY"
