#!/bin/bash
# Wave 2 — pos_target × neg_target 2D matrix (independent per 260513 absolute rule)
#
# User directive: pos_t >= 0.7, neg_t <= 0.3
# Matrix: pos_t ∈ {1.0, 0.95, 0.9, 0.85, 0.8, 0.75, 0.7} × neg_t ∈ {0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3}
# = 7×7 = 49 cells, ~5.8 hr sequential
#
# Frozen recipe (iter126e TIE winner base):
#   g=2 n=8 GRID=16 (winner candidate), cutmix-p=0.25, label-scale=0.5,
#   pair=masked, fill=corner, T7 (BCE+target), 10 epochs, seed=1
#
# Cell naming: W2_pt{P}_nt{N}    e.g. W2_pt90_nt10 = pos_t=0.9, neg_t=0.1

set -e
cd "$(dirname "$0")"

BACKBONE="convnextv2_base.fcmae_ft_in22k_in1k_384"
WEIGHTS="mega_matrix/weights/${BACKBONE}.pth"
EVAL_SET="D:/project/data/wm-811k/chip_multilabel_v15direct"

OUT_BASE="outputs"
SUMMARY="${OUT_BASE}/_W2_pt_nt_summary.log"
: > "$SUMMARY"
echo "$(date) [W2] pos_t × neg_t 2D matrix sweep start (49 cells, ~5.8 hr)" | tee -a "$SUMMARY"

train_one() {
    local PT_INT="$1"; local NT_INT="$2"   # pt_int = pos_t * 100, nt_int = neg_t * 100
    local PT=$(python -c "print($PT_INT / 100.0)")
    local NT=$(python -c "print($NT_INT / 100.0)")
    local TAG="W2_pt${PT_INT}_nt${NT_INT}"
    local OUT_ROOT="${OUT_BASE}/${TAG}"
    if [ -d "$OUT_ROOT" ]; then
        local DONE=$(find "$OUT_ROOT" -maxdepth 2 -name "best_model.pth" 2>/dev/null | wc -l)
        if [ "$DONE" -ge 1 ]; then
            echo "$(date) [W2] $TAG already done, skip" | tee -a "$SUMMARY"
            return 0
        else
            rm -rf "$OUT_ROOT"
        fi
    fi
    echo "$(date) [W2] TRAIN $TAG (pos_t=$PT neg_t=$NT)" | tee -a "$SUMMARY"

    python -X utf8 -m chip_multilabel._train_chip_variant \
        --variant T7 --pos-target $PT --neg-target $NT \
        --batch 2 --accum 8 --seed 1 \
        --epochs 10 --lr 1e-4 \
        --backbone-timm "$BACKBONE" --img-size 384 \
        --backbone-timm-weights "$WEIGHTS" \
        --no-normal --val-criterion margin_max --save-every-epoch \
        --cutmix-mode complement --cutmix-pair masked --cutmix-pair-fill corner \
        --cutmix-p 0.25 --cutmix-grid-dim 16 --cutmix-n-groups 2 \
        --cutmix-complete-label-scale 0.5 \
        --tag "$TAG" --out-root "$OUT_ROOT" \
        >> "${OUT_BASE}/_${TAG}_train.log" 2>&1
    local RUN=$(ls -d "$OUT_ROOT"/T*/ 2>/dev/null | head -1)
    [ -n "$RUN" ] && rm -f "$RUN"epoch_*.pth || true
    echo "$(date) [W2] DONE TRAIN $TAG" | tee -a "$SUMMARY"
}

eval_one() {
    local PT_INT="$1"; local NT_INT="$2"
    local TAG="W2_pt${PT_INT}_nt${NT_INT}"
    local OUT_ROOT="${OUT_BASE}/${TAG}"
    local RUN=$(ls -d "$OUT_ROOT"/T*/ 2>/dev/null | head -1)
    [ -z "$RUN" ] && return 0
    local EVAL_OUT="${RUN}eval_v15direct"
    if [ -d "$EVAL_OUT" ] && [ -n "$(ls -A "$EVAL_OUT" 2>/dev/null)" ]; then
        return 0
    fi
    echo "$(date) [W2] EVAL $TAG" | tee -a "$SUMMARY"
    python -X utf8 -m chip_multilabel.run_stage1 \
        --model "${RUN}best_model.pth" \
        --eval-set "$EVAL_SET" --out-root "$EVAL_OUT" \
        --variants I3,I7,I10,I13 --n-per-class 200 \
        --strength-min 0.0 --strength-max 1.0 --seed 42 \
        >> "${OUT_BASE}/_${TAG}_eval.log" 2>&1 || true
}

# 7x7 matrix: pos_t in {100,95,90,85,80,75,70}, neg_t in {0,5,10,15,20,25,30}
# Dispatch order: high pos_t first (most likely winners), then descend
for PT in 100 95 90 85 80 75 70; do
    echo "$(date) [W2] === pos_t = 0.${PT} phase ===" | tee -a "$SUMMARY"
    for NT in 0 5 10 15 20 25 30; do
        train_one $PT $NT
        eval_one $PT $NT
    done
done

echo "$(date) [W2] === pos_t × neg_t 2D matrix sweep COMPLETE ===" | tee -a "$SUMMARY"
