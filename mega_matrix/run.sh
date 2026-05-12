#!/bin/bash
# =====================================================================
# Mega matrix sweep — all-in-one runner (single GPU)
#
# train_n / class ∈ {50, 100, 200}
# eval_n / class  ∈ {200, 2000, 20000}
# selection       ∈ {val_f1, val_margin}
# + Stage 5: pseudo-label retrain
#
# Usage:
#   bash mega_matrix/run.sh                     # full
#   bash mega_matrix/run.sh --skip-data         # data 이미 있을 때
#   bash mega_matrix/run.sh --skip-train        # eval+report only
#   bash mega_matrix/run.sh --skip-pseudo       # stage 5 skip
#   bash mega_matrix/run.sh --report-only       # only summary.md regen
#
# DDP: bash mega_matrix/run_ddp.sh --gpus N
# =====================================================================
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJ_ROOT="$(dirname "$SCRIPT_DIR")"
cd "$PROJ_ROOT"

OUT_BASE=outputs/_mega_matrix    # shared train/eval data root
BACKBONE="convnextv2_base.fcmae_ft_in22k_in1k_384"
DO_DATA=1; DO_TRAIN=1; DO_EVAL=1; DO_PSEUDO=1; DO_REPORT=1
while [ $# -gt 0 ]; do
    case $1 in
        --skip-data) DO_DATA=0 ;;
        --skip-train) DO_TRAIN=0 ;;
        --skip-eval) DO_EVAL=0 ;;
        --skip-pseudo) DO_PSEUDO=0 ;;
        --skip-report) DO_REPORT=0 ;;
        --report-only) DO_DATA=0; DO_TRAIN=0; DO_EVAL=0; DO_PSEUDO=0; DO_REPORT=1 ;;
        --backbone) shift; BACKBONE="$1" ;;
        --backbone=*) BACKBONE="${1#--backbone=}" ;;
        --help|-h) head -20 "$0" | tail -16; exit 0 ;;
    esac
    shift
done

# Derive input img-size from backbone name pattern
case "$BACKBONE" in
    *384*) IMG_SIZE=384 ;;
    *256*) IMG_SIZE=256 ;;
    *)     IMG_SIZE=224 ;;
esac

MODEL_BASE="$OUT_BASE/$BACKBONE"   # per-backbone model namespace (data shared at OUT_BASE)
LOG="$MODEL_BASE/_run.log"
mkdir -p "$OUT_BASE" "$MODEL_BASE"
export MEGA_MODEL_BASE="$MODEL_BASE"   # consumed by pseudo_label.py / make_report.py
export MEGA_BACKBONE="$BACKBONE"
export MEGA_IMG_SIZE="$IMG_SIZE"

echo "$(date) [mega] start backbone=$BACKBONE img=$IMG_SIZE (data=$DO_DATA train=$DO_TRAIN eval=$DO_EVAL pseudo=$DO_PSEUDO report=$DO_REPORT)" > "$LOG"

# Offline weights (closed-network) — auto-passthrough if file exists
OFFLINE_WEIGHTS="mega_matrix/weights/${BACKBONE}.pth"
[ ! -f "$OFFLINE_WEIGHTS" ] && OFFLINE_WEIGHTS="mega_matrix/weights/${BACKBONE}.safetensors"
[ ! -f "$OFFLINE_WEIGHTS" ] && OFFLINE_WEIGHTS="mega_matrix/weights/${BACKBONE}.bin"
if [ -f "$OFFLINE_WEIGHTS" ]; then
    BACKBONE_WEIGHTS_FLAG="--backbone-timm-weights $OFFLINE_WEIGHTS"
    echo "$(date) [mega] offline weights: $OFFLINE_WEIGHTS" >> "$LOG"
else
    BACKBONE_WEIGHTS_FLAG=""
    echo "$(date) [mega] online mode (no $OFFLINE_WEIGHTS) - timm will fetch from HF" >> "$LOG"
fi

# ======================================================================
# Stage 1: data generation
# ======================================================================
if [ $DO_DATA -eq 1 ]; then
    echo "$(date) [mega] === STAGE 1: data generation ===" >> "$LOG"
    python -u -X utf8 mega_matrix/gen_data.py >> "$LOG" 2>&1
fi

# ======================================================================
# Stage 2: training (6 cells = 3 train_n × 2 selection)
# ======================================================================
train_one() {
    local TN=$1; local SEL=$2
    local TAG="train${TN}_${SEL}"
    local OUT_ROOT="${MODEL_BASE}/model_${TAG}"
    [ -d "$OUT_ROOT" ] && return 0
    echo "$(date) [mega] TRAIN ${TAG} ($BACKBONE)" >> "$LOG"
    set +e
    python -u -m chip_multilabel._train_chip_variant \
        --variant T7 --ls 0.30 --epochs 10 --batch 2 --accum 8 --seed 1 \
        --lr 1e-4 --no-normal --val-criterion ${SEL} --save-every-epoch \
        --data-root "${OUT_BASE}/train_n${TN}" \
        --cutmix-mode complement --cutmix-pair masked --cutmix-pair-fill corner \
        --cutmix-p 0.25 --cutmix-n-groups 3 --cutmix-complete-label-scale 0.5 \
        --backbone-timm "$BACKBONE" --img-size $IMG_SIZE \
        $BACKBONE_WEIGHTS_FLAG \
        --out-root "$OUT_ROOT" --tag "${TAG}" \
        >> "$LOG" 2>&1
    local RUN=$(ls -d "$OUT_ROOT"/T*/ 2>/dev/null | head -1)
    [ -n "$RUN" ] && rm -f "$RUN"epoch_*.pth
    set -e
    echo "$(date) [mega] DONE ${TAG}" >> "$LOG"
}
export -f train_one

if [ $DO_TRAIN -eq 1 ]; then
    echo "$(date) [mega] === STAGE 2: training (6 models) ===" >> "$LOG"
    for TN in 50 100 200; do
        for SEL in f1 margin_max; do
            train_one $TN $SEL
        done
    done
fi

# ======================================================================
# Stage 3: eval (18 cells = 6 models × 3 eval_n)
# ======================================================================
eval_one() {
    local TN=$1; local SEL=$2; local EN=$3
    local MODEL_ROOT="${MODEL_BASE}/model_train${TN}_${SEL}"
    local RUN=$(ls -d "$MODEL_ROOT"/T*/ 2>/dev/null | head -1)
    [ -z "$RUN" ] && return 0
    local EVAL_OUT="${RUN}eval_${EN}"
    [ -d "$EVAL_OUT" ] && return 0
    local EVAL_SET="${OUT_BASE}/eval_n${EN}"
    [ ! -d "$EVAL_SET" ] && return 0
    echo "$(date) [mega] EVAL train${TN}_${SEL} eval_${EN} ($BACKBONE)" >> "$LOG"
    python -u -m chip_multilabel.run_stage1 \
        --model "${RUN}best_model.pth" \
        --eval-set "$EVAL_SET" --out-root "$EVAL_OUT" \
        --variants I3,I7,I10,I13 --n-per-class 99999 \
        --strength-min 0.0 --strength-max 1.0 --seed 42 \
        >> "$LOG" 2>&1 || true
}

if [ $DO_EVAL -eq 1 ]; then
    echo "$(date) [mega] === STAGE 3: evaluation (18 cells) ===" >> "$LOG"
    for TN in 50 100 200; do
        for SEL in f1 margin_max; do
            for EN in 200 2000 20000; do
                eval_one $TN $SEL $EN
            done
        done
    done
fi

# ======================================================================
# Stage 4: pseudo-label retrain + eval
# ======================================================================
if [ $DO_PSEUDO -eq 1 ]; then
    echo "$(date) [mega] === STAGE 4: pseudo-label retrain ===" >> "$LOG"
    python -u -X utf8 mega_matrix/pseudo_label.py >> "$LOG" 2>&1
fi

# ======================================================================
# Stage 5: report (summary.md + plots)
# ======================================================================
if [ $DO_REPORT -eq 1 ]; then
    echo "$(date) [mega] === STAGE 5: report ===" >> "$LOG"
    python -u -X utf8 mega_matrix/make_report.py >> "$LOG" 2>&1
fi

echo "$(date) [mega] ALL DONE" >> "$LOG"
echo "→ docs/chip-multilabel/manager_report/summary_mega_sweep.md" >> "$LOG"
