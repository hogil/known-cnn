#!/bin/bash
# =====================================================================
# Mega matrix sweep — all-in-one runner (single GPU)
#
# train_n / class ∈ {50, 100, 200}
# eval_n / class  ∈ {200, 2000, 20000}
# selection       ∈ {val_f1, val_margin}
# recipe          = iter126e (LS=0.30, g=2, grid_dim=16, ep=10)
# + Optional Stage 4: pseudo-label retrain (OFF by default)
#
# Usage:
#   bash mega_matrix/run.sh                     # data + train + eval + report
#   bash mega_matrix/run.sh --skip-data         # data 이미 있을 때
#   bash mega_matrix/run.sh --skip-train        # eval+report only
#   bash mega_matrix/run.sh --with-pseudo       # also run pseudo-label retrain
#   bash mega_matrix/run.sh --report-only       # only summary.md regen
#
# DDP: bash mega_matrix/run_ddp.sh --gpus N
# =====================================================================
set -e
set -o pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJ_ROOT="$(dirname "$SCRIPT_DIR")"
cd "$PROJ_ROOT"

OUT_BASE=outputs/_mega_matrix    # shared train/eval data root
BACKBONE="convnextv2_base.fcmae_ft_in22k_in1k_384"
DO_DATA=1; DO_TRAIN=1; DO_EVAL=1; DO_PSEUDO=0; DO_REPORT=1
while [ $# -gt 0 ]; do
    case $1 in
        --skip-data) DO_DATA=0 ;;
        --skip-train) DO_TRAIN=0 ;;
        --skip-eval) DO_EVAL=0 ;;
        --with-pseudo) DO_PSEUDO=1 ;;
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

log() {
    echo "$(date) [mega] $*" | tee -a "$LOG"
}

: > "$LOG"
trap 'rc=$?; if [ $rc -ne 0 ]; then echo "$(date) [mega] EXIT_FAIL rc=$rc line=$LINENO" | tee -a "$LOG"; fi' EXIT
log "start backbone=$BACKBONE img=$IMG_SIZE (data=$DO_DATA train=$DO_TRAIN eval=$DO_EVAL pseudo=$DO_PSEUDO report=$DO_REPORT)"

# Offline weights (closed-network) — auto-passthrough if file exists
OFFLINE_WEIGHTS="mega_matrix/weights/${BACKBONE}.pth"
[ ! -f "$OFFLINE_WEIGHTS" ] && OFFLINE_WEIGHTS="mega_matrix/weights/${BACKBONE}.safetensors"
[ ! -f "$OFFLINE_WEIGHTS" ] && OFFLINE_WEIGHTS="mega_matrix/weights/${BACKBONE}.bin"
if [ -f "$OFFLINE_WEIGHTS" ]; then
    BACKBONE_WEIGHTS_FLAG="--backbone-timm-weights $OFFLINE_WEIGHTS"
    log "offline weights: $OFFLINE_WEIGHTS"
else
    BACKBONE_WEIGHTS_FLAG=""
    log "online mode (no $OFFLINE_WEIGHTS) - timm will fetch from HF"
fi

# ======================================================================
# Stage 1: data generation
# ======================================================================
if [ $DO_DATA -eq 1 ]; then
    log "=== STAGE 1: data generation ==="
    python -u -X utf8 mega_matrix/gen_data.py 2>&1 | tee -a "$LOG"
fi

# ======================================================================
# Stage 2: training (6 cells = 3 train_n × 2 selection)
# ======================================================================
train_one() {
    local TN=$1; local SEL=$2
    local TAG="train${TN}_${SEL}"
    local OUT_ROOT="${MODEL_BASE}/model_${TAG}"
    if [ -d "$OUT_ROOT" ]; then
        log "SKIP train ${TAG}: exists $OUT_ROOT"
        return 0
    fi
    log "TRAIN ${TAG} ($BACKBONE)"
    set +e
    python -u -m chip_multilabel._train_chip_variant \
        --variant T7 --ls 0.30 --epochs 10 --batch 2 --accum 8 --seed 1 \
        --lr 1e-4 --no-normal --val-criterion ${SEL} --save-every-epoch \
        --data-root "${OUT_BASE}/train_n${TN}" \
        --cutmix-mode complement --cutmix-pair masked --cutmix-pair-fill corner \
        --cutmix-p 0.25 --cutmix-grid-dim 16 --cutmix-n-groups 2 --cutmix-complete-label-scale 0.5 \
        --backbone-timm "$BACKBONE" --img-size $IMG_SIZE \
        $BACKBONE_WEIGHTS_FLAG \
        --out-root "$OUT_ROOT" --tag "${TAG}" \
        2>&1 | tee -a "$LOG"
    local TRAIN_RC=${PIPESTATUS[0]}
    local RUN=$(ls -d "$OUT_ROOT"/T*/ 2>/dev/null | head -1)
    [ -n "$RUN" ] && rm -f "$RUN"epoch_*.pth
    set -e
    if [ "$TRAIN_RC" -ne 0 ]; then
        log "TRAIN_FAIL ${TAG} rc=$TRAIN_RC"
        return 0
    fi
    log "DONE ${TAG}"
}
export -f train_one

if [ $DO_TRAIN -eq 1 ]; then
    log "=== STAGE 2: training (6 models) ==="
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
    if [ -z "$RUN" ]; then
        log "SKIP eval train${TN}_${SEL} eval_${EN}: no trained run under $MODEL_ROOT"
        return 0
    fi
    local EVAL_OUT="${RUN}eval_${EN}"
    if [ -d "$EVAL_OUT" ]; then
        log "SKIP eval train${TN}_${SEL} eval_${EN}: exists $EVAL_OUT"
        return 0
    fi
    local EVAL_SET="${OUT_BASE}/eval_n${EN}"
    if [ ! -d "$EVAL_SET" ]; then
        log "SKIP eval train${TN}_${SEL} eval_${EN}: missing $EVAL_SET"
        return 0
    fi
    log "EVAL train${TN}_${SEL} eval_${EN} ($BACKBONE)"
    python -u -m chip_multilabel.run_stage1 \
        --model "${RUN}best_model.pth" \
        --eval-set "$EVAL_SET" --out-root "$EVAL_OUT" \
        --variants I3,I7,I10,I13 --n-per-class 99999 \
        --strength-min 0.0 --strength-max 1.0 --seed 42 \
        2>&1 | tee -a "$LOG" || log "EVAL_FAIL train${TN}_${SEL} eval_${EN}"
}

if [ $DO_EVAL -eq 1 ]; then
    log "=== STAGE 3: evaluation (18 cells) ==="
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
    log "=== STAGE 4: pseudo-label retrain ==="
    python -u -X utf8 mega_matrix/pseudo_label.py 2>&1 | tee -a "$LOG"
fi

# ======================================================================
# Stage 5: report (summary.md + plots)
# ======================================================================
if [ $DO_REPORT -eq 1 ]; then
    log "=== STAGE 5: report ==="
    python -u -X utf8 mega_matrix/make_report.py 2>&1 | tee -a "$LOG"
fi

log "ALL DONE"
echo "-> docs/chip-multilabel/manager_report/summary_mega_sweep.md" | tee -a "$LOG"
