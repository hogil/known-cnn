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
#   bash mega_matrix/run.sh --data-base data/wm-811k
#   bash mega_matrix/run.sh --report-only       # only summary.md regen
#
# DDP: bash mega_matrix/run_ddp.sh --gpus N
# =====================================================================
set -e
set -o pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJ_ROOT="$(dirname "$SCRIPT_DIR")"
cd "$PROJ_ROOT"

export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export HF_HUB_DISABLE_TELEMETRY=1
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

OUT_BASE=outputs/_mega_matrix    # shared train/eval data root
BACKBONE="convnextv2_base.fcmae_ft_in22k_in1k_384"
DATA_BASE="${WM811K_ROOT:-$PROJ_ROOT/data/wm-811k}"
DO_DATA=1; DO_TRAIN=1; DO_EVAL=1; DO_PSEUDO=0; DO_REPORT=1
SMOKE=0
EPOCHS=10
TRAIN_SIZES_LIST="50 100 200"
SEL_LIST="f1 margin_max"
EVAL_SIZES_LIST="200 2000 20000"
while [ $# -gt 0 ]; do
    case $1 in
        --skip-data) DO_DATA=0 ;;
        --skip-train) DO_TRAIN=0 ;;
        --skip-eval) DO_EVAL=0 ;;
        --with-pseudo) DO_PSEUDO=1 ;;
        --skip-pseudo) DO_PSEUDO=0 ;;
        --skip-report) DO_REPORT=0 ;;
        --report-only) DO_DATA=0; DO_TRAIN=0; DO_EVAL=0; DO_PSEUDO=0; DO_REPORT=1 ;;
        --data-base) shift; DATA_BASE="$1" ;;
        --data-base=*) DATA_BASE="${1#--data-base=}" ;;
        --backbone) shift; BACKBONE="$1" ;;
        --backbone=*) BACKBONE="${1#--backbone=}" ;;
        --smoke) SMOKE=1 ;;
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

# H100 batch table — sized to ~60 GB / 80 GB (~75% utilization) to leave headroom
# for activations and FCM-PM forward-batch multiplier (up to 4x). All values
# reduced ~25% from prior table that filled VRAM completely (260514 user feedback).
case "$BACKBONE" in
    *convnextv2_large*384*) BATCH_PER_GPU=9 ;;
    *swinv2_base*384*)      BATCH_PER_GPU=12 ;;
    *vit_base*384*)         BATCH_PER_GPU=18 ;;
    *deit3_base*384*)       BATCH_PER_GPU=18 ;;
    *convnextv2_base*384*)  BATCH_PER_GPU=18 ;;
    *convnextv2_base*)      BATCH_PER_GPU=36 ;;   # 224
    *convnextv2_tiny*)      BATCH_PER_GPU=72 ;;   # 224
    *swin_tiny*224*)        BATCH_PER_GPU=72 ;;
    *maxvit_tiny*224*)      BATCH_PER_GPU=48 ;;
    *vit_base*clip*224*)    BATCH_PER_GPU=48 ;;
    *efficientnetv2*)       BATCH_PER_GPU=48 ;;
    *)                      BATCH_PER_GPU=12 ;;
esac

# Smoke mode override: 1 cell × 1 epoch × tiny data × micro batch
if [ $SMOKE -eq 1 ]; then
    BATCH_PER_GPU=2
    EPOCHS=1
    TRAIN_SIZES_LIST="10"
    SEL_LIST="margin_max"
    EVAL_SIZES_LIST="200"
    DO_PSEUDO=0
    DO_REPORT=0
    export MEGA_TRAIN_SIZES="10"
    export MEGA_EVAL_SIZES="200"
fi

TOTAL_CPU=$(nproc 2>/dev/null || echo 8)
TRAIN_WORKERS="$TOTAL_CPU"

MODEL_BASE="$OUT_BASE/$BACKBONE"   # per-backbone model namespace (data shared at OUT_BASE)
RUN_STAMP=$(date +%Y%m%d_%H%M%S)
LOG_BACKBONE="${BACKBONE//\//_}"
LOG_BACKBONE="${LOG_BACKBONE//:/_}"
LOG_BACKBONE="${LOG_BACKBONE// /_}"
LOG_DIR="$OUT_BASE/logs"
LOG="$LOG_DIR/${RUN_STAMP}_pid$$_${LOG_BACKBONE}_run.log"
mkdir -p "$OUT_BASE" "$MODEL_BASE" "$LOG_DIR"
export MEGA_MODEL_BASE="$MODEL_BASE"   # consumed by pseudo_label.py / make_report.py
export MEGA_BACKBONE="$BACKBONE"
export MEGA_IMG_SIZE="$IMG_SIZE"
export WM811K_ROOT="$DATA_BASE"

log() {
    echo "$(date '+%Y-%m-%d %H:%M:%S') [mega] $*" | tee -a "$LOG"
}

: > "$LOG"
trap 'rc=$?; if [ $rc -ne 0 ]; then echo "$(date "+%Y-%m-%d %H:%M:%S") [mega] EXIT_FAIL rc=$rc line=$LINENO" | tee -a "$LOG"; fi' EXIT
CUTMIX_FORWARD_MULT=4  # complement + masked + n_groups=2 expands train forward batch up to 4x
log "start backbone=$BACKBONE img=$IMG_SIZE batch=$BATCH_PER_GPU effective_forward_batch<=$((BATCH_PER_GPU * CUTMIX_FORWARD_MULT)) accum=1 workers=$TRAIN_WORKERS cuda_visible=$CUDA_VISIBLE_DEVICES data_base=$WM811K_ROOT (data=$DO_DATA train=$DO_TRAIN eval=$DO_EVAL pseudo=$DO_PSEUDO report=$DO_REPORT)"

# Offline weights (closed-network) — .pth only
OFFLINE_WEIGHTS="mega_matrix/weights/${BACKBONE}.pth"
if [ -f "$OFFLINE_WEIGHTS" ]; then
    BACKBONE_WEIGHTS_FLAG="--backbone-timm-weights $OFFLINE_WEIGHTS"
    log "offline weights: $OFFLINE_WEIGHTS"
else
    log "ERROR missing offline weights for $BACKBONE under mega_matrix/weights/"
    log "closed-network mode forbids HF/timm download; create mega_matrix/weights/${BACKBONE}.pth first"
    exit 2
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
    local OUT_ROOT="${MODEL_BASE}/${RUN_STAMP}_model_${TAG}"
    if [ -d "$OUT_ROOT" ]; then
        log "SKIP train ${TAG}: exists $OUT_ROOT"
        return 0
    fi
    log "TRAIN ${TAG} ($BACKBONE) batch=$BATCH_PER_GPU workers=$TRAIN_WORKERS"
    set +e
    local TRAIN_STAMP=$(date +%Y%m%d_%H%M%S)
    TRAIN_RUN_STAMP="$TRAIN_STAMP" python -u -m chip_multilabel._train_chip_variant \
        --variant T7 --ls 0.30 --epochs $EPOCHS --batch "$BATCH_PER_GPU" --accum 1 --seed 1 \
        --num-workers "$TRAIN_WORKERS" \
        --lr 1e-4 --no-normal --val-criterion ${SEL} --save-every-epoch \
        --data-root "${OUT_BASE}/train_n${TN}" \
        --cutmix-mode complement --cutmix-pair masked --cutmix-pair-fill corner \
        --cutmix-p 0.25 --cutmix-grid-dim 16 --cutmix-n-groups 2 --cutmix-complete-label-scale 0.5 \
        --backbone-timm "$BACKBONE" --img-size $IMG_SIZE \
        $BACKBONE_WEIGHTS_FLAG \
        --out-root "$OUT_ROOT" --tag "${TAG}" \
        2>&1 | tee -a "$LOG"
    local TRAIN_RC=${PIPESTATUS[0]}
    local RUN=$(find "$OUT_ROOT" -mindepth 2 -maxdepth 2 -name best_model.pth -printf '%h\n' 2>/dev/null | sort -r | head -1)
    [ -n "$RUN" ] && rm -f "$RUN"/epoch_*.pth
    set -e
    if [ "$TRAIN_RC" -ne 0 ]; then
        log "TRAIN_FAIL ${TAG} rc=$TRAIN_RC"
        return 0
    fi
    log "DONE ${TAG}"
}
export -f train_one

if [ $DO_TRAIN -eq 1 ]; then
    log "=== STAGE 2: training ==="
    for TN in $TRAIN_SIZES_LIST; do
        for SEL in $SEL_LIST; do
            train_one $TN $SEL
        done
    done
fi

# ======================================================================
# Stage 3: eval (18 cells = 6 models × 3 eval_n)
# ======================================================================
eval_one() {
    local TN=$1; local SEL=$2; local EN=$3
    local MODEL_ROOT=$(find "$MODEL_BASE" -mindepth 1 -maxdepth 1 -type d \( -name "*_model_train${TN}_${SEL}" -o -name "model_train${TN}_${SEL}" \) -printf '%p\n' 2>/dev/null | sort -r | head -1)
    local RUN=$(find "$MODEL_ROOT" -mindepth 2 -maxdepth 2 -name best_model.pth -printf '%h\n' 2>/dev/null | sort -r | head -1)
    if [ -z "$RUN" ]; then
        log "SKIP eval train${TN}_${SEL} eval_${EN}: no trained run under $MODEL_ROOT"
        return 0
    fi
    local EVAL_OUT="${RUN}/eval_${EN}"
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
        --model "${RUN}/best_model.pth" \
        --eval-set "$EVAL_SET" --out-root "$EVAL_OUT" \
        --variants I3,I7,I10,I13 --n-per-class 99999 \
        --strength-min 0.0 --strength-max 1.0 --seed 42 \
        2>&1 | tee -a "$LOG" || log "EVAL_FAIL train${TN}_${SEL} eval_${EN}"
}

if [ $DO_EVAL -eq 1 ]; then
    log "=== STAGE 3: evaluation ==="
    for TN in $TRAIN_SIZES_LIST; do
        for SEL in $SEL_LIST; do
            for EN in $EVAL_SIZES_LIST; do
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
