#!/bin/bash
# =====================================================================
# Mega matrix sweep — DDP-style parallel runner (server multi-GPU)
#
# Distributes 6 train cells across N GPUs (default 4).
# Recipe = iter126e (LS=0.30, g=2, grid_dim=16, ep=10).
#   GPU 0: train50_f1, train50_margin_max
#   GPU 1: train100_f1, train100_margin_max
#   GPU 2: train200_f1
#   GPU 3: train200_margin_max
#   (2-GPU mode: 3 cells/GPU; 6-GPU mode: 1 cell/GPU)
#
# Usage on server:
#   cd /path/to/known-cnn
#   bash mega_matrix/run_ddp.sh                # auto-detect GPU count
#   bash mega_matrix/run_ddp.sh --gpus 4       # force 4 GPU
#   bash mega_matrix/run_ddp.sh --skip-data    # skip data gen
#   bash mega_matrix/run_ddp.sh --with-pseudo  # also run pseudo-label retrain
#
# NOTE: vanilla PyTorch single-GPU training per cell (NOT torch.distributed DDP).
#       For true torchrun DDP inside one cell, trainer code needs modification
#       (DistributedSampler, init_process_group, etc.) — out of scope here.
#       This script uses "cell-level data parallelism" — different cells on
#       different GPUs simultaneously, single GPU per cell.
# =====================================================================
set -e
set -o pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJ_ROOT="$(dirname "$SCRIPT_DIR")"
cd "$PROJ_ROOT"

OUT_BASE=outputs/_mega_matrix
BACKBONE="convnextv2_base.fcmae_ft_in22k_in1k_384"

# Default flags
DO_DATA=1; DO_EVAL=1; DO_REPORT=1; DO_PSEUDO=0
NGPU=$(nvidia-smi -L 2>/dev/null | wc -l)
[ "$NGPU" -lt 1 ] && NGPU=1

while [ $# -gt 0 ]; do
    case $1 in
        --skip-data) DO_DATA=0 ;;
        --skip-eval) DO_EVAL=0 ;;
        --with-pseudo) DO_PSEUDO=1 ;;
        --skip-pseudo) DO_PSEUDO=0 ;;
        --skip-report) DO_REPORT=0 ;;
        --gpus) shift; NGPU=$1 ;;
        --gpus=*) NGPU="${1#--gpus=}" ;;
        --backbone) shift; BACKBONE="$1" ;;
        --backbone=*) BACKBONE="${1#--backbone=}" ;;
        --help|-h) head -25 "$0" | tail -20; exit 0 ;;
    esac
    shift
done

# Derive input img-size from backbone name pattern
case "$BACKBONE" in
    *384*) IMG_SIZE=384 ;;
    *256*) IMG_SIZE=256 ;;
    *)     IMG_SIZE=224 ;;
esac

# Per-rank batch for H100 80GB + bf16 amp (per-rank, NOT divided by world_size).
# Effective global batch = BATCH_PER_GPU * world_size.
case "$BACKBONE" in
    *convnextv2_large*384*) BATCH_PER_GPU=24 ;;
    *swinv2_base*384*)      BATCH_PER_GPU=32 ;;
    *vit_base*384*)         BATCH_PER_GPU=48 ;;
    *deit3_base*384*)       BATCH_PER_GPU=48 ;;
    *convnextv2_base*384*)  BATCH_PER_GPU=48 ;;
    *convnextv2_base*)      BATCH_PER_GPU=96 ;;   # 224
    *convnextv2_tiny*)      BATCH_PER_GPU=192 ;;  # 224
    *swin_tiny*224*)        BATCH_PER_GPU=192 ;;
    *maxvit_tiny*224*)      BATCH_PER_GPU=128 ;;
    *vit_base*clip*224*)    BATCH_PER_GPU=128 ;;
    *efficientnetv2*)       BATCH_PER_GPU=128 ;;
    *)                      BATCH_PER_GPU=32 ;;   # safe fallback
esac

# DataLoader workers: 64 CPU server / N GPU - 8 ~ 16 per rank safe.
TOTAL_CPU=$(nproc 2>/dev/null || echo 8)
WORKERS_PER_RANK=$(( TOTAL_CPU / NGPU / 2 ))
[ "$WORKERS_PER_RANK" -lt 4 ]  && WORKERS_PER_RANK=4
[ "$WORKERS_PER_RANK" -gt 16 ] && WORKERS_PER_RANK=16

MODEL_BASE="$OUT_BASE/$BACKBONE"
LOG="$MODEL_BASE/_run_ddp.log"
mkdir -p "$OUT_BASE" "$MODEL_BASE"
export MEGA_MODEL_BASE="$MODEL_BASE"

log() {
    echo "$(date) [ddp] $*" | tee -a "$LOG"
}

: > "$LOG"
trap 'rc=$?; if [ $rc -ne 0 ]; then echo "$(date) [ddp] EXIT_FAIL rc=$rc line=$LINENO" | tee -a "$LOG"; fi' EXIT
log "start backbone=$BACKBONE img=$IMG_SIZE N_GPU=$NGPU data=$DO_DATA eval=$DO_EVAL pseudo=$DO_PSEUDO report=$DO_REPORT"

# Offline weights (closed-network) — auto-passthrough
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
export BACKBONE_WEIGHTS_FLAG BACKBONE IMG_SIZE MODEL_BASE
export MEGA_MODEL_BASE="$MODEL_BASE"
export MEGA_BACKBONE="$BACKBONE"
export MEGA_IMG_SIZE="$IMG_SIZE"

# ======================================================================
# 1. Data generation (sequential, single CPU)
# ======================================================================
if [ $DO_DATA -eq 1 ]; then
    log "=== STAGE 1: data generation ==="
    python -u -X utf8 mega_matrix/gen_data.py 2>&1 | tee -a "$LOG"
fi

# ======================================================================
# 2. Training — distribute 6 cells across $NGPU GPUs
# ======================================================================
# 6 cells:
#   (50, f1)         (50, margin_max)
#   (100, f1)        (100, margin_max)
#   (200, f1)        (200, margin_max)

train_cell() {
    local TN=$1; local SEL=$2
    local TAG="train${TN}_${SEL}"
    local OUT_ROOT="${MODEL_BASE}/model_${TAG}"
    if [ -d "$OUT_ROOT" ]; then
        log "SKIP train ${TAG}: exists $OUT_ROOT"
        return 0
    fi
    log "TRAIN ${TAG} ($BACKBONE) NGPU=$NGPU batch_per_gpu=$BATCH_PER_GPU"
    # True torchrun DDP: each rank sees --batch as its OWN batch (per-rank).
    # effective global batch = BATCH_PER_GPU * NGPU
    set +e
    torchrun --standalone --nproc_per_node="$NGPU" \
        -m chip_multilabel._train_chip_variant \
        --variant T7 --ls 0.30 --epochs 10 \
        --batch "$BATCH_PER_GPU" --accum 1 --seed 1 \
        --num-workers "$WORKERS_PER_RANK" \
        --lr 1e-4 --no-normal --val-criterion ${SEL} --save-every-epoch \
        --data-root "${OUT_BASE}/train_n${TN}" \
        --cutmix-mode complement --cutmix-pair masked --cutmix-pair-fill corner \
        --cutmix-p 0.25 --cutmix-grid-dim 16 --cutmix-n-groups 2 --cutmix-complete-label-scale 0.5 \
        --backbone-timm "$BACKBONE" --img-size $IMG_SIZE \
        $BACKBONE_WEIGHTS_FLAG \
        --out-root "$OUT_ROOT" --tag "${TAG}" \
        2>&1 | tee -a "${LOG}.train"
    local TRAIN_RC=${PIPESTATUS[0]}
    set -e
    local RUN=$(ls -d "$OUT_ROOT"/T*/ 2>/dev/null | head -1)
    [ -n "$RUN" ] && rm -f "$RUN"epoch_*.pth
    if [ "$TRAIN_RC" -ne 0 ]; then
        log "TRAIN_FAIL ${TAG} rc=$TRAIN_RC"
        return 0
    fi
    log "DONE ${TAG}"
}

log "=== STAGE 2: sequential torchrun DDP (6 cells on $NGPU GPUs each) ==="
# True DDP: each train_cell uses ALL $NGPU GPUs via torchrun. Cells run
# sequentially (since they share GPUs). Effective global batch per cell =
# BATCH_PER_GPU * NGPU.
for TN in 50 100 200; do
    for SEL in f1 margin_max; do
        train_cell "$TN" "$SEL"
    done
done
log "all trainings complete"

# ======================================================================
# 3. Eval — distribute 18 cells across $NGPU GPUs
# ======================================================================
eval_cell() {
    local GPU=$1; local TN=$2; local SEL=$3; local EN=$4
    local MODEL_ROOT="${MODEL_BASE}/model_train${TN}_${SEL}"
    local RUN=$(ls -d "$MODEL_ROOT"/T*/ 2>/dev/null | head -1)
    if [ -z "$RUN" ]; then
        log "SKIP eval train${TN}_${SEL} eval${EN}: no trained run under $MODEL_ROOT"
        return 0
    fi
    local EVAL_OUT="${RUN}eval_${EN}"
    if [ -d "$EVAL_OUT" ]; then
        log "SKIP eval train${TN}_${SEL} eval${EN}: exists $EVAL_OUT"
        return 0
    fi
    local EVAL_SET="${OUT_BASE}/eval_n${EN}"
    if [ ! -d "$EVAL_SET" ]; then
        log "SKIP eval train${TN}_${SEL} eval${EN}: missing $EVAL_SET"
        return 0
    fi
    log "GPU${GPU} EVAL train${TN}_${SEL} eval${EN} ($BACKBONE)"
    CUDA_VISIBLE_DEVICES=$GPU python -u -m chip_multilabel.run_stage1 \
        --model "${RUN}best_model.pth" \
        --eval-set "$EVAL_SET" --out-root "$EVAL_OUT" \
        --variants I3,I7,I10,I13 --n-per-class 99999 \
        --strength-min 0.0 --strength-max 1.0 --seed 42 \
        2>&1 | tee -a "${LOG}.gpu${GPU}" || log "EVAL_FAIL train${TN}_${SEL} eval${EN}"
}

if [ $DO_EVAL -eq 1 ]; then
    log "=== STAGE 3: parallel eval (18 cells / $NGPU GPUs) ==="
    # Round-robin assignment
    IDX=0
    PIDS=()
    for TN in 50 100 200; do
        for SEL in f1 margin_max; do
            for EN in 200 2000 20000; do
                GPU=$((IDX % NGPU))
                eval_cell $GPU $TN $SEL $EN &
                PIDS+=($!)
                IDX=$((IDX+1))
                # Throttle — wait if >= NGPU concurrent
                if [ $((IDX % NGPU)) -eq 0 ]; then
                    for pid in "${PIDS[@]}"; do wait $pid; done
                    PIDS=()
                fi
            done
        done
    done
    for pid in "${PIDS[@]}"; do wait $pid; done
    log "all evals complete"
fi

# ======================================================================
# 4. Pseudo-label retrain (Stage 5 in main pipeline)
# ======================================================================
if [ ${DO_PSEUDO:-0} -eq 1 ]; then
    log "=== STAGE 4: pseudo-label retrain ==="
    python -u -X utf8 mega_matrix/pseudo_label.py 2>&1 | tee -a "$LOG"
fi

# ======================================================================
# 5. Report
# ======================================================================
if [ $DO_REPORT -eq 1 ]; then
    log "=== STAGE 5: report ==="
    python -u -X utf8 mega_matrix/make_report.py 2>&1 | tee -a "$LOG"
fi

log "ALL DONE"
