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
#   bash mega_matrix/run_ddp.sh --data-base data/images
#
# NOTE: vanilla PyTorch single-GPU training per cell (NOT torch.distributed DDP).
#       For true torchrun DDP inside one cell, trainer code needs modification
#       (DistributedSampler, init_process_group, etc.) — out of scope here.
#       This script uses "cell-level data parallelism" — different cells on
#       different GPUs simultaneously, single GPU per cell.
# =====================================================================
set -e
set -E
set -o pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJ_ROOT="$(dirname "$SCRIPT_DIR")"
cd "$PROJ_ROOT"
export PYTHONPATH="$PROJ_ROOT${PYTHONPATH:+:$PYTHONPATH}"

export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export HF_HUB_DISABLE_TELEMETRY=1
if [ -z "${CUDA_VISIBLE_DEVICES:-}" ] && [ -n "${CUDA_VISIBLE_DEVICE:-}" ]; then
    export CUDA_VISIBLE_DEVICES="$CUDA_VISIBLE_DEVICE"
fi
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

OUT_BASE=outputs/_mega_matrix
BACKBONE="convnextv2_base.fcmae_ft_in22k_in1k_384"
# 260520 — renamed env IMAGES_ROOT (was WM811K_ROOT) and default path data/images.
DATA_BASE="${IMAGES_ROOT:-$PROJ_ROOT/data/images}"
if [ -z "$IMAGES_ROOT" ] && [ ! -d "$DATA_BASE/classification_chips" ] && [ -d "E:/data/images/classification_chips" ]; then
    DATA_BASE="E:/data/images"
fi

# Default flags
DO_DATA=1; DO_EVAL=1; DO_REPORT=1; DO_PSEUDO=0
# CUDA_VISIBLE_DEVICES has higher priority than nvidia-smi -L (which ignores CVD)
if [ -n "$CUDA_VISIBLE_DEVICES" ]; then
    NGPU=$(echo "$CUDA_VISIBLE_DEVICES" | tr ',' '\n' | grep -c '[0-9]')
else
    NGPU=$(nvidia-smi -L 2>/dev/null | wc -l)
fi
[ "$NGPU" -lt 1 ] && NGPU=1
SMOKE=0
EPOCHS=30
TRAIN_SIZES_LIST="50 100 200 400"
SEL_LIST="f1 margin_max"
EVAL_SIZES_LIST="200 2000 20000"

while [ $# -gt 0 ]; do
    case $1 in
        --skip-data) DO_DATA=0 ;;
        --skip-eval) DO_EVAL=0 ;;
        --with-pseudo) DO_PSEUDO=1 ;;
        --skip-pseudo) DO_PSEUDO=0 ;;
        --skip-report) DO_REPORT=0 ;;
        --data-base) shift; DATA_BASE="$1" ;;
        --data-base=*) DATA_BASE="${1#--data-base=}" ;;
        --gpus) shift; NGPU=$1 ;;
        --gpus=*) NGPU="${1#--gpus=}" ;;
        --backbone) shift; BACKBONE="$1" ;;
        --backbone=*) BACKBONE="${1#--backbone=}" ;;
        --eval-batch) shift; EVAL_BATCH_OVERRIDE="$1" ;;
        --eval-batch=*) EVAL_BATCH_OVERRIDE="${1#--eval-batch=}" ;;
        # 260521 — custom sweep axes (comma or space separated)
        --train-sizes) shift; TRAIN_SIZES_LIST=$(echo "$1" | tr ',' ' ') ;;
        --train-sizes=*) TRAIN_SIZES_LIST=$(echo "${1#--train-sizes=}" | tr ',' ' ') ;;
        --eval-sizes) shift; EVAL_SIZES_LIST=$(echo "$1" | tr ',' ' ') ;;
        --eval-sizes=*) EVAL_SIZES_LIST=$(echo "${1#--eval-sizes=}" | tr ',' ' ') ;;
        --sels) shift; SEL_LIST=$(echo "$1" | tr ',' ' ') ;;
        --sels=*) SEL_LIST=$(echo "${1#--sels=}" | tr ',' ' ') ;;
        --epochs) shift; EPOCHS="$1" ;;
        --epochs=*) EPOCHS="${1#--epochs=}" ;;
        --smoke) SMOKE=1 ;;
        --help|-h) head -25 "$0" | tail -20; exit 0 ;;
    esac
    shift
done

# Propagate to gen_data.py + make_report.py so they generate / aggregate only the requested axes
export MEGA_TRAIN_SIZES=$(echo "$TRAIN_SIZES_LIST" | tr ' ' ',')
export MEGA_EVAL_SIZES=$(echo "$EVAL_SIZES_LIST" | tr ' ' ',')
export MEGA_SELS=$(echo "$SEL_LIST" | tr ' ' ',')

# Derive input img-size from backbone name pattern
case "$BACKBONE" in
    *384*) IMG_SIZE=384 ;;
    *256*) IMG_SIZE=256 ;;
    *)     IMG_SIZE=224 ;;
esac

# Per-rank batch for H100 80GB + small chip PNGs. Sized to ~60 GB / 80 GB
# (~75% utilization) for SERVER (mega_matrix exception — runs on server
# with no baseline GPU contention; local 30-40GB rule does NOT apply).
# Per-rank, NOT divided by world_size.
# Effective global batch = BATCH_PER_GPU * world_size.
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
    *)                      BATCH_PER_GPU=12 ;;   # safe fallback
esac

# Smoke mode override: 1 cell × 1 epoch × tiny data × micro batch
if [ $SMOKE -eq 1 ]; then
    BATCH_PER_GPU=2
    EPOCHS=1
    TRAIN_SIZES_LIST="10"
    SEL_LIST="f1 margin_max"
    EVAL_SIZES_LIST="200"
    DO_PSEUDO=0
    DO_REPORT=0
    export MEGA_TRAIN_SIZES="10"
    export MEGA_EVAL_SIZES="200"
    export MEGA_SELS="f1,margin_max"
fi

# DataLoader workers: use full logical CPU count per rank.
TOTAL_CPU=$(nproc 2>/dev/null || echo 8)
WORKERS_PER_RANK="$TOTAL_CPU"
EVAL_BATCH_SIZE="${EVAL_BATCH_OVERRIDE:-${EVAL_BATCH_SIZE:-$BATCH_PER_GPU}}"
EVAL_WORKERS="${EVAL_WORKERS:-0}"
SAVE_EVERY_EPOCH="${SAVE_EVERY_EPOCH:-0}"
SAVE_EVERY_EPOCH_FLAG=""
if [ "$SAVE_EVERY_EPOCH" = "1" ]; then
    SAVE_EVERY_EPOCH_FLAG="--save-every-epoch"
fi

GPU_IDS=()
if [ -n "${CUDA_VISIBLE_DEVICES:-}" ]; then
    IFS=',' read -ra GPU_IDS <<< "$CUDA_VISIBLE_DEVICES"
    if [ "$NGPU" -gt "${#GPU_IDS[@]}" ]; then
        NGPU="${#GPU_IDS[@]}"
    fi
else
    for ((i=0; i<NGPU; i++)); do
        GPU_IDS+=("$i")
    done
fi
[ "$NGPU" -lt 1 ] && NGPU=1

normalize_sel_list() {
    local out=""
    for raw in "$@"; do
        local sel="$raw"
        case "$sel" in
            val_f1) sel="f1" ;;
            margin|val_margin) sel="margin_max" ;;
        esac
        if [[ " $out " != *" $sel "* ]]; then
            out="${out}${out:+ }${sel}"
        fi
    done
    echo "$out"
}
SEL_LIST="$(normalize_sel_list $SEL_LIST)"
if [ "${MEGA_ALLOW_SINGLE_SELECTION:-0}" != "1" ]; then
    [[ " $SEL_LIST " == *" f1 "* ]] || SEL_LIST="f1 ${SEL_LIST}"
    [[ " $SEL_LIST " == *" margin_max "* ]] || SEL_LIST="${SEL_LIST} margin_max"
fi
export MEGA_SELS=$(echo "$SEL_LIST" | tr ' ' ',')

RUN_STAMP=$(date +%Y%m%d_%H%M%S)
LOG_BACKBONE="${BACKBONE//\//_}"
LOG_BACKBONE="${LOG_BACKBONE//:/_}"
LOG_BACKBONE="${LOG_BACKBONE// /_}"
# GROUP folder = single per-run namespace with TS prefix (same convention as run.sh)
GROUP_DIR="$OUT_BASE/${RUN_STAMP}_${LOG_BACKBONE}"
MODEL_BASE="$GROUP_DIR"
LOG_DIR="$GROUP_DIR/logs"
LOG="$LOG_DIR/run_ddp.log"
mkdir -p "$OUT_BASE" "$GROUP_DIR" "$LOG_DIR"
export MEGA_GROUP_DIR="$GROUP_DIR"
export MEGA_MODEL_BASE="$MODEL_BASE"

log() {
    echo "$(date '+%Y-%m-%d %H:%M:%S') [ddp] $*" | tee -a "$LOG"
}

: > "$LOG"
on_error() {
    local rc=$?
    local line="${1:-unknown}"
    echo "$(date '+%Y-%m-%d %H:%M:%S') [ddp] EXIT_FAIL rc=$rc line=$line" | tee -a "$LOG"
    exit "$rc"
}
trap 'on_error $LINENO' ERR
log "start backbone=$BACKBONE img=$IMG_SIZE N_GPU=$NGPU batch_per_gpu=$BATCH_PER_GPU eval_batch=$EVAL_BATCH_SIZE accum=1 workers_per_rank=$WORKERS_PER_RANK eval_workers=$EVAL_WORKERS save_every_epoch=$SAVE_EVERY_EPOCH data_base=$DATA_BASE data=$DO_DATA eval=$DO_EVAL pseudo=$DO_PSEUDO report=$DO_REPORT"

# Offline weights (closed-network) - .pth only.
# Only required for stages that init a fresh timm backbone (train, pseudo-label).
# data / eval / report don't need it (eval loads best_model.pth from disk).
BACKBONE_WEIGHTS_FLAG=""
OFFLINE_WEIGHTS="weights/${BACKBONE}.pth"
NEEDS_WEIGHTS=0
# run_ddp.sh always runs Stage 2 (train) unconditionally - no --skip-train flag exists.
# So weights are needed whenever this script is invoked normally (train) or pseudo.
if [ ${DO_PSEUDO:-0} -eq 1 ]; then
    NEEDS_WEIGHTS=1
fi
# Detect a "no-train" mode if user passed --skip-data --skip-eval --skip-report
# but otherwise treat as needing weights (default behavior).
if [ $DO_DATA -eq 1 ] || [ $DO_EVAL -eq 1 ] || [ $DO_REPORT -eq 1 ]; then
    # If all three are 1 and pseudo is 0 we still run training; play safe and require.
    NEEDS_WEIGHTS=1
fi
if [ -f "$OFFLINE_WEIGHTS" ]; then
    BACKBONE_WEIGHTS_FLAG="--backbone-timm-weights $OFFLINE_WEIGHTS"
    log "offline weights: $OFFLINE_WEIGHTS"
elif [ $NEEDS_WEIGHTS -eq 1 ]; then
    log "ERROR missing offline weights for $BACKBONE under weights/"
    log "closed-network mode forbids HF/timm download; create weights/${BACKBONE}.pth first"
    exit 2
else
    log "SKIP weight check (no stage requires fresh backbone)"
fi
export BACKBONE_WEIGHTS_FLAG BACKBONE IMG_SIZE MODEL_BASE
export MEGA_MODEL_BASE="$MODEL_BASE"
export MEGA_BACKBONE="$BACKBONE"
export MEGA_IMG_SIZE="$IMG_SIZE"
export IMAGES_ROOT="$DATA_BASE"

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
validate_train_data() {
    local TRAIN_ROOT=$1
    local TN=$2
    local EXPECTED_CLASSES=(bank_boundary fork scratch scratch_rot)

    if [ ! -d "$TRAIN_ROOT" ]; then
        log "TRAIN_DATA_FAIL missing train root: $TRAIN_ROOT"
        return 1
    fi

    local unexpected=()
    local d cls
    for d in "$TRAIN_ROOT"/*; do
        [ -d "$d" ] || continue
        cls=$(basename "$d")
        case "$cls" in
            bank_boundary|fork|scratch|scratch_rot|_*) ;;
            *) unexpected+=("$cls") ;;
        esac
    done
    if [ ${#unexpected[@]} -gt 0 ]; then
        log "TRAIN_DATA_FAIL unexpected class dirs under $TRAIN_ROOT: ${unexpected[*]}"
        log "  -> expected only: ${EXPECTED_CLASSES[*]}"
        return 1
    fi

    local missing=()
    local total=0
    local breakdown=""
    local n
    for cls in "${EXPECTED_CLASSES[@]}"; do
        if [ -d "$TRAIN_ROOT/$cls" ]; then
            n=$(find "$TRAIN_ROOT/$cls" -maxdepth 1 -name "*.png" 2>/dev/null | wc -l | tr -d ' ')
        else
            n=0
        fi
        total=$((total + n))
        breakdown="${breakdown}${cls}=${n} "
        if [ "$n" -lt "$TN" ]; then
            missing+=("$cls=$n/$TN")
        fi
    done
    if [ ${#missing[@]} -gt 0 ]; then
        log "TRAIN_DATA_FAIL insufficient class data under $TRAIN_ROOT: ${missing[*]}"
        log "  -> class_counts: ${breakdown}"
        return 1
    fi

    TRAIN_DATA_TOTAL=$total
    TRAIN_DATA_CLASS_N=${#EXPECTED_CLASSES[@]}
    TRAIN_DATA_CLASS_BREAK="$breakdown"
    return 0
}

# 6 cells:
#   (50, f1)         (50, margin_max)
#   (100, f1)        (100, margin_max)
#   (200, f1)        (200, margin_max)

train_cell() {
    local TN=$1; local SEL=$2
    local TAG="train${TN}_${SEL}"
    local OUT_ROOT="${MODEL_BASE}/${TAG}"
    if [ -d "$OUT_ROOT" ]; then
        log "SKIP train ${TAG}: exists $OUT_ROOT"
        return 0
    fi
    log "TRAIN ${TAG} ($BACKBONE) NGPU=$NGPU batch_per_gpu=$BATCH_PER_GPU workers_per_rank=$WORKERS_PER_RANK"
    local TRAIN_ROOT="${DATA_BASE}/train_n${TN}"
    validate_train_data "$TRAIN_ROOT" "$TN" || return 1
    log "  -> train_data: ${TRAIN_ROOT}"
    log "  -> chips=${TRAIN_DATA_TOTAL} (${TRAIN_DATA_CLASS_BREAK}) across ${TRAIN_DATA_CLASS_N} class; 0.9/0.1 train/val split"
    # True torchrun DDP: each rank sees --batch as its OWN batch (per-rank).
    # effective global batch = BATCH_PER_GPU * NGPU
    set +e
    local TRAIN_STAMP=$(date +%Y%m%d_%H%M%S)
    MULTI_VAL_FLAG=""
    # 260520 - bit_F1 / FAR every epoch (CLAUDE.md absolute rule 260512).
    for cand in 200 2000 100 50 30 20 10 5; do
        if [ -d "${DATA_BASE}/eval_n${cand}" ]; then
            npc=$cand
            [ "$npc" -gt 50 ] && npc=50
            MULTI_VAL_FLAG="--multi-val-set ${DATA_BASE}/eval_n${cand} --multi-val-n-per-class ${npc}"
            break
        fi
    done
    TRAIN_RUN_STAMP="$TRAIN_STAMP" torchrun --standalone --nproc_per_node="$NGPU" \
        -m chip_multilabel._train_chip_variant \
        --variant T7 --ls 0.30 --epochs $EPOCHS \
        --batch "$BATCH_PER_GPU" --accum 1 --seed 1 \
        --num-workers "$WORKERS_PER_RANK" \
        --lr 1e-4 --no-normal --val-criterion ${SEL} \
        $SAVE_EVERY_EPOCH_FLAG \
        $MULTI_VAL_FLAG \
        --data-root "${DATA_BASE}/train_n${TN}" \
        --cutmix-mode complement --cutmix-pair masked --cutmix-pair-fill corner \
        --cutmix-p 0.25 --cutmix-grid-dim 8 --cutmix-n-groups 3 --cutmix-complete-label-scale 0.5 \
        --backbone-timm "$BACKBONE" --img-size $IMG_SIZE \
        $BACKBONE_WEIGHTS_FLAG \
        --out-root "$OUT_ROOT" --tag "${TAG}" \
        2>&1 | tee -a "${LOG}.train"
    local TRAIN_RC=${PIPESTATUS[0]}
    set -e
    local RUN=$(find "$OUT_ROOT" -mindepth 2 -maxdepth 2 -name best_model.pth -printf '%h\n' 2>/dev/null | sort -r | head -1)
    [ -n "$RUN" ] && rm -f "$RUN"/epoch_*.pth
    if [ "$TRAIN_RC" -ne 0 ]; then
        log "TRAIN_FAIL ${TAG} rc=$TRAIN_RC"
        return 0
    fi
    log "DONE ${TAG}"
}

log "=== STAGE 2: sequential torchrun DDP ($NGPU GPUs each) ==="
# True DDP: each train_cell uses ALL $NGPU GPUs via torchrun. Cells run
# sequentially (since they share GPUs). Effective global batch per cell =
# BATCH_PER_GPU * NGPU.
for TN in $TRAIN_SIZES_LIST; do
    for SEL in $SEL_LIST; do
        train_cell "$TN" "$SEL"
    done
done
log "all trainings complete"

# ======================================================================
# 3. Eval — distribute 18 cells across $NGPU GPUs
# ======================================================================
eval_cell() {
    local GPU=$1; local TN=$2; local SEL=$3; local EN=$4
    local MODEL_ROOT=$(find "$MODEL_BASE" -mindepth 1 -maxdepth 1 -type d \( -name "train${TN}_${SEL}" -o -name "*_model_train${TN}_${SEL}" \) -printf '%p\n' 2>/dev/null | sort -r | head -1)
    local RUN=$(find "$MODEL_ROOT" -mindepth 2 -maxdepth 2 -name best_model.pth -printf '%h\n' 2>/dev/null | sort -r | head -1)
    if [ -z "$RUN" ]; then
        log "SKIP eval train${TN}_${SEL} eval${EN}: no trained run under $MODEL_ROOT"
        return 0
    fi
    local EVAL_OUT="${RUN}/eval_${EN}"
    if [ -d "$EVAL_OUT" ]; then
        log "SKIP eval train${TN}_${SEL} eval${EN}: exists $EVAL_OUT"
        return 0
    fi
    local EVAL_SET="${DATA_BASE}/eval_n${EN}"
    if [ ! -d "$EVAL_SET" ]; then
        log "SKIP eval train${TN}_${SEL} eval${EN}: missing $EVAL_SET"
        return 0
    fi
    log "GPU${GPU} EVAL train${TN}_${SEL} eval${EN} ($BACKBONE) batch=${EVAL_BATCH_SIZE} workers=${EVAL_WORKERS}"
    CUDA_VISIBLE_DEVICES=$GPU python -u -m chip_multilabel.run_stage1 \
        --model "${RUN}/best_model.pth" \
        --eval-set "$EVAL_SET" --out-root "$EVAL_OUT" \
        --variants I3,I7,I10,I13 --n-per-class 99999 \
        --batch-size "$EVAL_BATCH_SIZE" --num-workers "$EVAL_WORKERS" \
        --strength-min 0.0 --strength-max 1.0 --seed 42 \
        2>&1 | tee -a "${LOG}.gpu${GPU}" || log "EVAL_FAIL train${TN}_${SEL} eval${EN}"
}

if [ $DO_EVAL -eq 1 ]; then
    log "=== STAGE 3: parallel eval ($NGPU GPUs) ==="
    # Round-robin assignment
    IDX=0
    PIDS=()
    for TN in $TRAIN_SIZES_LIST; do
        for SEL in $SEL_LIST; do
            for EN in $EVAL_SIZES_LIST; do
                GPU_SLOT=$((IDX % NGPU))
                GPU="${GPU_IDS[$GPU_SLOT]}"
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
    REPORT_PATH="${GROUP_DIR}/summary_mega_sweep.md"
    if [ ! -s "$REPORT_PATH" ]; then
        log "REPORT_FAIL missing report: $REPORT_PATH"
        exit 1
    fi
    REQUIRED_PLOTS=(
        bit_F1_heatmap.png
        total_far_heatmap.png
        ni_far_heatmap.png
        ood_far_heatmap.png
        scaling_curves.png
        combined_bit_far_by_sel.png
        bit_F1_by_sel.png
        far_by_sel.png
    )
    if [[ " $SEL_LIST " == *" f1 "* && " $SEL_LIST " == *" margin_max "* ]]; then
        REQUIRED_PLOTS=(best_model_eval_by_selection.png "${REQUIRED_PLOTS[@]}")
    fi
    for plot in "${REQUIRED_PLOTS[@]}"; do
        PLOT_PATH="${GROUP_DIR}/figs_mega/${plot}"
        if [ ! -s "$PLOT_PATH" ]; then
            log "REPORT_FAIL missing plot: $PLOT_PATH"
            exit 1
        fi
    done
fi

log "ALL DONE"
echo "-> ${GROUP_DIR}/summary_mega_sweep.md" | tee -a "$LOG"
