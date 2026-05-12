#!/bin/bash
# =====================================================================
# Mega matrix sweep — DDP-style parallel runner (server multi-GPU)
#
# Distributes 6 train cells across N GPUs (default 4):
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
#
# NOTE: vanilla PyTorch single-GPU training per cell (NOT torch.distributed DDP).
#       For true torchrun DDP inside one cell, trainer code needs modification
#       (DistributedSampler, init_process_group, etc.) — out of scope here.
#       This script uses "cell-level data parallelism" — different cells on
#       different GPUs simultaneously, single GPU per cell.
# =====================================================================
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJ_ROOT="$(dirname "$SCRIPT_DIR")"
cd "$PROJ_ROOT"

OUT_BASE=outputs/_mega_matrix
LOG=$OUT_BASE/_run_ddp.log
mkdir -p "$OUT_BASE"

# Default flags
DO_DATA=1; DO_EVAL=1; DO_REPORT=1
NGPU=$(nvidia-smi -L 2>/dev/null | wc -l)
[ "$NGPU" -lt 1 ] && NGPU=1

for arg in "$@"; do
    case $arg in
        --skip-data) DO_DATA=0 ;;
        --skip-eval) DO_EVAL=0 ;;
        --skip-report) DO_REPORT=0 ;;
        --gpus) shift; NGPU=$1 ;;
        --gpus=*) NGPU="${arg#--gpus=}" ;;
        --help|-h) head -25 "$0" | tail -20; exit 0 ;;
    esac
done

echo "$(date) [ddp] start N_GPU=$NGPU" > "$LOG"

# ======================================================================
# 1. Data generation (sequential, single CPU)
# ======================================================================
if [ $DO_DATA -eq 1 ]; then
    echo "$(date) [ddp] === STAGE 1: data generation ===" >> "$LOG"
    python -u -X utf8 mega_matrix/gen_data.py >> "$LOG" 2>&1
fi

# ======================================================================
# 2. Training — distribute 6 cells across $NGPU GPUs
# ======================================================================
# 6 cells:
#   (50, f1)         (50, margin_max)
#   (100, f1)        (100, margin_max)
#   (200, f1)        (200, margin_max)

train_cell() {
    local GPU=$1; local TN=$2; local SEL=$3
    local TAG="train${TN}_${SEL}"
    local OUT_ROOT="${OUT_BASE}/model_${TAG}"
    if [ -d "$OUT_ROOT" ]; then
        echo "$(date) [ddp GPU${GPU}] skip ${TAG}" >> "$LOG"
        return 0
    fi
    echo "$(date) [ddp GPU${GPU}] TRAIN ${TAG}" >> "$LOG"
    CUDA_VISIBLE_DEVICES=$GPU python -u -m chip_multilabel._train_chip_variant \
        --variant T7 --ls 0.30 --epochs 10 --batch 2 --accum 8 --seed 1 \
        --lr 1e-4 --no-normal --val-criterion ${SEL} --save-every-epoch \
        --data-root "${OUT_BASE}/train_n${TN}" \
        --cutmix-mode complement --cutmix-pair masked --cutmix-pair-fill corner \
        --cutmix-p 0.25 --cutmix-n-groups 3 --cutmix-complete-label-scale 0.5 \
        --backbone-timm "convnextv2_base.fcmae_ft_in22k_in1k_384" --img-size 384 \
        --out-root "$OUT_ROOT" --tag "${TAG}" \
        >> "${LOG}.gpu${GPU}" 2>&1
    local RUN=$(ls -d "$OUT_ROOT"/T*/ 2>/dev/null | head -1)
    [ -n "$RUN" ] && rm -f "$RUN"epoch_*.pth
    echo "$(date) [ddp GPU${GPU}] DONE ${TAG}" >> "$LOG"
}

echo "$(date) [ddp] === STAGE 2: parallel training (6 cells / $NGPU GPUs) ===" >> "$LOG"

# Distribution strategy
# Cells (T=train_n, S=selection):
#   c1=(50, f1)    c2=(50, margin)   c3=(100, f1)
#   c4=(100, margin) c5=(200, f1)    c6=(200, margin)
PIDS=()
if [ "$NGPU" -ge 6 ]; then
    # 1 cell per GPU
    train_cell 0  50  f1         & PIDS+=($!)
    train_cell 1  50  margin_max & PIDS+=($!)
    train_cell 2  100 f1         & PIDS+=($!)
    train_cell 3  100 margin_max & PIDS+=($!)
    train_cell 4  200 f1         & PIDS+=($!)
    train_cell 5  200 margin_max & PIDS+=($!)
elif [ "$NGPU" -ge 4 ]; then
    # 1-2 cells/GPU on 4 GPUs
    ( train_cell 0 50  f1; train_cell 0 50  margin_max ) & PIDS+=($!)
    ( train_cell 1 100 f1; train_cell 1 100 margin_max ) & PIDS+=($!)
    train_cell 2 200 f1         & PIDS+=($!)
    train_cell 3 200 margin_max & PIDS+=($!)
elif [ "$NGPU" -ge 2 ]; then
    # 3 cells/GPU on 2 GPUs
    ( train_cell 0 50  f1; train_cell 0 100 f1; train_cell 0 200 f1 )            & PIDS+=($!)
    ( train_cell 1 50  margin_max; train_cell 1 100 margin_max; train_cell 1 200 margin_max ) & PIDS+=($!)
else
    # Single GPU sequential
    for TN in 50 100 200; do
        for SEL in f1 margin_max; do
            train_cell 0 $TN $SEL
        done
    done
fi

# Wait for all parallel trains
for pid in "${PIDS[@]}"; do
    wait $pid || echo "$(date) [ddp] train pid $pid failed" >> "$LOG"
done
echo "$(date) [ddp] all trainings complete" >> "$LOG"

# ======================================================================
# 3. Eval — distribute 18 cells across $NGPU GPUs
# ======================================================================
eval_cell() {
    local GPU=$1; local TN=$2; local SEL=$3; local EN=$4
    local MODEL_ROOT="${OUT_BASE}/model_train${TN}_${SEL}"
    local RUN=$(ls -d "$MODEL_ROOT"/T*/ 2>/dev/null | head -1)
    [ -z "$RUN" ] && return 0
    local EVAL_OUT="${RUN}eval_${EN}"
    [ -d "$EVAL_OUT" ] && return 0
    local EVAL_SET="${OUT_BASE}/eval_n${EN}"
    [ ! -d "$EVAL_SET" ] && return 0
    echo "$(date) [ddp GPU${GPU}] EVAL train${TN}_${SEL} eval${EN}" >> "$LOG"
    CUDA_VISIBLE_DEVICES=$GPU python -u -m chip_multilabel.run_stage1 \
        --model "${RUN}best_model.pth" \
        --eval-set "$EVAL_SET" --out-root "$EVAL_OUT" \
        --variants I3,I7,I10,I13 --n-per-class 99999 \
        --strength-min 0.0 --strength-max 1.0 --seed 42 \
        >> "${LOG}.gpu${GPU}" 2>&1 || true
}

if [ $DO_EVAL -eq 1 ]; then
    echo "$(date) [ddp] === STAGE 3: parallel eval (18 cells / $NGPU GPUs) ===" >> "$LOG"
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
    echo "$(date) [ddp] all evals complete" >> "$LOG"
fi

# ======================================================================
# 4. Pseudo-label retrain (Stage 5 in main pipeline)
# ======================================================================
if [ ${DO_PSEUDO:-1} -eq 1 ]; then
    echo "$(date) [ddp] === STAGE 4: pseudo-label retrain ===" >> "$LOG"
    python -u -X utf8 mega_matrix/pseudo_label.py >> "$LOG" 2>&1
fi

# ======================================================================
# 5. Report
# ======================================================================
if [ $DO_REPORT -eq 1 ]; then
    echo "$(date) [ddp] === STAGE 5: report ===" >> "$LOG"
    python -u -X utf8 mega_matrix/make_report.py >> "$LOG" 2>&1
fi

echo "$(date) [ddp] ALL DONE" >> "$LOG"
