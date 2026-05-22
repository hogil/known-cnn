#!/bin/bash
# Single-model SOTA runner: iter116J-style FCM-PM only.
#
# This intentionally does not run the mega-matrix axes. It trains one model,
# evaluates I10 only, and writes a compact bit_F1/FAR report.
#
# Defaults:
#   train: E:/data/images/classification_chips
#   eval : E:/data/images/chip_multilabel_v15direct_n2000
#
# Usage:
#   CUDA_VISIBLE_DEVICES=0 bash mega_matrix/run_single_sota.sh
#   bash mega_matrix/run_single_sota.sh --frozen-sota
#   bash mega_matrix/run_single_sota.sh --eval-set E:/data/images/chip_multilabel_v15direct_n2000
#   bash mega_matrix/run_single_sota.sh --skip-train --run-dir outputs/iter116J_exact_repro/<run>

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
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

BACKBONE="convnextv2_base.fcmae_ft_in22k_in1k_384"
IMG_SIZE=384
DATA_BASE="${IMAGES_ROOT:-$PROJ_ROOT/data/images}"
if [ -z "$IMAGES_ROOT" ] && [ ! -d "$DATA_BASE/classification_chips" ] && [ -d "E:/data/images/classification_chips" ]; then
    DATA_BASE="E:/data/images"
fi

TRAIN_ROOT="${DATA_BASE}/classification_chips"
EVAL_SET="${DATA_BASE}/chip_multilabel_v15direct_n2000"
EVAL_N_PER_CLASS=2000
OUT_ROOT="outputs/iter116J_exact_repro"
EPOCHS=10
BATCH=2
ACCUM=8
TRAIN_WORKERS="${TRAIN_WORKERS:-0}"
EVAL_BATCH="${EVAL_BATCH:-32}"
EVAL_WORKERS="${EVAL_WORKERS:-0}"
DO_DATA=1
DO_TRAIN=1
DO_EVAL=1
DO_REPORT=1
RUN_DIR=""
FROZEN_SOTA_RUN="outputs/iter116J_g3_ls30/T7_iter116J_g3_ls30_260513_010015"

while [ $# -gt 0 ]; do
    case "$1" in
        --frozen-sota) DO_TRAIN=0; RUN_DIR="$FROZEN_SOTA_RUN" ;;
        --train-root) shift; TRAIN_ROOT="$1" ;;
        --train-root=*) TRAIN_ROOT="${1#--train-root=}" ;;
        --eval-set) shift; EVAL_SET="$1" ;;
        --eval-set=*) EVAL_SET="${1#--eval-set=}" ;;
        --eval-n-per-class) shift; EVAL_N_PER_CLASS="$1" ;;
        --eval-n-per-class=*) EVAL_N_PER_CLASS="${1#--eval-n-per-class=}" ;;
        --out-root) shift; OUT_ROOT="$1" ;;
        --out-root=*) OUT_ROOT="${1#--out-root=}" ;;
        --epochs) shift; EPOCHS="$1" ;;
        --epochs=*) EPOCHS="${1#--epochs=}" ;;
        --batch) shift; BATCH="$1" ;;
        --batch=*) BATCH="${1#--batch=}" ;;
        --accum) shift; ACCUM="$1" ;;
        --accum=*) ACCUM="${1#--accum=}" ;;
        --eval-batch) shift; EVAL_BATCH="$1" ;;
        --eval-batch=*) EVAL_BATCH="${1#--eval-batch=}" ;;
        --skip-data) DO_DATA=0 ;;
        --skip-train) DO_TRAIN=0 ;;
        --skip-eval) DO_EVAL=0 ;;
        --skip-report) DO_REPORT=0 ;;
        --run-dir) shift; RUN_DIR="$1" ;;
        --run-dir=*) RUN_DIR="${1#--run-dir=}" ;;
        --help|-h)
            head -32 "$0" | tail -28
            exit 0
            ;;
        *)
            echo "unknown arg: $1" >&2
            exit 2
            ;;
    esac
    shift
done

LOG_DIR="$OUT_ROOT/logs"
mkdir -p "$OUT_ROOT" "$LOG_DIR"
LOG="$LOG_DIR/run_single_sota_$(date +%Y%m%d_%H%M%S).log"

log() {
    echo "$(date '+%Y-%m-%d %H:%M:%S') [single-sota] $*" | tee -a "$LOG"
}

on_error() {
    local rc=$?
    local line="${1:-unknown}"
    echo "$(date '+%Y-%m-%d %H:%M:%S') [single-sota] EXIT_FAIL rc=$rc line=$line" | tee -a "$LOG"
    exit "$rc"
}
trap 'on_error $LINENO' ERR

WEIGHTS="weights/${BACKBONE}.pth"
if [ $DO_TRAIN -eq 1 ] && [ ! -f "$WEIGHTS" ]; then
    log "ERROR missing offline weights: $WEIGHTS"
    exit 2
fi

if [ $DO_DATA -eq 1 ]; then
    log "data generation skipped: exact iter116J uses existing classification_chips and v15direct_n2000"
fi

for cls in bank_boundary fork scratch scratch_rot; do
    n=$(find "$TRAIN_ROOT/$cls" -maxdepth 1 -name "*.png" 2>/dev/null | wc -l | tr -d ' ')
    if [ "$n" -le 0 ]; then
        log "TRAIN_DATA_FAIL $TRAIN_ROOT/$cls has $n PNGs"
        exit 1
    fi
done
if [ ! -d "$EVAL_SET" ]; then
    log "EVAL_DATA_FAIL missing eval set: $EVAL_SET"
    exit 1
fi

train_total=$(find "$TRAIN_ROOT" -mindepth 2 -name "*.png" 2>/dev/null | wc -l | tr -d ' ')
eval_total=$(find "$EVAL_SET" -mindepth 2 -name "*.png" 2>/dev/null | wc -l | tr -d ' ')
log "SOTA recipe fixed: iter116J FCM-PM g=3 LS=0.30 pair=masked, val_margin, I10 only"
log "train_root=$TRAIN_ROOT train_png=$train_total (--no-normal; only 4 defect dirs used)"
log "eval_set=$EVAL_SET eval_png=$eval_total n_per_class_cap=$EVAL_N_PER_CLASS"
log "epochs=$EPOCHS batch=$BATCH accum=$ACCUM workers(train/eval)=$TRAIN_WORKERS/$EVAL_WORKERS cuda_visible=$CUDA_VISIBLE_DEVICES"

if [ $DO_TRAIN -eq 1 ]; then
    stamp=$(date +%Y%m%d_%H%M%S)
    TRAIN_RUN_STAMP="$stamp" python -u -m chip_multilabel._train_chip_variant \
        --variant T7 --ls 0.30 --epochs "$EPOCHS" --batch "$BATCH" --accum "$ACCUM" --seed 1 \
        --num-workers "$TRAIN_WORKERS" \
        --lr 1e-4 --no-normal --val-criterion margin_max \
        --multi-val-set "$EVAL_SET" --multi-val-n-per-class 50 \
        --data-root "$TRAIN_ROOT" \
        --cutmix-mode complement --cutmix-pair masked --cutmix-pair-fill corner \
        --cutmix-p 0.25 --cutmix-grid-dim 8 --cutmix-n-groups 3 --cutmix-complete-label-scale 0.5 \
        --backbone-timm "$BACKBONE" --img-size "$IMG_SIZE" \
        --backbone-timm-weights "$WEIGHTS" \
        --out-root "$OUT_ROOT" --tag "iter116J_exact" \
        2>&1 | tee -a "$LOG"
fi

if [ -z "$RUN_DIR" ]; then
    RUN_DIR=$(find "$OUT_ROOT" -mindepth 1 -maxdepth 1 -type d -name "*iter116J_exact*" -printf '%p\n' 2>/dev/null | sort -r | head -1)
fi
if [ -z "$RUN_DIR" ] || [ ! -f "$RUN_DIR/best_model.pth" ]; then
    log "RUN_DIR_FAIL no best_model.pth found. pass --run-dir explicitly."
    exit 1
fi
log "run_dir=$RUN_DIR"

if [ $DO_EVAL -eq 1 ]; then
    EVAL_OUT="$RUN_DIR/eval_sota_i10"
    if [ -d "$EVAL_OUT" ]; then
        log "eval exists; skip: $EVAL_OUT"
    else
        python -u -m chip_multilabel.run_stage1 \
            --model "$RUN_DIR/best_model.pth" \
            --eval-set "$EVAL_SET" --out-root "$EVAL_OUT" \
            --variants I10 --n-per-class "$EVAL_N_PER_CLASS" \
            --batch-size "$EVAL_BATCH" --num-workers "$EVAL_WORKERS" \
            --strength-min 0.0 --strength-max 1.0 --seed 42 \
            2>&1 | tee -a "$LOG"
    fi
fi

if [ $DO_REPORT -eq 1 ]; then
    python -u -X utf8 mega_matrix/make_single_sota_report.py \
        --run-dir "$RUN_DIR" --eval-name eval_sota_i10 \
        2>&1 | tee -a "$LOG"
fi

log "DONE"
echo "-> $RUN_DIR/single_sota_summary.md" | tee -a "$LOG"
echo "-> $RUN_DIR/single_sota_bit_far.png" | tee -a "$LOG"
