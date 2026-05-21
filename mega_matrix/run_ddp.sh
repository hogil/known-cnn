#!/bin/bash
# SOTA-only matrix runner (multi-GPU job scheduler).
#
# This is intentionally not true torch.distributed DDP. Each training job uses
# one GPU with the exact single-model SOTA recipe, preserving batch/accum.

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

BACKBONE="convnextv2_base.fcmae_ft_in22k_in1k_384"
IMG_SIZE=384
DATA_BASE="${IMAGES_ROOT:-$PROJ_ROOT/data/images}"
if [ -z "${IMAGES_ROOT:-}" ] && [ ! -d "$DATA_BASE/classification_chips" ] && [ -d "E:/data/images/classification_chips" ]; then
    DATA_BASE="E:/data/images"
fi

OUT_BASE="outputs/_mega_matrix_sota"
TRAIN_SIZES_LIST="50 100 200 400"
EVAL_SIZES_LIST="200 2000 20000"
EPOCHS=10
BATCH=2
ACCUM=8
TRAIN_WORKERS="${TRAIN_WORKERS:-0}"
EVAL_BATCH="${EVAL_BATCH:-32}"
EVAL_WORKERS="${EVAL_WORKERS:-0}"
EPOCH_EVAL_BATCH="${EPOCH_EVAL_BATCH:-32}"
EPOCH_EVAL_CAP="${EPOCH_EVAL_CAP:-0}"
TRAIN_EPOCH_PRINT_CAP="${TRAIN_EPOCH_PRINT_CAP:-50}"
ERROR_CAP="${ERROR_CAP:-200}"
DO_DATA=1
DO_TRAIN=1
DO_EPOCH_EVAL=1
DO_EVAL=1
DO_REPORT=1
FORCE=0
EXTRA_TRAIN_FLAGS=""
GROUP_DIR=""
NGPU=""

while [ $# -gt 0 ]; do
    case "$1" in
        --skip-data) DO_DATA=0 ;;
        --skip-train) DO_TRAIN=0 ;;
        --skip-epoch-eval) DO_EPOCH_EVAL=0 ;;
        --skip-eval) DO_EVAL=0 ;;
        --skip-report) DO_REPORT=0 ;;
        --report-only) DO_DATA=0; DO_TRAIN=0; DO_EPOCH_EVAL=0; DO_EVAL=0; DO_REPORT=1 ;;
        --data-base) shift; DATA_BASE="$1" ;;
        --data-base=*) DATA_BASE="${1#--data-base=}" ;;
        --out-base) shift; OUT_BASE="$1" ;;
        --out-base=*) OUT_BASE="${1#--out-base=}" ;;
        --group-dir) shift; GROUP_DIR="$1" ;;
        --group-dir=*) GROUP_DIR="${1#--group-dir=}" ;;
        --gpus) shift; NGPU="$1" ;;
        --gpus=*) NGPU="${1#--gpus=}" ;;
        --train-sizes) shift; TRAIN_SIZES_LIST=$(echo "$1" | tr ',' ' ') ;;
        --train-sizes=*) TRAIN_SIZES_LIST=$(echo "${1#--train-sizes=}" | tr ',' ' ') ;;
        --eval-sizes) shift; EVAL_SIZES_LIST=$(echo "$1" | tr ',' ' ') ;;
        --eval-sizes=*) EVAL_SIZES_LIST=$(echo "${1#--eval-sizes=}" | tr ',' ' ') ;;
        --epochs) shift; EPOCHS="$1" ;;
        --epochs=*) EPOCHS="${1#--epochs=}" ;;
        --batch) shift; BATCH="$1" ;;
        --batch=*) BATCH="${1#--batch=}" ;;
        --accum) shift; ACCUM="$1" ;;
        --accum=*) ACCUM="${1#--accum=}" ;;
        --eval-batch) shift; EVAL_BATCH="$1" ;;
        --eval-batch=*) EVAL_BATCH="${1#--eval-batch=}" ;;
        --epoch-eval-cap) shift; EPOCH_EVAL_CAP="$1" ;;
        --epoch-eval-cap=*) EPOCH_EVAL_CAP="${1#--epoch-eval-cap=}" ;;
        --train-workers) shift; TRAIN_WORKERS="$1" ;;
        --train-workers=*) TRAIN_WORKERS="${1#--train-workers=}" ;;
        --eval-workers) shift; EVAL_WORKERS="$1" ;;
        --eval-workers=*) EVAL_WORKERS="${1#--eval-workers=}" ;;
        --error-cap) shift; ERROR_CAP="$1" ;;
        --error-cap=*) ERROR_CAP="${1#--error-cap=}" ;;
        --grad-ckpt) EXTRA_TRAIN_FLAGS="$EXTRA_TRAIN_FLAGS --grad-checkpointing" ;;
        --force) FORCE=1 ;;
        --smoke)
            TRAIN_SIZES_LIST="${SMOKE_TRAIN_SIZES:-50}"
            EVAL_SIZES_LIST="${SMOKE_EVAL_SIZES:-200}"
            EPOCHS="${SMOKE_EPOCHS:-1}"
            EPOCH_EVAL_CAP="${SMOKE_EPOCH_EVAL_CAP:-10}"
            BATCH="${SMOKE_BATCH:-2}"
            ACCUM="${SMOKE_ACCUM:-1}"
            ;;
        --help|-h)
            sed -n '1,42p' "$0"
            exit 0
            ;;
        *)
            echo "unknown arg: $1" >&2
            exit 2
            ;;
    esac
    shift
done

GPU_IDS=()
if [ -n "${CUDA_VISIBLE_DEVICES:-}" ]; then
    IFS=',' read -ra GPU_IDS <<< "$CUDA_VISIBLE_DEVICES"
else
    detected=$(nvidia-smi -L 2>/dev/null | wc -l | tr -d ' ')
    [ "$detected" -lt 1 ] && detected=1
    for ((i=0; i<detected; i++)); do GPU_IDS+=("$i"); done
fi
if [ -z "$NGPU" ]; then
    NGPU="${#GPU_IDS[@]}"
fi
[ "$NGPU" -lt 1 ] && NGPU=1
if [ "$NGPU" -gt "${#GPU_IDS[@]}" ]; then
    NGPU="${#GPU_IDS[@]}"
fi

export IMAGES_ROOT="$DATA_BASE"
export MEGA_TRAIN_SIZES=$(echo "$TRAIN_SIZES_LIST" | tr ' ' ',')
export MEGA_EVAL_SIZES=$(echo "$EVAL_SIZES_LIST" | tr ' ' ',')
export MEGA_SELS="val_f1,val_margin"

mkdir -p "$OUT_BASE"
if [ -z "$GROUP_DIR" ]; then
    if [ "$DO_DATA$DO_TRAIN$DO_EPOCH_EVAL$DO_EVAL" = "0000" ]; then
        GROUP_DIR=$(find "$OUT_BASE" -mindepth 1 -maxdepth 1 -type d -printf '%p\n' 2>/dev/null | sort | tail -1)
        if [ -z "$GROUP_DIR" ]; then
            echo "no existing group under $OUT_BASE; pass --group-dir" >&2
            exit 1
        fi
    else
        GROUP_DIR="$OUT_BASE/$(date +%Y%m%d_%H%M%S)_iter116J_sota"
    fi
fi
LOG_DIR="$GROUP_DIR/logs"
mkdir -p "$LOG_DIR"
LOG="$LOG_DIR/run_ddp.log"
: > "$LOG"
export MEGA_GROUP_DIR="$GROUP_DIR"

log() {
    echo "$(date '+%Y-%m-%d %H:%M:%S') [sota-ddp] $*" | tee -a "$LOG"
}

on_error() {
    local rc=$?
    local line="${1:-unknown}"
    echo "$(date '+%Y-%m-%d %H:%M:%S') [sota-ddp] EXIT_FAIL rc=$rc line=$line" | tee -a "$LOG"
    exit "$rc"
}
trap 'on_error $LINENO' ERR

WEIGHTS="weights/${BACKBONE}.pth"
if [ "$DO_TRAIN" -eq 1 ] && [ ! -f "$WEIGHTS" ]; then
    log "ERROR missing offline weights: $WEIGHTS"
    exit 2
fi

log "start group=$GROUP_DIR GPUs=${GPU_IDS[*]} using=$NGPU"
log "fixed recipe: single-GPU SOTA jobs only, batch=$BATCH accum=$ACCUM epochs=$EPOCHS"
log "axes: train_sizes=[$TRAIN_SIZES_LIST] eval_sizes=[$EVAL_SIZES_LIST] selections=[val_f1 val_margin]"

count_pngs() {
    find "$1" -maxdepth 1 -type f -name '*.png' 2>/dev/null | wc -l | tr -d ' '
}

latest_train_run() {
    local tn=$1
    local root="$GROUP_DIR/train_n${tn}"
    [ -d "$root" ] || return 0
    find "$root" -mindepth 1 -maxdepth 1 -type d -name '*_T7_*' -printf '%p\n' 2>/dev/null | sort | tail -1
}

eval_cap_for() {
    local en=$1
    if [ "$EPOCH_EVAL_CAP" -gt 0 ] && [ "$EPOCH_EVAL_CAP" -lt "$en" ]; then
        echo "$EPOCH_EVAL_CAP"
    else
        echo "$en"
    fi
}

eval_stage_exists() {
    local root=$1
    [ -d "$root" ] || return 1
    find "$root" -mindepth 1 -maxdepth 1 -type d \( -name 'eval_*' -o -name 'stage1_*' \) -exec test -f '{}/bit_far_metrics.json' ';' -print -quit 2>/dev/null | grep -q .
}

if [ "$DO_DATA" -eq 1 ]; then
    log "DATA generate/check train_n and eval_n"
    python -u -X utf8 mega_matrix/gen_data.py 2>&1 | tee -a "$LOG"
fi

train_job() {
    local gpu=$1
    local tn=$2
    local job_log="$LOG_DIR/train_n${tn}.gpu${gpu}.log"
    local train_root="$DATA_BASE/train_n${tn}"
    local out_root="$GROUP_DIR/train_n${tn}"
    local run
    run=$(latest_train_run "$tn" || true)
    if [ -n "$run" ] && [ -f "$run/history.json" ] && [ "$FORCE" -eq 0 ]; then
        echo "[sota-ddp] TRAIN skip train_n${tn}: $run" | tee -a "$job_log"
        return 0
    fi
    local total=0
    local parts=""
    for cls in bank_boundary fork scratch scratch_rot; do
        local n=0
        [ -d "$train_root/$cls" ] && n=$(count_pngs "$train_root/$cls")
        parts="${parts}${cls}=${n} "
        total=$((total + n))
        if [ "$n" -lt "$tn" ]; then
            echo "[sota-ddp] TRAIN_DATA_FAIL $train_root/$cls has $n, need $tn" | tee -a "$job_log"
            return 1
        fi
    done
    local mv_set=""
    for cand in 200 2000 20000; do
        if [ -d "$DATA_BASE/eval_n${cand}" ]; then mv_set="$DATA_BASE/eval_n${cand}"; break; fi
    done
    if [ -z "$mv_set" ]; then
        echo "[sota-ddp] TRAIN_DATA_FAIL no eval_n* set exists for per-epoch print" | tee -a "$job_log"
        return 1
    fi
    mkdir -p "$out_root"
    echo "[sota-ddp] GPU${gpu} TRAIN train_n${tn}: before epoch 1 train_png=$total ($parts) eval_print_set=$mv_set cap=$TRAIN_EPOCH_PRINT_CAP/class" | tee -a "$job_log"
    local stamp
    stamp=$(date +%Y%m%d_%H%M%S)
    CUDA_VISIBLE_DEVICES="$gpu" TRAIN_RUN_STAMP="$stamp" python -u -m chip_multilabel._train_chip_variant \
        --variant T7 --ls 0.30 --epochs "$EPOCHS" --batch "$BATCH" --accum "$ACCUM" --seed 1 \
        --num-workers "$TRAIN_WORKERS" \
        --lr 1e-4 --no-normal --val-criterion margin_max \
        --save-every-epoch \
        --multi-val-set "$mv_set" --multi-val-n-per-class "$TRAIN_EPOCH_PRINT_CAP" \
        --data-root "$train_root" \
        --cutmix-mode complement --cutmix-pair masked --cutmix-pair-fill corner \
        --cutmix-p 0.25 --cutmix-grid-dim 8 --cutmix-n-groups 3 --cutmix-complete-label-scale 0.5 \
        --backbone-timm "$BACKBONE" --img-size "$IMG_SIZE" \
        --backbone-timm-weights "$WEIGHTS" \
        $EXTRA_TRAIN_FLAGS \
        --out-root "$out_root" --tag "sota_train_n${tn}" \
        2>&1 | tee -a "$job_log"
}

run_parallel_train() {
    local idx=0
    local pids=()
    for tn in $TRAIN_SIZES_LIST; do
        local gpu="${GPU_IDS[$((idx % NGPU))]}"
        log "dispatch GPU${gpu} train_n${tn}"
        train_job "$gpu" "$tn" &
        pids+=($!)
        idx=$((idx + 1))
        if [ "${#pids[@]}" -ge "$NGPU" ]; then
            for pid in "${pids[@]}"; do wait "$pid"; done
            pids=()
        fi
    done
    for pid in "${pids[@]}"; do wait "$pid"; done
}

select_all() {
    for tn in $TRAIN_SIZES_LIST; do
        local run
        run=$(latest_train_run "$tn" || true)
        if [ -z "$run" ]; then
            log "SELECT_FAIL train_n${tn}: missing run"
            return 1
        fi
        local flag=""
        [ "$FORCE" -eq 1 ] && flag="--force"
        python -u -X utf8 mega_matrix/select_sota_checkpoints.py --run-dir "$run" $flag 2>&1 | tee -a "$LOG"
    done
}

epoch_job() {
    local gpu=$1
    local tn=$2
    local en=$3
    local run
    run=$(latest_train_run "$tn")
    local eval_set="$DATA_BASE/eval_n${en}"
    local out_dir="$run/epoch_curves/eval_n${en}"
    local cap
    cap=$(eval_cap_for "$en")
    local job_log="$LOG_DIR/epoch_train${tn}_eval${en}.gpu${gpu}.log"
    if [ -s "$out_dir/epoch_metrics.csv" ] && [ -s "$out_dir/epoch_metrics.png" ] && [ "$FORCE" -eq 0 ]; then
        echo "[sota-ddp] EPOCH_EVAL skip train_n${tn} eval_n${en}: $out_dir" | tee -a "$job_log"
        return 0
    fi
    echo "[sota-ddp] GPU${gpu} EPOCH_EVAL train_n${tn} eval_n${en} cap=$cap/class" | tee -a "$job_log"
    CUDA_VISIBLE_DEVICES="$gpu" python -u -X utf8 mega_matrix/eval_epoch_curve.py \
        --run-dir "$run" --eval-set "$eval_set" --out-dir "$out_dir" \
        --n-per-class "$cap" --batch-size "$EPOCH_EVAL_BATCH" --num-workers "$EVAL_WORKERS" \
        2>&1 | tee -a "$job_log"
}

eval_job() {
    local gpu=$1
    local tn=$2
    local en=$3
    local sel=$4
    local run
    run=$(latest_train_run "$tn")
    local model="$run/selected/$sel/best_model.pth"
    local eval_set="$DATA_BASE/eval_n${en}"
    local out_root="$run/selected/$sel/eval_n${en}"
    local job_log="$LOG_DIR/eval_train${tn}_eval${en}_${sel}.gpu${gpu}.log"
    if eval_stage_exists "$out_root" && [ "$FORCE" -eq 0 ]; then
        echo "[sota-ddp] EVAL skip train_n${tn} eval_n${en} $sel: $out_root" | tee -a "$job_log"
        return 0
    fi
    echo "[sota-ddp] GPU${gpu} EVAL train_n${tn} eval_n${en} $sel n_per_class=$en" | tee -a "$job_log"
    CUDA_VISIBLE_DEVICES="$gpu" python -u -m chip_multilabel.run_stage1 \
        --model "$model" \
        --eval-set "$eval_set" --out-root "$out_root" \
        --variants I10 --n-per-class "$en" \
        --batch-size "$EVAL_BATCH" --num-workers "$EVAL_WORKERS" \
        --error-cap "$ERROR_CAP" \
        --strength-min 0.0 --strength-max 1.0 --seed 42 \
        2>&1 | tee -a "$job_log"
}

run_parallel_epoch() {
    local idx=0
    local pids=()
    for tn in $TRAIN_SIZES_LIST; do
        for en in $EVAL_SIZES_LIST; do
            local gpu="${GPU_IDS[$((idx % NGPU))]}"
            epoch_job "$gpu" "$tn" "$en" &
            pids+=($!)
            idx=$((idx + 1))
            if [ "${#pids[@]}" -ge "$NGPU" ]; then
                for pid in "${pids[@]}"; do wait "$pid"; done
                pids=()
            fi
        done
    done
    for pid in "${pids[@]}"; do wait "$pid"; done
}

run_parallel_eval() {
    local idx=0
    local pids=()
    for tn in $TRAIN_SIZES_LIST; do
        for en in $EVAL_SIZES_LIST; do
            for sel in val_f1 val_margin; do
                local gpu="${GPU_IDS[$((idx % NGPU))]}"
                eval_job "$gpu" "$tn" "$en" "$sel" &
                pids+=($!)
                idx=$((idx + 1))
                if [ "${#pids[@]}" -ge "$NGPU" ]; then
                    for pid in "${pids[@]}"; do wait "$pid"; done
                    pids=()
                fi
            done
        done
    done
    for pid in "${pids[@]}"; do wait "$pid"; done
}

if [ "$DO_TRAIN" -eq 1 ]; then
    run_parallel_train
fi

if [ "$DO_TRAIN" -eq 1 ] || [ "$DO_EPOCH_EVAL" -eq 1 ] || [ "$DO_EVAL" -eq 1 ] || [ "$DO_REPORT" -eq 1 ]; then
    select_all
fi

if [ "$DO_EPOCH_EVAL" -eq 1 ]; then
    run_parallel_epoch
fi

if [ "$DO_EVAL" -eq 1 ]; then
    run_parallel_eval
fi

if [ "$DO_REPORT" -eq 1 ]; then
    log "REPORT build final summary and plots"
    python -u -X utf8 mega_matrix/make_sota_matrix_report.py \
        --group-dir "$GROUP_DIR" \
        --train-sizes "$(echo "$TRAIN_SIZES_LIST" | tr ' ' ',')" \
        --eval-sizes "$(echo "$EVAL_SIZES_LIST" | tr ' ' ',')" \
        2>&1 | tee -a "$LOG"
    test -s "$GROUP_DIR/summary_sota_matrix.md"
    test -s "$GROUP_DIR/figs_sota/selection_bit_f1_far.png"
    test -s "$GROUP_DIR/figs_sota/bit_F1_heatmap.png"
    test -s "$GROUP_DIR/figs_sota/total_far_heatmap.png"
fi

log "DONE"
echo "-> $GROUP_DIR/summary_sota_matrix.md" | tee -a "$LOG"
