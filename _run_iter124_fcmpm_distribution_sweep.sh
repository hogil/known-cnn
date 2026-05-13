#!/bin/bash
# iter124 — 4-single-source FCM-PM distribution separation sweep.
#
# Absolute rule:
#   - train source remains 4 single defect classes only via classification_chips
#   - no Normal/Invalid/OOD/real combo/synthetic combo roots are used for training
#   - FCM-PM/CutMix pseudo-combo is generated on the fly from 4 single chips
#
# Primary target:
#   maximize single+2-combo bit-F1 while improving val2 bit-wise margins,
#   especially fork/scratch active lower tails.
set -e
cd /d/project/known-cnn

if [ "${1:-}" != "--run-full-sweep" ]; then
    echo "Refusing to run full sweep by default. Use PowerShell smoke:"
    echo "  ./_run_iter124_fcmpm_distribution_sweep.ps1 -Smoke"
    echo "Full sweep requires explicit approval and:"
    echo "  bash _run_iter124_fcmpm_distribution_sweep.sh --run-full-sweep"
    exit 2
fi
shift

RUN_LOG=outputs/_iter124_fcmpm_distribution_sweep.log
V15="D:/project/data/wm-811k/chip_multilabel_v15direct"
BACKBONE="convnextv2_base.fcmae_ft_in22k_in1k_384"
WEIGHTS="mega_matrix/weights/${BACKBONE}.pth"
WEIGHTS_ARG=""
[ -f "$WEIGHTS" ] && WEIGHTS_ARG="--backbone-timm-weights $WEIGHTS"

echo "$(date) [iter124] start 4-single-source FCM-PM distribution sweep" > "$RUN_LOG"
mkdir -p outputs

train_eval() {
    TAG=$1
    shift
    OUT_ROOT="outputs/iter124${TAG}"
    if [ -d "$OUT_ROOT" ]; then
        echo "$(date) [iter124-${TAG}] skip exists" >> "$RUN_LOG"
        return 0
    fi

    echo "$(date) [iter124-${TAG}] $*" >> "$RUN_LOG"
    set +e
    python -u -m chip_multilabel._train_chip_variant \
        --batch 2 --accum 8 --seed 1 \
        --lr 1e-4 --no-normal --val-criterion margin_max \
        --backbone-timm "$BACKBONE" --img-size 384 \
        $WEIGHTS_ARG \
        --out-root "$OUT_ROOT" --tag "iter124${TAG}" \
        "$@" >> "$RUN_LOG" 2>&1
    TRAIN_RC=$?
    set -e
    if [ "$TRAIN_RC" -ne 0 ]; then
        echo "$(date) [iter124-${TAG}] TRAIN_FAIL rc=$TRAIN_RC" >> "$RUN_LOG"
        return 0
    fi

    RUN=$(ls -d "$OUT_ROOT"/T*/ 2>/dev/null | head -1)
    if [ -z "$RUN" ]; then
        echo "$(date) [iter124-${TAG}] FAIL no run dir" >> "$RUN_LOG"
        return 0
    fi

    for CK in best_model final_epoch_model; do
        CKPT="${RUN}${CK}.pth"
        [ -f "$CKPT" ] || continue
        OUT_EVAL="${RUN}eval_v15direct_n200_${CK}"
        if [ -d "$OUT_EVAL" ]; then
            continue
        fi
        python -u -m chip_multilabel.run_stage1 --model "$CKPT" \
            --eval-set "$V15" --out-root "$OUT_EVAL" \
            --variants I3,I7,I10,I13 --n-per-class 200 \
            --strength-min 0.0 --strength-max 1.0 --seed 42 \
            >> "$RUN_LOG" 2>&1 || echo "$(date) [iter124-${TAG}] EVAL_FAIL ${CK}" >> "$RUN_LOG"
    done
    echo "$(date) [iter124-${TAG}] DONE" >> "$RUN_LOG"
}

# A. FCM-PM signal strength. Baseline neighborhood, one axis at a time.
train_eval A_p015 --variant T7 --ls 0.20 --epochs 10 \
    --cutmix-mode complement --cutmix-pair masked --cutmix-pair-fill corner \
    --cutmix-p 0.15 --cutmix-n-groups 3 --cutmix-complete-label-scale 0.5

train_eval B_p025_base --variant T7 --ls 0.20 --epochs 10 \
    --cutmix-mode complement --cutmix-pair masked --cutmix-pair-fill corner \
    --cutmix-p 0.25 --cutmix-n-groups 3 --cutmix-complete-label-scale 0.5

train_eval C_p035 --variant T7 --ls 0.20 --epochs 10 \
    --cutmix-mode complement --cutmix-pair masked --cutmix-pair-fill corner \
    --cutmix-p 0.35 --cutmix-n-groups 3 --cutmix-complete-label-scale 0.5

train_eval D_p040 --variant T7 --ls 0.20 --epochs 10 \
    --cutmix-mode complement --cutmix-pair masked --cutmix-pair-fill corner \
    --cutmix-p 0.40 --cutmix-n-groups 3 --cutmix-complete-label-scale 0.5

train_eval E_g2 --variant T7 --ls 0.20 --epochs 10 \
    --cutmix-mode complement --cutmix-pair masked --cutmix-pair-fill corner \
    --cutmix-p 0.25 --cutmix-n-groups 2 --cutmix-complete-label-scale 0.5

train_eval F_g4 --variant T7 --ls 0.20 --epochs 10 \
    --cutmix-mode complement --cutmix-pair masked --cutmix-pair-fill corner \
    --cutmix-p 0.25 --cutmix-n-groups 4 --cutmix-complete-label-scale 0.5

train_eval G_cls07 --variant T7 --ls 0.20 --epochs 10 \
    --cutmix-mode complement --cutmix-pair masked --cutmix-pair-fill corner \
    --cutmix-p 0.25 --cutmix-n-groups 3 --cutmix-complete-label-scale 0.7

train_eval H_cls10 --variant T7 --ls 0.20 --epochs 10 \
    --cutmix-mode complement --cutmix-pair masked --cutmix-pair-fill corner \
    --cutmix-p 0.25 --cutmix-n-groups 3 --cutmix-complete-label-scale 1.0

train_eval I_ab1008 --variant T7 --ls 0.20 --epochs 10 \
    --cutmix-mode complement --cutmix-pair masked --cutmix-pair-fill corner \
    --cutmix-p 0.25 --cutmix-n-groups 3 --cutmix-complete-label-scale 0.5 \
    --cutmix-ab-labels "1.0,0.8"

train_eval J_ab1010 --variant T7 --ls 0.20 --epochs 10 \
    --cutmix-mode complement --cutmix-pair masked --cutmix-pair-fill corner \
    --cutmix-p 0.25 --cutmix-n-groups 3 --cutmix-complete-label-scale 0.5 \
    --cutmix-ab-labels "1.0,1.0"

# B. Weak-pair focus: fork+scratch lower-tail boost.
train_eval K_bias_fs2 --variant T7 --ls 0.20 --epochs 10 \
    --cutmix-mode complement --cutmix-pair masked --cutmix-pair-fill corner \
    --cutmix-p 0.25 --cutmix-n-groups 3 --cutmix-complete-label-scale 0.5 \
    --cutmix-pair-bias "fork,scratch:2"

train_eval L_bias_fs3 --variant T7 --ls 0.20 --epochs 10 \
    --cutmix-mode complement --cutmix-pair masked --cutmix-pair-fill corner \
    --cutmix-p 0.25 --cutmix-n-groups 3 --cutmix-complete-label-scale 0.5 \
    --cutmix-pair-bias "fork,scratch:3"

# C. Spatial coherence alternatives.
train_eval M_bisect_h --variant T7 --ls 0.20 --epochs 10 \
    --cutmix-mode bisect_h --cutmix-pair masked --cutmix-pair-fill corner \
    --cutmix-p 0.25 --cutmix-complete-label-scale 0.5

train_eval N_bisect_v --variant T7 --ls 0.20 --epochs 10 \
    --cutmix-mode bisect_v --cutmix-pair masked --cutmix-pair-fill corner \
    --cutmix-p 0.25 --cutmix-complete-label-scale 0.5

train_eval O_bisect_rand --variant T7 --ls 0.20 --epochs 10 \
    --cutmix-mode bisect_rand --cutmix-pair masked --cutmix-pair-fill corner \
    --cutmix-p 0.25 --cutmix-complete-label-scale 0.5

# D. Calibration/regularization checks.
train_eval P_ls10 --variant T7 --ls 0.10 --epochs 10 \
    --cutmix-mode complement --cutmix-pair masked --cutmix-pair-fill corner \
    --cutmix-p 0.25 --cutmix-n-groups 3 --cutmix-complete-label-scale 0.5

train_eval Q_ls30 --variant T7 --ls 0.30 --epochs 10 \
    --cutmix-mode complement --cutmix-pair masked --cutmix-pair-fill corner \
    --cutmix-p 0.25 --cutmix-n-groups 3 --cutmix-complete-label-scale 0.5

train_eval R_dp003 --variant T7 --ls 0.20 --epochs 10 --drop-path-rate 0.03 \
    --cutmix-mode complement --cutmix-pair masked --cutmix-pair-fill corner \
    --cutmix-p 0.25 --cutmix-n-groups 3 --cutmix-complete-label-scale 0.5

train_eval S_dp005 --variant T7 --ls 0.20 --epochs 10 --drop-path-rate 0.05 \
    --cutmix-mode complement --cutmix-pair masked --cutmix-pair-fill corner \
    --cutmix-p 0.25 --cutmix-n-groups 3 --cutmix-complete-label-scale 0.5

train_eval T_pos125 --variant T7 --ls 0.20 --epochs 10 \
    --pos-weight "fork:1.25,scratch:1.25" \
    --cutmix-mode complement --cutmix-pair masked --cutmix-pair-fill corner \
    --cutmix-p 0.25 --cutmix-n-groups 3 --cutmix-complete-label-scale 0.5

train_eval U_pos150 --variant T7 --ls 0.20 --epochs 10 \
    --pos-weight "fork:1.5,scratch:1.5" \
    --cutmix-mode complement --cutmix-pair masked --cutmix-pair-fill corner \
    --cutmix-p 0.25 --cutmix-n-groups 3 --cutmix-complete-label-scale 0.5

train_eval V_t9_focal --variant T9 --ls 0.00 --epochs 10 \
    --cutmix-mode complement --cutmix-pair masked --cutmix-pair-fill corner \
    --cutmix-p 0.25 --cutmix-n-groups 3 --cutmix-complete-label-scale 0.5

# Refresh absolute-rule table and val2-margin audit for completed iter124 cells.
python -X utf8 _reeval_absolute_rule.py >> "$RUN_LOG" 2>&1 || true

MODELS=outputs/_iter124_val2_margin_models.txt
: > "$MODELS"
for RUN in outputs/iter124*/T*/ ; do
    [ -d "$RUN" ] || continue
    TAG=$(basename "$(dirname "$RUN")")
    [ -f "${RUN}best_model.pth" ] && echo "${TAG}_best=${RUN}best_model.pth" >> "$MODELS"
    [ -f "${RUN}final_epoch_model.pth" ] && echo "${TAG}_final=${RUN}final_epoch_model.pth" >> "$MODELS"
done

if [ -s "$MODELS" ]; then
    python -X utf8 _run_iter123_val2_margin_audit.py \
        --models-file "$MODELS" \
        --eval-set "$V15" \
        --n-per-class 200 \
        --strength-min 0.0 --strength-max 1.0 \
        --seed 42 --val-ratio 0.2 \
        --q-low 0.05 --q-high 0.95 \
        --alphas 0.50,0.65,0.80 \
        --out-prefix outputs/_iter124_val2_margin \
        >> "$RUN_LOG" 2>&1 || true
fi

echo "$(date) [iter124] DONE" >> "$RUN_LOG"
