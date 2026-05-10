#!/bin/bash
# 260509 — iter34: DINOv3-ConvNeXt backbone upgrade (Meta 2025 Nov)
# Eval both base (88M, fair-comp) + small (28M, production candidate).
# Chains AFTER iter33 KD sweep done.
set -e
cd /d/project/known-cnn

WAIT_LOG=outputs/_iter33_KD_sweep.log
RUN_LOG=outputs/_iter34_dinov3.log

V15="D:/project/data/wm-811k/chip_multilabel_v15direct"

echo "$(date) [iter34] waiting for iter33 to finish ..." > "$RUN_LOG"
until grep -q "\[iter33\] DONE" "$WAIT_LOG" 2>/dev/null; do
    sleep 30
done
echo "$(date) [iter34] iter33 done — start DINOv3 backbone eval" >> "$RUN_LOG"

# Step 0: download DINOv3 weights via HuggingFace cache (one-shot)
# - facebook/dinov3-convnext-base-pretrain-lvd1689m  (88M, fair vs ConvNeXtV2-base)
# - facebook/dinov3-convnext-small-pretrain-lvd1689m (28M, production candidate)
# trainer 의 build_model() 이 timm registry 확인 필요. ConvNeXtV2-base 와 같은 timm 호출 패턴
# (init_ckpt local path 또는 HuggingFace hub) 으로 가능한 만큼 fallback.

run_one() {
    TAG=$1
    BACKBONE=$2
    INIT_CKPT=$3
    OUT_ROOT="outputs/iter34${TAG}"
    echo "$(date) [iter34-${TAG}] backbone=${BACKBONE}  init=${INIT_CKPT}" >> "$RUN_LOG"
    set +e
    python -m chip_multilabel._train_chip_variant \
        --variant T7 --ls 0.20 --epochs 8 --batch 2 --accum 8 --seed 1 \
        --cutmix-p 0.25 --cutmix-mode complement --cutmix-n-groups 3 \
        --cutmix-complete-label-scale 0.50 --cutmix-pair masked --cutmix-pair-fill corner \
        --backbone "$BACKBONE" --init-ckpt "$INIT_CKPT" \
        --out-root "$OUT_ROOT" \
        --tag "iter34${TAG}_seed1" \
        >> "$RUN_LOG" 2>&1
    RUN=$(ls -d "$OUT_ROOT"/T*/ 2>/dev/null | head -1)
    if [ -z "$RUN" ]; then echo "$(date) [iter34-${TAG}] FAILED" >> "$RUN_LOG"; return 0; fi
    python -m chip_multilabel.run_stage1 --model "${RUN}best_model.pth" \
        --eval-set "$V15" \
        --out-root "${RUN}eval_v15direct" --variants I3,I6,I7,I10 \
        --n-per-class 50 --strength-min 0.0 --seed 42 >> "$RUN_LOG" 2>&1 || true
    echo "$(date) [iter34-${TAG}] DONE" >> "$RUN_LOG"
    set -e
}

# A. DINOv3-ConvNeXt-base (88M, fair architectural comparison vs ConvNeXtV2-base)
#    ★ NOTE: trainer 가 --backbone flag + HuggingFace ckpt path 지원해야 함.
#    현재 trainer 는 timm 의 ConvNeXtV2 만 build. dinov3 변종 add 필요 (별도 patch).
#    placeholder — patch 후 실행
run_one A_dinov3_base "convnextv2_base.dinov3_lvd1689m" "facebook/dinov3-convnext-base-pretrain-lvd1689m"

# B. DINOv3-ConvNeXt-small (28M, production candidate — 3× faster inference)
run_one B_dinov3_small "convnextv2_small.dinov3_lvd1689m" "facebook/dinov3-convnext-small-pretrain-lvd1689m"

echo "$(date) [iter34] DONE 2/2" >> "$RUN_LOG"
