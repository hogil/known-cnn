---
name: wafer-classifier
description: V3 chipgrid (1.16M ChipGridCNN, 32×32×5 one-hot obj only) wafer 분류기 학습/평가/추론. obj-only 4-layer / R-only ConvNeXt / compound 3ch 비교. block_expand 정책, TTA 금지 정책, active class YAML 정책 + 표 정책 (class별 sample / epoch best/total / val_f1 / val_p / val_r / val_err / test_f1 / test_acc / param / 학습 시간) 모두 enforced.
---

# wafer-classifier — V3 chipgrid + comparators

V3 best (val_f1 0.9946) 가 oracle ceiling 추월한 wafer 33-class (현재 active 20)
분류기 학습/평가/추론 패턴.

자세한 발견 / 결과 / 정책은 `D:/project/known-cnn/docs/wafer-ensemble/` 인덱스
참조. 본 스킬은 **명령어 + 표 정책 + 정책 enforce** 가 목적.

## ★ Fair-eval protocol (모든 비교 시 강제)

**모든 backbone 비교는 동일 조건**:
- active class (immediate): `experiments/active_classes_22.yaml` (22 class, 8 obj-less 미합성)
- active class (target): `experiments/active_classes_30.yaml` = `configs/chipgrid_class30_target.yaml`
- per-class sample: 200 (sorted file pick, deterministic)
- split: 0.8 / 0.1 / 0.1 stratified, seed 42
  - 22 class: n_train 3520 / val 440 / test 440
  - 30 class: n_train 4800 / val 600 / test 600
- epoch: 30 (early stop 끔, best val_f1 epoch model selection)
- optimizer: AdamW wd 0.05, cosine warmup 3ep
- batch/lr: small (≤2M) 64/1e-3, large (>10M) 16 / head 1e-3 backbone 1e-4
- augmentation: rotate ±15°, translate/scale ±3°, gaussian σ=0.01 (no flip/colorjitter/mixup/cutmix)
- TTA: 절대 금지

위반 시 결과 비교 불가 — 별 표 분리.

spec yaml: `experiments/fair_eval_protocol.yaml`
설명: `docs/wafer-ensemble/FAIR_EVAL_PROTOCOL.md`

## 핵심 명령

### V3 chipgrid (★ best, default)

```bash
# Active 20 (default 권장)
python cnn_eval_chipgrid.py --variant V3 --no-r-channel \
    --active-classes-yaml experiments/active_classes_20.yaml \
    --n-per-class 220 --epochs 30 --seed 42 \
    --model-tag v3_active20

# 5-seed 평균 (통계 유의성 검증)
for s in 42 1 7 100 234; do
    python cnn_eval_chipgrid.py --variant V3 --no-r-channel \
        --active-classes-yaml experiments/active_classes_20.yaml \
        --n-per-class 220 --epochs 30 --seed $s \
        --model-tag v3_seed${s}
done

# chip CNN noise robustness
python cnn_eval_chipgrid.py --variant V3 --no-r-channel \
    --active-classes-yaml experiments/active_classes_20.yaml \
    --chip-noise 0.10 --chip-noise-eval --model-tag v3_noise10
```

출력: `logs_chipgrid/<tag>_<TS>_<test_f1>_<val_f1>/best_model.pth`.

### obj-only 4-layer (비교 base)

```bash
python cnn_train_objonly.py --epochs 30 --batch 32 \
    --active-classes-yaml experiments/active_classes_20.yaml \
    --train-val-only --model-tag objonly_active20
```

출력: `logs_objid_ablation/<tag>_<TS>/best_model.pth`.

### R-only ConvNeXt (대형 비교 base)

```bash
python cnn_train_wafer.py --epochs 10 --batch 16 \
    --active-classes-yaml experiments/active_classes_20.yaml \
    --model-tag wafer_active20
```

12.5h. 출력: `logs_wafer/<tag>_<TS>/best_model.pth`.

### compound 3ch (R+G+B BICUBIC, ceiling 비교)

```bash
python cnn_train_compound.py --epochs 30 --batch 16 \
    --g-channel-mode onehot \
    --active-classes-yaml experiments/active_classes_20.yaml \
    --model-tag compound_active20
```

출력: `logs_compound/<tag>_<TS>/best_model.pth`.

### V3 fair eval (different split 비교)

```bash
# V3 ckpt 를 our 0.8/0.2 val 1420 위 fair eval (val_f1 0.9951)
python _v3_fair_eval.py
# 출력: results_v3_eval/v3_logits.npy, val_y.npy, info.json
```

### intra-distribution diagnostic

```bash
# distribution / object 분리 정확도 측정 (Edge-Bottom/Top weak point)
python _intra_dist_eval.py
# 출력: results_intra_dist/summary.json
```

## 표 정책 (모든 결과 보고 시)

새 학습 / 비교 결과 보고 시 **반드시** 다음 컬럼 포함:

| Model | input | encoding | params | n train | epoch (best/total) | val_f1 | val_p | val_r | val_err | test_f1 | test_acc | 학습 시간 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|

- `n train` = 학습 데이터 sample 수 (train split, n/class * n_class)
- `epoch (best/total)` = best epoch / total trained (early stop ep 포함, e.g. "6 / 13 (es)")
- `val_p`, `val_r` = val precision, recall (macro)
- `val_err` = val error count (P != Y 합)
- `test_f1`, `test_acc` = test split 의 macro F1, accuracy
- `params` = total trainable parameters (M 단위, e.g. 1.16M, 88M)
- `학습 시간` = wallclock (e.g. "2 min", "12.5h", "<1 min")

class별 sample 분포 보고 시 `cnn_eval_chipgrid.py` 의 stratified split (0.8/0.1/0.1)
또는 `cnn_train_objonly.py` 의 (0.8/0.2) 명시. seed 도 포함.

`docs/wafer-ensemble/RESULTS.md` 의 표 형식 그대로 따르기.

## 정책 enforce (절대 위반 금지)

### Block expand 정책
categorical map (obj_id, one-hot, prob) 의 spatial resize **`_chipgrid_resize.block_expand_2d` 만 사용**.
PIL/torch BICUBIC, NEAREST hardcode 금지.

```python
# ✅ 올바른 사용
from _chipgrid_resize import block_expand_2d
obj_384 = block_expand_2d(obj_32, 384, 384)

# ❌ 금지
PIL.Image.fromarray(obj_32).resize((384, 384), Image.BICUBIC)  # categorical 깨짐
F.interpolate(obj_t, size=(384, 384), mode='nearest')          # 정수 배수 가정
```

자세한 건 `~/.claude/projects/D--project-known-cnn/memory/feedback_block_expand_only.md`.

### TTA 절대 금지
wafer class identity 가 angle/위치에 묶여있음 (scratch_21deg, Edge-Top/Bottom).
rotation/flip TTA 시 다른 class 답 ensemble.

```python
# ❌ 금지
- rotation 4 ver (0/90/180/270) ensemble
- HFlip / VFlip / 180° ensemble
- multi-scale TTA, multi-crop TTA
```

자세한 건 `~/.claude/projects/D--project-known-cnn/memory/feedback_no_tta_wafer.md`.

### Active class 정책
- 학습 시 class subset 결정은 **active-class YAML** 만 사용
- `experiments/active_classes_20.yaml` (20 active), `experiments/archive_classes_14.yaml` (참조)
- 데이터 폴더 무단 삭제 금지 — `unknown_archive/` 로 copy 만
- `EXCLUDE_CLASSES` 같은 hardcoded list 에 새 class 추가 금지

자세한 건 `~/.claude/projects/D--project-known-cnn/memory/feedback_active_class_policy.md`,
`docs/wafer-ensemble/ACTIVE_CLASSES.md`.

### 학습 결과 폴더 절대 삭제 금지
`logs_chipgrid/`, `logs_objid_ablation/`, `logs_wafer/`, `logs_compound/` 등 어떤
training 결과도 사용자 명시 요청 없이 삭제 금지. 글로벌 룰.

## 학습 augmentation 정책 (도메인-safe only)

cnn_train.py / cnn_train_compound.py / cnn_train_objonly.py / cnn_eval_chipgrid.py
모두 동일:

- ✅ ±15° rotation: 검사장비 stage 회전 오차 범위 내
- ✅ 작은 translate/scale (±3%): alignment / magnification variability
- ✅ Gaussian noise σ=0.01: sensor pixel noise
- ❌ HFlip: scratch_21deg 등 angle = 클래스 정체성 (21° → -21°)
- ❌ VFlip / 180° rotation: Edge-Top ↔ Edge-Bottom 클래스 뒤집힘
- ❌ ColorJitter: palette grade 의미 손상
- ❌ MixUp / CutMix / Cutout: palette pixel 평균이 무의미한 grade 생성

## V3 best 결과 (참조)

| 메트릭 | 값 |
|---|---|
| val_f1 | **0.9946** |
| test_f1 | 0.9872 |
| val_f1 (our 0.8/0.2 fair eval) | 0.9951 (errors 7/1420) |
| params | 1.16M |
| best_ep / total | 6 / 13 (early stop) |
| 학습 시간 | <1 min |
| chip noise 10% val_f1 | 0.9870 |
| ckpt | `logs_chipgrid/v3_full_260503_160436_0.99_0.99/best_model.pth` |

비교: oracle ceiling (R+obj 2-stream) val_f1 0.9919. **V3 단독이 oracle 추월**.

## intra-distribution weak point

V3 의 진짜 ceiling = Edge-Bottom 0.9907, Edge-Top 0.9954 (chip 6 개 안 obj 식별).
다른 6 distribution 모두 ≥ 0.9954. 자세한 건 `INTRA_DIST.md`.

## 사용자 in-progress (touch X — read-only)

- `_chipgrid_kde_gmm.py` — Phase A: KDE/GMM sanity
- `cnn_train_chipgrid_fusion.py` — Phase B: V3 + KDE/GMM fusion
- `configs/chipgrid_class30_target.yaml` — Phase C: 8 새 wafer-canvas class target

자세한 건 `STATUS.md`.

## Cross-link

- 발견 → `D:/project/known-cnn/docs/wafer-ensemble/DISCOVERY.md`
- 결과 → `D:/project/known-cnn/docs/wafer-ensemble/RESULTS.md`
- intra-distribution → `D:/project/known-cnn/docs/wafer-ensemble/INTRA_DIST.md`
- active class → `D:/project/known-cnn/docs/wafer-ensemble/ACTIVE_CLASSES.md`
- 진행 + Phase A/B/C → `D:/project/known-cnn/docs/wafer-ensemble/STATUS.md`
- chipgrid V0~V6 sweep → `D:/project/known-cnn/docs/chipgrid/RESULTS.md`
- block_expand 정책 → `~/.claude/projects/D--project-known-cnn/memory/feedback_block_expand_only.md`
- active class 정책 → `~/.claude/projects/D--project-known-cnn/memory/feedback_active_class_policy.md`
- TTA 금지 → `~/.claude/projects/D--project-known-cnn/memory/feedback_no_tta_wafer.md`
- V3 best memory → `~/.claude/projects/D--project-known-cnn/memory/project_v3_chipgrid_best.md`
