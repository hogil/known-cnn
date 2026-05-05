# 02 — Method

> **APPEND-ONLY.** 이 파일은 누적 기록. 한 번 적힌 row/section 절대 삭제·수정 금지.

## 1. Data synthesis

### 1.1 Wafer-level distribution heatmap (8 종)

WM-811K cca/* 의 224×224 RGBA mask (값 = {0=outside, 128=normal, 255=defect}) 를 입력으로
8 클래스 (Center / Donut / Edge-Loc / Edge-Ring / Loc / Near-full / Random + Thick-Edge)
별로 wafer-fitting → 32×32 / 256×256 두 해상도의 P(defect | cell) heatmap 학습 (출처:
`docs/image-generation/PIPELINE.md` line 7-22).

```
acc_defect[cell] / acc_wafer[cell]   (cell P(wafer 안) ≥ 0.10 만 유효)
→ _dist_heatmaps/<Class>_p_defect_{32,256}.npy/.png    (gitignored 산출물)
```

부재 시 재생성 명령: `python dist_learn/_dist_learn.py` (출처: `CLAUDE.md` line 48).

### 1.2 Chip-internal alpha 매커니즘

각 chip 내부 200×200 픽셀에 대해 spatial alpha field α(y, x) ∈ [0, 1] 를 정의. baseline
grade dist 와 object peak dist 의 cumulative threshold 위 mix:

```
P(grade=g | chip=defect, position=(x,y)) = (1-α(x,y))·BASELINE[g] + α(x,y)·OBJECT_DIST[g]

cum_mixed = (1-α[..., None])·CUM_BASE + α[..., None]·cum_obj
u = rng.random((CHIP, CHIP))
grades = (u[..., None] < cum_mixed).argmax(axis=-1)
```

(출처: `docs/image-generation/SPEC.md` §5)

baseline (정상 chip / object 영역 밖):
```python
BASELINE = [0.83, 0.15, 0.012, 0.005, 0.002, 0.0008, 0.0001, 0.0001]   # P(0..7)
# P(0) = 0.83, P(0)+P(1) = 0.98 (정상 chip 압도적 grade 0/1)
```

5 chip-object α 함수 (출처: `docs/image-generation/SPEC.md` §6):
- **bank_boundary**: 3 vertical (x=50/100/150) + 1 horizontal (y=100) Gaussian line, σ=10
- **fork** (이전 `particle_blast`, round 26): 단일 Gaussian blob, σ ∈ [22, 35]
- **scratch**: 2-5 random vertical line, σ ∈ [3.0, 5.0], length partial
- **scratch_rot** (이전 `scratch_21deg`, round 26): 21° 회전 line, 동일 분포
- **invalid_main**: chip 전체 palette idx 31 (white) — α 적용 X

### 1.3 9 wafer-canvas pattern (`_sample_canvas_gen.py`, round 12-25)

obj-active 18 class 와 별도로 obj-less 9 class 를 chip-internal alpha 매커니즘 wafer
6400×6400 한 번에 적용 (출처: `docs/image-generation/CANVAS_9.md`):

| Class | spatial pattern | peak alpha |
|---|---|---|
| DiagonalSmear | 1 line, 45°±5°, along-fade | U(0.30, 0.50) |
| CrossScratch | 2 ⊥ line cross | U(0.30, 0.50) |
| CrescentArc | bottom-fixed 1/4-1/3 arc | U(0.30, 0.50) |
| ParallelScratches | 3-5 parallel line | U(0.30, 0.50) |
| BrokenRing | annular ring × 1-3 angular gap | U(0.30, 0.50) |
| RingDots | ring 위 14-23 angular dot | U(0.40, 0.60) |
| CenterDonut | center 얇은 ring | U(0.30, 0.50) |
| Row | direct PIL Draw line (Y-locked, 짧은 mini-line) | binary 1.0 |
| Starburst | center 빈 ring + 8-14 radial ray | center 0.60-0.85 |

핵심 alpha 분포 함수 — Lorentzian sharp + heavy tail sum (출처:
`docs/image-generation/CANVAS_9.md` §1.2):

```python
sharp = 1.0 / (1.0 + (d/(0.5σ))**2)   # 좁고 매우 sharp peak
wide  = 0.60 / (1.0 + (d/(5σ))**2)     # 크고 넓은 heavy tail
return min(sharp + wide, 1.0)
```

| d | 0 | σ | 2σ | 4σ | 8σ | 15σ | 25σ |
|---|---|---|---|---|---|---|---|
| α | 1.00 | 0.78 | 0.58 | 0.40 | 0.18 | 0.06 | 0.024 |

Gaussian 단일 / 단일 Lorentzian 모두 round 18-20 시도 후 사용자 reject (양 끝 자연 fade
실패). 두 Lorentzian sum 만이 "가운데 매우 sharp + 양 끝 자연 0 fade" 동시 만족.

### 1.4 chip border decision (round 23 strict primary filter)

```python
chip_alpha_mean = chip_alpha[200x200].mean()
chip_alpha_max  = chip_alpha[200x200].max()
if chip_alpha_mean < 0.10: continue       # primary filter (line 직접 통과 chip 만)
if chip_alpha_max  < 0.30: continue       # secondary
p_def = min(chip_alpha_mean * 3.0, 1.0)
```

기존 max-only filter 시 line 곁가지 chip 까지 BIN → 외곽 산만. **alpha mean primary
가 line 직접 통과 chip 만 BIN, 그 외 모두 normal** (출처:
`docs/image-generation/CANVAS_9.md` §1.3, `feedback_canvas_alpha_design.md`).

### 1.5 invalid 비례 fix (round 25)

기존 `n=15 fixed` invalid_inside_mask → defect 갯수의 ~15% 비례:

```python
invalid_inside_mask = select_random_invalid(rng, defect_mask, inside,
                                            n=max(2, int(defect_mask.sum() * 0.15)))
```

defect 적은 class 의 invalid 도 적게 (출처: `docs/image-generation/CANVAS_9.md` §1.4,
`_sample_gen.py`).

### 1.6 출력 위치

```
PNG : D:/project/data/wm-811k/unknown/<class>/*.png       (6400×6400 palette)
JSON: D:/project/data/positions/unknown/<class>/*.json    (chip BIN + FTN/QTN + partid/pgm)
chip crop : D:/project/data/wm-811k/classification_chips/<obj>/<basename>_x<x>_y<y>_b<bin>.png
            (5 obj label, wafer generation 시점에 inline 저장 — `_sample_gen.save_chip_crops`)
```

## 2. Architecture — 3-stage chain

```
┌──────────────────────────┐    ┌──────────────────────────┐    ┌──────────────────────────┐
│ Stage 1: chip 5-class    │    │ Stage 2: obj_id_maps     │    │ Stage 3: compound        │
│ cnn_train_chip.py        │ →  │ chip_tools/_build_obj_id │ →  │ cnn_train_compound.py    │
│ data: 200×200 chip crop  │    │ _maps.py                  │    │ R = failbit, G = obj_id  │
│ logs_chip/, 5 class      │    │ wafer 마다 32×32 obj_id  │    │ B = zero, 3-channel      │
│                           │    │ .npy cache               │    │ logs_compound/, 33-class │
│ ConvNeXtV2-base 88M       │    │ chip CNN forward inline  │    │ ConvNeXtV2-base 88M      │
└──────────────────────────┘    └──────────────────────────┘    └──────────────────────────┘
                                                                          ↑
                                  비교 base ────  ┌──────────────────────────┐
                                                  │ wafer-only (R 만)         │
                                                  │ cnn_train_wafer.py       │
                                                  │ logs_wafer/, 33-class    │
                                                  └──────────────────────────┘
```

(출처: `CLAUDE.md` line 117-160, `.claude/agents/orch-master.md`,
`.claude/agents/stage3-compound.md`)

### 2.1 백본 정책

`models/convnextv2_base.fcmae_ft_in22k_in1k_384.pth` (ImageNet FCMAE pretrained
ConvNeXtV2-base, 88M). 모든 trainer 의 init backbone. local-only — `cnn_train.py` 가
자동 로드 (출처: `CLAUDE.md` line 157-159).

### 2.2 학습 핵심 옵션 (출처: `cnn_train.py` engine + 각 wrapper config)

- **EMA**: decay 0.95, warmup 3 epoch (출처: `cnn_train_chip.py` hparams.yaml)
- **Label smoothing**: 0.02
- **bf16 AMP** (output: `amp: true`)
- **Stochastic depth**: 0.05
- **Gradient clip**: 0.5
- **class_weight**: effective (Cui et al. CVPR 2019, β=0.999)
- **Loss**: cross-entropy 기본 (focal_gamma=2.0 옵션)

### 2.3 Block expand policy (categorical resize)

obj_id (32×32 categorical) / one-hot binary / probability 등 categorical map 의 spatial
resize 는 **`_chipgrid_resize.block_expand_2d` 만 사용** (출처:
`feedback_block_expand_only.md`, `CLAUDE.md` line 273-298):

```python
# ✅ 올바른 사용
from _chipgrid_resize import block_expand_2d
obj_384 = block_expand_2d(obj_32, 384, 384)            # 정수 12 px/cell
obj_200 = block_expand_2d(obj_32, 200, 200)            # 6 px/cell + 8 cell 7px (균등 spread)

# ❌ 금지 — categorical 신호 깨짐
PIL.Image.fromarray(obj_32).resize((384, 384), Image.BICUBIC)
F.interpolate(obj_t, size=(384, 384), mode='nearest')   # 정수 배수 가정
```

이 정책이 V3 chipgrid (val_f1 0.9946) 의 enabling factor 였고, compound BICUBIC 384
ceiling 0.9784 대비 +75% error 감소를 만든 단일 변경이다.

## 3. Optimizer / LR (★ AD reference, round 28 update)

### 3.1 round 28 LR 변경 사항

`D:/project/anomaly-detection/train.py` line 1497-1530 (참조: AD repo) 의 spec 을
known-cnn 의 3 trainer 모두 적용 (출처: 사용자 round 28 명시 + AD `docs/summary.md`):

| 항목 | 이전 (round 27 이하) | 새 (round 28) | 이유 |
|---|---|---|---|
| `lr_backbone` | 1e-5 | **2e-5** | AD 검증값, fine-tune 더 적극적 |
| `lr_head` | 1e-3 | **2e-4** | head 도 backbone 과 비례 (10×) — gradient spike 회피 |
| `warmup_epochs` | 2 | **5** | 충분한 warmup 으로 초기 LR spike 완화 |
| `LinearLR start_factor` | 0.1 | **0.05** | 매우 낮은 시작값. AD seed 4 spike 사례 fix |
| Main scheduler | CosineAnnealingLR (동일) | CosineAnnealingLR `eta_min=1e-6` | 동일 유지 |

AD `train.py:1503-1506` 의 주석:
> `# warmup start_factor 0.05 — 매우 낮은 시작값으로 gradient spike 방지`
> `# 기존 0.1은 초기 LR이 너무 높아 seed 4 같은 경우에 ep 4-8에서 spike 발생`

### 3.2 새 LR spec (3 trainer 모두 동일)

```python
optimizer = torch.optim.AdamW([
    {"params": backbone_params, "lr": 2e-5},
    {"params": head_params,     "lr": 2e-4},
], weight_decay=0.05)

warmup = LinearLR(optimizer, start_factor=0.05, total_iters=5)
cosine = CosineAnnealingLR(optimizer, T_max=epochs - 5, eta_min=1e-6)
scheduler = SequentialLR(optimizer, schedulers=[warmup, cosine], milestones=[5])
```

### 3.3 적용 trainer

- `cnn_train_chip.py` (Stage 1)
- `cnn_train_wafer.py` (R-only base)
- `cnn_train_compound.py` (Stage 3)

(출처: round 28 사용자 명시 — `cnn-master` agent dispatch 시 자동 적용 spec)

## 4. Multi-label ablation (8-stage paper-style)

본 single-label trained 모델을 multi-label 추론에 활용하는 8 stage ablation (출처:
`docs/multi-label/STAGES.md`, `docs/multi-label/README.md`).

| Stage | 역할 | Status |
|---|---|---|
| 1. 분포 학습 | 5 method × 33 class × 5 data-amount surface sweep | ✅ 완료 (`_dist_learn_per_class.py`, 850 npy + 37 plot, commit 687448b) |
| 2. Hyperparameter | class_weight / label_smoothing / loss 영향 측정 | ⏳ TODO |
| 3. `unknown_multi/` 합성 | 1000-3000 wafer 에 known multi-label GT | ⏳ TODO |
| 4. 추론 path 비교 | Phase A heuristic / Phase B AdaGC / Phase C BCE/ASL | ⏳ TODO |
| 5. Threshold tuning | per-class F1 sweep + Temp + IDF + top-K floor mix | ⏳ TODO |
| 6. Chip-Wafer matching | heatmap + GMM ensemble + CRF | ⏳ TODO |
| 7. Prod predict 보강 | best 조합을 `cnn_predict_compound_prod.py` 통합 | ⏳ TODO |
| 8. Master comparison | paper-style table + figure | ⏳ TODO |

### 4.1 ★ 사용자 우선순위 3 영역 (출처: `feedback_multi_label_priority.md`)

> "loss 부분과 chip class 로 wafer class matching 하는 부분이 이론과 여러 기법 mix 등
>  굉장히 중요해 보인다 관건이다. 그리고 multi-label 판정 방식도."

1. **Loss design** (deep-dive `docs/multi-label/LOSS_DESIGN.md`) — M1-M7 mix combinations
   (CE / Focal / BCE / ASL / AdaGC + label_smoothing + class_weight)
2. **Chip-Wafer matching** (deep-dive `docs/multi-label/MATCHING_DESIGN.md`) — C1-C7
   (heatmap / GMM / KDE / hybrid + CRF + Mahalanobis + consistency)
3. **Multi-label decision rule** (deep-dive `docs/multi-label/DECISION_RULE.md`) —
   D1-D8 (per-class F1 + Temp + IDF + top-K floor)

단일 기법 SOTA 비교 X. **mix 조합 sweep 이 본 ablation 의 진짜 contribution**.

## 5. Active class policy

### 5.1 33 → 20 active + 14 archive (round 11 결정)

V3 chipgrid (val_f1 0.9946) 의 saturated 분류 결과 기반 (출처:
`docs/wafer-ensemble/ACTIVE_CLASSES.md`, `feedback_active_class_policy.md`):

| Group | Count | Classes |
|---|---|---|
| Donut × 5 obj | 5 | Donut_{bank_boundary, invalid_main, fork, scratch, scratch_rot} |
| Edge-Bottom × 5 obj ★ weak | 5 | Edge-Bottom_{...} |
| Edge-Ring × 4 obj | 4 | Edge-Ring_{bank_boundary, fork, scratch, scratch_rot} (-invalid_main) |
| Edge-Top × 5 obj ★ weak | 5 | Edge-Top_{...} |
| 특수 | 1 | Thick-Edge_invalid_main |
| **Total active 20** | **20** | (`experiments/active_classes_20.yaml`) |

### 5.2 active 27 (canvas 9 추가)

active 20 + 9 wafer-canvas - 2 archive (Center_invalid_main, Full_invalid_main) =
**27** (`experiments/active_classes_27.yaml`).

### 5.3 active 30 (target)

8 obj-less wafer-canvas 새 class 합성 후 사용 target (현재 9 canvas 합성 완료 →
`experiments/active_classes_30.yaml` = `configs/chipgrid_class30_target.yaml`).

### 5.4 현재 (round 28)

`unknown/` 안 모든 class 사용 — 43 active class (Normal_bank_boundary 1000 sample,
다른 42 class 200 sample). 이는 `--active-classes-yaml` 미지정 시 default behavior.

archive 14 class 데이터 = `D:/project/data/wm-811k/unknown_archive/<class>/` copy 보존
(원본 unknown/ 도 그대로 — 무단 삭제 금지).

## Cross-link

- 데이터 합성 spec → `docs/image-generation/{SPEC,PIPELINE,CANVAS_9}.md`
- 3-stage chain → `.claude/agents/stage3-compound.md`, `CLAUDE.md` Quickstart
- block_expand 정책 → `feedback_block_expand_only.md`
- LR AD reference → `D:/project/anomaly-detection/train.py:1497-1530`,
  `D:/project/anomaly-detection/docs/summary.md`
- multi-label → `docs/multi-label/{README,STAGES,LOSS_DESIGN,MATCHING_DESIGN,DECISION_RULE}.md`
- active class → `docs/wafer-ensemble/ACTIVE_CLASSES.md`
