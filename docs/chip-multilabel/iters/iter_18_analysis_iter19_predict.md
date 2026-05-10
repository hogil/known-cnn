# Iter 18 analysis + Iter 19 prediction (260508)

> Standalone analysis: iter18 6-cell soft-label + grid_complete sweep robust 진단,
> per-class cap diagnosis, iter19 complement CutMix sweep prior prediction, next
> atomic 변경 1 개. iter19 결과는 일부만 있음 (A crashed, B-L 진행 중).

## 1. iter18 winner (0.8272) — robust or noise?

### sweep spread vs historical noise

| sweep | iter18 6-cell macro_f1 spread | best - worst |
|---|---|---|
| range | 0.7843 ~ 0.8272 | **+0.0429** |
| std (6 cells) | ~0.014 | |

| historical noise reference | spread | source |
|---|---|---|
| iter 8 T9d seed=42 vs T9g seed=43 (same config) | 0.9705 vs 0.9408 | **±0.030** macro_f1 (single-seed) |
| iter 11 fork F1 across seeds (T9 LS=0.07) | 0.945 vs 0.815 | **±0.065** fork F1 |
| iter 14 v20 single-seed | CF1 0.9226 | n=1 ref |

### 진단

- iter18A 0.8214 → iter18D 0.8272 = **+0.0058**.
- Single-seed measurement noise (memory `feedback_atomic_method_iteration.md`) ≈ ±0.030.
- **+0.0058 is well within ±0.030 noise band** — iter18D winner는 statistically NOT
  distinguishable from iter18A baseline. iter18B (0.7843) drop도 noise band 의 lower tail.
- iter18F1 (0.8200) vs F2 (0.8036) vs F3 (0.8196) = label_scale 0.5/0.75/1.0 sweep
  spread ±0.0082 — 역시 noise 안. label_scale 의 monotone effect 없음.
- **결론**: iter18 의 spec 차이는 noise 보다 작거나 같음. winner 는 점추정으로만 의미.
  3-seed mean 측정 없이 "grid_pair = best" 결론 약함.

또한 iter18 의 macro_f1 은 **4-class only (bank_boundary/fork/scratch/scratch_rot) 의
single-class macro F1**. 6×4 confusion 검증 결과 모든 cell 에서 2-combo accuracy 95-100%,
3-combo zero-shot 0%, Normal/Invalid 는 inference variant (I3 vs I7/I10) 의 함수
(I3=0%, I10=97-100%). 즉 iter18 0.8272 ↔ 0.8214 의 차이는 4-class single F1 만의 미세
차이.

## 2. Per-class cap diagnosis (6 × 4 표)

iter18 6 cells × 4 train class F1 (각 report.md best cell, 4-class only):

| run | best cell | F1_bb | F1_fork | F1_sc | F1_sr | macro |
|---|---|---:|---:|---:|---:|---:|
| iter18A baseline single-CutMix | T0__I3 | 0.9800 | **0.7485** | 0.7912 | 0.7660 | 0.8214 |
| iter18B single + paired | T0__I3 | 0.9476 | **0.6519** | 0.7350 | 0.8028 | 0.7843 |
| iter18D grid + paired | T0__I3 | 0.9670 | **0.7266** | 0.7828 | 0.8325 | 0.8272 |
| iter18F1 grid_complete LS=0.5 | T0__I7 | 0.9527 | 0.7421 | 0.7915 | 0.7939 | 0.8200 |
| iter18F2 grid_complete LS=0.75 | T0__I10 | 0.9513 | **0.6880** | **0.7110** | 0.8641 | 0.8036 |
| iter18F3 grid_complete LS=1.0 | T0__I10 | 0.9669 | 0.7430 | **0.6853** | 0.8832 | 0.8196 |
| **mean** | | **0.9609** | **0.7167** | **0.7495** | **0.8237** | 0.8127 |
| **std** | | 0.013 | 0.034 | 0.044 | 0.044 | 0.015 |

### 핵심 cap 진단

1. **bank_boundary = saturated** (mean 0.961, std 0.013) — solved class.
2. **scratch_rot = strong & rising** with hard label (F2/F3 0.86-0.88).
3. **fork = the cap** (mean **0.717**, range 0.65-0.75). 일관되게 worst class. iter 8
   T9 sweep 에서 fork F1 ±0.065 seed variance 도 iter 11에서 확인됨 — fork 가
   single-seed noise 의 dominant source.
4. **scratch = secondary cap** (mean 0.750, std 0.044). 특히 hard label (F2/F3) 에서
   scratch F1 collapse (0.71/0.69) — scratch_rot 가 high prob으로 fire 하면서 scratch
   를 잡아먹음 (cross-class suppression, memory `feedback_cross_class_suppression.md`).
5. **fork-scratch trade-off** (F2/F3 hard label) — scratch_rot 강해지면 scratch + fork
   둘 다 약해짐. 같은 chip 위에서 fork 의 vertical 줄무늬가 scratch_rot 의 -21° 회전
   diagonal 과 perceptually 가까움 (특히 strong rotated scratch 가 vertical 분량을
   포함할 때).

**Cap = fork (mean F1 0.717)**. 모든 다음 변경은 fork 에 직접 효과 없으면 noise band
에서 못 빠져나감.

## 3. iter19 hypothesis sanity

### 12-cell sweep design check

- **2 dim** (group ∈ {2,3,4}, label_scale ∈ varying) × pair=masked + 2 baselines.
  Phase A / Phase F 도 2 dim grid sweep 이었으므로 본 단일 atomic change/iter
  policy (memory `feedback_atomic_method_iteration.md`) 위반 아님 — atomic
  change 는 "complement CutMix mode 도입", group/scale 은 그 mode 의 sub-knob.
- **다만 12 cell 모두 single seed = ±0.030 noise band 안** 결과만 나오면
  hypothesis falsification 어려움. 6-12 분/cell × 12 = 1-2 hr GPU 적정.

### Paper grade?

- arxiv 2405.13451 ([Label Propagation for CutMix multi-label remote sensing](https://arxiv.org/abs/2405.13451), 2024) — multi-label CutMix 의 label noise 문제를 pixel-level
  positional info pairing 으로 해결. complement CutMix 의 motivation 과 일치
  (sample 분산 → label noise 줄임). +2-4% mAP macro 개선 보고.
- ConCutMix (contrastive feature space rectification) / LGCOAMix (attention-weighted
  superpixel mixing) / TdAttenMix (top-down attention fusion) — 2024 SOTA 가
  attention/semantic-guided 방향. 우리의 complement 는 random partition 이라
  semantic-blind — bb/fork/scratch/sc_rot 의 spatial pattern (fork = 수직선,
  scratch = 대각선) 을 grid 8×8 에 random 분배. fork 의 1 픽셀 너비 vertical line
  이 group2 → 절반 cell 만 가지면 line 이 끊겨서 학습 difficulty 증가 가능.

→ **Paper grade hypothesis 는 yes (counterfactual + content-vs-location decoupling)
이지만 우리 chip 도메인에 직접 fit 하지는 않음**. fork 의 thin vertical line 이
random group 분배에 가장 손상 큼.

### 사전 prediction (iter19 best cell)

iter19A 회수 결과 (epoch 1 partial): macro_f1 **0.8078**, fork **0.7993** (iter18 mean
0.717 대비 +0.08! crashed best 도 fork lift). 진짜 epoch-8 결과 나오면:

| 가설 | 어느 cell 이 best | 예상 macro_f1 |
|---|---|---|
| group 2 + label_scale 0.5 (A 진짜) | **iter19A** (가장 작은 group, 가장 약한 label) | **0.83 ± 0.02** |
| group 2 + LS 1.0 | iter19C | 0.81 ± 0.02 (hard label = scratch suppress) |
| group 3 + LS 0.67 | iter19E | 0.82 ± 0.02 |
| group 4 + LS 0.5 | iter19H | 0.79 ± 0.03 (fork too fragmented) |
| baseline pair=none g2 | iter19K | 0.81 ± 0.02 (paired branch off, comparable to iter18A) |

★ **prior best prediction**: **iter19A (g=2, LS=0.5, masked)** — group 작을수록
fork 의 vertical line 살아남는 cell 이 많고, soft label 0.5 가 area-proportional 에
맞음. iter19A 의 epoch-1 회수가 fork 0.799 찍은 것이 강한 신호. 만약 iter19A 가
epoch-8 까지 갔으면 0.83-0.85 plausible.

★ **prior worst prediction**: **iter19J (g=4, LS=1.0)** — fork line 4 group 으로
shred + hard label 로 cross-class suppression 강화. 0.77-0.80 예상.

만약 iter19 best cell macro_f1 ≤ 0.835, complement CutMix 는 noise-band 안.
≥ 0.85 면 atomic 효과 입증.

## 4. Next atomic 변경 1 개 (iter19 외)

### Spec — fork-targeted positive sample oversample

**가설**: fork = thin vertical line (single bit width), v20 chip generator 에서 sigma
1.8-2.5 으로 두께 ↑ 했지만 iter18 mean F1=0.717 여전히 worst. **fork 학습 sample 만
2× weighted in DataLoader** — 모델이 fork visual feature 에 더 많은 gradient 받음.

```bash
python -m chip_multilabel._train_chip_variant \
    --variant T7 --ls 0.20 \
    --epochs 8 --batch 8 --accum 4 --seed 1 \
    --cutmix-p 0.25 --cutmix-rect 0.5 --cutmix-mode single \
    --cutmix-pair masked --cutmix-pair-loss-w 1.0 --cutmix-pair-fill corner \
    --class-sample-weight 'fork:2.0' \
    --tag T7_iter20A_forkweighted2x_seed1 \
    --out-root outputs/iter20A_forkweight2x
```

(★ `--class-sample-weight` 미구현 → `_train_chip_variant.py::collect_samples`에 1
function 추가: classes 명시된 weight ratio 로 WeightedRandomSampler. atomic patch
~30 line.)

### Paper 인용

- **Cui et al. 2019 ["Class-Balanced Loss Based on Effective Number of Samples"
  CVPR](https://arxiv.org/abs/1901.05555)** — single-label imbalance 의 simple
  re-weighting (1/n_class) 보다 effective number 가 sample 의 marginal contribution
  반영. multi-label 에서 minority class 의 effective sampling rate 상향 = 같은
  motivation.
- 우리 case 는 fork = imbalance 가 아닌 **difficulty imbalance** (4 class 모두
  200 sample 동수). 그러나 학습 signal 측면에서 fork loss gradient 가 다른 class
  보다 작아서 weighted sampling 이 effective sample size 의 fork 비율 ↑.

### Chip 도메인 reasoning

- fork = **wafer 의 die boundary 가 아닌 vertical line cluster** (수직 줄무늬).
  200×200 chip = 33×33 wafer cell 표현. fork single bit width 의 vertical line
  은 backbone (ConvNeXtV2) 의 first-stage stem 에서 4× downsample 후 50×50 에서
  1-2 pixel 너비로 줄어듦 — 정보량 작음.
- scratch_rot 와의 confusion: iter18F2/F3 hard label 에서 sc_rot F1 0.86-0.88 로
  올라가면 scratch + fork 같이 무너짐. scratch_rot 의 -21° 회전 (top tilts right,
  memory `feedback_v19_chip_strength_hierarchy.md`) 이 fork 의 vertical 과
  small-angle 에서 visually 가까움.
- **fork 2× weighted sampling 이 fork visual feature representation 을 강화** →
  fork ↔ sc_rot decision boundary 가 fork 쪽으로 push. iter18 의 fork F1 0.717
  → 0.78-0.82 expected (fork F1 +0.07-0.10).

### 예상 결과

- macro_f1 (4-class) baseline 0.8214 → **0.85 ± 0.02** (fork lift dominate).
- scratch 약간 hurt (cross-class suppression flip — 이번엔 fork 가 sc/sc_rot 잡아먹음).
- 만약 scratch F1 -0.05 이상 hurt 면 fork:1.5x 로 sweep 한 cell 추가.

## 5. 최종 한 줄 recommendation

> **iter19 끝나면 즉시 → fork-weighted 2× sampling (+`--class-sample-weight fork:2.0`,
> single CutMix + masked pair, T7 base, 1 atomic patch)**. fork = mean F1 0.717
> cap, complement CutMix sweep 결과와 무관하게 fork 직접 attack 이 noise band
> (±0.030) 위로 올라갈 가장 빠른 path. 6-12 분/job, 1 GPU.

## Sources

- iter18 reports: `D:/project/known-cnn/outputs/iter18{A,B,D,F1,F2,F3}_*/T7_*/eval_seed1/stage1_*/report.md`
- iter18D preds breakdown: 4-single 153/160 = 95.6%, 6 2-combo 239/240 = 99.6%,
  4 3-combo 0/160 zero-shot, Normal/Invalid I3=0% I10=97-100%.
- iter19A epoch-1 partial: `outputs/iter19A_complement_g2_l0.5_pmasked/.../eval_seed1/stage1_260508_141521/report.md`
  → macro_f1 0.8078 fork **0.7993** (already +0.08 over iter18 mean fork 0.717).
- iter19 launcher: `D:/project/known-cnn/_run_iter19_complement_resume.sh`
- arxiv [2405.13451](https://arxiv.org/abs/2405.13451) Multi-label CutMix label propagation 2024.
- Cui et al. [Class-Balanced Loss CVPR 2019](https://arxiv.org/abs/1901.05555).
- Memory: `feedback_atomic_method_iteration.md`, `feedback_cross_class_suppression.md`,
  `feedback_v19_chip_strength_hierarchy.md`.
