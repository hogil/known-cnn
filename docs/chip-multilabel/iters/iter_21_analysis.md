# Iter 21 — 8-model dual-eval (v14class / v15direct) deep analysis

> 260509. clean baseline 재학습 (classification_chips/ only, no pre-blended Normal,
> dual eval v14/v15direct). lens: **bit_F1** + **ni_chipFAR** only (no macro_f1).

## 1. Result snapshot (iter21A→H)

| iter21 | spec | v14 bF1 | v14 ni% | v15 bF1 | v15 ni% | v15 bb | v15 fk | v15 sc | v15 sr |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| A | 12-T5 baseline (no-Norm, no-CutMix) | 0.9745! | 100% | 0.7872 | 0.0% | 0.8612 | 0.9298 | 0.5841 | 0.7739 |
| B | T7N pure (no CutMix) | 0.8609! | 100% | 0.8089 | 2.5% | 0.9448 | 0.6420 | 0.7675 | 0.8811 |
| C | T7N + std CutMix (Yun 2019) | 0.9415! | 100% | 0.8457! | 100% | 0.9937 | 0.6498 | 0.8060 | 0.9333 |
| D | 18F1 grid_complete LS=0.5 | 0.9431 | 1.25% | 0.9252 | 2.5% | 0.9810 | 0.8698 | 0.8889 | 0.9610 |
| **E ★** | **19C compl g=2 LS=1.0 (FCM-PM)** | **0.9913** | **0.00%** | **0.9691** | 3.75% | **0.9905** | **0.9644** | 0.9439 | 0.9776 |
| F | 19E compl g=3 LS=0.67 | 0.9875 | 1.25% | 0.9676 | 1.25% | 0.9776 | 0.9404 | 0.9747 | 0.9776 |
| G | 19G compl g=4 LS=0.25 | 0.9674 | 2.5% | 0.9716! | 100% | 0.9682 | 0.9644 | 0.9666 | 0.9873 |
| H | 19I compl g=4 LS=0.75 | 0.9626 | 1.25% | 0.9346 | 0.0% | 0.8491 | 0.9511 | 0.9508 | 0.9873 |

`!` = ni_FAR ≥ 100% lookup deception (predicting any defect on every Normal —
bit-F1 inflated by hidden over-recall, not real performance). 정직한 Pareto
optimum 은 두 column 동시 통과해야 함.

A/B/C 의 v14 bit_F1 0.97/0.86/0.94 모두 100% Normal-FAR 으로 가짜 lift. v14
class-folder 단일 라벨 GT 가 Normal chip 을 평가에 포함 안 시키는 bias 가
mask. v15direct 가 진짜 ni% 를 노출.

## 2. ★ 12-T5 (A) → 19C (E) 의 v15 bit_F1 +0.182 mechanism (0.7872 → 0.9691)

class-별 lift:

| class | A v15 F1 | E v15 F1 | Δ |
|---|---:|---:|---:|
| bank_boundary | 0.8612 | 0.9905 | **+0.129** |
| fork | 0.9298 | 0.9644 | +0.035 |
| scratch | **0.5841** | **0.9439** | **+0.360** |
| scratch_rot | 0.7739 | 0.9776 | **+0.204** |

핵심 lift = **F1_scratch +0.360** (0.58→0.94). scratch_rot +0.204, bank_boundary
+0.129. fork 는 이미 0.93 으로 saturated.

### 왜 12-T5 (A) 에서 scratch 가 약한가

200×200 chip 8×8 grid 에서:
- **bank_boundary**: vertical+horizontal 격자 (전체 cell 이 신호 포함)
- **fork**: vertical short stroke cluster (3-5 cell 로 localized, narrow horizontal extent)
- **scratch**: 대각선 long line, 8-12 cell 가로지름, **각 cell 에서는 sparse**
- **scratch_rot**: 다른 angle 의 long line, 각 cell sparse

12-T5 = BCE pure (no CutMix, no LS, no Normal training). 학습 데이터 200 chip/class
에서 scratch 의 cell-sparse signature 를 BCE 가 약하게 학습 — single-positive
classification 의 implicit class prior 가 fork (dense local) 같은 pattern 을 favor.
v15direct eval (Normal chip 200) 에서 scratch head 가 임계 통과 못 해 F1 0.58.

### standard CutMix (C, Yun 2019) 의 ni_FAR 100%

C = T7N (Normal training ON) + std rect CutMix p=0.25 — eval ni_FAR 100%
양쪽 (v14 + v15 모두). **이게 paper-worthy negative**:

Yun 2019 std CutMix: `λ y_A + (1-λ) y_B` (proportional soft label). multi-label
sigmoid setup 에서 chip A (fork) 에 chip B (scratch) box-paste 시:
- box 영역 50% 면 fork target 0.5, scratch target 0.5
- model 이 soft target 따라 **두 head 모두 mid-prob (0.4-0.6)** 출력 학습
- inference 시 Normal chip 의 random green-speckle 위에서 두 head 모두
  threshold 근처 fire → ni 100%

[Bevandic 2024](https://arxiv.org/abs/2405.13451) 의 핵심 발견과 정확히 일치:
"directly applied in multi-label classification it can lead to the erasure or
addition of class labels (label noise)". area-proportional soft label 이
multi-label semantic 깨뜨림.

### vs E (19C complement g=2 LS=1.0) 의 결정적 효과

E = compl g=2 (그룹 2 chip 의 모든 cell 분산 fill, label_scale=1.0 hard
both-positive). 차이:
- **all cells preserved** (box 외 영역도 mix 하므로 location heuristic 차단)
- **hard label both** (label_scale=1.0 → A.label OR B.label 둘 다 1.0, soft
  area-proportional 폐기)
- **g=2 = 2-class compositional learning** — bb/fork, fork/scratch, scratch/sr
  pair 전부 자연 cover

→ scratch 의 long-line signal 이 chip A 의 cell 에 흩어진 chip B (예: bb 격자)
와 함께 학습되면서 model 이 **"sparse cell 에서도 scratch fire"** 를 배움.
F1_sc 0.58 → 0.94. snapmix 의 CAM-based label 과 컨셉 비슷하지만 grid
mechanism 이 더 단순 + multi-label 에 직접 적용 가능.

## 3. 19C (E) v15 ni_FAR 3.75% (3 chip / 80) 정체 분석

`outputs/iter21E_19C_repeat/.../eval_v15direct/.../errors/T0__I10/missed_normal/`
에 3 chip:

| chip | pred | per_class_probs (bb/fk/sc/sr) | 패턴 |
|---|---|---|---|
| Normal_0047 | fork | 0.40 / **0.66** / 0.16 / 0.20 | dense green-speckle, 우측 vertical 군집 |
| Normal_0104 | fork | 0.37 / **0.75** / 0.21 / 0.21 | very dense scattered grade-2 dots, 좌상 cluster |
| Normal_0152 | fork+scratch_rot | 0.32 / **0.60** / 0.15 / 0.29 | medium dense, 대각 + vertical accidental align |

**공통**: 모두 grade-2 (green) high-density Normal palette 의 stochastic
dot-cluster 가 우연히 vertical short-line 패턴 mimic → fork head 가 0.60-0.75
fire. Normal_0152 는 추가로 dot-density 의 대각 alignment 가 sr head 까지 fire.

**root cause**: Normal training (zero-vector target) 이 chip-level prob 를
잘 누르긴 하지만 (`ni% 100→3.75%`), **green-density extreme tail (top 2%)** 에서
fork head 의 vertical-pattern detector 가 여전히 false-fire. 200 Normal training
sample 의 green-density distribution coverage 부족. Cross-class suppression
(feedback_cross_class_suppression.md) 의 잔재 — fork prob 가 normal chip
mean 0.16 인데 dense Normal 에서 0.60+ 까지 spike.

**paper-grade fix 후보**:
- Normal training pool size 200 → 600+ (high-density tail sampling 추가)
- focal-style hard-negative mining: Normal training loss 에 fork prob > 0.4
  mask 로 weighting up
- pos_weight tuning: BCE 의 fork pos_weight 0.7 (default 1.0) 로 살짝 다운 →
  high-density false-fire 줄어듦, 단 fork recall 0.005-0.01 trade

## 4. iter23 후보 atomic 변경 spec

candidate ranking (최대 single-axis 변경 + 1 GPU job 6-12분):

### A1. **fork pos_weight 0.7** (recommended ★)
- atomic CLI: `--pos-weight 1.0,0.7,1.0,1.0`
- expectation: ni_FAR 3.75% → 1.25-2.5%, F1_fk -0.005 ~ -0.01 (Normal_0047/0104
  의 fork prob 0.66/0.75 → 0.5-0.6 으로 살짝 하향 → threshold 0.580 통과 못
  하게)
- rationale: dense Normal 의 fork false-fire 가 root cause. Cole 2021 SPML
  recommends class-specific pos_weight calibration. fork head 만 selective
  weakening 으로 다른 3 class 영향 minimal.
- paper: [Cole 2021 SPML](https://arxiv.org/abs/2106.09708) — single-positive
  multi-label needs per-class pos_weight when negatives dominate.

### A2. compl g=2 → g=3 ablation 다시 (LS=1.0 fix)
- iter21F (g=3 LS=0.67) 가 v15 0.9676 (E 0.9691 비슷). g=3 LS=1.0 미테스트.
- atomic CLI: `--cutmix-complement-group 3 --cutmix-label-scale 1.0`
- expectation: F1_sc 0.97+ 으로 lift 가능 (g=3 의 추가 class diversity), ni% 동등
- 단점: g 증가는 batch 내 mix 복잡도 ↑ + 6-12분 안정 안에 들어가는지 확인 필요

### A3. drop_path 0.05
- iter 8 negative 결과 있었으나 19C 위에서는 미테스트 (조합 변경)
- atomic CLI: `--drop-path 0.05`
- expectation: ni% +0.0 (정체) 또는 -1.25%, F1 ±0.005 noise
- single-axis 라 안전하나 lift 가능성 낮음

### A4. ASL light (γ_neg=2, γ_pos=0)
- 19C 의 BCE+LS 를 ASL 로 swap
- atomic CLI: `--loss asl --asl-gamma-neg 2`
- 기대 작음 (iter 8 에서 ASL light net negative)

### **Recommendation: A1 fork pos_weight 0.7**

CLI:
```bash
python -m chip_multilabel._train_chip_variant \
  --variant T7 --label-smoothing 1.0 \
  --cutmix-mode complement --cutmix-complement-group 2 --cutmix-label-scale 1.0 \
  --pair-mask none \
  --pos-weight "1.0,0.7,1.0,1.0" \
  --epochs 8 --batch 4 --accum 4 --lr-head 1e-4 --seed 1 \
  --output-dir outputs/iter23A_19C_pw_fork07
# eval dual: v14 + v15direct
```

## 5. statistical significance plan

iter22 (현재 dispatching, log `outputs/_iter22_19C_tune.log`):
- iter22A_seed7: T7 19C compl g=2 LS=1.0, seed=7
- iter22B_seed42: same, seed=42
- iter21E_seed1 (이미 끝남) + iter22A_seed7 + iter22B_seed42 = **3-seed mean ± std**

paper claim 강도:
- single-seed iter21E: v15 bit_F1 0.9691, ni 3.75%
- 3-seed mean target: 0.965 ± 0.005, ni 2.5-5%
- iter21A 12-T5 single-seed: v15 bit_F1 0.7872 → effect size = (0.965 - 0.787)
  / 0.005 = ~36σ → **paper-quality significant**

3-seed 가 모두 v15 bit_F1 ≥ 0.95 + ni ≤ 5% 로 들어오면 "FCM-PM (Full
Complement Mix - Pair Masked) significantly outperforms standard CutMix
(Yun 2019) and BCE+LS pure baseline in chip-level multi-label classification
on 200×200 wafer chip patches" claim 가능.

## Summary

12-T5 → 19C 의 v15 bit_F1 +0.182 lift 는 (a) **all-cells complement** location
heuristic 차단 + (b) **hard label both** (label_scale=1.0) multi-label
semantic 보존 의 곱 효과. F1_scratch +0.36 이 dominant. 19C 의 v15 ni 3.75%
잔재 3 chip 모두 dense green-speckle Normal 의 fork false-fire — fork
pos_weight 0.7 로 1-axis fix 가 가장 안전한 atomic next.

[OUT] D:/project/known-cnn/docs/chip-multilabel/iters/iter_21_analysis.md
