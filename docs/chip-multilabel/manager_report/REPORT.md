# Chip Multi-Label 결함 분류기

## 프로젝트

chip 이미지(200×200) 의 결함 4종(bank_boundary, fork, scratch, scratch_rot) 을 **multi-label** 로 분류 — 한 chip 에 여러 결함 동시 가능, sigmoid 4 head 독립 출력. 학습 = 단일 결함 4×200 + 정상 200 (1,000장), 평가 = 단일 + 2-combo + OOD-overlay + Normal/Invalid + OOD wafer 무늬 = 20 class 3,850장. chip 합성 = wafer 분포 → alpha map → 확률적 grade 픽셀 → 8-color palette PNG. combo 는 RGB pixel-wise min (두 결함 darker 보존).

## 데이터 — 4 group sample

### 학습 4 obj single defect (학습 + 평가)

| | | | |
|:---:|:---:|:---:|:---:|
| ![](figs/bank_boundary.png) | ![](figs/fork.png) | ![](figs/scratch.png) | ![](figs/scratch_rot.png) |
| bank_boundary | fork | scratch | scratch_rot |

### 6 2-combo (평가 only, min-blend)

| | | |
|:---:|:---:|:---:|
| ![](figs/bank_boundary_AND_fork.png) | ![](figs/bank_boundary_AND_scratch.png) | ![](figs/bank_boundary_AND_scratch_rot.png) |
| bb+fork | bb+scratch | bb+sr |
| ![](figs/fork_AND_scratch.png) | ![](figs/fork_AND_scratch_rot.png) | ![](figs/scratch_AND_scratch_rot.png) |
| fork+scratch | fork+sr | sc+sr |

### 4 OOD-overlay (평가 only — 2 trained + 1 OOD overlay, GT는 2 trained bit만)

bit 순서 = `[bb, fork, scratch, scratch_rot]`

| | |
|:---:|:---:|
| ![](figs/fork_AND_scratch_AND_OOD_DiagonalSmear.png) | ![](figs/bank_boundary_AND_fork_AND_OOD_CenterDonut.png) |
| fork+sc + OOD_DiagonalSmear<br>GT = [bb=0, **fork=1**, **sc=1**, sr=0] | bb+fork + OOD_CenterDonut<br>GT = [**bb=1**, **fork=1**, sc=0, sr=0] |
| ![](figs/fork_AND_scratch_rot_AND_OOD_CrossScratch.png) | ![](figs/scratch_AND_scratch_rot_AND_OOD_Starburst.png) |
| fork+sr + OOD_CrossScratch<br>GT = [bb=0, **fork=1**, sc=0, **sr=1**] | sc+sr + OOD_Starburst<br>GT = [bb=0, fork=0, **sc=1**, **sr=1**] |

### Normal / Invalid

| | |
|:---:|:---:|
| ![](figs/Normal.png) | ![](figs/Invalid.png) |
| Normal | Invalid |

### 4 OOD wafer-pattern (평가 only — 학습 안한 외형, false alarm 측정)

| | | | |
|:---:|:---:|:---:|:---:|
| ![](figs/DiagonalSmear.png) | ![](figs/CenterDonut.png) | ![](figs/CrossScratch.png) | ![](figs/Starburst.png) |
| DiagonalSmear | CenterDonut | CrossScratch | Starburst |

OOD chip 은 wafer 단위 합성한 불량 wafer 에서 결함 영역(bin ≥ 200) 의 chip 만 따왔다. 등록된 4 결함(bb/fork/sc/sr) 형태가 아니라 학습에 한 번도 들어가지 않은 외형이다. **현업 라인에서 학습 안 한 새 결함 패턴이 random 하게 들어오는 상황을 시뮬레이션** — 모델이 4 결함 중 하나로 잘못 fire 하면 false alarm (`ood_chip_FAR`).

## 평가 지표

각 chip 4-bit GT vs 4-bit pred — `[ bit 0 = bank_boundary, bit 1 = fork, bit 2 = scratch, bit 3 = scratch_rot ]`. 예: fork+scratch chip → GT `[0,1,1,0]`.

- **CF1** (paper main): 4 class 각 F1 산술평균, minority class 진단
- **ni_chip_FAR**: 정상/측정불능 chip 중 결함 bit 잘못 fire 비율 (운영 main)
- **ood_chip_FAR**: 학습 안한 OOD chip 중 fire 비율 (diagnostic)

운영 통과 = `CF1 ≥ 0.83` + `ni_chip_FAR ≤ 5%` ✅

## 핵심 결과

### 🏆 Paper Main (260511, n=200 robust)

**FCM-PM (Full-Cover Mixup with Pair Mask)** + 4-bag majority vote ensemble

- **Paper headline**: 4-bag {pure-hard composition} = **CF1 0.9953 / ni_FAR 0.00%** ✅
- **1× cost single SOTA**: 4-bag teacher KD distillation = **CF1 0.9872 / ni_FAR 0.50%**

### Two-tier FCM-PM method (paper §4)

1. **Pair Mask** (safety-critical): training 시 chip A 결함 + chip B 결함 paste 영역 의 pair-mask 가 Normal/Invalid 의 defect-prediction 차단. **제거 시 -0.18 CF1 catastrophic FAR collapse** (2.5% → 100%).
2. **Group Complete** (accuracy-critical, CutMix-complement mode): chip A 결함 영역 전체에 chip B 의 결함 합성 — combo 학습 효과. 제거 시 -0.035 (helpful but tunable).

### Production deployment 권장

| 비용 | model | CF1 | FAR | 권장 |
|---:|---|---:|---:|---|
| 1× | KD distill 4-bag teacher α=0.5 T=4 | 0.9872 | 0.5% | ★ standard production |
| 4× | 4-bag majority vote (pure-hard) | 0.9953 | 0% | high-accuracy SOTA |

## FCM-PM 학습 기법

### Pair Mask + Group Complete (paper main)

학습 시 chip A 위에 chip B 의 결함을 paste 해서 multi-label sample 생성. 두 메커니즘 핵심:

1. **Pair Mask**: paste 영역의 Normal background 부분에 mask 처리 → model 이 background 를 "정상" 으로 학습 → ni_FAR 차단
2. **Group Complete (complement mode)**: chip A 결함 영역 전체에 chip B 결함 채움 → 두 결함 fully overlap 학습 → combo 평가에 generalize

### CutMix base (Yun 2019)

학습 데이터에는 단일 결함 chip 만. CutMix 는 학습 중 일부 batch (25%) 에 chip A + chip B 합성으로 multi-label sample 생성. 학습 안한 combo 평가에 일반화.

| 원본 bank_boundary | 원본 scratch | CutMix 결과 |
|:---:|:---:|:---:|
| ![](figs/cutmix_demo/orig_bank.png) | ![](figs/cutmix_demo/orig_scratch.png) | ![](figs/cutmix_demo/cutmix_random_rect.png) |

### BCE + Label Smoothing (T7 BCE+LS, ls=0.20)

target soft-label `[bb=0.05, fork=0.85, sc=0.05, sr=0.05]` (LS ε=0.20). over-confidence 완화 + FAR 통제.

→ paper §6: **Pair Mask + BCE+LS + Knowledge Distillation = 3 orthogonal FAR-control mechanisms** (실험 35+ 종 결과)

### Knowledge Distillation (1× cost SOTA)

4-bag (NEW MAIN composition) 의 평균 sigmoid prob 를 teacher 로 → student α=0.5 T=4 학습. paper §5: 14-bag teacher (α=0.3) vs 4-bag teacher (α=0.5) — teacher prob sharpness 에 따라 α 조정. 4-bag teacher 가 sharper → α=0.5 sweet spot.

## paper grounding

BCE / CF1 / F1_bit (Tsoumakas 2007, Wang 2016, Chen 2019), Label Smoothing (Müller 2019), Focal (Lin 2017), ASL (Ridnik 2021), CutMix (Yun 2019, Walawalkar 2020, Sumbul 2024), Knowledge Distillation (Hinton 2015, Yang 2023 multi-label).
