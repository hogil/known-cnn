# Chip Multi-Label 결함 분류기

## 프로젝트

chip 이미지(200×200) 의 결함 4종(bank_boundary, fork, scratch, scratch_rot) 을 **multi-label** 로 분류 — 한 chip 에 여러 결함 동시 가능, sigmoid 4 head 독립 출력. 학습 = 단일 결함 4×200 + 정상 200 (1,000장), 평가 = 단일 + 2-combo + OOD-overlay + Normal/Invalid + OOD wafer 무늬 = 20 class 3,850장. chip 합성 = wafer 분포 → alpha map → 확률적 grade 픽셀 → 8-color palette PNG. combo 는 RGB pixel-wise min (두 결함 darker 보존).

## 평가 지표

각 chip은 4-bit GT 와 4-bit pred를 비교해서 **각 bit를 독립 binary classification으로 본다**. 즉 chip 한 장에서 4개의 binary 판정이 나온다. bit 순서는 정해져있다:

```
[ bit 0 = bank_boundary, bit 1 = fork, bit 2 = scratch, bit 3 = scratch_rot ]
```

예를 들어 fork + scratch가 같이 있는 chip의 GT는:

```
[ bb=0, fork=1, scratch=1, scratch_rot=0 ]   = [0, 1, 1, 0]
```

bit 1, bit 2 만 활성. 모델은 4개의 sigmoid 출력을 threshold 비교해서 같은 형식의 4-bit pred를 만든다.

- **CF1** (per-bit macro F1): 4 class 각각의 F1을 따로 구한 다음 평균. 모든 class에 동등 가중. 한 class가 약하면 평균이 떨어지므로 minority class 진단에 좋다. 논문 main metric.
- **F1_bit** (micro F1): 4 class 의 TP/FP/FN을 한꺼번에 모아서 F1 한 번 계산. bit 빈도에 비례 가중되므로 large class가 결과를 dominate.
- **chip_FAR**: 정상이어야 할 chip 중에서 모델이 결함 bit를 하나라도 잘못 fire 한 비율. 라인 운영에서 가장 중요한 false alarm 지표. 우린 이걸 정상/측정불능(Normal+Invalid) chip 만 보는 `ni_chip_FAR`(운영 main)와 학습 안한 OOD chip 만 보는 `ood_chip_FAR`(diagnostic) 둘로 분리해서 본다.

운영 통과 기준은 `CF1 ≥ 0.83` + `ni_chip_FAR ≤ 5%` 동시 만족.

## 현재 성능

| metric | 값 | 무엇을 보는가 |
|---|---:|---|
| **CF1** | **0.9406** | 4 class F1 의 산술평균 (모든 class 동등 가중) — 논문 main 지표 |
| F1_bit | 0.9375 | 4 bit 전체를 한꺼번에 합쳐 계산한 F1 (전반적 bit 정확도) |
| F1_bank_boundary | 0.9797 | bank_boundary bit 의 F1 (TP 와 FP, FN 균형) |
| F1_scratch | 0.9165 | scratch bit 의 F1 |
| F1_scratch_rot | 0.9979 | scratch_rot bit 의 F1 |
| **ni_chip_FAR** | **0.00%** | 정상/측정불능 chip 중에서 모델이 결함 bit 하나라도 잘못 fire 한 비율 — 운영 main |
| ood_chip_FAR | 1.41% | 학습 안한 OOD 무늬 chip 중 fire 한 비율 (진단용, 운영 통과 판정엔 X) |

운영 threshold (`CF1 ≥ 0.83` + `ni_chip_FAR ≤ 5%`) 통과 ✅.

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

## 성능 향상에 쓴 기법

- **Normal 학습 추가** — 정상 chip 200장을 zero-vector multi-hot label로 학습 데이터에 넣었다. 가장 큰 lever였고 chip_FAR가 80%대에서 0%까지 떨어졌다.
- **chip 합성 강화** — 결함 라인 두께와 라인별 길이/위치 산포를 조정해서 분류기가 학습할 결함 신호를 더 명확하게 만들었다.
- **Loss 설계** — 8가지 loss(CE, CE+LS, Focal, ASL, BCE, BCE→ASL, BCE+LS, Sigmoid Focal)를 비교했고 BCE+LS=0.20 + random CutMix p=0.25가 가장 안정적이었다. Focal/ASL은 calibration이 망가지거나 negative를 너무 강하게 눌러서 fork 같은 약한 신호를 죽였다.
- **chip_FAR split** — Normal+Invalid 와 OOD를 분리 측정해서 운영용 metric(`ni_chip_FAR`)을 OOD artifact에서 떼어냈다. bundled 96% 의 정체가 OOD 100% fire였다는 게 split 후에 보였다.
- **logit-avg ensemble** — Normal 학습한 모델과 안한 모델의 logit을 평균해 complementary 약점을 보완했다.

## paper grounding

BCE / CF1 / F1_bit는 Tsoumakas 2007 / Wang 2016 / Chen 2019 multi-label 표준. Label Smoothing은 Müller 2019, Focal은 Lin 2017, ASL은 Ridnik 2021. CutMix는 Yun 2019 / Walawalkar 2020 / Sumbul 2024. 우리가 신규로 한 건 chip_FAR 3-way split(Normal/Invalid vs OOD), OOD-overlay benchmark(2 trained + 1 OOD), with × without-Normal logit ensemble, 그리고 chip 합성 generator 자체.
