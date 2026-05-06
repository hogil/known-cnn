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

OOD chip 은 wafer 단위 합성한 불량 wafer 에서 결함 영역(bin ≥ 200) 의 chip 만 따왔다. 등록된 4 결함(bb/fork/sc/sr) 형태가 아니라 학습에 한 번도 들어가지 않은 외형이다. **현업 라인에서 학습 안 한 새 결함 패턴이 random 하게 들어오는 상황을 시뮬레이션**하려고 평가에만 추가했다 — 모델이 이걸 보고 학습된 4 결함 중 하나로 잘못 fire 하면 false alarm 으로 잡힌다 (`ood_chip_FAR`).

## 학습 기법 설명

### Loss

- **BCE (Binary Cross Entropy)** — 4 class 각각 독립 binary classification. multi-label 표준 loss. `L = -Σ [y·log(p) + (1-y)·log(1-p)]`.
- **CE (Cross Entropy)** — softmax 기반 single-class loss. multi-label 환경에 부적합 (한 chip 에 여러 결함이면 softmax 가 둘 중 하나만 살림).
- **Label Smoothing (LS)** — target 0/1 → 0.10/0.90 같이 soft 화 (ε=0.20 시). over-confidence 완화 (Müller 2019). BCE+LS 가 BCE 단독보다 안정적.
- **Focal Loss (Lin 2017)** — `-α(1-p)^γ log(p)`. 쉬운 example (p≈1) 의 loss 를 down-weight 해서 어려운 example 에 학습 집중. RetinaNet 의 sigmoid focal 버전 (T9) 도 있음.
- **ASL (Asymmetric Loss, Ridnik 2021)** — multi-label 전용. positive (실제 결함) 는 BCE-like, negative (정상) 는 focal 강화 (γ_neg=4). multi-label SOTA loss.

### CutMix — 학습 시 chip 두 장을 합쳐 새 학습 sample 생성

**random rectangle CutMix** (Yun 2019) — chip A 위에 chip B 의 직사각 patch 한 개를 paste. label 도 면적 비례로 합침.

| 원본 bank | 원본 scratch | CutMix 결과 (bank + scratch rect) |
|:---:|:---:|:---:|
| ![](figs/cutmix_demo/orig_bank.png) | ![](figs/cutmix_demo/orig_scratch.png) | ![](figs/cutmix_demo/cutmix_random_rect.png) |

**scattered CutMix** (Walawalkar 2020) — 큰 직사각 한 개 대신 여러 작은 patch 흩뿌림. 학습 시 결함이 chip 내 random 위치에 fragmented 로 보이도록.

| 원본 bank | 원본 scratch | scattered CutMix (5 patches) |
|:---:|:---:|:---:|
| ![](figs/cutmix_demo/orig_bank.png) | ![](figs/cutmix_demo/orig_scratch.png) | ![](figs/cutmix_demo/cutmix_scattered.png) |

학습 시 chip 의 일부 비율 (예: p=0.25) 이 CutMix 처리됨. 학습 데이터에 단일 결함만 있어도 모델이 multi-label 합성된 input 을 보게 되어 multi-label 일반화 가능.

### 기타

- **Normal training** — 정상 chip 을 zero-vector multi-hot label (`[0,0,0,0]`) 로 학습 데이터에 추가. 모델이 "결함 없음" 을 명시적으로 학습.
- **logit-avg ensemble** — 두 모델의 sigmoid 직전 logit 을 평균해서 complementary 약점 보완.
- **chip_FAR split** — false alarm 을 정상/측정불능 chip 만 보는 `ni_chip_FAR` 와 학습 안한 OOD 만 보는 `ood_chip_FAR` 로 분리.

## paper grounding

BCE / CF1 / F1_bit (Tsoumakas 2007, Wang 2016, Chen 2019), LS (Müller 2019), Focal (Lin 2017), ASL (Ridnik 2021), CutMix (Yun 2019, Walawalkar 2020, Sumbul 2024).
