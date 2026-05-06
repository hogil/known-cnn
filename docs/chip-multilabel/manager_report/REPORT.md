# Chip Multi-Label 결함 분류기 — 핵심 요약

## 1. 데이터 — 4 group sample 이미지 그리드

200×200 RGB palette PNG, `dist_apply/_sample_gen.py v20` 으로 합성. 학습 = 4 obj × 200 + Normal 200. 평가 = 20 class / 3,850 chip.

### 1.1 학습된 4 obj single defect (★ 학습 + 평가)

| | | | |
|:---:|:---:|:---:|:---:|
| ![](figs/bank_boundary.png) | ![](figs/fork.png) | ![](figs/scratch.png) | ![](figs/scratch_rot.png) |
| bank_boundary | fork | scratch | scratch_rot |

### 1.2 6 2-combo (★ 평가 only, min-blend)

| | | |
|:---:|:---:|:---:|
| ![](figs/bank_boundary_AND_fork.png) | ![](figs/bank_boundary_AND_scratch.png) | ![](figs/bank_boundary_AND_scratch_rot.png) |
| bb+fork | bb+scratch | bb+sr |
| ![](figs/fork_AND_scratch.png) | ![](figs/fork_AND_scratch_rot.png) | ![](figs/scratch_AND_scratch_rot.png) |
| fork+scratch | fork+sr | sc+sr |

### 1.3 4 OOD-overlay (★ 평가 only, 2 trained + 1 OOD)

GT bits = 2 trained 만 active. OOD pattern 은 visual noise — 모델이 무시하고 정답 2 bits 만 fire 해야 정답.

| | |
|:---:|:---:|
| ![](figs/fork_AND_scratch_AND_OOD_DiagonalSmear.png) | ![](figs/bank_boundary_AND_fork_AND_OOD_CenterDonut.png) |
| fork+sc+OOD_DS [0,1,1,0] | bb+fork+OOD_CD [1,1,0,0] |
| ![](figs/fork_AND_scratch_rot_AND_OOD_CrossScratch.png) | ![](figs/scratch_AND_scratch_rot_AND_OOD_Starburst.png) |
| fork+sr+OOD_CS [0,1,0,1] | sc+sr+OOD_SB [0,0,1,1] |

### 1.4 Normal / Invalid (정상 / 측정불능)

| | |
|:---:|:---:|
| ![](figs/Normal.png) | ![](figs/Invalid.png) |
| Normal | Invalid |

### 1.5 4 OOD wafer-pattern (★ 학습 안 한 외형, false alarm 측정)

| | | | |
|:---:|:---:|:---:|:---:|
| ![](figs/DiagonalSmear.png) | ![](figs/CenterDonut.png) | ![](figs/CrossScratch.png) | ![](figs/Starburst.png) |
| DiagonalSmear | CenterDonut | CrossScratch | Starburst |

wafer-canvas 합성에서 **defect cluster density 가장 높은 chip** (sub-region 25×25 max density). 모델이 한 번도 본 적 없는 외형.

---

## 2. ★ Multi-label 이해 — chip 1장에 여러 결함이 동시에

### 2.1 multi-class vs multi-label

| 구분 | 출력 | 예시 |
|---|---|---|
| multi-class (softmax) | 1 class 만 | "fork" |
| **multi-label (sigmoid 4 bit)** | 여러 class 동시 | "fork **and** scratch" |

### 2.2 4-bit 표기

```
position:  [bb,  fork,  scratch,  scratch_rot]
example:   [ 0,    1,        1,            0 ]   = "fork+scratch"
```

### 2.3 ★ combo chip → [0,1,1,0] 분해

**combo 이미지** (fork+scratch):

![](figs/fork_AND_scratch.png)

| 안에 들어있는 것 | 단독 이미지 | bit 위치 | 값 |
|---|:---:|---|:---:|
| ❌ bank_boundary 없음 | — | bit 0 (bb) | **0** |
| ✓ fork 있음 | ![](figs/fork.png) | bit 1 (fork) | **1** |
| ✓ scratch 있음 | ![](figs/scratch.png) | bit 2 (scratch) | **1** |
| ❌ scratch_rot 없음 | — | bit 3 (sr) | **0** |

→ **GT bits = [0, 1, 1, 0]**

합성: `np.minimum(fork_chip, scratch_chip)` (RGB pixel-wise min, palette 0=white 7=darkest 라 darker 보존 → 두 결함 모두 살아남음).

### 2.4 모델 동작

```
chip 입력  →  CNN  →  prob: bb=0.12  fork=0.90  sc=0.31  sr=0.13
                              ↓ threshold 비교
                            bits:    0      1      1      0
                              ↓
                          decision: fork+scratch ✓
```

★ softmax 가 아니라 **sigmoid 4 head 독립** — multi-label 의 핵심.

---

## 3. 이미지 합성 원리 (확률 분포 기반)

```
alpha map (obj 함수)  →  P(defect|alpha) per pixel
                      ↓
            grade 추출 (smoothstep 2-stage)
                      ↓
            8-color palette PNG
```

| step | 설명 |
|---|---|
| **alpha map** | obj 함수 — fork 7-9 다리, scratch 5-10 라인, scratch_rot -21° 7-12 라인, bank 격자 |
| **stage 1: P(defect)** | `is_defect = rand < alpha` |
| **stage 2: P(grade2 \| defect)** | smoothstep(alpha, 0.20, 0.50) — peak grade 2, halo grade 1 + 일부 3 |
| **palette PNG** | grade 0 white / 1 grey / **2 green** peak / **3 blue** strong |

---

## 4. 지표 — 어떻게 맞고 틀렸는지

GT [0,1,1,0] (fork+scratch) 일 때 다양한 pred:

| pred | 비교 | TP | FP | FN | TN |
|---|---|---:|---:|---:|---:|
| [0,1,1,0] **정답** | 정확히 일치 | 2 | 0 | 0 | 2 |
| [1,1,1,0] **over-fire** | bb FP 추가 | 2 | 1 | 0 | 1 |
| [0,1,0,0] **miss** | sc FN 누락 | 1 | 0 | 1 | 2 |

| 지표 | 측정 |
|---|---|
| **CF1** (per-bit macro F1) | `mean(F1_bb, F1_fk, F1_sc, F1_sr)` ★ paper main |
| **F1_bit** (micro F1) | `2·ΣTP / (2·ΣTP+ΣFP+ΣFN)` over 4N bits |
| **chip_FAR** | 정상 chip 중 ≥1 FP bit 비율 ★ 운영 main |
| **3plus_active%** | ≥3 bit 동시 fire (over-firing 진단) |

운영 통과 = **CF1 ≥ 0.83 + F1_fork ≥ 0.55 + ni_FAR ≤ 5%** 동시 만족.

---

## 5. ★ 성능 향상 핵심 기법 5

| # | 기법 | 효과 |
|---|---|---|
| **1** | **Normal training** (Normal 200 chip + zero-vector target) | **chip_FAR 80% → 0%** ★ 최대 lever |
| **2** | **chip 합성 v20** (fork peak σ ↑, line 두께 ↑, per-line 산포) | fork F1 **0.40 → 0.87** |
| **3** | **BCE + LS=0.20 + CutMix p=0.25** (8 loss variants 비교 winner) | CF1 +0.05 |
| **4** | **chip_FAR split** (Normal/Invalid vs OOD 분리) | bundled 96% artifact → ni_FAR 0.00% 추출 |
| **5** | **Logit-avg ensemble** (T7N + T5 with×without-Normal) | CF1 추가 +0.004 |

효과 적음: Sigmoid Focal (calibration ↓), ASL γ=4 (aggressive), scattered CutMix (FAR ↑), 3-combo (fork F1 ceiling).

---

## 6. 결과 누적

| iter | 변경 | CF1 | F1_fork | ni_FAR |
|---|---|---:|---:|---:|
| baseline (v19y T5) | BCE+CutMix | 0.8162 | 0.40 | 3.30% |
| + chip 합성 v19zpp + LS | T7 BCE+LS | 0.8490 | 0.52 | 80% (Normal X) |
| + Normal training | T7N | 0.9042 | 0.78 | 0.00% |
| + ensemble | T7N+T5 70:30 | 0.9083 | 0.77 | 0.50% |
| **★ 20-class master** | **T7N alone** | **0.9406** | **0.87** | **0.00%** |
| v20 fork 두께 ↑ retrain | T7N v20 | 0.9226 | 0.86 | 0.00% |

**누적 향상 v19y → 최고**: CF1 **+0.124**, fork F1 **+0.47**, FAR 안전권.

---

## 7. paper grounding

- BCE / CF1 / OF1: Tsoumakas 2007, Wang 2016, Chen 2019
- LS Müller 2019 / Focal Lin 2017 / ASL Ridnik 2021
- CutMix: Yun 2019, Walawalkar 2020, Sumbul 2024
- 신규 contribution: chip_FAR 3-way split, OOD-overlay benchmark, chip 합성 v20, with×without-Normal logit ensemble
