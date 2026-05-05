# Theoretical Background — Multi-label Wafer Classification

본 문서는 multi-label 학습 / SPML / calibration / density estimation / loss
formulation 의 **이론 + 수식 + 직관** 을 통합 정리. 3 deep-dive doc
(LOSS_DESIGN.md / MATCHING_DESIGN.md / DECISION_RULE.md) 의 base.

---

## 1. Multi-label Classification 이론

### 1.1 Problem Definition

```
Standard single-label:
   X = input space, Y = {1, ..., C}
   model f: X → Y
   target y_i ∈ Y (one class per sample)

Multi-label:
   X = input space, Y = {0, 1}^C (multi-hot vector)
   model f: X → Y
   target y_i ∈ Y (any subset of C classes)
   |y_i| ≥ 1 (atleast one positive)
```

### 1.2 Multi-label vs Multi-class

| Property | Multi-class | Multi-label |
|---|---|---|
| Label structure | one-hot | multi-hot |
| Mutual exclusion | yes (softmax) | no (sigmoid) |
| Loss | CE | BCE / ASL / etc |
| Eval metric | accuracy / F1 | mAP / hamming / macro F1 |
| Threshold | argmax | per-class threshold |

### 1.3 Multi-label Metrics

#### 1.3.1 Hamming Loss
```
hamming(y, ŷ) = (1/C) × Σ_c 1(y_c ≠ ŷ_c)
```
- bitwise mismatch
- 0 = perfect, 1 = all wrong
- imbalance 무시 (모든 bit 동등)

#### 1.3.2 Exact Match Ratio
```
exact_match = (1/N) × Σ_i 1(y_i = ŷ_i)
```
- strict — 모든 라벨 정확히
- 0~1
- multi-label 에서 보통 낮음 (10-50%)

#### 1.3.3 Macro F1
```
F1_c = 2 × precision_c × recall_c / (precision_c + recall_c)
macro_F1 = mean_c F1_c
```
- per-class F1 평균
- imbalance 에 robust
- 학계 표준

#### 1.3.4 Micro F1
```
TP = Σ_c TP_c, FP = Σ_c FP_c, FN = Σ_c FN_c
micro_F1 = 2 × TP / (2 × TP + FP + FN)
```
- 전체 sample 의 F1
- common class 에 dominated

#### 1.3.5 mAP (mean Average Precision)
```
AP_c = ∫₀¹ P_c(R) dR    (precision vs recall curve 면적)
mAP = mean_c AP_c
```
- threshold 무관 (rank 기반)
- 학계 multi-label benchmark 표준 (COCO, ChestX-ray)

#### 1.3.6 ECE (Expected Calibration Error)
```
ECE = Σ_b (|B_b| / N) × |acc(B_b) - conf(B_b)|

   B_b: prob 가 bin b 에 속하는 sample 모음
   acc(B_b): B_b 의 실제 positive 비율
   conf(B_b): B_b 의 평균 prob
```
- calibration quality
- 0 = perfect calibrated

→ paper-style table 에서 위 6 metric 모두 보고 (단일 metric 만 보면 mis-leading).

---

## 2. Single-Positive Multi-Label Learning (SPML)

### 2.1 SPML Setting

```
Standard multi-label: y_i = (1, 0, 1, 0, 1)  ← multi-hot, 모든 positive 알려짐
SPML:                  y_i = (1, ?, ?, ?, ?)  ← 한 positive 만 알려짐, 나머지 unknown
```

**우리 도메인** = SPML 의 정확한 setting:
- 합성 데이터 = single-label (한 wafer = 1 class)
- 추론 환경 = multi-label (한 wafer = 1+ class)
- 학습 시 알려진 라벨 = 한 class 만 (나머지 unknown)

### 2.2 Naive Approach — Assumed Negative

```
y_i = (1, 0, 0, 0, 0)   ← 알려진 1 외 나머지를 0 으로 가정
L = BCE(σ(z), y)        ← unknown 을 negative 로 학습
```

**문제**: 진짜 multi-label sample 에서 second positive 를 negative 로 학습 → false negative 학습 누적 → 모델이 second positive 의 logit 을 systematically underestimate.

### 2.3 SPML 문제 해결 paradigm

#### 2.3.1 Pseudo-labeling
```
모델 자체의 prediction 을 pseudo-label 로 활용
   if model predicts class c with high confidence, treat as positive
```

#### 2.3.2 Loss correction
```
unknown class 의 negative penalty 약화:
   L_corrected = BCE_known + λ × (negative penalty 보정)
```

#### 2.3.3 Regularization
```
모델이 너무 sharp 한 prediction 안 하도록:
   + entropy regularization, label smoothing, dropout
```

### 2.4 AdaGC (Adaptive Gradient Calibration, 2024 arXiv)

가장 최신 SPML 기법:
```
L_total = L_BCE_AN + λ × L_GC

L_GC = -(1/n) × Σ_i Σ_c 1(y_{i,c}=0) × log(1 - p_{i,c} × t_{i,c})

   p: student prediction
   t: teacher pseudo-label (dual EMA)
   λ: balance weight
```

**효과**: pseudo-label confidence 가 큰 unknown class 의 negative penalty 약화 → false negative 학습 보정.

→ 자세한 내용 LOSS_DESIGN.md 참조.

---

## 3. Loss Function 이론

### 3.1 Cross Entropy 의 limitation

```
L_CE = -Σ_c y_c × log(softmax(z)_c)
```

**Multi-label 문제**:
- softmax 는 mutually exclusive 가정 → second positive 신호 학습 안 됨
- 학습된 logit 은 top1 만 sharp, 나머지 noise

### 3.2 BCE 의 imbalance 문제

```
L_BCE = -Σ_c [y_c × log(σ(z_c)) + (1-y_c) × log(1-σ(z_c))]
```

**Class imbalance 문제**:
- 33-bit multi-hot 에서 평균 positive bit = 1.5 → 31.5 negative dominate
- BCE loss 의 99% 가 negative 의 easy negative 에서 발생
- positive 학습 신호 묻힘

### 3.3 Asymmetric Loss (ASL) 의 해결

```
L+_c = (1 - σ(z_c))^γ_pos × log(σ(z_c))
L-_c = (σ(z_c) - clip)_+^γ_neg × log(1 - σ(z_c) + clip)

   γ_pos = 1 (positive focus 약)
   γ_neg = 4 (negative focus 강)
   clip = 0.05
```

**Gradient 분석**:
- 쉬운 negative (p<clip) → loss 0 → gradient 0 → 무시
- 어려운 negative (p 큼) → strong down-weight
- 쉬운 positive (p 큼) → weak down-weight
- 어려운 positive (p 작음) → full weight

**수식적 직관**:
```
∂L+_c/∂z_c = γ_pos × (1-p)^(γ_pos-1) × p × (1-p) × log(p) - (1-p)^γ_pos × (1-p)
          ≈ -(1-p)^γ_pos × (1-p)    [γ_pos=1 일 때]
          = -(1-p)²

∂L-_c/∂z_c = γ_neg × (p-clip)^(γ_neg-1) × p × (1-p) × log(1-p+clip) + (p-clip)^γ_neg × p
          ≈ (p-clip)^γ_neg × p × γ_neg × log(1-p+clip)    [main term]
```

→ negative 의 gradient 가 (p-clip)^4 로 매우 sharp — 어려운 negative 만 학습.

→ 자세한 내용 LOSS_DESIGN.md 참조.

---

## 4. Calibration 이론

### 4.1 Calibrated Probability 의 정의

모델의 prob 출력이 calibrated 이면:
```
P(y = 1 | model predicts p) ≈ p
   예: p = 0.7 라고 모델이 예측한 sample 들의 실제 positive 비율은 0.7
```

→ **calibration = prob 가 진짜 confidence**.

### 4.2 Why neural networks are uncalibrated

Guo et al. (ICML 2017) 분석:
- modern neural network 은 over-confident
- batch normalization, deeper architecture → ECE 악화
- ImageNet ResNet-50 의 ECE ≈ 0.10

### 4.3 Calibration 방법

#### 4.3.1 Temperature Scaling
```
calibrated_p = sigmoid(z / T)
T 결정: val NLL minimize
```
- single hyperparameter
- global (모든 class 동일 T)
- 가장 simple, 가장 popular

#### 4.3.2 Platt Scaling (logistic regression)
```
calibrated_p = sigmoid(a × p + b)
```
- per-class (a_c, b_c)
- val set 으로 LogReg fit

#### 4.3.3 Isotonic Regression
```
calibrated_p = isotonic_fit(p, y, val)    # monotonic mapping
```
- non-parametric
- val 작으면 overfit

#### 4.3.4 Beta Calibration (Kull et al. 2017)
```
calibrated_p = sigmoid(a × log(p) + b × log(1-p) + c)
```
- Beta distribution 모델링
- Platt 의 일반화

→ 자세한 내용 DECISION_RULE.md 참조.

---

## 5. Density Estimation 이론

### 5.1 Histogram-based (Heatmap)

```
density(x, y) = count(x, y) / N
```
- non-parametric, simple
- discrete grid 의존

### 5.2 Kernel Density Estimation (KDE, Parzen 1962)

```
f̂(x, y) = (1 / N·h²) × Σ_i K((x - x_i) / h, (y - y_i) / h)

K(u, v) = (1 / 2π) × exp(-(u² + v²) / 2)    Gaussian kernel
```

**Bandwidth h 결정**:

#### 5.2.1 Silverman's rule of thumb (Silverman 1986)
```
h = (4σ⁵ / 3N)^(1/5) ≈ 1.06 × σ × N^(-1/5)
```
- analytical, fast
- Gaussian 가정 하 optimal

#### 5.2.2 Cross-validation
```
h* = argmin_h [-(1/N) × Σ_i log f̂_(-i)(x_i)]   # leave-one-out
```
- data-driven
- 시간 ↑

#### 5.2.3 Plug-in methods (Sheather-Jones 1991)
```
h 결정 위해 second derivative 추정 후 plug-in
```
- 학술 표준
- 구현 복잡

### 5.3 Gaussian Mixture Model (GMM)

```
f(x) = Σ_k π_k × N(x | μ_k, Σ_k)
```

**EM Algorithm** (Dempster, Laird, Rubin 1977):
```
E-step: 책임 (responsibility) 계산
   γ_{i,k} = π_k × N(x_i | μ_k, Σ_k) / Σ_j π_j × N(x_i | μ_j, Σ_j)

M-step: parameter update
   π_k = (1/N) × Σ_i γ_{i,k}
   μ_k = Σ_i γ_{i,k} × x_i / Σ_i γ_{i,k}
   Σ_k = Σ_i γ_{i,k} × (x_i - μ_k)(x_i - μ_k)^T / Σ_i γ_{i,k}

수렴 조건: log-likelihood 변화 < ε
```

**Component 수 K — BIC** (Schwarz 1978):
```
BIC = -2 × log_likelihood + k × log(N)

   k: parameter 수 (K × (1 + d + d(d+1)/2))
   N: sample 수
```

직관:
- 첫 항 = "데이터 잘 설명" (likelihood)
- 둘째 항 = "복잡도 페널티"
- minimum BIC 의 K = 적정.

### 5.4 Hybrid

```
hybrid(x, y) = α × heatmap_smooth(x, y) + (1-α) × gmm(x, y)
```
- 두 method 의 장점 결합
- α data-amount 의존 (적은 데이터 → α 작게, GMM 더)

→ 자세한 내용 MATCHING_DESIGN.md 참조.

---

## 6. Threshold Tuning 이론

### 6.1 Default 0.5 의 부적합성

```
Bayes optimal threshold:
   th* = P(y=0) / (P(y=0) + P(y=1)) × cost_FN / cost_FP

   class balanced + cost equal → th* = 0.5
   class imbalanced (positive rare) → th* < 0.5
```

→ multi-label 의 33-bit 에서 평균 positive ≈ 1.5/33 = 4.5% → th* ≈ 0.045 (per Bayes).

### 6.2 F1-optimal Threshold (Lipton et al. 2014)

```
F1 = 2P×R / (P+R)
F1 maximize 시 threshold ≈ F1*/2

   F1*: optimal F1 score
   조건: calibrated probability
```

→ per-class F1 sweep 의 이론 base.

### 6.3 Adaptive Threshold (Yan et al. 2025)

```
threshold(x, c) = α × IDF(c) + (1-α) × KNN_local(x, c)

   IDF(c) = log(N / N_c)    ← global signal
   KNN_local(x, c) = (이웃 K 중 class c positive 비율)    ← local signal
```

**효과**: per-instance per-class threshold — 가장 정교.

→ 자세한 내용 DECISION_RULE.md 참조.

---

## 7. Augmentation 이론 (multi-label 합성)

### 7.1 Mixup (Zhang et al. ICLR 2018)

```
x' = λ × x_a + (1-λ) × x_b
y' = λ × y_a + (1-λ) × y_b   (mixed target)
λ ~ Beta(α, α)    α=0.2
```

**우리 도메인 부적합**:
- palette idx 평균 의미 없음 (palette index 4.5 = ?)

### 7.2 CutMix (Yun et al. ICCV 2019)

```
x' = M ⊙ x_a + (1-M) ⊙ x_b   (M = patch mask)
y' = λ × y_a + (1-λ) × y_b
λ = patch area / total area
```

**우리 도메인 부적합**:
- chip grid 200×200 단위 안 맞으면 chip 경계 깨짐

### 7.3 Wafer-aware Multi-pattern Composition (★ 본 ablation)

```
combined_mask = OR over distributions of:
   chip_positions sampled from distribution's heatmap

each chip in combined_mask:
   assigned object = random sample from objects list
   render fail-bit pattern as standard
```

- 의미 단위 (chip) 합성
- chip grid + palette + fail-bit 모두 보존
- distribution-aware (heatmap mask 활용)

→ 자세한 내용 plan Stage 3 참조.

---

## 8. Conditional Random Field (CRF) 이론

### 8.1 CRF Energy Function (Lafferty et al. 2001)

```
E(L) = -Σ_i ψ_i(l_i) - Σ_{i,j ∈ N(i)} φ_{i,j}(l_i, l_j)

ψ_i: unary potential (chip i 의 single-cell 결정 score)
φ_{i,j}: pairwise potential (인접 chip 일관성 보너스)
```

**Inference**:
- argmin_L E(L) → 최적 라벨링
- 32×32 grid 작음 → simple message passing

### 8.2 Dense CRF (Krähenbühl & Koltun NeurIPS 2011)

```
모든 pair 고려 (not just N(i)):
   φ_{i,j}(l_i, l_j) = θ × δ(l_i = l_j) × exp(-||p_i - p_j||²/2σ²)
```

- 효율적 mean-field inference
- semantic segmentation 표준 (DeepLab 등)
- 우리 도메인은 32×32 작음 → dense CRF overkill (sparse 4-neighbor 충분)

→ 자세한 내용 MATCHING_DESIGN.md 참조.

---

## 9. 핵심 정리

본 문서는 4 영역 이론 통합:

1. **Multi-label classification** — single-label 과 차이, 6 metric, SPML setting
2. **Loss formulation** — CE / Focal / BCE / ASL / AdaGC 의 수식 + 직관
3. **Calibration** — Temperature / Platt / Isotonic 의 이론
4. **Density estimation** — Histogram / KDE / GMM 의 수식 + bandwidth/component 결정

각 deep-dive 는 별도 문서:
- Loss 단일 + mix → `LOSS_DESIGN.md`
- Density + ensemble + CRF → `MATCHING_DESIGN.md`
- Threshold + calibration + top-K → `DECISION_RULE.md`

논문 citation → `PAPERS.md`.

---

## 10. 참조

- 논문: `docs/multi-label/PAPERS.md`
- 사례: `docs/multi-label/EXAMPLES.md`
- stage motivation: `docs/multi-label/STAGES.md`
- plan: `~/.claude/plans/1-input-batch-hidden-patterson.md`
