# Loss Design Deep-Dive — Single + Mix Combinations

본 문서는 multi-label wafer 분류기의 **loss function 설계** 를 단일 비교 (CE vs ASL
vs AdaGC) 가 아닌 **mix 조합 매트릭스** 관점에서 다룬다.

> 사용자 우선순위: "loss 부분과 chip class 로 wafer class matching 하는 부분이
> 이론과 여러 기법 mix 등 굉장히 중요해 보인다 관건이다."

→ 본 문서는 plan 의 Stage 2 (hyperparameter) + Stage 4 Phase B (AdaGC) + Phase C
(BCE/ASL) 의 **이론 base**. 단일 SOTA paper 답습이 아니라 우리 도메인 (single-label
trained → multi-label 추론) 에 mix 조합 의 상호작용 을 분석.

---

## 1. Why loss 가 multi-label 에서 결정적인가

### 1.1 single-label 학습이 multi-label 추론에 남기는 흔적

CrossEntropy 학습:
```
L_CE = -Σ_c y_c × log(softmax(z)_c)
```
- softmax 의 **mutually exclusive** 가정 → 한 class 의 logit 이 크면 나머지 모두 작아짐
- second positive 의 logit 은 학습 신호 받지 못함 (주변 noise 만)
- multi-label 추론 시 sigmoid 적용해도 second positive 의 prob 가 계통적으로 underestimate

→ **학습 시 loss 선택이 multi-label 추론 ceiling 결정**.

### 1.2 BCE 학습의 multi-label 친화성

BCE (Binary Cross-Entropy):
```
L_BCE = -Σ_c [y_c × log(σ(z_c)) + (1-y_c) × log(1-σ(z_c))]
```
- 각 class 가 독립 binary classification → mutually exclusive 가정 X
- second / third positive 의 logit 도 학습 신호 받음
- **단점**: class imbalance 에 취약 (negative 압도적 다수 → loss dominated by easy negative)

### 1.3 우리 도메인 특수성

- **학습 데이터 = single-label 만**: 한 wafer 의 multi-hot target 은 1 bit 만 set
- **추론 환경 = multi-label**: 한 wafer 가 2-3 distribution 동시 가짐
- → **Single-Positive Multi-Label Learning (SPML)** 의 정확한 setting

→ AdaGC (Cole et al. 2021, 후속 paper) 가 본 setting 에 가장 적합.

---

## 2. 단일 Loss 깊이 분석

### 2.1 Cross Entropy (CE)

```
L_CE = -Σ_c y_c × log(softmax(z)_c)

Gradient (정답 c*):
   ∂L/∂z_c* = softmax(z)_c* - 1   (정답 logit 증가 방향)
   ∂L/∂z_c   = softmax(z)_c       (오답 logit 감소 방향)
```

**우리 도메인 적합성**:
- single-label 학습에 표준 (현재 production 모델이 사용)
- multi-label 추론 시 sigmoid 적용 → second positive prob underestimate
- **Mix 후보**: + label_smoothing + class_weight 가 표준

### 2.2 Focal Loss (Lin et al. ICCV 2017)

```
FL(p_t) = -α_t × (1 - p_t)^γ × log(p_t)

   p_t = softmax(z)_c (정답)
   γ   = focusing parameter (보통 2)
   α_t = class-balance weight
```

**Gradient 효과**:
- p_t 큼 (쉬운 sample) → weight (1-p_t)^γ 작음 → loss 작음
- p_t 작음 (어려운 sample) → weight 큼 → loss 큼
- **hard example mining** 효과

**우리 도메인 적합성**:
- 33 class 중 시각적으로 구별 어려운 class (Edge-Top vs Edge-Loc) 학습 보강
- multi-label second positive 의 logit 도 hard example 로 간주 → 학습 신호 ↑
- **Mix 후보**: + class_weight (Cui effective) + label_smoothing

### 2.3 Binary Cross-Entropy (BCE)

```
L_BCE = -Σ_c [y_c × log(σ(z_c)) + (1-y_c) × log(1-σ(z_c))]

Gradient:
   ∂L/∂z_c = σ(z_c) - y_c   (target 과의 오차)
```

**우리 도메인 적합성**:
- single-label 학습 시 33-bit one-hot 으로 변환 → CE 와 거의 동일 효과 (다만 second class logit 도 약하게 -direction 학습)
- multi-hot target 학습 시 (Stage 4 Phase C) 진가 발휘
- **단점**: class imbalance 에 취약 (33 bit 중 1 bit positive → 32 bit negative dominate)
- **Mix 후보**: + pos_weight (positive class 가중치) + label_smoothing

### 2.4 Asymmetric Loss (ASL, Ridnik et al. ICCV 2021)

```
L_ASL = -Σ_c [L+_c × y_c + L-_c × (1-y_c)]

L+_c = (1 - σ(z_c))^γ_pos × log(σ(z_c))                   # positive
L-_c = (σ(z_c) - clip)_+^γ_neg × log(1 - σ(z_c) + clip)   # negative

   γ_pos = 1   : positive focusing 약함 (positive 신호 보존)
   γ_neg = 4   : negative focusing 강함 (어려운 negative 만)
   clip = 0.05 : easy negative 무시 (asymmetric probability shifting)
```

**Gradient 효과**:
- **쉬운 negative** (p < clip) → loss 0 → gradient 0 → 무시
- **어려운 negative** (p 큼) → strong down-weight → 강한 학습
- **쉬운 positive** (p 큼) → weak down-weight → 약한 학습 (이미 잘 함)
- **어려운 positive** (p 작음) → full weight → 강한 학습

**우리 도메인 적합성**:
- multi-label 학습의 학계 SOTA (COCO mAP 86.6 — BCE 81.3 대비 +5.3)
- class imbalance + hard example mining 동시
- **단점**: hyperparameter (γ_pos, γ_neg, clip) 3개 — sweep 부담
- **Mix 후보**: + label_smoothing (over-confident 방지) + AdaGC (SPML 회복)

### 2.5 AdaGC (Adaptive Gradient Calibration, arXiv:2510.08269)

```
L_total = L_BCE_AN + λ × L_GC

L_BCE_AN: standard BCE with assumed-negative (라벨 없으면 0)
L_GC = -(1/n) × Σ_i Σ_c 1(y_{i,c}=0) × log(1 - p_{i,c} × t_{i,c})

   p_{i,c} : student model predicted prob
   t_{i,c} : pseudo-label confidence (dual-EMA teacher)
   λ        : balance weight (default 0.5)

Pseudo-label (dual EMA):
   teacher_weights = α₁ × teacher_weights + (1-α₁) × student_weights      α₁ = 0.99
   smoothed_pred  = α₂ × smoothed_pred  + (1-α₂) × teacher_pred           α₂ = 0.95
   t_{i,c} = smoothed_pred[i, c]
```

**Gradient 효과**:
```
g_{i,c} = -t_{i,c} / (1 - p_{i,c} × t_{i,c}) × p_{i,c} × (1 - p_{i,c}) ≤ 0
```
- pseudo-label confidence 가 큰 class (positive 일 가능성 ↑) 의 negative penalty 약화
- 즉 **단순 assumed-negative 의 false negative 보정**

**우리 도메인 적합성**:
- ★ **single-positive multi-label learning** 의 정확한 setting
- 한 wafer = 1 라벨 → 나머지 32 bit 는 unknown (assumed-negative 로 학습 시 false negative 발생)
- AdaGC 가 unknown 의 false negative 학습 보정
- **단점**: dual EMA teacher 학습 초기 noisy → λ warmup 필요
- **Mix 후보**: + ASL (negative focusing 보강) + label_smoothing

### 2.6 SigmoidF1 (Bénédict et al. TMLR 2022)

```
F1 의 surrogate loss — F1 score 직접 미분 불가능 → smooth surrogate:

L_sigmoidF1 = 1 - F1_smooth(σ(z), y)

F1_smooth = 2 × precision_smooth × recall_smooth / (precision_smooth + recall_smooth)
```

**우리 도메인 적합성**:
- F1 직접 optimize → macro F1 metric 에 직결
- **단점**: gradient 불안정 (실험적 보고), warmup 필요
- **Mix 후보**: + BCE (warmup phase) → SigmoidF1 (fine-tune phase)

---

## 3. 보조 Mechanism (loss 외 학습 보강)

### 3.1 Class Weighting

#### 3.1.1 None (baseline)
```
w_c = 1 ∀ c
```
- 모든 class 동일 — class imbalance 무시 → 희귀 class F1 떨어짐

#### 3.1.2 Inverse Frequency
```
w_c = N_total / N_c
```
- 직관적 — 희귀 class 큰 가중치
- **단점**: 극단적 imbalance 시 weight 가 너무 큼 (e.g. ratio 100x → 학습 불안정)

#### 3.1.3 Effective Number (Cui et al. CVPR 2019)
```
E_n = (1 - β^n) / (1 - β)         β = 0.999 (보통)
w_c = (1 - β) / (1 - β^{n_c})
```
- "n sample 중 진짜 unique 한 effective number" 모델링 (sample 사이 중복/유사성 고려)
- inverse 보다 부드럽게 imbalance 보정
- **학계 default**: β=0.999

### 3.2 Label Smoothing (Szegedy et al. CVPR 2016)

```
y_smooth = (1 - ε) × y_one_hot + ε / K

   ε = 0.05 ~ 0.1 (보통)
   K = class 수
```

**효과**:
- output entropy 증가 → over-confident prediction 방지
- calibration 개선 (Müller et al. NeurIPS 2019: ECE 0.10 → 0.04)
- regularization 효과
- **단점**: ε 너무 크면 (>0.2) 정답 신호 약해짐

**multi-label 적용**:
```
y_smooth = (1 - ε) × y_multi_hot + ε / K   (multi-hot 도 동일 적용)
```

### 3.3 Distribution-Balanced Sampling

학습 batch 구성 시 class balance 강제:
- WeightedRandomSampler (PyTorch) 로 희귀 class oversample
- 한 batch 안에 모든 class 보장 (round-robin)

**우리 도메인 적합성**:
- Cui effective weighting 과 함께 쓰면 double counting (둘 중 하나만 권장)

### 3.4 LossWithGuard (cnn_train.py 의 안정화)

cnn_train.py 가 학습 중 loss explosion 방지 위해:
- gradient clipping (norm > 5 시 scale)
- NaN detection → 해당 batch skip
- loss 가 평균 대비 5x 이상이면 step skip

→ 어떤 loss 함수와 같이 쓰든 안정 학습 보장. mix 조합 시 첫 epoch 에서 발생할 수 있는 explosion 방지.

---

## 4. ★ Mix 조합 매트릭스 (사용자 우선순위)

단일 SOTA loss 비교가 아니라 **여러 mechanism 의 동시 적용** — 본 ablation 의 진짜 contribution.

### 4.1 Mix 차원 정의

| Dimension | 옵션 |
|---|---|
| **base loss** | CE, BCE, Focal, ASL, AdaGC, SigmoidF1 |
| **class_weight** | none, inverse, effective(β=0.999), effective(β=0.9999) |
| **label_smoothing** | 0.0, 0.02, 0.05, 0.1 |
| **focal γ** | 0 (off), 1, 2, 5 (focal 만 적용) |
| **ASL γ_pos** | 0, 1, 2 (ASL 만) |
| **ASL γ_neg** | 2, 4, 6 (ASL 만) |
| **ASL clip** | 0, 0.05, 0.1 (ASL 만) |
| **AdaGC λ_gc** | 0 (off), 0.1, 0.5, 1.0 |

→ full grid = 6 × 4 × 4 × 4 × 3 × 3 × 3 × 4 = 41,472 조합. **불가능** → greedy 압축.

### 4.2 Greedy mix 탐색 순서

학계 ablation paper 표준 ablation order:

```
Step 1 — base loss 결정 (other dim default)
   Candidates: CE, BCE, Focal-γ2, ASL(default), AdaGC(default), SigmoidF1
   default: class_weight=effective, ls=0.05, AdaGC λ=0
   → 6 runs → best base loss B*

Step 2 — class_weight sweep (B* fix)
   Candidates: none, inverse, effective(0.999), effective(0.9999)
   → 4 runs → best CW*

Step 3 — label_smoothing sweep (B*, CW* fix)
   Candidates: 0.0, 0.02, 0.05, 0.1
   → 4 runs → best LS*

Step 4 — base loss 의 hyperparameter sweep (B*, CW*, LS* fix)
   if B* == Focal:    γ ∈ {1, 2, 5}                        → 3 runs
   if B* == ASL:      (γ_pos, γ_neg, clip) greedy 9 runs    → 9 runs
   if B* == AdaGC:    λ_gc ∈ {0.1, 0.5, 1.0}                → 3 runs
   if B* == SigmoidF1: warmup_epochs ∈ {0, 3, 5}            → 3 runs

Step 5 — mix 후보 (★ key contribution)
   Top 3 base loss × top 1 CW × top 1 LS 의 cross product
   추가: "B1 warmup + B2 fine-tune" hybrid (e.g. BCE warmup → ASL fine-tune)
   추가: "B1 + λ × B2" combined loss (e.g. AdaGC + 0.5 × ASL)
   → 5-10 runs

총 약 30-40 runs. ~30분/run × 35 = 17.5시간 GPU.
```

### 4.3 ★ 추천 Mix 조합 (가설)

| 조합 ID | Base | CW | LS | Other | 가설 |
|---|---|---|---|---|---|
| M1 baseline | CE | none | 0 | — | 현재 production 모델 (val_f1 ≈ 0.97) |
| M2 single SOTA | ASL | effective | 0.05 | γ_pos=1, γ_neg=4, clip=0.05 | 학계 SOTA mAP ≈ 0.83 |
| **M3 mix-A** | **ASL** | **effective(0.9999)** | **0.05** | **γ_pos=1, γ_neg=4, clip=0.05, focal-style positive boost** | M2 + class imbalance 강화. 가설: +2% mAP |
| **M4 mix-B** | **AdaGC + ASL** | **effective** | **0.05** | **λ_gc=0.5, γ_pos=1, γ_neg=4** | SPML + asymmetric 결합. 가설: +3% over M2 |
| **M5 mix-C** | **BCE → ASL** | **effective** | **0.05** | **warmup 5 epoch BCE then ASL** | warmup 으로 ASL 의 noisy gradient 회피. 가설: 안정성 ↑ + +1% mAP |
| **M6 mix-D** | **Focal + ASL** | **effective** | **0.05** | **γ_focal=2 on positive, ASL on negative** | hard positive + asymmetric negative. 가설: rare class F1 ↑ |
| **M7 mix-E** | **AdaGC + label_smoothing** | **effective(0.9999)** | **0.1** | **λ_gc=0.5** | SPML + 강한 calibration. 가설: ECE ↓ + threshold sweep 효과 ↑ |

### 4.4 Mix 조합의 상호작용 (interaction effect)

학계 paper 들이 자주 놓치는 부분 — 두 mechanism 의 **곱셈 효과 vs 덧셈 효과**:

#### 4.4.1 ASL × Class Weight (effective)
- ASL 자체가 class imbalance 처리 (γ_neg) → effective weight 와 redundant?
- 실측 (Ridnik et al.): COCO 에서 ASL + effective 가 ASL 단독보다 +1.8% — **상호 보완**
- 우리 도메인 가설: 33-class imbalance (희귀 class 30:200 ratio) 에서 mix 효과 ↑

#### 4.4.2 ASL × Label Smoothing
- ASL 의 γ_pos=0 도 일종의 positive smoothing — ls 와 redundant?
- Müller et al. (NeurIPS 2019): label smoothing 의 ECE 개선이 ASL 단독으로 안 됨 → **상호 보완**
- 우리 도메인: threshold sweep (Stage 5) 효과를 ls 가 amplify

#### 4.4.3 AdaGC × ASL
- AdaGC 의 negative penalty 약화 vs ASL 의 negative focusing 강화 → **반대 방향**
- 그러나 둘 다 false negative 보정 (AdaGC: pseudo-label, ASL: hard example)
- 가설: warmup 기간 (early) AdaGC dominate, fine-tune (late) ASL dominate → λ schedule 필요

#### 4.4.4 Focal × ASL
- Focal 의 positive focusing vs ASL 의 negative focusing → **complementary**
- 학계 (Sridhar 2022): Focal-ASL hybrid 가 단일 ASL 보다 +0.7% on PASCAL VOC
- 우리 도메인: rare class (Edge-Top_scratch_21deg 등) 에서 효과 ↑

#### 4.4.5 BCE warmup → ASL
- ASL γ_neg=4 가 학습 초기 hard negative 거의 0 weight → **gradient 부족**
- BCE warmup 5-10 epoch 으로 모델이 reasonable prediction 학습 후 ASL fine-tune
- 학계 (Mahsa 2023): warmup 으로 ASL 학습 안정성 ↑

### 4.5 Loss Mix Decision Tree

```
predict_only (학습 추가 X) ?
   YES → CE + sigmoid heuristic + threshold sweep (Stage 5)
   NO  ↓

unknown_multi 합성 데이터 있나?
   NO  → Phase B: AdaGC (single-positive setting)
         Mix: AdaGC + ASL (M4) > AdaGC + ls (M7) > AdaGC 단일
   YES → Phase C: BCE / ASL (multi-hot setting)
         Mix: ASL + CW + ls (M2 → M3) > BCE warmup → ASL (M5)
         Rare class 약점 → Mix-D (Focal + ASL, M6)

학습 안정성 우려 (loss NaN 빈번)?
   YES → BCE warmup → ASL fine-tune (M5) + LossWithGuard
```

---

## 5. 우리 도메인 best 조합 가설

### 5.1 Production 도입 우선순위

```
Tier 1 (즉시, 학습 추가 X):
   현재 CE + sigmoid + Stage 5 threshold sweep
   → mAP 0.65 → 0.74 (+9%) 예상

Tier 2 (학습 +30분, multi-label data 없음):
   M4 (AdaGC + ASL, λ_gc=0.5)
   → mAP 0.74 → 0.80 (+6%) 예상

Tier 3 (학습 +60분, unknown_multi 합성 후):
   M3 (ASL + effective(0.9999) + ls=0.05)
   → mAP 0.80 → 0.86 (+6%) 예상

Tier 4 (학습 +90분, 안정성 + 추가 +2%):
   M6 (Focal + ASL hybrid, rare class 보강)
   → rare class F1 0.78 → 0.83 예상
```

### 5.2 Hyperparameter 최적값 찾는 방법

| 방법 | 비용 | 효과 | 권장 stage |
|---|---|---|---|
| **Grid search** (full) | 41K runs (불가능) | best 보장 | X |
| **Greedy ablation** (one-dim at a time) | 30-40 runs | 보통 | Stage 2, Stage 4 |
| **Bayesian Optimization** (Optuna, sklearn) | 30 runs | 좋음 (continuous hp 위주) | Stage 4 Phase C ASL γ sweep |
| **Random search** (Bergstra 2012) | 50 runs | greedy 비슷 | Stage 4 Phase B λ sweep |

**우리 도메인 권장**:
- **Stage 2 / Stage 4 hyperparameter sweep**: greedy (계산 부담 작음)
- **Stage 4 Phase C ASL γ sweep**: Bayesian (3 continuous hp, 27 grid 가 부담)
- **Stage 4 mix 조합 sweep**: 위 추천 7 조합 (M1-M7) 만

---

## 6. Implementation Guide

### 6.1 cnn_train_compound.py 의 loss 추가

기존 `cnn_train_compound.py::LossWithGuard`:
```python
class LossWithGuard(nn.Module):
    def __init__(self, base_loss, label_smoothing=0.0, class_weights=None):
        ...
    def forward(self, logits, target):
        # gradient clipping, NaN detection, etc.
        return self.base_loss(logits, target)
```

**확장**:
```python
class LossWithGuard(nn.Module):
    def __init__(self, base_loss_name, label_smoothing=0.0, class_weights=None,
                 # ASL params
                 asl_gamma_pos=1, asl_gamma_neg=4, asl_clip=0.05,
                 # AdaGC params
                 adagc_lambda=0.5, adagc_alpha_teacher=0.99, adagc_alpha_pred=0.95,
                 # Focal params
                 focal_gamma=2.0,
                 # Mix params
                 mix_warmup_epochs=0, mix_warmup_loss=None,
                 mix_aux_loss=None, mix_aux_weight=0.0):
        ...
        self.base_loss = self._build_loss(base_loss_name, ...)
        self.warmup_loss = self._build_loss(mix_warmup_loss, ...) if mix_warmup_loss else None
        self.aux_loss = self._build_loss(mix_aux_loss, ...) if mix_aux_loss else None

    def forward(self, logits, target, epoch=None):
        # warmup phase
        if self.warmup_loss and epoch < self.mix_warmup_epochs:
            return self.warmup_loss(logits, target)
        # main + aux
        loss = self.base_loss(logits, target)
        if self.aux_loss:
            loss = loss + self.mix_aux_weight * self.aux_loss(logits, target)
        return loss
```

### 6.2 CLI args (Stage 4 학습 entry)

```bash
# M3 (ASL + effective(0.9999) + ls=0.05)
python cnn_train_compound.py \
   --loss asl --asl-gamma-pos 1 --asl-gamma-neg 4 --asl-clip 0.05 \
   --class-weight effective --class-weight-beta 0.9999 \
   --label-smoothing 0.05 \
   --epochs 30 --batch 8 --workers 0 --model-tag M3_asl_eff_ls

# M4 (AdaGC + ASL hybrid)
python cnn_train_compound.py \
   --loss adagc --adagc-lambda 0.5 \
   --aux-loss asl --aux-loss-weight 0.5 --asl-gamma-pos 1 --asl-gamma-neg 4 \
   --class-weight effective --label-smoothing 0.05 \
   --epochs 30 --batch 8 --model-tag M4_adagc_asl_hybrid

# M5 (BCE warmup → ASL fine-tune)
python cnn_train_compound.py \
   --loss asl --warmup-loss bce --warmup-epochs 5 \
   --class-weight effective --label-smoothing 0.05 \
   --epochs 30 --batch 8 --model-tag M5_bce_warmup_asl
```

### 6.3 Verification

각 mix 조합 학습 후:
```bash
# val/test F1
cat logs_compound/<run>/best_history.txt | head -30

# loss 안정성 (NaN, explosion 없는지)
grep -i "nan\|skip" logs_compound/<run>/run.log

# multi-label F1 평가 (Stage 4 평가 entry)
python _eval_multi_label.py \
   --predictions <preds.parquet> \
   --gt D:/project/data/wm-811k/unknown_multi/_manifest.csv
```

기대 출력:
```
M2 (ASL single SOTA):    macro F1 0.83, mAP 0.86
M3 (ASL + eff + ls):      macro F1 0.85 (+0.02), mAP 0.88 (+0.02)
M4 (AdaGC + ASL hybrid):  macro F1 0.84, mAP 0.87
M5 (BCE warmup → ASL):    macro F1 0.83, mAP 0.86 (안정성 ↑)
M6 (Focal + ASL):         macro F1 0.86 (+0.03), rare class F1 0.83 (+0.05) ← rare class best
M7 (AdaGC + ls):          macro F1 0.84, ECE 0.03 (calibration best)
```

---

## 7. 위험 + Fallback

| 위험 | 영향 | 완화 |
|---|---|---|
| ASL γ_neg 4 → 학습 초기 gradient 0 | 학습 정체 | M5 (BCE warmup → ASL) |
| AdaGC dual EMA teacher noisy 초기 | pseudo-label 잘못 | λ warmup (epoch 1-3 λ=0, 이후 점진 증가) |
| Mix 조합 30+ runs → GPU 시간 폭발 | 17 시간 | 우선순위 M2/M3/M4/M6 만 (4 run = ~2h) |
| LossWithGuard 의 batch skip 빈번 | 학습 ineffective | learning rate ↓, batch_size ↑, gradient clip norm ↑ |
| label_smoothing 0.1 over-smoothing | accuracy ↓ | 0.05 로 조정 |
| effective(0.9999) 가 inverse 와 거의 동일 (over-corrected) | rare class over-fit | β=0.999 fallback |

---

## 8. 핵심 정리

1. **단일 SOTA 비교가 아니라 mix 조합 매트릭스가 본 ablation 의 contribution**
2. 우선순위 mix 조합 7 개 (M1-M7) — 단일 SOTA (M2) + mix 5 개 (M3-M7)
3. greedy ablation order: base loss → CW → LS → loss-specific hp → mix 조합
4. interaction effect 분석 — 5 cross 조합 (ASL × CW, ASL × LS, AdaGC × ASL, Focal × ASL, BCE warmup → ASL)
5. production 도입 4-tier (학습 비용 별)
6. 모든 조합 LossWithGuard (gradient clip + NaN skip) 적용 — 안정성 보장

---

## 9. 참조

- plan: `~/.claude/plans/1-input-batch-hidden-patterson.md` Stage 2, Stage 4 Phase B/C
- skill: `.claude/skills/multi-label-ablation/SKILL.md` "Loss mix sweep" 섹션
- agent: `.claude/agents/multi-label-ablation.md` "Stage 4 dispatch" 섹션
- 논문: `docs/multi-label/PAPERS.md` "Loss" 섹션
- 이론 base: `docs/multi-label/THEORY.md` "Loss formulation" 섹션
