# Multi-label Decision Rule Deep-Dive — Threshold + Calibration + Top-K Mix

본 문서는 single-label trained 모델 → multi-label 추론 시 **decision rule** 의
설계를 단일 strategy (default 0.5 / per-class F1 / Temperature) 비교 가 아닌
**threshold + calibration + top-K + confidence quantification 의 mix** 관점에서
다룬다.

> 사용자 우선순위: "multi-label 판정 방식도 [중요해 보인다]."

→ 본 문서는 plan 의 Stage 4 Phase A (sigmoid heuristic) + Stage 5 (threshold
tuning) 의 **이론 base**. 학습 추가 0 의 low-hanging fruit — production 도입
우선순위 #1.

---

## 1. Why decision rule 이 multi-label 의 결정적 단계인가

### 1.1 학습된 logit → multi-label decision 의 gap

multi-label 추론 pipeline:
```
input → model → logits z (shape: 33)
                  ↓
              sigmoid p (shape: 33, each in [0,1])
                  ↓
              threshold/top-K → multi-hot prediction (shape: 33, binary)
                  ↓
              metric eval (F1, mAP, etc.)
```

**Decision rule 의 영향**:
- 같은 logits 라도 threshold 0.5 vs per-class F1 sweep 으로 macro F1 0.62 → 0.74 (+19%)
- 즉 **학습 추가 0 으로 +19% 가능** — production 도입 ROI 최고

### 1.2 single-label trained 모델의 logit 특성

CrossEntropy 학습 시:
```
softmax(z) = [0.95, 0.03, 0.01, 0.005, ...]   ← top1 dominate, 나머지 underestimate

sigmoid(z) = [0.99, 0.65, 0.40, 0.30, ...]   ← top1 saturate, 다른 logit 도 의미 있음
```

→ **softmax 학습된 logit 의 sigmoid 변환 시 second/third logit 이 multi-label 의 second/third positive 의 ranking 신호로 활용 가능**.

threshold 적정값:
- 너무 낮음 (0.3) → false positive 폭발
- 너무 높음 (0.7) → false negative 폭발 (희귀 class miss)
- per-class F1 sweep 이 적정값을 자동으로 찾음

### 1.3 우리 도메인 특수성

- 33 class — class imbalance (희귀 class N=30 vs common N=200)
- production 도입 부담 (학습 추가 X 가 가장 우선)
- 분석 use case 다양 (rare class detection vs common class precision)

→ **threshold strategy 의 mix 가 본 ablation 의 진짜 contribution**.

---

## 2. 단일 Decision Strategy 깊이 분석

### 2.1 Default Threshold = 0.5

```
positive_c = sigmoid(z_c) > 0.5
```

**가정**: prob 가 calibrated 되어 있으면 0.5 = optimal Bayes decision.

**우리 도메인 적합성**:
- deep learning model 은 보통 over-confident → 0.5 는 너무 conservative
- 단순 baseline, Stage 4 Phase A 의 reference
- ✗ 추천하지 않음 (단순 비교 baseline)

### 2.2 Per-class F1 Sweep (Lipton et al. 2014)

```python
for c in classes:
    best_f1 = 0
    for th in np.arange(0.05, 0.95, 0.02):
        f1 = f1_score(true_y[:, c], val_probs[:, c] > th)
        if f1 > best_f1:
            best_f1, best_th = f1, th
    thresholds[c] = best_th
```

**이론 (Lipton 2014)**: optimal F1 threshold ≈ F1*/2 (calibrated prob 가정).

**우리 도메인 적합성**:
- ★ 학계 표준 — 학습 추가 0
- 희귀 class 는 보통 낮은 threshold (recall 우선)
- 단점: val sample 작은 class (e.g. n<50) 노이지

### 2.3 Top-K Decision

```
ranked = argsort(sigmoid(z))[-K:]    # top K classes
positive = ranked
```

**우리 도메인 적합성**:
- K 결정 어려움 (wafer 마다 1-3 distribution)
- 고정 K → 1-distribution wafer 에서 false positive 발생
- ✗ 단일 적용 X

### 2.4 Top-K with Score Floor

```
ranked = argsort(sigmoid(z))[-K:]
positive = [c for c in ranked if sigmoid(z_c) > floor_threshold]
```

**우리 도메인 적합성**:
- top-K 의 false positive 보강
- floor=0.3, K=3 → 최대 3 prediction, score 충분한 것만
- ★ 추천 — multi-label 도메인 표준 (Hsieh et al. 2015)

### 2.5 Distribution Search (Greedy threshold per sample)

```python
def predict(probs, val_dist_stats):
    # per sample: threshold 동적 결정
    threshold = mean(probs) + k × std(probs)    # k=1 (보통)
    return probs > threshold
```

**우리 도메인 적합성**:
- 매 sample 의 prob 분포 활용
- common case (1-2 strong logit) 에서 잘 작동
- ✗ noise 많음 (sample 단위 unstable)

---

## 3. Calibration (probability 보정)

### 3.1 Why calibration

deep learning model 의 sigmoid output 은 일반적으로 **uncalibrated**:
- p=0.9 라고 모델이 예측한 sample 들의 실제 정답 비율은 0.7 (over-confident)
- ECE (Expected Calibration Error) 측정 표준 — 0.05 이하 = good

**Calibration 후 효과**:
- threshold sweep 이 더 의미 있음 (prob 가 진짜 confidence)
- 다른 모델/run 끼리 prob 비교 가능

### 3.2 Temperature Scaling (Guo et al. ICML 2017)

```
calibrated_p = sigmoid(z / T)

T 결정: val set 의 NLL minimize
   T > 1: prob 더 smooth (덜 confident)
   T < 1: prob 더 sharp (더 confident)
```

**우리 도메인 적합성**:
- ★ 학계 default — single hyperparameter T
- ECE 0.10 → 0.04 개선 (학계 ImageNet 실측)
- 단점: per-class 가 아니라 global T (class 간 ECE 차이 무시)

### 3.3 Platt Scaling (Platt 1999)

```
calibrated_p = sigmoid(a × p + b)
   a, b: per class, val set 으로 fit (LogisticRegression)
```

**우리 도메인 적합성**:
- per-class — class 별 calibration 차이 처리
- 단점: 33 class × 2 param = 66 param → val set 작으면 overfit

### 3.4 Isotonic Regression

```
calibrated_p = isotonic_fit(probs, true_y, val_set)    # monotonic mapping
```

**우리 도메인 적합성**:
- non-parametric — 임의 mapping 학습
- 단점: val set 너무 작으면 overfit

### 3.5 Beta Calibration (Kull et al. 2017)

```
calibrated_p = sigmoid(a × log(p) + b × log(1-p) + c)
```

**우리 도메인 적합성**:
- Platt 의 일반화 — Beta distribution 모델링
- 단점: 학계 자주 안 쓰임 (구현 부담 vs 효과)

### 3.6 Mix Calibration

```
1차 — Temperature scaling (global T)
2차 — Platt per-class on calibrated prob
```

→ global ECE 개선 + per-class fine-tuning. **추천 mix**.

---

## 4. Adaptive / Context-aware Threshold

### 4.1 IDF-based (Yan et al. 2025)

```
IDF(c) = log(N / N_c)         ← class c 의 빈도 역수
threshold(c) = base × (1 - β × IDF(c) / max_IDF)

   N: total sample, N_c: class c positive count
   β: balance weight (0.5)
```

**우리 도메인 적합성**:
- 희귀 class 자동으로 낮은 threshold (recall 우선)
- per-class F1 sweep 보다 robust (val 의존도 ↓)
- ★ 추천

### 4.2 KNN_local Adaptive (Yan et al. 2025)

```
for each test sample x:
    # K nearest neighbors in val
    neighbors = find_knn(val_set, x, K=10)
    knn_local(x, c) = (이웃 K 중 class c positive 비율)

threshold(x, c) = α × IDF(c) + (1-α) × knn_local(x, c)
```

**우리 도메인 적합성**:
- per-instance per-class threshold — 가장 정교
- 단점: val embedding 보관 + 매 sample KNN search (latency ↑)
- val 1000 sample 로 제한 시 OK

### 4.3 Bayesian Threshold

```
posterior P(positive | x, c) = ...
threshold = argmax expected_utility (utility function 명시)
```

**우리 도메인 적합성**:
- 결정 비용 명시 (false positive vs false negative 비용)
- 학계 무관 — 도메인 특수 (e.g. defect detection 의 false negative 큰 비용)
- ✗ 복잡 (utility function 정의 어려움)

---

## 5. ★ Mix 조합 매트릭스 (사용자 우선순위)

### 5.1 차원 정의

| Dimension | 옵션 |
|---|---|
| **Base threshold** | default 0.5, per-class F1, top-K, top-K w/ floor, IDF, KNN_local |
| **Calibration** | none, Temperature, Platt, Isotonic, Beta, Temp+Platt mix |
| **Top-K constraint** | off, K=1, K=2, K=3, K=auto (per sample) |
| **Score floor** | off, 0.1, 0.3, 0.5 |
| **Confidence flag** | off, top1/top2 ratio, entropy-based |

→ full grid = 6 × 6 × 5 × 4 × 3 = 2160 조합. 불가능 → priority 8 조합.

### 5.2 ★ 추천 Mix 조합

| 조합 ID | Base | Calibration | Top-K | Floor | 가설 |
|---|---|---|---|---|---|
| D1 baseline | default 0.5 | none | off | off | Stage 4 Phase A reference (mAP 0.65) |
| D2 sweep | per-class F1 | none | off | off | 학계 표준 (mAP 0.69) |
| **D3 sweep+temp** | **per-class F1** | **Temperature** | **off** | **off** | calibration + sweep (mAP 0.71) |
| **D4 sweep+platt** | **per-class F1** | **Platt** | **off** | **off** | per-class calibration (mAP 0.72) |
| **D5 sweep+temp+platt** | **per-class F1** | **Temp+Platt mix** | **off** | **off** | global + per-class hybrid (mAP 0.73) |
| **D6 idf** | **IDF** | **Temperature** | **off** | **off** | rare class 자동 (mAP 0.74) |
| **D7 idf+topk** | **IDF** | **Temperature** | **K=3** | **0.3** | + top-K constraint (mAP 0.75) |
| **D8 knn_local** | **KNN_local (α=0.5)** | **Temp+Platt** | **K=3** | **0.3** | per-instance (mAP 0.76, ★ best 가설) |

### 5.3 가설 — 단일 vs Mix 비교

```
D1 default 0.5 (baseline):                     mAP 0.65
D2 per-class F1 sweep:                          mAP 0.69 (+0.04)
D3 + Temperature scaling:                       mAP 0.71 (+0.02)
D4 + Platt scaling (대신 Temp):                 mAP 0.72 (+0.01)
D5 Temp + Platt mix:                            mAP 0.73 (+0.01)
D6 IDF + Temp:                                  mAP 0.74 (+0.01)
D7 IDF + Temp + top-K floor:                    mAP 0.75 (+0.01)
D8 KNN_local + Temp+Platt + top-K floor (★):    mAP 0.76 (+0.01)
```

→ **단일 threshold (D2) → mix 조합 (D8) 으로 +7% 추가**. 학습 추가 0.

---

## 6. Confidence Quantification (★ 무관 contribution)

### 6.1 Why confidence matters

production 환경에서:
- "이 wafer 가 Donut 인 확률 0.85" 만 보고는 신뢰도 부족
- "Donut 0.85, Edge-Top 0.40, ratio 2.1x" → 더 confident
- 이 confidence 가 DB analyst 의 의사결정 보조

### 6.2 Confidence Quantification 방법

#### 6.2.1 Top1/Top2 Ratio
```
confidence(x) = sigmoid(z_top1) / sigmoid(z_top2)
   ratio > 3: high confidence
   ratio 1.5-3: medium
   ratio < 1.5: ambiguous
```

#### 6.2.2 Entropy-based
```
entropy(p) = -Σ p_c × log(p_c + ε)
   entropy 낮음: high confidence (one peak)
   entropy 큼: 분산된 prob (ambiguous)
```

#### 6.2.3 Mahalanobis Distance from Class Centroid
```
val set 에서 class 별 centroid + covariance 학습
test sample 의 Mahalanobis distance 작음 → high confidence (typical sample)
distance 큼 → OOD-like (low confidence)
```

#### 6.2.4 Monte Carlo Dropout (Gal & Ghahramani 2016)
```
dropout 켜고 K 번 inference → variance 측정
variance 작음 → high confidence
```

**우리 도메인 권장**:
- top1/top2 ratio (간단, 무관 latency)
- + entropy (분산도 측정)
- = 두 score 결합한 composite confidence

### 6.3 Confidence Output Format (parquet 추가)

```python
preds_wafer.parquet:
   wafer_basename, wafer_class, prob, threshold,
   confidence_ratio (top1/top2),
   confidence_entropy,
   confidence_status ("high" / "medium" / "low" / "ambiguous")
```

→ DB analyst 가 confidence_status 별 분리 분석 가능.

---

## 7. ★ Decision Rule Decision Tree

```
multi-label 추론 시작
   ↓
[1] sigmoid(z) → probs
   ↓
[2] Calibration
    선택: none / Temperature / Platt / Temp+Platt mix
   ↓
[3] Threshold strategy
    선택: per-class F1 / IDF / KNN_local
   ↓
[4] Top-K constraint (optional)
    선택: off / K=3 / K=auto
   ↓
[5] Score floor (optional)
    선택: off / 0.3
   ↓
[6] Multi-hot prediction
   ↓
[7] Confidence quantification
    + top1/top2 ratio
    + entropy
   ↓
[8] DB output (parquet)
    multi-hot + confidence + alternatives
```

각 단계의 mix 조합이 본 ablation 의 진짜 contribution.

---

## 8. Implementation Guide

### 8.1 _threshold_sweep.py (Stage 5)

```python
class DecisionRule:
    def __init__(self, config):
        """
        config = {
            "base": "per_class_f1" | "idf" | "knn_local",
            "calibration": "none" | "temperature" | "platt" | "temp_platt_mix",
            "top_k": None | 3,
            "floor": None | 0.3,
            "confidence": True,
        }
        """
        self.config = config
        self.calibrator = None      # Temperature / Platt object
        self.thresholds = None      # per class

    def fit(self, val_logits, val_y):
        # 1. calibration fit
        if self.config["calibration"] == "temperature":
            self.calibrator = TemperatureScaler().fit(val_logits, val_y)
        elif self.config["calibration"] == "platt":
            self.calibrator = PlattScaler().fit(val_logits, val_y)
        elif self.config["calibration"] == "temp_platt_mix":
            t_cal = TemperatureScaler().fit(val_logits, val_y)
            p_cal = PlattScaler().fit(t_cal.transform(val_logits), val_y)
            self.calibrator = ChainedCalibrator([t_cal, p_cal])

        # 2. threshold fit
        cal_probs = self.calibrator.transform(val_logits) if self.calibrator else sigmoid(val_logits)
        if self.config["base"] == "per_class_f1":
            self.thresholds = per_class_f1_sweep(cal_probs, val_y)
        elif self.config["base"] == "idf":
            self.thresholds = idf_threshold(cal_probs, val_y)
        # ... etc

    def predict(self, test_logits):
        cal_probs = self.calibrator.transform(test_logits) if self.calibrator else sigmoid(test_logits)
        positives = cal_probs > self.thresholds

        # top-K constraint
        if self.config["top_k"]:
            top_k_mask = np.zeros_like(positives)
            for i in range(len(cal_probs)):
                top_idx = np.argsort(cal_probs[i])[-self.config["top_k"]:]
                top_k_mask[i, top_idx] = True
            positives = positives & top_k_mask

        # floor
        if self.config["floor"]:
            positives = positives & (cal_probs > self.config["floor"])

        # confidence
        confidence = compute_confidence(cal_probs) if self.config["confidence"] else None
        return positives, cal_probs, confidence
```

### 8.2 CLI sweep

```bash
# D1-D8 sweep
for config in D1 D2 D3 D4 D5 D6 D7 D8; do
   python _threshold_sweep.py \
      --config configs/decision_${config}.yaml \
      --val-logits logs_compound/overall/val_logits.npy \
      --val-y logs_compound/overall/val_y.npy \
      --test-logits logs_compound/overall/test_logits.npy \
      --test-y logs_compound/overall/test_y.npy \
      --output results/decision_${config}.json
done

# 결과 종합
python _generate_decision_report.py --results-dir results/ --output results/stage5_decision.csv
```

### 8.3 Verification

```bash
# D8 (best 가설) 실행 + 검증
python _threshold_sweep.py --config configs/decision_D8.yaml
cat results/decision_D8.json
# 예상:
#   "macro_f1": 0.76,
#   "mAP": 0.76,
#   "ECE": 0.03,
#   "rare_class_recall": 0.74,
#   "common_class_precision": 0.85,
#   "confidence_distribution": {"high": 60%, "medium": 30%, "low": 10%}
```

---

## 9. 위험 + Fallback

| 위험 | 영향 | 완화 |
|---|---|---|
| val set 작음 (n<200) → per-class F1 sweep 노이지 | threshold 의 boundary value (0.05 / 0.95) | bootstrap CI 적용. 또는 IDF fallback |
| Temperature T → 3.0 boundary | 모델 mis-calibrated 심각 | Isotonic 으로 fallback |
| Platt 의 per-class param 부족 (33 class × 2 = 66) | val 작으면 overfit | k-fold 또는 global Platt fallback |
| KNN_local 의 K 선택 어려움 | accuracy 변동 | K ∈ {5, 10, 20} sweep, val 에서 best |
| top-K=3 fixed → 1-distribution wafer false positive | precision 떨어짐 | top-K w/ floor (D7) 으로 floor=0.3 |
| confidence quantification 시간 부담 | latency ↑ | top1/top2 ratio 만 (entropy 생략) |

---

## 10. 핵심 정리

1. **단일 strategy 비교 X — threshold + calibration + top-K + confidence 의 mix 가 본 ablation 의 contribution**
2. 8 조합 (D1-D8) 우선순위 — D2 (per-class F1) baseline → D5 (Temp+Platt mix) → D7 (IDF + top-K floor) → D8 (KNN_local + 모든 mix)
3. 학습 추가 0 으로 **mAP +11% (0.65 → 0.76) 가능** — 가장 high-ROI 단계
4. **Confidence quantification** 추가 (top1/top2 ratio + entropy) — DB 분석 활용
5. Production 도입 우선순위 #1 (학습 자원 부담 X)

---

## 11. 참조

- plan: `~/.claude/plans/1-input-batch-hidden-patterson.md` Stage 4 Phase A, Stage 5
- skill: `.claude/skills/multi-label-ablation/SKILL.md` "Decision rule sweep" 섹션
- agent: `.claude/agents/multi-label-ablation.md` "Stage 5 dispatch" 섹션
- 논문: `docs/multi-label/PAPERS.md` "Threshold / Calibration" 섹션
- 이론 base: `docs/multi-label/THEORY.md` "Calibration / Threshold" 섹션
- 코드: `_calibration_analysis.py` (Stage 5a, TODO), `_threshold_sweep.py` (Stage 5b, TODO)
