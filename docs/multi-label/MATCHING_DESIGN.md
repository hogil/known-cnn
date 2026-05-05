# Chip-Wafer Matching Deep-Dive — Surface Ensemble + CRF + Consistency

본 문서는 multi-label compound 추론 시 **chip 단위 → wafer pattern 매칭** 의
설계를 단일 surface (heatmap / GMM / KDE / hybrid) 비교 가 아닌 **ensemble +
post-process + 의미 결합** 관점에서 다룬다.

> 사용자 우선순위: "chip class 로 wafer class matching 하는 부분이 이론과 여러
> 기법 mix 등 굉장히 중요해 보인다 관건이다."

→ 본 문서는 plan 의 Stage 1 (분포 학습) + Stage 6 (matching ablation) 의 **이론
base**. 학계가 잘 안 다루는 도메인-specific 부분 — chip object class 와 wafer
distribution pattern 의 의미적 결합.

---

## 1. Why chip-wafer matching 이 진짜 관건인가

### 1.1 multi-label 추론의 마지막 puzzle

compound 모델 결과:
```
wafer multi-label = ["Donut_scratch", "Edge-Bottom_scratch"]
                    ← 두 distribution + 한 chip object 의 결합
```

이 결과만으로는:
- ✗ **chip 단위 분석 불가** — 어느 chip 이 어느 패턴 일부인지 모름
- ✗ **outlier chip 검출 불가** — 두 패턴 모두 안 맞는 chip
- ✗ **품질 분석 어려움** — 어떤 패턴 영역에 어떤 chip object 우세

**chip-wafer matching = multi-label 추론의 마지막 puzzle piece**. SQL JOIN 으로
DB 분석 시 핵심 필드.

### 1.2 학계가 잘 안 다루는 부분

multi-label 학계 SOTA 들 (ASL, ML-Decoder, Q2L) 모두 **image-level multi-label**
까지만. **pixel/region level 어느 라벨에 속하는지** 는:
- semantic segmentation (per-pixel label) — 우리 도메인은 chip 단위라 segmentation 과 다름
- weakly-supervised localization (CAM 기반) — 우리는 chip 단위 GT 가 없음
- multi-instance multi-label learning (MIML) — chip 단위 instance 로 모델링 가능

→ MIML + heatmap surface 결합이 우리 도메인의 진짜 새로움.

### 1.3 우리 도메인 특수성 — 의미적 결합

```
wafer class 명 = (distribution × object) 의 결합
   "Donut_scratch"     = Donut 분포 × scratch chip object
   "Edge-Top_particle_blast" = Edge-Top 분포 × particle_blast chip object
```

→ **chip object class 정보 (chip CNN 결과) 가 matching 의 unary signal 로 활용 가능**:
```
wafer multi-label = ["Donut_scratch", "Edge-Top_scratch"]
chip CNN 예측     = "scratch"

→ 둘 다 "scratch" 를 포함 → consistency OK
→ 위치만으로 둘 중 어느 것인지 결정 (Donut surface vs Edge-Top surface)

만약 chip CNN 예측 = "particle_blast":
→ 둘 다 "particle_blast" 안 포함 → mismatch flag (별도 분석 필요)
```

이 **의미적 결합** 이 학계에 잘 안 다뤄짐 — 본 ablation 의 차별점.

---

## 2. 단일 Surface 깊이 분석

### 2.1 Heatmap (histogram-based)

```
density(x, y) = count(x, y) / N
```

**장점**:
- 즉시 학습 (counting), 어떤 분포 모양도 표현 가능
- 추론 instant (lookup)

**단점**:
- cell 독립 (인접 cell 정보 무시)
- 데이터 적으면 빈 cell 과다 (over-fit)
- noisy (outlier sensitive)

**우리 도메인**:
- full data (수천 chip) 시 충분
- 적은 데이터 시 (n=10) 빈 cell 70%+ → 일반화 떨어짐

### 2.2 Heatmap_smooth (histogram + Gaussian filter)

```
heatmap_smooth(x, y) = scipy.gaussian_filter(heatmap, σ=1.0)
```

**장점**:
- heatmap 의 noise 완화
- 인접 cell 정보 활용
- 적당한 smoothing 으로 일반화 ↑

**단점**:
- σ 결정 필요 (class 별 최적 다름)
- 너무 큰 σ → over-smoothing (Edge 띠 폭 흩뿌림)

**우리 도메인**:
- σ=1.0 default 적절
- ring 모양 (Donut, Edge-Ring) 에 효과적
- elongated (Edge-Top 띠) 에는 over-smoothing 가능 → σ=0.5 권장

### 2.3 Gaussian Mixture Model (GMM, sklearn)

```
f(x, y) = Σ_k π_k × N(x, y | μ_k, Σ_k)
```

**Component 수 K** = BIC sweep 후 minimum.

**장점**:
- parametric — 적은 데이터 (n=10) 도 robust
- covariance Σ 가 elongated 모양 (Edge 띠) 잘 표현
- mathematically well-defined (Mahalanobis distance 자연스럽게 derive)

**단점**:
- Gaussian 가정 — 비-가우시안 모양 (Donut ring) 표현 한계
- EM 의 random init → 비결정적
- BIC sweep 시간 (K=2~15 × 학습)

**우리 도메인**:
- Edge-Top, Edge-Bottom 등 elongated 패턴 best
- Donut (ring) 는 K=8+ 필요 → BIC penalty 로 K 작게 → 표현 부족

### 2.4 Kernel Density Estimation (KDE, sklearn)

```
f̂(x, y) = (1 / N·h²) × Σ_i K((x - x_i) / h, (y - y_i) / h)
```

**Bandwidth h** = Silverman's rule, cross-validation, plug-in methods.

**장점**:
- non-parametric — 어떤 분포 모양도 표현 (Donut ring 자연 표현)
- smooth boundary
- 적은 데이터도 reasonable surface 생성

**단점**:
- bandwidth h 결정 어려움 (class 별 다름)
- 추론 비용 큼 (모든 데이터 포인트 보관 + 매번 합산)
- high-dim 시 curse of dimensionality (우리는 2D 라 OK)

**우리 도메인**:
- Donut, CommaCluster, Starburst 같은 비-가우시안 모양 best
- bandwidth h ≈ 1.5 default 적절

### 2.5 Hybrid (Mixture of Methods)

```
hybrid(x, y) = α × heatmap_smooth(x, y) + (1-α) × gmm(x, y)
```

**장점**:
- heatmap 의 정확한 빈도 + GMM 의 smooth 일반화
- α 로 balance 조정 가능
- Stage 1 의 default 추천

**단점**:
- α 결정 필요 (data-amount 의존)
- 두 method 학습 + 합성 시간 (heatmap, GMM 둘 다 학습)

**우리 도메인**:
- α = 0.5 default (heatmap 빈도 + GMM smooth 균등)
- 적은 데이터 (n<30) 시 α=0.3 (GMM 의존 ↑)
- 많은 데이터 (n>200) 시 α=0.7 (heatmap 정확도 ↑)

---

## 3. ★ Surface Ensemble (사용자 우선순위)

단일 surface 결정이 아니라 **여러 surface 의 weighted ensemble** — 본 ablation 의 진짜 contribution.

### 3.1 Ensemble 차원

| Dimension | 옵션 |
|---|---|
| **Surface methods** | heatmap, heatmap_smooth, GMM, KDE, hybrid |
| **Ensemble weights** | uniform (1/N), validation-tuned, learned |
| **Aggregation** | weighted sum, max, geometric mean, log-sum-exp |
| **Temperature** | softmax temperature for sharp/smooth decision |

### 3.2 Ensemble 알고리즘 후보

#### 3.2.1 Linear Weighted Sum
```
ensemble(x, y, c) = Σ_m w_m × surface_m(x, y, c)
```
- weights w_m 결정: val set 의 matching accuracy maximize
- **장점**: simple, interpretable
- **단점**: surface 들 사이 scale 차이 (heatmap 1e-3 vs GMM 1e-1) → normalize 필요

#### 3.2.2 Geometric Mean (Bayesian product)
```
ensemble(x, y, c) = (Π_m surface_m(x, y, c) ^ w_m) ^ (1 / Σ w_m)
```
- 모든 surface 가 동의해야 score 큼
- **장점**: outlier robust (한 surface 만 큰 값이면 전체 작아짐)
- **단점**: surface 가 0 일 때 전체 0 (smoothing 필요)

#### 3.2.3 Log-Sum-Exp
```
ensemble(x, y, c) = log(Σ_m exp(w_m × log(surface_m(x, y, c))))
```
- soft maximum — 가장 큰 surface 가 dominate but 다른 surface 도 contribute
- **장점**: numerically stable
- **단점**: hyperparameter (LSE temperature)

#### 3.2.4 Stacking (meta-learner)
```
val set 으로 logistic regression 학습:
   y_chip = LogReg(surface_1(x, y), surface_2(x, y), ..., chip_obj_one_hot)
```
- 가장 정교 — surface 결과를 feature 로 메타 모델
- **장점**: surface 들의 의존성 학습
- **단점**: meta-train data 필요 (val set 사용)

### 3.3 ★ 추천 Ensemble 조합

| 조합 ID | 구성 | 가설 |
|---|---|---|
| E1 single | heatmap_smooth (Stage 6 단일 best) | baseline |
| E2 hybrid | 0.5 × heatmap_smooth + 0.5 × GMM (Stage 1 default) | 단일 hybrid |
| **E3 wide** | **0.4 × heatmap_smooth + 0.4 × GMM + 0.2 × KDE** | 3-method weighted, 다양성 ↑ |
| **E4 geo** | **(heatmap_smooth × GMM × KDE) ^ (1/3)** | geometric mean — outlier robust |
| **E5 stacking** | **LogReg(heatmap_smooth, GMM, KDE) on val** | 학습된 weight |
| **E6 class-adaptive** | **per-class best surface** (e.g. Donut→KDE, Edge→GMM, Center→heatmap) | per-class 최적화 |

### 3.4 ★ Ensemble Hyperparameter 결정

```
Step 1 — surface 단일 비교 (Stage 6.1, 5 surface)
   → 각 surface 별 matching accuracy 측정

Step 2 — uniform 2-method ensemble (10 조합)
   heatmap+GMM, heatmap+KDE, ..., KDE+hybrid
   → best 2-method combination

Step 3 — weighted 3-method (E3 후보)
   weight grid (0.2, 0.3, 0.4, 0.5) × 3 method = 64 combination → greedy 5

Step 4 — geometric mean (E4)
   smoothing parameter ε ∈ {1e-7, 1e-5, 1e-3}

Step 5 — stacking meta-learner (E5)
   val set 으로 LogReg 학습 → test 평가

Step 6 — per-class adaptive (E6)
   per-class accuracy 비교 → class 별 best surface 결정
   33 class × 5 surface = 165 cell heatmap 분석
```

### 3.5 Ensemble 가설 — 학계 reference

| 분야 | Ensemble 효과 |
|---|---|
| **Random Forest vs Decision Tree** | bagging +5-10% accuracy |
| **Image segmentation** (DeepLabV3+ vs DeepLabV3+ ensemble) | +1-2% mIoU |
| **Stacked Generalization** (Wolpert 1992) | +2-5% over best base learner |
| **GMM + KDE ensemble** (uncommon) | 학계 데이터 거의 없음 — 본 ablation contribution |

→ 우리 도메인 가설: **3-method weighted (E3) 가 단일 hybrid (E2) 대비 +2-3% matching accuracy**.

---

## 4. ★ CRF Post-Processing (인접 chip 일관성)

### 4.1 Why CRF — surface ensemble 만으로 부족한 이유

surface ensemble = **independent unary** decision (chip i 의 결정이 chip j 와 독립).

문제:
- 인접 chip 둘 (i, j) 이 **다른 wafer pattern** 으로 매칭되면 도메인 부자연스러움
- 같은 패턴 영역의 chip 들은 같은 wafer class 로 매칭되어야

→ **pairwise potential** 추가 (CRF, Conditional Random Field).

### 4.2 CRF 수식 (Lafferty et al. 2001)

```
E(L) = -Σ_i ψ_i(l_i) - Σ_{i,j ∈ N(i)} φ_{i,j}(l_i, l_j)

ψ_i: unary potential (chip i 의 surface ensemble score)
φ_{i,j}: pairwise potential
   φ_{i,j}(l_i, l_j) = θ × δ(l_i = l_j)
   인접 chip 같은 라벨이면 양수 보너스 θ
```

**Inference**:
- argmin_L E(L) → 최적 라벨링
- 우리 도메인 32×32 chip grid (1024 cell, 4 neighbor each) → simple message passing 가능
- 또는 dense CRF (Krähenbühl & Koltun NeurIPS 2011) — 모든 pair 고려

### 4.3 Pairwise potential 변형

| Variant | 수식 | 효과 |
|---|---|---|
| **constant** | φ = θ × δ(l_i = l_j) | simple smoothness |
| **distance-aware** | φ = θ × exp(-d/σ) × δ(l_i = l_j) | 가까운 chip 더 강한 보너스 |
| **feature-aware** | φ = θ × similarity(feature_i, feature_j) × δ(l_i = l_j) | chip object class similarity 고려 |
| **learned** | φ = NN(feature_i, feature_j) | meta-learned pairwise |

**우리 도메인 권장**: **feature-aware** — chip object class 가 같으면 보너스 ↑
```
φ_{i,j} = θ × δ(l_i = l_j) × δ(chip_obj_i = chip_obj_j)
```

### 4.4 Implementation — pydensecrf

```python
import pydensecrf.densecrf as dcrf

d = dcrf.DenseCRF2D(32, 32, n_classes)
unary = -np.log(ensemble_surface + 1e-7)    # (n_classes, 32*32)
d.setUnaryEnergy(unary.reshape(n_classes, -1))

# pairwise smoothness (Gaussian, distance-aware)
d.addPairwiseGaussian(sxy=1, compat=3)

# pairwise feature (chip_obj class)
d.addPairwiseBilateral(sxy=2, srgb=10, rgbim=chip_obj_grid, compat=10)

Q = d.inference(5)    # 5 iterations
labels = np.argmax(Q, axis=0).reshape(32, 32)
```

**우리 도메인 적합성**:
- 32×32 grid 작음 → fast
- chip_obj class 정보 활용 (bilateral pairwise)
- 학계 segmentation 표준 — 안정적

---

## 5. ★ Consistency Heuristic (의미 결합)

### 5.1 chip object × wafer class 의 의미적 결합

```
wafer multi-label = ["Donut_scratch", "Edge-Top_particle_blast"]
chip CNN 예측     = "scratch"

structural decomposition:
   "Donut_scratch"  = ("Donut", "scratch")    ← obj match
   "Edge-Top_particle_blast" = ("Edge-Top", "particle_blast") ← obj mismatch

→ chip 은 distribution 만 결정 (obj 는 fixed by chip CNN):
   "Donut" surface vs "Edge-Top" surface 비교
   "Donut" surface score 큼 → 매칭 = "Donut_scratch"

만약 chip obj = "particle_blast":
   "Donut_scratch" obj mismatch
   "Edge-Top_particle_blast" obj match → 자동 매칭
```

### 5.2 Consistency Filter Algorithm

```python
def match_with_consistency(chip_x, chip_y, chip_obj, wafer_multi_labels, surfaces,
                           outlier_threshold=0.01, ratio_threshold=2.0,
                           use_crf=False, ensemble_method="E3"):
    # 1. obj-consistent candidates
    consistent = [wc for wc in wafer_multi_labels if chip_obj in wc]
    if not consistent:
        # mismatch — chip obj 가 wafer multi-label 어디에도 없음
        # → chip CNN 잘못 OR wafer CNN 잘못 OR 진짜 outlier chip
        best = max(wafer_multi_labels, key=lambda wc: surfaces[wc][chip_y, chip_x])
        return {"status": "mismatch", "matched": best}

    # 2. unary score from ensemble
    scores = {wc: ensemble_score(surfaces, wc, chip_x, chip_y, ensemble_method)
              for wc in consistent}

    # 3. outlier check
    if max(scores.values()) < outlier_threshold:
        return {"status": "outlier", "matched": None}

    # 4. ambiguity check (top1 vs top2 ratio)
    sorted_s = sorted(scores.values(), reverse=True)
    if len(sorted_s) >= 2 and sorted_s[0] / max(sorted_s[1], 1e-7) < ratio_threshold:
        # top2 close → ambiguous
        best = max(scores, key=scores.get)
        return {"status": "ambiguous", "matched": best, "alternatives": list(scores.keys())}

    # 5. CRF post-process (optional, batch level)
    if use_crf:
        # collected over all chips of wafer, then apply CRF
        pass

    # 6. final decision
    best = max(scores, key=scores.get)
    return {"status": "ok", "matched": best, "score": scores[best]}
```

### 5.3 Status 의 의미

| Status | 의미 | DB 분석 시 활용 |
|---|---|---|
| `ok` | 정확 매칭 (top1 score 명확) | regular row, JOIN 안전 |
| `mismatch` | chip obj 가 wafer multi-label 어디에도 없음 | wafer/chip 모델 disagreement, 별도 cluster |
| `outlier` | 모든 wafer class surface 낮음 | noise chip, 분석 제외 |
| `ambiguous` | top1, top2 score 비슷 | 양쪽 분석 (alternatives 활용) |

---

## 6. ★ Mix Decision Tree (총합)

```
chip 매칭 시작
   ↓
[1] Surface ensemble 계산
    (heatmap_smooth, GMM, KDE → E3 weighted)
   ↓
[2] Consistency filter
    (chip obj 가 wafer multi-label 안 있나?)
       NO  → mismatch flag
       YES ↓
[3] Score 계산 (ensemble surface)
   ↓
[4] Outlier check
    (max score < 0.01?)
       YES → outlier flag
       NO  ↓
[5] Ambiguity check
    (top1 / top2 < 2?)
       YES → ambiguous flag (alternatives 보존)
       NO  ↓
[6] CRF post-process (optional, batch level)
    (인접 chip 같은 라벨로 강제)
   ↓
[7] Final 매칭 결과
```

각 step 의 조합이 **본 ablation 의 진짜 contribution**.

---

## 7. Mix 조합 매트릭스

### 7.1 차원

| Dimension | 옵션 |
|---|---|
| **Surface ensemble** | E1 (single heatmap_smooth), E2 (hybrid), E3 (3-method weighted), E4 (geo), E5 (stacking), E6 (class-adaptive) |
| **CRF** | off, constant pairwise, distance-aware, feature-aware (obj class) |
| **Consistency** | strict (obj in wc), soft (obj similar to wc), off |
| **Outlier threshold** | 0.001, 0.01, 0.05, percentile-based (top 5%) |
| **Ambiguity threshold** | 1.5x, 2x, 3x ratio |

→ full grid = 6 × 4 × 3 × 4 × 3 = 864 조합. **불가능** → priority 조합 10 개.

### 7.2 ★ 추천 Matching 조합

| 조합 ID | Surface | CRF | Consistency | Outlier | 가설 |
|---|---|---|---|---|---|
| C1 baseline | E1 | off | off | 0.01 | Stage 6 단일 baseline |
| C2 hybrid | E2 | off | strict | 0.01 | 단일 hybrid + consistency |
| **C3 wide** | **E3 (3-method)** | **off** | **strict** | **0.01** | surface ensemble 만 |
| **C4 wide+CRF** | **E3** | **distance-aware** | **strict** | **0.01** | + CRF smooth |
| **C5 wide+feat-CRF** | **E3** | **feature-aware (obj)** | **strict** | **0.01** | + CRF chip obj similarity |
| **C6 stacking** | **E5** | **off** | **strict** | **percentile (top 5%)** | 학습된 ensemble + dynamic outlier |
| **C7 class-adaptive** | **E6** | **feature-aware** | **strict** | **percentile** | per-class best + CRF + dynamic |

### 7.3 가설 — 우리 도메인

```
C1 baseline (E1 heatmap_smooth):  matching accuracy 0.78
C2 hybrid + consistency:          0.82 (+0.04)  ← consistency 효과
C3 E3 ensemble + consistency:      0.85 (+0.03)  ← ensemble 효과
C4 + distance CRF:                  0.86 (+0.01)
C5 + feature CRF (chip obj):        0.87 (+0.01)  ← 학계 무관 도메인-specific
C6 stacking + percentile outlier:  0.88 (+0.01)
C7 class-adaptive (★ best):        0.89 (+0.01)  ← per-class 최적
```

**진짜 contribution**: C5/C7 (feature-aware CRF + per-class adaptive surface).
학계 잘 안 다루는 도메인-specific 결합.

---

## 8. Implementation Guide

### 8.1 _eval_chip_matching.py (Stage 6)

```python
def evaluate_matching(unknown_multi_root, surfaces_root, config):
    """
    config = {
        "surface_ensemble": "E3",        # E1..E7
        "ensemble_weights": [0.4, 0.4, 0.2],   # for E3
        "crf": "feature-aware",           # off, constant, distance, feature
        "consistency": "strict",
        "outlier_threshold": 0.01,
        "ambiguity_threshold": 2.0,
    }
    """
    surfaces = load_surfaces(surfaces_root, config["surface_ensemble"])
    # 5 surface 종류 모두 로드 (heatmap, heatmap_smooth, GMM, KDE, hybrid)

    rows = []
    for json_path in unknown_multi_root.glob("**/*.json"):
        wafer_chips = parse_chips(json_path)
        wafer_multi = json.load(json_path)["multi_labels"]

        chip_results = []
        for chip in wafer_chips:
            result = match_with_consistency(
                chip["x"], chip["y"], chip["true_object"],
                wafer_multi, surfaces, config,
            )
            chip_results.append({**chip, **result})

        # CRF post-process (batch level, all chips of one wafer)
        if config["crf"] != "off":
            chip_results = apply_crf(chip_results, config["crf"])

        rows.extend(chip_results)

    # metrics
    metrics = compute_metrics(rows)    # accuracy, outlier rate, etc.
    return rows, metrics
```

### 8.2 CLI sweep

```bash
# C1-C7 모두 실행
for config in C1 C2 C3 C4 C5 C6 C7; do
   python _eval_chip_matching.py \
      --config configs/matching_${config}.yaml \
      --surfaces-root _dist_heatmaps_per_class/ \
      --gt-root D:/project/data/wm-811k/unknown_multi/ \
      --output results/matching_${config}.json
done

# 결과 종합
python _generate_matching_report.py --results-dir results/ --output results/stage6_matching.csv
```

### 8.3 Verification

```bash
# C7 (best 가설) 실행 + 검증
python _eval_chip_matching.py --config configs/matching_C7.yaml
cat results/matching_C7.json
# 예상:
#   "matching_accuracy": 0.89,
#   "outlier_rate": 0.04,
#   "mismatch_rate": 0.06,
#   "ambiguous_rate": 0.04
#   "per_class_acc": {Donut: 0.92, Edge-Top: 0.85, ...}
```

---

## 9. 위험 + Fallback

| 위험 | 영향 | 완화 |
|---|---|---|
| Stage 1 surface 학습 부실 (chip 데이터 부족) | matching accuracy 낮음 | Stage 1 의 5 method 중 best fallback (data-amount 별) |
| chip object class GT 없음 (positions JSON 부재) | consistency filter X | wafer 단위 surface (33 class) 만 사용 |
| CRF 라이브러리 (pydensecrf) 설치 어려움 (Windows) | C4-C7 적용 불가 | constant pairwise 직접 구현 (numpy) |
| Stacking (E5) val set 부족 | meta-learner overfit | k-fold CV |
| per-class adaptive (E6) class 별 데이터 부족 | unstable | global default fallback |
| 모든 surface ensemble 0 → outlier 폭발 | dataset issue | smoothing ε=1e-7 추가 |

---

## 10. 핵심 정리

1. **단일 surface 비교 X — surface ensemble + CRF + consistency 의 mix 가 진짜 contribution**
2. ensemble 6 조합 (E1-E6) 우선순위 — 단일 (E1) → hybrid (E2) → 3-method weighted (E3) → geometric (E4) → stacking (E5) → class-adaptive (E6)
3. CRF post-process 4 variant — feature-aware (chip obj) 가 우리 도메인 best 가설
4. consistency heuristic — chip obj × wafer class 의 의미적 결합 (학계 무관 contribution)
5. matching 조합 7 (C1-C7) — C5/C7 best 가설 (matching accuracy 0.87-0.89)
6. status 4 종 (ok / mismatch / outlier / ambiguous) — DB 분석 활용

---

## 11. 참조

- plan: `~/.claude/plans/1-input-batch-hidden-patterson.md` Stage 1, Stage 6
- skill: `.claude/skills/multi-label-ablation/SKILL.md` "Matching ensemble sweep" 섹션
- agent: `.claude/agents/multi-label-ablation.md` "Stage 6 dispatch" 섹션
- 논문: `docs/multi-label/PAPERS.md` "Matching / CRF" 섹션
- 이론 base: `docs/multi-label/THEORY.md` "Density estimation" 섹션
- 코드: `_dist_learn_per_class.py` (Stage 1, 완료), `_eval_chip_matching.py` (Stage 6, TODO)
