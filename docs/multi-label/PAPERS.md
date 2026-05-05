# Papers Reference — Multi-label Wafer Classification

본 문서는 plan 에서 인용한 논문들의 **상세 contribution + 우리 도메인 적용** 을
카테고리별로 정리. 단순 citation list 가 아니라 **각 논문의 idea 가 본 ablation
의 어느 stage 에서 어떻게 활용되는지** 명시.

---

## 1. Multi-label Classification

### 1.1 Ridnik et al. (ICCV 2021) — "Asymmetric Loss For Multi-Label Classification"

**arXiv**: 2009.14119

**Contribution**:
- multi-label 분류용 Asymmetric Loss (ASL) 제안
- positive/negative gradient 비대칭 (γ_pos ≠ γ_neg)
- COCO mAP 86.6 (BCE 81.3 대비 +5.3)

**핵심 수식**:
```
L_ASL = -Σ_c [L+_c × y_c + L-_c × (1-y_c)]
L+_c = (1 - σ(z_c))^γ_pos × log(σ(z_c))
L-_c = (σ(z_c) - clip)_+^γ_neg × log(1 - σ(z_c) + clip)
```

**우리 도메인 적용**:
- **Stage 4 Phase C**: ASL 학습 (default γ_pos=1, γ_neg=4, clip=0.05)
- **LOSS_DESIGN.md M2**: ASL single SOTA baseline
- **LOSS_DESIGN.md M3**: ASL + effective(0.9999) + ls=0.05 (mix)
- **LOSS_DESIGN.md M4**: ASL + AdaGC hybrid

### 1.2 Cole et al. (CVPR 2021) — "Multi-Label Learning from Single Positive Labels"

**Contribution**:
- Single Positive Multi-Label Learning (SPML) 의 founding paper
- 학습 시 한 라벨만 알려진 환경의 문제 정의
- 각 image 의 multi-label GT 수집 비용 절감

**우리 도메인 적용**:
- 우리 setting = SPML 의 정확한 setting (single-label 합성 → multi-label 추론)
- **Stage 4 Phase B**: AdaGC 가 SPML 의 후속 paper

### 1.3 Verelst et al. (2024) — "Adaptive Gradient Calibration for SPML"

**arXiv**: 2510.08269

**Contribution**:
- AdaGC loss 제안 (BCE + λ × GC term)
- dual-EMA pseudo-label
- remote sensing image scene classification 에 적용

**핵심 수식**:
```
L_total = L_BCE_AN + λ × L_GC
L_GC = -(1/n) × Σ_i Σ_c 1(y_{i,c}=0) × log(1 - p_{i,c} × t_{i,c})

   t_{i,c} = dual EMA teacher pseudo-label
```

**우리 도메인 적용**:
- **Stage 4 Phase B**: AdaGC retraining (default λ_gc=0.5)
- **LOSS_DESIGN.md M4**: AdaGC + ASL hybrid
- **LOSS_DESIGN.md M7**: AdaGC + label_smoothing

### 1.4 Liu et al. (ICML 2023) — "Revisiting Pseudo-Label for SPML"

**Contribution**:
- SPML 의 pseudo-labeling 방식 재분석
- entropy-based pseudo-label confidence

**우리 도메인 적용**:
- **LOSS_DESIGN.md**: AdaGC pseudo-label 의 alternative — entropy-based 도입 가능 (future)

### 1.5 Bénédict et al. (TMLR 2022) — "SigmoidF1: A Smooth F1 Score Surrogate Loss"

**Contribution**:
- F1 score 직접 optimize 하는 smooth surrogate loss
- Bénédict score → SigmoidF1

**우리 도메인 적용**:
- **LOSS_DESIGN.md**: 후보 base loss (BCE warmup → SigmoidF1 fine-tune mix)
- 단점: gradient 불안정 보고 → warmup 필수

### 1.6 Verma et al. (ICML 2021) — "Graph Transformer for SPML"

**Contribution**:
- graph-based label dependency 학습
- transformer architecture 활용

**우리 도메인 적용**:
- 직접 적용 X (architecture-level)
- **future**: label co-occurrence graph 학습 시 참조

### 1.7 Ben-Baruch et al. (2021) — "ML-Decoder"

**Contribution**:
- multi-label-specific decoder (transformer-based)
- query 형식 multi-label decoding

**우리 도메인 적용**:
- 직접 적용 X (architecture)
- **future**: decoder 변경 ablation 시 참조

---

## 2. Threshold Optimization

### 2.1 Lipton et al. (ECML PKDD 2014) — "Optimal Thresholding to Maximize F1"

**Contribution**:
- F1 maximize 의 optimal threshold ≈ F1*/2 (calibrated prob 가정)
- per-class threshold sweep 의 이론 base

**우리 도메인 적용**:
- **Stage 5 D2**: per-class F1 sweep
- **DECISION_RULE.md**: D2 strategy

### 2.2 Yang & Yu (2015) — "F1-Optimal Thresholding in Multi-Label Setting"

**Contribution**:
- multi-label 환경의 per-class F1 sweep 알고리즘 명시
- threshold 결정의 sample-wise 변동 분석

**우리 도메인 적용**:
- **Stage 5 _threshold_sweep.py**: 알고리즘 reference

### 2.3 Yan et al. (2025) — "Adaptive Thresholding via Global-Local Signal Fusion"

**arXiv**: 2505.03118

**Contribution**:
- IDF (global) + KNN_local (local) 의 fusion threshold
- per-instance per-class threshold
- COCO/VOC mAP +2-4%

**핵심 수식**:
```
threshold(x, c) = α × IDF(c) + (1-α) × KNN_local(x, c)

IDF(c) = log(N / N_c)         ← global signal
KNN_local(x, c) = (이웃 K 중 class c positive 비율)    ← local signal
```

**우리 도메인 적용**:
- **Stage 5 D6**: IDF threshold
- **Stage 5 D8**: KNN_local + Temp+Platt + top-K floor (★ best 가설)
- **DECISION_RULE.md**: D6, D8

### 2.4 Pillai et al. (Pattern Recognition 2013) — "Threshold optimisation for multi-label classifiers"

**Contribution**:
- 학계 multi-label threshold 비교 분석
- F1 / accuracy / AUC 의 trade-off

**우리 도메인 적용**:
- **DECISION_RULE.md**: threshold 비교 framework reference

---

## 3. Calibration

### 3.1 Guo et al. (ICML 2017) — "On Calibration of Modern Neural Networks"

**Contribution**:
- modern NN 의 over-confident 분석
- Temperature scaling 의 효과 증명 (ECE 0.10 → 0.04)

**핵심 알고리즘**:
```
calibrated_p = sigmoid(z / T)
T 결정: val NLL minimize
```

**우리 도메인 적용**:
- **Stage 5 D3**: Temperature scaling
- **Stage 5 D8**: Temp+Platt mix
- **DECISION_RULE.md**: 모든 calibration variant 의 base

### 3.2 Platt (1999) — "Probabilistic Outputs for SVMs"

**Contribution**:
- logistic regression 으로 sigmoid output 보정
- per-class fitting

**우리 도메인 적용**:
- **Stage 5 D4**: Platt scaling
- **DECISION_RULE.md**: per-class calibration

### 3.3 Niculescu-Mizil & Caruana (ICML 2005) — "Predicting Good Probabilities with Supervised Learning"

**Contribution**:
- Isotonic regression 의 calibration 효과 증명
- non-parametric mapping

**우리 도메인 적용**:
- **DECISION_RULE.md**: Isotonic 후보

### 3.4 Müller et al. (NeurIPS 2019) — "When Does Label Smoothing Help?"

**Contribution**:
- label smoothing 의 calibration 효과 분석
- ECE 개선 + over-confident 방지

**우리 도메인 적용**:
- **Stage 2c**: label_smoothing sweep
- **LOSS_DESIGN.md**: ls 0.05 default

### 3.5 Kull et al. (2017) — "Beta Calibration"

**Contribution**:
- Beta distribution 모델링한 calibration
- Platt 의 일반화

**우리 도메인 적용**:
- **DECISION_RULE.md**: Beta calibration 후보 (학계 자주 안 쓰임 → 낮은 priority)

---

## 4. Density Estimation

### 4.1 Parzen (1962) — "On Estimation of a Probability Density Function and Mode"

**Contribution**:
- KDE 의 founding paper
- non-parametric density estimation

**우리 도메인 적용**:
- **Stage 1 KDE**: chip 좌표 분포 학습
- **MATCHING_DESIGN.md**: KDE surface

### 4.2 Silverman (1986) — "Density Estimation for Statistics and Data Analysis"

**Contribution**:
- Silverman's rule of thumb (bandwidth 자동 결정)
- KDE 의 학술 표준 정리

**우리 도메인 적용**:
- **Stage 1**: bandwidth h=1.5 default
- **MATCHING_DESIGN.md**: KDE bandwidth 결정

### 4.3 Dempster, Laird, Rubin (1977) — "Maximum likelihood from incomplete data via the EM algorithm"

**Contribution**:
- EM algorithm 의 founding paper
- GMM 학습 표준

**우리 도메인 적용**:
- **Stage 1 GMM**: EM 학습
- **MATCHING_DESIGN.md**: GMM surface

### 4.4 Schwarz (1978) — "Estimating the Dimension of a Model"

**Contribution**:
- BIC (Bayesian Information Criterion)
- model selection (component K 결정)

**우리 도메인 적용**:
- **Stage 1 GMM**: BIC sweep [2, 15] 으로 K 결정
- **MATCHING_DESIGN.md**: GMM component K

### 4.5 Vassilev (2018) — "Comparison of GMM, KDE and Histogram for spatial data"

**Contribution**:
- 3 method 의 실증 비교
- data-amount 별 best method

**우리 도메인 적용**:
- **MATCHING_DESIGN.md**: 5 surface 비교 framework

---

## 5. Loss Functions (general)

### 5.1 Cui et al. (CVPR 2019) — "Class-Balanced Loss Based on Effective Number of Samples"

**Contribution**:
- Effective Number weighting (β=0.999)
- inverse 보다 부드러운 imbalance 보정

**핵심 수식**:
```
E_n = (1 - β^n) / (1 - β)
weight_c = (1 - β) / (1 - β^{n_c})
```

**우리 도메인 적용**:
- **Stage 2a**: class_weight=effective sweep
- **LOSS_DESIGN.md**: M2-M7 모두 적용

### 5.2 Lin et al. (ICCV 2017) — "Focal Loss for Dense Object Detection"

**Contribution**:
- Focal Loss (1 - p_t)^γ
- hard example mining

**핵심 수식**:
```
FL(p_t) = -(1 - p_t)^γ × log(p_t)    γ=2 default
```

**우리 도메인 적용**:
- **Stage 2c**: loss=focal γ=2 sweep
- **LOSS_DESIGN.md M6**: Focal + ASL hybrid

### 5.3 Szegedy et al. (CVPR 2016) — "Rethinking the Inception Architecture"

**Contribution**:
- Label Smoothing (1-ε one-hot + ε/K)
- regularization + over-confident 방지

**우리 도메인 적용**:
- **Stage 2b**: label_smoothing sweep
- **LOSS_DESIGN.md**: 모든 mix 조합에 적용

### 5.4 Wang et al. (ICCV 2017) — "Class-Balanced Loss for Multi-label Image Classification"

**Contribution**:
- multi-label 도 class-balanced logic 적용
- positive/negative imbalance 분리

**우리 도메인 적용**:
- **LOSS_DESIGN.md**: ASL 의 base idea (asymmetric weighting)

---

## 6. Data Augmentation

### 6.1 Zhang et al. (ICLR 2018) — "mixup: Beyond Empirical Risk Minimization"

**Contribution**:
- pixel-level linear interpolation
- multi-label 자동 (target 도 mix)

**우리 도메인 적용**:
- **Stage 3**: 부적합 (palette idx 평균 의미 없음)
- **THEORY.md augmentation**: 비교 baseline

### 6.2 Yun et al. (ICCV 2019) — "CutMix"

**Contribution**:
- patch-level mixing
- local feature preservation

**우리 도메인 적용**:
- **Stage 3**: 부적합 (chip grid 200×200 단위 안 맞으면 chip 깨짐)
- **THEORY.md augmentation**: 비교 baseline

### 6.3 Inoue (2018) — "Data Augmentation by Pairing Samples for Images"

**Contribution**:
- sample pairing strategy

**우리 도메인 적용**:
- **THEORY.md**: 비교 baseline

---

## 7. Wafer Defect Domain

### 7.1 Wang et al. (Soft Computing 2020) — "MixedWM38"

**Contribution**:
- 학계 multi-label wafer benchmark (38,015 wafer)
- 38 mixed defect class
- GAN augmentation

**우리 도메인 적용**:
- **EXAMPLES.md**: benchmark 비교
- **Stage 3**: mix 비율 reference (single 23.6%, 2-mix 34.2%, 3-mix 31.6%, 4-mix 10.5%)

### 7.2 Nag et al. (Computers in Industry 2022) — "WaferSegClassNet"

**Contribution**:
- light-weight classification + segmentation
- semiconductor wafer defect

**우리 도메인 적용**:
- **EXAMPLES.md**: 학계 wafer 모델 비교

### 7.3 (RDP-Net) "Wafer composite defect recognition based on residual dynamic perception" (2025)

**Contribution**:
- composite defect recognition
- asymmetric multi-label loss (ASL 변형)

**우리 도메인 적용**:
- **LOSS_DESIGN.md**: ASL 변형 후보 (future)

---

## 8. Hierarchical Classification

### 8.1 Kowsari et al. (Information 2020) — "HMIC: Hierarchical Medical Image Classification"

**arXiv**: 2006.07187

**Contribution**:
- 의료 image hierarchical classification
- coarse → fine class 학습

**우리 도메인 적용**:
- 우리 setting = flat 33-class (factorize 하지 않음)
- **future**: distribution × object factorize 시 참조

### 8.2 Huo et al. (Biomedical Signal Processing 2024) — "HiFuse"

**Contribution**:
- hierarchical multi-scale feature fusion
- medical image classification

**우리 도메인 적용**:
- 직접 적용 X (architecture)
- **future**: 비교 SOTA reference

---

## 9. Conditional Random Fields (CRF)

### 9.1 Lafferty et al. (ICML 2001) — "Conditional Random Fields"

**Contribution**:
- CRF 의 founding paper
- sequence labeling

**우리 도메인 적용**:
- **Stage 6 C4-C7**: CRF post-process
- **MATCHING_DESIGN.md**: CRF 이론 base

### 9.2 Krähenbühl & Koltun (NeurIPS 2011) — "Efficient Inference in Fully Connected CRFs"

**Contribution**:
- dense CRF (모든 pair)
- efficient mean-field inference

**우리 도메인 적용**:
- **Stage 6**: pydensecrf 라이브러리 base
- **MATCHING_DESIGN.md**: CRF 알고리즘

### 9.3 Mahalanobis (1936) — "On the generalized distance"

**Contribution**:
- Mahalanobis distance 의 founding
- covariance-aware distance

**우리 도메인 적용**:
- **Stage 6 GMM matching**: Mahalanobis distance 활용
- **MATCHING_DESIGN.md**: GMM 의 unary potential

---

## 10. Confidence Quantification

### 10.1 Gal & Ghahramani (ICML 2016) — "Dropout as Bayesian Approximation"

**Contribution**:
- MC Dropout 으로 prediction variance 측정
- uncertainty quantification

**우리 도메인 적용**:
- **DECISION_RULE.md**: confidence quantification 후보
- **future**: prod predict 에 추가

### 10.2 Hsu et al. (CVPR 2018) — "Generalized Zero-Shot Recognition"

**Contribution**:
- hierarchical class matching
- semantic embedding

**우리 도메인 적용**:
- **MATCHING_DESIGN.md**: consistency heuristic 의 의미적 결합 reference

---

## 11. Architecture

### 11.1 Liu et al. (CVPR 2023) — "ConvNeXt V2"

**Contribution**:
- ConvNeXt V2 architecture
- FCMAE pretraining
- ImageNet SOTA 81%+ accuracy

**우리 도메인 적용**:
- **현재 production 모델**: ConvNeXtV2 base FCMAE
- **모든 stage 의 base backbone**

### 11.2 Wolpert (1992) — "Stacked Generalization"

**Contribution**:
- stacking ensemble
- meta-learner combine multiple base models

**우리 도메인 적용**:
- **Stage 6 E5**: surface stacking ensemble
- **MATCHING_DESIGN.md**: ensemble framework

---

## 12. 종합 — 카테고리별 핵심 paper

| Category | Top 3 paper |
|---|---|
| **Multi-label SOTA loss** | Ridnik 2021 (ASL), Verelst 2024 (AdaGC), Wang 2017 (CB-loss) |
| **Threshold tuning** | Lipton 2014 (F1 sweep), Yan 2025 (IDF/KNN), Pillai 2013 (비교) |
| **Calibration** | Guo 2017 (Temperature), Platt 1999, Müller 2019 (label smoothing) |
| **Density estimation** | Parzen 1962 (KDE), Schwarz 1978 (BIC), Dempster 1977 (EM) |
| **Loss imbalance** | Cui 2019 (CB-loss), Lin 2017 (Focal), Szegedy 2016 (LS) |
| **Augmentation** | Zhang 2018 (Mixup), Yun 2019 (CutMix) |
| **Wafer domain** | Wang 2020 (MixedWM38), Nag 2022 (WSC-Net) |
| **CRF** | Lafferty 2001, Krähenbühl 2011 (dense) |
| **Architecture** | Liu 2023 (ConvNeXt V2) |

---

## 13. 참조

- 이론 base: `docs/multi-label/THEORY.md`
- 사례: `docs/multi-label/EXAMPLES.md`
- stage motivation: `docs/multi-label/STAGES.md`
- plan: `~/.claude/plans/1-input-batch-hidden-patterson.md`
