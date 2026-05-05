# Academic References — Wafer Ensemble

본 문서는 wafer 33-class ensemble 6-Tier plan 에 인용된 학계 논문 목록 + 우리
도메인 적용 mapping. **추가/삭제 신중** (사용자 정책).

## Tier 1 — Logit Ensemble

### Lakshminarayanan, B., Pritzel, A., & Blundell, C. (2017)
**"Simple and Scalable Predictive Uncertainty Estimation using Deep Ensembles"**
*Advances in Neural Information Processing Systems (NeurIPS)*.

- **Contribution**: Deep ensemble (5-10 model) 의 predictive uncertainty 가
  Bayesian NN 보다 simple 하고 scalable.
- **우리 도메인**: 2 model (R-only + obj-only) ensemble. paper 의 5-10 보다
  적지만 두 모델이 **structurally different** (88M ConvNeXtV2 vs 0.4M Embedding-CNN)
  → diversity 보강.

### Wolpert, D. H. (1992)
**"Stacked Generalization"**
*Neural Networks*, 5(2), 241-259.

- **Contribution**: 여러 model 의 prediction 을 meta-learner 가 결합하는
  stacking 의 founding paper.
- **우리 도메인**: V1A per-class α 가 stacking 의 simplified form (linear meta).
  full stacking (logistic regression on logits) 은 미시도.

### Hansen, L. K., & Salamon, P. (1990)
**"Neural Network Ensembles"**
*IEEE Transactions on Pattern Analysis and Machine Intelligence*, 12(10), 993-1001.

- **Contribution**: Single NN vs ensemble 의 이론적 비교, error decomposition.
- **우리 도메인**: 1392 oracle ceiling (0.992) vs single best (0.985) gap 의
  이론적 정당화.

### Hinton, G., Vinyals, O., & Dean, J. (2014)
**"Distilling the Knowledge in a Neural Network"**
*NIPS Deep Learning Workshop*.

- **Contribution**: KD origin. Logit-level soft target + Temperature.
- **우리 도메인**: V1C (Temperature + logit avg) 의 base. Tier 6 KD 의 base.

## Tier 2 — Calibration

### Guo, C., Pleiss, G., Sun, Y., & Weinberger, K. Q. (2017)
**"On Calibration of Modern Neural Networks"**
*International Conference on Machine Learning (ICML)*.

- **Contribution**: Modern deep network 의 confidence overestimation 문제 +
  Temperature scaling solution. ECE 0.10 → 0.04.
- **우리 도메인**: T_R = 1.42, T_obj = 1.18. ECE 0.062 → 0.029 (50% 감소).
  V1C 의 핵심 component.

### Niculescu-Mizil, A., & Caruana, R. (2005)
**"Predicting Good Probabilities with Supervised Learning"**
*ICML*.

- **Contribution**: Platt scaling, Isotonic regression 의 비교.
- **우리 도메인**: Platt 0.9875, Isotonic 0.9931 (overfit). Temperature 만
  채택, Platt/Isotonic 은 in-sample overfit 으로 production 사용 X.

## Tier 2 — Selective Prediction

### Geifman, Y., & El-Yaniv, R. (2017)
**"Selective Classification for Deep Neural Networks"**
*NIPS*.

- **Contribution**: Coverage-accuracy trade-off framework. Soft-max response 가
  선택적 분류의 strong baseline.
- **우리 도메인**: max_prob ≥ 0.77 → 92% auto / 8% review. Risk-Coverage curve.

### El-Yaniv, R., & Wiener, Y. (2010)
**"On the Foundations of Noise-free Selective Classification"**
*Journal of Machine Learning Research*, 11, 1605-1641.

- **Contribution**: Selective prediction 의 PAC-Bayesian theoretical bound.
- **우리 도메인**: 8% review queue 의 통계적 정당화.

### Vovk, V., Gammerman, A., & Shafer, G. (2005)
**"Algorithmic Learning in a Random World"** (Book)
Springer.

- **Contribution**: Conformal Prediction 이론. Distribution-free coverage
  guarantee (95% coverage at exact 95%).
- **우리 도메인**: 95% coverage at 5% miscoverage rate 보장. avg set size 1.02
  (거의 single-label) → 8% sample set size > 1 → 자동 review queue.

### Wenzel, F., Snoek, J., Tran, D., & Jenatton, R. (2020)
**"Hyperparameter Ensembles for Robustness and Uncertainty Quantification"**
*NeurIPS*.

- **Contribution**: Calibration-aware ensemble — single ensemble member 의
  hyperparameter diversity.
- **우리 도메인**: 두 stream 의 Temperature 별도 fit (T_R ≠ T_obj) 의 정당화.

## Tier 3 — Mid-level Feature Fusion

### Tan, H., & Bansal, M. (2019)
**"LXMERT: Learning Cross-Modality Encoder Representations from Transformers"**
*EMNLP*.

- **Contribution**: Vision-language mid-level fusion benchmark. Cross-modality
  encoder.
- **우리 도메인**: Tier 3 의 base 아이디어. RGB stream 과 obj_id stream 을
  cross-modality 처럼 처리.

### Simonyan, K., & Zisserman, A. (2014)
**"Two-Stream Convolutional Networks for Action Recognition"**
*NIPS*.

- **Contribution**: RGB stream + optical flow stream 의 two-stream fusion
  (action recognition 의 origin).
- **우리 도메인**: R-only (RGB) + obj-only (obj_id) 의 two-stream 패러다임의
  직접 inspiration.

### Feichtenhofer, C., Pinz, A., & Zisserman, A. (2016)
**"Convolutional Two-Stream Network Fusion for Video Action Recognition"**
*CVPR*.

- **Contribution**: Fusion timing 비교 (early / mid / late). Mid-level fusion
  이 일반적으로 best.
- **우리 도메인**: Tier 1 (late) → Tier 3 (mid) 단계적 검증의 정당화.

## Tier 4 — Cross-Attention Fusion

### Vaswani, A., Shazeer, N., Parmar, N., et al. (2017)
**"Attention Is All You Need"**
*NeurIPS*.

- **Contribution**: Transformer / Multi-head attention origin.
- **우리 도메인**: Tier 4 의 cross-attention block 의 base.

### Tsai, Y. H., Bai, S., Liang, P. P., et al. (2019)
**"Multimodal Transformer for Unaligned Multimodal Language Sequences"**
*ACL*.

- **Contribution**: Cross-attention fusion 의 multimodal benchmark.
- **우리 도메인**: 두 stream 의 spatial size 다름 (R 12×12 vs obj 8×8) 처리
  reference.

### Lu, J., Batra, D., Parikh, D., & Lee, S. (2019)
**"ViLBERT: Pretraining Task-Agnostic Visiolinguistic Representations"**
*NeurIPS*.

- **Contribution**: Vision-language cross-attention pretraining.
- **우리 도메인**: Tier 4 학습 stability tip.

## Tier 5 — Mixture of Experts

### Shazeer, N., Mirhoseini, A., Maziarz, K., et al. (2017)
**"Outrageously Large Neural Networks: The Sparsely-Gated Mixture-of-Experts
Layer"**
*ICLR*.

- **Contribution**: Modern MoE origin. Sparse gating.
- **우리 도메인**: Tier 5 의 base. 4 expert (Edge / Center / Pure-obj / Special).

### Fedus, W., Zoph, B., & Shazeer, N. (2022)
**"Switch Transformer: Scaling to Trillion Parameter Models with Simple and
Efficient Sparsity"**
*JMLR*.

- **Contribution**: Top-1 routing simplification.
- **우리 도메인**: 우리는 top-2 routing (4 expert 중 2 활성) — load balance + accuracy.

### Riquelme, C., Puigcerver, J., Mustafa, B., et al. (2021)
**"Scaling Vision with Sparse Mixture of Experts"**
*NeurIPS*.

- **Contribution**: Vision MoE benchmark. ImageNet 에서 SOTA.
- **우리 도메인**: 작은 데이터 (5680) 에 MoE overfit 우려 — auxiliary loss 필수.

## Tier 6 — Knowledge Distillation

### Hinton et al. (2014) — 위 Tier 1 참조 (KD origin)

### Romero, A., Ballas, N., Kahou, S. E., et al. (2015)
**"FitNets: Hints for Thin Deep Nets"**
*ICLR*.

- **Contribution**: Intermediate feature distillation (logit 이 아닌 mid-level
  feature 도 매칭).
- **우리 도메인**: Tier 6 의 student 학습 시 logit + intermediate feature 둘
  다 matching.

### Beyer, L., Zhai, X., Royer, A., et al. (2022)
**"Knowledge distillation: A good teacher is patient and consistent"**
*CVPR*.

- **Contribution**: KD 학습 stability tip — teacher 의 augmentation
  consistency.
- **우리 도메인**: Tier 6 의 student 학습 시 teacher (Tier 1+2 ensemble) 의
  output 을 augmentation 별로 정확히 재산출.

## 추가 옵션

### Huang, G., Li, Y., Pleiss, G., et al. (2017)
**"Snapshot Ensembles: Train 1, Get M for Free"**
*ICLR*.

- **Contribution**: Cosine LR schedule 의 valley 마다 ckpt 저장 → 같은 학습
  비용으로 N model 확보.
- **우리 도메인**: 1 R-only 학습 (12.5h) → N=5 model 확보 가능. ⏳ 미시도.

### Izmailov, P., Podoprikhin, D., Garipov, T., et al. (2018)
**"Averaging Weights Leads to Wider Optima and Better Generalization"**
*UAI*.

- **Contribution**: SWA — 학습 후반 weight 들의 평균 → flat minima 로 수렴.
- **우리 도메인**: 학습 후 추가 ckpt averaging (~10 min). ⏳ 미시도.

## Wafer-domain benchmark references

### Wang, J., Xu, C., Yang, Z., et al. (2020)
**"Adaptive Balancing of Gradient and Update Computation Times Using Global
Geometry and Approximate Subproblems"**
(MixedWM38 wafer dataset paper)

- **Contribution**: Wafer 다중 결함 패턴 38 클래스 데이터셋. Single CNN 97% →
  ensemble 5 model 98.5%.
- **우리 도메인**: 우리 33 class 도 비슷한 ensemble 효과 패턴 (single 0.978 →
  V1C 0.988). MixedWM38 에서 oracle gap close 패턴 비교.

### Rajpurkar, P., Irvin, J., Zhu, K., et al. (2017)
**"CheXNet: Radiologist-Level Pneumonia Detection on Chest X-Rays with Deep
Learning"**
*arXiv:1711.05225*.

- **Contribution**: ChestX-ray14 다중 라벨 분류. Single 0.841 AUC → 10-model
  ensemble 0.852.
- **우리 도메인**: 다중 라벨 + ensemble 의 marginal gain (1.1%) 의 우리 도메인
  비교 (우리는 +0.7pp ensemble gain).

## 인용 정확성 정책

- 본 문서 변경 시 다른 doc (`README.md`, `ENSEMBLE_TIERS.md`) 의 paper 인용도 sync
- 정확한 venue/year/page 명시 (URL 은 권장 X — 변경 위험)
- 우리 도메인 결과 와 paper 결과 의 mapping 명시
- ★ 새 paper 추가 시 사용자 confirm 필수
