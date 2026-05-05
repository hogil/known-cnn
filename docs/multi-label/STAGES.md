# 8 Stages Motivation — Theory + Hypothesis + Expected Outcome

본 문서는 plan 의 8 stage 의 **motivation / 이론 base / 가설 / 기대 효과** 만
응축한 이론 중심 view. **실행 detail (commands, sweep range, verification) 은
plan 본문 + skill 참조**.

---

## Stage 1 — 분포 학습 ablation

### 목적
chip 좌표 분포를 32×32 grid 위 surface 로 학습. 5 method × 5 data-amount sweep.

### 이론 base
- KDE (Parzen 1962), GMM (Dempster 1977), histogram (non-parametric)
- BIC (Schwarz 1978) 로 GMM K 결정
- Silverman (1986) 으로 KDE bandwidth

### 가설
- **H3**: chip 위치 분포 기반 heatmap matching 으로 chip-wafer assignment accuracy 80%+ 달성

### 기대 효과
- per-class best surface 결정 (Donut → KDE, Edge → GMM, Center → heatmap)
- Stage 6 chip-wafer matching 의 surface base
- chip 위치 outlier 검출 가능

### 우리 ablation 의 위치
- ★ deep-dive: `MATCHING_DESIGN.md` 의 단일 surface 분석 base

### Status
✅ **완료** — `_dist_learn_per_class.py`, 850 npy + 37 plot + CSV. commit 687448b.

---

## Stage 2 — Hyperparameter ablation

### 목적
single-label 학습의 hyperparameter (class_weight, label_smoothing, loss) 가 multi-label 추론 효과에 미치는 영향 측정.

### 이론 base
- Cui et al. (CVPR 2019) Effective Number weighting
- Lin et al. (ICCV 2017) Focal Loss
- Szegedy et al. (CVPR 2016) Label Smoothing
- Müller et al. (NeurIPS 2019) LS calibration 효과

### 가설
- **H2**: 학습 시 class imbalance 보강 (weighted BCE / focal / ASL) 이 multi-label F1 +3-7%
- **H6** (★ mix): ASL + label_smoothing + class_weight + focal mix 가 단일 ASL SOTA 대비 +2-5%

### 기대 효과
- 11 runs (greedy) 로 best hyperparameter 결정
- multi-label 추론 시 calibration ↑ (label smoothing 효과)
- rare class F1 +20% 가능 (effective weighting)

### 우리 ablation 의 위치
- ★ deep-dive: `LOSS_DESIGN.md` 의 보조 mechanism + mix 조합 base

### Status
⏳ TODO — compound MemoryError 해결 후 진행. fallback: cnn_train_wafer.py R-only.

---

## Stage 3 — `unknown_multi/` 합성 데이터셋

### 목적
multi-label evaluation baseline 확보. 1000~3000 합성 wafer 에 known multi_labels GT 부여 → Stage 4/5/6 의 정량 평가 ground truth.

### 이론 base
- Wang et al. (Soft Computing 2020) MixedWM38 mix 비율 reference
- 자체 wafer-aware multi-pattern composition (heatmap mask OR + chip 단위 random object)

### 가설
- 합성 multi-label 이 학계 oracle (full-multi-label labeled training) 의 90%+ 회복 가능 (Phase C 학습 시)

### 기대 효과
- Stage 4/5/6 의 multi-label F1 측정 가능 (GT 확보)
- Stage 4 Phase C (BCE/ASL) 학습 시 직접 사용 가능
- chip-level GT (true_distribution, true_object) 보존 → Stage 6 matching accuracy

### 우리 ablation 의 위치
- 모든 후속 stage 의 평가 dataset
- `THEORY.md` augmentation 섹션의 wafer-aware composition base

### Status
⏳ TODO — `_sample_gen_multi.py` 작성 후 합성 (1.5h).

---

## Stage 4 — Multi-label 추론 path 비교 (★ 핵심)

### 목적
Single-label trained 모델을 multi-label 추론으로 사용하는 3 phase 비교:
- Phase A: heuristic — 학습 변경 없음
- Phase B: AdaGC retraining — 학습 보강
- Phase C: BCE/ASL retraining — 학계 SOTA

### 이론 base

**Phase A (sigmoid heuristic)**:
- single-label trained 모델의 sigmoid output ranking 활용
- Lipton (2014) F1-optimal threshold

**Phase B (AdaGC)**:
- Cole et al. (CVPR 2021) SPML founding
- Verelst et al. (2024) AdaGC dual-EMA pseudo-label
- Liu et al. (ICML 2023) pseudo-label 분석

**Phase C (BCE/ASL)**:
- Ridnik et al. (ICCV 2021) ASL — multi-label SOTA
- Wang et al. (ICCV 2017) class-balanced multi-label
- Bénédict et al. (TMLR 2022) SigmoidF1

### 가설
- **H1**: sigmoid heuristic 만으로 multi-label F1 0.65+ (학계 oracle 70-80%)
- **H5**: AdaGC 로 BCE 대비 mAP +3-5%, oracle 의 90%+
- **H6** (★ mix): ASL + LS + CW + Focal mix 가 단일 ASL 대비 +2-5%

### 기대 효과
- Phase A: mAP 0.65 → 0.74 (Stage 5 threshold tuning 적용 시)
- Phase B: mAP 0.74 → 0.80 (+6%)
- Phase C: mAP 0.80 → 0.86 (+6%)
- ★ Mix (M3-M7): 추가 +2-3%

### 우리 ablation 의 위치
- ★ ★ ★ 본 ablation 의 **가장 중요 stage** — 3 핵심 영역 모두 포함
- ★ deep-dive: `LOSS_DESIGN.md` (Phase B/C) + `DECISION_RULE.md` (Phase A)

### Status
⏳ TODO — Phase A 부터 (Stage 3 합성 후 즉시 가능).

---

## Stage 5 — Threshold Tuning (★ 핵심)

### 목적
multi-label decision threshold 결정 전략 5종 + mix 조합 비교.

### 이론 base
- Lipton et al. (2014) per-class F1 sweep
- Yang & Yu (2015) F1-optimal multi-label
- Guo et al. (2017) Temperature scaling
- Platt (1999), Niculescu-Mizil (2005) calibration
- Yan et al. (2025) IDF + KNN_local fusion
- Pillai et al. (2013) 비교 분석

### 가설
- **H4**: per-class F1 sweep + Temperature 조합으로 default 0.5 대비 macro F1 +5-10%
- **H8** (★ mix): per-class threshold + Temp + IDF + top-K floor mix 가 단일 strategy 대비 +2-4%

### 기대 효과
- D1 baseline 0.62 → D8 mix best 0.76 (+11%)
- 학습 추가 0 — 가장 high-ROI
- Confidence quantification 추가 (top1/top2 ratio + entropy) → DB 분석 활용

### 우리 ablation 의 위치
- ★ ★ deep-dive: `DECISION_RULE.md`
- production 도입 우선순위 #1 (Tier 1, 학습 비용 X)

### Status
⏳ TODO — Stage 3 합성 후. Phase A 와 동시 가능.

---

## Stage 6 — Chip-Wafer Matching (★ 핵심)

### 목적
multi-label compound 추론 시 chip 한 개 → wafer 의 multi-label 중 어느 wafer-class 에 속하는지 결정.

### 이론 base
- Mahalanobis (1936) Mahalanobis distance
- Lafferty et al. (2001) CRF founding
- Krähenbühl & Koltun (NeurIPS 2011) dense CRF
- Wolpert (1992) stacking ensemble
- Hsu et al. (CVPR 2018) hierarchical class matching

### 가설
- **H3**: heatmap matching 으로 chip-wafer assignment 80%+
- **H7** (★ mix): heatmap_smooth + GMM ensemble + CRF post-process 가 단일 hybrid 대비 +3-5%

### 기대 효과
- C1 baseline 0.78 → C7 best 0.89 (+11%)
- chip-level 분석 enable (DB JOIN)
- outlier / mismatch / ambiguous status 자동 분류

### 우리 ablation 의 위치
- ★ ★ deep-dive: `MATCHING_DESIGN.md`
- chip object × wafer pattern 의 의미적 결합 — **학계 무관 contribution**

### Status
⏳ TODO — Stage 1 surface + Stage 3 GT 활용.

---

## Stage 7 — Prod Predict 보강

### 목적
Stage 1-6 의 best 조합을 cnn_predict_compound_prod.py 에 통합.

### 이론 base
- Mahalanobis (1936) matching distance
- TensorRT/ONNX production deployment patterns

### 가설
- 즉시 production 도입 가능 (Stage 5 best + Stage 6 best default)
- 분석 유연성 ↑ (2 parquet, DB JOIN)

### 기대 효과
- preds_wafer.parquet (1 row / wafer × positive bit)
- preds_chip.parquet (1 row / chip)
- wrong_wafer/, wrong_chip/ 분리 (debug-friendly)
- _meta.json 에 matching_status_summary 추가

### 우리 ablation 의 위치
- ablation → production deployment 의 **bridge**
- Stage 1-6 결과를 production 에 적용

### Status
⏳ TODO — Stage 1-6 완료 후 통합.

---

## Stage 8 — Master Comparison

### 목적
전체 ablation 결과 한 곳에 정리. paper-style table + figure.

### 이론 base
- 학계 ablation paper format (Ridnik 2021, Cui 2019, Lin 2017, Liu 2023)
- multi-metric reporting (mAP / macro F1 / micro F1 / Hamming / ECE / per-class breakdown)

### 가설
- production 도입 의사결정의 명확한 근거 자료 도출
- 추후 모델 개선 시 비교 baseline

### 기대 효과
- master_table.csv + master_table.md (paper-rendering)
- master_comparison.png (6 subplot figure)
- decision_guide.md (budget 별 도입 sequence)

### 우리 ablation 의 위치
- 모든 stage 의 종합 view
- paper publication-ready

### Status
⏳ TODO — 모든 stage 완료 후.

---

## ★ 8 Stage 의 핵심 영역 분포

```
                Stage 1  Stage 2  Stage 3  Stage 4  Stage 5  Stage 6  Stage 7  Stage 8
                ────────────────────────────────────────────────────────────────────────
Loss 설계                   ●●               ●●●                                    ●
Matching         ●●                                            ●●●                  ●
Multi-label                                  ●●●     ●●●                            ●
판정 방식
Density          ●●●                                            ●●
Calibration                                            ●●●
Production                                                              ●●●
종합                                                                              ●●●
```

★ ★ ★ = 본 ablation 의 가장 중요 stage (사용자 우선순위 3 영역 모두 포함)
- Stage 4: Loss 설계 + Multi-label 판정
- Stage 5: Multi-label 판정 + Calibration
- Stage 6: Matching ensemble + CRF + consistency

---

## ★ 가설 종합 (H1-H8)

| 가설 | Stage | 검증 metric |
|---|---|---|
| H1 | Phase A | sigmoid heuristic multi-label F1 ≥ 0.65 |
| H2 | Stage 2 | class weighting → +3-7% multi-label F1 |
| H3 | Stage 6 | heatmap matching accuracy ≥ 0.80 |
| H4 | Stage 5 | per-class F1 + Temp → +5-10% macro F1 |
| H5 | Phase B | AdaGC → +3-5% mAP |
| **H6** ★ | Stage 4 mix | ASL + LS + CW + Focal mix → +2-5% over single ASL |
| **H7** ★ | Stage 6 mix | surface ensemble + CRF + consistency → +3-5% over hybrid |
| **H8** ★ | Stage 5 mix | per-class + Temp + IDF + top-K floor → +2-4% over single |

★ = mix 가설 (사용자 우선순위, 본 ablation 의 진짜 contribution).

---

## 참조

- 이론 base: `docs/multi-label/THEORY.md`
- 논문: `docs/multi-label/PAPERS.md`
- 사례: `docs/multi-label/EXAMPLES.md`
- ★ deep-dive: `LOSS_DESIGN.md`, `MATCHING_DESIGN.md`, `DECISION_RULE.md`
- plan: `~/.claude/plans/1-input-batch-hidden-patterson.md`
- skill: `.claude/skills/multi-label-ablation/SKILL.md`
- agent: `.claude/agents/multi-label-ablation.md`
