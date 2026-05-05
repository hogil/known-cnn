# Multi-label Ablation — Stage Status

**Last updated**: 2026-05-03 (round 3 — full 실측 round)

## ★ Round 3 핵심 결과

### Stage 5 best (학습 추가 0, ★ 즉시 도입 권장)
- **D2 per-class F1 sweep**: macro_F1=0.4648, mAP=0.4694
- baseline (D1 default 0.5): macro_F1=0.2673
- → **+74% macro F1 향상, 학습 비용 0**

### Stage 6 best (chip-wafer matching)
- **C7 (4-method geometric ensemble + CRF + percentile outlier)**: accuracy=0.7005
- baseline (C1 single heatmap_smooth): 0.6511
- → **+5.4% accuracy, surface ensemble + percentile outlier 효과 검증**

### 모든 결과 학습 추가 0 — production 즉시 도입 가능
- master_table.csv (15 row), decision_guide.md, master_comparison.png 산출

### 미흡 (다음 round 예정)
- Stage 4 Phase B/C 학습 (AdaGC + ASL mix M3-M7) — 학습 시간 미확보
- Stage 5 D8 KNN_local 가 가설 미달 (D2 per-class F1 가 단순 + best)
- rare_recall 0 — multi-label setting 에서 일부 class support 너무 적음

---

본 문서는 8 stage 의 진행 상태 + 산출 path + 다음 step 을 추적.

---

## 종합 진행 상태

| Stage | Status | 코드 | 산출 | 진척 |
|---|---|---|---|---|
| Stage 1 — 분포 학습 | ✅ COMPLETE | `_dist_learn_per_class.py` | 850 npy + 37 plot + CSV | 100% |
| Stage 2 — Hyperparameter | ⏳ TODO | (cnn_train.py 기존 활용) | (compound MemoryError 해결 후) | 0% |
| Stage 3 — `unknown_multi/` 합성 | ✅ COMPLETE | `_sample_gen_multi.py`, `_verify_multi.py` | 2000 PNG + JSON + manifest, verify pass | 100% |
| Stage 4 Phase A | ✅ COMPLETE | `_eval_multi_label.py` (D1-D6 wrap) | results/stage4_phaseA via stage5 | 100% |
| Stage 4 Phase B/C | ⏳ TODO | `multi_label_losses.py`, `cnn_train_multilabel.py` ✓ | (학습 dispatch 필요) | 50% |
| Stage 5 — Threshold tuning | ✅ COMPLETE | `_calibration_analysis.py`, `_threshold_sweep.py` | results/stage5_decision.csv (8 strategy) | 100% |
| Stage 6 — chip-wafer matching | ✅ COMPLETE | `_eval_chip_matching.py` | results/stage6_matching.csv (7 combo) | 100% |
| Stage 7 — Prod predict 보강 | ✅ COMPLETE | `cnn_predict_compound_prod.py` patched (`--multi-label-mode`, `--thresholds-json`, `--matching-method`, 2 parquet output) + `results/thresholds_per_class_D2.json` | 100% |
| Stage 8 — Master comparison | ✅ COMPLETE | `_generate_master_report.py` | results/master_table.{csv,md} + decision_guide.md + master_comparison.png | 100% |

**총 진척**: 코드 86% / 실측 25%

## 작성 완료 (Round 2)

- `_sample_gen_multi.py` — multi-pattern wafer 합성 (heatmap mask OR + chip 단위 random object). smoke 10 pass.
- `_verify_multi.py` — multi_labels JSON / chip-level GT / PNG / manifest 검증.
- `_calibration_analysis.py` — model + val inference → 33 class sigmoid prob 분포 + reliability + ECE. val_logits/val_y_multihot 저장.
- `_threshold_sweep.py` — D1-D8 8 strategy (Temperature / Platt / Isotonic / IDF / KNN_local / Temp+Platt mix / top-K floor). smoke pass.
- `_eval_multi_label.py` — Stage 4 Phase A wrapper. D1/D2/D3/D6 variant 평가.
- `_eval_chip_matching.py` — C1-C7 7 mix combination (surface ensemble + CRF + consistency).
- `multi_label_losses.py` — BCELossWithSmoothing / AsymmetricLoss (Ridnik) / AdaGCLoss (Verelst) / ChainedLoss / MixLoss / build_loss factory M1-M7. self-test pass.
- `_generate_master_report.py` — Stage 1-6 CSV 종합 → master_table + decision_guide + master_comparison.png. smoke pass.

---

## Stage 1 산출물 (✅ COMPLETE)

### 코드
- `_dist_learn_per_class.py` (437 lines) — chip 좌표 분포 학습 (5 method × 33 class × 5 data-amount)

### 데이터
- `_dist_heatmaps_per_class/` (gitignored)
  - `<class>__<method>__n=<n>.npy` × 825 = 33 × 5 × 5
  - 총 850 npy (33 class × 5 method × 5 data-amount + 25 hybrid)

### Plot
- `plots/dist_compare_<class>.png` × 33 (5 method × 5 n grid)
- `plots/dist_summary_hybrid.png` (33 class × hybrid full data)
- `plots/dist_bic_curves.png` (33 class × GMM BIC)
- 총 37 plot

### CSV
- `results/stage1_distribution.csv` — class × method × n × log_likelihood × best_K × n_chips_full

### Commit
- 687448b (commit message: "[Stage 1] _dist_learn_per_class.py: 5 method × 33 class × 5 data-amount sweep")

---

## Stage 2 산출물 (⏳ TODO)

### 의존
- compound 학습 MemoryError 해결 (workers=0, batch=4 이하 — 또는 wafer R-only fallback)

### 진행 시 산출 (예정)
- 11 runs (greedy: cw=3 + ls=5 + loss=3)
- `logs_compound/<run>/best_history.txt` 11 개
- `results/stage2_hyperparameter.csv` — run × val_f1 × test_f1 × per_class_F1 × ECE × multi_label_F1

### Fallback
- compound 메모리 부족 → `cnn_train_wafer.py` (R-only) 로 ablation
- `logs_wafer/` 산출 사용

---

## Stage 3 산출물 (⏳ TODO)

### 코드 (예정)
- `_sample_gen_multi.py` (~300 lines) — multi-pattern wafer 합성

### 데이터 (예정)
- `D:/project/data/wm-811k/unknown_multi/<basename>__d-...__o-....png` × 2000
- `D:/project/data/positions/unknown_multi/<basename>__d-...__o-....json` × 2000
- `D:/project/data/wm-811k/unknown_multi/_manifest.csv` (basename × distributions × objects)

### 검증 (예정)
- `_verify_multi.py` — multi_labels 필드, chip-level GT 일관성, PNG 무결성

---

## Stage 4 산출물 (⏳ TODO)

### Phase A (즉시 가능)
- 코드: `_eval_multi_label.py`
- `cnn_predict_compound.py` 결과 + sigmoid heuristic
- `results/stage4_phaseA.csv` (4 variant × metric)

### Phase B (학습 필요)
- 코드: `cnn_train_compound_adagc.py` (~600 lines)
- 학습: λ_gc ∈ {0.1, 0.5, 1.0} × ~30분 = 1.5h
- `logs_compound/adagc_*/best_model.pth`
- `results/stage4_phaseB.csv`

### Phase C (학습 + multi-label data 필요)
- 코드: `cnn_train_compound_bce.py` (~600 lines)
- 학습: BCE baseline + ASL 9 runs (greedy hyperparameter sweep)
- `logs_compound/bce_*/`, `logs_compound/asl_*/`
- `results/stage4_phaseC.csv`

### Mix 조합 (★ key contribution)
- 추가 5-10 runs (M3-M7)
- `logs_compound/M3_*/` 등
- `results/stage4_mix.csv`

### 의존
- Phase A: Stage 3 GT 필요
- Phase B: Stage 3 GT (validation 만)
- Phase C: Stage 3 GT (학습 + validation)

---

## Stage 5 산출물 (⏳ TODO)

### Stage 5a (calibration 분석)
- 코드: `_calibration_analysis.py` (~250 lines)
- `plots/calibration/dist_<class>.png` × 33
- `plots/calibration/reliability_overall.png`
- `plots/calibration/ece_per_class.png`

### Stage 5b (threshold sweep)
- 코드: `_threshold_sweep.py` (~400 lines)
- `results/stage5_threshold.csv` (8 strategy × 6 metric)
- `plots/threshold/threshold_vs_f1_<class>.png` × 33
- `plots/threshold/method_comparison_bar.png`

### 의존
- Stage 3 multi-label GT
- Stage 4 Phase A 의 prediction (sigmoid logits)

---

## Stage 6 산출물 (⏳ TODO)

### 코드
- `_eval_chip_matching.py` (~300 lines)

### 결과
- `results/stage6_matching.csv` (5 surface × 7 mix combination × accuracy/outlier/compute time)
- `plots/matching/confusion_matrix_*.png` × 7
- `plots/matching/surface_method_compare.png`
- `plots/matching/failure_cases/` (debug visualization)

### 의존
- Stage 1 surface (✅ 있음)
- Stage 3 chip-level GT

---

## Stage 7 산출물 (⏳ TODO)

### 코드 변경
- `cnn_predict_compound_prod.py` — multi-label + matching 옵션 추가
- 새 CLI: `--multi-label-threshold-strategy`, `--matching-method`, `--surfaces-root`, `--outlier-threshold`

### 출력 변경
- `result_compound/<product>/<line>/<date>/preds_wafer.parquet`
- `result_compound/<product>/<line>/<date>/preds_chip.parquet`
- `wrong_wafer/`, `wrong_chip/`
- `_meta.json` 에 `matching_status_summary` 추가

### 의존
- Stage 5 best (`thresholds_per_class.json`)
- Stage 6 best matching method

---

## Stage 8 산출물 (⏳ TODO)

### 코드
- `_generate_master_report.py` (~400 lines)

### 결과
- `results/master_table.csv`
- `results/master_table.md` (markdown rendering)
- `results/decision_guide.md` (budget 별 도입 sequence)
- `plots/master_comparison.png` (6 subplot)
- `plots/per_class_f1_stacked.png`
- `plots/calibration_compare.png`

### 의존
- 모든 stage CSV (stage1-stage6)

---

## 다음 Step (우선순위)

### 즉시 가능
1. **Stage 3 시작** (`_sample_gen_multi.py` 작성) — Stage 4/5/6 의 GT 확보
2. **Stage 5a 동시 시작** (`_calibration_analysis.py`) — 기존 single-label 모델 + val set 으로 가능

### Stage 3 후
3. **Stage 4 Phase A** (`_eval_multi_label.py` + sigmoid heuristic 평가)
4. **Stage 5b** (`_threshold_sweep.py` — 8 strategy)
5. **Stage 6** (`_eval_chip_matching.py` — 7 mix combination)

### 학습 자원 확보 후
6. **Stage 2** (compound 메모리 해결 후)
7. **Stage 4 Phase B** (AdaGC 학습)
8. **Stage 4 Phase C** (BCE/ASL 학습 + mix M3-M7)

### 모든 stage 완료 후
9. **Stage 7** (prod predict 보강)
10. **Stage 8** (master report)

---

## ★ 핵심 영역 우선순위 (사용자 강조)

| 영역 | 관련 Stage | 우선순위 |
|---|---|---|
| **Loss 설계 + mix** | Stage 4 Phase B/C | 1 (학습 추가 필요하나 효과 큼) |
| **Multi-label 판정** | Stage 4 Phase A + Stage 5 | ★ 최우선 (학습 추가 0) |
| **Chip-wafer matching** | Stage 6 | 2 (Stage 1 surface 이미 있음) |

→ **권장 순서**: Stage 3 합성 → Stage 5 (판정) → Stage 6 (matching) → Stage 4 (loss)

---

## 참조

- plan: `~/.claude/plans/1-input-batch-hidden-patterson.md`
- 이론: `THEORY.md`
- ★ deep-dive: `LOSS_DESIGN.md`, `MATCHING_DESIGN.md`, `DECISION_RULE.md`
- skill: `.claude/skills/multi-label-ablation/SKILL.md`
- agent: `.claude/agents/multi-label-ablation.md`
- memory: `~/.claude/projects/.../memory/project_multi_label_ablation.md`
