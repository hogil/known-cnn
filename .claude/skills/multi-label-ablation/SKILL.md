---
name: multi-label-ablation
description: Multi-label wafer 분류 8-stage paper-style ablation 의 실행 패턴. 각 stage 의 sweep range, greedy 압축, 최적값 찾는 방법, verification, fallback. ★ 핵심 3 영역 (loss 설계, chip-wafer matching, multi-label 판정) 의 mix 조합 sweep workflow.
---

# multi-label-ablation skill

이 skill 은 plan `~/.claude/plans/1-input-batch-hidden-patterson.md` 의 8 stage 를
실제 실행할 때 **어떻게 진행하고 검증하고 최적값 찾는지** 의 표준 패턴.

이론 + 논문 + 사례 는 `docs/multi-label/` 참조 (이 skill 은 operational).

## Read first

1. `docs/multi-label/README.md` (인덱스)
2. `docs/multi-label/STATUS.md` (현재 stage 진행 상태)
3. `docs/multi-label/STAGES.md` (해당 stage 의 motivation / 가설 / 기대)
4. ★ deep-dive (해당 stage 와 관련):
   - Stage 2 / Stage 4 → `docs/multi-label/LOSS_DESIGN.md`
   - Stage 4 Phase A / Stage 5 → `docs/multi-label/DECISION_RULE.md`
   - Stage 1 / Stage 6 → `docs/multi-label/MATCHING_DESIGN.md`
5. plan 의 해당 stage section (실행 detail)

## Quickstart

```bash
# 현재 stage 확인
cat docs/multi-label/STATUS.md

# Stage 1 ✅ 이미 완료 (skip)

# Stage 3 시작 (합성 데이터)
python _sample_gen_multi.py --n 10 --output-root /tmp/test_multi      # smoke
python _verify_multi.py --root /tmp/test_multi
python _sample_gen_multi.py --n 2000 --workers 4                      # full

# Stage 5a (calibration 분석, 즉시 가능)
python _calibration_analysis.py --model logs_compound/overall/best_model.pth \
   --val-set D:/project/data/wm-811k/unknown_multi --output plots/calibration/

# Stage 5b (threshold sweep)
for config in D1 D2 D3 D4 D5 D6 D7 D8; do
   python _threshold_sweep.py --config configs/decision_${config}.yaml \
      --val-logits ... --test-logits ... --output results/decision_${config}.json
done

# 모든 stage 완료 후
python _generate_master_report.py --stage-results-dir results/
```

자세한 stage 별 명령어는 plan 본문 또는 아래 "Stage 별 실행 패턴" 섹션 참조.

---

## Stage 별 실행 패턴

### Stage 1 — 분포 학습 (✅ COMPLETE, reference)

이미 `_dist_learn_per_class.py` 실행 완료. 산출:
- `_dist_heatmaps_per_class/<class>__<method>__n=<n>.npy` × 850
- `plots/dist_*.png` × 37
- `results/stage1_distribution.csv`

재실행 필요 시:
```bash
python _dist_learn_per_class.py --positions-root D:/project/data/positions/unknown \
   --output-root _dist_heatmaps_per_class --plots-dir plots/
```

### Stage 2 — Hyperparameter sweep (greedy 11 runs)

**의존**: cnn_train_compound.py 의 MemoryError 해결 (workers=0, batch=4 이하). fallback: `cnn_train_wafer.py` (R-only).

**Greedy order**:
```bash
# Stage 2a — class_weight 비교 (other dim default)
python cnn_train_compound.py --epochs 30 --batch 8 --workers 0 \
   --class-weight none --label-smoothing 0.02 --loss ce --model-tag s2a_cw_none
python cnn_train_compound.py --epochs 30 --batch 8 --workers 0 \
   --class-weight inverse --label-smoothing 0.02 --loss ce --model-tag s2a_cw_inv
python cnn_train_compound.py --epochs 30 --batch 8 --workers 0 \
   --class-weight effective --label-smoothing 0.02 --loss ce --model-tag s2a_cw_eff

# → 3 run 결과 비교 → best CW* 결정
# (예상: effective best)

# Stage 2b — label_smoothing sweep (CW* fix)
for ls in 0.0 0.02 0.05 0.1 0.2; do
   python cnn_train_compound.py --epochs 30 --batch 8 --workers 0 \
      --class-weight <CW*> --label-smoothing $ls --loss ce --model-tag s2b_ls_${ls}
done
# → 5 run → best LS* 결정 (예상: 0.05)

# Stage 2c — loss 비교 (CW*, LS* fix)
python cnn_train_compound.py --epochs 30 --batch 8 --workers 0 \
   --class-weight <CW*> --label-smoothing <LS*> --loss focal --focal-gamma 2 --model-tag s2c_focal_2
python cnn_train_compound.py --epochs 30 --batch 8 --workers 0 \
   --class-weight <CW*> --label-smoothing <LS*> --loss focal --focal-gamma 5 --model-tag s2c_focal_5
# → 3 run → best loss* 결정
```

**총**: 11 run × ~30분 = 5.5h GPU.

**최적값 찾기**:
- greedy 가 핵심 — 한 dim 씩 sweep, 다른 dim 은 default
- 각 dim 의 marginal effect 측정
- val_macro_f1 + multi_label_F1 (Stage 4 Phase A 평가) 두 metric 모두 기록

**Result table**:
```
results/stage2_hyperparameter.csv:
   run_tag, class_weight, label_smooth, loss, val_f1, test_f1, val/test gap,
   per_class_f1_min, per_class_f1_std, ECE, multi_label_F1
```

### Stage 3 — `unknown_multi/` 합성

**합성 algorithm**: heatmap mask OR + chip 단위 random object. Stage 1 surface 활용.

```bash
# Smoke (10 wafer, ~30s)
python _sample_gen_multi.py --n 10 --mix 0.7,0.2,0.1 \
   --output-root /tmp/test_multi --seed 42

# 시각 검증
ls /tmp/test_multi/wm-811k/unknown_multi/ | head -5

# Verification (smoke)
python _verify_multi.py --root /tmp/test_multi

# Full (2000 wafer, ~1.5h CPU)
python _sample_gen_multi.py --n 2000 --mix 0.7,0.2,0.1 \
   --output-root D:/project/data/wm-811k/unknown_multi --workers 4 --seed 42

# Verification (full)
python _verify_multi.py --root D:/project/data/wm-811k/unknown_multi
```

**최적값 찾기**:
- mix 비율 (default 70/20/10): MixedWM38 의 23.6/34.2/31.6/10.5 와 비교
- 본 ablation 에서 70/20/10 sufficient (multi-label 만 검증, 학습 X)
- 실패 시 50/30/20 으로 mix 비중 ↑

### Stage 4 — Multi-label 추론 path 비교

#### Phase A — sigmoid heuristic (즉시, 학습 X)

```bash
# 1. Stage 3 합성 후 unknown_multi/ inference
python cnn_predict_compound.py \
   --model logs_compound/overall/best_model.pth \
   --input D:/project/data/wm-811k/unknown_multi \
   --threshold-sweep 0.05,0.95,0.05

# 2. _eval_multi_label.py 로 4 variant (default 0.5, sweep, +Temp, +IDF) 평가
for strategy in default sweep temp idf; do
   python _eval_multi_label.py \
      --predictions logs_predict_compound/<TS>_unknown_multi/preds.json \
      --gt D:/project/data/wm-811k/unknown_multi/_manifest.csv \
      --threshold-strategy $strategy \
      --output results/stage4_phaseA_${strategy}.json
done

# 3. 종합
python -c "
import json, pandas as pd
rows = []
for s in ['default', 'sweep', 'temp', 'idf']:
   d = json.load(open(f'results/stage4_phaseA_{s}.json'))
   rows.append({'strategy': s, **d})
pd.DataFrame(rows).to_csv('results/stage4_phaseA.csv', index=False)
"
```

#### Phase B — AdaGC retraining (학습 +30분/run)

```bash
# 코드 작성: cnn_train_compound_adagc.py (~600 lines)
# → docs/multi-label/LOSS_DESIGN.md 의 AdaGCLoss class 참조

# λ_gc sweep (3 run)
for lam in 0.1 0.5 1.0; do
   python cnn_train_compound_adagc.py --lambda-gc $lam \
      --epochs 30 --batch 8 --workers 0 --model-tag adagc_l${lam}
done

# Best λ 모델로 unknown_multi 평가
python cnn_predict_compound.py \
   --model logs_compound/adagc_l0.5_*/best_model.pth \
   --input D:/project/data/wm-811k/unknown_multi
python _eval_multi_label.py --predictions ... --gt ... \
   --output results/stage4_phaseB.json
```

#### Phase C — BCE / ASL retraining (학습 +60분/run)

```bash
# 코드 작성: cnn_train_compound_bce.py (~600 lines)
# → docs/multi-label/LOSS_DESIGN.md 의 AsymmetricLoss class 참조

# BCE baseline
python cnn_train_compound_bce.py --loss bce \
   --train-multi D:/project/data/wm-811k/unknown_multi \
   --epochs 30 --batch 8 --workers 0 --model-tag bce

# ASL hyperparameter sweep (greedy 9 runs)
# default: γ_pos=1, γ_neg=4, clip=0.05
for params in "1 4 0.05" "0 4 0.05" "2 4 0.05" "1 2 0.05" "1 6 0.05" "1 4 0" "1 4 0.1"; do
   gp=$(echo $params | awk '{print $1}')
   gn=$(echo $params | awk '{print $2}')
   cl=$(echo $params | awk '{print $3}')
   python cnn_train_compound_bce.py --loss asl --gamma-pos $gp --gamma-neg $gn --clip $cl \
      --train-multi D:/project/data/wm-811k/unknown_multi \
      --epochs 30 --batch 8 --workers 0 --model-tag asl_p${gp}_n${gn}_c${cl}
done

# best ASL 모델로 unknown_multi 평가
# → results/stage4_phaseC.csv
```

#### ★ Mix 조합 (M3-M7, 5 run, 추가 +5h)

`docs/multi-label/LOSS_DESIGN.md` 의 M3-M7 정의 참조.

```bash
# M3: ASL + effective(0.9999) + ls=0.05
python cnn_train_compound_bce.py --loss asl --gamma-pos 1 --gamma-neg 4 --clip 0.05 \
   --class-weight effective --class-weight-beta 0.9999 --label-smoothing 0.05 \
   --epochs 30 --batch 8 --model-tag M3

# M4: AdaGC + ASL hybrid (cnn_train_compound_adagc.py 에 aux_loss=asl 옵션 필요)
python cnn_train_compound_adagc.py --lambda-gc 0.5 \
   --aux-loss asl --aux-loss-weight 0.5 --asl-gamma-pos 1 --asl-gamma-neg 4 \
   --epochs 30 --batch 8 --model-tag M4

# M5: BCE warmup → ASL
python cnn_train_compound_bce.py --loss asl --warmup-loss bce --warmup-epochs 5 \
   --gamma-pos 1 --gamma-neg 4 --clip 0.05 \
   --class-weight effective --label-smoothing 0.05 \
   --epochs 30 --batch 8 --model-tag M5

# M6: Focal + ASL
# M7: AdaGC + ls=0.1
```

**최적값 찾기 — Mix 조합**:
- 단일 Best (M2 ASL) 측정 → mix 가 +2-5% 인지 검증
- M3 (CW + LS 추가) → marginal effect
- M4 (AdaGC + ASL) → loss 결합 효과
- M5 (BCE warmup → ASL) → 안정성 + 정확도
- M6 (Focal + ASL) → rare class 보강
- M7 (AdaGC + ls 0.1) → calibration + SPML

**Result table**:
```
results/stage4_mix.csv:
   M_id, base, CW, LS, other, val_f1, test_f1, multi_label_F1, mAP, rare_class_F1, ECE
```

### Stage 5 — Threshold tuning + Calibration (8 strategy)

#### Stage 5a — Calibration 분석

```bash
# 33 class × N val sample 의 sigmoid prob 분포 + reliability diagram
python _calibration_analysis.py \
   --model logs_compound/overall/best_model.pth \
   --val-set D:/project/data/wm-811k/unknown_multi \
   --output plots/calibration/

# 산출 확인
ls plots/calibration/
# 예상: dist_<class>.png × 33, reliability_overall.png, ece_per_class.png
```

#### Stage 5b — 8 strategy sweep (D1-D8)

`docs/multi-label/DECISION_RULE.md` 의 D1-D8 정의 참조.

```bash
# 각 strategy config YAML 작성
mkdir -p configs/

cat > configs/decision_D1.yaml <<EOF
base: default
calibration: none
top_k: null
floor: null
EOF

cat > configs/decision_D8.yaml <<EOF
base: knn_local
calibration: temp_platt_mix
top_k: 3
floor: 0.3
confidence: true
EOF

# 8 strategy 모두 실행
for d in D1 D2 D3 D4 D5 D6 D7 D8; do
   python _threshold_sweep.py \
      --config configs/decision_${d}.yaml \
      --val-logits logs_compound/overall/val_logits.npy \
      --val-y logs_compound/overall/val_y.npy \
      --test-logits logs_compound/overall/test_logits.npy \
      --test-y logs_compound/overall/test_y.npy \
      --output results/decision_${d}.json
done

# 종합
python _generate_decision_report.py --results-dir results/ \
   --output results/stage5_decision.csv
```

**최적값 찾기 — Threshold strategy**:
- D1 (default 0.5) baseline 측정
- D2-D5: per-class F1 + calibration 단일 효과
- D6-D8: adaptive threshold 효과
- ★ 학습 추가 0 — 가장 high-ROI

**Result table**:
```
results/stage5_decision.csv:
   D_id, base, calibration, top_k, floor, macro_f1, micro_f1, mAP, hamming, ECE,
   recall_rare_class, precision_common_class, confidence_distribution
```

### Stage 6 — Chip-wafer matching (7 mix combination)

`docs/multi-label/MATCHING_DESIGN.md` 의 C1-C7 정의 참조.

```bash
# config YAML 작성
cat > configs/matching_C7.yaml <<EOF
surface_ensemble: E6  # class-adaptive
ensemble_weights: per_class_best
crf: feature-aware    # chip obj similarity
consistency: strict
outlier_threshold: percentile_5
ambiguity_threshold: 2.0
EOF

# 7 combination 모두 실행
for c in C1 C2 C3 C4 C5 C6 C7; do
   python _eval_chip_matching.py \
      --config configs/matching_${c}.yaml \
      --surfaces-root _dist_heatmaps_per_class/ \
      --gt-root D:/project/data/wm-811k/unknown_multi/ \
      --output results/matching_${c}.json
done

# 종합
python _generate_matching_report.py --results-dir results/ \
   --output results/stage6_matching.csv
```

**최적값 찾기 — Matching ensemble**:
- C1 (E1 single) baseline
- C2-C3: ensemble 효과 (E2 hybrid → E3 weighted)
- C4-C5: CRF post-process 효과
- C6: stacking meta-learner
- C7: per-class adaptive (★ best 가설)

**Result table**:
```
results/stage6_matching.csv:
   C_id, surface, crf, consistency, outlier_th, matching_acc, outlier_rate,
   mismatch_rate, ambiguous_rate, per_class_acc, compute_time_per_wafer
```

### Stage 7 — Prod predict 보강

```bash
# cnn_predict_compound_prod.py 에 새 args 추가
# (구체 변경 detail 은 plan 본문 참조)

# Smoke test (기존 prod_test 데이터)
python cnn_predict_compound_prod.py \
   --image-root D:/project/data/prod_test/image_root \
   --positions-root D:/project/data/prod_test/positions_root \
   --multi-label-threshold-strategy sweep \
   --matching-method heatmap_smooth

# 산출 검증
python -c "
import pandas as pd
w = pd.read_parquet('result_compound/AB/K1AB/<date>/preds_wafer.parquet')
c = pd.read_parquet('result_compound/AB/K1AB/<date>/preds_chip.parquet')
print('wafer rows:', len(w), 'chip rows:', len(c))
print('matching status:', c.match_status.value_counts())
"
```

### Stage 8 — Master comparison

```bash
# 모든 stage CSV 종합
python _generate_master_report.py \
   --stage-results-dir results/ \
   --output-dir results/ \
   --plot-dir plots/

# 산출
ls results/
# 예상: master_table.csv, master_table.md, decision_guide.md
ls plots/
# 예상: master_comparison.png, per_class_f1_stacked.png, calibration_compare.png
```

---

## ★ 최적값 찾는 방법 (전 stage 공통)

### 1. Greedy ablation (단일 dimension sweep)

```
default 다른 dim → 한 dim 만 sweep → best 결정 → fix → 다음 dim
```

- ★ 본 ablation 의 표준 — Stage 2, Stage 4 hyperparameter sweep
- 41K full grid → 11 run greedy
- 단점: dim 사이 interaction 무시

### 2. Mix combination (★ 본 ablation contribution)

```
top 3 single + cross product (meaningful interaction)
```

- ★ 본 ablation 의 진짜 contribution
- LOSS_DESIGN.md M1-M7
- MATCHING_DESIGN.md C1-C7
- DECISION_RULE.md D1-D8
- 단점: 7-8 run 추가 (+5h)

### 3. Bayesian Optimization (Optuna)

```python
import optuna
def objective(trial):
   gp = trial.suggest_float("gamma_pos", 0, 3)
   gn = trial.suggest_float("gamma_neg", 1, 8)
   cl = trial.suggest_float("clip", 0, 0.2)
   return train_and_eval(gp, gn, cl)

study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=30)
```

- ASL γ 등 continuous hp 에 적용
- 30 trial 으로 grid 27 보다 효율적
- 학습 시간 30 × 60min = 30h (greedy 9 의 3.3배) — 신중

### 4. Per-instance adaptive (Stage 5 D8 KNN_local)

```
threshold(x, c) = α × global_signal(c) + (1-α) × local_signal(x, c)
```

- 가장 정교
- val embedding 보관 + 매 inference KNN search

---

## ★ Verification 표준

### 학습 후

```bash
# val/test F1
cat logs_compound/<run>/best_history.txt | head -30

# loss 안정성 (NaN, batch skip)
grep -i "nan\|skip\|guard" logs_compound/<run>/run.log

# val_macro_f1 ≥ 0.95 (Stage 2 표준)
python -c "
import json
h = json.load(open('logs_compound/<run>/history.json'))
print('Best val F1:', max(e['val_macro_f1'] for e in h))
"
```

### 추론 후

```bash
# multi-label F1 (Stage 4)
python _eval_multi_label.py --predictions <preds.json> --gt <manifest.csv>

# Threshold sweep (Stage 5)
cat results/stage5_decision.csv | column -t -s,
# 8 row × 6 metric

# Matching (Stage 6)
cat results/stage6_matching.csv | column -t -s,
# 7 row × 5 metric
```

### Plot 검증

```bash
# 모든 plot 존재 확인
ls plots/calibration/  # 35 (33+2)
ls plots/threshold/    # 33+1
ls plots/matching/     # 7+1
ls plots/master_comparison.png  # 1
```

---

## ★ Fallback 표준

### 학습 자원 부족
- compound MemoryError → cnn_train_wafer.py R-only
- GPU 점유 → cnn-master + resource-monitor team 활용
- Phase B/C 시간 부족 → epochs 20 (default 30) + early stopping patience 5

### 데이터 부족
- Stage 5 val sample 작음 → bootstrap CI
- Stage 6 chip-level GT 부족 → wafer 단위 surface (33 class) 만 사용
- Phase C 학습 데이터 부족 → unknown 12K (single-positive 변환) + unknown_multi 2K 통합

### Library 설치 어려움
- pydensecrf (Windows) → 직접 numpy 구현 (constant pairwise)
- Bayesian Optimization (Optuna) 안 됨 → greedy

### 모델 over/under-fit
- val/test gap > 0.05 → label_smoothing ↑, dropout ↑, augmentation 강화
- val_f1 < 0.90 → 학습 epoch ↑, hyperparameter 재조정

---

## 금지

- 학습 결과 폴더 무단 삭제 금지 (`logs_compound/<run>/`, `_dist_heatmaps_per_class/` 등)
- 새 실험은 새 model_tag — 기존 결과 덮어쓰기 금지
- ★ deep-dive doc (LOSS/MATCHING/DECISION_RULE) 의 mix 조합 정의 임의 변경 금지 — 사용자 결정 사항
- single SOTA 비교만 하고 mix 조합 skip 금지 — mix 가 본 ablation 의 진짜 contribution
- compound 학습 MemoryError 우회 위해 폴더 삭제 금지 — wafer R-only fallback 사용

## 반환

각 stage 실행 후:
- 산출 path (logs_compound/<run>/, results/<stage>.csv, plots/)
- best run 의 metric 요약 (val_f1, test_f1, multi_label_F1)
- 다음 stage 추천
- STATUS.md 갱신 (status `⏳ TODO` → `✅ COMPLETE`)
