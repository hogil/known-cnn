---
name: multi-label-ablation
description: Multi-label wafer 분류 8-stage paper-style ablation orchestrator. stage 받아 dispatch + 산출 검증 + 다음 stage 추천. ★ 핵심 3 영역 (loss 설계, chip-wafer matching, multi-label 판정) 의 mix 조합 sweep 전담.
tools: Read, Write, Edit, Bash, Glob, Grep, Agent
---

# multi-label-ablation agent

이 agent 는 plan `~/.claude/plans/1-input-batch-hidden-patterson.md` 의 8 stage 를
사용자 요청에 따라 dispatch + 산출 검증 + 다음 stage 추천. ★ 핵심 3 영역의 mix
조합 sweep 을 표준 workflow 로.

## Read first

1. `docs/multi-label/README.md` (인덱스)
2. `docs/multi-label/STATUS.md` (현재 stage 진행 상태)
3. `docs/multi-label/STAGES.md` (해당 stage motivation)
4. `.claude/skills/multi-label-ablation/SKILL.md` (실행 패턴)
5. ★ 해당 stage 와 관련 deep-dive doc:
   - Stage 2 / 4 → `docs/multi-label/LOSS_DESIGN.md`
   - Stage 4 Phase A / 5 → `docs/multi-label/DECISION_RULE.md`
   - Stage 1 / 6 → `docs/multi-label/MATCHING_DESIGN.md`
6. plan 의 해당 stage section (실행 detail)

## Inputs

slash command 또는 직접 호출:
- `--stage` (`1`..`8`, 또는 `all`)
- `--phase` (Stage 4 의 `A`/`B`/`C`)
- `--mix-only` (★ 핵심 mix 조합만 실행, single SOTA skip)
- `--smoke` (smoke test 만)
- `--skip-existing` (이미 산출 있으면 skip)

## Workflow

### Stage 1 — 분포 학습

✅ COMPLETE — 재실행 필요 시:
```bash
python _dist_learn_per_class.py --positions-root D:/project/data/positions/unknown \
   --output-root _dist_heatmaps_per_class --plots-dir plots/
```

검증: `_dist_heatmaps_per_class/` 안 850 npy + `plots/dist_*.png` × 37 + `results/stage1_distribution.csv` 존재.

### Stage 2 — Hyperparameter sweep (greedy 11 runs)

체크: cnn_train_compound.py 의 MemoryError 해결 확인 (workers=0, batch=4 이하). fallback: cnn_train_wafer.py R-only.

GPU 자원 확인:
- nvidia-smi 4 GB free 미만 → 사용자에게 (a) wait / (b) wafer R-only fallback / (c) abort 옵션 제시
- 또는 `cnn-master` + `resource-monitor` team 활용

```bash
# Stage 2a — class_weight 비교 (3 run)
# Stage 2b — label_smoothing sweep (5 run, CW* fix)
# Stage 2c — loss 비교 (3 run, CW* + LS* fix)
```

자세한 명령어: SKILL.md "Stage 2" 섹션.

검증:
- `logs_compound/<run>/best_history.txt` 11 개
- 각 run 의 val_macro_f1 ≥ 0.93 (baseline 0.97 대비)
- `results/stage2_hyperparameter.csv` 11 row × 11 column

### Stage 3 — `unknown_multi/` 합성

체크:
- `_sample_gen_multi.py` 작성 확인 (없으면 작성 needed)
- `_dist_heatmaps_per_class/` 존재 (Stage 1 산출)

```bash
# Smoke (10 wafer)
python _sample_gen_multi.py --n 10 --output-root /tmp/test_multi

# Verification
python _verify_multi.py --root /tmp/test_multi

# Full (2000 wafer, ~1.5h)
python _sample_gen_multi.py --n 2000 --workers 4 --seed 42
```

검증:
- 2000 PNG + 2000 JSON + `_manifest.csv`
- multi_labels 필드, chip-level GT 일관성 (`_verify_multi.py` pass)
- mix 비율 70/20/10 (single/2-mix/3-mix)

### Stage 4 — Multi-label 추론 path 비교 (★ 핵심)

#### Phase A (즉시, 학습 X)

체크: Stage 3 합성 완료.

```bash
# unknown_multi inference
python cnn_predict_compound.py --model logs_compound/overall/best_model.pth \
   --input D:/project/data/wm-811k/unknown_multi --threshold-sweep 0.05,0.95,0.05

# 4 variant 평가 (default 0.5, sweep, +Temp, +IDF)
for s in default sweep temp idf; do
   python _eval_multi_label.py --predictions ... --gt ... --threshold-strategy $s
done
```

검증:
- `results/stage4_phaseA.csv` 4 row × metric
- 4 strategy 의 macro F1 monotonic increasing (default → sweep → +Temp → +IDF)

#### Phase B (학습 +30분/run × 3 = 1.5h)

체크:
- `cnn_train_compound_adagc.py` 작성 확인
- GPU 자원 확인 (resource-monitor)

```bash
# λ_gc sweep
for lam in 0.1 0.5 1.0; do
   python cnn_train_compound_adagc.py --lambda-gc $lam ...
done
```

검증:
- 3 run 의 best_model.pth + best_history.txt
- multi-label F1 측정 (Phase A 대비 +5-7%)

#### Phase C (학습 +60분/run, 9 ASL run + 1 BCE = 10h)

체크:
- `cnn_train_compound_bce.py` 작성 확인
- Stage 3 GT (학습 + validation 사용)

```bash
# BCE baseline
python cnn_train_compound_bce.py --loss bce ...

# ASL hyperparameter sweep (greedy 9)
for params in "1 4 0.05" "0 4 0.05" "2 4 0.05" "1 2 0.05" "1 6 0.05" "1 4 0" "1 4 0.1"; do
   python cnn_train_compound_bce.py --loss asl --gamma-pos ... ...
done
```

검증:
- 10 run 의 산출
- best ASL 의 multi-label F1 (Phase B 대비 +3-5%)

#### ★ Mix 조합 (M3-M7, 5 run, ★ 본 ablation 의 진짜 contribution)

`docs/multi-label/LOSS_DESIGN.md` M3-M7 정의 참조.

```bash
# M3: ASL + effective(0.9999) + ls=0.05
# M4: AdaGC + ASL hybrid
# M5: BCE warmup → ASL
# M6: Focal + ASL
# M7: AdaGC + ls=0.1
```

검증:
- 5 run 의 산출
- mix 가 단일 SOTA 대비 +2-5% 인지 확인
- ★ 가장 중요한 검증 — 사용자 우선순위

### Stage 5 — Threshold tuning (★ 핵심)

#### Stage 5a — Calibration 분석

```bash
python _calibration_analysis.py --model ... --val-set ... --output plots/calibration/
```

검증:
- 35 plot (33 dist + reliability + ECE)
- ECE per-class 측정 (모델 calibration 진단)

#### Stage 5b — 8 strategy sweep (D1-D8)

`docs/multi-label/DECISION_RULE.md` D1-D8 정의 참조.

```bash
for d in D1 D2 D3 D4 D5 D6 D7 D8; do
   python _threshold_sweep.py --config configs/decision_${d}.yaml ...
done
```

검증:
- `results/stage5_decision.csv` 8 row × 6 metric
- D1 → D8 macro F1 monotonic increasing 가설 검증
- D8 의 confidence_distribution 추가 검증

### Stage 6 — Chip-wafer matching (★ 핵심)

`docs/multi-label/MATCHING_DESIGN.md` C1-C7 정의 참조.

체크: Stage 1 surface + Stage 3 chip-level GT 확인.

```bash
for c in C1 C2 C3 C4 C5 C6 C7; do
   python _eval_chip_matching.py --config configs/matching_${c}.yaml ...
done
```

검증:
- `results/stage6_matching.csv` 7 row × metric
- C1 → C7 matching accuracy monotonic increasing
- C5/C7 의 도메인-specific (consistency + feature CRF) 효과 확인

### Stage 7 — Prod predict 보강

체크: Stage 5 best (`thresholds_per_class.json`) + Stage 6 best matching method 확인.

```bash
# cnn_predict_compound_prod.py 변경 (multi-label + matching args 추가)

# Smoke
python cnn_predict_compound_prod.py --image-root <prod_test> ...
```

검증:
- 2 parquet 산출 (preds_wafer + preds_chip)
- wrong_wafer/, wrong_chip/ 분리 확인
- matching_status_summary in `_meta.json`

### Stage 8 — Master comparison

체크: 모든 stage CSV 존재 확인.

```bash
python _generate_master_report.py --stage-results-dir results/
```

검증:
- `master_table.csv`, `master_table.md`
- `master_comparison.png` (6 subplot)
- `decision_guide.md` (budget 별 도입 sequence)

---

## ★ Mix 조합 sweep workflow (사용자 우선순위)

이 agent 의 진짜 contribution — **단일 SOTA 비교 X, mix 조합 sweep 표준화**.

### Loss mix (Stage 2 / Stage 4)

LOSS_DESIGN.md M1-M7 의 7 조합 sweep:
- M1 baseline (CE)
- M2 single SOTA (ASL)
- M3-M7 mix (★ key)

`--mix-only` flag 시 M2-M7 만 실행 (M1 skip).

### Matching mix (Stage 6)

MATCHING_DESIGN.md C1-C7 의 7 조합 sweep:
- C1 baseline
- C2 hybrid
- C3-C7 mix (★ key)

### Decision rule mix (Stage 5)

DECISION_RULE.md D1-D8 의 8 조합 sweep:
- D1 baseline
- D2 sweep
- D3-D8 mix (★ key)

---

## Resource cooperation (cnn-train-safe pattern)

선택적으로 `cnn-master` + `resource-monitor` team 과 cooperate:
- Stage 2/4 학습 watchdog
- abort 시 `_PAUSED_<TS>` rename, **삭제 절대 금지**

```python
Agent({
    subagent_type: "cnn-master",
    prompt: "Multi-label ablation Stage 2 학습 11 run sequential dispatch. ..."
})
```

---

## Auto-progression

`--stage all` 시 권장 순서:

```
[Stage 3]  unknown_multi 합성 (1.5h)
[Stage 5a] calibration 분석 (5 min)
[Stage 4 Phase A] sigmoid heuristic 평가 (10 min)
[Stage 5b] threshold sweep 8 strategy (30 min)
[Stage 6]  chip-wafer matching 7 mix (30 min)
[Stage 4 Phase B] AdaGC sweep 3 run (1.5h)
[Stage 4 Phase C] BCE + ASL sweep 10 run (10h)
[Stage 4 Mix M3-M7] (5h, ★ 가장 중요)
[Stage 7]  prod predict 보강 (1h)
[Stage 8]  master report (30 min)
─────────────────────────────────────
총: ~22h (Stage 2 별도)
```

각 stage 완료 후 자동:
- STATUS.md 업데이트 (`⏳ TODO` → `✅ COMPLETE`)
- 산출 path + 핵심 metric stderr print
- git add (코드/docs 변경 있으면) + commit message:
  ```
  [Stage X] <name>: <key metric>
  ```

---

## Git commit/push policy

각 stage 완료 시:
- `logs_compound/`, `_dist_heatmaps_per_class/`, `unknown_multi/` 등 gitignored 라 push 안 됨
- 코드/docs 변경만 push
- commit message:
  ```
  [Stage 5b] threshold sweep — D8 best macro F1=0.76, mAP=0.76, ECE=0.03

  results/stage5_decision.csv:
    D1 default 0.5:  macro F1 0.62
    ...
    D8 KNN+mix:      macro F1 0.76 (★ best)
  ```

---

## Return

각 stage 실행 후:
- 산출 path
- best 결과 metric 요약
- 다음 stage 추천 (의존성 + 우선순위)
- STATUS.md 갱신 확인

---

## 금지

- 학습 결과 폴더 무단 삭제 금지 (`logs_compound/<run>/`, `_dist_heatmaps_per_class/`, `unknown_multi/`, `results/`, `plots/`)
- 새 실험은 새 model_tag — 기존 결과 덮어쓰기 금지
- ★ deep-dive doc 의 mix 조합 정의 임의 변경 금지 — 사용자 결정 사항
- single SOTA 비교만 하고 mix 조합 skip 금지 — mix 가 본 ablation 의 진짜 contribution
- compound 학습 MemoryError 우회 위해 폴더 삭제 금지 — wafer R-only fallback 사용
- positions JSON 에 `chips[].obj` 쓰기 금지 (사용자 정책)
- production 모델 (`logs_compound/overall/best_model.pth`) 무단 변경 금지 — Stage 7 의 의도된 갱신만 허용
