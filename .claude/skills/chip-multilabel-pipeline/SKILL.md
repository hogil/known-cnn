---
name: chip-multilabel-pipeline
description: chip 200x200 single-label train -> multi-label predict 평가 파이프라인. classification_chips/ 4 class (bank_boundary, fork, scratch, scratch_rot) 학습 + 합성 12-class eval set (4 single + 6 combo + Normal + Invalid) 평가 매트릭스. ★ 단일 master 폴더 chip_multilabel/ + manifest 기반 runtime sampling (subset 폴더 절대 안 만듦). 저장 (master) defect 200/class 강한 것만, 사용 (eval) runtime --n-per-class 50. Stage 1 = 기존 모델 + inference variants (I0-I11; I5 TTA 영구 금지), Stage 2 = T1/T4/T5/T6 학습 후 inference 매트릭스. 결과는 outputs/stage1_<TS>/ 또는 outputs/stage2_<TS>/ 의 results_matrix.parquet + report.md.
---

## ★ 단일 master 폴더 정책 (260506)

**저장 (master)** = `D:/project/data/wm-811k/chip_multilabel/`
- defect 10 class × **200** chip (강한 것만, `--source-strength-pct 50`)
- Normal × **200** chip
- Invalid × **50** chip
- 총 **2450 chip** + `manifest.csv` (defect_pixel_ratio 컬럼 포함)

**사용 (eval at runtime)** = manifest 에서 sample
- `run_stage1.py --n-per-class 50` → defect 10×50 + Normal 50 + Invalid 50 = 600 chip
- `--strength-min 0.1 --strength-max 0.5` → defect_pixel_ratio 범위 슬라이스 가능
- `--include-classes bank_boundary,fork,Normal` → class subset
- subset 폴더 **절대 안 만듦** (memory rule `feedback_no_subset_archive_folders.md`, `feedback_master_storage_vs_runtime_sampling.md`)

## 사용 시나리오

### 1. (한 번만) master 합성

```bash
python -X utf8 -m chip_multilabel.gen_eval_set \
  --out-root D:/project/data/wm-811k/chip_multilabel \
  --per-defect 200 --per-normal 200 --per-invalid 50 \
  --source-strength-pct 50 --seed 42
```

→ 2450 chip, 12 폴더 + `manifest.csv` + `_preview/`. 강한 defect (top 50% by defect_pixel_ratio) 만 source.

### 2. Stage 1 inference (runtime sampling)

```bash
# 기본 — n_per_class=50 (defect 10×50 + Normal 50 + Invalid 50 = 600 chip)
python -X utf8 -m chip_multilabel.run_stage1 \
  --eval-set D:/project/data/wm-811k/chip_multilabel \
  --n-per-class 50 \
  --out-root outputs --batch-size 32 \
  --variants I3,I7,I10,I11

# defect 만 강한 것 (defect_pixel_ratio >= 0.15)
python -X utf8 -m chip_multilabel.run_stage1 \
  --eval-set D:/project/data/wm-811k/chip_multilabel \
  --n-per-class 50 --strength-min 0.15 \
  --out-root outputs --batch-size 32

# Normal 압도적 (real-env 80% 비율 모사) — Normal 200 + defect 10×50
python -X utf8 -m chip_multilabel.run_stage1 \
  --eval-set D:/project/data/wm-811k/chip_multilabel \
  --n-per-class 200 --include-classes Normal \
  --out-root outputs --batch-size 32
# (그 다음 별도로 defect 만 50, 합산)
```

→ 9 cell 매트릭스 (~1-3분 GPU). batch=32 → ~2GB GPU.

### 3. Stage 2 학습 + inference 매트릭스

```bash
python -X utf8 -m chip_multilabel.run_stage2 \
  --eval-set D:/project/data/wm-811k/chip_multilabel \
  --n-per-class 50 \
  --epochs 8 --batch 16 --accum 2
```

→ 4 train × 9 inference = 36 cell (~30분).

### GPU 사용량 target (260506)

다른 python 작업 (contrastive 등) 외 **chip-multilabel 만으로 GPU 30%** (~5GB / 16GB) 점유.
- 학습: `--batch 16 --accum 2` (effective 32 동일) — 약 5GB
- 추론: `--batch-size 32` — 약 2GB
- 시작 후 `nvidia-smi` 로 monitor — 35%↑면 batch ↓, 25%↓면 batch ↑

## Hard rules (위반 금지)

- **TTA (4-view averaging) 영구 금지** — chip 회전 의존적 (scratch vs scratch_rot). iter 1 실측 -0.018 macro_f1.
- **Rotation/Flip aug 학습 영구 금지** — RandomAffine translate+scale 만.
- **subset/archive 폴더 만들기 금지** — manifest + runtime sampling 만.
- **Master 저장 ≠ runtime 사용** — defect 200 저장, runtime --n-per-class 50 사용. 절대 storage = usage 로 만들지 않음.
- **outputs/ 결과 폴더 무단 삭제 금지** — CLAUDE.md global.
- **1 atomic method change / iter** — paper protocol.
- **decision rule 변경**: 새 I-variant 추가만 (I7/I10/I11 덮어쓰기 금지).

## 결과 해석 가이드

| 메트릭 | 의미 | 기준선 |
|---|---|---|
| macro_f1 (4-multi) | 4-dim multi-hot per-class binary F1 평균 | T9d 0.9687 (strong-50, 11-class) |
| **10-defect macro F1** ★ | 4 single + 6 combo per-class F1 평균 (Normal/Invalid 제외) | T9d__I3 = **0.9095** (260506 baseline, 12-class with sc+sr) |
| top1_11class | 11-class exact match | T9d 0.92 |
| ECE pre/post | calibration error | < 0.05 desirable |
| temperature | scalar T (I4/I9) | T<1 sharpens |

## 12 class breakdown (T9d__I3 strong-50 latest, 260506 첫 측정)

| class | F1 | 비고 |
|---|---:|---|
| bank_boundary | 0.904 | FP 17.5% |
| fork | 0.973 | 강함 |
| scratch | 0.894 | FP 19% |
| scratch_rot | 0.821 | **FP 30% — 가장 약점 single** |
| bank_boundary+fork | 0.849 | recall 26% missing |
| bank_boundary+scratch | 0.958 | 강함 |
| bank_boundary+scratch_rot | 1.000 | I11 pair-aware 효과 |
| fork+scratch | 0.968 | 강함 |
| fork+scratch_rot | 0.974 | 강함 |
| **scratch+scratch_rot** ★ | **0.755** | **39% missing — sc+sr CutMix disallowed 학습 결과** |
| Normal | 0.937 | I10 entropy gate 있어도 12% defect 누출 |
| Invalid | 1.000 | heuristic 완벽 |

## 오답 분석

`outputs/<run>/errors/<cell_id>/<error_type>/<chip>.png` 와 sidecar `.json`
- `false_positive_<class>` / `false_negative_<class>` / `wrong_combo` / `missed_invalid` / `missed_normal`

## Agent 호출

세션 재시작 후 4 agent 자동 등록:
- `chip-multilabel-runner` — 학습/inference dispatch + GPU 가드
- `chip-multilabel-analyst` (opus) — 결과 분석 + 다음 실험 제안
- `chip-multilabel-logger` — notes.md / docs 업데이트
- `chip-multilabel-paper-narrator` (opus) — paper section narrative

```
Agent(subagent_type='chip-multilabel-runner', prompt='Stage 1 만 돌리고 best cell 보고')
Agent(subagent_type='chip-multilabel-analyst', prompt='최신 stage1 결과 분석 + 다음 실험 제안')
```

## 자료 + 의견 누적

- 실시간 의견 / hparam 실험: `chip_multilabel/notes.md` 에 iter 단위 append
- paper-grade 결과: `docs/chip-multilabel/iters/iter_<N>_*.md`
- canonical 표: `docs/chip-multilabel/02_results.md`, `docs/chip-multilabel/tables/all_runs_macro_f1.csv`
