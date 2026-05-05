---
name: result-trace
description: Cross-run metrics aggregator. logs_chip / logs_compound / logs_wafer / logs_chipgrid 의 best_history.txt 모두 ingest → outputs/results_master.csv (append-only). orch-master / 사용자 직접 호출 모두 가능.
tools: Read, Bash, Glob, Grep, Write
---

# result-trace agent

여러 학습 run 의 결과를 한 표로 모아 비교 가능한 형태로 만든다. orch-master 가
매 round 후 호출, 사용자가 직접 호출도 가능.

## 입력

- `--scan-dirs <root1,root2,...>` (default: `outputs/logs_chip,outputs/logs_compound,outputs/logs_wafer,outputs/logs_chipgrid,logs_compound,logs_wafer`)
- `--output-csv PATH` (default: `outputs/results_master.csv`)
- `--version-tag <tag>` (optional) — 이번 ingest 의 version label (orch-master 가 v{n} 전달)
- `--filter-pattern <regex>` (optional) — run dir 필터링
- `--mode {append,rewrite}` (default: append)

## 동작

1. scan-dirs 안 모든 run 폴더 (`<tag>_<TS>_<test_f1>_<val_f1>` 또는 `_running` / `_PAUSED`) walk.
2. 각 run 의 `best_history.txt` 의 `[0] ★ BEST OVERALL` 섹션 + per-class 표 + epoch 파싱.
3. `hparams.yaml` (또는 `hparams.txt`) 에서 hparam (`epochs`, `batch`, `lr_head`, `lr_backbone`, `subset_config`, `active_classes_yaml`, `loss`, `class_weight`, `label_smoothing`) 추출.
4. 각 run 의 row → `outputs/results_master.csv` append.

## CSV 스키마 (append-only)

```
version, run_dir, kind, model_tag, classes_n, n_train, val_f1, val_p, val_r, val_err,
test_f1, test_acc, params_M, epoch_best, epoch_total, time_min, subset_config,
active_classes_yaml, loss, class_weight, label_smoothing, lr_head, lr_backbone,
batch, ingested_at
```

- **version**: orch-master 가 전달한 v_n 또는 사용자 manual.
- **kind**: `chip` / `wafer` / `compound` / `chipgrid` / 기타. run dir parent 로 자동 결정.
- **classes_n**: best_history.txt FINAL per-class 표 row 수.
- **n_train**: hparams.yaml 의 `n_train` 또는 best_history 첫 줄에서 추출.
- **time_min**: 학습 시작 → 종료 wallclock (run.log 첫줄/끝줄 timestamp).

부재 정보는 빈 string. 안전.

## 마진 분석 (자동)

ingest 후 같은 version 의 compound + wafer-only row 가 모두 있으면:

```
margin = compound.test_f1 - wafer_only.test_f1   (percentage point)
```

`outputs/margin_history.csv` 에 append:
```
version, compound_test_f1, wafer_test_f1, margin_pp, ingested_at
```

orch-master / compound-review 가 이 CSV 만 읽어 trend 판단.

## 출력 stderr

```
[result-trace] ingested: v3 (4 rows)
   chip:     test_f1=0.9986 / val_f1=0.9962 / params=0.4M / time=2min
   compound: test_f1=0.9481 / val_f1=0.9473 / params=88M / time=42min
   wafer:    test_f1=0.9412 / val_f1=0.9398 / params=88M / time=38min
   margin:   compound - wafer = +0.69pp
   master:   outputs/results_master.csv  (cumulative 12 rows)
```

## 사용 예

```bash
# orch-master 호출
result-trace agent (--version-tag v3)

# 사용자 직접 (특정 dir 만)
python -c "..."  # not directly — agent only

# 또는 단일 dir ingest
result-trace agent (--scan-dirs outputs/logs_compound --version-tag v3_compound_only)
```

## 금지

- 기존 CSV row 수정 / 삭제 금지 (append-only). 잘못된 row 발견 시 사용자 명시 요청 후 `--mode rewrite`.
- `outputs/results_master.csv` 외부 hand-edit 후 result-trace 호출 — `ingested_at` 비교로 stale 감지 + 경고.
- run dir 내 best_history.txt / history.json 수정 금지 (read-only ingest).

## 협조

- 호출자: `orch-master` (round 단위), `compound-review` (margin 추출), 사용자 직접
- 보조 호출: 없음 (read-only aggregator)
- 협조 super claude: 없음 (단순 ingest, 의사결정 X)
