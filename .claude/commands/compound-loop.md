---
description: compound > wafer-only iterative training loop (orch-master dispatch)
---

# /compound-loop

본 프로젝트의 핵심 목적 — **compound test_f1 > wafer-only test_f1 + margin** — 도달까지
v1 → v2 → v3 ... 자동 iterative 학습 + aggregation + next-action 결정.

`orch-master` agent 를 entry point 로 호출. 매 round 마다:

```
canvas-verify (데이터 sanity)
   ↓
[Branch A] wafer-only baseline (1 stage)
   ↓
[Branch B] chip CNN → obj_id_maps build → compound (3 stage chain, ★)
   ↓
result-trace (cross-run aggregation → outputs/results_master.csv + margin_history.csv)
   ↓
compound-review (margin gate Δ = compound - wafer; next_action 결정)
   ↓
cnn-analyze (per-class 진단)
   ↓
cnn-plan + sc:sc-performance-engineer (v_{n+1} 권장)
   ↓
converge or loop
```

## Args (`$ARGUMENTS`)

- `--max-rounds N` (default 5)
- `--converge-margin F` (default 1.5pp)
- `--converge-window N` (default 2 round 연속 유지)
- `--start-round N` (default 1, resume 지원)
- `--n-per-class-chip N` (default 100)
- `--n-per-class-wafer N` (default 50)
- `--active-classes-yaml PATH` (optional, 미지정 시 unknown/ 모든 class)
- `--skip-stage 1,2` (optional, chip / obj_id_maps 재사용)
- `--auto-loop` (default off — round 끝마다 사용자 confirm 받음)
- `--use-sc` (default on — super claude bridge agent 활성)

## 예

```bash
# 기본: 5 round 까지, margin 1.5pp 2 round 연속 유지 시 converge, 사용자 confirm 모드
/compound-loop

# 자동 loop, max 3 round
/compound-loop --max-rounds 3 --auto-loop

# v3 부터 resume, chip CNN + obj_id_maps 는 v3 이미 있어서 skip
/compound-loop --start-round 3 --skip-stage 1,2

# super claude 호출 비활성 (비용 절감)
/compound-loop --max-rounds 3 --use-sc off
```

## 정지 조건

- converge: margin >= --converge-margin 가 --converge-window round 연속 유지
- max-rounds 도달
- 자원 watchdog abort 3회 이상 (resource-monitor)
- 학습 자체 fail (loss NaN / OOM 반복)
- 사용자 Ctrl-C

각 정지 시 `outputs/orch_summary.md` 작성 — 전체 round 결과 + best compound + 사용 hparam.

## 자원 가드

`cnn-master` + `resource-monitor` team (`team_name=cnn-team`) 자동 사용:
- RAM 80% / GPU 90% / CPU 90% polling (60s)
- 한계 초과 시 process kill + `_PAUSED_<TS>` rename + 자원 회복 polling + 새 model_tag 재시작

## 결과 폴더 보존

각 round 의 `logs_chip/v{n}_*`, `logs_compound/v{n}_*`, `logs_wafer/v{n}_*` 모두 보존
(글로벌 룰: 학습 결과 무단 삭제 금지). 새 round = 새 model_tag.

## 다음 step

`/compound-loop` 종료 후:
- `outputs/orch_summary.md` 검토
- `outputs/results_master.csv` per-version margin trend 확인
- 사용자 결정:
  (a) converge → production 모델 = `outputs/logs_compound/<best_version>/best_model.pth`
  (b) iterate 계속 → `/compound-loop --start-round <next> --max-rounds N`
  (c) 정책 변경 → 새 active class / 새 backbone / 새 loss 후 다시 `/compound-loop`
