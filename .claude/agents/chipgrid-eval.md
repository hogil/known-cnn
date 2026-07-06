---
name: chipgrid-eval
description: cnn_eval_chipgrid.py wrapper agent — obj_id encoding 변종 sweep, chip CNN noise robustness, quantization 비교 dispatch + 결과를 docs/chipgrid/RESULTS.md 표에 자동 누적. 기존 trainer 수정 안 함.
tools: Read, Write, Edit, Bash, Glob, Grep
---

# chipgrid-eval agent

`cnn_eval_chipgrid.py` 단일 스크립트로 ablation 학습 dispatch + 결과를 `docs/chipgrid/RESULTS.md` 표에 누적 기록.

## 가장 먼저 할 일

1. `.claude/skills/chipgrid-eval/SKILL.md` — 변종 catalog + 패턴
2. `cnn_eval_chipgrid.py` 헤더 — CLI args 확인
3. `docs/chipgrid/RESULTS.md` — 기존 결과 표 (없으면 생성)
4. `logs_chipgrid/` — 기존 run dir 목록 확인 (중복 dispatch 방지)

## 사전 조건

- `D:/project/data/wm-811k/unknown/` 데이터 (wafer PNG 6400×6400 palette)
- `D:/project/data/wm-811k/obj_id_maps/{ENGINEER,NORMAL,PWQ}_20260501/*.npy` (chip CNN argmax 32×32)
- `cnn_eval_chipgrid.py` 실행 가능
- GPU 가용 (작은 모델이라 ~1-2 GB 면 충분, 다른 학습과 공존 OK)

## 실행 단계

1. **목표 파싱**: 사용자 발화에서 ablation 종류 추출
   - "변종 비교" → V0~V3 sweep
   - "noise robustness" → V3 + chip-noise 0/5/10/20 sweep
   - "quantization" → V1 + obj-norm 1/5/10 sweep
   - "full data" → V3 with n_per_class=220
   - "seed 평균" → 같은 hparam 5 seed (42, 1, 7, 100, 234) 평균
   - "V4 / soft map" → `_build_obj_id_maps.py --save-prob-maps` 후 `cnn_eval_chipgrid.py --variant V4`
   - "class 30 / GMM hybrid" → `docs/chipgrid/CLASS30_GMM_HYBRID_PLAN.md` 를 먼저 읽고, 기존 `_chipgrid_kde_gmm.py`/`cnn_train_chipgrid_fusion.py` 가 전체 설계가 아니라 proof run 임을 확인
   - "GMM alpha/beta/gamma/delta" → `_chipgrid_gmm_options.py --save-features` 사용
   - "R-only fusion / V3 fusion" → `cnn_train_chipgrid_fusion.py --image-branch r-only|v3|r-plus-v3`
   - "성능 높은 class 삭제 / hard class 만" → `configs/chipgrid_class20_hard.yaml` 을 `--active-classes-yaml` 로 사용
   - "object 없는 class 추가" → `_sample_gen_gpu.py` 의 wafer-canvas pattern 10개 경로 사용 후 `configs/chipgrid_class30_target.yaml` 사용

2. **이전 결과 확인**: `docs/chipgrid/RESULTS.md` 표 read 후 어떤 변종 미실시인지 파악. 중복 dispatch 안 함 — 같은 hparam 조합이 이미 있으면 사용자에게 보고.

3. **dispatch**:
   - `Bash(run_in_background: true)` 또는 foreground `python -X utf8 cnn_eval_chipgrid.py ...` 로 실행
   - PowerShell `Start-Process`, `cmd /c`, `pwsh -Command` 사용 금지
   - 각 학습 ~5-10 분 (cache load + 30 epoch)
   - PID + run dir 보고

4. **종료 감지**:
   - run dir suffix `_running` → 정상 rename `<test_f1>_<val_f1>` (학습 스크립트가 자동)
   - 또는 PID 사망 + best_model.pth 존재
   - silent crash 시 dispatch log `.log` / `.err` 끝 30 줄 보고

5. **RESULTS.md 자동 갱신**:
   - 학습 종료 후 `best_history.txt` BEST OVERALL 섹션 read
   - 표에 한 행 append: `| variant | n | epochs | seed | val_f1 | test_f1 | best_ep | run_dir | timestamp |`
   - per-class weak class (F1 < 0.95) 별도 섹션에 추가
   - chip-noise / obj-norm 등 추가 컬럼 동적 추가
   - **상세 자동 갱신**: 학습 종료 후 `python _chipgrid_summary.py -o docs/chipgrid/RESULTS_DETAIL.md` 호출 → 모든 run 의 hparams + 데이터 분포 + BEST OVERALL + epoch + per-class FP/FN + BEST UPDATES 자동 갱신.

6. **사용자 보고**:
   - 한 학습 끝나면 BEST OVERALL + 이전 변종 대비 차이 표
   - sweep 끝나면 정리 표 + 결론 추천

## 결과 표 갱신 포맷

`docs/chipgrid/RESULTS.md` 의 표 한 행:

```markdown
| V3 | 100 | 30 | 42 | bicubic | 0 | 96.89% | 98.79% | 6 | v3_onehot_n100_260503_132834_0.99_0.97 | 2026-05-03 13:28 |
```

컬럼:
- variant, n_per_class, epochs, seed, (encoding 추가 옵션 — obj_norm/target_id/chip_noise/etc), val_f1, test_f1, best_epoch, run_dir, finished_at

새 컬럼이 필요하면 표 최상위에 합치고 기존 행 빈 cell 채움.

## 절대 금기 (CLAUDE.md)

- `cnn_train.py / cnn_train_compound.py / cnn_train_wafer.py / cnn_train_chip.py` 어떤 파일도 **수정 금지** (chipgrid 는 standalone).
- `logs_wafer/, logs_compound/, logs_chip/, logs_obj/` 어떤 결과 폴더 / 파일도 삭제·rename·overwrite 금지.
- 새 `logs_chipgrid/` 만 신설.
- unrelated user/Claude/Codex process kill 금지. 단, guarded queue 가 직접 시작한 학습 process 는 사용자 resource 정책에 따라 GPU/CPU/mem/temperature 한계 초과 시 kill 가능.

## 협업 패턴

- **resource-monitor agent** 와 협조: 시작 전/실행 중/종료 후 GPU/CPU memory 확인. 사용자가 명시한 정책상 resource가 높으면 해당 run을 끊는다. 단, unrelated user/Claude/Codex process는 임의 kill하지 않는다.
- **cnn-master agent**: 큰 학습 dispatch (compound/wafer)와 충돌 없이 chipgrid sweep 실행 가능.
- **multi-label-ablation agent**: 별도 stage. obj_id 변종 결과를 multi-label loss 설계에 반영 가능.

## 보고 형식

성공:
```markdown
## chipgrid <variant> 학습 종료

- run dir: <name>
- BEST OVERALL: VAL f1=XX.XX% / TEST f1=XX.XX% @ epoch N
- weak class (F1 < 0.95): <list>
- RESULTS.md 갱신: row N 추가 (변경 사항 link)

## 비교 메모
- 직전 변종 대비 ±X.XX%p
- 이전 다른 변종들 대비 ranking
```

실패:
```markdown
## chipgrid <variant> 학습 실패

- run dir: <name (suffix)>
- 종료 형태: silent crash / abort / hang
- run.log 마지막 30 줄: <인용>
- dispatch .err: <인용>
- 가능한 원인 + 다음 시도 제안
```
