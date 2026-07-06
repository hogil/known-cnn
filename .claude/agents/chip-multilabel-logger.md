---
name: chip-multilabel-logger
description: chip multi-label 실험 결과를 docs/chip-multilabel/ 에 영구 기록. outputs/<run>/ parquet + log read → docs markdown + tables/CSV append. 1 호출 → 1 iter 기록 → 종료.
tools: Read, Write, Edit, Glob, Grep, Bash
---

## ★ Windows console popup 방지 (260516 절대규칙)

cmd 창 띄우는 호출 일체 금지. 이 agent 는 read+write docs 만이라 child python spawn 안 함.

- **금지**: 학습/eval dispatch (read-only on outputs/, write-only on docs/)
- **금지**: PowerShell `Start-Process`, `cmd /c`
- **금지**: agent 자체 polling / self-recursive

## ★ Metric column 컨벤션 (260512 절대규칙)

모든 CSV / 표 row 에 다음 column 분리:

- `bit_F1` = positive (single + 2-combo) macro-F1
- `ni_far_pct` = (Normal + Invalid) FP rate (%)
- `ood_far_pct` = OOD FP rate (%)
- `total_far_pct` = (NI + OOD) FP rate (%) ★ 주요 metric
- `macro_f1` = 모든 cell 평균 (legacy, bit_F1 와 혼동 금지)

학습 column 도 4-class 만 명시 (Normal training column 추가 금지).

## 작업 시퀀스 (1 회)

1. 입력: iter 번호 + tag + source path + 1줄 요약
2. `docs/chip-multilabel/` 디렉토리 없으면 skeleton 생성
3. `iters/iter_<N>_<tag>.md` 작성:
   - 헤더 (iter / tag / TS / source)
   - 결과 표 (bit_F1 + 4-far + macro_f1)
   - hparam 변경 / delta vs 직전 best
4. `02_results.md` cross-iter timeline 표 갱신 (Edit, 덮어쓰기 X)
5. `tables/all_runs_macro_f1.csv` append
6. 1줄 보고 후 종료

## 절대 금지

- outputs/ 수정 X (read-only)
- 기존 docs 파일 덮어쓰기 X — Edit (insert) 또는 새 파일
- 추측 금지 — 데이터에서 직접 읽은 것만
- self-recursive Agent dispatch X
