# Paper draft — known-cnn

본 디렉토리는 `paper-scribe` agent 가 누적 기록하는 **논문 자료**.
모든 `.md` 는 **append-only** — 한 번 적힌 row 절대 삭제·수정 금지 (정정도 새 row).

## 핵심 목표 (본 프로젝트의 본질)

**compound (R+G+B = failbit + obj_id + zero) 가 wafer-only (R 만) 보다 wafer 분류 test_f1 가 높도록** — chip-level obj 정보 (chip CNN inference 결과) 가 wafer-level 분류에 보탬을 주는지 검증.

다양한 확률 분포 / loss 설계 / matching / threshold 등은 모두 이 목적을 위한 수단.

## 디렉토리

| 파일 | 내용 | 갱신 trigger |
|---|---|---|
| `README.md` (이 파일) | 인덱스 + 현재 상태 | 매 trigger |
| `00_abstract.md` | 1-paragraph 추상 | 진행도에 따라 |
| `01_motivation.md` | 동기 — compound > wafer-only | 1회 작성 + 정책 변경 시 |
| `02_method.md` | 3-stage 아키텍처 (chip CNN → obj_id maps → compound) + LR/optimizer 정책 | design decision |
| `03_data_synthesis.md` | round 1-28+ history, alpha 함수, 분포 heatmap, sample 분포 | data_synth_round |
| `04_experiments.md` | per-version run 표 (chip / wafer-only / compound) | training_run |
| `05_results.md` | margin trend (compound - wafer), per-class | round_end |
| `06_analysis.md` | weak class root cause, ablation, intra-distribution, GPU resource lesson | analysis |
| `07_iteration_log.md` | round 끝마다 next_action + rationale 누적 | round_end |
| `08_conclusion.md` | 진행 따라 점진 작성 | converge / 사용자 명시 |
| `figures/` | curves, confusion matrix, sample wafer PNG, distribution plot | training_run / data_synth |
| `CHANGELOG.md` | 시간순 한 줄 entry (single source of truth) | **매 trigger** |

## 현재 상태 (자동 갱신, paper-scribe 가 마지막 호출 시 update)

| 항목 | 값 |
|---|---|
| 마지막 갱신 | (미갱신 — bootstrap mining 대기) |
| 현재 round | v3 (data 합성 100% 완료, obj_id_maps build 84% 진행 중) |
| Best compound test_f1 | TBD (Branch B Stage 3 미실행) |
| Best wafer-only test_f1 | TBD (Branch A 미실행) |
| Best margin Δ | TBD |

## 인용 정책

- 사용자 발화 한국어 그대로 (번역 X) + (출처: 대화 시점) 표기
- 코드 / 데이터 path 인용 시 (출처: `D:/project/...`) 명시
- 외부 참조 (anomaly-detection 등) 명시: (참조: `D:/project/anomaly-detection/...`)
- `outputs/results_master.csv` 와 `04_experiments.md` row 동기화 (paper-scribe 가 검증)

## 워크플로우 ↔ paper

```
canvas-verify → paper-scribe (event=data_synth_round)
                  ↓
                03_data_synthesis.md row + figures/round{n}_<class>_sample.png

cnn-master 학습 끝 → paper-scribe (event=training_run)
                       ↓
                     04_experiments.md row + figures/v{n}_{kind}_cm.png

result-trace → paper-scribe (event=round_end)
                 ↓
               05_results.md margin row + 07_iteration_log.md next_action

compound-review → 자체 결정만, paper-scribe 가 read 해서 07 에 기록
```
