# 05 — Results: compound vs wafer-only Margin

> **APPEND-ONLY.** 이 파일은 누적 기록. 한 번 적힌 row 절대 삭제·수정 금지 (수정도 새 row).
>
> 핵심 = compound test_f1 가 wafer-only test_f1 보다 높아야 한다. margin Δ = compound - wafer.
> converge gate (출처: `.claude/agents/orch-master.md`): margin ≥ 1.5 pp 가 2 round 연속 유지.

## 1. Margin 표 (round 별 누적)

| Round | compound test_f1 | wafer test_f1 | Δ (pp) | converge? | sc:sc-self-review |
|---|---|---|---|---|---|
| v3 (round 28, 2026-05-05) | TBD (학습 미실행) | TBD (학습 미실행) | — | — | — |

## 2. 각 row 의 상태

### v3 (round 28)

- **chip CNN (Stage 1) 완료**: val_f1 1.0000 / test_f1 0.9799 / 6.7 min (출처:
  `logs_chip/v3_chip_260505_142036_running/best_history.txt`)
- **obj_id_maps build (Stage 2) 진행 중**: 7965 / ~8600 (~92.6%)
- **wafer-only (Branch A) TBD**: `cnn_train_wafer.py` 미실행 (계획: round 28 후반 dispatch
  예상)
- **compound (Stage 3) TBD**: `cnn_train_compound.py` 미실행 (Stage 2 완료 + wafer-only
  완료 후 dispatch)

다음 row 갱신 시점: Branch A wafer-only 와 Branch B Stage 3 compound 양쪽 학습 완료 후
즉시 paper-scribe (event=round_end) trigger.

## 3. Compound vs V3 chipgrid 비교 (이전 protocol B 기록, 참고용)

(출처: `docs/wafer-ensemble/RESULTS.md` §Compound)

protocol B = 33-class 전체 + 0.8/0.2 split + n=220/cls. **현재 round 28 (protocol C)
와 직접 비교 불가** — 별 표.

| 측면 | compound 3ch BICUBIC 384 | V3 chipgrid 32 one-hot (참고) |
|---|---|---|
| 입력 해상도 | 384 | **32** |
| 입력 채널 | 3 (R+G+B) | **6 (R + 5 obj one-hot)** |
| obj_id 인코딩 | 정수 BICUBIC → categorical 깨짐 | **per-class binary, 보간 0** |
| 백본 | ConvNeXtV2-base 88M | tiny CNN 1.16M |
| 학습 데이터 | 5,680 train | 2,656 train (n=100/cls) |
| 학습 시간 | 8 min/epoch (compound 12 ep ≈ 1.5 h) | <1 sec/ep (V3 30 ep ≈ <1 min) |
| val_f1 | 0.9784 | **0.9689** |
| test_f1 | 0.9736 | **0.9879** |
| 종료 형태 | epoch 13 MemoryError crash | 정상 종료 |

이 비교는 **현재 round 28 의 fair-eval 비교가 아님**. round 28 compound vs wafer-only
비교는 같은 protocol C 위에서만 의미를 가진다. compound 가 wafer-only 를 못 넘으면
V3 chipgrid (block_expand) 으로 fallback 검토.

## 4. Per-class breakdown (TBD — 학습 후 갱신)

학습 완료 시 다음 항목 추가:
- per-class F1 (compound vs wafer-only) — Edge-Bottom × 5 / Edge-Top × 5 weak class 마진
- intra-distribution obj 식별 정확도 (출처: `docs/wafer-ensemble/INTRA_DIST.md` 형식)
- confusion matrix off-diagonal pair (compound vs wafer-only)

## 5. 운영 threshold 측정 (TBD)

`cnn-pipeline` skill 의 자동 추천 threshold (Normal pool max_prob 분포 기반) — round 28
결과 후 측정. (출처: `.claude/skills/cnn-pipeline/SKILL.md`)

## Cross-link

- 본질 statement (compound > wafer-only) → `docs/paper/01_motivation.md`
- 학습 결과 raw → `docs/paper/04_experiments.md`
- intra-distribution weak point → `docs/paper/06_analysis.md`
- iteration 다음 step → `docs/paper/07_iteration_log.md`
- compound-loop converge gate → `.claude/agents/orch-master.md`
