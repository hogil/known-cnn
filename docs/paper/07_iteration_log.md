# 07 — Iteration Log

> **APPEND-ONLY.** 이 파일은 누적 기록. 한 round 의 row 는 그 round 가 끝난 시점에 한 번
> 적힌다 — 절대 삭제·수정 금지 (정정도 새 row).
>
> 매 round end 에 paper-scribe (event=round_end) trigger 시 새 row 추가.

## 1. round 표

| Round | data v | chip CNN | wafer-only | compound | margin Δ | next_action | rationale |
|---|---|---|---|---|---|---|---|
| v3 (round 28) | 9600 wafer (43 active class, 200/cls + Normal 1000) | val_f1 1.0000 / test_f1 0.9799 / 6.7 min ✅ | TBD (학습 미실행) | TBD (Branch B Stage 3 미실행, obj_id_maps build 80%+ 진행 중) | TBD | TBD (Branch B Stage 3 미실행) — Stage 2 obj_id_maps build 완료 (≥99% .npy) → Branch A wafer-only dispatch → Branch B Stage 3 compound dispatch → margin 측정 → orch-master converge 판단 | round 28 의 LR spec 변경 (AD reference: lr_backbone 2e-5, lr_head 2e-4, warmup 5, start_factor 0.05) 적용 후 첫 fair-eval. Stage 1 chip CNN 이 100% saturated (5-class on n=100 subset) — Stage 2/3 의 obj_id distillation 은 충분히 신뢰 가능. compound > wafer-only 가 정말 입증되는지가 핵심. |

## 2. round 28 의 next_action 상세

### 2.1 즉시 next (Branch A — wafer-only baseline)

```bash
# protocol C (active 미적용, 전 unknown class)
python cnn_train_wafer.py --epochs 30 --batch 16 --model-tag wafer_v3 \
    --lr-backbone 2e-5 --lr-head 2e-4 --warmup-epochs 5
```

dispatch 책임자: `cnn-master` agent (auto via `/cnn-train-safe`). 결과:
`logs_wafer/wafer_v3_<TS>_<test>_<val>/`. 예상 학습 시간: ~5-6 h (R-only ConvNeXtV2 base
12.5 h on full 5680 → 9600 sample 비례 ~21 h... but 1 epoch 단축 가능; ConvNeXtV2 base
@1024 channels_last + bf16 RTX 4060 Ti 16GB).

### 2.2 Branch B Stage 3 (compound)

```bash
# Stage 2 obj_id_maps build 완료 (≥99% npy) 확인 후
python cnn_train_compound.py --epochs 30 --batch 16 --model-tag compound_v3 \
    --lr-backbone 2e-5 --lr-head 2e-4 --warmup-epochs 5 \
    --g-channel-mode default
```

dispatch 책임자: `stage3-compound` skill. 결과:
`logs_compound/compound_v3_<TS>_<test>_<val>/`.

### 2.3 round_end 직전 trigger sequence

```
result-trace
    ↓
compound-review (자체 결정만)
    ↓
paper-scribe (event=round_end)
    ↓
docs/paper/05_results.md margin row 추가
docs/paper/07_iteration_log.md row 추가 (이 파일)
docs/paper/CHANGELOG.md round_end timestamp
    ↓
orch-master converge 판단:
    margin ≥ 1.5 pp 가 2 round 연속? → CONVERGE 로 종료, 08_conclusion.md 작성
    아니면 → round 29 (v4) spec 결정 (compound-review 권고 따라)
```

## 3. round 28 진입 전 변경 history (compact)

(자세한 timeline → `CHANGELOG.md`)

| Date | Event |
|---|---|
| 2026-04-30 | known-cnn 자매 repo 분리 (initial commit `247b640`) |
| 2026-05-03 | V3 chipgrid 1.16M val_f1 0.9946 발견 (oracle ceiling 추월) |
| 2026-05-03 | 33 → 20 active + 14 archive 결정 |
| 2026-05-05T11:38 | experiments/ 폴더 restore (commit `6b825c7`) |
| 2026-05-05T13:53 | particle_blast → fork, scratch_21deg → scratch_rot rename (commit `f359fee`) |
| 2026-05-05T14:06 | compound-loop orchestration team 도입 (4 agents + slash, commit `c3356f9`) |
| 2026-05-05T14:20 | chip CNN v3 dispatched |
| 2026-05-05T14:27 | chip CNN v3 완료 (val_f1 1.0000 / test_f1 0.9799, 6.7 min) |
| 2026-05-05T14:30 | canvas v2 + obj_id_maps build dispatched |
| 2026-05-05T14:48 | CenterCircle round 28 alpha redesign + 사용자 OK |
| 2026-05-05T15:00 | 모든 unknown/ class 200/200 통일 완료 (43 class × 200 + Normal 1000) |
| 2026-05-05T15:15 | LR policy update (AD reference) — 3 trainer 모두 |
| 2026-05-05T15:30 | paper-scribe agent + docs/paper/ skeleton |

## 4. 이후 round 가이드 (general)

각 round 가 끝날 때:

1. compound test_f1 vs wafer-only test_f1 비교 → margin Δ 계산
2. converge_margin (1.5 pp) 미달 시 compound-review 가 다음 spec 권고:
   - data: 합성 round 추가 (사용자 합의 후)
   - architecture: G channel encoding 변경 (default / sparse / onehot)
   - hparam: LR / batch / epoch 조정
   - resize: block_expand 정책 위반 발견 시 패치
3. converge 조건: margin ≥ 1.5 pp 가 2 round 연속 유지

## 5. 절대 금기 (사용자 글로벌 룰 + 본 정책)

- `logs_*/<run>/` 학습 결과 폴더 무단 삭제 금지 (글로벌 + 본 정책)
- `unknown/<class>` 데이터 폴더 무단 삭제 금지 (글로벌 + 본 정책)
- experiments/ YAML 무단 삭제 금지 (active class spec)
- archive 14 class (Center×5, Full×5, Edge-Ring_invalid_main, Normal_bank_boundary,
  Starburst, CommaCluster) 의 `unknown_archive/` copy 삭제 금지

## Cross-link

- round_end 시 trigger sequence → `.claude/agents/orch-master.md`
- compound-review 권고 input → `.claude/agents/compound-review.md`
- result aggregation → `.claude/agents/result-trace.md`
- margin 표 → `docs/paper/05_results.md`
- timeline raw → `docs/paper/CHANGELOG.md`
