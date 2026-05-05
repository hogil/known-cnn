# Fair-eval protocol

모든 wafer backbone 비교의 single source of truth. Spec yaml: `experiments/fair_eval_protocol.yaml`.

## TL;DR

같은 active class, 같은 sample 수, 같은 split, 같은 epoch, 같은 augmentation,
TTA 없음 → best val_f1 epoch model 의 test 결과로 비교.

## 왜 필요한가

현재 결과 표가 split/epoch/sample 수 backbone 마다 달라서 직접 비교 불가:

- V3 chipgrid `val_f1 0.9946` — `cnn_eval_chipgrid.py`, 0.8/0.1/0.1 split, 220/class, 13 epoch (early stop)
- R-only ConvNeXt `val_f1 0.9851` — `cnn_train_wafer.py`, 0.8/0.2 split, 200~/class, 10 epoch
- objonly 4-layer `val_f1 0.9844` — `cnn_train_objonly.py`, 0.8/0.2 split, 220/class, 10 epoch

split 이 다르면 val sample 자체가 다른 wafer set 이라 비교 무의미. 본 protocol 로 통일.

## 고정 spec

| 항목 | 값 |
|---|---|
| active class | `experiments/active_classes_20.yaml` (20 class, V3 weak point + reference) |
| per-class sample | 220 (sorted file pick, deterministic) |
| split | 0.8 / 0.1 / 0.1 stratified, seed 42 |
| n_train | 3520 (176/class × 20) |
| n_val | 440 (22/class × 20) |
| n_test | 440 (22/class × 20) |
| epoch | 30 (early stop 끔) |
| selection | best val_f1 macro epoch |
| optimizer | AdamW wd 0.05, cosine warmup 3ep |
| augmentation | rotate ±15°, translate/scale ±3%, gaussian σ=0.01 (no flip/colorjitter/mixup/cutmix) |
| TTA | 금지 |

### Backbone 크기 별 batch/lr

| size | param | batch | lr |
|---|---|---|---|
| small | ≤ 2M | 64 | uniform 1e-3 |
| large | > 10M | 16 | head 1e-3, backbone 1e-4 |

(param 차이로 batch/lr 동일하면 large 모델 OOM. 외 모든 항목 동일)

## 보고 표 형식

모든 비교 표 필수 컬럼:

```
| Model | input | encoding | params | n_train | epoch (best/total) | val_f1 | val_p | val_r | val_err | test_f1 | test_acc | 학습 시간 |
```

예시 (V3 baseline):

| Model | input | encoding | params | n_train | epoch (best/total) | val_f1 | val_p | val_r | val_err | test_f1 | test_acc | 학습 시간 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| V3 chipgrid | obj_id 32×32 | one-hot 5ch | 1.16M | 3520 | 6 / 30 | TBD | TBD | TBD | TBD | TBD | TBD | <1 min |

## 통계적 claim 시

production decision 또는 paper 수준 claim 시 5-seed [42, 1, 7, 100, 234] mean ± std.

```bash
for s in 42 1 7 100 234; do
    python <script> --active-classes-yaml experiments/active_classes_20.yaml \
        --n-per-class 220 --split 0.8 0.1 0.1 \
        --epochs 30 --seed $s --model-tag v3_seed${s}
done
```

5-run mean val_f1 ± std 보고.

## Multi-protocol 비교 (예외)

다른 split (e.g. 0.8/0.2) 또는 다른 active class set (e.g. 30 class with new wafer-canvas) 결과는
**별도 표** 로 분리, header 에 protocol 명시.

```
### Protocol A (active 20, 0.8/0.1/0.1, ep 30)
| ... |

### Protocol B (active 30, 0.8/0.1/0.1, ep 30)
| ... |
```

## Enforcement

- 새 학습 script 작성 시 본 yaml 의 default 따르기 (CLI 로 override 가능 하지만 표 보고 시 명시)
- 보고 시 한 표 안 row 들의 protocol 항목 동일성 확인
- 위반: 표 분리 또는 재학습

## 절대 금기 (재확인)

| 위반 | 결과 |
|---|---|
| TTA (rotation/flip ensemble) | 결과 거부 — wafer class identity = angle/위치 |
| BICUBIC/NEAREST hardcode (categorical resize) | 결과 거부 — `_chipgrid_resize.block_expand_2d` 만 |
| `EXCLUDE_CLASSES` hardcode list 추가 | 결과 거부 — active YAML 만 |
| 학습 결과 폴더 무단 삭제 | 결과 거부 — global rule |

## Cross-link

- spec yaml: `experiments/fair_eval_protocol.yaml`
- active class: `experiments/active_classes_20.yaml`
- 표 정책 상세: `docs/wafer-ensemble/RESULTS.md`
- block_expand 정책: `~/.claude/projects/D--project-known-cnn/memory/feedback_block_expand_only.md`
- TTA 금지: `~/.claude/projects/D--project-known-cnn/memory/feedback_no_tta_wafer.md`
- active class 정책: `~/.claude/projects/D--project-known-cnn/memory/feedback_active_class_policy.md`
