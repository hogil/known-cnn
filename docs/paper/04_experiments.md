# 04 — Experiments

> **APPEND-ONLY.** 이 파일은 누적 기록. 한 번 적힌 row 절대 삭제·수정 금지 (수정도 새 row).
>
> 표 정책 (모든 wafer 분류 결과 보고 시 강제 — 출처: `CLAUDE.md` line 300-311,
> `.claude/skills/wafer-classifier/SKILL.md`):
>
> `Model | input | encoding | params | n train | epoch (best/total) | val_f1 | val_p | val_r | val_err | test_f1 | test_acc | 학습 시간`

## 1. 표 1 — V3 chipgrid sweep (n=100/cls, seed 42, protocol A)

protocol A: active class 미적용 (33-class 또는 24-class 시점), 0.8/0.1/0.1 split,
n=100 또는 220/cls, 30 epoch, batch 16. **다른 row 와 비교 시 protocol 다름 주의.**

(출처: `docs/chipgrid/RESULTS.md`, `logs_chipgrid/v*/best_history.txt`)

| Model | input | encoding | params | n train | epoch (best/total) | val_f1 | val_p | val_r | val_err | test_f1 | test_acc | 학습 시간 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| V0 (R-only chipgrid) | 32×32 | R only | ~1M | 2400 | 13 / 30 | 0.4359 | — | — | — | 0.4385 | — | <1 min |
| V1 (argmax /5) | 32×32 | argmax obj_id 1ch | ~1M | 2400 | 10 / 30 | 0.9505 | — | — | — | 0.9726 | — | <1 min |
| V2 (binary fork) | 32×32 | binary single-obj | ~1M | 2400 | 13 / 30 | 0.6543 | — | — | — | 0.6479 | — | <1 min |
| **V3 (one-hot 5ch)** | 32×32 | one-hot 5 + R | 1.16M | 2400 | 6 / 30 | **0.9689** | — | — | — | **0.9879** | — | <1 min |
| V0 full | 32×32 | R only | ~1M | 5280 | 5 / 30 | 0.4698 | — | — | — | 0.4039 | — | <1 min |
| V1 full | 32×32 | argmax /5 | ~1M | 5280 | 11 / 30 | 0.9805 | — | — | — | 0.9735 | — | <1 min |
| **V3 full ★** | 32×32 | one-hot 5ch | 1.16M | 5280 | 5 / 30 | **0.9945** | — | — | — | 0.9866 | — | <1 min |

핵심:
- V0 → V1 = +51 %p (val) — obj_id 채널이 wafer 분류의 dominant 신호
- V1 → V3 = +1.84 %p (val, n=100) — one-hot binary 가 정수 압축보다 우위
- V3 (n=100) → V3 (n=220) = +2.56 %p — full data 효과 큼

## 2. 표 2 — V3 chip CNN noise robustness (n=100/cls, seed 42, protocol A)

(출처: `docs/chipgrid/RESULTS.md`)

| Model | input | encoding | params | n train | epoch (best/total) | val_f1 | val_p | val_r | val_err | test_f1 | test_acc | 학습 시간 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| V3 noise 0% | 32×32 | one-hot 5ch | 1.16M | 2400 | 6 / 30 | 0.9689 | — | — | — | 0.9879 | — | <1 min |
| V3 noise 5% | 32×32 | one-hot 5ch | 1.16M | 2400 | 5 / 30 | 0.9667 | — | — | — | 0.9910 | — | <1 min |
| V3 noise 10% | 32×32 | one-hot 5ch | 1.16M | 2400 | 8 / 30 | 0.9707 | — | — | — | 0.9919 | — | <1 min |
| V3 noise 20% | 32×32 | one-hot 5ch | 1.16M | 2400 | 18 / 30 | 0.9595 | — | — | — | 0.9636 | — | <1 min |

→ chip CNN 10% noise 까지 robust. V3 5-seed std ±0.92 %p 범위 내 (0~10% 차이 통계
유의 X). **production chip CNN 이 90%+ 정확하면 V3 그대로 적용 가능.**

## 3. 표 3 — V3 5-seed mean ± std (n=100/cls, protocol A)

(출처: `docs/chipgrid/RESULTS.md` §V3 5-seed)

| seed | val_f1 | test_f1 |
|---|---|---|
| 42 | 0.9689 | 0.9879 |
| 1 | 0.9901 | 0.9838 |
| 7 | 0.9821 | 0.9868 |
| 100 | 0.9842 | 0.9935 |
| 234 | 0.9936 | 0.9826 |
| **mean ± std** | **0.9838 ± 0.0092** | **0.9869 ± 0.0041** |

## 4. 표 4 — Wafer ensemble fair compare (protocol B = 0.8/0.2 split, 33-class, n=220)

protocol B: 0.8/0.2 split (no test), n=220/cls, 5680 train, 33-class 전체. (출처:
`docs/wafer-ensemble/RESULTS.md`)

| Model | input | encoding | params | n train | epoch (best/total) | val_f1 | val_p | val_r | val_err | test_f1 | test_acc | 학습 시간 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| **V3 obj-only chipgrid** | 32×32 | one-hot 5ch (no R) | **1.16M** | 5680 | 6 / 13 (es) | **0.9946** | — | — | — | 0.9872 | — | <1 min |
| V3 R+5obj chipgrid | 32×32 | R + one-hot 5ch | 1.16M | 5680 | — | 0.9707 | — | — | — | — | — | <1 min |
| V3 fair eval (val 1420) | 32×32 | R + one-hot 5ch | 1.16M | 5680 | 6 / 13 | **0.9951** | — | — | **7** | — | — | inference |
| R-only ConvNeXtV2 base | 1024 RGB | palette PNG | 88M | 5680 | 10 / 10 | 0.9851 | — | — | 22 | — | — | 12.5 h |
| obj-only 4-layer CNN | 32×32 | embedding | 0.4M | 5680 | 8 / 10 | 0.9844 | — | — | 23 | — | — | 2 min |
| Tier 1 simple α=0.35 | combined | softmax avg | 88M+0.4M | 5680 | — | 0.9886 | — | — | — | — | — | inference |
| Oracle ceiling (R+obj) | combined | "either correct" | — | 5680 | — | 0.9919 | — | — | 12 | — | — | inference |
| compound33 baseline | 384 RGB | R+obj_id+0 BICUBIC | 88M | 5680 | 12 / 13 (crash) | 0.9784 | — | — | — | 0.9736 | — | ~1.5 h (12 ep) |

→ V3 단독 0.9946 이 oracle ceiling 0.9919 도 추월. compound BICUBIC 보다 +1.62 %p
(val) — block_expand 가 BICUBIC categorical 손상 회피한 효과 (출처:
`feedback_block_expand_only.md`).

## 5. 표 5 — Round 28 progress (protocol C = unknown/ 모든 class, n=200/cls 기본)

protocol C: `--active-classes-yaml` 미지정 → unknown/ 안 모든 class. seed 42, batch 16,
epoch 30 (early stop 7), AdamW (round 28 LR spec: lr_backbone=2e-5, lr_head=2e-4,
warmup_epochs=5, LinearLR start_factor=0.05, CosineAnnealing). 출처:
`logs_chip/v3_chip_260505_142036_running/{best_history.txt,hparams.yaml,run.log}`,
사용자 round 28 명시.

| Model | input | encoding | params | n train | epoch (best/total) | val_f1 | val_p | val_r | val_err | test_f1 | test_acc | 학습 시간 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| **chip CNN v3 (Stage 1)** | 384 RGB | chip palette PNG | 88M | 400 (5 cls × 100, n=100/cls subset) | 3 / 10 (es) | **1.0000** | 1.000 | 1.000 | 0 | **0.9799** | 0.980 | **6.7 min** (14:20:36 → 14:27:16) |
| wafer-only (Stage 2 base) | 1024 RGB | R only (palette) | 88M | TBD (~7680, 43 cls × 200, 0.8/0.1/0.1) | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD (학습 미실행) |
| compound v3 (Stage 3) | 384 (R + obj_id block_expand) | R+G+B 3ch | 88M | TBD (~7680) | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD (Branch B Stage 3 미실행) |

chip CNN v3 학습 환경 (출처: `logs_chip/v3_chip_260505_142036_running/hparams.yaml`):
- 5 class: bank_boundary / fork / invalid_main / scratch / scratch_rot
- subset: `experiments/chip_object_n100.yaml` (100/cls), 510 → 500 sample
- split: 400 / 50 / 50 (0.8 / 0.1 / 0.1)
- epoch 1 best (smoothed val F1 0.9799), epoch 2 → 0.9900, epoch 3 → **1.0000** (final best),
  epoch 4-9 모두 1.0000 유지, ep 10 early stop
- TEST per-class F1: bank_boundary 0.952 / fork 1.000 / invalid_main 1.000 / scratch
  0.947 / scratch_rot 1.000 (1 wrong test sample, 0 wrong val)
- 학습 시간 6 min 40 sec (= 6.7 min)

## 6. obj_id_maps build (Stage 2) — round 28 진행 중

`chip_tools/_build_obj_id_maps.py --batch 64` (출처: 사용자 round 28 dispatch).

| 항목 | 값 |
|---|---|
| input wafer (unknown/) | ~9600 PNG (43 class × 200 + Normal 1000) |
| output (.npy) | `D:/project/data/wm-811k/obj_id_maps/<TD>_<YYYYMMDD>/<basename>.npy` |
| 현재 진행 (2026-05-05T15:30 기준) | 7965 / 8600 ≈ **92.6%** |
| GPU 사용 | 7-8 GB (batch 64) — round 28 GPU resource lesson 적용 가능 (다음 build batch 32) |

(자세한 GPU resource lesson → `06_analysis.md`)

## 7. Multi-label stage 1 — per-class surface (출처: `docs/multi-label/STATUS.md`)

| Stage | 산출 | n | commit |
|---|---|---|---|
| Stage 1 (분포 학습) | `_dist_heatmaps_per_class/<class>/<method>_<n>.npy` | **850** (5 method × 33 class × 5 data-amount) + 37 plot + CSV | `687448b` |

이 산출은 multi-label stage 6 chip-wafer matching 의 surface base.

## 8. 학습 entry / 출력 컨벤션

(출처: `CLAUDE.md` line 117-160)

| Kind | 학습 entry | data root | log root |
|---|---|---|---|
| chip 5-class | `cnn_train_chip.py` | `data/wm-811k/classification_chips/` | `logs_chip/` |
| wafer 33-class R-only | `cnn_train_wafer.py` | `data/wm-811k/unknown/` | `logs_wafer/` |
| wafer 33-class compound | `cnn_train_compound.py` | `data/wm-811k/unknown/` + `obj_id_maps/` | `logs_compound/` |

학습 출력 컨벤션: `logs_<kind>/{model_tag}_{YYMMDD_HHMMSS}_{test_f1:.2f}_{val_f1:.2f}/`
(3-way) 또는 `logs_<kind>/{model_tag}_{YYMMDD_HHMMSS}_{val_f1:.2f}/` (`--train-val-only`).

각 logs_* 안 `overall/` 폴더 — 학습 종료 시 val F1 이 그 폴더 내 best 면 현재 run
폴더 통째 복사 교체. `_overall_meta.json` 에 source_run + val_f1 기록.

## 9. 자원 가드 (출처: `CLAUDE.md` line 148-156)

| Agent | 역할 |
|---|---|
| `cnn-master` | 학습 dispatch + kill + resume orchestrator (slash `/cnn-train-safe`) |
| `resource-monitor` | RAM 80% / GPU 90% 한계 자동 polling |

한계 초과 시 process kill + `log/<run>` `_PAUSED` rename (삭제 절대 금지) +
자원 회복 polling + 재시작.

## Cross-link

- chipgrid run 표 시간순 → `docs/chipgrid/RESULTS.md`
- compound vs V3 비교 → `docs/wafer-ensemble/RESULTS.md`
- 표 정책 → `CLAUDE.md` line 300-311
- fair-eval protocol → `docs/wafer-ensemble/FAIR_EVAL_PROTOCOL.md`
- chip CNN v3 학습 결과 raw → `logs_chip/v3_chip_260505_142036_running/best_history.txt`
