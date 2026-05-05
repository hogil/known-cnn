# Results — Wafer classifier all models compared

V3 발견 round 의 모든 model 결과 표. Same data, fair comparison
(0.8/0.1/0.1 또는 0.8/0.2 split, seed 42).

## 핵심 결과 (FAIR comparison, V3 best)

| Model | input | encoding | params | n train | epoch (best/total) | val_f1 | val_p | val_r | val_err | test_f1 | test_acc | 학습 시간 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| **V3 obj-only chipgrid** | 32×32×5 | one-hot 5ch (no R) | **1.16M** | 5680 (220/cls) | 6 / 13 (es) | **0.9946** | — | — | — | 0.9872 | — | <1 min |
| V3 R+5obj chipgrid | 32×32×6 | R + one-hot 5ch | 1.16M | 5680 | — | 0.9707 | — | — | — | — | — | <1 min |
| **V3 fair eval (our val 1420)** | 32×32×6 | R + one-hot 5ch | 1.16M | 5680 | 6 / 13 | **0.9951** | — | — | **7** | — | — | inference |
| R-only ConvNeXtV2 base | 1024 RGB | palette PNG | 88M | 5680 | 10 / 10 | 0.9851 | — | — | 22 | — | — | 12.5h |
| obj-only 4-layer CNN | 32×32 | embedding | 0.4M | 5680 | 8 / 10 | 0.9844 | — | — | 23 | — | — | 2 min |
| Tier 1 simple α=0.35 | combined | softmax avg | 88M+0.4M | 5680 | — | 0.9886 | — | — | — | — | — | inference |
| Oracle ceiling (R+obj) | combined | "either correct" | — | 5680 | — | 0.9919 | — | — | 12 | — | — | inference |

→ **V3 단독 0.9946 이 oracle ceiling 0.9919 도 추월**.

## V3 chipgrid sweep (full)

V0~V4 변종 비교 (n=100/class, seed=42, 30 epoch).

| variant | n/cls | encoding | in_ch | val_f1 | test_f1 | best_ep | run_dir |
|---|---|---|---|---|---|---|---|
| V0 | 100 | R only | 1 | 0.4359 | 0.4385 | 13 | v0_n100_260503_130816_0.44_0.44 |
| V1 | 100 | argmax/5 1ch | 2 | 0.9505 | 0.9726 | 10 | v1_n100_260503_131535_0.97_0.95 |
| V2 | 100 | binary particle_blast | 2 | 0.6543 | 0.6479 | 13 | v2_particle_n100_260503_132203_0.65_0.65 |
| **V3** | **100** | **one-hot 5ch** | 6 | **0.9689** | **0.9879** | 6 | v3_onehot_n100_260503_132834_0.99_0.97 |
| V3 | 220 | one-hot 5ch | 6 | **0.9945** | 0.9866 | 5 | v3_full_260503_160436_0.99_0.99 |
| V0 | 220 | R only | 1 | 0.4698 | 0.4039 | 5 | v0_full_260503_163546_0.40_0.47 |
| V1 | 220 | argmax/5 1ch | 2 | 0.9805 | 0.9735 | 11 | v1_full_260503_163549_0.97_0.98 |

핵심:
- V0 → V1: +51%p (val) — obj_id 채널이 wafer 분류의 dominant 신호
- V1 → V3: +1.84%p (val, n=100) — one-hot binary 가 정수 압축보다 우위
- V3 (n=100) → V3 (n=220): +2.56%p — full data 효과 큼

## V3 chip CNN noise robustness

n=100/class, seed=42, --chip-noise --chip-noise-eval (학습/평가 양쪽 적용).

| noise | val_f1 | test_f1 | val 차이 (vs V3 0%) |
|---|---|---|---|
| 0% | 0.9689 | 0.9879 | — |
| 5% | 0.9667 | 0.9910 | -0.22%p |
| 10% | 0.9707 | 0.9919 | +0.18%p |
| 20% | 0.9595 | 0.9636 | -0.94%p |

n=220/class:

| noise | val_f1 | test_f1 |
|---|---|---|
| 0% | 0.9946 | 0.9872 |
| 10% | 0.9870 | — |

→ 10% 까지 robust. 단일 seed 분산 ±0.92%p 이라 0~10% 차이 통계적 유의 X,
20% 부터 확실 degrade. **production chip CNN 90%+ 정확하면 V3 그대로 적용 가능**.

## V3 5-seed 평균 (통계 유의성)

같은 hparam (n=100, epochs=30) 으로 seed 5개:

| seed | val_f1 | test_f1 |
|---|---|---|
| 42 | 0.9689 | 0.9879 |
| 1 | 0.9901 | 0.9838 |
| 7 | 0.9821 | 0.9868 |
| 100 | 0.9842 | 0.9935 |
| 234 | 0.9936 | 0.9826 |
| **mean ± std** | **0.9838 ± 0.0092** | **0.9869 ± 0.0041** |

## obj-only 4-layer CNN sweep

| run | epochs | best_ep | val_f1 | 비고 |
|---|---|---|---|---|
| objonly_ep10 | 10 | 8 | 0.9844 | fair compare base |
| objonly_ep15 | 15 | 14 | 0.9778 | over-fit (학습 noise) |
| objonly_ep30_initial | 30 | — | 0.9804 | initial unfair compare |

## R-only ConvNeXtV2 sweep

| run | epochs | best_ep | val_f1 | 학습 |
|---|---|---|---|---|
| wafer33_full ep10 | 10 | 10 | 0.9851 | 12.5h |
| wafer33_full ep15 | 15 | (학습 중) | — | |

## Compound (R+G+B BICUBIC 384) — ceiling

| run | encoding | epochs | best_ep | val_f1 | test_f1 |
|---|---|---|---|---|---|
| compound33 baseline | R+obj_id+0 BICUBIC | 13/30 (crash) | 12 | 0.9784 | 0.9736 |

V3 가 compound 보다 +1.62%p 절대 (val) — block_expand 가 BICUBIC 손상 회피한 효과.

## Edge-Bottom / Edge-Top per-class breakdown

intra-distribution 분석 → `INTRA_DIST.md` 참조. V3 의 진짜 weak point.

## Path 위치

| 파일/폴더 | 내용 |
|---|---|
| `logs_chipgrid/v3_full_260503_160436_*/best_model.pth` | V3 best ckpt (n=220 baseline) |
| `logs_objid_ablation/overall/best_model.pth` | obj-only 4-layer best |
| `logs_wafer/overall/best_model.pth` | R-only ConvNeXt best |
| `logs_compound/overall/best_model.pth` | compound 3ch best |
| `results_v3_eval/v3_logits.npy` | V3 inference on our val 1420 |
| `results_ensemble_ep10/{r,obj}_logits.npy`, `val_y.npy` | ep10 fair compare base |
| `results_intra_dist/summary.json` | distribution / obj 분리 정확도 |
| `results_disagree/oracle_summary.json` | oracle ceiling 0.9919 |

## Cross-link

- 7 핵심 발견 → `DISCOVERY.md`
- intra-distribution 분석 → `INTRA_DIST.md`
- 33 → 20 active 결정 → `ACTIVE_CLASSES.md`
- 진행 상태 + Phase A/B/C → `STATUS.md`
- chipgrid V0~V6 sweep 전체 (시간순) → `D:/project/known-cnn/docs/chipgrid/RESULTS.md`
