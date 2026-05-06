# 02 — Results

All numbers reported to 4 decimal places. Eval set: 2200 chips, 11-class
multi-label.

## Cross-iter best timeline

| iter | best_cell           | macro_f1   | top1_11    | Δ macro_f1 | Δ top1_11  | source                                                                |
|-----:|---------------------|-----------:|-----------:|-----------:|-----------:|-----------------------------------------------------------------------|
|  0\* | T0__I0              |     0.7302 |     0.4472 |          — |          — | outputs/stage1_260505_162842 (argmax baseline)                        |
|    1 | T0__I3              |     0.8466 |     0.6017 |    +0.1164 |    +0.1545 | outputs/stage1_260505_162842                                          |
|    2 | T0__I7              |     0.8485 |     0.6210 |    +0.0019 |    +0.0193 | outputs/stage1_260505_165400                                          |
|    3 | T0__I10             |     0.8542 |     0.6517 |    +0.0057 |    +0.0307 | outputs/stage1_260505_170827                                          |
|    4 | T1__I10             |     0.8634 |     0.7006 |    +0.0092 |    +0.0489 | outputs/stage1_260505_173649                                          |
|    5 | **T1_LS20__I7**     | **0.9268** | **0.8449** | **+0.0634**| **+0.1443**| outputs/phase_a_260505_175105                                         |
|    6 | (no new best)       |     0.9268 |     0.8449 |    +0.0000 |    +0.0000 | outputs/phase_a_260505_185805 — Phase A3 confirms ep=8 is global best |
|    7 | **T7c__I10**        | **0.9271** |     0.8307 | **+0.0003**|    -0.0142 | outputs/stage1_260505_195730 — BCE+LS=0.20+CutMix(p=0.5), bb+sr 0.32→0.96 |
|    8 | **T9d__I7** ☆       | **0.9705** | **0.9267** | **+0.0434**| **+0.0960**| outputs/stage1_260505_211334 — BCE+LS=0.07+CutMix(p=0.5), seed=42 (lucky outlier) |
|    8 | T9g__I7 (realistic) |   0.9408 |  0.8307 |    +0.0137 |    +0.0000 | outputs/stage1_260505_212557 — same config, seed=43; variance ±0.030 |
|    9 | (no new best)       |   0.9705 |  0.9267 |    +0.0000 |    +0.0000 | drop_path / cutmix-rect / two-LR all regress (see iter_09)             |
|   10 | **baseline+C_44 ENS** ★★★ | **0.9950** (10-def) | **0.9396** | **+0.0245** (over T9d) | **+0.0129** | ad-hoc ensemble, baseline T9d + C_44 (Normal trained, cutmix=0.25) logit avg. 5-sample-seed mean **0.9930±0.005**, FAR 0.00%. See `iters/iter_10_master_consol_sc_sr.md`. |
|   11 | (no new best — single) | 0.9050 | 0.8646 | -0.0900 vs iter10 ENS | — | iter 11 paper-style 6-train × 6-inf × 3-phase ablation. Best single = T6+I3 (BCE→ASL) macro 0.905 BUT Normal F1=0.000, FAR=100% — **operationally unusable**. Confirms ensemble necessity. See `iters/iter_11_paper_ablation_matrix.md`. |

\*iter 0 = the argmax baseline cell that lives inside iter 1's run.
☆ T9d is a single-seed favorable outlier (seed=42). The matched-config
seed=43 run (T9g) gives 0.9408. **Single-seed variance ±0.030 macro-F1**
at this config is now the dominant uncertainty above the LS axis (see
iter 8 for the full curve / variance discussion).

**Phase A final winner (closed): `T1_LS20_ep8 + I7`  →  macro_f1 = 0.9268, top1\_11 = 0.8449.**
Iter 6 (Phase A3 epochs sweep over {3, 5, 12} at LS=0.20) does not beat ep=8;
ep=12 → 0.8926 (I3), ep=3 → 0.8763 (I10), ep=5 → 0.8567 (I10). See
`iters/iter_06_phase_a3_epochs_sweep.md` for the full nine cells and the
training-duration → best-inference-variant regime change.

**Phase F (iter 7) outcome — two faces.** Anomaly-detection BKM transfer
(F1 warmup 2ep, F2 EMA 0.95) regressed by −0.109 / −0.089 macro-F1 — small
data + TAPT init does not need warmup, and EMA over-smooths under ~12
effective steps. The I11 pair-aware threshold band-aid (no retrain) lost
−0.007 net (bb+sr recall +25 chips, but bb+fork over-trigger 31 FP). The
T7 atomic decomposition (T1 → T7a CE→BCE → T7c +CutMix p=0.5) ties macro_f1
on the surface (0.9268 → 0.9271, +0.0003) but **flips the operational
profile**: bb+sr combo recall jumps from 0.32 to 0.96 (+0.63), and
`scratch_rot` per-class F1 reaches 1.0000 in T7c. CutMix-p sweep peaks
sharply at 0.5 (→ 0.7: 0.9038, → 0.3: 0.8626, → 0.0: 0.8577). See
`iters/iter_07_phase_f_warmup_ema_t7_cutmix.md`.

**Iter 8 (T9 LS sweep + variance verify).** Re-sweeping LS under the
BCE+CutMix recipe shifts the optimum to **LS=0.07** (T9d__I7 = 0.9705,
seed=42). The curve has a sharp cliff at LS=0.08 (T9e=0.8085) and a
broad 0.94 plateau across LS∈{0.05, 0.06, 0.10}. The same config at
seed=43 (T9g) drops to 0.9408 — a **single-seed variance of ±0.030**,
roughly the same magnitude as the LS=0.05→0.07 gap. The variance is
concentrated in fork F1 (0.945 vs 0.815 across seeds). T9d 0.9705 is
the best observed, T9g 0.9408 is the realistic point estimate. See
`iters/iter_08_T9_LS_sweep_variance.md`.

**Iter 9 (drop_path / cutmix-rect / two-LR — all atomic-failed).**
Three orthogonal regularizers/training-regime BKMs probed on top of
the iter-8 LS=0.07 recipe; **all regress**. T10 drop_path 0.05 (n=2
seeds): −0.054 / −0.049. T11 cutmix-rect 0.25: −0.106 (confounded with
0.5→0.25 ratio drop, which alone matches iter-7's p=0.3 result). T12
two-LR backbone/head: −0.084 (top1_11 drops 0.27, killing combos). All
three fail under the same diagnosis as iter 7 warmup/EMA: long-training
regime BKMs that need many effective steps to stabilize, on a small-
data + TAPT-init + 8-epoch budget that doesn't provide them. See
`iters/iter_09_negative_axis_drop_cutmix_two_lr.md`.

Cumulative gain vs argmax baseline (best observed, T9d__I7): **+0.2403
macro-F1**, **+0.4795 top1\_11class**. Realistic gain (T9g__I7, same
config seed=43): **+0.2106 macro-F1**, **+0.3835 top1\_11class**.

_Source: outputs/stage1_260505_162842/results_matrix.parquet,
outputs/stage1_260505_165400/results_matrix.parquet,
outputs/stage1_260505_170827/results_matrix.parquet,
outputs/stage1_260505_173649/results_matrix.parquet,
outputs/phase_a_260505_175105/sweep_log.csv,
outputs/phase_a_260505_185805/sweep_log.csv,
outputs/stage1_260505_195730/results_matrix.parquet (T7c, iter 7),
outputs/stage1_260505_211334/results_matrix.parquet (T9d, iter 8 ☆),
outputs/stage1_260505_212557/results_matrix.parquet (T9g, iter 8 — variance verify),
outputs/stage1_260505_{213423,213817,214222,214634}/results_matrix.parquet (iter 9)._

## Top-15 all-time cells (by macro_f1)

| rank | iter | cell_id            | macro_f1   | top1_11 | source                                                                  |
|-----:|-----:|--------------------|-----------:|--------:|-------------------------------------------------------------------------|
|    1 |    8 | **T9d__I7** ☆      | **0.9705** |  0.9267 | outputs/stage1_260505_211334 — LS=0.07 seed=42 (lucky outlier)          |
|    2 |    8 | T9d__I10 ☆         |     0.9705 |  0.9267 | outputs/stage1_260505_211334 — same checkpoint, I10 ties I7              |
|    3 |    8 | T9d__I3 ☆          |     0.9673 |  0.9187 | outputs/stage1_260505_211334                                            |
|    4 |    8 | T9d__I11 ☆         |     0.9654 |  0.9205 | outputs/stage1_260505_211334                                            |
|    5 |    8 | T9b__I7            |     0.9449 |  0.8670 | outputs/stage1_260505_210535 — LS=0.05                                  |
|    6 |    8 | T9b__I10           |     0.9449 |  0.8670 | outputs/stage1_260505_210535                                            |
|    7 |    8 | T9b__I11           |     0.9440 |  0.8659 | outputs/stage1_260505_210535                                            |
|    8 |    8 | T9b__I3            |     0.9424 |  0.8614 | outputs/stage1_260505_210535                                            |
|    9 |    8 | T9g__I7 (realistic)|     0.9408 |  0.8307 | outputs/stage1_260505_212557 — LS=0.07 seed=43, variance verify         |
|   10 |    8 | T9g__I10           |     0.9408 |  0.8307 | outputs/stage1_260505_212557                                            |
|   11 |    8 | T9g__I11           |     0.9408 |  0.8307 | outputs/stage1_260505_212557                                            |
|   12 |    8 | T9f__I3            |     0.9401 |  0.8648 | outputs/stage1_260505_212153 — LS=0.06                                  |
|   13 |    8 | T9a__I10           |     0.9364 |  0.8489 | outputs/stage1_260505_210059 — LS=0.10                                  |
|   14 |    8 | T9a__I7            |     0.9346 |  0.8443 | outputs/stage1_260505_210059                                            |
|   15 |    8 | T9a__I11           |     0.9346 |  0.8443 | outputs/stage1_260505_210059                                            |

☆ T9d (rank 1–4) is a single-seed favorable outlier (seed=42); the
matched-config seed=43 run T9g lands at rank 9–11 (0.9408). The realistic
gap between rank-1-observed (T9d, 0.9705) and rank-9-realistic (T9g,
0.9408) is **0.0297 macro-F1** at fixed config — see iter 8 for the
variance discussion.

Iter 8 reshuffles the **entire top-15 to T9 cells** — every iter-7 cell
(T7c, T7d) and every iter-5 cell (T1_LS20, T1_LS15) is pushed off the
list by the BCE+LS=0.07+CutMix(p=0.5) recipe. The previous rank-1
(T7c__I10 = 0.9271) is now rank 16+. Iter 9's drop_path/cutmix-rect/
two-LR experiments did not produce any cells that crack the top-15.

_Source: docs/chip-multilabel/tables/all_runs_macro_f1.csv (all rows incl. ranks 16+)._

## Per-iter winner — per-class F1 detail

### iter 1 — T0__I3 (frozen, F1-max + top-K rescue)

| class          | threshold | precision | recall | F1     | AP     |
|----------------|----------:|----------:|-------:|-------:|-------:|
| bank_boundary  |    0.4994 |    0.9788 | 0.9391 | 0.9585 | 0.9752 |
| fork           |    0.1195 |    0.4843 | 0.9141 | 0.6331 | 0.5762 |
| scratch        |    0.7682 |    1.0000 | 0.9438 | 0.9711 | 0.9723 |
| scratch_rot    |    0.8355 |    1.0000 | 0.7000 | 0.8235 | 0.8700 |

_Source: outputs/stage1_260505_162842/per_class_metrics.parquet (cell_id=T0__I3)._

### iter 2 — T0__I7 (frozen, F1-max + step-search Δ=0.02)

| class          | threshold | precision | recall | F1     | AP     |
|----------------|----------:|----------:|-------:|-------:|-------:|
| bank_boundary  |    0.5000 |    0.9788 | 0.9391 | 0.9585 | 0.9752 |
| fork           |    0.1400 |    0.5005 | 0.8609 | 0.6330 | 0.5762 |
| scratch        |    0.7400 |    1.0000 | 0.9479 | 0.9733 | 0.9723 |
| scratch_rot    |    0.8200 |    1.0000 | 0.7083 | 0.8293 | 0.8700 |

_Source: outputs/stage1_260505_165400/per_class_metrics.parquet (cell_id=T0__I7)._

### iter 3 — T0__I10 (frozen, I7 + entropy Normal gate)

| class          | threshold | precision | recall | F1     | AP     |
|----------------|----------:|----------:|-------:|-------:|-------:|
| bank_boundary  |    0.5000 |    0.9786 | 0.9297 | 0.9535 | 0.9752 |
| fork           |    0.1400 |    0.5360 | 0.8609 | 0.6607 | 0.5762 |
| scratch        |    0.7400 |    1.0000 | 0.9479 | 0.9733 | 0.9723 |
| scratch_rot    |    0.8200 |    1.0000 | 0.7083 | 0.8293 | 0.8700 |

_Source: outputs/stage1_260505_170827/per_class_metrics.parquet (cell_id=T0__I10)._

The Normal-gate gain comes mostly from `fork` precision (0.5005 → 0.5360, recall held).

### iter 4 — T1__I10 (CE+LS=0.10 retrain, I7+entropy)

| class          | threshold | precision | recall | F1     | AP     |
|----------------|----------:|----------:|-------:|-------:|-------:|
| bank_boundary  |    0.4600 |    1.0000 | 0.7781 | 0.8752 | 0.8969 |
| fork           |    0.2200 |    0.7014 | 0.7891 | 0.7426 | 0.6607 |
| scratch        |    0.6600 |    0.9803 | 0.9354 | 0.9574 | 0.9824 |
| scratch_rot    |    0.5000 |    1.0000 | 0.7833 | 0.8785 | 0.9614 |

_Source: outputs/stage1_260505_173649/per_class_metrics.parquet (cell_id=T0__I10 row, but the model was T1)._

The big jump is **fork F1 0.6607 → 0.7426** (+0.082): label smoothing
flattens the runner-up logit so multi-label thresholding actually has a
distinguishable score for fork-in-combo chips.

### iter 5 — T1_LS20__I7 (CE+LS=0.20 retrain, I7) — overall best (until iter 7)

`per_class_metrics.parquet` is not stored for sweep cells; per-class
breakdown is the next thing to capture in iter 6 if needed.
Aggregate: macro_f1 = 0.9268, top1\_11 = 0.8449.

_Source: outputs/phase_a_260505_175105/sweep_log.csv (LS=0.20, inference_id=I7)._

### iter 7 — T7c__I10 (BCE+LS=0.20+CutMix p=0.5, I10) — new overall best

| class          | threshold | precision | recall | F1     | AP     |
|----------------|----------:|----------:|-------:|-------:|-------:|
| bank_boundary  |    0.6200 |    0.9345 | 0.8469 | 0.8885 | 0.9575 |
| fork           |    0.1400 |    0.8815 | 0.8484 | 0.8647 | 0.7547 |
| scratch        |    0.7400 |    1.0000 | 0.9146 | 0.9554 | 0.9725 |
| scratch_rot    |    0.4200 |    1.0000 | 1.0000 | 1.0000 | 1.0000 |

_Source: outputs/stage1_260505_195730/per_class_metrics.parquet (cell_id=T0__I10)._

The headline shifts in T7c vs T1_LS20:

- **`scratch_rot` perfect** (F1 1.0000, AP 1.0000) — CutMix multi-hot
  training directly teaches the bb+sr visual co-occurrence, so the
  scratch-rot signal is no longer collapsed by the bb signal.
- **`fork` precision 0.70 → 0.88** at almost identical recall — BCE +
  CutMix sharpens fork's negative discrimination far more than CE+LS could.
- **bank_boundary** trades a small F1 drop (0.8974 → 0.8885) for the
  bb+sr gain — net positive on combo recall.
- **ECE_post 0.1788 → 0.0446** (4× lower) — BCE + CutMix produces a much
  better-calibrated probability surface as a side benefit.

The headline operational metric: **bb+sr combo recall 0.32 → 0.96** (from
T1_LS20 baseline → T7c). See `iters/iter_07_phase_f_warmup_ema_t7_cutmix.md`
for the full atomic decomposition (T1 → T7a → T7c) and CutMix-p sweep.

### iter 8 — T9d__I7 (BCE+LS=0.07+CutMix p=0.5, seed=42) — observed best

| class          | threshold | precision | recall | F1     | AP     |
|----------------|----------:|----------:|-------:|-------:|-------:|
| bank_boundary  |    0.9000 |    0.9919 | 0.9578 | 0.9746 | 0.9818 |
| fork           |    0.2200 |    1.0000 | 0.8953 | 0.9448 | 0.9877 |
| scratch        |    0.5800 |    0.9912 | 0.9354 | 0.9625 | 0.9759 |
| scratch_rot    |    0.1800 |    1.0000 | 1.0000 | 1.0000 | 1.0000 |

_Source: outputs/stage1_260505_211334/per_class_metrics.parquet (cell_id=T0__I7)._

vs **T9g__I7 (same config, seed=43, realistic point)**:

| class          | threshold | precision | recall | F1     | AP     |
|----------------|----------:|----------:|-------:|-------:|-------:|
| bank_boundary  |    0.3800 |    0.9984 | 0.9641 | 0.9809 | 0.9817 |
| fork           |    0.1000 |    0.7392 | 0.9078 | 0.8149 | 0.8253 |
| scratch        |    0.7400 |    0.9726 | 0.9625 | 0.9675 | 0.9804 |
| scratch_rot    |    0.3000 |    1.0000 | 1.0000 | 1.0000 | 1.0000 |

_Source: outputs/stage1_260505_212557/per_class_metrics.parquet (cell_id=T0__I7)._

The seed-driven variance is **concentrated in fork** (F1 0.9448 vs 0.8149,
AP 0.9877 vs 0.8253). bank_boundary, scratch, scratch_rot are stable
across seeds (Δ ≤ 0.013 F1). fork is the diffuse / longest-tail defect
per iter-1 error analysis — its sigmoid mass is most sensitive to which
fork patches end up in CutMix mosaics. See `iters/iter_08_T9_LS_sweep_variance.md`.

## Iter 10 — Ensemble FINAL winner (260506)

**Trigger** (260506): user re-added `scratch+scratch_rot` to COMBO_KEYS (was excluded in iter 1-9 design as "same defect family"). T9d on 12-class showed sc+sr F1=0.755 (R=0.606) — model never learned sc+sr CutMix pair (was disallowed).

**Phase journey**:
1. A1 retry (cutmix-p=0.5 + sc+sr CutMix on): sc+sr 1.000 ✅ but other classes ↓ (Normal 0.000) — net negative
2. D (cutmix-p=0.25, gentler): macro 0.9116, sc+sr 0.947 — partial recovery, Normal still 0
3. C (Normal training added, 5-class with y=-1 sentinel): Normal F1 1.000 ± 0.000 lock ✅, fork+scratch 0.673 NEW weakness (cross-class suppression)
4. F (fork↔scratch CutMix pair bias): fork+sc 0.95 but bb/fork+sr ↓ — net negative
5. **H Ensemble** (baseline T9d + C_44 logit avg): **10-defect macro 0.9950**, all class F1 ≥ 0.987, FAR 0%. **Diversity (with-Normal vs without) > Quantity (multi-seed)** — baseline + 1 C_44 (0.995) > baseline + 3 C seeds (0.966).

**Key memory rules added** (260506):
- `feedback_logit_ensemble_complementary.md` — H finding永久 룰
- `feedback_normal_training_open_set.md` — Normal training non-negotiable
- `feedback_cross_class_suppression.md` — fork combo prob 3× collapse mechanism
- `feedback_master_storage_vs_runtime_sampling.md` — single SoT folder
- `feedback_chip_train_batch_safe.md` — shared GPU batch=8 강제

**Final 5-sample-seed mean** (eval on master n=50):
- 4-single macro F1 = 0.9963 ± 0.0045
- 6-combo macro F1 = 0.9908 ± 0.0063
- 10-defect macro F1 = **0.9930 ± 0.0049**
- Normal F1 = 1.0000 ± 0.0000, sc+sr F1 = 1.0000 ± 0.0000
- False Alarm Rate = **0.00% ± 0.00%** (Normal 800 → 0 false alarms in 1000-chip wafer)

See `iters/iter_10_master_consol_sc_sr.md`.

## Iter 11 — Paper-style 4-row Ablation Matrix (260506)

108 cells = 6 train (T1/T3/T4/T5/T6/T7, all 4-class only via `--no-normal`) × 6 inference (I3/I7/I10/I11/I12/I13) × 3 phases (p50 simple Normal / p30 simple / p50 diverse Normal).

**Best single per train** (cross-phase):

| Train | Best | macro | Normal | FAR |
|---|---|---:|---:|---:|
| T6 (BCE→ASL) | P1+I3 | **0.905** | 0.000 | 100% ❌ |
| T5 (BCE) | P3+I11 | 0.894 | 0.000 | 100% ❌ |
| T7 (BCE+LS) | P2+I7 | 0.860 | 0.000 | 100% ❌ |
| T4 (ASL) | P1+I10 | 0.803 | 0.857 | 18% ⚠ |
| T1 (CE+LS) | P3+I11 | 0.620 | 0.000 | 100% ❌ |
| T3 (Focal) | P1+I11 | 0.513 | 0.974 | 5% ⚠ |

**Key findings**:
1. ★ Normal training 누락 = catastrophic FAR (T1/T5/T6/T7 모두 100% FAR) — operationally unusable
2. **Asymmetric (T4 ASL) / Focal (T3) 만 4-class only 환경에서 Normal 자연스럽게 generalize** (asymmetric/focal mechanism)
3. p50 → p30 distribution-shift Δ < 0.02 — 모든 model robust
4. Normal diversity 효과 marginal — T4 ASL 만 receptive (+0.07 N, -12.5% FAR)
5. **iter 10 ensemble (0.995, FAR 0%) 이 모든 single iter 11 model 압도** — single 의 한계 입증

See `iters/iter_11_paper_ablation_matrix.md`.

## Table dump

`tables/all_runs_macro_f1.csv` contains every iter-1-through-11 cell (283 rows)
with columns: `iter, cell_id, train_id, inference_id, macro_f1, micro_f1,
mAP, top1_11class, temperature, ece_post, source`.
