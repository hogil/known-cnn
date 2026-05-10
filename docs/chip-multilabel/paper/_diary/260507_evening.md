# Diary — 2026-05-07 evening

## What happened today

The chip multi-label paper sections were brought up-to-date to
reflect three iters and one synthesis-spec reset that had landed
in the codebase since the last paper update (260506 final):

1. **Iter 10** (260506): master folder consolidation, sc+sr
   re-addition to COMBO_KEYS, Normal training (T7N family / C
   variant), and the H ensemble (baseline + C_44 logit-avg)
   reaching **0.9950 single-seed / 0.9930 ± 0.005 over 5 sample
   seeds, FAR 0.0%**. First cell in the project to clear 0.99
   macro-F1 with operational-grade FAR.
2. **Iter 11** (260506): paper-style 4-row 6×6 ablation matrix
   (108 cells) confirms that **no single (loss × inference) cell
   beats the H ensemble at FAR ≤ 5%**. Best single = T6 + I3 =
   0.905 macro-F1 / FAR = 100%. Asymmetric / Focal losses
   (T3, T4) Normal-generalise without explicit Normal training
   but cap at 10-def macro 0.51–0.80.
3. **Iter 12** (260506–07): chip-strength elevation via v19
   synthesis (fork weak-tier 0.45–0.55 → 0.70–0.85, smear-factor
   1.5–2.5 → 5.0–8.0) + FAR-split metric (paper switches
   `chip_FAR` headline to `normal_invalid_chip_FAR`) + T7N+T5
   ensemble (CF1 = 0.9083, ni_FAR = 0.50%) + v20 fork-thickness
   ↑ retrain (single-seed CF1 −0.018 noise but fork single recall
   1.000 saturated, fork+sr recall +0.094).
4. **v5.2 baseline reset** (260507 evening): wafer-side synthesis
   fixes (bank-seam removal, wafer pink uniform-spread, RingDots
   fixed positions, Edge defect-budget 6→20). Chip-level
   synthesis logic invariant — chip multi-label results carry
   forward.

## Design decisions captured

- **Master folder + runtime sampling > subset folders.** User
  directive 260506: "다시는 이런 subset 폴더 만들지마라". §3.8
  in 03_data.md.
- **Normal training non-negotiable.** User directive 260506:
  "Normal 학습에 들어갔어야". Empirical confirmation: iter 11
  35/36 cells fail FAR with 4-class only training; iter 12 T7N
  drops ni_FAR from 80% to 0% in one single-axis change.
- **OOD class metrics never reported.** User directive 260506:
  "OOD class 의 어떤 metric 도 보여주지 않음". 5 wafer-pattern
  classes are diagnostic only.
- **`scratch + scratch_rot` re-added with measurement-only
  stake** (iter 10). User directive: measure but accept it as
  ill-defined.
- **`chip_FAR` headline switched to `normal_invalid_chip_FAR`**
  (iter 12). The bundled metric obscured an 80× single-axis
  intervention.
- **Logit-averaging over complementary models (diversity >
  quantity)** — iter 10 demonstrates that pairing baseline with
  one well-chosen C seed beats baseline + 3 C seeds.
- **No multi-seed for v20** — single-seed CF1 −0.018 is within
  the §6.7 σ ≈ 0.030 noise floor; queued for confirmation.
- **TTA / rotation aug remain permanently disallowed** (iter 1
  finding extends through iter 12 — scratch / scratch_rot
  identity is rotation-tied).

## Where the numbers live

| iter | source                                                                         | run dirs                                                                                              |
|------|--------------------------------------------------------------------------------|-------------------------------------------------------------------------------------------------------|
| 10   | `outputs/T7_T9d_*_seed{42,43,44}_*` + `outputs/_logit_avg/T9d_C_44_*`          | baseline T9d + C_42/43/44 + H ensemble                                                                |
| 11   | `outputs/stage1_260506_092731 .. 094032/results_matrix.parquet`                | 108 cells (3 phases × 6 trains × 6 inferences)                                                        |
| 12 (v19y) | `outputs/T*_iter12_master_seed42_*/eval_I3/`                              | 8 variants on v19y master                                                                             |
| 12 (v19zpp) | `outputs/T*_T*_v19zpp_seed42_*/eval_I3/bit_metrics_split.json` + `outputs/T7_T7_with_normal_v19zpp_seed42_v2_260507_002217/` | 8 split-FAR + 1 T7N |
| 12 (v20) | `outputs/T7_T7N_v20_seed42_260507_063032/`                                   | v20 fork-thickness retrain                                                                            |
| 12 (ensemble) | `outputs/_iter12_v19zpp_logs/ensemble/T7N_T5_w70_30.json` + 16 other configs | 17 ensemble configurations                                                                            |
| v5.2 | `_uniform_linear_sample/`, `_v5_2_smoke/`, `_floor_cap_4opts/`, `_edge_check_preview/` | visual sanity preview folders                                                                         |

## What is still open

1. **Multi-seed v20.** Single-seed CF1 −0.018 may be noise; n≥3
   seed retrain required to confirm fork-thickness lift direction.
2. **v5.2-locked T7N + T5 ensemble.** Iter 13 should re-run the
   T7N + T5 70:30 recipe on the v5.2-locked master and report
   3-seed mean ± std; current §5.13 number (CF1 = 0.9083) is
   single-seed on v19zpp.
3. **Phase B ASL γ sweep** (queued from iter 5–9). Hypothesis is
   capped at +0.05 macro-F1 vs T9 by iter-9 T13a result; would
   close the question.
4. **Phase H replacement-not-additive regularisation test**
   (queued from iter 9). Swap LS=0.07 out for drop_path-only at
   varying rates to test the §7.4.4 regularisation-ceiling
   hypothesis.

## Carry-forward sections updated

- `03_data.md` — added §3.8 (master folder + runtime sampling),
  §3.9 (FAR-split metric), §3.10 (v5.2 baseline reset).
- `04_methods.md` — added T7c/T9 description and T7N variant
  under T-section, added §4.3 (logit-averaging ensemble) with
  iter-10 H + iter-12 T7N+T5 details, renumbered §4.3/§4.4 →
  §4.4/§4.5, added §4.5.1 FAR-split metrics.
- `05_experiments.md` — appended §5.11 (iter 10), §5.12 (iter
  11), §5.13 (iter 12 / v19y / v19zpp / v20), §5.14 (v5.2
  baseline reset).
- `_diary/260507_evening.md` — this entry.

## Sections deferred for next iter

- `01_introduction.md` — contributions list still mentions only
  iter 1–5 / "9 iters". Update queued for after multi-seed v20
  result.
- `02_related_work.md` — open-set chip multi-label, complementary
  ensemble (Hu et al.), label smoothing extension to
  multi-label (Cole et al. SPML) all remain to be cited
  inline at the iter-10 / iter-12 paragraphs added today.
- `06_analysis.md` — §6.3 / §6.4 fork / scratch_rot weakness
  analysis pre-dates v19/v20 strength elevation; should be
  augmented with a §6.9 documenting which weak points were
  attacked and which remain.
- `07_discussion.md` — §7.5 asymmetric BKM transfer story
  predates the iter-10 H ensemble; should be amended with a
  §7.5.5 documenting that *post-hoc complementary ensembling*
  is a third axis (alongside LS and CutMix) that *transfers
  cleanly* to our regime.
- `09_conclusion.md` — best-known result needs the iter-10 H
  ensemble (0.9930 ± 0.005) added as the new operational headline.
- `abstract.md` — current abstract caps at iter-9 0.9305 ± 0.046
  T9 family-mean; the iter-10 H ensemble (0.9930 ± 0.005, FAR
  0.0%) is the new milestone and should replace the headline
  number at the next abstract revision.

These cross-section updates are queued for the next narrator
session — the current update prioritises §3 (data), §4 (methods),
§5 (experiments) where iter 10 / 11 / 12 / v5.2 directly land.
