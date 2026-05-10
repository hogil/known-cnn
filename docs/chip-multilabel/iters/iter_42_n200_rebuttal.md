# iter 42 — n=200 Rebuttal: Pure-Hard 4-bag REAL HEADLINE = 0.9955 (not 0.9992)

- **Iter**: 42
- **Tag**: `n200_rebuttal`
- **Date**: 2026-05-10
- **Type**: ★★★ REAL HEADLINE — supersedes iter39 n=50 finding (small-sample artifact)
- **Mode**: INFERENCE-ONLY re-evaluation — same checkpoints as iter21–37, but
  evaluated on **v15direct n=200 (3080 chips, 4× larger eval set)** instead of n=50.
- **Headline**: pure hard-label 4-bag `{24_LS030_seed42 + 26B + 26D + 26H}` thr ≥ 2/4
  majority vote → **v15 bit_F1 = 0.9955**, **ni_FAR = 0.00%**.
  Per-class: bb=0.9984, fk=0.9881, sc=0.9953, sr=1.0000.

## Motivation

User flagged that the iter39 NEW HEADLINE result (v15 bit_F1 = 0.9992 / 0% on
n=50) might be a small-sample artifact: only 50 chips per class on the v15direct
eval set means 1–2 wrong predictions can move bit_F1 by **±0.005 or more**, and
the spread between competing 4-bag configurations (iter33 / iter34 / iter37 /
iter39) was already inside that noise band.

Hypothesis: at **n=200 per class** (eval set 4× larger, 3080 chips total),
all "winning" 4-bag configurations should collapse to within a tighter sampling
band, and the "+0.0016 pure-hard wins" thesis from iter39 may not survive.

## n=200 single-model re-eval (Phase 27 — 9 cells)

Same checkpoints, identical inference recipe, only eval set size differs.

| model               | n=50 best (FAR-safe) | n=200 best (FAR-safe) | ΔbF1 (n200-n50) | dual |
|---------------------|---------------------:|----------------------:|----------------:|------|
| 26B (g=3 LS=0.50)   |    0.9791 / 1.25%    |     0.9781 / 2.50%    |         −0.0010 | PASS |
| 26D (g=4 LS=0.40)   |    0.9353 / 0%       |     0.9533 / 0%       |         +0.0180 | PASS |
| 26H (g=3 LS=0.67 w) |    0.9687 / 2.50%    |     0.9857 / 3.50%    |         +0.0170 | PASS |
| 24_LS030_seed42     |    FAIL (21.25% FAR) |     FAIL (20.5% FAR)  |               — | FAIL |
| 24_LS030_seed7      |    FAIL (12.5% FAR)  |     FAIL (46.0% FAR)  |               — | FAIL |
| 33A (KD α=0.3 T=4)  |    0.9840 / 0%       |     0.9770 / 0%       |         −0.0070 | PASS |
| 33D (KD α=0.5 T=8)  |    0.9695 / 0%       |     0.9728 / 0.5%     |         +0.0033 | PASS |
| 37E (g=3 (1.0,0.5)) |    0.9604 / 1.25%    |     0.9782 / 1.0%     |         +0.0178 | PASS |
| 21H (g=4 LS=0.75)   |    0.9346 / 0%       |     0.9585 / 1.5%     |         +0.0239 | PASS |

★ **24_LS030 (BOTH seeds)** is single-model FAR-fragile under both n=50 AND n=200
— always FAILS dual-gate alone, but contributes to the 4-bag ensemble PASS.
This is the canonical paper §6.17.2 example of "ensemble cancels single-model
fragility."

## n=200 ensemble 4-bag re-test (3080 chips)

Same 5 ensemble compositions retested at n=200.

| 4-bag config                                  | n=50 bF1 / ni  | n=200 bF1 / ni | per-class @ n=200 (bb/fk/sc/sr)         | dual |
|-----------------------------------------------|---------------:|---------------:|------------------------------------------|------|
| ★★★ NEW REAL MAIN: 24_30s42+26B+26D+26H (pure-hard) | 0.9992 / 0%    | **0.9955 / 0%** | 0.9984 / 0.9881 / 0.9953 / 1.0000        | PASS |
| alt seed7: 24_30s7+26B+26D+26H                | 0.9992 / 1.25% |  0.9959 / 4.50% | 1.0000 / 0.9873 / 0.9961 / 1.0000        | PASS (borderline) |
| iter34: 26B+26D+33A+37E (KD+asym)             | 0.9976 / 0%    |  0.9945 / 0%    | 0.9984 / 0.9865 / 0.9929 / 1.0000        | PASS |
| iter33: 26B+21H+26D+24_30s42 (pure-hard alt)  | 0.9945 / 0%    |  0.9953 / 0%    | 0.9992 / 0.9873 / 0.9945 / 1.0000        | PASS |
| hard+KD: 24_30s42+26B+26H+33D                 | 0.9984 / 0%    |  0.9953 / 0%    | 0.9992 / 0.9881 / 0.9937 / 1.0000        | PASS |

**All 4-bag configs collapse to 0.9945–0.9959 at n=200 — spread = 0.0014, fully
inside the n=200 sampling band.** The n=50 spread of 0.0047 (0.9945 → 0.9992)
**does not survive** the 4× eval-set expansion.

## Key paper rebuttal claims

1. **n=50 → n=200 supersedes the headline number itself**: 0.9992 was a
   single-chip-difference artifact (n=50 has ~50–80 defect chips per class so
   a single fork miss = 0.005 bF1). At n=200 (~770 defect chips per class),
   the same ensemble lands at **0.9955** — still SOTA, still 0% FAR, but
   0.0037 lower than the n=50 reading.

2. **"Pure-hard wins by +0.0016" thesis is FALSIFIED.** At n=200 the pure-hard
   4-bag (0.9955) is statistically tied with hard+KD (0.9953), iter33 alt
   pure-hard (0.9953), and iter34 KD+asym (0.9945). The iter39 ranking
   advantage of +0.0016 disappears in the larger eval. **No specialty-axis
   argument follows from the data alone** — the diversity-axis-discovery
   research strategy is supported only at the level of "all 4 axes contribute
   roughly equally," not "axis X dominates."

3. **"Ensemble cancels fragility" thesis is STRENGTHENED.** Both
   `24_LS030_seed42` (n=200 ni_FAR = 20.5%) and `24_LS030_seed7` (n=200
   ni_FAR = 46.0%) FAIL single-model dual-gate at the larger eval set, **yet
   their inclusion in the 4-bag ensemble keeps ni_FAR at 0%**. The other 3
   bags absorb the seed-fragile model's false positives. This is the cleanest
   single-data-point demonstration of the ensemble-cancels-fragility mechanism
   in the project's history.

## Source paths (n=200 evals)

```
outputs/iter21H_19I_repeat/T7_iter21H_19I_repeat_seed1_260509_110530/eval_v15direct_n200/stage1_*/preds_chip.parquet
outputs/iter24_LS030_seed42/T7_iter24_LS030_seed42_260509_144238/eval_v15direct_n200/stage1_*/preds_chip.parquet
outputs/iter24_LS030_seed7/T7_iter24_LS030_seed7_260509_143534/eval_v15direct_n200/stage1_*/preds_chip.parquet
outputs/iter26B_g3_LS050/T7_iter26B_g3_LS050_seed1_260509_154354/eval_v15direct_n200/stage1_*/preds_chip.parquet
outputs/iter26D_g4_LS040/T7_iter26D_g4_LS040_seed1_260509_162552/eval_v15direct_n200/stage1_*/preds_chip.parquet
outputs/iter26H_g3_LS067_white/T7_iter26H_g3_LS067_white_seed1_260509_165535/eval_v15direct_n200/stage1_*/preds_chip.parquet
outputs/iter33A_alpha03_T4/T7_iter33A_alpha03_T4_seed1_260509_233558/eval_v15direct_n200/stage1_*/preds_chip.parquet
outputs/iter33D_alpha05_T8/T7_iter33D_alpha05_T8_seed1_260509_235817/eval_v15direct_n200/stage1_*/preds_chip.parquet
outputs/iter37E_g3_1.0_0.5/T7_iter37E_g3_1.0_0.5_seed1_260510_024859/eval_v15direct_n200/stage1_*/preds_chip.parquet
```

## Phase 28 n=500 confirmation (added 2026-05-10)

Same checkpoints, same inference recipe, eval expanded to **v15direct_n1000 n=500
(5250-chip per-class eval, 7080 intersection chips)** — most reliable evaluation
to date.

### Single models n=500 (FAR-safe filter, 5250-chip eval)

| model           | bF1    | ni_FAR | dual | per-class bb/fk/sc/sr        |
|-----------------|-------:|-------:|------|------------------------------|
| 24_LS030_seed42 | 0.9867 | 22.50% | FAIL | 0.9691/0.9902/0.9915/0.9959  |
| 24_LS030_seed7  | 0.9919 | 68.00% | FAIL | 0.9925/0.9828/0.9950/0.9972  |
| 26B             | 0.9795 |  2.50% | PASS | 0.9704/0.9813/0.9665/1.0000  |
| 26D             | 0.9605 |  0.00% | PASS | 0.9453/0.9811/0.9431/0.9724  |
| 26H             | 0.9708 |  4.00% | PASS | 0.9981/0.9909/0.8941/1.0000  |
| 33A             | 0.9860 |  0.00% | PASS | 0.9854/0.9902/0.9683/1.0000  |
| 33D             | 0.9792 |  0.00% | PASS | 0.9674/0.9912/0.9581/1.0000  |
| 37E             | 0.9800 |  0.50% | PASS | 0.9494/0.9851/0.9882/0.9972  |
| 21H             | 0.9586 |  2.50% | PASS | 0.9474/0.9450/0.9586/0.9835  |

### 4-bag ensembles at n=500 (7080-chip intersection)

| 4-bag config                                 | n=50    | n=200   | **n=500 FINAL** | per-class @ n=500          |
|----------------------------------------------|--------:|--------:|----------------:|-----------------------------|
| ★★★ NEW HEADLINE pure-hard (24_30s42+26B+26D+26H) | 0.9992/0% | 0.9955/0% | **0.9953/0%** ★ | 0.9959/0.9915/0.9937/1.0000 |
| ★★★ hard+KD TIE (24_30s42+26B+26H+33D)        | 0.9984/0% | 0.9953/0% | **0.9953/0%** ★ TIE | 0.9962/0.9912/0.9937/1.0000 |
| alt seed7 (24_30s7+26B+26D+26H)              | 0.9992/1.25% | 0.9959/4.5% | 0.9963/4.5% | 0.9994/0.9915/0.9944/1.0000 |
| iter33 with 21H                              | 0.9945/0% | 0.9953/0% | 0.9935/0% | 0.9947/0.9903/0.9890/1.0000 |
| iter34 KD+asym                               | 0.9976/0% | 0.9945/0% | 0.9922/0% | 0.9912/0.9899/0.9878/1.0000 |

### Phase 28 conclusions

1. **FINAL HEADLINE confirmed**: pure-hard 4-bag at **bit_F1 = 0.9953 / ni_FAR =
   0.00%** at n=500 — within 0.0002 of n=200 (0.9955), confirming n=200 was
   already the stable number. The n=50 0.9992 reading is now definitively a
   small-sample artifact.

2. **Hard+KD 4-bag TIES** the pure-hard headline at **0.9953/0%** with per-class
   F1 differing by ≤0.0003 — pure noise. Replacing 26D (g=4 LS=0.40 hard) with
   33D (KD α=0.5 T=8) yields an identical headline. **The KD-axis vs hard-label
   diversity distinction is indistinguishable at the 4-bag level.** Two
   independent 4-bag compositions land on the same headline number.

3. **Single 24_LS030 seeds STAY FAR-fragile at n=500**: seed=42 ni_FAR=22.50%,
   seed=7 ni_FAR=68.00% — both single-model FAILs. Yet the 4-bag containing
   seed=42 holds at 0.00% ni_FAR. This is the **strongest single-data-point
   demonstration of "ensemble cancels single-model fragility"** in the project's
   history (5250-chip eval, ≥1000-chip Normal probe).

4. **Stability across n**: pure-hard 4-bag 0.9992 (n=50) → 0.9955 (n=200) →
   0.9953 (n=500) shows monotonic convergence to the true value as eval set
   grows. **0.9953 is the stable answer.**

## Records updated

- `tables/paper_main_headline.csv` — added `iter39_ensemble_4bag_pureHard_n200_REBUTTAL`
  as the new top row (0.9955 / 0% at n=200), prior iter39 / iter37 / iter34 rows
  annotated with the n=50 caveat.
- `tables/all_runs_macro_f1.csv` — appended 9 single-model and 5 ensemble n=200 rows.
- `02_results.md` — replaced top PAPER MAIN WINNER pointer to the n=200 0.9955 result;
  added top-of-timeline row for iter 42 rebuttal; iter 39 row annotated as rebutted.
- Paper sections (`paper/05_experiments.md`, `06_analysis.md`, `abstract.md`) are
  paper-narrator's domain and are NOT updated by this iter.

## Summary

The n=50 → n=200 re-evaluation reveals that the four leading 4-bag ensemble
configurations are **statistically indistinguishable** at the larger eval set
(spread 0.0014 vs n=50 spread 0.0047). The new headline is **0.9955 v15 bit_F1
at 0% ni_FAR on 3080 chips**. The "pure-hard wins" thesis from iter39 does not
survive. The "ensemble cancels fragility" thesis is **strengthened** — the
`24_LS030_seed42` slot has 20.5% ni_FAR alone but the 4-bag holds at 0%.
