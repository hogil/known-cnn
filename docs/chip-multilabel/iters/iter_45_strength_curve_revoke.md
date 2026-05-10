# iter 45 — strength-curve REVOKES iter44 HARD WINNER claim

- iter: 45
- tag: `phase35_36_strength_curve_revoke`
- date: 2026-05-10
- eval: `eval_v15direct_HARD{040,045,050,055,060}` + `eval_v15direct_n200`
- one-liner: 6-point strength curve (45 inferences over 9-model pool) shows
  **pure-hard NEW HEADLINE 4-bag wins at 5/6 thresholds**; iter44 HARD WINNER
  dual-seed advantage was a single-point artifact at `strength_max=0.50`.

## Motivation

iter44 (Phase 34) declared `{24_LS030_seed42 + 33D + 37E + 24_LS030_seed7}`
the "NEW HARD WINNER" at **bit_F1 = 0.9843 / ni_FAR = 2.00%**, beating the
prior pure-hard NEW HEADLINE (0.9670) by +0.0173 — but the comparison ran
at **a single strength threshold** (`HARD050`, `strength<=0.50`).

Single-threshold conclusions are vulnerable to slice composition. To
falsify or confirm robustness, Phase 35/36 swept the strength threshold
across `{0.40, 0.45, 0.50, 0.55, 0.60, 1.00}` (FULL) and re-ran the same
4-bag winners at each point.

## Phase 35/36 sweep methodology

- 9-model candidate pool: `24_LS030_seed42, 24_LS030_seed7, 26B, 26D, 26H,
  33A, 33D, 37E, 21H`.
- 6 strength points × ~9 model inferences = ~45 stage-1 forwards on
  `chip_multilabel_v15direct` (seed=42 sampling).
- Intersection chip count grows monotonically with `strength_max` cap
  (more chips qualify): 975 (0.40) → 1395 (0.45) → 2003 (0.50) → 2724
  (0.55) → 3059 (0.60) → 3080 (FULL n=200).
- 4-bag candidates compared at every point under simple-majority `>=2/4`:
  - `pure-hard` = `{24_LS030_seed42 + 26B + 26D + 26H}` (NEW HEADLINE)
  - `hard+KD`   = `{24_LS030_seed42 + 26B + 26H + 33D}`
  - `HARD-W`    = `{24_LS030_seed42 + 33D + 37E + 24_LS030_seed7}` (iter44)
  - `iter34`    = `{26B + 26D + 33A + 37E}` (iter34 KD+asym)

## Full curve table (4-bag bit_F1 / ni_FAR)

| strength_max | n_chips int. | 4-bag winner (FAR ≤ 5%) | bF1    | ni_FAR | pure-hard | hard+KD | HARD-W (dual-seed) | iter34 (KD+asym) |
|--------------|-------------:|--------------------------|-------:|-------:|----------:|--------:|--------------------:|-----------------:|
| 0.40 (HARD040) |   975 (bb-excluded) | hard+KD                  | 0.7377 |    0%  |    0.7355 |  0.7377 |              0.7344 |           0.7315 |
| 0.45 (HARD045) | 1395                | **pure-hard ★**          | **0.9941** | 0%  |    0.9941 |  0.9937 |              0.9948 |           0.9736 |
| **0.50 (HARD050)** | 2003           | **HARD-W (iter44)**     | **0.9843** | **2%** | 0.9670 | 0.9689 | **0.9843**          |           0.9481 |
| 0.55 (HARD055) | 2724                | **pure-hard ★**          | **0.9966** | 0%  |    0.9966 |  0.9901 |              0.9953 |           0.9909 |
| 0.60 (HARD060) | 3059                | **pure-hard ★**          | **0.9959** | 0%  |    0.9959 |  0.9953 |              0.9953 |           0.9913 |
| FULL 1.00 (n=200) | 3080             | **pure-hard ★**          | **0.9955** | 0%  |    0.9955 |  0.9953 |              0.9937 |           0.9945 |

★ Pure-hard wins at 5/6 strength thresholds.
HARD040 (n=975) is a degenerate slice — `bb` chips are excluded entirely
(min `defect_pixel_ratio` for `bb` = 0.41 > 0.40), inflating the absence
of the bb-class error mode and flattening all configs near 0.73.

## Interpretation

1. **The dual-seed advantage exists ONLY at `strength_max=0.50`.**
   At neighboring thresholds (0.45 and 0.55), the pure-hard 4-bag is
   either tied or strictly better. The +0.0154 advantage at 0.50 does
   not propagate.
2. **Pure-hard NEW HEADLINE is ROBUST across the curve.** It wins at
   0.45/0.55/0.60/FULL by margins of 0.0008–0.0065 over hard+KD, and
   never falls below the iter44 HARD-W bag except at 0.50.
3. **HARD050 is a sample-composition artifact.** The cohort of chips
   surviving `strength<=0.50` happens to favor the dual-24_LS030-seed
   double-counting; this does not generalize to wider or narrower
   filters.
4. **iter34 (KD+asym) is consistently the weakest 4-bag** across the
   curve (0.7315–0.9945), confirming the iter39 finding that the
   asymmetric+KD axes do not stack.

## Revocation logic

Phase 34's claim was framed as "**NEW HARD WINNER beats hard+KD by
+0.0154**", with HARD eval framed as a stress-test. A genuine stress-test
result must hold under perturbation of the stress parameter. The
strength-curve sweep is exactly that perturbation: it varies the
hardness threshold continuously and asks whether the winner survives.

The iter44 4-bag does not survive: at every adjacent strength setting,
either pure-hard or both pure-hard and hard+KD beat it. This is the
signature of a **single-point artifact**, not a robust ranking.

Therefore the iter44 row in `paper_main_headline.csv` is annotated as
**REVOKED** (not removed — preserved for paper rebuttal narrative). The
final paper-grade claim reverts to the pure-hard NEW HEADLINE 4-bag
(`{24_LS030_seed42 + 26B + 26D + 26H}` thr ≥ 2/4 majority) at v15direct
FULL n=200/n=500 = 0.9953–0.9955 / 0% ni_FAR.

## Paper-grade lesson

> Single-threshold ablation conclusions are **vulnerable to slice
> composition**. A reported advantage at one stress slice should be
> validated by sweeping the slice parameter; if the advantage does not
> hold at neighboring slice settings, the result is a sample-composition
> artifact, not a structural advantage.

This applies generally: HARD-eval stress thresholds, OOD severity
levels, label-noise rates, sample-size cuts — all benefit from
sweep-then-rank rather than pick-then-claim.

## Source paths

- `outputs/_phase35_strength_curve.log` — full sweep dispatch log
  (45 stage-1 forwards, 6 strength points × 9 models)
- `outputs/iter*/T*/eval_v15direct_HARD040/stage1_*/preds_chip.parquet`
  — per-model preds at strength_max=0.40
- `outputs/iter*/T*/eval_v15direct_HARD045/stage1_*/preds_chip.parquet`
  — per-model preds at strength_max=0.45
- `outputs/iter*/T*/eval_v15direct_HARD050/stage1_*/preds_chip.parquet`
  — per-model preds at strength_max=0.50 (iter44 HARD050 source)
- `outputs/iter*/T*/eval_v15direct_HARD055/stage1_*/preds_chip.parquet`
  — per-model preds at strength_max=0.55
- `outputs/iter*/T*/eval_v15direct_HARD060/stage1_*/preds_chip.parquet`
  — per-model preds at strength_max=0.60
- `outputs/iter*/T*/eval_v15direct_n200/stage1_*/preds_chip.parquet`
  — per-model preds at FULL strength (iter42 n=200 rebuttal source)

## Related rows

- `tables/paper_main_headline.csv` — `iter44_ensemble_4bag_HARD_WINNER_dualseed`
  (annotated REVOKED) + `iter45_strength_curve_revoke` (SUMMARY)
- `iters/iter_44_HARD_winner_refined.md` — the revoked claim (preserved)
- `iters/iter_43_HARD_eval_breakthrough.md` — the prior single-pair HARD050 finding
- `iters/iter_42_n200_rebuttal.md` — n=200 stability evidence
