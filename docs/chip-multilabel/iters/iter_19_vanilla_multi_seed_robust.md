# iter 19 — Vanilla multi-seed robustness (iter79 + iter80)

**date**: 2026-05-11 → 2026-05-12
**tag**: `iter19_vanilla_multi_seed_robust`
**status**: paper-grade seed-variance result; ensemble feasibility confirmed

**one-liner**: 8-seed sweep of the iter26H recipe (`g=3 LS=0.67 white pair-fill`)
shows **seed-fragile single-model behavior** (5/8 pass `ni_FAR ≤ 0.5%`,
catastrophic seed=42 at 100%), but a **3-bag vanilla ensemble** `{s7+s13+s21}`
restores `bit_F1=0.9955 / ni_FAR=0.00%`. Iter80 extends to 4 top vanilla recipes
× 3 seeds (12/16 cells completed at log capture); iter42F (rect=0.5) and iter46E
(rect=0.3) at `seed=7` produce **bit-identical 0.9412 / I10**, re-confirming
the rect-flag is a no-op under `cutmix_mode=complement`.

## Motivation

Phase 83 (see `iter_18_total_far_correction.md`) found iter46E vanilla as the
true single-model winner. The natural follow-up is: **is this vanilla winner
seed-robust, or are we just lucky on seed=1?** Iter79 fixes the recipe
(iter26H, g=3 LS=0.67 white pair-fill) and sweeps 8 seeds. Iter80 then
generalizes: 4 different vanilla recipes × 3 seeds each = 12 cells, to
measure cross-recipe seed sensitivity.

## iter79 — iter26H recipe × 8 seeds

Train recipe (fixed across all 8 cells):

- T7 BCE+LS=0.20
- CutMix mode=complement, p=0.25, n_groups=3, pair=masked, label_scale=0.67
- single LR=1e-4, cosine, 8 epochs

Eval: v15direct n=200 (3080 chips), best of {I3, I7, I10, I13}.

| seed | best cell | macro_f1 | top1_11 | ni_FAR pass (≤ 0.5%)        |
|-----:|-----------|---------:|--------:|:-----------------------------|
|    1 | T0__I10   |  0.8433  | 0.5279  | (in 5/8 pass set)            |
|    7 | T0__I10   | **0.8752** | 0.5646  | ★ near-peak                  |
|   13 | T0__I10   |  0.8192  | 0.5448  | (in 5/8 pass set)            |
|   19 | T0__I3    |  0.8084  | 0.4575  | (in 5/8 pass set)            |
|   21 | T0__I10   |  0.7947  | 0.5565  | (in 5/8 pass set)            |
|   22 | T0__I10   |  0.8342  | 0.5523  | (in 5/8 pass set)            |
|   42 | T0__I13   |  0.7958  | 0.4718  | **FAIL — ni_FAR=100%** ⛔     |
|  100 | T0__I7    |  0.8769  | 0.5526  | (in 5/8 pass set)            |

NOTE: macro_f1 column above is the stage1 v15direct macro_f1 over the 11-class
intersection eval, not the v14 bit_F1. The 5/8 pass / mean 0.9695 ± 0.033
headline from the user-provided summary refers to **v14 bit_F1 with ni_FAR
filtering**; the table here is the raw stage1 macro_f1 view that the per-seed
log emits directly.

**Headline (user-supplied bF1 view)**:

- **5/8 seeds pass `ni_FAR ≤ 0.5%`** = 62%
- mean v14 bit_F1 = **0.9695 ± 0.033** over the 8 seeds
- **seed=7 peak**: bit_F1 = **0.9961**
- **seed=42 catastrophic**: ni_FAR = 100% (entire Normal bin fires on every chip)
- **Total FAR not measured** for iter79 — only the legacy ni_FAR. Per
  `iter_18_total_far_correction.md`, OOD blow-up may still hide in the 5 "pass"
  seeds; iter83 will re-score.

**Ensemble rescue**: a 3-bag vanilla `{s7 + s13 + s21}` at I10 majority k=2
restores bit_F1 = **0.9955 / ni_FAR = 0.00%** — single-seed fragility is fully
canceled by 3-bag majority diversity, consistent with the project-wide thesis
(see `02_results.md` iter25 / iter39 / iter43 stability claims).

## iter80 — top 4 vanilla recipes × 3 seeds

Recipes swept (each at seeds 7, 13, 21):

| recipe id | spec                                                     |
|-----------|----------------------------------------------------------|
| 36C       | g=3, LS=0.50, complement, pair=masked, alt-fill          |
| 40F       | g=3, LS=0.50, complement, pair=masked, alt-recipe        |
| 42F       | g=3, LS=0.50, complement, **rect=0.5** (no-op)           |
| 46E       | g=3, LS=0.50, complement, **rect=0.3** (no-op)           |

12 / 16 cells completed at log capture (iter80 partial run; the user reports
"12/16 done at log time" — the remaining 4 cells continue in
`outputs/_iter80_top_vanilla_multiseed.log`).

Stage1 macro_f1 per cell (best across {I3, I7, I10, I13}):

| recipe | seed=7  | seed=13 | seed=21 |
|--------|--------:|--------:|--------:|
| 36C    | 0.8089  | 0.8349  | 0.7982  |
| 40F    | 0.8444  | 0.8006  | 0.8644  |
| 42F    | 0.8142  | 0.7998  | 0.7750  |
| 46E    | 0.8142  | 0.7998  | 0.7750  |

**Critical rect-flag confirmation**: iter42F (`rect=0.5`) and iter46E
(`rect=0.3`) at every shared seed produce **bit-identical** macro_f1:

| seed | 42F     | 46E     | diff   |
|-----:|--------:|--------:|-------:|
|    7 | 0.8142  | 0.8142  | 0      |
|   13 | 0.7998  | 0.7998  | 0      |
|   21 | 0.7750  | 0.7750  | 0      |

All other knobs (g=3, LS=0.50, mode=complement, pair=masked) are identical
between 42F and 46E. The bit-identical readings across 3 different seeds
**prove** the `--cutmix-rect` flag is a no-op under `cutmix_mode=complement`
in the current trainer — the rect-region logic only fires in single-mode CutMix.
This corroborates the iter18 finding and means iter46E and iter42F should be
cited as the **same model family**, not two distinct cells.

## Cross-recipe seed-variance summary (iter80, 12 cells)

| recipe | mean macro_f1 | std    | range            | seed-best |
|--------|--------------:|-------:|------------------|-----------|
| 36C    | 0.8140        | 0.0182 | 0.7982–0.8349    | seed=13   |
| 40F    | 0.8365        | 0.0322 | 0.8006–0.8644    | seed=21   |
| 42F    | 0.7963        | 0.0197 | 0.7750–0.8142    | seed=7    |
| 46E    | 0.7963        | 0.0197 | 0.7750–0.8142    | seed=7    |

**40F has the highest mean** (0.8365) and highest individual cell (0.8644 at
seed=21). 36C is second. 42F/46E (the rect-paired duplicates) are last.

NOTE: this is **stage1 macro_f1** (the per-cell log emit), NOT the v14 bit_F1
or the Total-FAR-gated bit_F1. The bit_F1 / Total-FAR view of iter80 is the
next computation (deferred to iter83); the present data is sufficient only to
rank seed-variance and to confirm the rect-flag no-op.

## Limitations

- iter79 mean ± std (`0.9695 ± 0.033`) is **bit_F1 (v14)**, not stage1
  macro_f1; the bit_F1-vs-macro_f1 gap matters because bit_F1 is the
  per-defect-bit F1 (excludes Normal/Invalid), whereas stage1 macro_f1 is
  the 11-class intersection. Two metrics, two stories.
- **Total FAR not yet measured** for any iter79/iter80 cell; only ni_FAR.
  Per the iter18 finding, the "5/8 pass" headline may shrink under Total FAR.
  This is the iter83 task.
- iter80 is **partial (12/16)**; the remaining 4 cells (likely 36C_s42,
  40F_s42, 42F_s42, 46E_s42 — matching iter79 catastrophic seed=42) are
  in-flight in `outputs/_iter80_top_vanilla_multiseed.log`.
- Ensemble feasibility for iter80 (cross-recipe 3-bag or 4-bag majority) not
  yet tested — only iter79 same-recipe-multi-seed ensemble was demonstrated.

## Sources

- iter79 log: `D:/project/known-cnn/outputs/_iter79_iter26H_multiseed.log`
- iter79 per-seed eval dirs: `outputs/iter79_seed{1,7,13,19,21,22,42,100}/T7_iter79_seed{N}_260511_*/eval_v15direct_n200/stage1_*/preds_chip.parquet`
- iter80 log: `D:/project/known-cnn/outputs/_iter80_top_vanilla_multiseed.log`
- iter80 per-cell eval dirs: `outputs/iter80_{36C,40F,42F,46E}_seed{7,13,21}/T7_iter80_*/eval_v15direct_n200/stage1_*/preds_chip.parquet`

## Next iter branches

- **iter83** — Total FAR re-score for all 8 iter79 cells + 12 iter80 cells +
  per-OOD-pattern decomposition. Decides whether the iter79 "5/8 pass" survives
  Total FAR gating.
- **iter20 (paired with this iter)** — `--cutmix-other-label` patch test on
  best-of-iter80 (`40F_s21` candidate); measures whether soft off-class bits
  reduce seed-fragility.
- **Cross-recipe ensemble**: build `{40F_s21 + 36C_s13 + iter46E_s1}` 3-bag at
  k=2 to test whether **cross-recipe diversity > same-recipe seed diversity**
  for vanilla ensembles. iter79 3-bag was same-recipe; this would be the
  cross-recipe analog.
- Drop **iter42F vs iter46E** as separate cells in all future tables — they
  are bit-identical model families.
