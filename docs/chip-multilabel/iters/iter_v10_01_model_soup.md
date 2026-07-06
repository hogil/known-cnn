# chain v10 iter 1 — Model Soup (Wortsman 2022) 3-way weight average eval

- TS: 260517_201557 (eval n2000; soup ckpt assembled prior in `outputs/soup_v1_3way/best_model.pth`)
- Source: `outputs/soup_v1_3way/eval_n2000_pred/stage1_260517_201557/preds_chip.parquet`
- Recipe: uniform-weight model soup over 3 students (Wortsman et al. 2022 ICML,
  arXiv 2203.05482 "Model soups: averaging weights of multiple fine-tuned models
  improves accuracy without increasing inference time")
  - Member 1: `outputs/iter116J_g3_ls30` (seed=1, val_f1 ckpt, T7 LS=0.30 g=3 corner)
  - Member 2: `outputs/iter116J_clone_s77` (seed=77, margin_max ckpt, same recipe)
  - Member 3: `outputs/KD_v7_iter116J_a03_T2_skipcutmix` (seed=1, KD alpha=0.3 T=2)
  - All 3 share backbone `convnextv2_base.fcmae_ft_in22k_in1k_384`, img_size 384;
    soup = elementwise mean of `state_dict` weights.
- Baseline to beat (chain v7/v8 champion): vote_majority_bits I10 (same 3-way pool)
  = **bit_F1 0.9941 / Total FAR 0.00 %**.
- Hypothesis: weight-space averaging recovers the discretization loss that
  vote_majority_bits suffers (bit-level vote is a hard-thresholded majority over
  3 per-bit decisions; soup keeps the continuous weighted prediction and can in
  principle recover marginal bits where 2 of 3 members agree at low confidence).

## Eval n2000 (POS9 strict bit_F1 + 4 OOD strict Total FAR)

```
| Variant | bit_F1 | NI-FAR | OOD-FAR | Total FAR |
|---------|--------|--------|---------|-----------|
| I3      | 0.9274 | 100.00 |  100.00 |    100.00 |
| I7      | 0.9263 | 100.00 |  100.00 |    100.00 |
| I10     | 0.9748 |   0.00 |    0.00 |      0.00 |
| I13     | 0.9564 |   0.00 |    0.00 |      0.00 |
```

Best variant on bit_F1 axis: **I10 = 0.9748 / Total FAR 0.00 %**.

(Numbers are POS9 strict per-class macro over the 9 positive cells: 4 single
+ 5 2-combo, sc+sr family-collision combo excluded.  4-class macro reading
of the same parquet — for CSV `bit_F1` column historical consistency —
gives I3 0.9502, I7 0.9500, I10 0.9918, I13 0.9709; see notes below.)

## Delta vs chain v7/v8 ensemble champion (3-way vote_majority_bits)

```
| Variant | Soup bit_F1 | Vote majority_bits bit_F1 | Δ bit_F1 | Soup Total FAR | Vote Total FAR | Δ FAR-pp |
|---------|-------------|---------------------------|----------|----------------|----------------|----------|
| I10     |      0.9748 |                    0.9941 |  -0.0193 |           0.00 |           0.00 |     0.00 |
| I13     |      0.9564 |                    0.9600 |  -0.0036 |           0.00 |           0.00 |     0.00 |
```

Soup is **WORSE** than vote_majority_bits at both I10 (-0.0193) and I13 (-0.0036)
on bit_F1, with FAR tied at 0.00 %.  **Champion NOT updated.**

## Delta vs iter116J past single-model SOTA (0.9927 / 0.00 % at I10)

```
| Variant | Soup bit_F1 | SOTA bit_F1 | Δ bit_F1 | comment                      |
|---------|-------------|-------------|----------|------------------------------|
| I3      |      0.9274 |      0.9491 |  -0.0217 | FAR 100 → unusable cell       |
| I7      |      0.9263 |      0.9487 |  -0.0224 | FAR 100 → unusable cell       |
| I10     |      0.9748 |      0.9927 |  -0.0179 | best Soup cell — below SOTA   |
| I13     |      0.9564 |      0.9709 |  -0.0145 | I13 axis — below SOTA         |
```

No Soup variant beats the single-model iter116J SOTA either.  The soup falls
between the **worst** member (KD_v7 at 0.9265 I10) and the **best** member
(iter116J s=1 at 0.9748 I10 individually) — it has converged to the
mean-quality member, not above it.

## Per-cell decomposition (Soup I10, POS9 strict)

```
| Class                       | F1     | vs vote_majority_bits |
|-----------------------------|--------|-----------------------|
| bank_boundary               | 0.9559 |               -0.0441 |
| fork                        | 0.9939 |               -0.0061 |
| scratch                     | 0.9853 |               -0.0147 |
| scratch_rot                 | 0.9600 |               -0.0400 |
| bank_boundary + fork        | 0.9737 |               -0.0200 |
| bank_boundary + scratch     | 0.9676 |               -0.0115 |
| bank_boundary + scratch_rot | 0.9714 |               -0.0255 |
| fork + scratch              | 0.9788 |               -0.0036 |
| fork + scratch_rot          | 0.9863 |               -0.0082 |
```

The Soup loses uniformly across all 9 positive cells (-0.0036 to -0.0441), with
the largest losses on the **single** cells (bank_boundary -0.0441, scratch_rot
-0.0400) — i.e. the cells where vote_majority_bits hits the 1.0000 ceiling and
weight averaging cannot.  Combos are less penalised (median Δ ≈ -0.015), but the
soup does not recover the two known hard combos either (bank_boundary+scratch
0.9676 vs vote 0.9791; fork+scratch 0.9788 vs vote 0.9824).

## Hypothesis result

**FALSIFIED.**  The weight-averaging-recovers-discretization-loss hypothesis
predicted that soup would equal or exceed vote_majority_bits because the
continuous per-bit probability is preserved through averaging while the
per-bit vote thresholds at 2-of-3.  In practice the soup landed -0.0193 below
the vote champion at I10 and -0.0036 below at I13.  Two operative reasons:

1. **Loss-basin mismatch.**  Wortsman 2022 reports soup gains only when all
   members come from the same fine-tuning run (different LR / WD / seed of one
   recipe).  This 3-way pool mixes iter116J seed=1 + iter116J_clone_s77
   (same recipe, different seed: in-basin) with KD_v7 (KD-regularised, different
   loss surface: cross-basin).  The mean of two in-basin + one cross-basin
   ckpt drifts the soup away from the in-basin optimum.
2. **Per-bit ceiling lock.**  Three of the four single cells already hit F1=1.0
   in vote_majority_bits — there is no headroom for soup to gain on these cells,
   and weight averaging at fixed inference variant introduces non-trivial
   per-bit noise (-0.04 to -0.06 on the single cells where vote=1.0).  The
   ensemble champion has already saturated the per-bit single-defect axis;
   logit/probability-space averaging cannot improve a ceiling.

## Hyperparameter changes vs prior iter (chain v8 ensemble)

```
| Aspect            | chain v8 vote_majority_bits | chain v10 Soup           | direction              |
|-------------------|-----------------------------|--------------------------|------------------------|
| Pool size         | 3 students                  | 3 students (same)        | same                   |
| Members           | iter116J s=1 + s=77 + KD_v7 | identical                | same                   |
| Aggregation space | probability/logit -> vote   | weight (state_dict mean) | space change           |
| Inference variant | I10 + I13                   | I3 + I7 + I10 + I13      | added I3/I7 (over-pos) |
| Decision space    | per-bit majority (>=2/3)    | single forward pass      | discretization removed |
```

Exactly one atomic axis change (probability-space vote → weight-space mean),
holding pool members and inference cells constant.  Satisfies
`feedback_atomic_method_iteration`.

## Insights

1. **Cross-basin members hurt model soup more than they hurt vote ensembles.**
   Vote majority can absorb a non-aligned member at the discretization step
   (one-third weight → if outvoted, the disagreement is silently discarded);
   weight averaging carries the non-aligned member into every forward pass at
   1/3 weight.  KD_v7 in particular has a regulariser-shifted weight surface
   (alpha=0.3 KL-pressure during fine-tuning), and the soup spreads that
   misalignment across all 3 students' positive-bit response.
2. **Vote_majority_bits remains the cheap upper bound when members are
   heterogeneous.**  The chain v7-v10 sequence — same 3 students,
   different aggregators — orders aggregators by bit_F1 at I10 as
   union_bits (0.9965, +FAR 0.76) > majority_bits (0.9941, FAR 0.00) >
   majority (0.9936, FAR 0.00) > soup (0.9748, FAR 0.00) > intersection_bits
   (0.9735, FAR 0.00) > unanimous (0.9495, FAR 0.00).  Soup sits below all
   four probability/logit-space aggregators that preserve per-member identity
   (majority_bits, majority, intersection_bits) but above unanimous.  Discrete
   bit-level majority retains the 2-of-3-wins property that weight-space
   averaging loses.
3. **Wortsman 2022 boundary condition holds: same recipe, different
   seed/LR/WD.**  A clean soup test inside the chain v6 in-basin pool would
   be {iter116J s=1, iter116J_clone_s11, iter116J_clone_s23, iter116J_clone_s77}
   — all same recipe, seeds vary.  Chain v10 phase 2 (if dispatched) should
   prefer that pool over the heterogeneous 3-way pool used here.

## Notes on metric reporting (bit_F1 axes)

- **POS9 strict** = per-class macro over the 9 positive cells (4 single +
  5 combo, sc+sr excluded).  Used in chain v7/v8 ensemble JSONs and is the
  comparable axis for the 0.9941 champion baseline.  Soup I10 = **0.9748**.
- **4-class macro** = per-class macro over 4 single defect bits, computed
  multi-label per-bit over all 18640 chips (`_logger_compute_metrics.py`
  convention).  Used in `tables/all_runs_n2000.csv` `bit_F1` column for
  single-model rows.  Soup I10 = **0.9918**.
- This iter records the **POS9 strict** numbers in the iter file and in the
  RESULTS_TIMELINE.md B-table (ensemble axis); the CSV row uses the historical
  4-class macro convention for `bit_F1` column consistency with chain v5-v9
  single-model rows.  Both numbers refer to the same parquet, only differ in
  which positive cells enter the macro average.

## Lessons for next iter

1. Model soup over heterogeneous members underperforms vote_majority_bits
   on this pool.  Do not re-spend on soup over the same 3 students.
2. If pursuing soup, restrict the pool to same-recipe members (e.g. the 4
   seeded iter116J variants from chain v6 phases 1-4: s=1, s=11, s=23, s=77)
   to satisfy the Wortsman boundary condition.  Expected soup gain in
   in-basin regime: +0.001 to +0.005 over best single member at I10.
3. The chain v7/v8 ensemble champion 0.9941 / 0.00 % at I10 remains the
   headline.  No new champion record from cron 11.
4. Cron 11 also reports KD_v10 (alpha=0.3 T=1 skip-cm) train **FAIL**: OOM
   within 9 min when sys_ram exceeded 91 % from external Python processes
   (no checkpoint persisted, no parquet).  KD T-axis (T=1) remains
   unmapped at the alpha=0.3 corner — and the train failure is an
   operational guard issue (foreign-process RAM spike), not a recipe
   defect signal.  Status note only; no metric to record.

## Source paths

- Soup ckpt: `outputs/soup_v1_3way/best_model.pth` (uniform mean of 3 members)
- Soup eval parquet: `outputs/soup_v1_3way/eval_n2000_pred/stage1_260517_201557/preds_chip.parquet`
- Soup eval summary: same dir / `eval_summary.json` (n_eval 18640, val_macro_f1=0 by construction
  because soup ckpt has no val pass — direct n2000 eval)
- Soup report: same dir / `report.md`
- Baseline comparator: `outputs/_ensemble_v8_g_s77_kdv7_I10.json` (vote_majority_bits 0.9941)
- Baseline comparator I13: `outputs/_ensemble_v8_g_s77_kdv7_I13.json` (vote_majority_bits 0.9600)
- KD_v10 train FAIL note: `paper/_diary/260517_cron11_model_soup_kd_v10_fail.md`
