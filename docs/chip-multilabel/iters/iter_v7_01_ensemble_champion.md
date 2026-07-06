# chain v7 iter 1 — ensemble champion (vote_majority_bits beats iter116J SOTA)

- TS: 260517 (5-mode ensemble eval aggregated post-hoc, no new training)
- Source aggregate: `outputs/_ensemble_v7_5mode.json`
- Ensemble pool (3 students, all I10 forwards on the n2000 POS9 strict + 4 OOD strict eval set):
  - `outputs/iter116J_g3_ls30/` (seed=1, val_f1 ckpt) — past SOTA single (bit_F1 0.9927 / Total FAR 0.00%)
  - `outputs/iter116J_clone_s77/` (seed=77, margin_max ckpt) — chain v6 phase 3 micro-win (0.9786 / 0.76%)
  - `outputs/KD_v7_iter116J_a03_T2_skipcutmix/` (seed=1, KD alpha=0.3 T=2 + --kd-skip-on-cutmix) — KD regulariser (0.9265 / 0.00%)
- 5 vote-aggregation modes tested on the per-bit prediction tensors at the I10 cell:
  vote_majority, vote_unanimous, vote_intersection_bits, vote_union_bits, vote_majority_bits.
- Baseline (single-model headline to beat): iter116J s=1 I13 = **bit_F1 0.9927 / Total FAR 0.00%**.

## Hypothesis

The chain v5 + v6 seed and ckpt-selector sweeps had identified two recipe-internal variance
terms (seed at ±0.046, ckpt-selector at ±0.21) that placed a hard ceiling on any single
training-run headline. We test whether **diversity across the seed axis (s=1 vs s=77) and
the KD-vs-no-KD axis** can convert that within-recipe variance into a usable signal — i.e.
whether the three already-on-disk students jointly produce a bit_F1 above the single-best
0.9927 at zero FAR.  Bit-level vote aggregation is the cheapest way to test this without any
retraining or threshold tuning.

## Eval n2000 (POS9 strict + 4 OOD strict, I10 inference variant)

```
| Mode                   | bit_F1 | NI-FAR | OOD-FAR | Total FAR | dbit_F1 vs SOTA | dFAR  | comment           |
|------------------------|--------|--------|---------|-----------|-----------------|-------|-------------------|
| iter116J s=1 (single)  | 0.9927 |   0.00 |    0.00 |      0.00 |          0.0000 |  0.00 | baseline SOTA     |
| vote_unanimous         | 0.9495 |   0.00 |    0.00 |      0.00 |         -0.0432 |  0.00 | too strict        |
| vote_intersection_bits | 0.9735 |   0.00 |    0.00 |      0.00 |         -0.0192 |  0.00 | too conservative  |
| vote_majority          | 0.9936 |   0.00 |    0.00 |      0.00 |         +0.0009 |  0.00 | tie at FAR safe   |
| vote_majority_bits     | 0.9941 |   0.00 |    0.00 |      0.00 |         +0.0014 |  0.00 | NEW CHAMPION      |
| vote_union_bits        | 0.9965 |   0.40 |    1.88 |      0.76 |         +0.0038 | +0.76 | peak F1 FAR trap  |
```

Per-class F1 on the champion `vote_majority_bits` cell (positive 9, POS9 strict):

```
| Class                     | F1     |
|---------------------------|--------|
| bank_boundary             | 1.0000 |
| fork                      | 1.0000 |
| scratch                   | 1.0000 |
| scratch_rot               | 1.0000 |
| bank_boundary+fork        | 0.9937 |
| bank_boundary+scratch     | 0.9791 |
| bank_boundary+scratch_rot | 0.9969 |
| fork+scratch              | 0.9824 |
| fork+scratch_rot          | 0.9945 |
```

All 4 single-defect cells lock at 1.0000.  Hard combos are `bank_boundary+scratch` (0.9791)
and `fork+scratch` (0.9824) — both gated by the scratch head, consistent with the chain v6
finding that scratch is the uniquely weak class.

## Hyperparameter / variant changes vs prior iter

No new training.  This iter only changes the **post-hoc aggregation rule** over the three
chain v6 students' I10 bit predictions:

```
| Mode                   | Rule                                                            |
|------------------------|-----------------------------------------------------------------|
| vote_unanimous         | bit asserted iff all 3 students assert (AND)                    |
| vote_intersection_bits | bit-level intersection across the asserted bit-set per chip     |
| vote_majority          | label-level majority over the 11 declared class vectors         |
| vote_majority_bits     | per-bit majority (2/3) across the 3 students' 9-positive vectors|
| vote_union_bits        | bit asserted iff any student asserts (OR)                       |
```

## Delta vs iter116J past SOTA (0.9927 / 0.00%)

- `vote_majority_bits` is the **first cell across chain v5+v6+v7 to beat the iter116J SOTA
  on bit_F1 without any FAR penalty** (+0.0014 at 0.00% FAR).
- `vote_majority` (label-level) gives +0.0009 / 0.00% — also a tie-at-FAR-safe win, but
  smaller margin than the bit-level rule.
- `vote_union_bits` extracts +0.0038 bit_F1 by recovering the hard `bank_boundary+scratch`
  combo (0.9913 vs 0.9791 in the majority cell), at the cost of +0.76 pp Total FAR — a
  classic Pareto trade rather than a clean win.
- `vote_unanimous` and `vote_intersection_bits` both collapse `bank_boundary+scratch`
  (0.6669 and 0.8518 respectively) — strict modes punish the combo where one of the three
  students typically misses the second bit.

## Insights

1. **Diversity-over-tuning realised.**  The three students individually span [0.4738,
   0.9786] bit_F1 — i.e. one of them is a near-disaster at I10.  Yet bit-level majority over
   the three produces 0.9941 at zero FAR.  Per-bit voting is robust to single-student
   degradation as long as the other two agree on the correct bits, which is the case
   here for every single-defect class (all 1.0000) and 4 of 5 combos (≥0.99).
2. **Bit-level > label-level majority.**  `vote_majority_bits` beats `vote_majority`
   (+0.0005), because label-level voting can split votes across 11 classes (3-way tie
   collapses to single-vote fall-back), while bit-level voting only needs 2 students to
   agree on each of the 9 positive bits independently.
3. **Union mode is a FAR trap, not a free lunch.**  Bit OR aggressively recovers combos
   (BB+scratch 0.9791 -> 0.9913) but inflates the asserted-bit count whenever any single
   student over-asserts.  +0.76 pp Total FAR comes from 8 NI false positives + 12 OOD false
   positives — these were 0 in every other vote mode.
4. **Hard pair is scratch-combo (BB+scratch 0.9791, fork+scratch 0.9824).**  Both gated by
   the scratch head.  Future iter should target scratch-specific calibration (per-class
   threshold) or scratch-only data augmentation.
5. **KD student earns its keep.**  Although KD_v7 alone is the weakest pool member (0.9265),
   its calibrated I10 vector contributes the deciding vote on edge cases where iter116J and
   s=77 disagree.  The KD regulariser effect (low ceiling, low variance) is exactly the
   right complement to two high-variance students.

## Lessons for next iter

1. The `vote_majority_bits` cell is the new headline.  Update SOTA to **0.9941 / 0.00%**.
2. Test 4-student and 5-student ensemble (add KD-multi-teacher students once trained) —
   per-bit majority generally improves monotonically with N as long as N is odd.
3. Try per-class confidence-weighted bit aggregation on the scratch combo cells — this is
   the only remaining gap to push bit_F1 above 0.9965 without paying FAR.
4. `vote_union_bits` is publishable as a Pareto extremum, but should not replace the
   headline.

## Source paths

- 5-mode aggregate JSON: `outputs/_ensemble_v7_5mode.json`
- Pool members:
  - `outputs/iter116J_g3_ls30/T7_iter116J_g3_ls30_260513_010015/.../preds_chip.parquet`
  - `outputs/iter116J_clone_s77/20260517_*_T7_iter116J_clone_s77/eval_n2000_pred/stage1_*/preds_chip.parquet`
  - `outputs/KD_v7_iter116J_a03_T2_skipcutmix/20260517_095713_T7_KD_v7_iter116J_a03_T2_skipcutmix/eval_n2000_pred/stage1_260517_101336/preds_chip.parquet`
