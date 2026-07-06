# chain v9 iter 1 — KD_v8 (alpha=0.5 T=2 skip-cm) re-eval landing, collapse confirmed

- TS: 260517_121833 (train) / 260517_150413 (eval n2000)
- Source: `outputs/KD_v8_a05_T2_skipcm/20260517_121833_T7_KD_v8_a05_T2_skipcm/eval_n2000_pred/stage1_260517_150413/preds_chip.parquet`
- Recipe: T7 BCE+LS=0.30 + FCM-PM CutMix g=3 corner + KD (teacher=iter116J g3_ls30 single,
  **alpha=0.5**, T=2, --kd-skip-on-cutmix), seed=1
- Teacher: `outputs/iter116J_g3_ls30/T7_iter116J_g3_ls30_260513_010015` (single member, bit_F1 0.9748)
- Baseline to compare: KD_v7 same recipe but **alpha=0.3** → bit_F1 0.9265 / Total FAR 0.00 % (I10)
- Run health: `val_macro_f1: 0.0` at `epoch: 1` in `eval_summary.json` model_meta — training
  collapsed at ep01, the eval is on the ep01 ckpt because no later epoch beat it.
- Eval health: re-eval was GPU-gated by `_run_chain_v9.sh::wait_gpu_free` (60 % threshold);
  dispatched after foreign GPU process dropped to 55 % usage (cron 6 → cron 7 window).

## Hypothesis

KD_v7 at alpha=0.3 (cron 3, chain v6) was the first non-collapse KD result and reached
bit_F1 0.9265 / FAR 0.00 % (I10) — a regulariser-quality student.  The chain v8 phase 0
re-dispatch of KD_v8 at alpha=0.5 tested whether pushing KD weight up by +0.2 would let
the student inherit more of the teacher's saturation (teacher val_acc ≈ 0.9969) and
exceed the alpha=0.3 ceiling.

Plausible-positive: more KD weight → student probability mass concentrates on teacher
soft target → 9 positive cells get cleaner bit decoding, FAR stays 0 because the teacher
target on Normal/Invalid/OOD chips is itself near-zero on the 4 active bits.

Plausible-negative: alpha=0.5 inflates the soft target's residual probability mass on
all 4 active bits even on negative-cell chips → over-positive predictions → Total FAR
explodes.  This is the documented chain v7 KD_v4 failure (alpha=0.5, T=4, LS=0.20:
bit_F1 0.8298, Total FAR 22.77).

## Eval n2000 (POS9 strict + 4 OOD strict)

```
| Variant | bit_F1 | NI-FAR | OOD-FAR | Total FAR |
|---------|--------|--------|---------|-----------|
| I3      | 0.9274 | 100.00 |  100.00 |    100.00 |
| I7      | 0.9227 | 100.00 |  100.00 |    100.00 |
| I10     | 0.8924 |  32.15 |   79.46 |     57.15 |
| I13     | 0.8365 |  31.45 |   71.12 |     52.41 |
```

Best variant on bit_F1 axis: I3 (0.9274) — but FAR 100 % renders the cell unusable.
Lowest-FAR variant: I13 (52.41 %) — still 2 orders of magnitude above KD_v7 I13 (0.00 %).

## Delta vs KD_v7 (alpha=0.3 with same skip-cm recipe)

```
| Variant | dbit_F1 vs KD_v7 | dTotal_FAR_pp vs KD_v7  | direction      |
|---------|------------------|-------------------------|----------------|
| I3      |          +0.0189 |                  +96.25 | bit up, FAR up |
| I7      |          +0.0113 |                  +79.58 | bit up, FAR up |
| I10     |          -0.0341 |                  +57.15 | both worse     |
| I13     |          -0.0534 |                  +52.37 | both worse     |
```

The I10 / I13 cells regress on both axes simultaneously — the alpha bump did not pay
off even on bit_F1, and it exploded FAR.  The I3 / I7 cells appear to "win" on bit_F1
but the gain is a meaningless artefact of an over-positive student that fires every
bit on every chip (which incidentally also activates the 9 positive cells' bits and
inflates micro-recall toward 1 on those rows).  This is the same collapse signature
as KD_v1-v6: the I3/I7 cells (no gate) inherit the over-positive output, while the
I10/I13 cells (entropy gate / max-prob + dist-band gate) catch most of it but not all.

## Delta vs iter116J past best (0.9927 / 0.00 %)

```
| Variant | dbit_F1 vs SOTA | dTotal_FAR_pp vs SOTA |
|---------|-----------------|-----------------------|
| I3      |         -0.0653 |               +100.00 |
| I7      |         -0.0700 |               +100.00 |
| I10     |         -0.1003 |                +57.15 |
| I13     |         -0.1562 |                +52.41 |
```

No variant of KD_v8 lands inside the SOTA's Pareto envelope.

## Hyperparameter changes vs prior iter (KD_v7)

```
| Aspect          | KD_v7                | KD_v8                | direction |
|-----------------|----------------------|----------------------|-----------|
| KD alpha        | 0.3                  | 0.5                  | up +0.2   |
| KD temperature  | 2                    | 2                    | same      |
| --kd-skip-on-cm | yes                  | yes                  | same      |
| LS              | 0.30                 | 0.30                 | same      |
| CutMix g        | 3 corner             | 3 corner             | same      |
| seed            | 1                    | 1                    | same      |
| teacher         | iter116J single      | iter116J single      | same      |
```

Exactly one atomic change (alpha 0.3 → 0.5) — clean ablation, satisfies
`feedback_atomic_method_iteration`.

## Insights

1. **KD viable corner at T=2 is alpha ∈ [0.3, 0.3]**, i.e. a single point.
   alpha=0.2 (KD_v5) collapses (bit_F1 0.1093, FAR 99.47).  alpha=0.3 (KD_v7) viable
   (0.9265, 0.00).  alpha=0.5 (KD_v8 this iter) collapses on the FAR axis (57.15 % at
   I10) and on the bit axis at the gated cells.  alpha=0.7 (KD_v2) was previously
   shown over-smoothed (0.7874, 0.08).  The KD search budget across the 5-point alpha
   grid (0.2, 0.3, 0.5, 0.7) at T=2 has therefore evidentiarily established alpha=0.3
   as the **only** viable corner — there is no second cell to fall back to.
2. **The training-time collapse signature `val_macro_f1=0 at epoch=1`** persists even
   with `--kd-skip-on-cutmix`.  The skip-cm flag was the breakthrough that enabled
   alpha=0.3 to converge, but at alpha=0.5 the KD loss on the **non-CutMix** batches
   alone is enough to drag the student into the degenerate solution by ep01 (no later
   epoch ever exceeds, so the best ckpt is the ep01 collapsed state).
3. **I3 / I7 cells lie when the student is over-positive**.  At Total FAR 100 % the
   I3/I7 bit_F1 numbers (0.9274, 0.9227) are mechanically high because the prediction
   fires every bit on every chip — the per-cell positive recall approaches 1, the
   precision drops to (true positives) / (n_chip × 4 bits), but the per-class F1
   averaged over the 9 positive cells still reads ≈ 0.92.  Reporting bit_F1 without
   the paired Total FAR (as the `feedback_chip_multilabel_train_eval_composition`
   absolute rule mandates) would have hidden this collapse.
4. **Ensemble pool is unchanged.**  KD_v8 cannot enter the chain v7+v8 3-student
   majority-vote pool (would drag the per-bit majority toward over-positive on
   Normal/Invalid/OOD chips).  Champion remains `vote_majority_bits` of
   {iter116J s=1, iter116J_clone_s77, KD_v7} at 0.9941 / 0.00 % (I10).

## Lessons for next iter

1. KD α grid at T=2 is closed at α=0.3.  Do not re-dispatch KD_v9 (α=0.2) or KD_v10
   (α=0.3 T=1) inside the same recipe family — the T=1 cell could in principle
   widen the viable window, but the alpha grid evidence (collapse at both α=0.2 and
   α=0.5) suggests the regulariser-only regime is too narrow to mine further at this
   eval-set / teacher pairing.
2. If chain v9 needs a 4th student to extend the ensemble pool, do **not** seek it
   in the KD recipe family.  Either (a) train a fresh seed of the base T7 g=3 LS=0.30
   recipe at seed 42 / 11 / 23 / 77 (cheaper, known-viable), or (b) try a different
   cutmix-p (chain v9 phase 2 already queues 0.15 / 0.20 / 0.30 / 0.35 / 0.40 at
   seed 42) — both of those have a non-zero probability of contributing a
   complementary deciding vote without the KD collapse risk.
3. The chain v9 `wait_gpu_free` gate worked: the re-eval that previously OOM'd at
   chain v8 cron 4 and cron 5 now landed cleanly.  Keep the gate for chain v10.

## Source paths

- preds parquet: `outputs/KD_v8_a05_T2_skipcm/20260517_121833_T7_KD_v8_a05_T2_skipcm/eval_n2000_pred/stage1_260517_150413/preds_chip.parquet`
- eval summary: same dir / `eval_summary.json` (model_meta confirms `val_macro_f1=0` at ep01)
- supervisor log: `outputs/_chain_v9_summary.log`
- runner log: `outputs/_chain_v9_runner.log`
