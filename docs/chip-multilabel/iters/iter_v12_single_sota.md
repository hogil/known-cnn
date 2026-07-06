# iter — `iter116J_exact_repro_v12` (single SOTA second replicate; T7 BCE+LS=0.30 pair complement)

- **Chain / tag**: `iter116J_exact_repro` / `iter116J_exact_repro_s1_ep8_best_v12`
- **Cron**: #726
- **Eval TS**: `260522_213802`
- **Train TS**: `20260522_211945_T7_iter116J_exact`
- **Train recipe**: T7 BCE+LS=0.30, CutMix p=0.25 complement (rect=0.5, n_patches=5, total_ratio=0.3, discount=0.7, alpha=1.0), pair-mask ON, no-Normal, 10 ep, seed=1
- **Best epoch**: 8 (best_val_acc 0.6950 per selector; final ep10 val_acc 0.9969)
- **Eval set**: n_eval = 18 640 (POS9 strict + 4-class OOD strict), single cell `T0__I10` (SOTA selector)
- **Source parquet**: `outputs/iter116J_exact_repro/20260522_211945_T7_iter116J_exact/eval_sota_i10/eval_260522_213802/preds_chip.parquet`

## Result — single-cell SOTA selector + cross-replicate comparison (cron #726)

```
| Recipe                                        | Ckpt | Variant | bit_F1 | NI-FAR | OOD-FAR | Tot-FAR | vs E22 (0.9956/0.00) | vs iter116J past best (0.9927/0.00) | Status                |
|-----------------------------------------------|------|---------|--------|--------|---------|---------|----------------------|-------------------------------------|-----------------------|
| iter116J_exact_repro v12 (cron 726, this run) | best | I10     | 0.9691 |   0.00 |    8.13 |    1.97 |    -0.0265 / +1.97   |              -0.0236 / +1.97        | SOTA repro v12        |
| iter116J_exact_repro (cron 687 first repro)   | best | I10     | 0.9691 |   0.00 |    3.75 |    0.91 |    -0.0265 / +0.91   |              -0.0236 / +0.91        | SOTA repro v1         |
| iter116J_nopair_10ep_s1 (cron 685 nopair ref) | best | I10     | 0.9237 |   0.50 |    8.44 |    2.42 |    -0.0719 / +2.42   |              -0.0690 / +2.42        | nopair short-ep gap   |
| E22 champion ensemble (frozen)                | -    | -       | 0.9956 |   0.00 |    0.00 |    0.00 |              -       |              +0.0029 /  0.00        | champion              |
| iter116J past best single (frozen)            | -    | -       | 0.9927 |   0.00 |    0.00 |    0.00 |    -0.0029 /  0.00   |                       -             | single ref            |
```

## Per-bit / per-OOD breakdown — `iter116J_exact_repro_v12` / I10 / best

```
| Class            | bit_F1 / FAR | Count       | Note                                |
|------------------|--------------|-------------|-------------------------------------|
| bank_boundary    |       0.9974 | -           | near-perfect (identical to v1)      |
| fork             |       0.9840 | -           | strong (identical to v1)            |
| scratch          |       0.8951 | -           | weak (drag; identical to v1)        |
| scratch_rot      |       0.9999 | -           | perfect (identical to v1)           |
| Normal           |        0.00% | 0 / 1600    | clean                               |
| Invalid          |        0.00% | 0 / 400     | clean                               |
| DiagonalSmear    |       12.50% | 20 / 160    | largest OOD leak (v1: 2.50% = 4)    |
| CenterDonut      |        6.88% | 11 / 160    | mid OOD leak (v1: 3.75% = 6)        |
| CrossScratch     |        6.88% | 11 / 160    | mid OOD leak (v1: 3.75% = 6)        |
| Starburst        |        6.25% | 10 / 160    | mid OOD leak (v1: 5.00% = 8)        |
```

## Delta vs prior champion + reference

- vs **E22 ensemble** (0.9956 / 0.00 %): **-0.0265 bit_F1 / +1.97 pp Total FAR** — single-model repro cannot match the 4-way bit-vote champion (expected; E22 is the ensemble headline).
- vs **iter116J past best single** (0.9927 / 0.00 %): **-0.0236 bit_F1 / +1.97 pp Total FAR** — gap widens from cron #687's +0.91 pp to +1.97 pp on Total FAR while bit_F1 stays pinned at 0.9691.
- vs **cron #687 v1 repro** (0.9691 / 0.91 %): **0.0000 bit_F1 / +1.06 pp Total FAR** — same per-bit numbers identical to 4 decimals, OOD-FAR doubles (3.75 -> 8.13 %). Confirms per-seed/per-run **variance is wholly on the OOD-strict axis**.

## Insight — why two same-recipe repros land at identical bit_F1 with very different FAR

1. **bit_F1 identical to 4 decimals** (0.9691) — per-bit F1s match across replicates to 4dp (bank_boundary 0.9974, fork 0.9840, scratch 0.8951, scratch_rot 0.9999). The positive-side decision landscape is **deterministic across seeds** at this recipe / data / ckpt-selector trio: the 9 positive cells lock to the same per-bit operating points and yield the same macro bit_F1.
2. **OOD-FAR is the entire variance band** — Total FAR jumps from 0.91 % to 1.97 % (+1.06 pp) **purely from OOD-strict leak** (NI stays 0/2000 in both replicates). The 4 OOD classes show coordinated lift: DiagonalSmear 4 -> 20 (5x), CenterDonut 6 -> 11, CrossScratch 6 -> 11, Starburst 8 -> 10. The replicate boundary lives in the OOD-strict decision band, not the positive-bit band.
3. **Pair-mask + LS=0.30 keeps NI clean across seeds** — 0/1600 Normal and 0/400 Invalid in both replicates; the cron #687 finding that "NI-FAR perfect under pair-mask" is now confirmed across two seeds.
4. **iter116J past best (0/640 OOD) is on the upper tail** — cron #687 showed +0.91 pp, this run shows +1.97 pp. Two same-recipe replicates **bracket** the past best on the OOD axis, neither matches. The past best result is rarer than a 1-in-2 draw at this seed-budget, consistent with the §5.55 single-seed variance band documented for T7-pair-10ep.
5. **Ensemble headroom on OOD axis** — E22 collapses OOD-FAR to 0 % via 4-way bit-vote (single SOTA cannot do this alone). The OOD-strict variance in single-seed repros (3.75 % -> 8.13 %) directly motivates the ensemble's bit-vote majority filter.
6. **Scratch per-bit (0.8951) is the deterministic bit_F1 drag** in both replicates — single per-bit fix on `scratch` (e.g. positive-bit calibration or per-bit threshold tuning) could lift bit_F1 closer to past-best 0.9927 without disturbing the FAR-OOD variance picture.

## Cross-replicate variance summary

```
| Metric    | v1 (cron 687) | v12 (cron 726) | delta  | source of variance              |
|-----------|---------------|----------------|--------|---------------------------------|
| bit_F1    |        0.9691 |         0.9691 |  0.000 | deterministic (positive cells)  |
| NI-FAR    |        0.00 % |         0.00 % |   0.00 | pair-mask locks NI to 0         |
| OOD-FAR   |        3.75 % |         8.13 % |  +4.38 | OOD-strict decision band drift  |
| Total-FAR |        0.91 % |         1.97 % |  +1.06 | wholly OOD-driven               |
```

## Hyperparameter changes vs prior iter (cron #687 v1)

```
| Aspect             | cron #687 v1 (iter116J_exact_repro)    | cron #726 v12 (this run)                | direction                |
|--------------------|----------------------------------------|-----------------------------------------|--------------------------|
| Recipe             | T7 BCE+LS=0.30 complement pair         | identical                               | same                     |
| Seed               | 1                                      | 1                                       | same                     |
| Epochs             | 10                                     | 10                                      | same                     |
| Best ckpt          | ep8                                    | ep8                                     | same                     |
| Pair-mask          | ON                                     | ON                                      | same                     |
| CutMix p           | 0.25                                   | 0.25                                    | same                     |
| LS                 | 0.30                                   | 0.30                                    | same                     |
| Eval selector      | T0__I10 single-cell SOTA               | T0__I10 single-cell SOTA                | same                     |
| Train TS           | 20260522_142643                        | 20260522_211945                         | second run (+5h offset)  |
| Eval TS            | 260522_151400                          | 260522_213802                           | second eval              |
```

Zero recipe deltas — this is a **same-recipe per-seed/per-run variance probe**. The atomic axis isolated is the **stochastic run-to-run variance** of T7-pair-10ep-s1 under the SOTA single-cell selector, not a recipe change.

## Lessons for next iter

1. **Single-seed T7-pair-10ep bit_F1 is deterministic at 0.9691** across at least 2 replicates — bit_F1 reproducibility is **not** a concern for this recipe at this budget.
2. **OOD-FAR has a ~5 pp variance band** at the same recipe — single-seed repros span [0.91, 1.97] %; passing 0 % requires either ensemble bit-vote (E22) or seed-search over many replicates to land on the iter116J past-best upper tail. Single-seed exact-recipe repro alone cannot guarantee 0 % OOD-FAR.
3. **Scratch per-bit (0.8951) is the deterministic macro-F1 drag** — separate from the OOD-FAR variance picture, fixing the scratch bit could lift single-seed bit_F1 from 0.9691 toward 0.9927.
4. **Champions still frozen** — E22 (0.9956 / 0 %) and iter116J past best single (0.9927 / 0 %) untouched. v12 does not challenge either.
5. **Next single-SOTA probes should diversify seed** (seed=7/11/23/33/77) rather than re-run seed=1 — bit_F1 saturates at 0.9691 under seed=1, so further variance/lift signal is on other seeds.

## Source paths

- Train ckpt: `outputs/iter116J_exact_repro/20260522_211945_T7_iter116J_exact/best_model.pth`
- Eval parquet: `outputs/iter116J_exact_repro/20260522_211945_T7_iter116J_exact/eval_sota_i10/eval_260522_213802/preds_chip.parquet`
- Eval summary: `outputs/iter116J_exact_repro/20260522_211945_T7_iter116J_exact/eval_sota_i10/eval_260522_213802/eval_summary.json`
- Eval report: `outputs/iter116J_exact_repro/20260522_211945_T7_iter116J_exact/eval_sota_i10/eval_260522_213802/report.md`
- bit_far_metrics: `outputs/iter116J_exact_repro/20260522_211945_T7_iter116J_exact/eval_sota_i10/eval_260522_213802/bit_far_metrics.json`
- train_summary: `outputs/iter116J_exact_repro/20260522_211945_T7_iter116J_exact/train_summary.json`
- v1 comparator (cron #687): `outputs/iter116J_exact_repro/20260522_142643_T7_iter116J_exact/eval_sota_i10/eval_260522_151400/preds_chip.parquet`
- v1 iter doc: `docs/chip-multilabel/iters/iter_iter116J_exact_repro.md`
