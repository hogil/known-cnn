# chain v5 iter 2 — iter50_clone_seed42_v4

- TS: 260516_105111 (train) / 260516_115724 (eval n2000)
- Source: `outputs/iter50_clone_seed42_v4/20260516_105111_T7_iter50_clone_seed42_v4/eval_n2000_pred/stage1_260516_115724/preds_chip.parquet`
- Recipe: T7 BCE+LS=0.30 + FCM-PM CutMix g=3 corner, seed=42 (vs iter1 seed=99)
- Baseline (iter116J): bit_F1 0.9927 / Total FAR 0.00%

## Eval n2000 (POS9 strict + 4 OOD strict)

```
| Variant | bit_F1 | NI-FAR | OOD-FAR | Total FAR |
|---------|--------|--------|---------|-----------|
| I3      | 0.9596 |   8.10 |   13.59 |      9.43 |
| I7      | 0.9532 |   4.80 |    9.22 |      5.87 |
| I10     | 0.9583 |   0.00 |    1.25 |      0.30 |
| I13     | 0.9320 |   0.00 |    1.09 |      0.27 |
```

best variant: I10 — bit_F1 0.9583, Total FAR 0.30% (8 OOD FP / 2640)

## Delta vs iter116J baseline

```
| Variant | dbit_F1 | dTotal_FAR_pp |
|---------|---------|---------------|
| I3      | -0.0331 |         +9.43 |
| I7      | -0.0395 |         +5.87 |
| I10     | -0.0344 |         +0.30 |
| I13     | -0.0607 |         +0.27 |
```

Much closer to the iter116J baseline than iter1 — bit_F1 only -0.034 at
the I10 cell. Total FAR rises 0.30pp (8 OOD FP), still strong gating.

## Insights / hypotheses (vs iter1 seed=99)

- Seed=42 is a **~0.08 better bit_F1 draw than seed=99** at I10
  (0.9583 vs 0.8778). That alone is a 9× variance signal — chain v5
  hypothesis "T7 LS=0.30 g=3 is single-seed optimum" is contradicted
  by this 2-point spread.
- I13 NI-FAR identical (0%) across both seeds — gate is bullet-proof on
  Normal/Invalid regardless of seed draw.
- I3/I7 (non-gated) FAR spreads widely: iter1 OOD-FAR 5-7%, iter2 9-13%.
  The gating cells (I10/I13) absorb this variance — confirming gates are
  the **robustness mechanism**, not the headline-recipe.
- Still 0.034 below iter116J at the best cell — seed=1 of iter116J might
  itself be a positive outlier (will revisit after iters 3/4 sample more
  of the seed distribution).
