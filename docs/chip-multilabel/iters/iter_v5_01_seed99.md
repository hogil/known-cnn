# chain v5 iter 1 — iter50_clone_seed99_v3

- TS: 260516_073008 (train) / 260516_080044 (eval n2000)
- Source: `outputs/iter50_clone_seed99_v3/20260516_073008_T7_iter50_clone_seed99_v3/eval_n2000_pred/stage1_260516_080044/preds_chip.parquet`
- Recipe: T7 BCE+LS=0.30 + FCM-PM CutMix g=3 corner, seed=99, batch=2 accum=8, lr=1e-4
- Baseline (iter116J): bit_F1 0.9927 / Total FAR 0.00% (T7 LS=0.30 g=3 seed=1)

## Eval n2000 (POS9 strict + 4 OOD strict)

```
| Variant | bit_F1 | NI-FAR | OOD-FAR | Total FAR |
|---------|--------|--------|---------|-----------|
| I3      | 0.8880 |   1.30 |    5.16 |      2.23 |
| I7      | 0.8757 |   1.55 |    7.50 |      2.99 |
| I10     | 0.8778 |   0.00 |    0.16 |      0.04 |
| I13     | 0.8634 |   0.00 |    0.16 |      0.04 |
```

best variant: I10 — bit_F1 0.8778, Total FAR 0.04% (1 OOD FP / 2640)

## Delta vs iter116J baseline

```
| Variant | dbit_F1 | dTotal_FAR_pp |
|---------|---------|---------------|
| I3      | -0.1047 |         +2.23 |
| I7      | -0.1170 |         +2.99 |
| I10     | -0.1149 |         +0.04 |
| I13     | -0.1293 |         +0.04 |
```

bit_F1 regresses ~0.11 across the board (seed=99 unlucky draw vs the seed=1
iter116J optimum). NI-FAR holds at 0 for the gated variants I10/I13.
OOD-FAR for I10/I13 is a single Starburst/CenterDonut FP. The non-gated
I3/I7 variants leak a non-trivial OOD-FAR (5-8%) — gating remains
essential at this seed.

## Insights / hypotheses

- Recipe is stable but seed-sensitive: seed=99 shifts bit_F1 down by 0.11
  vs seed=1 (iter116J = 0.9927). Variance estimate alone is meaningful —
  need more seeds for a confidence interval.
- I10 > I13 on bit_F1 (0.8778 vs 0.8634) at identical FAR (both 0.04%).
  The 0.0144 gap is consistent with prior FCM-PM observations that I13
  trades bit_F1 for tighter calibration; here calibration is already
  saturated (NI 0%, OOD 1/640), so I10 is the cleaner pick.
- This is the **1st of 4 seed-variance points** in chain v5; conclusions
  reserved for iter 4.
