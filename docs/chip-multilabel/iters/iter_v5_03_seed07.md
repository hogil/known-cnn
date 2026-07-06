# chain v5 iter 3 — iter50_clone_seed07_v4

- TS: 260516_103926 (train) / 260516_121239 (eval n2000)
- Source: `outputs/iter50_clone_seed07_v4/20260516_103926_T7_iter50_clone_seed07_v4/eval_n2000_pred/stage1_260516_121239/preds_chip.parquet`
- Recipe: T7 BCE+LS=0.30 + FCM-PM CutMix g=3 corner, seed=7 (vs iter1 seed=99, iter2 seed=42)
- Baseline (iter116J): bit_F1 0.9927 / Total FAR 0.00%

## Eval n2000 (POS9 strict + 4 OOD strict)

```
| Variant | bit_F1 | NI-FAR | OOD-FAR | Total FAR |
|---------|--------|--------|---------|-----------|
| I3      | 0.8757 | 100.00 |  100.00 |    100.00 |
| I7      | 0.8633 | 100.00 |  100.00 |    100.00 |
| I10     | 0.8787 |  19.40 |   14.53 |     18.22 |
| I13     | 0.7864 |  23.65 |   19.06 |     22.54 |
```

best variant: I10 — bit_F1 0.8787, Total FAR 18.22% — **catastrophic
gate failure**. I3/I7 are full collapse (every Normal/Invalid/OOD chip
mis-predicted as a defect class).

## Delta vs iter116J baseline

```
| Variant | dbit_F1 | dTotal_FAR_pp |
|---------|---------|---------------|
| I3      | -0.1170 |        +100.0 |
| I7      | -0.1294 |        +100.0 |
| I10     | -0.1140 |        +18.22 |
| I13     | -0.2063 |        +22.54 |
```

## Insights / hypotheses

- **Seed=7 is a complete training failure** — collapse pattern (all
  negatives predicted as defects in I3/I7) indicates either (a) the
  threshold-free cells lock onto a degenerate logit floor, or (b) early
  CutMix coverage saturated training before the model learned the
  negative manifold.
- I10/I13 gate partially rescues bit_F1 (0.88 vs full collapse 0.87) but
  FAR is 18-23% — **gate provides ~80% rescue but not full**. iter1 and
  iter2 had I10/I13 FAR ≤ 0.30%; iter3 jumps 60× higher. This is
  evidence that the gate is **not unconditionally robust** — it depends
  on the underlying logit distribution being well-separated.
- Combined with iter1 (bit_F1 0.88) and iter2 (0.96), the 3-seed bit_F1
  spread is now [0.88, 0.88, 0.96] at I10 — std ≈ 0.046, mean ≈ 0.905.
  iter116J's 0.9927 is **>2σ above the seed mean** — likely a positive
  outlier, not the median expectation of the recipe.
- This **kills the chain v5 hypothesis** that the iter116J recipe is
  seed-robust. The recipe needs either (a) a wider seed sweep at every
  iter, or (b) a fundamentally different stabilisation (longer warmup,
  smaller initial LR, EMA).
