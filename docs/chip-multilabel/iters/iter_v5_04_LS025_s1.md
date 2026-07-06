# chain v5 iter 4 — iter50_clone_LS025_s1_v4

- TS: 260516_113254 (train) / 260516_122822 (eval n2000)
- Source: `outputs/iter50_clone_LS025_s1_v4/20260516_113254_T7_iter50_clone_LS025_s1_v4/eval_n2000_pred/stage1_260516_122822/preds_chip.parquet`
- Recipe: T7 BCE+LS=**0.25** + FCM-PM CutMix g=3 corner, seed=1
  (vs iter1-3 LS=0.30, vs iter116J LS=0.30 seed=1 — same seed, atomic LS delta)
- Baseline (iter116J): bit_F1 0.9927 / Total FAR 0.00% (T7 LS=0.30 g=3 seed=1)

## Eval n2000 (POS9 strict + 4 OOD strict)

```
| Variant | bit_F1 | NI-FAR | OOD-FAR | Total FAR |
|---------|--------|--------|---------|-----------|
| I3      | 0.8697 |  69.95 |   80.62 |     72.54 |
| I7      | 0.8695 |  69.80 |   80.78 |     72.46 |
| I10     | 0.9121 |   0.05 |    1.56 |      0.42 |
| I13     | 0.8839 |   0.05 |    0.94 |      0.27 |
```

best variant: I10 — bit_F1 0.9121, Total FAR 0.42% (11 FP / 2640)

## Delta vs iter116J baseline (same seed=1, atomic LS delta 0.30→0.25)

```
| Variant | dbit_F1 | dTotal_FAR_pp |
|---------|---------|---------------|
| I3      | -0.1230 |        +72.54 |
| I7      | -0.1232 |        +72.46 |
| I10     | -0.0806 |         +0.42 |
| I13     | -0.1088 |         +0.27 |
```

## Insights / hypotheses

- **LS=0.25 is strictly worse than LS=0.30 at seed=1**: -0.08 bit_F1 at
  the best (I10) cell, +0.42pp Total FAR. The chain v5 hypothesis "lower
  LS → tighter calibration" is rejected for this recipe family.
- I3/I7 collapse to 70%+ FAR is **even worse than seed=7 (iter3)**, but
  notably I10/I13 hold at <0.5% FAR — the gate works at seed=1 in a way
  it did not at seed=7. This suggests **the gate failure in iter3 is
  seed-specific**, not LS-specific.
- LS=0.25 reduces the spread between training-time positive/negative
  logits → I3/I7 (no gating) become much more vulnerable to OOD bleed,
  but the I10/I13 entropy/distance gate has enough margin to compensate.
  The LS↓ regression manifests primarily as **bit_F1 loss**, not FAR.
- This is a clean atomic LS ablation (seed identical to iter116J), and
  it confirms **LS=0.30 is at or near the local optimum** for this
  recipe — lowering by 0.05 already costs 0.08 bit_F1 at the best cell.

## Combined chain v5 verdict (iter 1-4)

```
| iter | seed | LS   | best variant | bit_F1 | Total FAR | comment           |
|------|------|------|--------------|--------|-----------|-------------------|
| 1    |  99  | 0.30 | I10          | 0.8778 |      0.04 | gated robust      |
| 2    |  42  | 0.30 | I10          | 0.9583 |      0.30 | best seed of 3    |
| 3    |   7  | 0.30 | I10          | 0.8787 |     18.22 | gate failure      |
| 4    |   1  | 0.25 | I10          | 0.9121 |      0.42 | LS-0.05 regress   |
| 116J |   1  | 0.30 | I13          | 0.9927 |      0.00 | recorded SOTA     |
```

- 3-seed (LS=0.30) bit_F1 spread at I10: [0.88, 0.96, 0.88], mean 0.905,
  std 0.046. **iter116J's 0.9927 is +1.9σ above mean — almost certainly
  a positive seed outlier.**
- iter116J reported Total FAR 0.00% is also a tail event: the 3-seed
  mean Total FAR at I10 is (0.04 + 0.30 + 18.22) / 3 = 6.19% — driven
  entirely by iter3.
- **Next-iter recommendation**: kill the single-seed reporting practice
  for this recipe family. Either (a) report 3-seed mean + std as the
  headline, or (b) add EMA / warmup to stabilise seed=7-class failures
  before quoting the recipe as SOTA.
