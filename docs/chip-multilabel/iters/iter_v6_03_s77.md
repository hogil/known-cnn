# chain v6 iter 3 — iter116J_clone_s77 (seed=77, NEW bit_F1 micro-champ)

- TS: 260517_091330 (train) / 260517_092932 (eval n2000)
- Source: `outputs/iter116J_clone_s77/20260517_091330_T7_iter116J_clone_s77/eval_n2000_pred/stage1_260517_092932/preds_chip.parquet`
- Recipe: T7 BCE+LS=0.30 + FCM-PM CutMix g=3 corner, seed=77 (same family as iter116J seed=1)
- Baseline (iter116J g3_ls30 T7): I10 bit_F1 0.9748 / Total FAR 0.00%
- Selected ckpt: ep8 (val_margin 0.6978); train val_f1 0.9969 by ep10

## Eval n2000 (POS9 strict + 4 OOD strict)

```
| Variant | bit_F1 | NI-FAR | OOD-FAR | Total FAR |
|---------|--------|--------|---------|-----------|
| I3      | 0.9352 |  82.85 |  100.00 |     87.01 |
| I7      | 0.9344 |  82.95 |  100.00 |     87.08 |
| I10     | 0.9786 |   0.40 |    1.88 |      0.76 |
| I13     | 0.8621 |   0.05 |    1.72 |      0.45 |
```

Best variant: I10 — bit_F1 0.9786, Total FAR 0.76%

## Delta vs iter116J baseline (atomic seed delta 1 -> 77)

```
| Variant | dbit_F1 | dTotal_FAR_pp |
|---------|---------|---------------|
| I3      | +0.0078 |        -12.99 |
| I7      | +0.0081 |        -12.92 |
| I10     | +0.0038 |         +0.76 |
| I13     | -0.0943 |         +0.45 |
```

## Insights

- **Marginal new bit_F1 best on I10 (+0.0038)** — first seed in chain v5+v6 to
  exceed iter116J g3_ls30 baseline on bit_F1. But +0.76 pp Total FAR penalty.
- per_defect_F1 (I3): bank_boundary 0.997, fork 0.980, scratch 0.847, scratch_rot 0.999.
  scratch is the weak class (0.85 vs baseline ~0.99).
- I3/I7 still have catastrophic FAR (~87%) — the basic threshold gate is not enough;
  I10's max_prob entropy gate and I13's invalid_score gate are essential.
- I13 (strict gate) takes 0.94 hit on bit_F1 to drop FAR by 0.31 pp — strictness costly here.
- **NOT a clean win** vs baseline: +0.0038 bit_F1 at cost of +0.76 FAR.
  Combined: bit_F1 micro-improve does not justify FAR regression.
- Chain v5+v6 seed scan (s1, s7, s11, s23, s42, s77, s99) shows extreme seed
  variance under this recipe — only s1 (baseline) and s77 reach competitive
  I10 bit_F1 ~ 0.97.
