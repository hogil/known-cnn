# chain v6 iter 2 — iter116J_clone_s23 (seed=23, val_margin_max collapsed)

- TS: 260517_085059 (train) / 260517_090654 (eval n2000)
- Source: `outputs/iter116J_clone_s23/20260517_085059_T7_iter116J_clone_s23/eval_n2000_pred/stage1_260517_090654/preds_chip.parquet`
- Recipe: T7 BCE+LS=0.30 + FCM-PM CutMix g=3 corner, seed=23 (same family as iter116J seed=1)
- Baseline (iter116J g3_ls30 T7): I10 bit_F1 0.9748 / Total FAR 0.00%
- Selected ckpt: ep9 (val_margin 0.6954); train val_f1 reached 0.9969 by ep10

## Eval n2000 (POS9 strict + 4 OOD strict)

```
| Variant | bit_F1 | NI-FAR | OOD-FAR | Total FAR |
|---------|--------|--------|---------|-----------|
| I3      | 0.5592 | 100.00 |  100.00 |    100.00 |
| I7      | 0.5069 | 100.00 |  100.00 |    100.00 |
| I10     | 0.4738 |  63.20 |   76.56 |     66.44 |
| I13     | 0.4311 |  62.40 |   76.25 |     65.76 |
```

Best variant: I10 — bit_F1 0.4738, Total FAR 66.44%

## Delta vs iter116J baseline (atomic seed delta 1 -> 23)

```
| Variant | dbit_F1 | dTotal_FAR_pp |
|---------|---------|---------------|
| I3      | -0.3682 |          0.00 |
| I7      | -0.4194 |          0.00 |
| I10     | -0.5010 |        +66.44 |
| I13     | -0.5253 |        +65.76 |
```

## Insights

- **Catastrophic seed variance**: even with ep9 ckpt (later than s11's ep1), the model's
  positive-class calibration is far from baseline. bit_F1 0.4738 means roughly half the
  defects in 9 POS keys are missed/mislabeled.
- per_defect_F1 breakdown (I3): bank_boundary 0.988, fork 0.613, scratch 0.796, scratch_rot 0.919.
  → fork is weak link (-0.36 vs baseline 0.97). Indicates seed 23 trained an under-confident fork head.
- val_f1 trajectory misled us: train val_f1 reached 0.9969 (ep10), but val_margin peaked
  at ep9 (0.6954), and the actual generalization to synthetic n2000 eval is much worse.
- The same recipe family with different seeds (s11=70% FAR, s23=66% FAR, s77=0.76% FAR)
  spans Total FAR by ~70 pp — extreme seed sensitivity at this recipe.
