# chain v6 iter 4 — KD_v7_iter116J_a03_T2_skipcutmix (KD skip-on-cutmix prevents collapse)

- TS: 260517_095713 (train) / 260517_101336 (eval n2000)
- Source: `outputs/KD_v7_iter116J_a03_T2_skipcutmix/20260517_095713_T7_KD_v7_iter116J_a03_T2_skipcutmix/eval_n2000_pred/stage1_260517_101336/preds_chip.parquet`
- Recipe: T7 BCE+LS=0.30 + FCM-PM CutMix g=3 corner + KD (teacher=iter116J g3_ls30 single,
  alpha=0.3, T=2, --kd-skip-on-cutmix), seed=1
- Teacher: `outputs/iter116J_g3_ls30/T7_iter116J_g3_ls30_260513_010015` (1 member, 2015 chips, 31s prob gen)
- Baseline (teacher = iter116J g3_ls30 T7): I10 bit_F1 0.9748 / Total FAR 0.00%
- Prior KD attempts (6 KD runs without skip-on-cutmix): ALL collapsed (bit_F1 < 0.5 or NaN)
- This iter's KEY HYPOTHESIS: KD loss term + complement CutMix patches conflict
  (teacher prob computed on clean chip, student sees mixed chip → KL mismatch).
  --kd-skip-on-cutmix disables KD loss on the 25% cutmix-active batches.
- Selected ckpt: ep7 (val_margin 0.6962); train val_f1 0.9969 by ep10

## Eval n2000 (POS9 strict + 4 OOD strict)

```
| Variant | bit_F1 | NI-FAR | OOD-FAR | Total FAR |
|---------|--------|--------|---------|-----------|
| I3      | 0.9085 |   4.00 |    2.97 |      3.75 |
| I7      | 0.9114 |  20.60 |   19.84 |     20.42 |
| I10     | 0.9265 |   0.00 |    0.00 |      0.00 |
| I13     | 0.8899 |   0.00 |    0.16 |      0.04 |
```

Best variant: I10 — bit_F1 0.9265, Total FAR 0.00%

## Delta vs iter116J baseline (KD-applied vs no-KD same recipe)

```
| Variant | dbit_F1 | dTotal_FAR_pp |
|---------|---------|---------------|
| I3      | -0.0189 |        -96.25 |
| I7      | -0.0149 |        -79.58 |
| I10     | -0.0483 |          0.00 |
| I13     | -0.0665 |         +0.04 |
```

## Insights

- **KD did NOT collapse** — major breakthrough vs prior 6 KD attempts.
  The --kd-skip-on-cutmix flag (skip KD loss on the 25% batches where CutMix-complement
  is active) prevented the teacher-student gradient conflict that previously caused collapse.
- **But KD did NOT exceed baseline either.** bit_F1 0.9265 < 0.9748 (-0.048).
  Single-teacher KD seems to act as a regularizer (lowers variance, lowers ceiling).
- I3/I7 bit_F1 ~ 0.91 with FAR 4-20% — KD significantly improves I3 over no-KD seeds
  (s11 I3=0.86 FAR 100%, s23 I3=0.56 FAR 100%) — KD calibrates the threshold gate even
  without the I10 entropy gate. This is interesting: KD makes simpler variants viable.
- Maintains 0% Total FAR on I10 same as baseline.
- scratch class still the weak point: I10 per_defect_F1 = bb 0.997, fork 0.987, scratch 0.93,
  scratch_rot 1.000. KD did not specifically help scratch.
- **Hypothesis: multi-teacher KD (bag of 3-5 teachers) might add enough diversity to
  beat single baseline.** Single-teacher KD ceiling = teacher bit_F1.

## Lessons for next iter

1. --kd-skip-on-cutmix is a keeper flag — solves the collapse problem.
2. Try multi-teacher KD: aggregate probs from s77 (0.978) + iter116J (0.975) + others.
3. KD alpha=0.3 + T=2 is OK but maybe alpha=0.1 (more student) + T=1 (sharper teacher)
   could let student exceed teacher.
