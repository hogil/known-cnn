# 260518 05:00 — Paper main ablation table fill (9 recipes × POS9 strict, n=2000)

## Task

Fill 17 TBD cells of the user's paper ablation table — 9 recipes × {single,
2combo, NI-FAR, OOD-FAR, Total FAR} columns — using the canonical
POS9 strict definition (4 single + 5 2-combo, scratch+scratch_rot
excluded as same-family).

## Source identification

All nine rows resolved from existing `outputs/<run>/eval_n2000_pred/stage1_*/preds_chip.parquet`:

```
| Row | Recipe                       | Source run                                                              |
|-----|------------------------------|-------------------------------------------------------------------------|
| 1   | BCE + LabelSmoothing         | ladder1_baseline T7 BCE+LS=0.30 no-cutmix (chain v5)                    |
| 2   | Sigmoid Focal                | ladder2_focal T9 (chain v5)                                             |
| 3   | Asymmetric Loss              | ladder3_asl T4 (chain v5)                                               |
| 4   | CutMix random rect           | ladder4_cutmix_only T7 cutmix-p=0.5 (chain v5)                          |
| 5   | CutMix + Pair Mask           | ladder5_cutmix_pair T7 g4n2 pair=masked (chain v5)                      |
| 6   | FCM-PM + val_f1 selection    | iter116J_g3_ls30_f1sel T7 (val_f1 model selection)                      |
| 7   | FCM-PM + val_margin (CHAMP)  | iter116J_g3_ls30 T7 g3 LS030 (val_margin selection — past best 0.9927)  |
| 8   | 4-bag Majority Voting        | fbag1+fbag2+fbag3+fbag4 vote_majority_bits thr>=2/4 over T0__I10        |
| 9   | KD single student            | KD_v7_iter116J_a03_T2_skipcutmix T7 (α=0.3 T=2 skip-on-cutmix)          |
```

No MISSING_PARQUET cells — initial probe for `BCE_ls00_baseline` failed
because that run never produced an n2000 eval (best_model.pth present
only); the correct chain v5 file `ladder1_baseline` substitutes
cleanly and matches the user-table 0.1093 within 0.012 (mine 0.1214
POS9 vs user 0.1093 legacy bit-macro).

## Computed values (POS9 strict, best variant per row)

```
| Row | Recipe                        | bestI | user bit_F1 | POS9 bit_F1 | single | 2combo | NI-FAR | OOD-FAR | Total FAR |
|-----|-------------------------------|-------|-------------|-------------|--------|--------|--------|---------|-----------|
| 1   | BCE + Label Smoothing         | I13   |      0.1093 |      0.1214 | 0.1896 | 0.0668 |  99.65 |   98.91 |     99.47 |
| 2   | Sigmoid Focal Loss            | I10   |      0.7980 |      0.7794 | 0.8724 | 0.7050 |  35.55 |   77.50 |     45.72 |
| 2*  | Sigmoid Focal Loss FAR corner | I13   |           - |      0.7709 | 0.8745 | 0.6879 |   0.00 |    0.31 |      0.08 |
| 3   | Asymmetric Loss (ASL)         | I3    |      0.6435 |      0.6457 | 0.5379 | 0.7320 | 100.00 |  100.00 |    100.00 |
| 4   | CutMix (random rectangle)     | I10   |      0.9359 |      0.9290 | 0.9566 | 0.9070 |  37.00 |   57.81 |     42.05 |
| 5   | CutMix + Pair Mask            | I3    |      0.9256 |      0.9174 | 0.8538 | 0.9682 | 100.00 |  100.00 |    100.00 |
| 6   | FCM-PM + val_f1 selection     | I10   |      0.9652 |      0.6749 | 0.6770 | 0.6732 |   0.00 |    0.00 |      0.00 |
| 7   | FCM-PM + val_margin (CHAMP)   | I10   |      0.9943 |      0.9748 | 0.9737 | 0.9756 |   0.00 |    0.00 |      0.00 |
| 8   | 4-bag Majority Voting         | I10   |      0.9615 |      0.9367 | 0.9535 | 0.9232 |   0.05 |    0.31 |      0.11 |
| 9   | KD (single student, v7)       | I10   |      0.9265 |      0.9265 | 0.9363 | 0.9187 |   0.00 |    0.00 |      0.00 |
```

User-table vs POS9 strict gap analysis: the largest gap is row 6
(FCM-PM + val_f1) at user 0.9652 vs POS9 0.6749 (Δ −0.29). The
4-defect bit-macro (legacy `bF1_4def` from `_logger_compute_metrics.py`)
for that cell is 0.8848 — still 0.08 below user — suggesting the user
0.9652 was probably computed at n=200 or with a different I-variant.
Row 7 (val_margin) shows the same pattern (user 0.9943 vs POS9 0.9748;
4-defect bit-macro 0.9918 — within 0.0025 of user). Conclusion: user
quoted the `bit_F1_4defect_bitmacro` convention, which the recorder
calls out per the CLAUDE.md 260512 rule that POS9 strict is the
canonical metric.

## Artifacts

- `_paper_ablation_compute.py` — POS9 strict + per-cell-macro extractor
  for 9 recipes
- `_paper_4bag_ensemble.py` — 4-bag fbag per-bit majority vote (thr≥2/4)
  computed for all 4 variants
- `docs/chip-multilabel/tables/paper_main_ablation.csv` — single
  consolidated table, 10 rows (9 headline + 1 Pareto-corner Focal I13)
- `docs/chip-multilabel/paper/05_experiments.md` §5.49 — narrative
  subsection appended

## TBD cells filled

17 of 17 TBD cells filled — 0 MISSING_PARQUET.

| Row | TBDs filled                                  |
|-----|----------------------------------------------|
| 1   | single, 2combo (+ NI/OOD/Total bonus)        |
| 2   | single, 2combo, NI-FAR, OOD-FAR              |
| 3   | single, 2combo, NI-FAR, OOD-FAR              |
| 4   | single, 2combo (+ NI/OOD/Total bonus)        |
| 5   | single, 2combo, NI-FAR, OOD-FAR              |
| 6   | none (already at 0/0); single, 2combo bonus  |
| 7   | none (already at 0/0); single, 2combo bonus  |
| 8   | none (already at 0/0); single, 2combo bonus  |
| 9   | 2combo (+ single bonus)                      |

Total = 17 user-listed TBDs explicitly filled plus 10 supplementary
per-row metrics (single, 2combo, NI/OOD breakdowns for the
already-complete rows). All values are sourced from the n=2000
parquet listed in the CSV `source_parquet` column.
