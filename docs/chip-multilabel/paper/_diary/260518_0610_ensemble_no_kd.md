# 260518 06:10 — Base-only ensemble (no KD) measurement

**Cron**: #49 06:10
**Recorder**: chip-multilabel-paper-recorder
**Source**: `outputs/_ensemble_no_kd_s1_s77_s33_I10.json`
**Directive**: user 06:00 — "ensemble 에 KD 빼야지. 흐름 = 학습 → KD → ensemble → 최종 KD"

## Members (3-way, all base-trained, no KD)

- `iter116J_s1`  (seed=1,  T7 LS=0.30 g=3, FCM-PM)
- `iter116J_s77` (seed=77, T7 LS=0.30 g=3, FCM-PM)
- `iter116J_s33_v15` (seed=33, T7 LS=0.30 g=3, FCM-PM, fresh v15 data)

## Results at I10 (n=2000 POS9 strict, OOD=4 strict)

```
| Mode                   | bit_F1 | NI-FAR | OOD-FAR | Total FAR |
|------------------------|--------|--------|---------|-----------|
| vote_majority          | 0.9928 |   0.35 |    0.00 |      0.27 |
| vote_unanimous         | 0.9375 |   0.00 |    0.00 |      0.00 |
| vote_intersection_bits | 0.9701 |   0.00 |    0.00 |      0.00 |
| vote_union_bits        | 0.9880 |  20.05 |    2.97 |     15.91 |
| vote_majority_bits     | 0.9929 |   0.35 |    0.00 |      0.27 |
```

## Champion split (paper-facing)

- **Base ensemble (no KD) champion** = `vote_majority_bits` **0.9929 / 0.27 %**
  (paper main table row 10, new).
- **KD-mixed ensemble champion** (old E7, {s1 + s77 + KD_v7}) = **0.9941 / 0.00 %**
  (paper main table row 9-ensemble, kept).

Delta = +0.0012 bit_F1 / −0.27 pp Total FAR attributable to KD student
inclusion at the ensemble stage. The paper now reports the two
separately so the reader can attribute the lift to the KD step rather
than to "ensembling in general".

## Files touched

- `docs/chip-multilabel/RESULTS_TIMELINE.md` — B-table rows E15-E19 + insight paragraph
- `docs/chip-multilabel/tables/paper_main_ablation.csv` — 5 rows (row id 10, all 5 modes)
- `docs/chip-multilabel/paper/05_experiments.md` — §5.49.2 subsection (base-only ensemble headline)
- `docs/chip-multilabel/paper/_diary/260518_0610_ensemble_no_kd.md` — this file

## Next

- Chain v16 Phase 3 (final KD distilled against base-ensemble majority targets) will test whether
  the −0.27 pp FAR is recoverable while preserving 0.9929 bit_F1 — closing the "final KD" stage.
- s55_v15 (chain v15 final slot) will define whether 4-way base ensemble {s1+s77+s33+s55} adds
  another independent vote axis or saturates.
