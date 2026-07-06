# 260517 — ensemble champion dispatch (chain v7 first headline)

**TS:** 2026-05-17 (post-hoc aggregation, no new training)

**Trigger:** User-issued parallel dispatch to record the first iter116J
SOTA replacement.  Post-hoc 5-mode bit/label vote aggregation across the
three chain v6 students (iter116J s=1 val_f1 ep6, iter116J_clone_s77
margin_max ep8, KD_v7_iter116J_a03_T2_skipcutmix --kd-skip-on-cutmix
ep7) was scored on the same n2000 POS9 strict + 4 OOD strict eval set at
the I10 inference variant.

**Source aggregate:** `outputs/_ensemble_v7_5mode.json`.

**Headline (paper cell):**

```
| Mode                   | bit_F1 | Total FAR | dbit_F1 vs SOTA | dFAR  | status            |
|------------------------|--------|-----------|-----------------|-------|-------------------|
| vote_majority_bits     | 0.9941 |      0.00 |         +0.0014 |  0.00 | NEW CHAMPION      |
| vote_majority          | 0.9936 |      0.00 |         +0.0009 |  0.00 | tie at FAR safe   |
| vote_union_bits        | 0.9965 |      0.76 |         +0.0038 | +0.76 | peak F1 FAR trap  |
| vote_intersection_bits | 0.9735 |      0.00 |         -0.0192 |  0.00 | too conservative  |
| vote_unanimous         | 0.9495 |      0.00 |         -0.0432 |  0.00 | too strict        |
| iter116J s=1 (single)  | 0.9927 |      0.00 |          0.0000 |  0.00 | prior SOTA        |
```

**Key calls:**

- New paper headline cell = `vote_majority_bits` at **bit_F1 0.9941 /
  Total FAR 0.00%**.  This is the first non-FAR-paying improvement over
  iter116J across chain v5+v6+v7 (9-day SOTA broken).
- `vote_union_bits` reported as Pareto extremum, not headline.
- Residual gaps both scratch-combo (BB+scratch 0.9791, fork+scratch
  0.9824) — same scratch-head weakness chain v6 already isolated.

**Docs written/updated this dispatch:**

- New: `iters/iter_v7_01_ensemble_champion.md` — 5-mode table, per-class
  F1 on champion cell, delta vs SOTA, insights, lessons.
- Prepended: `02_results.md` — new section `## 2026-05-17 ensemble
  champion — chain v7 (vote_majority_bits beats iter116J SOTA)` placed
  immediately above the chain v6 4-phase summary block.
- Appended: `tables/all_runs_n2000.csv` — 5 rows under `chain=v7,
  iter=1`, one per aggregation mode (all sourced to the JSON aggregate).
- Appended: `paper/05_experiments.md` — standalone subsection
  `### Ensemble champion contribution` (paper-grade narrative,
  ~600 words) after the existing chain v6+v7 progression narrative.
- New: this diary entry.

**No training, no eval dispatch, no outputs/ touched.**  All five
docs are pure read of the pre-existing aggregate JSON.

**Next-iter recommendation (recorded for chain v7 iter 2):**

1. Extend ensemble to 4 or 5 students once additional KD-multi-teacher
   members exist — per-bit majority typically improves monotonically
   with odd N.
2. Per-class confidence-weighted bit aggregation on the scratch-combo
   cells — the remaining 0.97-0.98 → 0.99+ gap.
3. Maintain `vote_majority_bits` as default reporting cell for any
   single-recipe seed sweep going forward.
