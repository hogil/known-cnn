# chain v8 iter 1 — ensemble re-confirmation (champion stable, I13 axis added)

- TS: 260517 14:00:01 (post-hoc 5-mode ensemble eval, no new training)
- Source aggregates (NEW for chain v8):
  - I10 variant: `outputs/_ensemble_v8_g_s77_kdv7_I10.json`
  - I13 variant: `outputs/_ensemble_v8_g_s77_kdv7_I13.json`
- Pool (same 3 students as chain v7 phase 1, all on n2000 POS9 strict + 4 OOD strict):
  - `outputs/iter116J_g3_ls30/` (seed=1, val_f1 ckpt)
  - `outputs/iter116J_clone_s77/` (seed=77, margin_max ckpt)
  - `outputs/KD_v7_iter116J_a03_T2_skipcutmix/` (seed=1, KD alpha=0.3 T=2)
- Baseline to beat (chain v7 headline): vote_majority_bits I10 = **bit_F1 0.9941 / Total FAR 0.00%**.

## Hypothesis

Chain v7 phase 1 only ran 5 vote modes at the I10 cell.  Chain v8 supervisor
re-ran the same 3-student pool but added the I13 inference variant to test
whether the strict-decision cell (I13: max-prob + dist-band gate) closes the
remaining hard-combo gap (`bank_boundary+scratch` 0.9791, `fork+scratch` 0.9824)
or instead inherits the I10 ranking with worse absolute numbers.

## Eval n2000 (POS9 strict + 4 OOD strict, I10 + I13)

```
| Variant | Mode                   | bit_F1 | NI-FAR | OOD-FAR | Total FAR | dbit_F1 vs v7 | comment             |
|---------|------------------------|--------|--------|---------|-----------|---------------|---------------------|
| I10     | vote_majority_bits     | 0.9941 |   0.00 |    0.00 |      0.00 |       +0.0000 | champion (re-conf)  |
| I10     | vote_majority          | 0.9936 |   0.00 |    0.00 |      0.00 |       +0.0000 | re-conf             |
| I10     | vote_union_bits        | 0.9965 |   0.40 |    1.88 |      0.76 |       +0.0000 | re-conf (FAR trap)  |
| I10     | vote_intersection_bits | 0.9735 |   0.00 |    0.00 |      0.00 |       +0.0000 | re-conf             |
| I10     | vote_unanimous         | 0.9495 |   0.00 |    0.00 |      0.00 |       +0.0000 | re-conf             |
| I13     | vote_majority_bits     | 0.9600 |   0.00 |    0.00 |      0.00 |        new    | I13 axis below I10  |
| I13     | vote_majority          | 0.9595 |   0.00 |    0.00 |      0.00 |        new    | I13 axis below I10  |
| I13     | vote_union_bits        | 0.9923 |   0.05 |    1.88 |      0.49 |        new    | I13 union closes gap|
| I13     | vote_intersection_bits | 0.8240 |   0.00 |    0.00 |      0.00 |        new    | I13 strict collapse |
| I13     | vote_unanimous         | 0.7973 |   0.00 |    0.00 |      0.00 |        new    | I13 worst           |
```

Per-class F1 on the v8 I10 champion `vote_majority_bits` (identical to v7):

```
| Class                     | F1     |
|---------------------------|--------|
| bank_boundary             | 1.0000 |
| fork                      | 1.0000 |
| scratch                   | 1.0000 |
| scratch_rot               | 1.0000 |
| bank_boundary+fork        | 0.9937 |
| bank_boundary+scratch     | 0.9791 |
| bank_boundary+scratch_rot | 0.9969 |
| fork+scratch              | 0.9824 |
| fork+scratch_rot          | 0.9945 |
```

I13 `vote_majority_bits` per-class — note the fork-scratch combo regression:

```
| Class                     | F1     |
|---------------------------|--------|
| bank_boundary             | 1.0000 |
| fork                      | 1.0000 |
| scratch                   | 1.0000 |
| scratch_rot               | 1.0000 |
| bank_boundary+fork        | 0.9931 |
| bank_boundary+scratch     | 0.9767 |
| bank_boundary+scratch_rot | 0.9913 |
| fork+scratch              | 0.8471 |
| fork+scratch_rot          | 0.8323 |
```

## Hyperparameter / variant changes vs prior iter

- No new training (same 3-student pool as chain v7).
- Only **inference-variant axis added** (I13 in addition to I10).

```
| Aspect           | chain v7 phase 1     | chain v8 iter 1      |
|------------------|----------------------|----------------------|
| pool size        | 3 students           | 3 students (same)    |
| vote modes       | 5                    | 5 (same)             |
| inference cells  | I10 only             | I10 + I13            |
| total cells      | 5                    | 10                   |
```

## Delta vs iter116J past SOTA (0.9927 / 0.00%) and chain v7 (0.9941 / 0.00%)

- v8 I10 vote_majority_bits **re-confirms 0.9941 / 0.00% bit-exact** (no
  numerical drift from v7 phase 1 — pool members on disk are identical and
  the aggregator is deterministic).  Champion stable.
- v8 I13 vote_majority_bits = **0.9600 / 0.00%**, well below the I10 cell and
  below the iter116J SOTA (0.9927).  I13 collapses `fork+scratch` (0.8471)
  and `fork+scratch_rot` (0.8323) — strict decision rule loses the second
  bit on these combos because per-class max-prob falls below the gate.
- v8 I13 vote_union_bits = **0.9923 / 0.49%** — recovers most of the I13 gap
  (BB+scratch 0.9913 vs 0.9767 in majority) but with FAR penalty similar to
  I10 union mode.  Not a headline candidate.

## Insights

1. **Champion is numerically stable across cron ticks.**  v7 phase 1 (cron 3)
   and v8 iter 1 (cron 5) produce bit-identical 0.9941 from the same pool —
   confirms the aggregator is reproducible and no pool ckpt has drifted.
2. **I13 axis is uniformly worse than I10 for ensemble.**  Across all 5 vote
   modes, I13 bit_F1 trails I10 by 0.0040 (majority_bits), 0.0042 (union),
   0.1495 (intersection), 0.1522 (unanimous).  The cost is concentrated in
   the fork-scratch combo cells where I13's stricter gate suppresses the
   second positive bit.  I10 (soft top-2) remains the correct inference
   variant for ensemble headline reporting.
3. **Strict mode collapse worsens at I13.**  I10 unanimous = 0.9495, I13
   unanimous = 0.7973 (-0.15).  I13 intersection = 0.8240, I10 intersection
   = 0.9735 (-0.15).  Strict bit aggregation × strict decision rule is
   doubly punishing on fork-scratch combos.
4. **I13 union is a publishable mid-Pareto point.**  0.9923 bit_F1 at 0.49 pp
   Total FAR sits between I10 majority_bits (0.9941 / 0.00) and I10 union
   (0.9965 / 0.76) — useful as a "moderate FAR budget" reference for
   pipelines that cannot tolerate the I10 union trade-off but want above-
   majority bit_F1.

## Lessons for next iter (and phase 2/3 pending)

1. Champion 0.9941 / 0.00% is the chain v8 headline.  No re-train needed.
2. The v8 supervisor kicked off `iter116J_cmp015` (T7 LS=0.5 cutmix p=0.15
   seed=42, RunDir `outputs/iter116J_cmp015/20260517_140019_T7_iter116J_cmp015/`)
   at 14:00:06 — phase 2 (cutmix-p sweep) is **training, not yet
   evaluable**.  No metric to record.
3. Phase 3 (label-scale sweep ls030/ls070/ls100) is **queued behind phase 2**.
   No outputs yet.
4. KD_v8 (alpha=0.5 T=2 skip-cm) re-eval at cron 4 was an OOM-fail; KD_v9
   and KD_v10 had no checkpoint to re-eval.  Already recorded in
   `paper/_diary/260517_cron4_kd_sweep_finale.md`.

## Source paths

- v8 I10 5-mode aggregate JSON: `outputs/_ensemble_v8_g_s77_kdv7_I10.json`
- v8 I13 5-mode aggregate JSON: `outputs/_ensemble_v8_g_s77_kdv7_I13.json`
- supervisor timeline: `outputs/_chain_v8_summary.log`
- Pool members (identical to chain v7 phase 1):
  - `outputs/iter116J_g3_ls30/T7_iter116J_g3_ls30_260513_010015/.../preds_chip.parquet`
  - `outputs/iter116J_clone_s77/20260517_*_T7_iter116J_clone_s77/eval_n2000_pred/stage1_*/preds_chip.parquet`
  - `outputs/KD_v7_iter116J_a03_T2_skipcutmix/20260517_095713_T7_KD_v7_iter116J_a03_T2_skipcutmix/eval_n2000_pred/stage1_260517_101336/preds_chip.parquet`
