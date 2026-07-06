# cron 6 tick — chain v6 / v7 / v8 catch-up + chain v9 GPU guard

TS: 260517 15:37 (cron 6 1h tick).  This entry catches up the directly
preceding 1 h window (cron 5 → cron 6) and back-fills the chain v6+v7
journey into the cross-iter `RESULTS_TIMELINE.md` (Sections B, C, D, E)
which had been lagging behind the per-iter docs.

## What is new on disk since cron 5 (14:37 → 15:37)

```
| Time      | Event                                                        |
|-----------|--------------------------------------------------------------|
| 13:59:57  | chain v8 phase 0 KD_v8 re-eval FAILED (OOM, cron 4-5)        |
| 14:00:01  | chain v8 phase 1 ensemble re-confirmation (I10 + I13) DONE   |
| 14:00:06  | chain v8 phase 2 iter116J_cmp015 train dispatched            |
| 15:03:58  | chain v9 phase 0 KD_v8 re-eval gated-dispatch (gpu 55 % free)|
| 15:37     | cron 6 read — KD_v8 re-eval running, no parquet yet          |
```

No new bit_F1 / Total FAR row landed on disk between cron 5 and cron 6.
The supervisor-side news is the **GPU guard** introduced in
`_run_chain_v9.sh`: every train / eval call now waits for free GPU memory
below a threshold (50 % train, 60 % eval) with a 30-120 min fallback.

## What was logged at cron 6 (this tick)

```
| File                                                          | Action   |
|---------------------------------------------------------------|----------|
| docs/chip-multilabel/RESULTS_TIMELINE.md (Section B)          | appended |
| docs/chip-multilabel/RESULTS_TIMELINE.md (Section C)          | appended |
| docs/chip-multilabel/RESULTS_TIMELINE.md (Section D)          | updated  |
| docs/chip-multilabel/RESULTS_TIMELINE.md (Section E)          | rewrote  |
| docs/chip-multilabel/02_results.md (chain v9 GPU guard)       | prepended|
| docs/chip-multilabel/paper/_diary/260517_cron6_v6_v7_summary.md| created  |
```

`iters/iter_v6_0[1-4]_*.md`, `iters/iter_v7_01_ensemble_champion.md`,
`iters/iter_v8_01_ensemble_reconfirm.md` were all written by cron 3-5 and
need no change at this tick (per-iter reports immutable, 260512 rule).

## Why touch `RESULTS_TIMELINE.md` now and not earlier

The B (ensemble) and C (KD) tables had not been updated since the chain v5
era.  They omitted:

- **E7 vote_majority_bits = 0.9941 / 0.00 %** — the new champion, recorded
  in iters/iter_v7_01 and 02_results.md but missing from the headline
  Section B step list.
- **E8 vote_union_bits = 0.9965 / 0.76 %** — Pareto extremum, same
  omission.
- **E10/E11/E12 chain v8 reconfirm rows** — I10 bit-identical to v7, plus
  the new I13 datapoint.
- **C row KD_v7_iter116J_a03_T2_skipcutmix** — the first non-collapse KD
  result (bit_F1 0.9265 / 0.00 % FAR at I10), which corrects the
  pre-existing "KD = negative result" conclusion at Section D row 5.
- **C rows KD_v8 / KD_v9 / KD_v10** — record of the chain v7-v9 retry
  attempts so the negative-results trail is auditable in one place.

Section D row 5 was rewritten from "negative result" to **"collapse fix
via `--kd-skip-on-cutmix`"** and Section D gained two new rows: ensemble
champion (D6) and Pareto extremum (D7), both with explicit
diversity-over-tuning insight (D8).

## Headline numbers reaffirmed at cron 6 (4-decimal, no new evaluation)

```
| run                                  | I10 bit_F1 | Total FAR | source                              |
|--------------------------------------|------------|-----------|-------------------------------------|
| iter116J s=1 SOTA (val_f1, ep6)      |     0.9927 |      0.00 | outputs/iter116J_g3_ls30/           |
| chain v7 vote_majority_bits          |     0.9941 |      0.00 | outputs/_ensemble_v7_5mode.json     |
| chain v7 vote_union_bits             |     0.9965 |      0.76 | outputs/_ensemble_v7_5mode.json     |
| chain v8 vote_majority_bits (re-run) |     0.9941 |      0.00 | outputs/_ensemble_v8_g_s77_kdv7_I10 |
| KD_v7 single (I10)                   |     0.9265 |      0.00 | outputs/KD_v7_iter116J_a03_T2_skipcm|
```

The headline `vote_majority_bits = 0.9941 / 0.00 %` has now held across
three independent cron cycles (cron 4 dispatch, cron 5 re-run, cron 6
docs-only).  This is the longest stable-champion streak of the chain
v5+ era.

## What cron 7 should look for

1. `outputs/KD_v8_a05_T2_skipcm/.../eval_n2000_pred/.../preds_chip.parquet`
   — phase 0 re-eval completion.  If parquet exists, compute I3/I7/I10/I13
   bit_F1 + 3-FAR and append a row to `tables/all_runs_n2000.csv`
   (chain=v9, iter=0, tag=KD_v8_a05_T2_skipcm_reeval) plus an iter doc
   `iter_v9_00_KDv8_reeval.md`.
2. `outputs/KD_v9_a02_T2_skipcm/` or `outputs/KD_v10_a03_T1_skipcm/` train
   logs — phase 1 KD retries.  Record collapse-vs-non-collapse outcome.
3. `outputs/iter116J_cmp015/` (and 020/030/035/040) — phase 2 cutmix-p
   sweep cells.  Record each as a row in the all_runs CSV.
4. `outputs/iter116J_ls030/` (and 070/100) — phase 3 complement-label-
   scale sweep cells.

## Cross-references

- per-iter detail (immutable):
  - `iters/iter_v6_0[1-4]_*.md`
  - `iters/iter_v7_01_ensemble_champion.md`
  - `iters/iter_v8_01_ensemble_reconfirm.md`
- prior diary entries:
  - `paper/_diary/260517_cron5_chain_v8_progress.md`
  - `paper/_diary/260517_chain_v6_v7_narrative_appended.md`
- supervisor logs:
  - `outputs/_chain_v6_summary.log`
  - `outputs/_chain_v7_summary.log`
  - `outputs/_chain_v8_summary.log`
  - `outputs/_chain_v9_summary.log`
- ensemble JSONs:
  - `outputs/_ensemble_v7_5mode.json` (5 modes × I10)
  - `outputs/_ensemble_v8_g_s77_kdv7_I10.json` + `_I13.json`
