# cron 5 tick — chain v8 progress snapshot

TS: 260517 cron5 (1h tick #5).  chain v8 supervisor is autonomous-dispatching
on its own; this entry only **records** what is verifiable on disk, no new
train/eval is launched from this tick.

## Phase 0 — KD_v8 re-eval

- Status: **FAILED (OOM)** at cron 4.  Already recorded in
  `260517_cron4_kd_sweep_finale.md`.  No new artefact this tick.
- Source: `outputs/_KD_v8_a05_T2_skipcm_reeval_n2000.log`,
  `outputs/_chain_v8_summary.log` line 2 (`RE_EVAL KD_v8_a05_T2_skipcm FAILED`).
- KD_v9 / KD_v10 had no checkpoint to re-eval (skipped — same source line 3-4).

## Phase 1 — ensemble re-confirmation (DONE)

- Supervisor ran both I10 and I13 ensemble at 13:59:57 + 14:00:01 (see
  `outputs/_chain_v8_summary.log` lines 5-6).
- I10 vote_majority_bits = **0.9941 / 0.00% Total FAR** — bit-identical to
  chain v7 phase 1 champion.  Champion stable.
- I13 vote_majority_bits = **0.9600 / 0.00%** — new datapoint, below I10
  (gate collapses fork-scratch combos).
- I13 vote_union_bits = **0.9923 / 0.49%** — moderate Pareto.
- Recorded as `iters/iter_v8_01_ensemble_reconfirm.md`; CSV append at
  `tables/all_runs_n2000.csv` (10 new rows: 5 modes × 2 variants).

## Phase 2 — cutmix-p sweep (TRAINING, no eval yet)

- Supervisor kicked off `iter116J_cmp015` at 14:00:06 (T7 LS=0.5 cutmix p=0.15
  seed=42).  RunDir: `outputs/iter116J_cmp015/20260517_140019_T7_iter116J_cmp015/`.
- Train log `outputs/_iter116J_cmp015_train.log` is **0 bytes** at cron 5 read
  time — process either still spinning up or already redirected logs to
  RunDir.  No metric.
- Sibling sweep cells (cmp020 / cmp030 / cmp035 / cmp040) have **no dir on
  disk yet** — supervisor queues them sequentially behind cmp015 train+eval.
- No iter doc / CSV row generated this tick for phase 2.

## Phase 3 — label-scale sweep ls030 / ls070 / ls100 (NOT STARTED)

- No dir on disk under `outputs/iter116J_ls*`.  Queued behind phase 2 in
  supervisor script.  No artefact.  No iter doc / CSV row this tick.

## Cron 5 docs touched

```
| Path                                                       | Action   |
|------------------------------------------------------------|----------|
| docs/chip-multilabel/iters/iter_v8_01_ensemble_reconfirm.md| created  |
| docs/chip-multilabel/02_results.md                         | appended |
| docs/chip-multilabel/tables/all_runs_n2000.csv             | appended |
| docs/chip-multilabel/paper/_diary/260517_cron5_chain_v8_progress.md | created |
```

## Open

- cron 6 (1h later) should look for `outputs/iter116J_cmp015/.../eval_n2000_pred/`
  parquet — if present, extract I3/I7/I10/I13 metrics into a new
  `iter_v8_02_cmp015.md` and update `02_results.md` + CSV.
- If phase 2 sweep cells (cmp020+) appear, batch-record under
  `iter_v8_03_cmpp_sweep.md`.
- Phase 3 ls sweep likely only visible at cron 7 or later.
