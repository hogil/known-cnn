# 260517 cron 9 — no new metric this cycle

**Cycle TS**: 2026-05-17 18:18:11 KST
**Recorder**: chip-multilabel-paper-recorder
**Trigger**: 5-min infinite loop (cron tick 9, ~3h+ session)

## Scan result

```
| Check                                          | Found | Latest mtime              | Note                                |
|------------------------------------------------|-------|---------------------------|-------------------------------------|
| outputs/**/preds_chip.parquet newer than KD_v8 | 0     | 2026-05-17 15:04 (KD_v8)  | KD_v8 already recorded in cron 7    |
| outputs/KD_v9_a02_T2_skipcm/**/best_model.pth  | 1     | 2026-05-17 17:24:31       | 53 min stale (no further progress)  |
| outputs/_KD_v9_a02_T2_skipcm_train.log         | 1     | 2026-05-17 17:05 (0 bytes)| train log empty since dispatch      |
| outputs/_chain_v9b_runner.log                  | 1     | 2026-05-17 17:05 (0 bytes)| runner silent since dispatch        |
```

## Resource-guard tail (last 3 polls, every 3 min)

```
| Tick  | Time      | gpu_mem | util | used      | chip_pids | status |
|-------|-----------|---------|------|-----------|-----------|--------|
| 118   | 18:09:54  | 91.7%   | 100% | 15026 MiB | 0         | ABORT  |
| 119   | 18:12:54  | 91.7%   | 100% | 15026 MiB | 0         | ABORT  |
| 120   | 18:15:54  | 91.6%   | 100% | 15001 MiB | 0         | ABORT  |
```

GPU at 91% from external (non-chip) process — 15 GB occupied.  Resource-guard is
stuck in continuous ABORT but has nothing to kill (chip_pids=0); KD_v9 train
process already exited or never wrote to its log.

## Diagnosis

KD_v9 train (alpha=0.2 T=2 skip-on-cutmix, seed=1) dispatched 17:05:06 according
to chain v9b summary.  Wrote best_model at 17:24:31 (epoch checkpoint or initial
warm save) — then nothing.  Train log + runner log both 0 bytes => process was
killed early (likely OOM during epoch 2 forward pass when external GPU spike hit
91%; cuda OOM raises before Python can print to a redirected stderr).

This matches the chain v7 failure pattern for KD_v9/v10 (OOM at α=0.2 / T=1
corners).  The chain v9b retry under the GPU-gated supervisor still fails
because the gate fires after dispatch, not before each batch.

## Champion table (no change this cron)

```
| Tier                | Model                                              | bestI | bit_F1 | NI-FAR | OOD-FAR | Total FAR |
|---------------------|----------------------------------------------------|-------|--------|--------|---------|-----------|
| Single SOTA         | iter116J_g3_ls30 (s=1)                             | I10   | 0.9927 |   0.00 |    0.00 |      0.00 |
| Ensemble champion   | vote_majority_bits {iter116J_s1, _s77, KD_v7}      | I10   | 0.9941 |   0.00 |    0.00 |      0.00 |
| Pareto F1-max       | vote_union_bits {iter116J_s1, _s77, KD_v7}         | I10   | 0.9959 |   0.92 |    0.66 |      0.76 |
```

## Next cycle (cron 10, ~18:23)

- Re-scan outputs/ for any KD_v9 partial eval if external GPU clears.
- If KD_v9 train log still 0-bytes at next tick, flag the cron diary as
  "supervisor v9b dispatch failed — escalate to master to relaunch with
  pre-batch GPU gate".
- Otherwise just log "no new metric" again.

**No appends to**: iters/, RESULTS_TIMELINE.md, tables/, 05_experiments.md.
Champion unchanged.
