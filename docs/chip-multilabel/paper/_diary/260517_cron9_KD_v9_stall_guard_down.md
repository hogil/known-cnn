# 2026-05-17 cron 9 (19:07) — KD_v9 train stall + resource_guard down + watchdog regex bug

## Tick summary

```
| Component                    | State at 19:07                                          |
|------------------------------|---------------------------------------------------------|
| KD_v9 python trainer pid     | 17680 alive, CPU_s 1859, WS_MB 343                      |
| KD_v9 best_model.pth mtime   | 17:24 (1h 43m stale; no new best since cron 8)          |
| KD_v9 train log file         | 0 bytes (buffered stdout, sealed on process exit)       |
| chain v9b supervisor bash    | pid 30968 / 51676 alive, waiting on trainer exit        |
| eval_n2000_pred (KD_v9)      | absent — train not yet returned ckpt                    |
| resource_guard               | DEAD at 18:18:54 ("budget 6h reached, exit")            |
| resource_watchdog_v2 polls   | #1-4 all show known_proc=0 (regex bug; PID 17680 not matched) |
| 4-agent cron cycle           | cycle 1 done, cron-driven only (no self-recurse, Windows popup policy) |
```

## New facts this tick

1. **KD_v9 (α=0.2 T=2 skipcm) training stall confirmed.**
   PID 17680 still alive (CPU_s 1859 ≈ 31 min wall-equivalent on single GPU,
   WS_MB 343) but best_model.pth mtime frozen at 17:24:31 — no new best
   checkpoint write in 1h 43m.  This pattern matches **chain v6 KD_v5
   (α=0.2 T=4)**: early-epoch best (val_macro_f1 high once), then
   degenerate solution thereafter.  KD_v5 final result was bit_F1 0.8658
   I3 with Total FAR 100 % (full collapse).  Direction-of-failure for
   α=0.2 thus reproduces across two T values (T=2 here, T=4 prior),
   confirming the cron 7 closure note that **the KD α-grid at T=2 has
   only one viable cell — α=0.3**.

2. **resource_guard exited at 18:18:54** with `"budget 6h reached, exit"`.
   Guard is the process that enforces the GPU 30-40 % shared-budget rule.
   With guard down, any subsequent dispatch from cron may exceed the
   30-40 % envelope — re-start required before chain v9b moves to phase 2
   (cutmix-p sweep) or phase 3 (complement-label-scale sweep).

3. **resource_watchdog_v2 keyword regex bug.**  Polls #1-4 all report
   `known_proc=0` despite python trainer PID 17680 being alive.  The
   watchdog's process-classification regex does not match
   `chip_multilabel._train_chip_variant`.  Action item for master:
   widen regex to also catch `_train_chip_variant`, `KD_v9`, or
   `outputs/KD_v9_*` substrings on the command line.

4. **chain v9b supervisor sequential-blocked.**  bash pids 30968 / 51676
   alive but idle — `train_eval_gated` waits on trainer subprocess exit
   before invoking `run_stage1` eval.  Eval will not start while pid
   17680 hangs.  No intervention this tick (per-policy "kill-all + restart"
   reserved for explicit user directive; cron is observer-only).

5. **Analyst cycle 1 recommendation (not dispatched).**  Model Soup
   (Wortsman 2022) — uniform weight average of `iter116J_s1 + s77 + KD_v7`
   checkpoints.  Expected gain +0.001 ~ +0.003 bit_F1, 0 train cost,
   ~5 min single-eval, targets the scratch-combo weak point of the
   current ensemble champion (vote_majority_bits 0.9941 / 0 %).  Master
   has not dispatched (cycle 1 single-pass policy); recorded here for
   cycle 2 pickup.

## Champion (unchanged)

```
| Tier     | Recipe                                                | Variant | bit_F1 | Total FAR |
|----------|-------------------------------------------------------|---------|--------|-----------|
| single   | iter116J seed=1 (T7 LS=0.30 g=3 cutmix-pair=masked)   | I10     | 0.9927 |      0.00 |
| ensemble | vote_majority_bits {s1 + s77 + KD_v7}                 | I10     | 0.9941 |      0.00 |
```

## Action items (for next cron tick or user directive)

```
| Priority | Item                                                                  | Owner        |
|----------|-----------------------------------------------------------------------|--------------|
| P0       | Restart resource_guard (30-40 % GPU budget enforcement)               | master       |
| P1       | Patch resource_watchdog_v2 regex to catch _train_chip_variant         | master       |
| P2       | Decide: kill PID 17680 (stall confirmed) vs let supervisor timeout    | user         |
| P3       | Dispatch Model Soup eval (analyst rec) — uniform avg s1+s77+KD_v7     | master       |
| P4       | After KD_v9 closure: skip α=0.2 cells in any future KD chain          | plan         |
```

## No metric changes this tick

- iter doc: not written (no eval results from KD_v9 yet)
- tables/all_runs_n2000.csv: not appended (no new eval row)
- RESULTS_TIMELINE.md section A / B / E: unchanged
- RESULTS_TIMELINE.md section C **KD_v9 row status note** updated:
  "train stall (best mtime 17:24, 1h 43m stale), α=0.2 collapse direction suspected, eval pending"

## Files touched this tick

- `docs/chip-multilabel/paper/_diary/260517_cron9_KD_v9_stall_guard_down.md` (this file)
- `docs/chip-multilabel/RESULTS_TIMELINE.md` (C-table KD_v9 row Status string only)

## Source

- KD_v9 trainer: pid 17680, started 17:05:18, alive at 19:07 (uptime 2h 02m)
- KD_v9 best ckpt: `outputs/KD_v9_a02_T2_skipcm/20260517_170518_T7_KD_v9_a02_T2_skipcm/best_model.pth` (mtime 17:24:31, 350 MB)
- KD_v9 train log: `outputs/_KD_v9_a02_T2_skipcm_train.log` (0 bytes)
- chain v9b supervisor: `outputs/_chain_v9b_summary.log`, `outputs/_chain_v9b_runner.log`
- resource_guard exit timestamp: 18:18:54 ("budget 6h reached, exit")
- resource_watchdog_v2 poll log: `outputs/_resource_watchdog_v2.log` (polls 1-4 all `known_proc=0`)
- prior tick: `docs/chip-multilabel/paper/_diary/260517_cron8_chain_v9b_KD_v9_train_in_progress.md`
- comparator KD_v5 collapse precedent: `docs/chip-multilabel/RESULTS_TIMELINE.md` § C
