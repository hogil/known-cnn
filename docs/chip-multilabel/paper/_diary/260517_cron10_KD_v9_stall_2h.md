# 2026-05-17 cron 10 (19:34) — KD_v9 stall reaches 2h 10m, chain v9b 9 phases blocked

## Tick summary

```
| Component                       | State at 19:34                                       |
|---------------------------------|------------------------------------------------------|
| KD_v9 python trainer pid        | 17680 still alive (uptime 2h 29m since 17:05:18)     |
| KD_v9 best_model.pth mtime      | 17:24 (2h 10m stale; +27 min stall vs cron 9)        |
| KD_v9 train log file            | unchanged (buffered, sealed on exit)                 |
| chain v9b queue                 | 9 phases blocked (KD_v10 + cmp p×5 + ls×3)           |
| user decision on KD_v9 kill     | not received (P2 from cron 9 still open)             |
| resource_guard                  | RESTARTED (poll #20 at 19:32; _resource_guard.log refreshing) |
| resource_watchdog_v2            | poll #16 / #36 cycling normally on 5 min cadence     |
| resource_watchdog_v2 known_proc | 0 (regex bug from cron 9 still in effect; permanent) |
| champion (single + ensemble)    | unchanged                                            |
```

## New facts this tick (no new metrics)

1. **KD_v9 stall extended to 2h 10m.**  best_model.pth mtime frozen at
   17:24 since cron 8 — no new best in past three cron ticks (8 → 9 → 10).
   PID 17680 still alive but eval cannot start because chain v9b supervisor
   bash waits on trainer subprocess exit before invoking `run_stage1`.
   Same α=0.2 collapse direction-of-failure as KD_v5 (T=4) confirmed by
   prolonged no-best window.

2. **chain v9b queue of 9 phases now blocked downstream of KD_v9.**
   Phase 1 second slot KD_v10 (α=0.3 T=1) + cutmix-p sweep (5 cells:
   p=0.05/0.10/0.15/0.20/0.25) + complement-label-scale sweep (3 cells:
   ls=0.20/0.25/0.35) all hold pending KD_v9 trainer exit.

3. **User decision on KD_v9 kill not received.**  Cron 9 P2 action item
   "Decide: kill PID 17680 (stall confirmed) vs let supervisor timeout"
   remains open.  Per cron-observer policy + 260515 "Problem → kill-all +
   restart" rule reserved for explicit user directive, no in-place
   intervention this tick.  Hold (boryu) status.

4. **resource_guard back up.**  Was DEAD at cron 9 (18:18:54 budget-6h
   exit).  Now refreshing — poll #20 at 19:32 in `_resource_guard.log`.
   GPU 30-40 % shared-budget envelope re-enforced for any subsequent
   chain dispatch.

5. **resource_watchdog_v2 polls on schedule but regex bug persistent.**
   Poll #16 / #36 visible on 5 min cycle, no log corruption, but
   `known_proc=0` continues for PID 17680 (regex does not match
   `chip_multilabel._train_chip_variant` command line).  Action item P1
   from cron 9 still open for master patch.

## Champion (unchanged from cron 9)

```
| Tier     | Recipe                                                | Variant | bit_F1 | Total FAR |
|----------|-------------------------------------------------------|---------|--------|-----------|
| single   | iter116J seed=1 (T7 LS=0.30 g=3 cutmix-pair=masked)   | I10     | 0.9927 |      0.00 |
| ensemble | vote_majority_bits {s1 + s77 + KD_v7}                 | I10     | 0.9941 |      0.00 |
```

## No metric changes this tick

- iter doc: not written (no eval results from KD_v9 yet)
- tables/all_runs_n2000.csv: not appended (no new eval row)
- RESULTS_TIMELINE.md section A / B / E: unchanged
- RESULTS_TIMELINE.md section C **KD_v9 row status note** only updated:
  "2h+ stalled (best mtime 17:24, 2h 10m stale at 19:34); user kill-decision boryu"

## Files touched this tick

- `docs/chip-multilabel/paper/_diary/260517_cron10_KD_v9_stall_2h.md` (this file)
- `docs/chip-multilabel/RESULTS_TIMELINE.md` (C-table KD_v9 row Status string only)

## Source

- KD_v9 trainer: pid 17680, started 17:05:18, alive at 19:34 (uptime 2h 29m)
- KD_v9 best ckpt mtime: 17:24:31 (frozen since cron 8)
- resource_guard restart evidence: poll #20 at 19:32, `outputs/_resource_guard.log`
- resource_watchdog_v2 polls: #16 / #36 visible in `outputs/_resource_watchdog_v2.log`
- prior tick: `docs/chip-multilabel/paper/_diary/260517_cron9_KD_v9_stall_guard_down.md`
