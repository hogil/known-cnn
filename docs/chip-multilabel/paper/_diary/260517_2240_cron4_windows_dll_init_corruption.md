# 260517 22:40 — cron #4 — Windows DLL init system corruption confirmed

## TL;DR

Analyst-confirmed escalation of the chain v12 dispatch fault
(§6.32.6.2-6.32.6.3) to a **system-level Windows DLL init
corruption** state. PowerShell sessions silent-fail; every fresh
python spawn deadlocks at torch CUDA init; per-process
`Stop-Process -Name python -Force` no longer recovers a clean
dispatch state. **OS reboot required.** Champion unchanged
(iter116J single 0.9927 / 3-way vote 0.9941).

## What changed since cron #3 (22:30)

- §6.32.6.3 protocol was "kill all python + verify idle + redispatch"
  (sufficient for the 22:15-22:30 window).
- 22:30-22:40 window: the same protocol stopped working — fresh
  dispatches no longer reach the dispatcher, PowerShell itself
  silent-fails.
- Analyst escalated the diagnosis to OS-level DLL loader corruption.

## Why this is paper-level, not just an ops note

Three nested failure modes are now characterised:

1. per-trainer config bug → 1-epoch smoke catches
2. per-host zombie accumulation → `Stop-Process -Name python -Force` catches (§6.32.6.2-6.32.6.3)
3. per-host DLL init corruption → only OS reboot catches (this cron)

The §7 discussion paragraph on reproducibility gains a stronger
disclosure: chain studies in the saturated 0.985-0.995 bit_F1
regime have headline-altering dependencies on the dispatch
infrastructure's survivor set, and a replication on a different
OS + CUDA stack can legitimately differ at the third decimal
place purely because the alternative infrastructure encountered
a different bag of survivor trainers.

## Paper update

- §6.32.6.4 added — "Escalation to system-level DLL init
  corruption — Windows reproducibility constraint (260517 22:40)".
- §6.32.7 connection paragraph unchanged structurally; chain v12
  data still pending.

## Champion status

Unchanged.

```
| Recipe                                 | bestI | bit_F1 | Total FAR | Status     |
|----------------------------------------|-------|--------|-----------|------------|
| iter116J s=1 T7 BCE+LS=0.30 g=3 FCM-PM | I10   | 0.9927 |      0.00 | single ch  |
| 3-way vote_majority_bits ensemble      | -     | 0.9941 |      0.00 | ensemble   |
```

## Action items (not for the agent — for the orchestrator)

- OS reboot per analyst recommendation
- Post-reboot: re-dispatch chain v12 phases 2-8
- Verify enforcer v6 30% CUDA cap still alive post-reboot
- No new metric expected from this cycle

_Source: cron fire #4 narrative update; analyst diagnosis of
Windows DLL init system corruption; champion unchanged._
