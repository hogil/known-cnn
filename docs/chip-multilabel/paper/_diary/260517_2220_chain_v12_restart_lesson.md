# 260517 22:20 — chain v12 restart, Windows zombie + torch CUDA init hang lesson

## Trigger (cron fire #2)

User directive at 22:15:
- chain v12 trainer pattern = first trainer (BCE_ls00) succeeds, all subsequent trainers hang at torch CUDA init
- Diagnosis: Windows + python + CUDA zombie process accumulation deadlocks new context init
- Command: `Stop-Process -Name python -Force` of all python, idle confirm, re-dispatch chain v12

## Findings (this cron tick)

1. **Pattern characterised** — first-trainer-success / nth-trainer-hang is the canonical
   signature of CUDA zombie deadlock on shared Windows GPU hosts. No exception, no OOM,
   just an indefinite hang on torch CUDA init when the trainer process enters.

2. **Root cause** — orphaned python processes (from prior Bash `run_in_background`
   dispatches and prior chain attempts) hold partial CUDA contexts. Driver-side
   cleanup only runs on full process exit and serialises new context requests
   behind orphans once accumulation passes ~4 zombies.

3. **Only reliable recovery** — `Stop-Process -Name python -Force` of all python,
   not single-process kill, not in-place restart. Single-process kill leaves
   at least one zombie behind and next trainer re-hangs within seconds.

4. **Champion unchanged** — single 0.9927 (iter116J), ensemble 0.9941 (chain v8
   3-way vote) hold. Chain v12 has not yet produced a new candidate metric;
   restart is awaiting first new eval result.

## Paper updates this tick

- `06_analysis.md` §6.32.6.2 appended — new subsection "Operational infrastructure
  lesson — Windows zombie accumulation + torch CUDA init hang (260517 22:20)".
  Covers pattern, root cause, operational protocol, generalisation, paper
  implication for §7 (loss budget on shared Windows GPU hosts, ~1.1× wall-clock
  nominal recommendation, sensitivity disclosure for saturated-regime results).

- This diary entry.

- §5 experiments and §6.32 numeric results table unchanged — no new metric to
  append (champion still 0.9941 ensemble / 0.9927 single).

## What to monitor next cron tick

- Whether the re-dispatched chain v12 trainers (post-kill) make it past torch
  CUDA init this time.
- Whether any of {BCE_ls02 re-run, KD alpha corner sweep, ensemble member sweep}
  produce a new metric that displaces 0.9941 or fills the §6.32.5 KD-alpha
  basin geometry question.
- If pattern repeats (first trainer ok, second hangs), confirm the kill-all
  protocol is the right rule rather than dispatcher-side fix attempts.

## Cross-references

- `memory/feedback_windows_python_dispatch.md` (260506) — session-rule version
  of the same finding.
- `memory/feedback_problem_kill_restart_rule.md` (260515) — absolute rule
  "any problem = kill all + restart" derived from this and other failure modes.
- `paper/_diary/260517_chain_v12_dispatch.md` — chain v12 dispatch event.
- `paper/_diary/260517_2210_narrator_BCE_LS_collapse_boundary.md` — adjacent
  finding (BCE multi-label LS viable window single-point at 0.30) that the
  chain v12 dispatch was designed to characterise.
