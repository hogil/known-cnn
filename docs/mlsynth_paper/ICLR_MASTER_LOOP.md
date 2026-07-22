# ICLR master loop -- orchestration spec

GOAL: maximize honest ICLR acceptance probability for the mlsynth paper.
User target = 60%. HONEST current (JUDGE r1) = ~18%. The loop pushes probability
UP every round via real theory/evidence work; it NEVER fabricates a number.
Reaching 60% is not guaranteed and may be impossible for this paper direction --
the loop reports the true probability each round and says so if it plateaus.

## Roles (master dispatches these each round)
- MASTER (this loop, in main thread): owns the round cycle, applies FIX edits +
  commits, updates status, decides continue/stop, reports honest probability.
- JUDGE (opus subagent, read-only): adversarial ICLR review -> top reject levers
  + ONE highest-leverage next action + honest probability. Never edits.
- THEORY (opus subagent, read-only): attacks a specific theorem/positioning gap;
  returns paste-ready proof/section text + honest verdict (proven vs assumed).
- EXPERIMENT (subagent, GPU under resource gate): only when JUDGE prescribes a
  PRE-REGISTERED gated experiment; never retries a seen test; freezes config+hash.
- FIX = MASTER applies the returned text to the paper, runs texcheck if latex,
  commits + pushes, appends round log.

## Round cycle
1. JUDGE reviews current state -> (reject levers, action, prob).
2. MASTER routes action: theory-deepening -> THEORY; gated experiment -> EXPERIMENT
   (resource-check first); positioning/writing -> MASTER direct.
3. Worker returns text/result. MASTER applies + commits + logs.
4. JUDGE re-reviews -> new prob. If prob rose, keep the change; if a THEORY result
   is "still assumed/false", record honestly and try the next-best action.
5. STOP when: prob reaches 60%, OR 3 consecutive rounds with <+1pp gain
   (plateau), OR user stops. On plateau, MASTER states the honest ceiling and
   recommends venue (ICLR vs TMLR).

## Guardrails (non-negotiable)
- Never fabricate probability; JUDGE sets it adversarially.
- Never claim method-superiority (SVHN GATE-2 closed that path).
- Never retry the seen SVHN test; new benchmarks need a fresh pre-registered gate.
- GPU experiments only after a resource-monitor check (this session hit fork-fail
  /OOM; kill my own runs + retry, never touch codex/MCP python).
- Do not stage other authors' (codex) uncommitted files.
- Every round: commit evidence, append round log, report honest prob to user.

## Probability ledger
- r1 JUDGE: ~18% (two-point theorems read as expected; T4(a) trivial; no method win).
- (subsequent rounds appended)
