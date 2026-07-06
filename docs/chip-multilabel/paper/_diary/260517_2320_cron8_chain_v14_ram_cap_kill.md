# 260517 23:20 — cron #8 chain v14 RAM cap kill, chain v15 pending decision

**Trigger.** Cron fire #8 at 23:20. Narrator cycle.

**New fact.**
- Chain v14 trainer config `batch=1 img=224` (floor knob configuration)
  still violated the 30 % host-RAM enforcer cap → all chain v14 trainers
  killed.
- Analyst pivot recommendation: grad-checkpointing + `img=384` (restoring
  native backbone resolution) as the chain v15 candidate.
- User decision on chain v15 dispatch: **pending**.

**Champion (unchanged).**
- Single: iter116J **0.9927** bit_F1.
- Ensemble: 3-way vote **0.9941** bit_F1 / **0%** Total FAR.
- No new metric produced this cycle.

**Paper change.**
- Appended `§6.32.6.5` to `06_analysis.md` — "Trainer footprint vs 30 %
  RAM cap — incompatible without grad-checkpointing".
- Extends the §6.32.6.2-6.32.6.4 nested failure-mode hierarchy from
  three levels (config bug / zombie accumulation / DLL init corruption)
  to **four levels**, adding **(iv) per-trainer activation-memory
  floor above the enforcer cap**.
- §6.32.6.5 frames the floor-knob exhaustion (`batch=1`, `img=224`
  minimum) as the diagnostic for level (iv): when the cheapest
  reduction knobs trip the cap, the activation-storing forward pass
  is the irreducible cost, and the trainer recipe must be
  re-architected (grad-checkpointing, mixed-precision activation,
  backbone swap) rather than further knob-tuned.

**WHY this is a paper-level finding, not just an ops note.** The
§6.32.6.2-6.32.6.4 arc established that infrastructure failures
(zombie accumulation, DLL init corruption) are paper-level findings
because they constrain the reproducibility budget of any saturation-
regime claim. §6.32.6.5 adds the same kind of constraint at the
**trainer-recipe level**: a chip-multi-label result claimed under a
30-40 % host-RAM share, with a backbone whose forward pass stores
activations natively, cannot be replicated by naive batch/img
reduction alone. Grad-checkpointing (or equivalent) becomes a
required infrastructure dependency, not an optional optimisation.

**Status.**
- Chain v14: dead (RAM cap kill).
- Chain v15 (grad-checkpointing + img=384): candidate, awaiting user
  decision.
- Narrative pending chain v15 decision.

_Sources: cron #8 23:20 trigger; chain v14 enforcer kill log;
analyst pivot recommendation grad-checkpointing + img=384; champion
unchanged iter116J single 0.9927 / 3-way vote 0.9941 / 0% Total FAR;
appended §6.32.6.5 to `D:/project/known-cnn/docs/chip-multilabel/paper/06_analysis.md`._
