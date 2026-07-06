# 260520 cron #441 — cls=0.3 catastrophic collapse, §5.52 cls-axis Addendum 2

**Trigger.** Autoloop cron #441 (10min tick). Cron #440 had dispatched the fair-train cls=0.3 8ep follow-up that §5.52 Addendum 1 (cron #438) flagged as required. Result extraction from cron #440's `fcm_nopair_cls03` run now ready.

**Headline finding.** `fcm_nopair_cls03` (cls=0.3, pair-off, g=3, **full 8ep** matched to iter116J protocol) collapses into universal scratch over-positive assertion:

- **Total FAR = 100.00 %** (all Normal + Invalid + OOD chips assert at least one positive)
- **Normal FP = 1506 / 2000** chips receive scratch prediction (75.3 % Normal-only FP rate; remaining 494 receive bb/fork/scratch_rot)
- bit_F1 effectively collapsed (model is positive-saturated on everything)

**Comparison to Addendum 1 (cron #438).** The 1ep cls=0.3 cell (bit_F1 0.8876 / Total FAR 4.55 %) was **not** an early under-fit version of the same regime — it was a pre-collapse cell that had not yet crossed the leakage knee. Matched 8ep training drives cls=0.3 catastrophically past the leak point.

**§5.52 update.** Appended "Addendum 2 — cls=0.3 catastrophic collapse at fair 8ep" to §5.52 cls-axis paragraph. Refined interpretation:

- cls=0.3 = catastrophic, **discarded** from §4 design-space
- cls=0.5 = **confirmed sweet spot** (pair-on iter116J 0.9927 / 0 % FAR + pair-off §5.51 0.9943 / 11.81 % FAR, both stable)
- cls=0.7 = **upper bound TBD** (cron #440 currently training)
- cls axis simplifies from "continuous knob" to "knife-edge with collapse below cls=0.4"
- §4 design-space simplifies: cls fixed at 0.5, active knobs = pair-mask × g-group

**Champion frozen.** E22 (§5.49.7, bit_F1 0.9956 / 0.00 % Total FAR) unchallenged.

**Source.** `outputs/fcm_nopair_cls03_260520_*/eval_n2000_pred/stage1_*/preds_chip.parquet + eval_summary.json`.

**WHY paper-worth.** Documented failed-promotion case strengthening §1 commitment to surface failed iterations. The cls axis turns out to be discontinuous rather than smooth, which is itself a method-section narrative simplification.
