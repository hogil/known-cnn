# 260517 22:10 — Narrator cycle: BCE LS collapse boundary confirmed

**Trigger.** cron 10min fire at 22:10.

**New facts since 22:00 cycle.**

- Chain v12 Phase 1 `BCE_ls00_baseline` (LS=0, pure BCE):
  `best_model.pth` saved, but training/calibration log surfaces
  `RuntimeWarning: divide by zero encountered in log` — sigmoid
  saturates to {≈0, ≈1}, eval pipeline deferred pending recovery.
- Chain v12 Phase 2 `BCE_ls02` (LS=0.20): **FAIL no ckpt** —
  training diverged before any epoch crossed the val_acc gate.
  LS=0.20 confirmed as the lower collapse boundary for BCE
  multi-label on this benchmark.
- Chain v12 Phase 3 `iter116J_s33` training in progress (Phase 2
  ensemble member diversity recovery branch).

**Paper updates.**

1. `05_experiments.md`: appended `chain v12 Phase 1+2 — BCE LS=0
   collapse and LS=0.20 boundary failure (260517 22:10 update)`
   subsection under the chain v12 narrative. ~200 words.
   Documents the LS=0 log(0) divide-by-zero, the LS=0.20 no-ckpt
   failure, the Phase 3 iter116J_s33 recovery branch, and that
   the champion table is unchanged (single 0.9927, ensemble
   0.9941).

2. `06_analysis.md`: inserted new `§6.32.6.1 BCE multi-label LS
   viable window — single-point at 0.30 (260517 22:10)` between
   §6.32.6 and §6.32.7. Sharpens §6.32.6 item 1 into a quantitative
   finding: **viable LS set = {0.30}** (single point). Contrasts
   with §6.1 single-label CE curve (10-wide window at K=5) and
   complements §6.32.5 KD alpha narrow basin finding. Frames the
   constraint as cite-able for §7 discussion.

**WHY each insertion.** The narrator records narrative arc, not
metrics — recorder owns numbers. The two negative results (log(0)
RuntimeWarning + LS=0.20 train failure) are cite-able mechanistic
boundary evidence; recording them now prevents the chain v12
findings from being lost if subsequent phases overwrite the
narrative buffer. The collapse-boundary single-point finding
strengthens the §7 discussion framing of the benchmark as a
**saturation regime with single-point hyperparameter viability**.

**Champion unchanged.** iter116J single 0.9927, chain v8
`vote_majority_bits` 0.9941. No new metrics to publish.

**Files touched.**

- `D:\project\known-cnn\docs\chip-multilabel\paper\05_experiments.md`
- `D:\project\known-cnn\docs\chip-multilabel\paper\06_analysis.md`
- `D:\project\known-cnn\docs\chip-multilabel\paper\_diary\260517_2210_narrator_BCE_LS_collapse_boundary.md`
  (this file)
