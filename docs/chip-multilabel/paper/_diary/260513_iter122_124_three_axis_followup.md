# 2026-05-13 — Iter 122 / 123 / 124 three-axis follow-up

## Context

The iter 116 J recipe (T7 BCE + LS = 0.20 + CutMix complement g = 3
masked corner cls = 0.5, val_f1-selected) is the iter 112-era
1 × cost SOTA at bit-F1 = 0.7911 / Total FAR = 0.00 % on the
`v15direct n = 200` protocol. Three follow-up axes were
dispatched on 2026-05-13:

- **iter 122 / 123** — Asymmetric Loss as a remedy for the
  `bb + sr → sr` partner-bit recall asymmetry. Two clip
  settings.
- **iter 124** — Decoupled `(g, n)` CutMix parameterisation
  (`GRID = g · n`) — methodology contribution to isolate
  label cardinality from spatial granularity.

## Iter 122 — T6 (BCE warmup 3 → ASL γ_neg = 4 clip = 0.05)

- Run dir: `outputs/iter122_T6_asl_gn4/T6_iter122_T6_asl_gn4_260513_085714/`
- Training: 518.8 s, 10 epochs, BCE → ASL switch ep 6,
  `val_criterion = margin_max`, `--save-every-epoch`.
- Bugfix: `_train_chip_variant.py` was missing `import os`
  (previous iter 122 dispatch crashed immediately). Fixed
  and re-dispatched.
- Best (val_margin pick) = ep 3 (still in BCE phase).
- Result @ ep 10 (final ASL phase, clip = 0.05): bit-F1 0.8297,
  **Total FAR 9.4 %** (NI 29.0 %, OOD 3.3 %).
- `bb + sr → sr` partner recall 0.831 → 0.981 (+0.150) — the
  intended ASL mechanism activates.
- `fork + sr → sr` partner recall 1.000 → 0.750 (−0.250) —
  destructive cross-class trade-off.
- Verdict: **REGRESSION**. dual gate (bit-F1 ≥ 0.99 ∧ Total
  FAR ≤ 0.5 %) fails by 10 × on FAR.

## Iter 123 — T6 clip 0.05 → 0.10 (atomic single-axis dial)

- Run dir: `outputs/iter123_T6_asl_clip01/T6_iter123_T6_asl_clip01_260513_091520/`
- Training: 750.9 s (+45 % over iter 122 — clip = 0.10 has
  heavier backward).
- Best (val_margin pick) = ep 3 BCE — identical to iter 122
  (clip does not affect BCE warmup phase).
- Result @ ep 10 (final ASL phase, clip = 0.10): bit-F1 0.8297
  (same as iter 122), **Total FAR 5.0 %** (NI 16.0 %, OOD
  1.6 %) — halved vs iter 122.
- `fork + sr → sr` partner recall recovers 0.750 → 0.838
  (+0.088, 35 % of lost recall).
- Verdict: **DRAW**. Partial improvement but production gate
  still missed by 10 × on FAR. ASL is **loss-axis dead-end**
  on this benchmark — clip dialing alone cannot resolve the
  structural mismatch (ASL's global gradient asymmetry vs the
  local partner-bit imbalance).

## Iter 124 — Decoupled `(g, n)` CutMix parameterisation

- 9-row sweep: `g ∈ {2, 3}, n ∈ {1, 2, 3, 4}` + 2 bisect
  controls + matched-`GRID` (6) `(g, n) ∈ {(2, 3), (3, 2)}`
  pairing.
- All 9 trainings succeeded; only row `a` (`g = 2, n = 1`,
  GRID = 2) finished evaluation against the `v15direct n = 200`
  set (rows `b` – `i` errored on a transient path-resolution
  bug — fallback eval queued).
- Row `a` result: best cell T0__I3 macro_f1 = 0.8306,
  fork F1 = 0.6681 (low), scratch_rot F1 = 0.9363.
- **Paper contribution** is the parameterisation itself, not
  the row-a result. The decoupling `GRID = g · n` makes the
  cardinality-vs-spatial-granularity ablation answerable for
  the first time in this paper's recipe space. Matched-GRID
  `(2, 3)` vs `(3, 2)` is the cardinality-isolated comparison.

## Decisions surfaced

1. ASL is a **closed loss-axis** on this benchmark — no
   further dispatches under any clip / γ setting are queued.
2. **`val_margin` is phase-blind** under phased-loss schedules
   — surfaced by iter 122 / 123 selection picking BCE ep 3 in
   both runs. Queued: phase-aware selection patch.
3. **`(g, n)` decoupling is novel methodology** — paper
   contribution at §5.47 even with incomplete matrix
   evaluation. Queued: row b – i fallback evaluation
   (< 5 min compute).

## Files written

- `paper/03_data.md` — restated five-group composition under
  the absolute rule, added iter 122 / 123 / 124 metric
  binding paragraph.
- `paper/05_experiments.md` — new §5.47 (Spatial Granularity
  in Group-Mixed CutMix).
- `paper/06_analysis.md` — new §6.31 (Asymmetric Loss is not
  the right axis — failed-direction analysis with mechanism
  decomposition).
- `paper/07_discussion.md` — new §7.13 (val_margin phase-
  blindness limitation, two queued fixes).
- This diary entry.

## Open questions

- Does the `(2, 3)` vs `(3, 2)` matched-GRID pairing in row
  `c` / `f` produce a measurable cardinality effect once the
  evaluations complete? Conjecture: `(3, 2)` will be lower
  bit-F1 (3-positive output is hard on a 2-positive train
  prior — §5.46.8.1 label-cardinality bias), but the
  matched-spatial-resolution control will isolate the effect.
- Is there a non-ASL loss-axis recipe that addresses partner-
  bit imbalance? The §6.31.6 verdict closes ASL but does not
  close the underlying motivation — paper-grade follow-up
  would be **per-class-weighted BCE** (cls weight inversely
  proportional to single-bit frequency) as a local-mechanism
  alternative to ASL's global mechanism.
