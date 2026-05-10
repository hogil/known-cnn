# Diary — 2026-05-09 — Iter 25 6-seed I10 majority vote ensemble (BREAKTHROUGH)

## Context

Iter 21 closed with FCM-PM 19C as the single-model headline:
v14 = 0.9913 / 0.00 %, v15 = 0.9691 / 3.75 %. Iters 22–24 ran
hyperparameter sweeps (LS, fork pos_weight, alternate seeds,
auxiliary regularisers) on top of that base. Iter 25 ensembled
the resulting 6 single models.

## What changed in the paper

Five paper sections updated, append/edit only (no overwrites):

1. **`abstract.md`** — appended a final paragraph promoting the
   iter-25 6-seed I10 majority vote ensemble as the paper headline.
   Numbers cited from `paper_main_headline.csv` row
   `iter25_ensemble_majority`: v14 = 0.9976 / 0.00 %,
   v15 = 0.9913 / 0.00 %, all 4 defect F1 ≥ 0.987. +0.2041 v15
   bit-F1 vs 12-T5, +0.0222 vs 21 E single best.

2. **`04_methods.md`** — added new **§ 4.7 Ensemble Inference**
   subsection. Describes the 6-cell bag construction
   (LS × seed = 2 × 3), the 4-of-6 cell-vote aggregator, the
   aggregator choice (cell-vote > logit-mean for the bimodal-`ni_FAR`
   regime), pseudocode, and literature pointers (Hansen & Salamon
   1990 voting, Krogh & Vedelsby 1995 ambiguity decomposition,
   Tsoumakas & Katakis 2007 multi-label ensemble). Generalisation
   of the iter-10 H-ensemble (bag 2 → 6, aggregator logit-mean →
   cell-vote, diversity Normal-axis → LS × seed).

3. **`05_experiments.md`** — added new **§ 5.16 Phase 4 hparam tune
   + 6-seed ensemble** subsection (7 sub-subsections). Iter 22
   single-axis sweep table, iter 23 fork pos_weight (negative),
   iter 24 LS = 0.30 3-seed verify (bimodal-seed proof), iter 25
   ensemble headline + comparison vs 12-T5 / 21 E, why-it-works
   mechanism, hparam-axes summary, paper-claims unlocked.

4. **`06_analysis.md`** — added new **§ 6.11 Seed instability +
   vote-ensemble fix** subsection (6 sub-subsections). Decomposes
   `ni_FAR` (bimodal seed-axis) vs bit-F1 (unimodal); explains the
   threshold-position mechanism; updates §6.10 single-model claim
   to the bimodal-seed reading; extends iter-10 disjoint-failure-
   mode claim to LS × seed.

5. **`07_discussion.md`** — added new **§ 7.5.7 ensemble cost /
   readiness / limitations** + **§ 7.5.8 framing** subsections.
   6× train compute, 6× inference compute, 0× hparam retune at
   deployment. Production-ready on operational FAR + defect F1
   floor + seed-stability axes. Limitations: bag-size minimum 6,
   single-domain validation, rule-of-three FAR bound, no
   class-incremental support.

6. **`09_conclusion.md`** — added new **§ 9.4 Final paper headline
   + bimodal-seed lesson** section. Codifies the fifth lesson
   (bimodal-seed `ni_FAR` + vote-rule fix) and recommends
   ≥ 2/3-of-bag-size majority vote as the operational pattern.
   Lists 3 open questions for future work.

## Design decisions registered

- **Headline reset.** The paper's primary recommendation is now
  the 6-seed ensemble, **not** the iter-21 E single best. The
  single-best is retained as the strongest single-model baseline.
- **Numbers cited from logger sources only**:
  - `docs/chip-multilabel/iters/iter_22_25_full_phase4.md`
  - `docs/chip-multilabel/tables/paper_main_headline.csv`
- **No overwrites.** All 5 changed paper sections received
  *append* edits at their existing tail; the iter 1–21 narrative
  is preserved verbatim.

## Why the ensemble works (one-paragraph capsule)

Iter 24 surfaced that v15 `ni_FAR` is **bimodal in the seed axis**
at every operating point — at LS = 0.30, seed = 1 → 1.25 %,
seed = 7 → 67.5 %, seed = 42 → 50.0 %, while v15 bit-F1 stays
at 0.9921–0.9929. The bimodality comes from per-class threshold
position drift across seeds at saturated bit-F1, landing the
threshold on opposite sides of a v15 Normal-chip cluster. The
4-of-6 cell-vote aggregator is the smallest threshold that
out-votes a single bad-`ni_FAR` seed at one LS regime; defect
chips clear it at 5–6 / 6 (consensus), Normal chips fall to
0–2 / 6 (rejected), and the 2–3 / 6 ambiguous tail (where the
bimodality lives) is converted to 0-output. v15 `ni_FAR`
collapses to 0.00 % while bit-F1 lifts to 0.9913.

## Production guidance written into the paper

- 6-cell bag is the **minimum** bag size; 4 cells cannot suppress
  the worst-case bimodal draws.
- LS × seed are the **only two** ensemble axes recommended; all
  other hparam axes (CutMix-p, EMA, warmup, drop_path, fork
  pos_weight, lr-head) are net-negative in iter 22 and not
  worth ensembling.
- 6× compute is paid **once at deployment**; no retune at
  inference. Per-chip latency ≈ 600 ms is well within the
  operational budget.
- Rule-of-three: 0 / 280 v15 false-alarms gives a one-sided
  95 % upper bound of `ni_FAR` ≤ 1.07 %; we report 0.00 %
  empirical and flag the upper bound in §7.5.7.
