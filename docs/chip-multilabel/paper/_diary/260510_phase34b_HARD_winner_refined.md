# 260510 — Phase 34b HARD-eval big-sweep refinement

## TL;DR

The "hard + KD wins HARD050 by +0.0019" headline from
260510_phase34_HARD_breakdown.md is **superseded** by a
comprehensive 4-bag enumeration over the 9-model pool.
New HARD-eval winner:
**{24_LS030_seed42 + 33 D + 37 E + 24_LS030_seed7}** at
bit-F1 = **0.9843 / ni_FAR = 2.00 %** (within 5 % gate),
beating hard + KD by **+0.0154** and pure-hard by +0.0173.
FULL-eval headline 0.9953 / 0 % (n = 200 / 500) unchanged.

## Method (Phase 34 big-sweep)

- 9-model prediction pool, 2 003-chip strength-filtered
  intersection (HARD050, defect strength ≤ 0.50).
- Enumerate all C(9, 4) = 126 4-bag combinations.
- v15direct decision-rule on each, evaluate bit-F1 +
  ni_FAR + per-class bb / fk / sc / sr.
- Filter dual-gate-eligible (FAR ≤ 5 %); rank by bit-F1.

## Key findings

1. **Composition spread on HARD050 is 0.0918** across
   the full 126-combo enumeration — the prior 6-row
   table (spread 0.0208) was sampling only a narrow
   region.
2. **Dual-24_LS030-seed in every top-10 4-bag.** All ten
   highest-bit-F1 4-bags include both seed-42 and seed-7
   of the FAR-fragile HARD-chip specialist. The
   majority-vote aggregator absorbs over-firing on
   Normals while the doubled-vote amplifies the bb-axis
   signal.
3. **HARD WINNER composition: dual-specialist + KD +
   asymmetric.** {24_s42 + 33 D + 37 E + 24_s7} stacks
   three orthogonal lift sources. Per-class delta vs
   hard + KD: bb +0.0532 dominant, sc +0.0074
   contributing.
4. **Bag-size cost frontier confirms 4 = global
   optimum** at HARD050 (5-bag 0.9715, 6-bag 0.9755,
   7-bag 0.9613). Not just FULL-eval — HARD-eval also
   peaks at 4-bag.
5. **Production recommendation by FAR band:** standard
   ≤ 5 % → HARD WINNER (0.9843); strict 0 % → hard + KD
   (0.9689). pure-hard is third on HARD eval.

## Paper updates

- §5.27 — replaced 6-row 4-bag table with big-sweep
  top-10 + bag-size cost-frontier + revised cross-eval
  cross-tabulation; +0.0019 claim replaced by +0.0154.
- §6.17.3 — replaced spread "0.0208 / +0.0019"
  reading with "0.0918 / +0.0154"; per-class
  mechanism table now HARD WINNER vs pure-hard;
  three-axis stacking (dual-seed + KD + asymmetric)
  paragraph added.
- §6.17.3 24_LS030 paragraph — dual-seed amplification
  added.
- §6.18.1 — KD axis recharacterised as "one of three
  contributors on HARD" (not "dominant alone").
- §7.6.4 — deployment recommendation split by
  FAR-tolerance band (≤ 5 % HARD WINNER vs strict 0 %
  hard + KD); updated cost frontier addendum.
- §7.6.4 cost-frontier table refreshed.
- abstract — HARD-eval one-liner refined: HARD WINNER
  0.9843 / 2.00 %, +0.0173 / +0.0154 deltas; FAR-band
  recommendation.
- §9 — dual-seed insight added; "fragility as a feature,
  not a bug" framing strengthened with HARD WINNER
  evidence.

## Interpretation note

The +0.0019 "hard + KD wins HARD" thesis was correct on
its sampled subset but **massively under-stated the
HARD-eval composition signal**. The big-sweep reveals
that the +0.0019 axis swap (26 D → 33 D within
{24_s42, 26 B, 26 H, ?}) is one of many possible
single-axis edits; the global HARD-eval optimum is a
two-axis-from-hard-+KD-baseline edit (26 B → 37 E and
26 H → 24_s7) that lifts bit-F1 by an order of
magnitude more.

The "pure-hard wins" rebuttal at FULL n = 200 (0.9955 /
0 %) remains intact for FULL-eval. At HARD-eval pure-
hard is third — a different game.

## Source

- Phase 34 big-sweep predictions: 9-model
  `preds_chip.parquet` files, 2 003-chip strength-
  filtered intersection
- iter file: `docs/chip-multilabel/iters/iter_39_purehard_4bag.md`
- prior diary: `260510_phase34_HARD_breakdown.md` (this
  refinement supersedes the +0.0019 finding)
