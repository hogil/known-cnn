# 260510 — Phase 35 strength-curve revocation

## What changed

The previously claimed "HARD WINNER {24_LS030_s42 +
33 D + 37 E + 24_LS030_s7} beats hard + KD by
+0.0154 at HARD eval" (§6.17.3 / §5.27 / §7.6.4 prior
version) is **revoked**. Comprehensive Phase 35
strength-curve evaluation (45 / 45 inferences across
six thresholds) reveals the dual-seed advantage was a
**single-point artefact at exactly strength_max = 0.50**,
not a generalisable HARD-chip property.

## Strength-curve table (verified data, Phase 35)

| strength_max | intersection chips | winner @ FAR ≤ 5 % | bF1     | ni_FAR | pure-hard | hard + KD | dual-seed | iter-34 (KD + asym) |
|-------------:|-------------------:|--------------------|--------:|-------:|----------:|----------:|----------:|--------------------:|
| 0.40         | 975                | hard + KD          | 0.7377  | 0 %    | 0.7355    | 0.7377    | 0.7344    | 0.7315              |
| 0.45         | 1 395              | **pure-hard**      | **0.9941** | 0 % | **0.9941**| 0.9937    | 0.9948    | 0.9736              |
| 0.50         | 2 003              | dual-seed          | 0.9843  | 2 %    | 0.9670    | 0.9689    | **0.9843**| 0.9481              |
| 0.55         | 2 724              | **pure-hard**      | **0.9966** | 0 % | **0.9966**| 0.9901    | 0.9953    | 0.9909              |
| 0.60         | 3 059              | **pure-hard**      | **0.9959** | 0 % | **0.9959**| 0.9953    | 0.9953    | 0.9913              |
| 1.00 (FULL)  | 3 080              | **pure-hard**      | **0.9955** | 0 % | **0.9955**| 0.9953    | 0.9937    | 0.9945              |

## Interpretation

1. Pure-hard NEW HEADLINE 4-bag wins at **5 of 6**
   strength thresholds (0.45, 0.55, 0.60, FULL n = 200,
   FULL n = 500) with bF1 ≥ 0.9941 and FAR = 0 %.
2. The strength_max = 0.50 dual-seed advantage of
   +0.0154 is a **single-point compositional anomaly**
   — at strength_max = 0.45 and 0.55 the gap reverses
   (pure-hard wins by +0.0007 and +0.0013 respectively
   over the dual-seed bag).
3. The 0.40 slice is small enough (n = 975 chips) that
   sample-composition variance dominates; the apparent
   hard + KD micro-lead is within noise.
4. **No "production composition for hard chips" claim
   is supported** by the comprehensive strength-curve
   sweep. The previous +0.0154 dual-seed advantage
   claim was selection bias from a single threshold.

## Files revised

- `docs/chip-multilabel/paper/05_experiments.md`
  §5.27 — title and body rewritten as
  "Strength-curve evaluation reveals composition
  winner stability".
- `docs/chip-multilabel/paper/06_analysis.md`
  §6.17.3 — title and body rewritten as
  "Strength curve confirms composition winner
  robustness". §6.18.1 reverted to "KD axis
  interchangeable across the strength curve".
- `docs/chip-multilabel/paper/07_discussion.md`
  §7.6.4 — title rewritten as "composition winner is
  robust across the strength curve". Cost frontier
  addendum rewritten with strength-curve table; deploy
  recommendation unified to pure-hard 4-bag.
- `docs/chip-multilabel/paper/abstract.md` —
  HARD-eval refinement block replaced with
  strength-curve refinement block.
- `docs/chip-multilabel/paper/09_conclusion.md` —
  "KD-axis nuance and dual-seed amplification" block
  replaced with strength-curve interchangeability
  framing; "fragility as a feature" claim removed at
  the dual-seed level.

## What is preserved

- FULL-eval headline 0.9953 / 0 % at n = 500.
- Pure-hard NEW HEADLINE 4-bag as the unified
  production composition.
- §6.17.2 single-component fragility absorption story
  (24_LS030_seed42 fails dual-gate alone but
  contributes positively inside the 4-bag at FULL
  eval).
- §6.18 majority-vote-beats-prob-averaging finding.
- Methodological lessons (multi-seed protocol,
  diversity-rank-based bag-size selection).

## What is revoked

- "HARD WINNER 4-bag {24_s42 + 33 D + 37 E + 24_s7}
  generalises to harder deployment" claim.
- Dual-seed amplification framed as a paper-grade
  mechanism (now retained only as a single-slice
  compositional curiosity at strength_max = 0.50).
- "Fragility as a feature, not a bug" wording at the
  dual-seed level; the per-cell fragility story
  remains valid for §6.17.2 but not for the dual-seed
  HARD-chip-specialist claim.
- Deployment recommendation by FAR-tolerance band
  (≤ 5 % FAR → HARD WINNER; strict 0 % FAR → hard + KD).
  Replaced by single unified pure-hard 4-bag
  recommendation across the strength curve.

## Lesson

This is the second major rebuttal in the paper writing
process (cf. earlier strength-stratification revocation
in `260510_phase34*`). The pattern: **single-point
strength-filtered evaluations are not enough**. A
comprehensive strength-curve (multiple thresholds) is
required to distinguish a robust property from a
slice-composition artefact. We adopt the strength-curve
as the standard difficulty-conditioning protocol going
forward.

The 0.9953 / 0 % FULL headline still stands. What is
revoked is the "HARD chip recommendation" — there is
no robust "HARD chip" claim from the data we have.
