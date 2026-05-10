# 260510 — Phase 44 n = 200 4-bag big-sweep

## Setup

- Eval: v15direct n = 200, 3 080-chip intersection across 9 model preds
- Pool: 14 cells (FCM-PM bank + KD-student + asymmetric + iter-46 white-fill cells)
- Combinations exhaustively scored: **1 001** (C(14, 4))
- Aggregation: τ = 2 / 4 simple-majority vote on per-cell I10 cell-decisions

## Top 10 4-bags (by v15direct bit-F1)

| rank | 4-bag                                                              | bF1    | ni_FAR | bb / fk / sc / sr               |
|-----:|--------------------------------------------------------------------|-------:|-------:|---------------------------------|
| 1    | 24_LS030_seed7 + 26 B + 26 H + 37 E                                | 0.9966 | 4.50 % | 1.0000 / 0.9873 / 0.9992 / 1.0000 |
| 2 ★  | **24_LS030_seed42 + 26 H + 33 A + 37 E**                           | 0.9964 | 0.00 % | 0.9992 / 0.9881 / 0.9984 / 1.0000 |
| 3    | 26 B + 26 H + 37 E + 42 C                                          | 0.9963 | 0.00 % | 0.9992 / 0.9882 / 0.9977 / 1.0000 |
| 4    | 24_LS030_seed42 + 26 B + 26 H + 37 E                               | 0.9963 | 0.00 % | 0.9992 / 0.9881 / 0.9977 / 1.0000 |
| 5    | 24_LS030_seed7 + 26 B + 37 E + 42 C                                | 0.9962 | 2.50 % | 0.9977 / 0.9873 / 1.0000 / 1.0000 |
| 6    | 24_LS030_seed7 + 26 H + 37 E + 42 C                                | 0.9961 | 5.00 % | 0.9969 / 0.9882 / 0.9992 / 1.0000 |
| 7    | 24_LS030_seed42 + 21 H + 26 H + 37 E                               | 0.9961 | 0.00 % | 0.9984 / 0.9874 / 0.9984 / 1.0000 |
| 8    | 24_LS030_seed42 + 26 B + 37 E + 42 C                               | 0.9961 | 0.00 % | 0.9977 / 0.9881 / 0.9984 / 1.0000 |
| 9    | 26 B + 37 E + 42 C + 46 E                                          | 0.9961 | 0.00 % | 0.9969 / 0.9881 / 0.9992 / 1.0000 |
| 10   | 24_LS030_seed42 + 26 B + 33 A + 42 C                               | 0.9961 | 0.00 % | 1.0000 / 0.9881 / 0.9961 / 1.0000 |
| paper main | 24_LS030_seed42 + 26 B + 26 D + 26 H (pure-hard)             | 0.9955 | 0.00 % | 0.9984 / 0.9881 / 0.9953 / 1.0000 |

## Findings

1. **Top-10 spread = 0.0005 bit-F1** — at the n = 200 sampling-noise floor (≈ 1 chip
   out of 2 000 defect chips per ±0.0005 step). No single 4-bag is statistically
   distinguished from another in the top-10.

2. **NEW best at n = 200**: {24_LS030_seed42 + 26 H + 33 A + 37 E} = **0.9964 / 0 %**.
   This combines all four diversity axes (hard + hard-fill-variant + KD + asymmetric).
   +0.0011 over the paper main (0.9955 / 0 %) — within sampling noise but consistent
   across the top-10 ranks.

3. **Asymmetric axis (37 E) reasserted** — appears in **9 of top 10** rows. The Phase 36
   strength-curve revocation was about HARD050-specific dual-seed claims; the
   asymmetric axis itself remains paper-relevant as a free diversity axis at the
   n = 200 single-strength evaluation.

4. **Pure-hard vs all-4-axes is at the noise floor** — the 5/6 strength-threshold win
   for pure-hard (§6.17.3) remains the deciding factor. At single-strength n = 200,
   all-4-axes blends edge slightly higher; across the strength curve, pure-hard
   dominates. Both compositions are valid 4-bag deployments.

## Interpretation (paper-grade honest framing)

- The 0.9953 NEW HEADLINE pure-hard 4-bag remains valid as **one of the tied 4-bag
  configurations** at the noise floor.
- The Phase 44 sweep finds slightly higher-bF1 alternatives at n = 200 that include
  the asymmetric (37 E) and KD (33 A) axes.
- The "pure-hard wins" / "all-axes wins" dichotomy is **at the sampling-noise floor**
  at single-strength n = 200; the strength-curve robustness analysis (§6.17.3)
  retains pure-hard as the deployment recommendation.

## Paper updates (Phase 44)

- §5.31 — new subsection with top-10 table + noise-floor interpretation
- §6.17.4 — new sub-subsection: all-4-axes blend at the noise floor
- §7.6 — appended Phase 44 refinement to the cost frontier (deployment unchanged)
- abstract — appended single paragraph on the n = 200 nuance

This is **not a self-correction** — it is a refinement adding nuance. The 0.9953
headline pure-hard 4-bag still stands as the recommended deployment composition.

_Source: Phase 44 sweep results, paper §5.31, §6.17.4, §7.6, abstract paragraph._
