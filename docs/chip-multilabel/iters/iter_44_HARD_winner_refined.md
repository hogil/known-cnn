# iter 44 — HARD050 winner refined (Phase 34 big-sweep)

- iter: 44
- tag: `phase34_HARD050_bigsweep_dualseed_winner`
- date: 2026-05-10
- eval: `eval_v15direct_HARD050` (`strength <= 0.50`, 2003 intersection chips)
- one-liner: comprehensive C(9,4)=126 4-bag sweep at HARD050 reveals
  **NEW HARD WINNER = `{24_LS030_seed42 + 33D + 37E + 24_LS030_seed7}`**
  at **bit_F1 = 0.9843 / ni_FAR = 2.00%**, beating prior hard+KD winner
  (0.9689) by **+0.0154** and pure-hard 4-bag (0.9670) by **+0.0173**.

## Motivation

Phase 31b (iter 43) tested only the **directly comparable** 4-bag swap
{pure-hard ↔ hard+KD} at HARD050 and concluded "hard+KD wins by +0.0019".
This was a single-pair test, not exhaustive. Phase 34 ran the full
exhaustive sweep over 9 candidate cells (24_LS030_s42, 24_LS030_s7, 26B,
26D, 26H, 33A, 33D, 37E, 21H) → C(9,4) = 126 four-bag combinations + bag
size frontier (5/6/7-bag), all evaluated at HARD050.

Result: the **dual-24_LS030-seed strategy** (both seeds 42 and 7 in the
bag) systematically dominates. The HARD-chip specialist `24_LS030`
(single bF1 0.9767 on HARD, but ni_FAR=20.50% alone) double-votes on
hard chips when both seeds are in the bag, and the remaining two slots
(33D KD-student + 37E asymmetric label) cancel its FAR over-firing while
preserving HARD-chip strength.

## Top-10 4-bag at HARD050 (FAR ≤ 5% gate)

| rank | combo                                           |   bF1 | ni_FAR | bb     | fk     | sc     | sr  |
|-----:|-------------------------------------------------|------:|-------:|--------|--------|--------|-----|
| 1 ★  | 24_LS030_s42 + 33D + 37E + 24_LS030_s7          | 0.9843|  2.00% | 0.9517 | 0.9891 | 0.9964 | 1.0 |
| 2    | 24_LS030_s42 + 33A + 37E + 24_LS030_s7          | 0.9833|  2.00% | 0.9479 | 0.9891 | 0.9964 | 1.0 |
| 3    | 24_LS030_s42 + 26D + 33A + 24_LS030_s7          | 0.9826|  2.00% | 0.9498 | 0.9854 | 0.9952 | 1.0 |
| 4    | 24_LS030_s42 + 26H + 33D + 24_LS030_s7          | 0.9823|  2.00% | 0.9460 | 0.9882 | 0.9952 | 1.0 |
| 5    | 24_LS030_s42 + 26D + 33D + 24_LS030_s7          | 0.9821|  2.00% | 0.9479 | 0.9854 | 0.9952 | 1.0 |
| 6    | 24_LS030_s42 + 26D + 37E + 24_LS030_s7          | 0.9819|  2.00% | 0.9422 | 0.9891 | 0.9964 | 1.0 |
| 7    | 24_LS030_s42 + 33A + 33D + 24_LS030_s7          | 0.9817|  2.00% | 0.9479 | 0.9835 | 0.9952 | 1.0 |
| 8    | 24_LS030_s42 + 26D + 26H + 24_LS030_s7          | 0.9816|  2.00% | 0.9441 | 0.9872 | 0.9952 | 1.0 |
| 9    | 24_LS030_s42 + 26H + 33A + 24_LS030_s7          | 0.9814|  2.00% | 0.9422 | 0.9882 | 0.9952 | 1.0 |
| 10   | 24_LS030_s42 + 26B + 33D + 24_LS030_s7          | 0.9807|  2.00% | 0.9403 | 0.9872 | 0.9952 | 1.0 |

★ **All top-10 4-bag include BOTH 24_LS030 seeds** — dual-seed strategy
is the structural enabler at HARD eval.

## HARD050 bag-size cost-frontier

| bag size | best (FAR ≤ 5%)                                   |   bF1 | ni_FAR |
|---------:|---------------------------------------------------|------:|-------:|
| **4 ★**  | 24_LS030_s42 + 33D + 37E + 24_LS030_s7            | 0.9843|  2.00% |
| 5        | 24_LS030_s42 + 26H + 33A + 33D + 24_LS030_s7      | 0.9715|  0.00% |
| 6        | 24_LS030_s42 + 26D + 26H + 33A + 33D + 24_LS030_s7| 0.9755|  0.00% |
| 7        | 24_LS030_s42 + 26D + 26H + 33A + 33D + 21H + 24_LS030_s7 | 0.9613 | 0.00% |

★ **4-bag is the global optimum at HARD eval.** Larger bags (5/6/7)
regress on bF1 (consensus dilution: more cells with weaker HARD-chip
performance pull the threshold-2 majority away from the dual-seed
specialist signal). Trade-off: larger bags drive ni_FAR to 0.00%.

## Strict 0% FAR 4-bag at HARD050 (production safety-critical)

| 4-bag                                              |   bF1 | ni_FAR |
|----------------------------------------------------|------:|-------:|
| **hard+KD** (24_LS030_s42 + 26B + 26H + 33D)       | 0.9689|  0.00% |
| **pure-hard** NEW HEADLINE (24_LS030_s42 + 26B + 26D + 26H) | 0.9670 | 0.00% |

No FAR=0% 4-bag matches the HARD WINNER's 0.9843 — the WINNER carries
2% FAR (within 5% paper-spec gate). For deployments demanding strict 0%
FAR, hard+KD remains the recommendation.

## Dual-24_LS030-seed insight

`24_LS030` is the HARD-chip specialist:
- single bF1 0.9767 at HARD050 (highest single bF1 of all candidate cells)
- single ni_FAR 20.50% / 22.50% at FULL — FAR-fragile alone
- per-class 0.9307 / 0.9891 / 0.9892 / 0.9977 — strong on bb (0.9307)
  where most other cells collapse below 0.90 on HARD chips

When **both seeds** are in the 4-bag, the threshold-2 majority vote
double-counts the HARD-chip evidence, lifting `bb` F1 to 0.9517
(highest of all top-10). The remaining two slots (33D KD-student + 37E
asymmetric) provide diversity from non-correlated decision boundaries
to cancel FAR over-firing.

This is structurally distinct from prior 4-bag designs (iter33 g/LS
spread, iter34 KD axis, iter37 asymmetric axis, iter39 pure-hard) which
treated `24_LS030_seed42` as a single slot. Phase 34 reveals that on
HARD chips the **specialist signal benefits from dual-vote
amplification**.

## Paper §6 implication

Phase 31b (iter 43) claim: "hard+KD wins HARD by +0.0019 over pure-hard"
was a directed-comparison limited to two prior-best 4-bag compositions.
Phase 34 exhaustive sweep reframes:

- HARD eval **is** sensitive to ensemble composition (FULL eval
  saturates → all 4-bag types tie within 0.0008)
- The right axis at HARD is **specialist-amplification** (dual-seed of
  the HARD-chip dominant cell), not KD vs hard-label diversity
- KD (33D) and asymmetric (37E) still contribute as the two complementary
  slots — but the headline gain comes from dual-24_LS030-seed structure
- Prior "hard+KD wins by +0.0019" claim should be reframed: hard+KD
  beats pure-hard within the **single-seed** 4-bag family; once
  dual-seed is allowed, both are surpassed by +0.0154/+0.0173

## Sources

Each model's HARD050 prediction parquet:
- `outputs/eval_v15direct_HARD050/24_LS030_seed42/preds_chip.parquet`
- `outputs/eval_v15direct_HARD050/24_LS030_seed7/preds_chip.parquet`
- `outputs/eval_v15direct_HARD050/26B/preds_chip.parquet`
- `outputs/eval_v15direct_HARD050/26D/preds_chip.parquet`
- `outputs/eval_v15direct_HARD050/26H/preds_chip.parquet`
- `outputs/eval_v15direct_HARD050/33A/preds_chip.parquet`
- `outputs/eval_v15direct_HARD050/33D/preds_chip.parquet`
- `outputs/eval_v15direct_HARD050/37E/preds_chip.parquet`
- `outputs/eval_v15direct_HARD050/21H/preds_chip.parquet`

Phase 34 sweep summary: 126 4-bag combos + 4 bag-size frontier rows.
2003 intersection chips after merge across 9 cell parquets.
