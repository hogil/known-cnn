# Phase 43 — Pair-fill axis revocation (4th self-correction)

_2026-05-10. Diary entry following iter 48._

## Trigger

Phase 42 (paper §6.20) elevated pair-fill (corner vs
white-fill) to a "5th FCM-PM axis" capable of flipping
the dual-gate PASS / FAIL boundary on the LS axis. The
elevation rested on a single comparison: iter 30 D
(corner, FAIL) vs. iter 47 F (white-fill, PASS, borderline
5 % `ni_FAR`) at g = 2, LS = 0.50.

## Falsification — iter 48

Tested 4 additional corner-FAIL points with white-fill:

| cell | (g, LS)        | corner ref | white bF1 | white ni_FAR | dual |
|------|----------------|------------|----------:|-------------:|------|
| 48 A | (3, 0.40)      | 40 C FAIL  |   0.9719  |     100.00 % | FAIL |
| 48 B | (4, 0.50)      | 40 E FAIL  |   0.9396  |     100.00 % | FAIL |
| 48 C | (2, 0.45)      | 36 B FAIL  |   0.8703  |     100.00 % | FAIL |
| 48 D | (2, 0.65)      | 36 E FAIL  |   0.9345  |     100.00 % | FAIL |
| 47 F | (2, 0.50)      | 30 D FAIL  |   0.9795  |       5.00 % | PASS (borderline) |

White-fill rescues 1 of 5 corner-FAILs. Not systematic.

## Revocation

§6.20 rewritten as **"Pair-fill is hyperparameter-tunable,
not a method axis."** Restores the §5.28 5-axis ablation
reading (pair-fill in the tunable-hyperparameter bucket
alongside g, LS, cutmix-p, cutmix-rect; not an essential
method axis). §5.30 added with the iter 48 rescue table.
Abstract did not contain pair-fill elevation language —
no edit needed.

## Self-correction count

1. n = 200 → n = 500 sample-size rebuttal (§5.26 / §6.18)
2. HARD WINNER revoke (§5.27 / §6.18.x)
3. Continuous PASS region falsification (§5.29.2 / §6.20.4)
4. **Pair-fill axis revocation (this entry)**

The headline 0.9953 / 0 % FAR at n = 500 (§5.26) is
unaffected. Periphery claims are now polished.

## Lesson

Single-comparison mechanism claims (one PASS vs one FAIL)
are unsafe even when `(g, LS, seed, schedule, aug)` are
held fixed. The dual-gate threshold is at 5 % FAR; a
borderline PASS is one sample-composition perturbation
away from FAIL. Future apparent boundary flips require
≥ 3 same-direction observations before paper-text
elevation.
