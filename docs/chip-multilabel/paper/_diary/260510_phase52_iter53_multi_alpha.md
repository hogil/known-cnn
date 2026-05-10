# 260510 Phase 52 — iter 53 multi-teacher fusion + α rescue

## Context

After §5.32–§5.34 stabilised the 1× cost tier around the
4-bag teacher with bag-size-dependent α, two open questions
remained:

1. **Does fusing two competent 4-bag teachers help?** —
   textbook KD literature (arXiv:1503.02531, arXiv:2106.05237)
   implicitly endorses teacher averaging; our setting hadn't
   tested it directly.

2. **Was iter 51 C's "pure-hard teacher fails" claim a
   teacher property or a (teacher, α) interaction?** — at
   α = 0.5 the pure-hard 4-bag teacher had failed dual-gate;
   we had not swept α for the pure-hard composition.

iter 53 sweeps both axes in 6 cells (FULL n = 200, single
seed = 1, T = 4 fixed).

## Result

| cell | spec | bF1 | ni_FAR | dual |
|------|------|----:|-------:|:----:|
| A | multi-teacher avg (NEW MAIN ⊕ iter 33) α = 0.5 | 0.8986 | 100 % | FAIL |
| B | multi-teacher avg (NEW MAIN ⊕ pure-hard) α = 0.5 | 0.9524 | 100 % | FAIL |
| C | multi-teacher avg (all 3 4-bag teachers) α = 0.5 | 0.9268 | 0 % | weak PASS |
| D | iter-33 4-bag α = 0.3 | 0.9785 | 3 % | PASS |
| E | iter-33 4-bag α = 0.7 | 0.9825 | 0 % | PASS |
| **F** | **pure-hard 4-bag α = 0.3** ★ | **0.9843** | **0 %** | **PASS** |

## Findings

**Finding 1 — Multi-teacher fusion fails (counter-textbook).**
A and B both FAIL with `ni_FAR = 100 %`; even the all-3
average (C) only weakly passes at 0.9268 (−0.060 vs single-
best teacher). Mechanism: at saturated bit-F1 each teacher
is already at ≥ 0.9945, so the residual disagreement chips
are *genuinely hard* (not noise). Averaging two sharp-but-
different posteriors flattens the KD target on disagreement
chips; the student's hard-label gradient toward GT defects
overshoots without the diluted KD counter-evidence, and the
student over-predicts defects on Normal / Invalid chips at
deployment. **Single-best-teacher beats multi-teacher
average in this regime.**

**Finding 2 — Pure-hard teacher rescue at α = 0.3.**
The pure-hard 4-bag teacher had failed at α = 0.5 (51 C:
0.9630 / 100 %); at α = 0.3 it reaches **0.9843 / 0 % PASS**.
The previous categorical "pure-hard fails" claim is *partial*:
the failure was a (teacher sharpness × α) mismatch.
Mechanism: pure-hard per-class posteriors are sharper (≈ 0.99
on the modal class) than NEW MAIN's mixed-axis composition
(KD + asymmetric + LS axes). At α = 0.5 the student
over-mimics the over-sharp teacher; at α = 0.3 the
hard-label weight (1 − α = 0.7) balances the over-sharpness.

**Finding 3 — iter-33 teacher is α-robust.**
At α ∈ {0.3, 0.5, 0.7} the iter-33 teacher gives PASS at
0.9785 / 0.9790 / 0.9825 (FAR ≤ 3 %). The iter-33 4-bag
includes mid-LS axes (paper §5.21) whose per-class posteriors
sit between NEW MAIN's mixed composition and the pure-hard
extremes, giving the student a wider safe α window.

## Refinement of §6.21.2

The bag-size ↔ α anti-correlation is **a function of teacher
per-class posterior sharpness**, of which bag size is one
driver (small bag → sharper) but pure-hard composition is
another (already sharp at any bag size ≤ 4). Pure-hard 4-bag
needs α = 0.3 like 14-bag needs α = 0.3 — same α but
distinct mechanism (over-sharp vs over-smooth).

## 1× tier expansion

The 1× cost production tier now offers **three operating
points**: iter 50 B (0.9872 / 0.5 %), iter 51 D
(0.9790 / 0 %), iter 53 F (0.9843 / 0 %). 53 F is the new
strict-zero-FAR ≥ 0.98 bit-F1 pareto point.

## Sources

- Run dir: `outputs/phase_52_iter53_*` (6 cells)
- Paper edits: §5.35 (05_experiments), §6.21.1 partial
  revocation + §6.21.2 refinement + §6.21.6 new
  (06_analysis), §7.10.2 new (07_discussion), abstract
  Phase 52 paragraph appended
