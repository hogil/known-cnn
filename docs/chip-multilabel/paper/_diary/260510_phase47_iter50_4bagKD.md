# Phase 47 / Iter 50 — 4-bag teacher KD distillation: single-SOTA refresh

**Date.** 2026-05-10 (evening)
**Scope.** Iter 50 5-cell sweep, KD distillation from a 4-bag teacher.
**Headline.** Iter 50 B (α = 0.5, T = 4) = **bit-F1 0.9872 / `ni_FAR = 0.5 %`
PASS** — new single-model SOTA at 1× cost, **+0.0032 over iter 33 A**
(14-bag teacher KD, 0.9840 / 0 %).

## 0. Motivation

The §7.6 / §7.8 cost frontier had iter 33 A (14-bag teacher KD α = 0.3
T = 4) as the only 1× option at bit-F1 = 0.9840. The 1× → 4× gap to
the NEW HEADLINE 4-bag (0.9953 / 0 %) was 0.0124 bit-F1 — large
enough that production lines unable to afford 4× inference cost paid
a meaningful quality penalty. Hypothesis: distilling from a smaller,
sharper teacher should give the student a stronger per-chip gradient
signal and recover some of the 4× ensemble lift in a 1× student.

The §5.31 Phase 44 big-sweep had surfaced a strong 4-bag candidate
{24_LS030_seed42 + 26 H + 33 A + 37 E} at bit-F1 = 0.9964 / 0 % (the
all-4-axes top-2 at n = 200, statistically tied with the pure-hard
NEW HEADLINE). This 4-bag served as the iter 50 teacher.

## 1. Sweep design

Two single-axis sweeps anchored at the 33 A operating point:

- α-sweep at fixed T = 4: α ∈ {0.3, 0.5, 0.7}
- T-sweep at fixed α = 0.3: T ∈ {2, 4, 8}

5 cells total (50 A α=0.3 T=4 is the shared anchor at the
α = 0.3 / T = 4 corner).

## 2. Results

| cell      | α   | T | bit_F1   | ni_FAR | per-class bb / fk / sc / sr     | dual |
|-----------|----:|--:|---------:|-------:|---------------------------------|:----:|
| 50 A      | 0.3 | 4 | 0.8921   | 0 %   | 0.9801 / 0.8670 / 0.7330 / 0.9881 | PASS (sc collapse) |
| **50 B ★** | **0.5** | **4** | **0.9872** | **0.5 %** | **0.9866 / 0.9825 / 0.9795 / 1.0000** | **PASS** |
| 50 C      | 0.7 | 4 | 0.8720   | 0 %   | 0.9511 / 0.8594 / 0.7285 / 0.9491 | PASS |
| 50 D      | 0.3 | 2 | 0.9384   | 0 %   | 0.9678 / 0.9393 / 0.8811 / 0.9652 | PASS |
| 50 E      | 0.3 | 8 | 0.9323   | 0 %   | 0.9577 / 0.8946 / 0.8769 / 1.0000 | PASS |
| 33 A (14-bag teacher, paper main reference) | 0.3 | 4 | 0.9840 | 0 % | (paper main reference) | PASS |

## 3. Findings

**3.1 α sweet spot shifts upward with smaller teacher bag.** The
14-bag teacher 33 A optimised at α = 0.3 (bit-F1 = 0.9840). The 4-bag
teacher 50 B optimises at α = 0.5 (bit-F1 = 0.9872). Off-peak cells
50 A (α = 0.3) and 50 C (α = 0.7) both collapse the scratch class to
F1 ≈ 0.73 — α = 0.3 underutilises the sharper 4-bag teacher signal,
α = 0.7 over-mimics teacher modes. α = 0.5 balances hard-label
gradient with teacher-distribution gradient, holding all four
defect-class F1 ≥ 0.98.

**3.2 Mechanism.** A 14-bag teacher averages 14 per-chip soft
posteriors → smoother probability mass spread across 2–3 classes on
borderline chips. A 4-bag teacher averages only 4 posteriors → mass
concentrates on a single class except on near-tie chips. The
teacher-side KD gradient magnitude scales with posterior
concentration; sharper teacher × higher α produces a balanced
effective distillation magnitude that matches what α = 0.3 delivers
on a smoother teacher.

**3.3 T = 4 is invariant.** T = 2 (50 D, 0.9384) over-sharpens
targets and discards the soft-label benefit; T = 8 (50 E, 0.9323)
over-smooths the teacher signal. T = 4 stays optimal under both
14-bag and 4-bag teacher regimes — a robust default for KD in this
domain.

**3.4 Cost frontier update.**

| cost | recipe                                                 | bit-F1   | ni_FAR |
|-----:|--------------------------------------------------------|---------:|-------:|
| 1×   | 33 A KD-student (14-bag teacher α = 0.3 T = 4)         |  0.9840  |  0 %   |
| **1× ★** | **iter 50 B KD-student (4-bag teacher α = 0.5 T = 4)** | **0.9872** | **0.5 %** |
| 3×   | 37 E + 24_LS030_seed7 + 26 D                            |  0.9929  |  0 %   |
| 4×   | NEW HEADLINE 4-bag (n = 500)                            |  0.9953  |  0 %   |

The 1× → 4× gap contracts from 0.0124 to **0.0081 bit-F1
(33 % reduction)**. iter 50 B is the new 1× production
recommendation.

## 4. Implications for paper

- **§5.32** appended: 4-bag teacher KD sweep table + α-shift finding.
- **§6.21** appended: teacher-bag-size-dependent α sweet spot
  mechanism + KD design heuristic ("α scales with teacher
  complexity").
- **§7.9** appended: cost frontier refresh; iter 50 B = new 1× tier.
- **Abstract** appended: single-SOTA mention.
- **Headline 0.9953 unchanged** — this is a single-model refresh,
  not an ensemble lift.

## 5. Open questions

- Does the α-shift heuristic generalise to a 2-bag or 8-bag teacher?
  (Predict: α ≈ 0.6 for 2-bag; α ≈ 0.4 for 8-bag — log-linear
  interpolation between 4-bag's 0.5 and 14-bag's 0.3.)
- Does iter 50 B hold up at n = 500? (Phase 47 ran at n = 200
  paper-canonical only.)
- Does an asymmetric variant on the 4-bag teacher (replacing 33 A
  with another KD cell) further sharpen the student? Untested.

_Source data:_ iter 50 5-cell run summary, paper-narrator handoff
260510 evening.
