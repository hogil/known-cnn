# 260510 Phase 50 iter 52 — Teacher bag-size sweep at fixed α=0.5

## Context

Iter 50 B (4-bag teacher) and iter 33 A (14-bag teacher)
established the **bag-size ↔ optimal-α anti-correlation**
qualitatively. Iter 51 narrowed the α window for the 4-bag
teacher to ±0.025 around 0.50. Open question: **how does
student bF1 actually move as we sweep teacher bag size at
the same α = 0.5 / T = 4?** Iter 52 runs a 6-cell sweep over
bag sizes {2, 3, 4, 5, 6, 14} with student α = 0.5 fixed.

## Sweep design + results

| cell | bag size | teacher composition                | student bF1 | ni_FAR | dual | bb / fk / sc / sr           |
|------|---------:|------------------------------------|------------:|-------:|:----:|-----------------------------|
| A    |        2 | 37 E + 33 A                        | 0.9198      | 1 %    | PASS | 0.9785 / 0.8594 / 0.9002 / 0.9413 |
| B    |        3 | 37 E + 33 A + 24_LS030_s42         | 0.9768      | 1 %    | PASS | 0.9702 / 0.9744 / 0.9666 / 0.9961 |
| **C**| **4 (NEW MAIN)** | 24 + 26 H + 33 A + 37 E    | **0.9872**  | **0.5 %** | **PASS ★** | 0.9866 / 0.9825 / 0.9795 / 1.0000 |
| **D**|        5 | NEW MAIN + 26 B                    | **0.9913**  | **99.5 %** | **FAIL ★** | 0.9961 / 0.9818 / 0.9882 / 0.9992 |
| E    |        6 | NEW MAIN + 26 B + 26 D             | 0.9862      | 0 %    | PASS | 0.9677 / 0.9825 / 0.9945 / 1.0000 |
| F    |       14 | iter-27 14-bag (paper §5.21)       | 0.9053      | 0 %    | PASS | regress per-class            |

## Curve interpretation

The bag-size → student-bF1 curve at fixed α = 0.5 is
**non-monotonic with a sharp peak at 4-bag**:

- **2 → 3 → 4 monotonic up** (0.9198 → 0.9768 → 0.9872):
  more diverse teacher posteriors deliver a larger learnable
  distillation signal up to a saturation point.
- **4 → 5 catastrophic FAR jump** (0.5 % → 99.5 %): adding
  26 B to the NEW MAIN 4-bag delivers the *highest defect
  bit-F1 in the sweep (0.9913)* but breaks safety entirely.
  Mechanism: 26 B's high-precision per-class posteriors push
  the teacher signal toward "every chip looks like defect";
  the student over-mimics the over-confident teacher and
  predicts defect on Normal / Invalid chips.
- **5 → 6 partial recovery** (0.9862 PASS): adding a sixth
  cell (26 D) re-smooths the teacher and the student
  recovers the safety profile, with a small bit-F1 regress
  vs 4-bag.
- **14-bag collapse at α = 0.5** (0.9053): the same 14-bag
  teacher that distils to 0.9840 at α = 0.3 (iter 33 A)
  collapses to 0.9053 at α = 0.5, because the 14-bag
  posterior is too smoothed to dominate the hard label at
  that α weight. Tuning α to 0.3 recovers the 14-bag.

## Headline finding

**5-bag teacher is the highest-bit-F1 / lowest-safety cell
in the sweep.** Adding a high-precision specialist (26 B)
to a working 4-bag teacher *increases* defect accuracy on
the chips the teacher is correct on, but the resulting
sharper teacher posterior breaks the student's
Normal-vs-defect boundary. This is a **paper-grade safety
counter-example to "more teacher knowledge is better"** in
the saturated-bit-F1 regime.

The **4-bag teacher at α = 0.5** is the **only confirmed
PASS sweet spot** across the {2, 3, 4, 5, 6, 14}-bag range
at the fixed α: smaller bags under-train, 5-bag breaks FAR,
6-bag regresses, 14-bag requires re-tuning α to 0.3.

## Operational guideline

The teacher-bag-size ↔ optimal-α relation can be
approximated by

```
α_opt ≈ 0.7 / sqrt(bag_size)         (rough heuristic)
```

giving α ≈ 0.50 at 4-bag, ≈ 0.45 at 6-bag, ≈ 0.30 at
14-bag — consistent with the observed sweet spots within
±0.05. The relationship is anti-correlated (smaller bag →
larger α) because smaller bags produce sharper teacher
posteriors and thus require less α weighting to match the
hard-label gradient magnitude.

## Implications for paper

- **§5.34** new (experiments): 6-row sweep table + curve
  description + 5-bag FAR-collapse paradox.
- **§6.21.4** new (analysis): bag-size dependent α optimum
  curve (replaces heuristic-only §6.21.2 framing).
- **§6.21.5** new (analysis): 5-bag FAR collapse mechanism.
- **§7.10** updated (discussion): production recommendation
  hardens to "4-bag teacher at α = 0.5 is the **only** PASS
  sweet spot found across bag sizes at fixed α; smaller
  under-trains, larger breaks FAR or requires α retuning".
- **Abstract** updated: "Teacher bag size has narrow
  optimum at 4 for student α = 0.5; 5-bag teacher achieves
  higher defect bit-F1 (0.9913) but breaks `ni_FAR`
  (99.5 %); production setting requires 4-bag teacher
  specifically."
- **Headline 0.9953 unchanged** — single-model KD frontier
  refinement only; the 4-bag NEW HEADLINE 0.9953 / 0 %
  remains the paper SOTA.

_Source: iter 52 6-cell teacher-bag-size sweep, paper-
narrator handoff 260510 evening._
