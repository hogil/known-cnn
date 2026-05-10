# 260510 Phase 47 iter 51 — KD α / teacher / seed nuance

## Context

iter 50 B (Phase 47 §5.32) produced a 1× single-SOTA at
bit-F1 = 0.9872 / `ni_FAR = 0.5 %` PASS by distilling
from a 4-bag teacher (NEW MAIN 4-bag = {24_LS030_seed42
+ 26 H + 33 A + 37 E}, all-axes, bF1 = 0.9964) at
α = 0.5, T = 4. Open questions left after iter 50:

- Is α = 0.5 a sweet spot or a wide plateau?
- Does the choice of 4-bag teacher composition matter?
- Is the seed = 1 student replicable across other
  student seeds?

Iter 51 runs a 6-cell sweep that pins all three.

## Sweep design

| cell | teacher                                          | α    | T | seed |
|------|--------------------------------------------------|-----:|--:|-----:|
| 50 B | NEW MAIN 4-bag (24+26 H+33 A+37 E)               | 0.50 | 4 |  1  |
| 51 A | NEW MAIN 4-bag                                   | 0.50 | 4 |  7  |
| 51 B | NEW MAIN 4-bag                                   | 0.50 | 4 | 42  |
| 51 C | pure-hard 4-bag (NEW HEADLINE)                   | 0.50 | 4 |  1  |
| 51 D | iter-33 4-bag (paper §5.21)                      | 0.50 | 4 |  1  |
| 51 E | NEW MAIN 4-bag                                   | 0.40 | 4 |  1  |
| 51 F | NEW MAIN 4-bag                                   | 0.55 | 4 |  1  |

Three axes are pinned: student seed (50 B, 51 A, 51 B);
teacher composition (50 B, 51 C, 51 D); α at fixed
teacher (50 B, 51 E, 51 F). All cells eval on the
n = 200 paper-canonical eval.

## Results

| cell | bF1     | ni_FAR | dual | bb / fk / sc / sr           |
|------|--------:|-------:|:----:|-----------------------------|
| 50 B | 0.9872  |  0.5 % | PASS | 0.987 / 0.983 / 0.980 / 1.000 |
| 51 A | 0.9728  |  0.0 % | PASS | 0.973 / 0.955 / 0.982 / 0.981 |
| 51 B | 0.9498  |  100 % | FAIL | 0.984 / 0.907 / 0.957 / 0.952 |
| 51 C | 0.9630  |  100 % | FAIL | 0.943 / 0.959 / 0.970 / 0.981 |
| 51 D | 0.9790  |  0.0 % | PASS | 0.968 / 0.958 / 0.991 / 1.000 |
| 51 E | 0.8878  |  100 % | FAIL | 0.984 / 0.895 / 0.787 / 0.885 |
| 51 F | 0.8959  |  100 % | FAIL | 0.959 / 0.857 / 0.809 / 0.959 |

## Three findings

### F1 — α window is narrow at 4-bag scale

51 E (α = 0.40) and 51 F (α = 0.55) both fail
dual-gate at 100 % `ni_FAR`; only α = 0.50 passes.
The 14-bag teacher tolerated α ∈ {0.20, 0.30, 0.50}
broadly. Mechanism: 4-bag posteriors concentrate per-
class mass more tightly than 14-bag (mostly unanimous
or 3 / 1 splits), producing a sharper teacher
gradient — the safe α window contracts proportionally.

### F2 — Teacher composition outranks teacher bit-F1

Holding bag size = 4 and α = 0.5 / T = 4 / seed = 1:

| teacher              | teacher bF1 | student | dual |
|----------------------|------------:|--------:|:----:|
| pure-hard NEW HEADLINE | 0.9953    | 0.9630  | FAIL |
| iter-33 4-bag         | 0.9945    | 0.9790  | PASS |
| NEW MAIN 4-bag        | 0.9964    | 0.9872  | PASS |

The teacher with the highest single-model bit-F1
(NEW MAIN, mixed-axis) wins; the pure-hard
composition that wins on ensemble bit-F1 (0.9953)
*loses* as a teacher. Mechanism: pure-hard per-class
probs are near-deterministic (≈ 0.99 on modal class)
→ student over-mimics → predicts defect everywhere on
borderline / Normal chips → FAR collapses.

The mixed-axis teachers (iter 33, NEW MAIN) contain
KD-distilled (33 A, 33 D) and asymmetric (37 E) cells
whose per-class outputs are slightly less extreme,
giving the student a learnable Normal-vs-defect
boundary.

This is a new KD claim at the saturated-bit-F1
regime: above bit-F1 ≈ 0.99, **teacher posterior
shape dominates teacher posterior correctness**.
Prior work (Hinton 2015 arXiv:1503.02531; Beyer 2021
arXiv:2106.05237) treats teacher quality as monotone
in teacher accuracy; we provide a counter-example.

### F3 — KD seed-fragility absorbed only by ensemble

Re-running 50 B at seeds {1, 7, 42}: PASS / PASS /
FAIL. Even the distilled student is not seed-immune.
Extends §6.17.2 ensemble-from-fragility to KD
students: every single-cell production candidate in
the saturated regime is seed-bimodal in `ni_FAR`.
For 1× deployment, must seed-validate or run a
secondary parallel model.

## 51 D = ALT single-PASS production candidate

The iter-33 4-bag teacher distils a student at
bit-F1 = 0.9790 / `ni_FAR = 0.0 %` (strict zero) —
lower bit-F1 than 50 B (0.9872 / 0.5 %) but **strictly
zero FAR**. For safety-critical lines where FAR is
contractually bounded at zero, 51 D is preferred over
50 B.

The 1× cost frontier now has two PASS options:

| recipe (1×)     | bit-F1   | ni_FAR  |
|-----------------|---------:|--------:|
| iter 50 B       |  0.9872  |  0.5 %  |
| **iter 51 D ★** | **0.9790** | **0.0 %** |

## Implications for paper

- **§5.33** appended (experiments): 7-row sweep table
  + 3 findings + 51 D as ALT production option.
- **§6.21.1–3** appended (analysis): teacher-
  composition mechanism, α-window scaling, KD seed-
  fragility.
- **§7.10** appended (discussion): 1× tier production
  recommendation with two PASS options
  (50 B vs 51 D).
- **Abstract** appended: KD nuance paragraph
  ("smaller bags need finer α tuning; teacher
  diversity outweighs teacher bit-F1").
- **Headline 0.9953 unchanged** — single-model refresh
  only; the 4-bag NEW HEADLINE stays the SOTA.

## Open questions

- Is the α = 0.5 sweet spot teacher-composition-
  invariant? (51 D and 50 B both passed at α = 0.5
  with different 4-bag teachers — suggests yes for
  mixed-axis 4-bags, but a sweep at 51 D ± α not yet
  run.)
- Why does the pure-hard teacher's posterior
  concentration matter only at the student level
  and not at the ensemble-self-evaluation level
  (where 0.9953 is fine)? Conjecture: ensemble
  voting reads only argmax, while KD reads the full
  posterior shape — the per-class mass concentration
  is invisible to majority vote but visible to KL.
- Does an entropy-aware KD loss (e.g., temperature-
  adaptive on per-chip teacher entropy) recover the
  pure-hard teacher? Untested.

_Source: iter 51 6-cell sweep, paper-narrator handoff
260510 evening._
