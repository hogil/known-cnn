# Phase 62 — iter 58 pure-asym teacher + circular distillation (260511)

6-cell sweep on top of 50 B testing two paper-novel directions
(pure-asymmetric 4-bag teacher; circular distillation where KD
students themselves serve as teacher) and three optimisation
hyperparameters (two-LR, mild warmup, tighter grad-clip). FULL
n = 200.

## Result matrix (6/6 complete, n = 200)

| cell | spec | bF1 | ni_FAR | per-class bb/fk/sc/sr | dual | Δ vs 50 B |
|------|------|----:|-------:|-----------------------|------|----------:|
| 58 A | pure-asym 4-bag teacher (37A+D+E+H) α=0.5 | 0.8670 | 2 % | 0.9641 / 0.7914 / 0.8314 / 0.8811 | PASS | − 0.120 |
| **58 B** | **pure-asym teacher α=0.3** | **0.9880** | **100 %** | **0.9977 / 0.9761 / 0.9785 / 1.0000** | **FAIL** | **+ 0.001 / FAR break** |
| **58 C ★** | **pure-KD teacher (33A+B+C+D, circular)** α=0.5 | **0.9310** | **0 %** | **0.9421 / 0.8870 / 0.9389 / 0.9560** | **PASS** | − 0.056 |
| 58 D | 50B + two-LR (bb 5e-5 / head 2e-4) | 0.9618 | 4 % | regress | PASS | − 0.025 |
| 58 E | 50B + warmup-epochs = 1 | 0.9869 | 54.5 % | FAR break | FAIL | bF1 ≈ |
| 58 F | 50B + grad-clip = 0.5 | 0.8971 | 0 % | regress | PASS | − 0.090 |
| **50 B** | paper KD canonical (ref) | 0.9872 | 0.5 % | 0.9866 / 0.9825 / 0.9795 / 1.0000 | PASS ★ | baseline |

## Two paper-novel findings

### 1. ★ Pure-asym teacher α = 0.3 — absolute reachable peak BUT FAR-broken (58 B)

- bit-F1 = **0.9880** (highest single-model in project, + 0.001 over 50 B)
- per-class: bb 0.9977 / fk 0.9761 / sc 0.9785 / sr 1.0000 — all near-perfect
- `ni_FAR = 100 %` — catastrophic
- **Decisive evidence for §6.21 trade-off thesis**: alternative configs
  CAN exceed 50 B on bit-F1, but only at cost of FAR collapse.
- "Honest 1× SOTA" 0.9872 / 0.5 % (50 B) is the **FAR-conforming peak**,
  NOT absolute reachable peak.
- Mechanism: pure-asymmetric teacher produces extremely sharp positive
  posteriors but lacks calibration mass on negative side (Normal /
  Invalid). Student inherits sharpness on positives and breaks
  Normal-suppression boundary.
- Removing one of the three FAR-control mechanisms (§6.19 pair-mask
  data, §6.22 KD soft-targets, §6.23 BCE + LS calibration) suffices to
  break FAR even at maximal bit-F1.

### 2. ★ Circular distillation works at 0.9310 / 0 % (58 C)

- Teacher: four prior KD students (33 A / 33 B / 33 C / 33 D) soft-
  target average
- Student passes dual gate: 0.9310 / 0 % FAR
- Paper-novel: KD chains are **feasible** (no collapse, dual-gate PASS)
- But − 0.056 weaker than NEW MAIN teacher path — information loss
  across distillation generations
- Implication: distillation chains are operationally viable but **not
  strict improvements** within the saturated 1× regime
- Connects to §6.21.1 — 4-bag teacher's α window is narrow; KD-student
  teacher widens it but at lower bit-F1 ceiling

## Four secondary findings

3. **58 A pure-asym α = 0.5 weak**: − 0.120 vs 50 B. Asymmetric-axis
   alone lacks information density of mixed-axis teacher. Confirms
   §6 recipe diversity requirement.

4. **58 D two-LR regression**: − 0.025. Backbone 5e-5 / head 2e-4
   does not compose with KD recipe — KD already shapes the late-
   epoch posterior surface; LR-split has nothing left to tune.

5. **58 E warmup = 1 FAR break**: bit-F1 OK at 0.9869 but FAR
   54.5 %. Consistent with iter 54 C where warmup = 3 broke FAR —
   even mild warmup pushes confidence past the FAR gate when KD is
   active.

6. **58 F grad-clip = 0.5 regression**: − 0.090. Over-tight
   clipping starves gradient flow during the late KD distillation
   phase; default 1.0 is the sweet spot.

## Paper sections updated

- **§5.41 Pure-asym teacher + circular distillation** (05_experiments.md):
  7-row table + 4 findings + paper claim on FAR-conforming peak.
- **§6.26 FAR-conforming SOTA vs absolute single-model peak**
  (06_analysis.md): mechanism for 58 B FAR break; three FAR-control
  mechanisms jointly necessary; circular distillation information-
  lossy.
- **§7.10.7** (07_discussion.md): production gate IS the discriminator;
  without it recipe selection collapses.
- **Abstract** small note: production-deployable peak 0.9872 (50 B) vs
  absolute reachable peak 0.9880 (58 B at FAR = 100 %).

## Headline unchanged

- 4× cost NEW HEADLINE: **0.9953 / 0 %** (4-bag pure-hard / hard + KD,
  n = 500 stabilised)
- 1× cost FAR-conforming SOTA: **0.9872 / 0.5 %** (50 B; KD α = 0.5,
  T = 4; 4-bag NEW MAIN teacher)
- 1× cost strict-zero-FAR option: **0.9843 / 0 %** (53 F pure-hard
  teacher α = 0.3) OR **0.9840 / 0 %** (33 A NEW MAIN teacher α = 0.3)
- **Absolute reachable single-model bit-F1** (no FAR gate): **0.9880**
  (58 B; pure-asymmetric 4-bag teacher α = 0.3) — NOT recommended,
  `ni_FAR = 100 %`

## Sources

- 6/6 cells FULL n = 200 evaluation completed 260511
- Recipe specs: pure-asym teacher = (37 A + 37 D + 37 E + 37 H)
  soft-target average; circular teacher = (33 A + 33 B + 33 C + 33 D)
  soft-target average; KD α / T as listed in cell column
- Eval set: chip multi-label 11-class n = 200 per class, FCM-PM
  v15direct + v14class dual gate
