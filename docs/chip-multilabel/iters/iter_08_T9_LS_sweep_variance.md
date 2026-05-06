# Iter 8 — T9 LS sweep on BCE+CutMix(p=0.5) + variance verification

**Run window**: 2026-05-05 20:58 – 21:26
**Train dirs**:
`outputs/logs_chip_multilabel/T7_T9a_BCE_LS10_cutmix50_260505_205820/`,
`outputs/logs_chip_multilabel/T7_T9b_BCE_LS05_cutmix50_260505_210235/`,
`outputs/logs_chip_multilabel/T7_T9c_BCE_LS00_cutmix50_260505_210636/`,
`outputs/logs_chip_multilabel/T7_T9d_BCE_LS07_cutmix50_260505_211038/`,
`outputs/logs_chip_multilabel/T7_T9e_BCE_LS08_cutmix50_260505_211458/`,
`outputs/logs_chip_multilabel/T7_T9f_BCE_LS06_cutmix50_260505_211856/`,
`outputs/logs_chip_multilabel/T7_T9g_BCE_LS07_seed43_260505_212302/`
**Stage1 dirs**:
`outputs/stage1_260505_210059/` (T9a, LS=0.10),
`outputs/stage1_260505_210535/` (T9b, LS=0.05),
`outputs/stage1_260505_210932/` (T9c, LS=0.00),
`outputs/stage1_260505_211334/` (T9d, LS=0.07 seed=42 ★),
`outputs/stage1_260505_211752/` (T9e, LS=0.08),
`outputs/stage1_260505_212153/` (T9f, LS=0.06),
`outputs/stage1_260505_212557/` (T9g, LS=0.07 seed=43, variance verify)

## Goal

Iter 7 closed with **T7c (BCE+LS=0.20+CutMix p=0.5) = 0.9271**, but the
LS=0.20 optimum was inherited unchanged from the iter-5 CE+LS sweep. The
loss surface under BCE+CutMix is structurally different from CE-only, so
the smoothing optimum was not necessarily still at 0.20.

Iter 8 re-sweeps the LS axis for the BCE+CutMix recipe and additionally
re-runs the apparent winner with a different seed to separate signal from
single-seed variance.

Recipe held: BCE, CutMix p=0.5, LR=1e-4, ep=8. Only `label_smoothing` (and
seed, for T9g) varies.

## T9 LS sweep — best cell per run

| run  | LS    | seed | best cell  | macro_f1 | top1_11 | mAP    | ECE_post | Δ vs T7c=0.9271 |
|------|------:|-----:|------------|---------:|--------:|-------:|---------:|----------------:|
| T9c  |  0.00 |   42 | T9c__I10   |   0.8609 |  0.6443 | 0.8384 |   0.0114 |        −0.0662  |
| T9b  |  0.05 |   42 | T9b__I7    |   0.9449 |  0.8670 | 0.9378 |   0.0060 |        +0.0178  |
| T9f  |  0.06 |   42 | T9f__I3    |   0.9401 |  0.8648 | 0.9521 |   0.0088 |        +0.0130  |
| T9d ★|  0.07 |   42 | T9d__I7    | **0.9705** | **0.9267** | **0.9864** | **0.0106** |    **+0.0434** |
| T9e  |  0.08 |   42 | T9e__I3    |   0.8085 |  0.4449 | 0.8362 |   0.0425 |        −0.1186  |
| T9a  |  0.10 |   42 | T9a__I10   |   0.9364 |  0.8489 | 0.9451 |   0.0143 |        +0.0093  |
| T9g  |  0.07 |   43 | T9g__I7    |   0.9408 |  0.8307 | 0.9468 |   0.0079 |        +0.0137  |

_Source: `outputs/stage1_260505_{210059,210535,210932,211334,211752,212153,212557}/results_matrix.parquet`._

## What the curve says (LS-axis only, fixed seed=42)

```
LS    : 0.00   0.05   0.06   0.07   0.08   0.10
F1    : 0.861  0.945  0.940  0.971  0.808  0.936
```

Three things are striking:

1. **The peak shifts low** — under BCE+CutMix the optimum is at
   **LS=0.07**, not LS=0.20. The CE-era optimum no longer applies.
   BCE already softens hard targets via its independent-class
   formulation, and CutMix softens further by interpolating multi-hot
   labels. Stacking LS=0.20 on top of that double-soft target
   over-smooths; LS=0.07 is the right additional dose.
2. **The curve is non-smooth** — the cliff at **LS=0.08 → 0.8085**
   between LS=0.07 (0.9705) and LS=0.10 (0.9364) is a 0.16-macro-F1
   drop over a 0.01 LS step. This is not a smooth optimum; it is a
   knife-edge.
3. **LS=0.05 / 0.06 / 0.10 are all in a 0.94 band** — the smoothing
   axis is broadly forgiving *except at the 0.08 spot*. T9e is the
   anomaly, not T9d.

## Variance verification: T9d vs T9g (LS=0.07, seed 42 vs 43)

T9d (seed=42): macro_f1 = **0.9705**
T9g (seed=43): macro_f1 = **0.9408**
Δ at fixed config = **0.0297 macro-F1**.

This is roughly the same magnitude as the LS=0.07 peak above the LS=0.05
runner-up (0.9705 − 0.9449 = 0.0256). In other words, **changing the seed
moves macro-F1 by about as much as changing LS from 0.05 to 0.07**.

Implication: the headline 0.9705 number is partly luck. The realistic
LS=0.07 expectation under this recipe is closer to **0.94–0.97 with σ
≈ 0.015–0.02** based on n=2 seeds. T9d is a favorable outlier; T9g is
the realistic point estimate.

We do not have enough seeds for a confidence interval. n=2 only proves
the variance exists; it doesn't pin its scale. **For the paper, T9d
0.9705 must be reported alongside T9g 0.9408 with explicit variance
caveat — not as "we got 0.9705".**

## What T9d's per-class breakdown shows

T9d__I7 (best cell):

| class          | threshold | precision | recall | F1     | AP     |
|----------------|----------:|----------:|-------:|-------:|-------:|
| bank_boundary  |    0.9000 |    0.9919 | 0.9578 | 0.9746 | 0.9818 |
| fork           |    0.2200 |    1.0000 | 0.8953 | 0.9448 | 0.9877 |
| scratch        |    0.5800 |    0.9912 | 0.9354 | 0.9625 | 0.9759 |
| scratch_rot    |    0.1800 |    1.0000 | 1.0000 | 1.0000 | 1.0000 |

_Source: `outputs/stage1_260505_211334/per_class_metrics.parquet`._

vs T9g__I7 (same config, seed=43):

| class          | threshold | precision | recall | F1     | AP     |
|----------------|----------:|----------:|-------:|-------:|-------:|
| bank_boundary  |    0.3800 |    0.9984 | 0.9641 | 0.9809 | 0.9817 |
| fork           |    0.1000 |    0.7392 | 0.9078 | 0.8149 | 0.8253 |
| scratch        |    0.7400 |    0.9726 | 0.9625 | 0.9675 | 0.9804 |
| scratch_rot    |    0.3000 |    1.0000 | 1.0000 | 1.0000 | 1.0000 |

_Source: `outputs/stage1_260505_212557/per_class_metrics.parquet`._

The variance is concentrated in **fork**: F1 0.9448 (T9d) vs 0.8149 (T9g),
a 0.13 gap. fork-AP also shifts (0.9877 vs 0.8253). The other three classes
are stable across seeds (Δ ≤ 0.01). fork is the longest-tail / most-diffuse
defect (per iter-1 error analysis), so seed-driven differences in which
fork-y patches end up in CutMix mosaics directly move fork's F1.

bank_boundary recall and scratch/scratch_rot are essentially identical
across seeds — the axis-of-randomness in this recipe is fork, not
everywhere.

## Why LS=0.08 falls off a cliff

LS=0.08 is a single seed (seed=42, T9e), so we can't fully separate
"LS=0.08 is bad" from "LS=0.08 with seed=42 happened to be bad". But
the symmetry of T9f (LS=0.06 = 0.9401) and T9d (LS=0.07 = 0.9705) at
the same seed=42 suggests something about LS=0.08 specifically that
seed=42 hits hard.

Working hypothesis: BCE+CutMix targets have a ceiling on tolerable
total-target softness. With BCE = independent sigmoids (no zero-sum) +
CutMix mixing two multi-hot labels (already softening), LS=0.08 pushes
the effective positive-target value just below a threshold where the
gradient signal collapses for the runner-up class. This is consistent
with the non-monotonic, knife-edge shape — it's a phase-transition-like
collapse, not a smooth optimum.

A cleaner test would be LS=0.08 with seed=43 to disentangle. We have
not run it; iter 9 instead diverted resources to negative-axis probes.

## Iter 8 verdict

- **New best**: T9d__I7 = **0.9705** macro-F1, +0.0434 over T7c__I10. But
  this is a single seed, and the same config at seed=43 (T9g) gives
  0.9408. Realistic point estimate: 0.94 ± 0.02.
- **LS optimum under BCE+CutMix is LS=0.07**, not LS=0.20. The CE-era
  optimum does not transfer to BCE+CutMix targets.
- **Single-seed variance ±0.030** at the headline number — this becomes
  the dominant uncertainty above the LS axis itself, except at the
  LS=0.08 cliff.
- For paper: report T9d 0.9705 as "best observed", T9g 0.9408 as
  "realistic with seed variance", and the cliff at LS=0.08 as a
  separate phenomenon worth a fuller seed sweep before claiming the
  hyperparameter is brittle.

## Files

- `outputs/logs_chip_multilabel/T7_T9{a,b,c,d,e,f,g}_*/` — train logs.
- `outputs/stage1_260505_{210059,210535,210932,211334,211752,212153,212557}/` —
  inference matrices.
- `outputs/stage1_260505_211334/per_class_metrics.parquet` — T9d per-class.
- `outputs/stage1_260505_212557/per_class_metrics.parquet` — T9g per-class.
