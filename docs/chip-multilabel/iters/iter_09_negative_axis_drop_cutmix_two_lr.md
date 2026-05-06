# Iter 9 — Negative axes: drop_path / cutmix-rect / two-LR

**Run window**: 2026-05-05 21:31 – 21:50
**Train dirs**:
`outputs/logs_chip_multilabel/T7_T10a_BCE_LS07_dp05_s42_260505_213119/`,
`outputs/logs_chip_multilabel/T7_T10b_BCE_LS07_dp05_s43_260505_213522/`,
`outputs/logs_chip_multilabel/T7_T11a_cutmixrect25_LS07_s42_260505_213927/`,
`outputs/logs_chip_multilabel/T7_T12a_2LR_LS07_s42_260505_214338/`
**Stage1 dirs**:
`outputs/stage1_260505_213423/` (T10a, drop_path 0.05, seed=42),
`outputs/stage1_260505_213817/` (T10b, drop_path 0.05, seed=43),
`outputs/stage1_260505_214222/` (T11a, cutmix-rect 0.25, seed=42),
`outputs/stage1_260505_214634/` (T12a, two-LR backbone/head, seed=42)

## Goal

Iter 8 placed the LS=0.07 optimum but also exposed a single-seed variance
of ±0.030 at the headline cell. Iter 9 probes three orthogonal axes
**on top of** the iter-8 BCE+LS=0.07+CutMix(p=0.5) recipe to test whether
any of them can lift the realistic baseline (T9g = 0.9408) above the
seed-noise floor:

1. **T10 — drop_path** (stochastic depth on backbone), regularizer hint
   from anomaly-detection / large-batch literature. Two seeds (42, 43)
   to control for the iter-8 variance finding.
2. **T11 — cutmix-rect** (rectangular mask cuts, vs the default square
   patch). Tests whether the CutMix mask shape carries signal vs being
   essentially geometry-free at the chip-grid scale.
3. **T12 — two-LR** (lower LR for backbone, default LR for head). Anomaly-
   detection BKM transfer; tests whether selectively-frozen-ish backbone
   helps preserve TAPT init while still letting the new head adapt to
   BCE+CutMix targets.

All four runs hold: BCE, LS=0.07, CutMix p=0.5, ep=8, LR=1e-4 (where
applicable). Only the listed axis varies per run.

## Headline table

Baseline reference: **T9g (LS=0.07, seed=43) = 0.9408** (realistic
point) and **T9d (LS=0.07, seed=42) = 0.9705** (favorable outlier).

| run  | axis change           | seed | best cell  | macro_f1 | top1_11 | mAP    | ECE_post | Δ vs T9d (0.9705) |
|------|-----------------------|-----:|------------|---------:|--------:|-------:|---------:|------------------:|
| T10a | drop_path 0.05        |   42 | T10a__I3   |   0.9160 |  0.7335 | 0.8870 |   0.0063 |          **−0.0545** |
| T10b | drop_path 0.05        |   43 | T10b__I11  |   0.8918 |  0.7511 | 0.8997 |   0.0269 |          **−0.0787** |
| T11a | cutmix-rect 0.25      |   42 | T11a__I7   |   0.8646 |  0.6551 | 0.8980 |   0.0175 |          **−0.1059** |
| T12a | two-LR backbone/head  |   42 | T12a__I10  |   0.8862 |  0.6511 | 0.8718 |   0.0203 |          **−0.0843** |

_Source: `outputs/stage1_260505_{213423,213817,214222,214634}/results_matrix.parquet`._

All four runs **regress**. The three axes are atomic-failed.

## T10 — drop_path 0.05 (−0.054 / −0.049 vs T9d)

drop_path adds stochastic depth to backbone blocks. Standard intuition: it
regularizes deep nets in long training runs.

Two seeds, both worse than the corresponding LS=0.07 baseline:

|         | T9d (seed=42, no dp) | T10a (seed=42, dp=0.05) | Δ        |
|---------|---------------------:|------------------------:|---------:|
| macro_f1|              0.9705  |                  0.9160 |  −0.0545 |

|         | T9g (seed=43, no dp) | T10b (seed=43, dp=0.05) | Δ        |
|---------|---------------------:|------------------------:|---------:|
| macro_f1|              0.9408  |                  0.8918 |  −0.0490 |

Diagnosis is **the same Phase F (iter 7) lesson**: drop_path is a
long-training regularizer, and our 8-epoch budget on a TAPT-initialized
backbone doesn't have enough effective steps for the stochastic-depth
expectation to settle. Each epoch's pass through a partially-dropped
network is essentially noise without enough averaging to recover the
clean signal at convergence. Same structural-mismatch pattern as
warmup/EMA.

The fact that **both seeds lose ~0.05** rules out "drop_path happened to
be unlucky on seed=42". The axis itself doesn't transfer to this regime.

## T11 — cutmix-rect 0.25 (−0.106 vs T9d)

Replaces the default square CutMix patch with a rectangular mask
(aspect-ratio-aware) and reduces the cutmix ratio to 0.25 (vs the
T9-line default of 0.5).

Result: **0.8646** — the worst of the four iter-9 runs. Two simultaneous
changes (rect mask, 0.5→0.25 ratio) confound which one hurts, but
the dominant suspect is the **0.5 → 0.25 ratio**: iter 7's CutMix-p
sweep already showed p=0.3 = 0.8626 vs p=0.5 = 0.9271. T11a at 0.8646
lands almost exactly on top of the iter-7 p=0.3 number, which strongly
suggests:

- The rect-vs-square shape contributes ~zero signal (chip-grid CutMix
  is essentially geometry-free at this resolution — the model sees a
  patch boundary and learns whatever the multi-hot label says).
- The 0.5 → 0.25 ratio reduction is the active ingredient and it's a
  monotonic regression, exactly tracking the iter-7 CutMix-p curve.

**Interpretation**: rect-vs-square is not a real axis at this
resolution; the 0.5 → 0.25 change is the iter-7 result re-confirmed.
We do not get a separate iter-9 lesson from this run — the CutMix
fraction's 0.5-peak is already known.

## T12 — two-LR backbone/head (−0.084 vs T9d)

Splits LR: lower LR for the backbone (TAPT-init we want to preserve),
default LR for the new classification head (random init, needs to
adapt to BCE+CutMix multi-hot targets).

Result: **0.8862** vs T9d 0.9705. Diagnosis:

1. With BCE + CutMix soft targets, the backbone *does* need to update
   its representation — softmax-pretrained features over-emphasize
   single-class winners, and BCE+CutMix asks for sharper independent
   discrimination per class. A lowered backbone LR starves this update.
2. The 8-epoch budget compounds the problem: at LR_backbone < 1e-4,
   8 epochs is not enough effective movement.
3. Two-LR is again a long-training regime BKM (often paired with
   cosine schedules and 50+ epochs). Same structural-mismatch problem
   as warmup/EMA/drop_path.

Top1_11 also drops from 0.9267 (T9d) to 0.6511, a 0.27 collapse — much
larger than macro-F1's 0.084 — which confirms it's the *combo*
predictions that suffer most, exactly the regime that benefits from
unrestricted backbone updates under CutMix multi-hot targets.

## Cross-run pattern: "the same structural mismatch"

iter 7 atomic-failed warmup and EMA on the same diagnosis: both are
long-training-regime BKMs that don't transfer to small-data + TAPT
init + 8-epoch budgets. **Iter 9 adds three more to the same list**:
drop_path, two-LR, and (implicitly) cutmix-rect-with-low-ratio.

The growing pattern: anything that **needs many effective steps to
average out / stabilize / recover from initial-condition noise** loses
in this regime. The 8-epoch + small-data + TAPT setup has a narrow
window of techniques that transfer cleanly, and iter-7/8 already found
most of them (BCE + LS≈0.07 + CutMix p=0.5).

## Iter 9 verdict — paper-relevant negative result

Three more negative results, all consistent with the iter-7 "structural
mismatch" diagnosis:

| axis              | best Δ | seeds | verdict                                                        |
|-------------------|-------:|------:|----------------------------------------------------------------|
| drop_path 0.05    |  −0.05 |   2   | regularizer-needs-many-steps, same as warmup/EMA               |
| cutmix-rect 0.25  |  −0.11 |   1   | confounded with 0.5→0.25 ratio drop; rect shape adds no signal |
| two-LR            |  −0.08 |   1   | starves backbone of BCE+CutMix-driven updates                  |

Combined with iter 7, the **negative-axis catalogue** for the chip-
multi-label benchmark now contains: TTA(I5), ASL(T4), BCE-only(T5),
BCE→ASL(T6), min-floor(I6), temperature-only(I9), warmup(F1), EMA(F2),
I11 pair-aware threshold, CutMix p=0.7(T7d), drop_path(T10a/b),
cutmix-rect(T11a), two-LR(T12a). All atomic-failed, all explained by
the same small-data + warm-init + few-step regime story.

This catalogue is itself a paper contribution: it tells a future
practitioner *what not to try* in this regime, which is the most
consistently useful kind of negative result.

## Files

- `outputs/logs_chip_multilabel/T7_T10a_BCE_LS07_dp05_s42_260505_213119/`
- `outputs/logs_chip_multilabel/T7_T10b_BCE_LS07_dp05_s43_260505_213522/`
- `outputs/logs_chip_multilabel/T7_T11a_cutmixrect25_LS07_s42_260505_213927/`
- `outputs/logs_chip_multilabel/T7_T12a_2LR_LS07_s42_260505_214338/`
- `outputs/stage1_260505_213423/` — T10a inference matrix.
- `outputs/stage1_260505_213817/` — T10b inference matrix.
- `outputs/stage1_260505_214222/` — T11a inference matrix.
- `outputs/stage1_260505_214634/` — T12a inference matrix.
