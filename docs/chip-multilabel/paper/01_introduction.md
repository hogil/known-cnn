# 1. Introduction

Wafer fail-bit maps frequently contain *more than one* defect signature on
a single 200×200 chip patch: a `bank_boundary` ring around an edge can
co-occur with a vertical `fork` stripe; a horizontal `scratch` and its
rotated variant `scratch_rot` are visually distinct but spatially adjacent;
a chip can be `Normal` (no defect at all) or `Invalid` (white background
with the orange QC border). Industrial defect-classification CNNs are
typically trained **single-label** (one CE softmax over K classes), because
single-label labels are cheaper to obtain and the per-class precision is
required for downstream automation.

This paper addresses a concrete instance of the resulting mismatch.
We have a chip-level CNN, `chip5_round4_v14_260505_061558_running`,
trained single-label cross-entropy over 5 classes
(`bank_boundary`, `fork`, `scratch`, `scratch_rot`, `invalid_main`)
on 327 chips. We wish to deploy the same checkpoint on a multi-label
benchmark of 2200 chips covering 11 logical classes (4 single defects,
5 two-defect combos, `Normal`, `Invalid`; the `scratch+scratch_rot`
combo is excluded as ill-defined). The naive readout — argmax over
softmax — yields macro-F1 = **0.7302**, with `Normal` and combo classes
essentially never recovered.

**Research question.** How far can such a single-label model be pushed
on a multi-label benchmark, *without* re-collecting labels and *without*
augmentations that violate the class taxonomy (specifically, rotation
that conflates `scratch` with `scratch_rot`)?

**Contributions.**

1. **Inference-side calibration.** We characterise a sequence of
   training-free decoders (I0–I10) that climb from 0.7302 macro-F1 to
   0.8542 on a *frozen* backbone. The single biggest jump (+0.1142) is
   per-class F1-max thresholds (Lipton et al. 2014, arXiv:1402.1892);
   the second is an entropy-based `Normal` gate that gives a previously
   unsupervised class an explicit decoding rule. We hypothesise that
   single-label CE collapses runner-up logits, and that an aggressive
   F1-max threshold sweep recovers them.
2. **Training-side intervention.** Among CE+LS (T1, Müller et al. 2019,
   arXiv:1906.02629), ASL (T4, Ridnik et al. 2021, arXiv:2009.14119),
   BCE (T5), and BCE→ASL (T6), only **T1** improves on the frozen
   baseline; the multi-label-native losses regress in our small-data
   strong-pretrain regime. We then sweep T1's single hyperparameter
   (label-smoothing strength α) on a fixed grid (Phase A1) and observe
   a sharp non-monotonic peak at α=0.20, lifting macro-F1 to **0.9268**.
3. **Negative results.** TTA (test-time rotation), min-floor thresholds,
   per-class temperature scaling alone, BCE, ASL, and BCE→ASL all
   regress against equally-priced alternatives. We document each as a
   first-class ablation rather than dropping it from the writeup.

**Hypothesis-driven iteration.** All five iterations were sequenced
under a strict GPU=1-job rule and a coordinate-descent discipline:
each iter starts with the prior best cell, names the failure mode
the next change is intended to address, then verifies on the same
2200-chip eval set. We treat this protocol itself as a methodological
contribution and describe it in §8.

**Section roadmap.** §2 surveys related work on multi-label decoding
from single-label models, F1 thresholding, and label smoothing.
§3 details the synthetic chip dataset, eval set construction, and
sister-repo data pipeline. §4 formalises every inference and training
variant. §5 reports the iter-by-iter narrative and numbers. §6
analyses the LS curve and the entropy gate's regime change. §7
discusses why "obvious" multi-label losses lost to a one-line CE+LS
modification. §8 documents the iteration protocol. §9 concludes
and lists Phase B–F sweeps that remain.

## 1.1 Motivation: why pixel-level α-blend mixers fail in the palette-discrete regime

_Added 2026-05-09 (iter 28 / iter 29 evidence)._

A second motivation for our **Full-Cover Mixup with Pair Mask**
(FCM-PM, §4.6) is now available as direct empirical evidence
rather than a-priori reasoning. Mixup (Zhang et al. 2018,
arXiv:1710.09412) is the canonical pixel-level α-blend mixer:
two chips are linearly interpolated as
`x_mix = α x_A + (1−α) x_B` with a soft `λ`-weighted target. We
swept Mixup's single hyperparameter α over the full
operating range used in the natural-image literature
(α ∈ {0.1, 0.2, 0.4, 0.8, 1.0, 2.0}, including the Zhang 2018
default α = 0.2) and **all six variants exhibit `ni_FAR = 100 %`
on the v15 (production-realistic) eval set** (§5.18.1). The α = 0.2
variant — which is the field default — additionally retains a
non-trivial v14 (in-distribution) `ni_FAR = 5.0 %`.

The mechanism is direct: a pixel α-blend with α ∈ (0, 1) produces
**non-palette mid-grade values** at every blended pixel. Our chip
images are 8-bit palette-discrete (5 grades + background), and the
chip-CNN has been trained exclusively on palette-valid pixel
distributions. Mixup's training-time fabrication of mid-grade
values therefore creates a **synthetic data manifold disjoint from
the deployment manifold**: the model overfits to the synthetic
mid-grade noise, generalises arbitrarily on real palette-discrete
Normal/OOD chips, and over-fires every defect head with `ni_FAR = 100 %`.

This is not a hyperparameter-tuning failure. The α sweep covers
0.1×, 1×, 2×, 4×, 5×, and 10× the Zhang 2018 default; none
recover. The result is a **palette-discreteness violation** that
mixers must respect, motivating the *region paste* mechanism of
CutMix (Yun et al. 2019, arXiv:1905.04899) and our four-design
extension FCM-PM (§4.6 / §4.7), where every blended pixel is
either pure-A or pure-B and the palette discreteness is preserved
by construction.

## 1.2 Motivation: why each of the FCM-PM four designs is necessary

_Added 2026-05-09 (iter 29 component-decomposition evidence)._

The FCM-PM recipe (§4.6) is the conjunction of four design
decisions: (i) **region paste** (no α-blend, palette preserved);
(ii) **full coverage** via complementary group masks (no
information loss); (iii) **pair mask** for grounded single-class
supervision; (iv) **hard label** (union target, no λ-mix). The
iter 29 ablation isolates the label and spatial axes into a
**2 × 3 component matrix** (§5.18.2) and shows that **dropping any
single design produces a broken configuration**. Ablating *region
paste* alone reduces FCM-PM to standard λ-mixup and reproduces the
§1.1 catastrophic failure (Mixup variants, all six v15 `ni_FAR =
100 %`). Ablating *full coverage* (cell 21C, std box-cut + hard
label) yields v15 broken `ni_FAR`. Ablating *pair mask* alone
(cell 29C, complement region paste + hard label, no mask) yields
bit-F1 = 0.92 and v15 `ni_FAR = 100 %`. Ablating *hard label*
alone (cell 29B, complement region paste + pair mask + λ-soft
label) **maximises bit-F1 to 0.99 but drives v15 `ni_FAR = 100 %`** —
the most surprising outcome of the entire ablation, formalising a
recall–FAR trade-off that the soft-label literature has not
quantified for our regime. **Only the conjunction of all four
designs (cell 21E, FCM-PM) clears both gates simultaneously**
(bit-F1 ≈ 0.97, v15 `ni_FAR` pass).

This component-decomposition evidence (six cells, four ablation
axes, one winner) is the §5.18 paper-main ablation and the
strongest necessity argument we have for each design choice.
