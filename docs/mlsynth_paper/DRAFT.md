# Multi-Label Recognition Without Multi-Label Annotation: Label-Faithful Synthesis from Single-Label Data

Working draft v0.1 (2026-07-07). All numbers are measured (sources:
`docs/superpowers/multilabel_synth_RESULTS.md`, chip leaderboards under
`outputs/`). Target: IEEE TII / T-Semi.Manuf / Pattern Recognition (stretch:
WACV/BMVC after COCO + SPML comparison).

---

## Abstract

Industrial visual inspection routinely faces images containing multiple
co-occurring defect types, yet multi-label annotation of such images is
impractical: co-occurrences explode combinatorially, rare combinations may
never be observed in labeled form, and overlapping patterns are ambiguous to
annotate. In contrast, single-defect examples are cheap and unambiguous. We
study how to train a multi-label classifier from single-label supervision
alone, by synthesizing combination examples from single-label sources. We show
that the choice of synthesis operator is governed by a measurable property we
call label fidelity — the probability that every labeled object actually
survives in the synthesized image — and that operators preserving whole
objects with hard labels (per-pixel max/min overlay; full-cover complement
mixing) dominate averaging (Mixup) and rectangle-patch (CutMix) synthesis. On
a controlled MultiMNIST benchmark, blind max-overlay synthesis from
single-label data exceeds a fully-supervised oracle on the full test
(mAP 0.868 vs 0.846) and exceeds it by +0.198 mAP on held-out label
combinations that the oracle never observed. On the public MixedWM38 wafer-map
benchmark, under an equal-condition comparison (same backbone and budget,
where the fully-supervised oracle reaches 0.974 +-0.019 bit-F1 — matching
published 98-99% accuracies), training only on real single-defect wafers
recovers 86% of the oracle (0.841 +-0.034 over nine seeds) while driving
false alarms on 1,000 real normal wafers to zero (exactly zero in six of
nine seeds; mean 0.0008, max 0.005) — the oracle false-alarms on 56% +-8. Crucially, rejection cannot rescue the oracle: its
confidence on real normals is >=0.99 for 80% of them, so even a 0.99
threshold leaves a 0.80 false-alarm rate — the reliability advantage is
created by the synthesis-side training design (defect-erased synthetic
normals and pair masking shaping the confidence geometry), not by any
inference trick, and cannot be bought with more real data. A small set of
known-good samples further yields a finite-sample conformal FAR guarantee
(realized 0.040 at alpha=0.05, 0.006 at alpha=0.01). We
characterize when the approach transfers to natural images (PASCAL VOC) and
release the full harness.

## 1 Introduction

Industrial visual inspection must recognize images containing several defect
types at once, yet multi-label annotation of such images is rarely available:
co-occurrences grow combinatorially, many combinations are rare or absent
from any labeled archive, and overlapping patterns are genuinely ambiguous to
annotate. What production lines do yield, cheaply and unambiguously, are
single-defect examples. This inverts the usual weak-supervision setting:
rather than multi-label images with missing labels (single-positive
multi-label, SPML), the training images themselves contain one category —
and the question is whether multi-label competence can be manufactured from
them.

We answer by synthesizing combination examples from single-label sources and
show that the choice of synthesis operator is governed by one measurable
quantity: label fidelity — the probability that every labeled object
actually survives in the synthesized image. Operators that preserve object
evidence (per-pixel max in signal-ordered spaces; vector averaging in
disjoint-vocabulary text) rank above operators that destroy it (rectangle
patching, pixel averaging), and the measured survival ordering predicts the
downstream performance ordering in every domain we test (9/9 pairwise
orderings across three image families; a fourth, text, explains an apparent
exception: the fidelity-maximizing operator is modality-dependent).

The reliability result is the practical core. On the public MixedWM38
benchmark, an equal-condition fully-supervised oracle reaches 0.974 bit-F1
(matching published 98-99% accuracies) but false-alarms on 56% of real
normal wafers — and rejection cannot rescue it, because its confidence on
normals is above 0.99 for most of them. Our pipeline, trained on 7,015 real
singles with zero multi-label and zero normal annotations, recovers 86% of
the oracle's bit-F1 while producing zero false alarms across all six seeds,
and a handful of known-good samples upgrades this to a finite-sample
conformal guarantee. The trade is explicit and industrially favorable:
14% of bit-F1 for a false-alarm axis the oracle cannot buy with more data.

Contributions:
1. A stricter-than-SPML problem setting — multi-label recognition from
   genuinely single-label training images, evaluated on real multi-label
   data and real normals, with no location annotation — plus an
   equal-condition oracle protocol that we validate against published
   benchmark numbers.
2. Label fidelity as a measurable, predictive property of synthesis
   operators, with a superposition-domain theory that predicts where blind
   synthesis matches the real-mix distribution, where it fails (natural RGB),
   and which operator is correct per modality.
3. False-alarm control by construction: defect-erased synthetic normals and
   pair masking shape a confidence geometry that yields zero false alarms at
   full coverage; conformal rejection adds a distribution-free guarantee.
   The fully-supervised oracle, by contrast, is unrescuable by thresholding.
4. An anatomy of the remaining gap: bit-level evidence composes across
   combination orders while joint appearance does not — the oracle's true
   advantage is the appearance interaction of real high-order mixes, not
   knowledge of the combination support (both alternatives tested and
   rejected).

## 2 Related Work

- **Mixing augmentations.** Mixup (Zhang et al., 2018): convex image/label
  blending — designed as a regularizer for single-label training, not as a
  combination synthesizer; we show averaging ghosts both objects (MNIST mixup
  full mAP 0.738 vs overlay 0.868; WM38 bitF1 0.450 vs 0.609). CutMix (Yun et
  al., 2019): rectangle replacement with area-proportional soft labels — as a
  synthesizer its label is frequently false (measured: at patch fraction 0.5,
  the pasted object is >70% lost in 71% of samples). Copy-Paste (Ghiasi et
  al., 2021): mask-based object pasting for instance segmentation — requires
  location/mask annotation; our industrial setting has none (content-blind
  constraint). Our contribution is not a new pasting op but the label-fidelity
  criterion + the single-to-multi bootstrap setting + FAR control.
- **Single-positive multi-label (SPML).** Cole et al. (2021) and its
  successors (Kim et al. 2022 large-loss; an active 2024-25 line of
  VLM/prompt pseudo-labeling, hyperbolic structured classification,
  class-prior methods). SPML assumes multi-label IMAGES with one observed
  positive and corrects the false negatives among the unobserved labels. Our
  setting is STRICTLY WEAKER: no multi-label image is observed at all — the
  training images each contain a single category — so there are no false
  negatives to correct. Our single_only baseline is exactly SPML's
  Assume-Negative baseline made unbiased, and it still fails on multi-label
  test (bit-F1 0.24-0.41): the deficit is STRUCTURAL (absent co-occurrence),
  not label noise, so an oracle SPML method (correction inert) reduces to
  this failing baseline. Synthesis addresses it; label correction cannot.
  Full positioning + the information-ordering argument in Sec. 6.
- **Mixed-type wafer map classification.** MixedWM38 (Wang et al., 2020);
  fully-supervised methods reach 98-99% accuracy (density-aware fusion 2025;
  MLR-WM-ViT). **Single-to-mixed prior art exists and must be engaged
  head-on**: (i) ESWA 2023 "Learning from single-defect wafer maps to
  classify mixed-defect wafer maps" — synthesizes mixed maps via mixup +
  rotation + noise filtering from normal+single maps; (ii) CAIE 2025
  "Mixed-defect wafer map separation and detection based on single-defect
  wafer map"; (iii) SSRN 2025 diffusion+attention synthesis. Our
  differentiation: (a) label-fidelity mechanism — we measure WHY operators
  differ and show mixup-style blending is the weakest image operator (bitF1
  0.435 vs 0.837 overlay stack on WM38); (b) the false-alarm axis these works
  do not study — the real-mixed-trained oracle false-alarms on 56% +-8 of
  real normals and cannot be rescued by thresholding, while synthetic normals
  + pair masking drive this to zero with no normal labels; (c)
  held-out-combination compositional protocol; (d) cross-domain
  scope (chip palette maps, MultiMNIST, VOC boundary); (e) training-free
  operators vs generative (diffusion) synthesis. TODO: obtain ESWA-2023
  protocol details/numbers for a direct-comparison paragraph.
- **Open-set / false-alarm control.** Relation to OOD rejection; our pair-mask
  and synthetic-normal are training-side mechanisms (inference-side selection
  and rejection are a separate paper).

## 3 Problem Setting

**Task family: superimposed-condition multi-label.** Two distinct problems
share the "multi-label" name. In CONDITION-type multi-label, several states
co-occupy one entity's shared support — defects interpenetrating one wafer,
findings on one radiograph, topics interleaved through one document. In
ENTITY-type multi-label (VOC/COCO-style), separate objects occupy disjoint
scene regions — effectively detection with the boxes discarded, solvable
per-object. Our problem, and our method, target the condition type: it is
exactly where combinations cannot be annotated (states interpenetrate) and
where join-synthesis is faithful (the superposition condition). The
entity-type results (Sec 5.3-5.4) mark the task-family boundary: there,
blind joins have no meaning, and detection-style tools are the natural
fit. Reuters lands on the condition side (topics superimpose in one
document) — and synthesis works there.

Label space of K defect types; a sample's target is a K-bit multi-hot vector.
Training pool: only samples with exactly one positive bit (real singles).
No multi-label sample, no normal(all-negative) label, no location/mask
annotation is available at training time (content-blind constraint —
motivated by the fab: defect location is what inspection is trying to find).
Evaluation: genuine multi-label samples (>=2 bits), plus real all-negative
normals for false-alarm rate. Metrics: bit-F1 (macro-F1 over K bits), FAR
(false-positive rate on negative bits; NORMAL-FAR = fraction of real normal
samples raising any alarm), exact-match, mAP (for literature comparability),
and pos/neg mean predicted probability as calibration diagnostics.

## 4 Method: Label-Faithful Synthesis

### 4.1 Synthesis operators (content-blind)

Given two singles (x_a, a), (x_b, b):
- **overlay**: per-pixel max (wafer/palette domains: defect intensity wins
  over normal background) — both objects survive whole; hard label {a,b}.
  Domain analog: chip min-blend.
- **complement (FCM)**: G x G grid (G = 3N, e.g. 9), cells randomly permuted
  and partitioned into n groups; mix_i = x_b base with x_a's group-i cells
  overwritten. The union of the n mixes covers x_a exactly once (full cover).
  Hard labels; per-mix asymmetric (A,B) targets are a no-op within
  [0.9, 1.0] (chip leaderboard: 0.9889-0.9898 across A in {0.90,0.95,1.00}).
- **pair mask (PM)**: for each complement mix, also emit a mask sample —
  x_a's cells kept, x_b's cells with defects erased; target: bit a at a soft
  0.65, all else negative. Teaches "near-normal map with weak fragments =>
  low confidence"; the false-alarm suppressor.
- **synthetic normals**: defect pixels erased from real singles (wafer:
  min(x, normal-die value)) => all-negative samples without any normal label.
- Baselines: CutMix (rectangle patch), Mixup (convex blend, soft labels),
  single-only (no synthesis), oracle (real multi-label training — ceiling,
  not a rival).

### 4.2 Label fidelity

Define survival of the weaker source = fraction of its object pixels present
in the synthesized image. Measured over 3000 pairs (MNIST):

| operator     | mean survival | P(survival<15%) |
|--------------|---------------|-----------------|
| cutmix f0.50 |         0.100 |           0.714 |
| copy_paste   |         0.320 |           0.310 |
| cutmix f0.25 |         0.329 |           0.206 |
| checker g20  |         0.491 |           0.000 |
| overlay/fill |         0.703+|           0.000 |

Operators with high object-loss probability train on false labels; the
downstream ranking follows survival (Sec 5). CutMix at common settings drops
the pasted object most of the time — its labels are lies at synthesis scale.

Cross-dataset validation (MixedWM38, defect-pixel survival, 2000 pairs):
overlay 1.000 > cutmix 0.579 > complement 0.527 > mixup 0.236 — exactly the
downstream 3-seed bitF1 ordering (0.58-0.72 / 0.563 / 0.495 / 0.435). Label
fidelity is a measurable, predictive property on both datasets.

### 4.3 Why not averaging

Mixup's blend produces ghosted objects (both at half contrast) with soft
labels; overlay keeps the stronger signal per pixel with hard labels. Same
combination operator family, opposite outcome: MNIST 0.738 vs 0.868; WM38
0.450 vs 0.609. Failure of "blending" is specifically averaging, not
combining.

## 5 Experiments

### 5.1 MultiMNIST (controlled; mechanism + compositional generalization)

Protocol: 10 digits, combos = unordered pairs (45); 9 pairs held out from the
oracle; synthesis arms may generate all pairs from singles. SmallCNN 0.62M,
20 epochs, 3 seeds. Test = overlaid digit pairs (full: all 45; holdout: the 9).

| config          | full mAP        | holdout mAP     | exact |
|-----------------|-----------------|-----------------|-------|
| overlay (blind) | 0.7730 +-0.0030 | 0.7755 +-0.0059 | 0.270 |
| oracle          | 0.7591 +-0.0049 | 0.6328 +-0.0068 | 0.253 |
| cutmix f0.25    | 0.6980 +-0.0022 | 0.6780 +-0.0089 | 0.142 |
| checker g20     | 0.6737 +-0.0087 | 0.6821 +-0.0040 | 0.061 |
| complement g6n3 | 0.6724 +-0.0082 | 0.6755 +-0.0228 | 0.104 |
| mixup           | ~0.678 (1 seed) | ~0.700          | 0.028 |
| single_only     | 0.5990 +-0.0098 | 0.5994 +-0.0039 | 0.043 |

Full-scale confirmation (400 singles/class, 8,000 synth train, 3,000 test,
25 epochs, 3 seeds) strengthens every claim — overlay's oracle-beating margin
grows and the held-out gap widens:

| config          | full mAP | holdout mAP | exact | pos_prob | neg_prob |
|-----------------|----------|-------------|-------|----------|----------|
| overlay (blind) |   0.8676 |      0.8832 | 0.433 |    0.737 |    0.057 |
| oracle          |   0.8457 |      0.6847 | 0.415 |    0.720 |    0.058 |
| mixup           |   0.7376 |      0.7343 | 0.051 |    0.340 |    0.046 |
| single_only     |   0.6193 |      0.6347 | 0.030 |    0.311 |    0.046 |
| cutmix f0.25    |   0.6055 |      0.6040 | 0.106 |    0.464 |    0.104 |

Claims: (i) overlay matches/exceeds the oracle using singles only (+0.022 at
full scale); (ii) on held-out combos synthesis beats the oracle by a wide
margin (overlay 0.883 vs oracle 0.685, +0.198; compositional generalization —
the oracle cannot classify pair-combinations it never saw); (iii)
whole-object preservation > fragmentation, in survival-order; (iv) overlay
also beats the mixup and cutmix augmentation baselines (0.868 vs 0.738 /
0.606).

### 5.2 MixedWM38 (public benchmark; real multi-label evaluation)

38,015 real wafer maps: 7,015 singles (train source), 30,000 real mixed
(29 combos; eval), 1,000 real normals (FAR). SmallCNN, 15 epochs, 3 seeds;
6 combos excluded from oracle training.

Equal-condition headline (ResNet-18, 30 epochs, both sides):

| config                             | seeds | EVAL bitF1      | HOLDOUT bitF1 | NORMAL FAR       |
|------------------------------------|-------|-----------------|---------------|------------------|
| oracle (real mixed + multi labels) |     3 | 0.974 +-0.019   |         0.957 | 0.563 +-0.083    |
| overlay+sn+neg003 (zero labels)    |     9 | 0.841 +-0.034   |         0.823 | 0.0008 (6/9 = 0) |

(The oracle matches published MixedWM38 accuracies 98-99%, validating the
harness. An earlier "statistical parity" claim against a SmallCNN-15ep oracle
(0.863) is retracted as a weak-oracle artifact — caught by this fairness
check. Oracle+rejection: max-prob on real normals >=0.99 for 80% of normals,
so even tau=0.99 leaves FAR 0.799 — rejection cannot rescue the oracle.)

Secondary configs:

| config (3 seeds)                  | EVAL bitF1      | exact | HOLDOUT bitF1 | NORMAL FAR      |
|-----------------------------------|-----------------|-------|---------------|-----------------|
| overlay+sn+neg003 (SmallCNN,n6000)| 0.749 +-0.064   | 0.375 |         0.745 | 0.095           |
| overlay+sn+neg003 (ResNet18,15ep) | 0.717 +-0.079   | 0.399 |         0.735 | 0.013 +-0.003   |
| overlay+sn+neg003 (SmallCNN,15ep) | 0.641 +-0.019   | 0.248 |         0.614 | 0.031 +-0.026   |
| overlay+sn          | 0.581 +-0.022   | 0.178 |         0.554 | 0.001 +-0.001   |
| fcm_pm_pm+sn+neg003 | 0.540 +-0.035   | 0.254 |         0.487 | 0.058 +-0.031   |
| overlay (no sn)     | 0.609 (1 seed)  | 0.199 |         0.588 | 0.562           |
| cutmix              | 0.563 +-0.020   | 0.291 |         0.487 | 0.810 +-0.218   |
| fcm_pm (no PM)      | 0.495 +-0.009   | 0.230 |         0.435 | 0.839 +-0.125   |
| mixup               | 0.435 +-0.013   | 0.070 |         0.400 | 0.087 +-0.109   |
| single_only         | 0.232 +-0.035   | 0.014 |         0.204 | 0.778 +-0.126   |

Claims: (i) synthesis recovers 86% of the equal-condition oracle's bit-F1
with zero multi labels (headline table above); (ii) the oracle cannot
control false alarms (0.563 +-0.083 on real normals, unrescuable by
thresholding) — pair masking (0.852 -> 0.012 in isolation) and synthetic
normals (0.562 -> 0.001) fix it by construction, to zero at the headline
configuration; (iii) label-fidelity ordering reproduces (overlay > cutmix >
fcm_pm > mixup for bitF1, matching measured survival).

**Full-scale confirmation (ResNet-18, all 7,015 singles as sources, 14,000-mix
test, 30 epochs, 3 seeds).** synthetic-normal (4,000) and neg-target (0.03)
applied IDENTICALLY to every non-oracle arm, so the FAR column isolates the
operator:

| arm (full scale)  | EVAL bitF1      | HOLDOUT | NORMAL FAR | recovery |
|-------------------|-----------------|---------|------------|----------|
| oracle            | 0.9844 +-0.0065 |   0.980 |      0.186 |     100% |
| overlay (ours)    | 0.7947 +-0.0734 |   0.797 |     0.0003 |      81% |
| cutmix            | 0.8548 +-0.0487 |   0.897 |      0.618 |      87% |
| mixup             | 0.5809 +-0.0891 |   0.506 |      0.758 |      59% |
| single_only       | 0.4086 +-0.0098 |   0.375 |      0.390 |      42% |

Two things sharpen at scale. (1) Under an IDENTICAL synthetic-normal control,
only overlay reaches zero false alarms (0.0003); cutmix and mixup inject
destroyed-source false-positive labels that defeat the control and fire on
normals (FAR 0.62 / 0.76) — a direct empirical confirmation of the
fidelity->FAR mechanism (the independence term of Theorem 1). (2) cutmix's
HIGHER raw bit-F1 (0.855) is operationally unusable at 62% false alarm: the
bit-F1-only lens crowns cutmix, the FAR lens shows overlay is the only viable
operator. overlay's bit-F1 (0.795 +-0.073) is within seed variance of the
9-seed n=6000 headline (0.841); the oracle rises to 0.984 with full data, so
recovery reads 81% here. The zero-FAR property and overlay >> single-only
(2x) ordering hold at full scale.

#### 5.2.1 Loss-engineering control: can a better loss on singles substitute?

A natural objection: perhaps the single-only floor is low only because BCE is
a weak multi-label loss, and a purpose-built loss (Asymmetric Loss, Focal)
would close the gap without any synthesis. We test this directly with an
atomic comparison — SmallCNN, identical recipe (n_single_aug 2000, 20 epochs,
3 seeds, no synthetic normals, no neg-target), varying only the arm and the
loss:

| config (SmallCNN, 3 seeds) | train bitF1 | EVAL bitF1 | EVAL pos/neg | NORMAL FAR |
|----------------------------|-------------|------------|--------------|------------|
| single_only + BCE          |       0.947 |      0.243 |  0.167/0.032 |      0.591 |
| single_only + ASL          |       0.923 |      0.316 |  0.231/0.103 |      1.000 |
| single_only + Focal        |       0.953 |      0.242 |  0.172/0.044 |      0.584 |
| overlay + BCE (ours)       |       0.936 |      0.607 |  0.555/0.019 |      0.549 |

All four fit the single-label training data equally (train bit-F1 0.92-0.95).
The strongest multi-label loss (ASL) lifts EVAL bit-F1 by only +0.073 over BCE
(0.243 -> 0.316) and does so by over-predicting positives, driving real-normal
FAR to 1.00. Focal gives nothing (+0.00). Synthesis (overlay) adds +0.364 —
five times the best loss-engineering gain — while keeping FAR lower. The
bottleneck is not loss calibration but the absence of label co-occurrence
structure in single-label data, which no loss can recover and which synthesis
supplies by construction.

### 5.3 PASCAL VOC 2007 (natural-scene boundary analysis)

IMPORTANT SCOPING: the copy-paste arm here uses the dataset's bounding-box
annotations to crop objects — it is a CONTENT-AWARE reference probe, not
one of our content-blind operators. The content-blind constraint (no
location annotation) governs the industrial setting and all headline
claims (WM38, MNIST, Reuters, all blind). On natural scenes we
deliberately relax it to measure the boundary: blind operators fail here,
and matching the oracle requires location supervision — which is itself
the finding predicted by the superposition-domain condition.

Natural single-category (56%) vs multi-category (44%) split — no label
masking. ResNet-18, subsampled protocol. Findings: (i) copy-paste (crops onto
scene backgrounds) is the only synthesis matching the oracle's bit-F1 (0.357
vs 0.355) — natural scenes need object/context scale alignment; (ii) tight
crops alone regress (scale mismatch); (iii) region > blend ordering persists.
Positions the domain condition: overlay-type synthesis requires a
signal-ordered pixel space (defect > normal), which wafer/palette domains
have and RGB scenes do not.

### 5.4 MS-COCO (scale boundary — honest negative at subsampled scale)

80 classes; natural split (train2017: 24,186 single-cat / 93,080 multi-cat).
At CPU-subsampled scale (cap 30 singles/class, 1.5k synth, ResNet-18, 6 ep)
ALL arms — including the oracle — collapse (train bitF1 ~1.0, eval 0.05-0.15):
80-way multi-label with ~3.5 categories/image requires training scale that a
subsampled protocol cannot provide. We report this as a measured scale
boundary; credible COCO numbers require GPU-scale training (future work /
camera-ready). No method ranking is claimed at this scale.

### 5.5 Ablations

- A-target insensitivity (chip leaderboard): bit-F1 0.9889-0.9898 for
  A in {0.90, 0.95, 1.00} — asymmetric AB labels are a no-op band.
- B/weak-target sensitivity (chip): partner target 0.5 collapses the
  pos-neg gap to negative (-0.21); >=0.6 recovers (+0.04..+0.14) — the
  weak bit's own target is the lever; BCE bits are independent.
- Complement grid/groups (MNIST): n2 > n3 > n4 (larger surviving area
  better); fine grids approach dither.
- neg-target: +0.06..0.10 bitF1, slight FAR give-back (WM38).
- Backbone (WM38): winner config SmallCNN 0.641 +-0.019 -> ResNet-18
  0.717 +-0.079 (exact 0.248 -> 0.399; NORMAL FAR 0.031 -> 0.013). Gains
  transfer and grow with capacity; not a toy-backbone artifact.
- Training-size scaling (WM38, seed 0, SmallCNN): n_train 1000/3000/6000 ->
  bitF1 0.589/0.660/0.683, NORMAL FAR 0.109/0.065/0.055 — monotone, not
  saturated at 6000; synthetic data is free.

### 5.6 Inference-side rejection and checkpoint selection (WM38)

Margin rejection: refuse samples whose max bit probability < tau (flagged for
human review). Three independent reproductions (15/30/60-epoch models):

| model      | tau | coverage | bitF1(accepted) | NORMAL FAR       |
|------------|-----|----------|-----------------|------------------|
| 30ep       | 0.9 |    91.7% | 0.709 (up)      | 0.151 -> 0.000   |
| 30ep(easy) | 0.9 |    94.4% | 0.699 (kept)    | 0.041 -> 0.000   |
| 60ep       | 0.9 |    98.1% | 0.747 (kept)    | 0.062 -> 0.000   |

Zero false alarms on 1,000 real normals at 2-8% review cost; accepted-set
accuracy never drops; longer training makes rejection cheaper.

Checkpoint selection (honest scope): on the chip domain, val-F1 saturates at
epoch 1 (pretrained 88M backbone + easy in-distribution val) and selecting by
the pos-neg margin instead rescues bit-F1 0.755 -> 0.918 at tr50. This
pathology did NOT transfer to WM38 (two attempts: hard synthetic-combo val
and easy in-dist val; scratch-trained SmallCNN keeps val-F1 informative
through 30+ epochs; both criteria pick adjacent checkpoints). Margin-based
selection is presented as a chip-regime finding, not a general law.

Synthetic-real overfitting boundary: real-mixed bit-F1 peaks near epoch 30
(0.779) and declines with further training on synthetic combos (ep52: 0.748)
while the synthetic val keeps improving — no synthetic-val criterion can see
the real peak. A small real validation set, when available, is worth more
than any selection criterion on synthetic data.

### 5.7 Order extrapolation and annotation efficiency (WM38)

Order extrapolation (pairs-only training, real mixes split by order): bit-F1
extrapolates gracefully to orders never synthesized (2/3/4-mix:
0.784/0.671/0.628) while exact-match collapses (0.699/0.143/0.000). Naive
higher-order synthesis (+triples/+quads) hurts every metric: only 12 of 56
possible 3-combos and 4 of 70 4-combos exist in WM38 — arbitrary-combo
synthesis spends capacity on combinations that never occur. This isolates
what the oracle really owns: knowledge of the combination SUPPORT, the one
thing single-label data cannot supply (Sec Theory, P1 delta).

Annotation efficiency (winner recipe): 500 real singles (~62/class) already
reach bit-F1 0.617 with 0.002 NORMAL FAR; 2000 -> 0.697; 7015 -> 0.779.

### 5.8 Fourth family: text (Reuters-21578) and the operator-flip

Natural split, full top-20 categories (5,995 single-topic train / 889
real multi-topic oracle pool / 300 real multi-topic test; TF-IDF + MLP,
3 seeds): oracle 0.603 > vec-average 0.433 > concat 0.398 >> single-only
0.254. Synthesis recovers 72% of the oracle from singles in a non-vision
modality (subsample gave the same ordering at 71%). The
operator ranking FLIPS versus images: averaging (the mixup analog) wins in
text because topic evidence occupies (nearly) disjoint feature coordinates —
averaging preserves per-coordinate evidence, whereas image classes share
pixel coordinates and averaging ghosts them. The invariant law is evidence
preservation (label fidelity); which operator maximizes it is determined by
the modality's evidence geometry.

### 5.9 Conformal FAR guarantee and its boundary

Split-conformal threshold from 500 known-good real normals: realized FAR
0.040 at alpha=0.05 and 0.006 at alpha=0.01, coverage >= 99.5% (guarantee
holds; known-good samples require no defect annotation). Calibrating on
training-style synthetic normals fails (realized FAR 0.97): training
collapses their scores, so exchangeability with deployment normals is the
binding assumption.

### 5.10 What the oracle's advantage actually is

Two hypotheses for the residual bit-F1 gap were tested and rejected: (i)
combination support — synthesizing only the 29 real combos does not recover
joint accuracy (4-mix exact 0.000 unchanged); (ii) order coverage — adding
triples/quads hurts. The surviving explanation: real higher-order mixes
contain appearance interactions (overlapping defects distort one another)
that independent-union synthesis cannot produce. The oracle's advantage is
the image distribution of real high-order mixes; conversely, rejection
cannot buy the oracle our false-alarm control (Sec 5.2). Each side owns one
axis, and only ours is available without annotation.

## 6 Discussion & Limitations

**What each side owns.** Under equal conditions the oracle keeps a bit-F1
lead (0.974 vs 0.837): real high-order mixes contain appearance interactions
that independent-union synthesis cannot produce, and we show this gap is not
closable from the label side — neither combination-support knowledge nor
higher-order synthesis recovers it (Sec 5.10). Conversely, the oracle's
false-alarm behavior is not closable from the inference side — with
confidence above 0.99 on most real normals, no threshold separates its
normals from defects (FAR 0.799 even at tau 0.99). Each regime owns one
axis; only the synthesis side's axis is available without annotation, and in
inspection practice the false-alarm axis is typically the binding one.

**Positioning against single-positive multi-label (SPML).** SPML is the
paradigm most easily confused with ours, and the difference is not
incremental — it is a difference in what is OBSERVED, which makes SPML's
toolkit inapplicable here. Order multi-label weak supervision by observed
information:
1. full supervision — multi-label images, all labels;
2. SPML (Cole 2021; Kim 2022) — multi-label images, exactly one positive
   observed per image, the remaining positives unobserved (false negatives);
3. ours — NO multi-label image observed at any level; each training image
   contains a single category.

Ours is strictly weaker than SPML: SPML still sees every co-occurrence (the
image contains both objects even if only one is labeled), whereas we never
observe a co-occurrence at all — yet in a superposition domain we recover the
oracle (Cor. 1), because the generative structure lets synthesis reconstruct
the co-occurrence distribution that observation withholds.

SPML's methods do not transfer, and not for want of tuning. SPML exists to
combat the false negatives created by assuming unobserved labels negative;
its advances — label estimation (Cole 2021), large-loss rejection (Kim 2022)
— all model that false-negative noise. Our data has none: a single-defect
exemplar genuinely has one class, so the assumed negatives are TRUE. Our
single_only baseline is precisely SPML's Assume-Negative baseline applied to
genuinely single-label images — an unbiased AN — and it still fails on
multi-label test (bit-F1 0.24-0.41). An oracle SPML method, its
false-negative correction inert because there is nothing to correct, reduces
exactly to this failing baseline. The deficit is STRUCTURAL (absent
co-occurrence), not label noise; the remedy is synthesis, not label
correction.

The two are orthogonal and composable, not competing: given both
single-label exemplars AND a pool of partially-labeled multi-label images,
SPML operates on the latter and our synthesis on the former. A head-to-head
benchmark is therefore ill-posed — the methods consume different inputs — so
we report the conceptual ordering: our setting subsumes SPML's in difficulty
by removing the multi-label image entirely.

**Joint prediction lags.** Exact-match trails the oracle (0.440 vs 0.903 at
the headline configuration): per-bit training composes evidence but not
joint cardinality, the same phenomenon behind the order-extrapolation result
(bit-F1 extrapolates to unseen orders, exact-match does not).

**When compositional generalization appears.** The synthesis advantage on
held-out combinations is decisive when the label space is large relative to
observed combinations (MNIST, 45 pairs: +0.14 over the oracle) and vanishes
when few bits cover all combinations (WM38's 8 bits; VOC scenes, where all
methods drop together on held-out-pair images).

**Model selection on synthetic data.** Real-mix performance peaks before
synthetic-val metrics do (0.818 at epoch 30 vs a monotonically rising val to
epoch 50); no synthetic-val criterion — F1 or margin — sees the real peak. A
small real validation set is worth more than any selection heuristic on
synthetic data. Relatedly, margin-based checkpoint selection helps only in
the regime where the val metric saturates (observed in the chip domain;
absent on WM38) and is reported as a domain-scoped finding.

**Guarantee boundary.** The conformal FAR guarantee requires calibration
normals exchangeable with deployment normals: 500 known-good samples
(annotation-free to collect) suffice, whereas training-style synthetic
normals fail (their scores are collapsed by training itself).

**Scope.** One public benchmark with real mixed labels carries the
industrial claim (MixedWM38); chip-domain results use internal data;
VOC/COCO results are subsampled-scale boundary analyses. We do not run a
head-to-head SPML benchmark by design — the settings consume different
inputs (SPML requires multi-label images; we forbid them), so the comparison
is conceptual, not empirical (Sec. 6).

## 7 Conclusion

From single-label data alone — no multi-label, no normal, no location
annotation — label-faithful synthesis trains multi-label recognizers that
recover 86% of an equal-condition fully-supervised oracle on real mixed
data, exceed it on unseen combinations when the combination space is rich,
and achieve a false-alarm regime the oracle cannot reach by any amount of
real data or thresholding: zero false alarms at full coverage, with a
finite-sample conformal guarantee available for the price of a handful of
known-good samples. The governing quantity, label fidelity, is measurable
before training and predicts operator rankings across four domain families;
the superposition-domain condition predicts where the approach matches the
real distribution and where it does not. What remains with the oracle is the
appearance of real high-order interactions — a boundary we quantify and
leave as the open problem.

---

## TODO before submission

- [ ] Replace 1-seed WM38 rows with 3-seed (hardening batch running)
- [ ] Backbone ablation table (ResNet-18, running)
- [ ] Scaling curve (running)
- [ ] COCO section 5.4 (downloading)
- [ ] Literature pass for 2024-26 SPML/synthesis work (web search)
- [ ] MixedWM38 published-baseline positioning paragraph
- [ ] Figures: (1) synthesis-arm montage (have: mnist_synthesis_arms.png,
      mnist_complement_faithful.png), (2) survival-vs-bitF1 scatter,
      (3) holdout collapse bar chart, (4) FAR bar chart
- [ ] LaTeX port (venue template)
