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
single-label data matches a fully-supervised oracle on the full test
(mAP 0.773 vs 0.759) and exceeds it by +0.14 mAP on held-out label
combinations that the oracle never observed. On the public MixedWM38 wafer-map
benchmark, under an equal-condition comparison (same backbone and budget,
where the fully-supervised oracle reaches 0.974 +-0.019 bit-F1 — matching
published 98-99% accuracies), training only on real single-defect wafers
recovers 86% of the oracle (0.837 +-0.039 over six seeds) while producing
ZERO false alarms on 1,000 real normal wafers in every seed — the oracle
false-alarms on 56% +-8. Crucially, rejection cannot rescue the oracle: its
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
  full mAP 0.678 vs overlay 0.773; WM38 bitF1 0.450 vs 0.609). CutMix (Yun et
  al., 2019): rectangle replacement with area-proportional soft labels — as a
  synthesizer its label is frequently false (measured: at patch fraction 0.5,
  the pasted object is >70% lost in 71% of samples). Copy-Paste (Ghiasi et
  al., 2021): mask-based object pasting for instance segmentation — requires
  location/mask annotation; our industrial setting has none (content-blind
  constraint). Our contribution is not a new pasting op but the label-fidelity
  criterion + the single-to-multi bootstrap setting + FAR control.
- **Single-positive multi-label (SPML).** Cole et al. (2021) and an active
  2024-25 line (VLM/prompt pseudo-labeling with RAM; hyperbolic structured
  classification; class-prior methods). SPML assumes multi-label IMAGES with
  one observed positive; our setting is stricter and industrially natural —
  the training images themselves contain a single category, and we evaluate
  on real multi-label test data. SPML methods recover missing labels of
  existing images; we synthesize combination images that never existed.
- **Mixed-type wafer map classification.** MixedWM38 (Wang et al., 2020);
  fully-supervised methods reach 98-99% accuracy (density-aware fusion 2025;
  MLR-WM-ViT). **Single-to-mixed prior art exists and must be engaged
  head-on**: (i) ESWA 2023 "Learning from single-defect wafer maps to
  classify mixed-defect wafer maps" — synthesizes mixed maps via mixup +
  rotation + noise filtering from normal+single maps; (ii) CAIE 2025
  "Mixed-defect wafer map separation and detection based on single-defect
  wafer map"; (iii) SSRN 2025 diffusion+attention synthesis. Our
  differentiation: (a) label-fidelity mechanism — we measure WHY operators
  differ and show mixup-style blending is the weakest operator (bitF1 0.435
  vs 0.717 overlay on WM38); (b) the false-alarm axis these works do not
  study — the real-mixed-trained oracle itself false-alarms on 55% of real
  normals, while pair-mask + synthetic normals cut this 42x with zero normal
  labels; (c) held-out-combination compositional protocol; (d) cross-domain
  scope (chip palette maps, MultiMNIST, VOC boundary); (e) training-free
  operators vs generative (diffusion) synthesis. TODO: obtain ESWA-2023
  protocol details/numbers for a direct-comparison paragraph.
- **Open-set / false-alarm control.** Relation to OOD rejection; our pair-mask
  and synthetic-normal are training-side mechanisms (inference-side selection
  and rejection are a separate paper).

## 3 Problem Setting

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
combination operator family, opposite outcome: MNIST 0.678 vs 0.773; WM38
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

Claims: (i) overlay matches/exceeds the oracle using singles only; (ii) on
held-out combos every synthesis arm beats the oracle (compositional
generalization; oracle -0.126 collapse); (iii) whole-object preservation >
fragmentation, in survival-order.

### 5.2 MixedWM38 (public benchmark; real multi-label evaluation)

38,015 real wafer maps: 7,015 singles (train source), 30,000 real mixed
(29 combos; eval), 1,000 real normals (FAR). SmallCNN, 15 epochs, 3 seeds;
6 combos excluded from oracle training.

Equal-condition headline (ResNet-18, 30 epochs, both sides):

| config                             | seeds | EVAL bitF1      | HOLDOUT bitF1 | NORMAL FAR      |
|------------------------------------|-------|-----------------|---------------|-----------------|
| oracle (real mixed + multi labels) |     3 | 0.974 +-0.019   |         0.957 | 0.563 +-0.083   |
| overlay+sn+neg003 (zero labels)    |     6 | 0.837 +-0.039   |         0.815 | 0.000 (all 6)   |

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

Claims: (i) synthesis recovers 74% of the oracle's bit-F1 with zero multi
labels; (ii) the oracle cannot control false alarms (0.55 +-0.18 on real
normals) — pair masking (0.852 -> 0.012 alone) and synthetic normals
(0.562 -> 0.001) fix it by construction, 18-48x below the oracle; (iii)
label-fidelity ordering reproduces (overlay > cutmix > mixup for bitF1).

### 5.3 PASCAL VOC 2007 (natural-scene boundary analysis)

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

Natural split (5,995 single-topic train / 300 real multi-topic test, top-20
categories; TF-IDF + MLP, 3 seeds): oracle 0.567 +-0.025 > vec-average
0.402 +-0.023 > concat 0.359 +-0.020 >> single-only 0.221 +-0.007. Synthesis
recovers 71% of the oracle from singles in a non-vision modality. The
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

- The oracle remains ahead on full-test bit-F1 on WM38 (0.86 vs 0.64): real
  mixed wafers contain interactions beyond independent-union synthesis. The
  industrial trade is explicit: -26% bit-F1 for 18x lower false alarms and
  zero multi-label annotation.
- Exact-match lags (joint calibration), consistent with per-bit training.
- Compositional advantage is decisive when label space is large relative to
  observed combos (MNIST 45 pairs); with 8 bits / 29 combos (WM38) the oracle
  generalizes across combos and the holdout gap narrows.
- Natural RGB scenes lack a signal ordering; blind overlay does not port —
  content-aware pasting (with masks) or scene-aligned copy-paste is required.
- Single real-world benchmark with public mixed labels (MixedWM38); chip-domain
  results use internal data.

## 7 Conclusion

Single-label supervision plus label-faithful synthesis trains multi-label
recognizers that approach fully-supervised performance, exceed it on unseen
label combinations, and — with pair masking and synthetic normals — beat it
outright on false-alarm control, all without a single multi-label or normal
annotation.

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
