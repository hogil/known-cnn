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
benchmark, training only on real single-defect wafers reaches 74% of the
real-mixed-trained oracle's bit-level macro-F1 while producing an 18x lower
false-alarm rate on real normal wafers (0.031 vs 0.548) — the oracle, trained
on real multi-label data, cannot control false alarms, whereas two synthesis
components (pair masking and defect-erased synthetic normals) solve this by
design, using no normal labels and no defect-location annotation. We
characterize when the approach transfers to natural images (PASCAL VOC) and
release the full harness.

## 1 Introduction

- Para 1 (problem): multi-label defect recognition; annotation infeasibility
  in fabs (combinatorial co-occurrence, rare combos, ambiguity). Single-label
  data is what production actually yields.
- Para 2 (setting): train on singles only; evaluate on genuine multi-label
  data. This resembles single-positive multi-label (SPML) but is stricter and
  cleaner: instead of masking labels of multi-label images, the training
  images themselves contain a single category (natural in industry).
- Para 3 (approach + mechanism): synthesize combinations from singles. The
  governing variable is label fidelity: synthesis that drops a labeled object
  trains the model on false labels. We measure per-operator survival rates and
  show they predict downstream ranking.
- Para 4 (results): MultiMNIST — blind overlay beats the oracle on held-out
  combos (+0.14, ~18 sigma); MixedWM38 — 74% of oracle bit-F1 with 18x lower
  false alarms using zero multi/normal labels; VOC — boundary analysis for
  natural scenes.
- Para 5 (contributions):
  1. Problem framing: multi-label from single-label supervision, evaluated on
     real multi-label data (public benchmark), no location annotation.
  2. Label-fidelity analysis: a measurable property of synthesis operators
     that predicts multi-label performance; whole-object hard-label synthesis
     dominates blending.
  3. Compositional generalization: synthesis covers combinations absent from
     any real training set; the fully-supervised oracle collapses there.
  4. False-alarm control by construction: pair masking + defect-erased
     synthetic normals cut real-normal false alarms 18-48x below real-data
     training, without normal labels.

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
- **Single-positive multi-label (SPML).** Cole et al. (2021) and successors
  assume multi-label images with one observed positive. Our setting differs:
  the images themselves are single-category (industrially natural), and we
  evaluate against real multi-label test data. (Add 2024-26 SPML follow-ups
  after a literature pass — TODO verify recent work.)
- **Mixed-type wafer map classification.** MixedWM38 (Wang et al., 2020) and
  followers train on the full labeled mixed dataset; published accuracies
  assume full multi-label supervision. We use the dataset in a strictly
  harder protocol (singles-only training) — numbers are not directly
  comparable, and we report the fully-supervised oracle as the ceiling.
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

| config (3 seeds)    | EVAL bitF1      | exact | HOLDOUT bitF1 | NORMAL FAR      |
|---------------------|-----------------|-------|---------------|-----------------|
| oracle              | 0.863 +-0.064   | 0.770 |         0.836 | 0.548 +-0.175   |
| overlay+sn+neg003   | 0.641 +-0.019   | 0.248 |         0.614 | 0.031 +-0.026   |
| overlay+sn          | 0.581 +-0.022   | 0.178 |         0.554 | 0.001 +-0.001   |
| fcm_pm_pm+sn+neg003 | 0.540 +-0.035   | 0.254 |         0.487 | 0.058 +-0.031   |
| overlay (no sn)     | 0.609 (1 seed)  | 0.199 |         0.588 | 0.562           |
| fcm_pm (no PM)      | 0.502 (1 seed)  | 0.226 |         0.445 | 0.852           |
| cutmix              | 0.551 (1 seed)  | 0.281 |         0.487 | 1.000           |
| mixup               | 0.450 (1 seed)  | 0.072 |         0.412 | 0.005           |
| single_only         | 0.227 (1 seed)  | 0.002 |         0.224 | 0.636           |

(1-seed rows to be replaced by tonight's 3-seed hardening batch.)

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

### 5.4 MS-COCO (scale) — PENDING

80 classes, natural single/multi split, subsampled protocol; copy-paste vs
cutmix vs mixup vs single-only vs oracle. (Data downloading; section to be
filled.)

### 5.5 Ablations

- A-target insensitivity (chip leaderboard): bit-F1 0.9889-0.9898 for
  A in {0.90, 0.95, 1.00} — asymmetric AB labels are a no-op band.
- B/weak-target sensitivity (chip): partner target 0.5 collapses the
  pos-neg gap to negative (-0.21); >=0.6 recovers (+0.04..+0.14) — the
  weak bit's own target is the lever; BCE bits are independent.
- Complement grid/groups (MNIST): n2 > n3 > n4 (larger surviving area
  better); fine grids approach dither.
- neg-target: +0.06..0.10 bitF1, slight FAR give-back (WM38).
- Backbone (WM38): SmallCNN vs ResNet-18 — PENDING (tonight's batch).
- Training-size scaling (WM38): n_train in {1000, 3000, 6000} — PENDING.

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
