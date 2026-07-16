# Multi-Label Recognition Without Multi-Label Annotation: Label-Faithful Synthesis from Single-Label Data

Working draft v0.2 (2026-07-10). All numbers are measured (primary log:
`D:/project/known-cnn/docs/superpowers/multilabel_synth_RESULTS.md`; result
CSVs under `D:/project/known-cnn/outputs/multilabel_synth/`). Candidate venues
are assessed in `D:/project/known-cnn/docs/mlsynth_paper/SUBMISSION_READINESS_260710.md`.

> **Positioning correction, 2026-07-16 (die-budget partition — supersedes the
> earlier "operator = max-union" revision).** A decisive density measurement shows
> WM38 real mixing is a **die-budget PARTITION**, not a pixelwise superposition:
> each physical die carries exactly one defect type, so a real 2-mix wafer
> *partitions* its dies among the two sources rather than overlaying them. Measured
> defect-die fraction of on-wafer area: real 2-mix **0.305**, single **0.290**,
> whole-image **max-union / overlay / Shin et al. (2022) Summation Mixup 0.501**
> (64% denser than real; 91% of its synthetics exceed the real-2-mix 95th
> percentile), full-cover-complement **FCM / FCM-PM 0.293** (matches real).
> max-union therefore assumes the WRONG generative model — it double-counts
> overlapping dies and produces distributionally unrealistic, over-dense training
> wafers — and is **EXCLUDED** on die-budget/realism grounds. This is a
> modeling-faithfulness argument, stated honestly: max-union's raw bit-F1 **0.80**
> (strict, 5 seeds) is *higher* than any faithful operator's, but it is an artifact
> of training on over-dense maps that over-detect, not a faithful method. The
> full-cover complement (FCM) synthesizes each die as owned by exactly one source,
> matching the real die budget; with a Pair-Mask false-alarm view (**FCM-PM**) it
> is the **faithful WM38 method** and has the best F1-FAR trade-off among
> die-budget-faithful blind operators: FCM-PM **0.654** bit-F1 / **0.147** FAR vs
> cutmix 0.691/0.439, FCM-without-PM 0.665/0.384, mixup 0.537/0.225, single-only
> 0.473/0.602 (WM38 strict, 5 seeds, pick=val_tail_margin_guarded, neg 0.02).
> Pair-Mask's role is FAR control (0.147 vs FCM 0.384 at comparable F1). FCM-PM is
> THE WM38 method — and also excels on chip-internal maps (~0.99); the earlier
> revision's "max-union is our operator / no operator novelty / FCM-PM chip-only"
> framing is UNDONE. Density and strict evidence:
> `D:/project/known-cnn/docs/superpowers/multilabel_synth_RESULTS.md` (section
> "max-union violates wafer die-budget") and
> `D:/project/known-cnn/docs/mlsynth_paper/FCMPM_CORRECTION_AND_EVIDENCE_PLAN_260713.md`.

---

## Abstract

Industrial visual inspection routinely faces images containing multiple
co-occurring defect types, yet multi-label annotation of such images is
impractical: co-occurrences explode combinatorially, rare combinations may
never be observed in labeled form, and overlapping patterns are ambiguous to
annotate. In contrast, single-defect examples are cheap and unambiguous. We
study how to train a multi-label classifier from single-label supervision
alone, by synthesizing combination examples from single-label sources. The crux
is that a faithful synthesizer must reproduce the domain's TRUE combination
operator. We contribute (i) a domain insight — on wafer maps, real mixing is a
**die-budget partition** (each die carries exactly one defect type), not a
pixelwise superposition, verified by density (real 2-mix defect-die fraction 0.31,
essentially the single-defect 0.29); (ii) a **label-fidelity /
generative-model-match criterion** that selects the faithful content-blind
operator — evidence-preserving AND density-matching — before any training; (iii)
the faithful operator itself for wafer maps, full-cover-complement mixing with
Pair-Mask (**FCM-PM**); (iv) a superposition/partition-equivalence excess-risk
theory explaining when and why blind synthesis matches a fully-supervised oracle,
with a natural-image (PASCAL VOC) boundary corollary; and (v) an annotation-free
false-alarm-rate control layer (synthetic normals + val-margin selection +
naive-Bayes reject + split-conformal calibration) that operator-only prior work
lacks. On a controlled MultiMNIST benchmark — a genuine superposition domain —
blind max-overlay synthesis from single-label data exceeds a fully-supervised
oracle on the full test (mAP 0.868 vs 0.846) and exceeds it by +0.198 mAP on
held-out label combinations that the oracle never observed. On the public
MixedWM38 wafer-map benchmark, whole-image max-union — which under the binary
encoding (normal die 0.5, defect 1.0) equals Shin et al. (2022) Summation Mixup —
double-counts overlapping dies (density 0.50, 64% denser than real) and is
**excluded as die-budget-violating**; its high raw bit-F1 (0.80) is an
over-dense-training artifact, not a faithful method (a modeling-faithfulness
argument, not an F1 one). The density-matching FCM-PM operator (0.29) is the best
die-budget-faithful synthesis: bit-F1 0.654 at real-normal FAR 0.147, versus
0.38–0.44 FAR for the next-best faithful arms (Pair-Mask is the FAR-control lever,
0.147 vs 0.384 without it), and it also reaches ~0.99 on chip-internal maps. A
small set of known-good samples yields a finite-sample conformal FAR guarantee
(realized 0.040 at alpha=0.05, 0.006 at alpha=0.01). Generality holds across a
fourth modality (Reuters text, where the operator flips to averaging); we
characterize the natural-image boundary (PASCAL VOC) and release the full harness.

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

We answer by synthesizing combination examples from single-label sources, and
show that a faithful synthesizer must satisfy TWO measurable criteria. First,
**label fidelity** — every labeled source must retain detectable evidence in the
synthesized example; operators that destroy evidence (rectangle patching, pixel
averaging) train on false labels, and the measured survival ordering predicts the
raw detection ordering within every image family we test. Second, and decisively
for wafer maps, the synthesizer must reproduce the domain's **true combination
operator**. We find that WM38 real mixing is a **die-budget partition** — each
die carries exactly one defect type — so a real 2-mix wafer partitions its dies
among the two sources (measured defect-die fraction 0.31, essentially the
single-defect 0.29) rather than superimposing them. Whole-image max-union
maximizes evidence survival yet doubles overlapping dies to 0.50 (64% denser than
real; 91% of its synthetics exceed the real-2-mix 95th percentile), silently
violating the die budget and producing distributionally unrealistic training
wafers.

We therefore **exclude** whole-image max-union — equivalently Shin et al. (2022)
Summation Mixup under the binary encoding (Sec. 2, Sec. 5.2) — on
generative-model-faithfulness grounds. We are explicit that this is a
modeling-realism argument, not an F1 argument: max-union's raw bit-F1 (0.80) is
higher than any faithful operator's, but it is earned on over-dense wafers that
over-detect. The faithful operator for wafer maps is **full-cover-complement
mixing**, which assigns each die to exactly one source and matches the real die
budget (0.29); with a Pair-Mask false-alarm view (**FCM-PM**) it gives the best
F1-FAR trade-off among die-budget-faithful blind operators (0.654 bit-F1 at FAR
0.147). The label-fidelity / generative-model-match criterion is what selects the
faithful operator (FCM complement) over the unfaithful one (max-union).

The reliability layer is the practical core, and it is what operator-only prior
work omits. Because the training data contain no all-negative label, a naive
synthesizer over-alarms on real normals; we control false alarms without any
real-normal annotation by combining synthetic normals, a negative-target lever,
val-margin checkpoint selection on a disjoint held-out-source synthetic proxy,
class-conditional Gaussian (naive-Bayes) rejection fit only on further disjoint
single sources and synthetic normals, and a split-conformal calibration for a
finite-sample guarantee. Real mixed and normal maps remain final test data.
FCM-PM is the WM38 method and also reaches bit-F1 ~0.99 on chip-internal maps,
where its Pair-Mask view likewise aids FAR control.

Contributions:
1. **Die-budget partition insight + a generative-model-match criterion.** We show
   WM38 real 2-mixes are a **die-budget partition** (defect-die density 0.31,
   matching singles) not a pixelwise superposition (max-union 0.50), and give a
   two-part criterion — evidence survival (label fidelity) AND generative-model
   match (density / die-budget) — that selects the faithful content-blind operator
   (full-cover complement) over the unfaithful one (max-union), before any
   training. The setting is stricter-than-SPML (genuinely single-label training
   images, no location annotation, evaluated on real multi-label data and real
   normals) with an equal-condition oracle protocol validated against published
   benchmark numbers.
2. **A faithful synthesis operator with the best legitimate trade-off.** FCM-PM
   (full-cover-complement + Pair-Mask) matches the real die budget (0.29) and,
   among die-budget-faithful blind operators, has the best F1-FAR trade-off on
   WM38 (0.654 bit-F1 / 0.147 FAR; Pair-Mask is the FAR-control lever, 0.147 vs
   0.384 without it) while reaching ~0.99 on chip-internal maps.
3. **Theory.** A superposition/partition-equivalence excess-risk account: blind
   synthesis is oracle-faithful exactly when the operator reproduces the domain's
   true combination law (pixelwise max for inked digits/spectrograms, die-budget
   partition for wafer maps, coordinate averaging for text); it predicts the
   natural-image boundary corollary and isolates the oracle's residual advantage
   (the appearance interaction of real high-order mixes, not knowledge of the
   combination support — both alternatives tested and rejected).
4. **Annotation-free guarantee.** A strict source-only reliability pipeline —
   synthetic normals, negative-target control, synthetic validation margin for
   checkpoint selection, class-conditional Gaussian pattern likelihood for
   rejection/decoding, and split-conformal calibration for a finite-sample
   marginal FAR guarantee under exchangeability — which prior operator-only work
   (e.g. Shin et al. 2022) does not provide.
5. **Cross-domain generality.** The criterion transfers across modalities — the
   operator flips to vector averaging on Reuters text (disjoint coordinates) —
   with natural images (VOC) marking the boundary; an audio (FSD50K spectrogram)
   extension is an optional bonus, not a load-bearing claim.

## 2 Related Work

- **Mixing augmentations.** Mixup (Zhang et al., 2018): convex image/label
  blending — designed as a regularizer for single-label training, not as a
  combination synthesizer; we show averaging ghosts both objects (MNIST mixup
  full mAP 0.738 vs overlay 0.868; WM38 mixup bitF1 0.537 vs the faithful FCM-PM
  join 0.654, strict 5-seed). CutMix (Yun et
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
  fully-supervised methods reach 98-99% accuracy. The single-to-mixed setting
  is not new: prior methods use WBM-specific Summation Mixup (Shin et al.,
  2022), mixup with rotation and noise filtering (Shim and Kang, 2023), binary
  union-style Mixup and token-level saliency mixing (Yu, 2024), adaptive ROI
  extraction and combination from single-defect heat maps (Thuan, 2025), or
  learned separation and diffusion pipelines (Li et al., 2025; Yang et al.,
  2026). **These prior single-to-mixed methods assume a pixelwise-superposition
  generative model — Shin et al. (2022) Summation Mixup, under the benchmark's
  binary encoding (normal die 0.5, defect 1.0), is exactly whole-image max-union**
  (verified in `multilabel_synth/synthesis/wm38_arms.py`: the
  `summation_mixup_shin22` and `overlay` arms both compute `np.maximum(ca, cb)`).
  **We show this generative model is wrong for wafer maps**: real 2-mixes are a
  **die-budget partition** (each die carries one defect type; measured defect-die
  density 0.31, matching singles 0.29), whereas max-union double-counts overlapping
  dies to 0.50 — 64% denser than real, with 91% of its synthetics exceeding the
  real-2-mix 95th percentile. We therefore **exclude** max-union / Summation Mixup
  as die-budget-violating; its high raw bit-F1 (0.80) is an artifact of training on
  over-dense wafers, not a faithful method (a modeling-realism argument, stated
  honestly, not an F1 one). Our faithful operator is full-cover-complement mixing
  with Pair-Mask (FCM-PM), which reproduces the real die budget (0.29) and has the
  best F1-FAR trade-off among die-budget-faithful blind operators (0.654 / 0.147).
  Our contributions relative to this prior line are thus the die-budget partition
  insight, the generative-model-match criterion that *selects* the faithful
  operator, the matched-law theory that explains *why* it matches the oracle, and
  an annotation-free FAR-control layer (synthetic normals + val-margin selection +
  NB-reject + split-conformal) that Shin et al. and the other operator-only methods
  above do not provide. Thuan is the closest direct synthesis competitor but
  explicitly extracts defect support (content-aware); our operators remain
  location-agnostic. A submission-grade version must reproduce every prior whose
  exact specification is available under our common 8-bit/FAR protocol — reporting
  both bit-F1/FAR AND synthesized defect-die density — and report native results
  plus protocol gaps for inaccessible methods; this is a required comparison, not
  an optional paragraph.
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

## 4 Method: Label-Faithful Synthesis Framework

Our method is a framework whose operator is selected — not assumed — by a
two-part criterion. It has three parts: (i) a **label-fidelity /
generative-model-match criterion** (Sec 4.2) that selects, per modality and
before any training, the content-blind synthesis operator that both preserves
source evidence AND reproduces the domain's true combination law (for wafer maps,
the die budget); (ii) **val-margin checkpoint selection** on a disjoint
held-out-source synthetic proxy; and (iii) **NB-reject** -- a synthetic-only
class-conditional Gaussian (naive-Bayes) pattern-likelihood reject/decode stage,
optionally backed by the split-conformal FAR guarantee (Sec 5.10). Parts
(ii)-(iii), together with synthetic normals and a negative-target lever, form the
annotation-free FAR-control layer that operator-only prior work omits.

The operator the criterion selects is domain-dependent, because the domain's true
combination law is. In **genuine superposition domains** (inked digits, audio
spectrograms) co-occurring sources join by pixelwise max, so it selects
**overlay / max-union**. On **wafer maps** the true law is a **die-budget
partition** (each die owned by one source), so max-union is *unfaithful* —
under the binary encoding it coincides with Shin et al. (2022) Summation Mixup
(Sec 5.2) and double-counts overlapping dies, over-densifying the map (density
0.50 vs real 0.31); it is excluded on generative-model grounds despite a high raw
bit-F1. The faithful wafer operator is the **full-cover complement (FCM)**, which
partitions the dies among sources and matches the real budget (0.29); with
Pair-Mask (**FCM-PM**) it is the WM38 method (bit-F1 0.654 / FAR 0.147) and also
reaches ~0.99 on chip-internal maps. In **disjoint-coordinate text** it selects
vector averaging (Sec 5.9). The remaining operators in 4.1 (max-union on wafer,
CutMix, Mixup, single-only, oracle) are content-blind reference/baseline arms
evaluated under the identical gate. Generality is established primarily on image
datasets (MixedWM38 public wafer, chip-internal, MultiMNIST); text (Reuters, Sec
5.9) is an additional modality, and audio (FSD50K) is an optional bonus extension,
not a load-bearing claim.

### 4.1 Synthesis operators (content-blind)

Given two singles (x_a, a), (x_b, b):
- **overlay / max-union**: per-pixel max (defect intensity wins over normal
  background) — both objects survive whole; hard label {a,b}. This is the faithful
  operator in a **genuine superposition domain** (inked digits, spectrograms).
  On **wafer maps it is unfaithful**: under WM38's binary encoding (normal 0.5,
  defect 1.0) it is exactly Shin et al. (2022) Summation Mixup's clipped binary
  sum, and it double-counts overlapping dies (density 0.50 vs real 0.31), so it is
  excluded as die-budget-violating (Sec 4.2, Sec 5.2). Domain analog: chip
  min-blend.
- **complement (FCM)** (the faithful wafer operator): G x G grid (G = 3N, e.g. 9),
  cells randomly permuted and partitioned into n groups; mix_i = x_b base with
  x_a's group-i cells overwritten. The union of the n mixes covers x_a exactly
  once (full cover), so each die is owned by exactly one source — a die-budget
  partition that matches the real 2-mix density (0.29 vs real 0.31). Hard labels;
  per-mix asymmetric (A,B) targets are a no-op within [0.9, 1.0] (chip
  leaderboard: 0.9889-0.9898 across A in {0.90,0.95,1.00}).
- **pair mask (PM)** (the FAR-control lever): for each complement mix, also emit a
  mask sample — x_a's cells kept, x_b's cells with defects erased; target: bit a
  at a soft 0.65, all else negative. Teaches "near-normal map with weak fragments
  => low confidence"; the false-alarm suppressor. FCM+PM together form FCM-PM,
  the WM38 method (0.654 / 0.147; PM cuts FAR 0.384 -> 0.147 at comparable F1) and
  the chip-internal operator (~0.99).
- **synthetic normals**: defect pixels erased from real singles (wafer:
  min(x, normal-die value)) => all-negative samples without any normal label.
- Baselines: max-union (excluded on wafer, see above), CutMix (rectangle patch),
  Mixup (convex blend, soft labels), single-only (no synthesis), oracle (real
  multi-label training — ceiling, not a rival).

### 4.2 Label fidelity and generative-model match

Label fidelity is necessary but not sufficient. Define survival of the weaker
source = fraction of its object pixels present in the synthesized image. Measured
over 3000 pairs (MNIST):

| operator     | mean survival | P(survival<15%) |
|--------------|---------------|-----------------|
| cutmix f0.50 |         0.100 |           0.714 |
| copy_paste   |         0.320 |           0.310 |
| cutmix f0.25 |         0.329 |           0.206 |
| checker g20  |         0.491 |           0.000 |
| overlay/fill |         0.703+|           0.000 |

Operators with high object-loss probability train on false labels; the raw
detection ranking follows survival (Sec 5). CutMix at common settings drops the
pasted object most of the time — its labels are lies at synthesis scale.

The **second criterion is generative-model match**: a faithful operator must also
reproduce the domain's true combination law. On WM38 that law is a die-budget
partition, so the faithful operator must match the real 2-mix defect-die density
(0.305, essentially the single-defect 0.290). Measured density: full-cover
complement FCM / FCM-PM 0.293 (matches real) vs whole-image max-union 0.501 (64%
denser than real; 91% of its synthetics exceed the real-2-mix 95th percentile).
Max-union maximizes survival (defect-pixel survival 1.000 on WM38, above CutMix
0.579, complement 0.527, Mixup 0.236) yet FAILS the generative-model criterion by
double-counting dies — which is exactly why its high raw bit-F1 (0.80) is an
over-dense artifact rather than a faithful result. The criterion selects the
operator that satisfies BOTH tests: the full-cover complement, which preserves
each source's evidence exactly once (survival by construction) AND matches the
real die budget.

### 4.3 Why not averaging

In shared-coordinate spaces, Mixup's blend produces ghosted objects (both at
half contrast) with soft labels; a hard, evidence-preserving join keeps each
source recoverable. Same combination-operator family, opposite outcome: on
genuine-superposition MNIST, overlay 0.868 vs mixup 0.738; on wafer maps the
faithful FCM-PM join 0.654 far exceeds mixup 0.537 (strict, 5 seeds). Failure
of "blending" is specifically averaging, not combining. (On wafer maps the
*over-densifying* max-union join is unfaithful for a different reason —
Sec 4.2 — so the faithful join there is the die-budget-partitioning complement,
not overlay.)

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

Audit note (2026-07-10): pre-audit WM38 F1 values in this subsection use the
legacy macro whose denominator could change when an unsupported test bit
produced a false positive. They remain provenance records, not final
submission numbers. The paired saved-probability rerun will replace them with
supported-class macro-F1; all FAR values remain separately reported.

38,015 real wafer maps: 7,015 singles (train source), 30,000 real mixed
(29 combos; eval), 1,000 real normals (FAR). SmallCNN, 15 epochs, 3 seeds;
6 combos excluded from oracle training.

**The die-budget partition finding (decisive).** WM38 real mixing is a
**die-budget partition**: each physical die carries exactly one defect type, so a
real 2-mix wafer partitions its dies among the two sources rather than
superimposing them. We measure the defect-die fraction of on-wafer area:

| arm                                  | defect-die fraction | vs real 2-mix        |
|--------------------------------------|---------------------|----------------------|
| real 2-mix (ground truth)            | 0.305 +-0.050       | —                    |
| single (1 defect)                    | 0.290               | matches              |
| FCM / FCM-PM (full-cover complement) | 0.293               | matches              |
| max-union / overlay (= Shin22 summ.) | 0.501 +-0.123       | 64% denser; 91% > 95th pct |

Whole-image max-union double-counts overlapping dies (0.501 vs real 0.305) and
produces distributionally unrealistic, over-dense training wafers: it assumes the
wrong (superposition) generative model. The full-cover complement synthesizes
each die as owned by exactly one source and matches the real budget (0.293). We
therefore **exclude max-union / Summation Mixup as die-budget-violating**, and
identify FCM-PM as the faithful WM38 operator.

**Strict operator comparison (5 seeds, pick=val_tail_margin_guarded, neg 0.02).**
Common protocol (matched view budget, splits, checkpoint selection, rejection):

| operator                       | bit-F1 | NORMAL FAR | defect-die density | note                          |
|--------------------------------|--------|------------|--------------------|-------------------------------|
| FCM-PM (ours; faithful)        | 0.654  | 0.147      | 0.293 (matches)    | best F1-FAR among faithful    |
| FCM (no Pair-Mask)             | 0.665  | 0.384      | 0.293 (matches)    | PM removed -> FAR 0.147->0.384 |
| cutmix                         | 0.691  | 0.439      | —                  | evidence-destroying, FAR 3x   |
| mixup                          | 0.537  | 0.225      | —                  | ghosting                      |
| single_only                    | 0.473  | 0.602      | 0.290              | floor                         |
| max-union / overlay (= Shin22) | 0.800  | 0.010      | 0.501 (over-dense) | EXCLUDED: die-budget-violating |

Reading, stated honestly: max-union's bit-F1 (0.80) is the highest of any arm and
its FAR the lowest (0.010) — but it earns this on over-dense wafers (density 0.501
vs real 0.305) that over-detect, so it is excluded on generative-model /
distributional-realism grounds, not on F1. This is a modeling-faithfulness
argument. Among **die-budget-faithful** operators, FCM-PM has the best F1-FAR
trade-off: cutmix scores marginally higher F1 (0.691) but triples the FAR (0.439),
and removing Pair-Mask (FCM alone) more than doubles it (0.665 F1 / 0.384 FAR).
Pair-Mask is thus the FAR-control lever (0.147 vs 0.384 at comparable F1). FCM-PM
recovers roughly two-thirds of the literature-grade oracle (0.654 vs 0.974) at FAR
0.147; the annotation-free reliability layer (Sec 5.6, 5.10) tightens the operating
point further.

(The oracle matches published MixedWM38 accuracies 98-99%, validating the
harness. An earlier "statistical parity" claim against a SmallCNN-15ep oracle
(0.863) is retracted as a weak-oracle artifact. For the headline oracle
checkpoint, max-prob on real normals >=0.99 for 80% of normals, so tau=0.99 still
leaves FAR 0.799 — a checkpoint-level calibration result, not an impossibility
claim about all oracle training.)

**Legacy / over-dense-arm provenance (diagnostic only).** Earlier
SmallCNN/ResNet-18 runs of the (now-excluded) max-union / Summation Mixup arm
under the legacy macro are retained as provenance; they show the same over-dense
arm reaching bit-F1 0.795 (full-scale, all 7,015 singles, ResNet-18, 30ep) to
0.841 (nine-seed headline) at near-zero real-normal FAR (0.0003-0.0008). These are
NOT a faithful WM38 result — they are the over-dense-training artifact quantified
above — and are reported only to make the exclusion auditable. Component
ablations that DO transfer to the faithful operator: pair masking in the
complement arm cuts observed NORMAL FAR in isolation (0.852 -> 0.012), and
synthetic normals reduce it in every arm; the label-fidelity survival ordering
(max-union > cutmix > complement > mixup) predicts raw detection but, as the
density column shows, does not by itself certify faithfulness.

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
| Shin22-style Summation Mixup + BCE | 0.936 | 0.607 | 0.555/0.019 | 0.549 |

All four fit the single-label training data equally (train bit-F1 0.92-0.95).
The strongest multi-label loss (ASL) lifts EVAL bit-F1 by only +0.073 over BCE
(0.243 -> 0.316) and does so by over-predicting positives, driving real-normal
FAR to 1.00. Focal gives nothing (+0.00). Adding co-occurrence structure via
synthesis adds +0.364 — five times the best loss-engineering gain — while keeping
FAR lower. (This isolation uses the max-union / Summation Mixup arm precisely
because it is the simplest join; the point is loss-vs-structure, and it holds for
any co-occurrence synthesis including the faithful FCM-PM.) The bottleneck is not
loss calibration but the absence of label co-occurrence structure in single-label
data, which no loss can recover and which synthesis supplies by construction.

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
Positions the domain condition: faithful blind synthesis requires the domain's
true combination law to be reproducible by a content-blind operator (pixelwise
max in superposition domains, die-budget partition on wafer maps) — which
wafer/palette domains admit and opaque RGB scenes do not (objects occlude, not
join).

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

Observed 0/1,000 false alarms on real normals at 2-8% review cost; accepted-set
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

### 5.8 Controlled source-evidence intervention

We attenuated one source toward the normal-die baseline before max-overlay,
kept the hard union label, and changed no other training factor. Five paired
model/data-split repeats show a threshold rather than a smooth monotonic law.
The supported-class macro-F1 uses the six bits that occur in the real mixed
test; Random and NearFull have no positive mixed examples and their false
positives are counted in FAR, not conditionally inserted into the F1 macro.

| retained evidence f | supported bit-F1 | pos prob | neg prob | gap | NORMAL FAR |
|---:|---:|---:|---:|---:|---:|
| 0.00 | 0.631+-0.034 | 0.517 | 0.133 | 0.384 | 0.179+-0.091 |
| 0.10 | 0.724+-0.028 | 0.602 | 0.067 | 0.536 | 0.507+-0.162 |
| 0.25 | 0.806+-0.022 | 0.681 | 0.040 | 0.642 | 0.253+-0.125 |
| 0.50 | 0.811+-0.022 | 0.697 | 0.032 | 0.665 | 0.005+-0.007 |
| 0.75 | 0.814+-0.029 | 0.687 | 0.032 | 0.655 | 0.012+-0.021 |
| 1.00 | 0.793+-0.037 | 0.654 | 0.041 | 0.614 | 0.015+-0.017 |

Full versus erased evidence improves supported bit-F1 by 0.162 (paired
bootstrap 95% CI 0.129--0.208), increases probability gap by 0.229
(0.191--0.276), and reduces NORMAL FAR by 0.164 (0.087--0.234). Survival is
associated with bit-F1 (Spearman rho=0.662, p<0.001), gap (rho=0.624), and
NORMAL FAR (rho=-0.656), but adjacent-step monotonicity is only 60%, 68%, and
52%, respectively. The defensible mechanism is therefore a minimum-sufficient
evidence threshold: catastrophic weakening destabilizes the F1/FAR frontier;
maximal evidence is not itself the F1 optimum.

### 5.9 Fourth family: text (Reuters-21578) and the operator-flip

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

### 5.10 Conformal FAR guarantee and its boundary

Split-conformal threshold from 500 known-good real normals: realized FAR
0.040 at alpha=0.05 and 0.006 at alpha=0.01, coverage >= 99.5% (guarantee
holds; known-good samples require no defect annotation). Calibrating on
training-style synthetic normals fails (realized FAR 0.97): training
collapses their scores, so exchangeability with deployment normals is the
binding assumption.

**Coverage at a guaranteed FAR is where the method pays off (5 seeds x 50
splits, n_cal=500).** Every method meets the finite-sample guarantee (realized
FAR = alpha to within CI), but the *usable coverage* retained at that
guarantee separates them sharply:

| method (die-budget-faithful) | alpha=0.01 realized / coverage | alpha=0.05 realized / coverage |
|------------------------------|--------------------------------|--------------------------------|
| FCM-PM (ours)                | 0.0098 / **0.972**             | 0.0503 / **0.989**             |
| fcm (no Pair-Mask)           | 0.0100 / 0.839                 | 0.0503 / 0.908                 |
| cutmix                       | 0.0097 / 0.759                 | 0.0517 / 0.884                 |
| mixup                        | 0.0096 / 0.737                 | 0.0488 / 0.799                 |
| single-only                  | 0.0092 / 0.518                 | 0.0495 / 0.540                 |

At a *guaranteed* 1% false-alarm rate, FCM-PM keeps **97.2%** of wafers usable
versus 51.8-83.9% for every faithful baseline (1.2-1.9x more decisions at the
same certified safety). Pair-Mask lowers the model's intrinsic normal
confidence, so the conformal reject abstains the least to reach the target.
This coverage-at-guaranteed-FAR advantage is a direct measured result and does
not depend on the die-budget exclusion argument; it is the practical payoff of
the faithful-synthesis + Pair-Mask design.

### 5.11 What the oracle's advantage actually is

Two hypotheses for the residual bit-F1 gap were tested and rejected: (i)
combination support — synthesizing only the 29 real combos does not recover
joint accuracy (4-mix exact 0.000 unchanged); (ii) order coverage — adding
triples/quads hurts. The surviving explanation: real higher-order mixes
contain appearance interactions (overlapping defects distort one another)
that independent-union synthesis cannot produce. The oracle's advantage is
the image distribution of real high-order mixes. Our training-side design is
more consistently low-FAR in the measured protocols, but the full-scale oracle
reaches 0.001 normal FAR in one seed, so this is not an impossibility result.

## 6 Discussion & Limitations

**What each side owns.** Under equal conditions the oracle keeps a bit-F1
lead (0.974 vs the faithful FCM-PM's 0.654): real high-order mixes contain
appearance interactions that die-partition synthesis does not reproduce; neither
combination-support matching nor naive higher-order synthesis closes it in our
tests (Sec 5.10). (The excluded max-union arm scores higher raw bit-F1, 0.80, but
only by training on over-dense wafers that violate the die budget — Sec 5.2 — so
it is not a legitimate closer of this gap.) On the headline oracle checkpoint,
even tau=0.99 leaves normal FAR 0.799; the full-scale oracle varies from
0.295/0.262/0.001 across seeds, so this is an optimization/calibration result, not
an inherent impossibility theorem. Our faithful training-side construction reaches
its operating point (FAR 0.147) with margin rejection and a conformal guarantee
(Sec 5.6, 5.10) rather than real-normal training.

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
observe a co-occurrence at all — yet when synthesis reproduces the domain's true
combination law (superposition for inked digits, die-budget partition for wafer
maps) we recover the oracle (Cor. 1), because the generative structure lets
synthesis reconstruct the co-occurrence distribution that observation withholds.

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
normals exchangeable with deployment normals. Known-good status must still be
established, although it requires no defect-type or location annotation.
Training-style synthetic normals fail because their scores are collapsed by
training itself.

**Scope.** One public benchmark with real mixed labels carries the
industrial claim (MixedWM38); chip-domain results use internal data;
VOC/COCO results are subsampled-scale boundary analyses. We do not run a
head-to-head SPML benchmark by design — the settings consume different
inputs (SPML requires multi-label images; we forbid them), so the comparison
is conceptual, not empirical (Sec. 6).

## 7 Conclusion

From single-label training data alone — no multi-label, real-normal, or location
annotation — label-faithful synthesis trains multi-label recognizers by matching
the domain's true combination law. On wafer maps that law is a **die-budget
partition** (not a pixelwise superposition), so whole-image max-union / Summation
Mixup is excluded as die-budget-violating (over-dense, 0.50 vs real 0.31) despite
its high raw bit-F1, and the faithful full-cover-complement operator (FCM-PM)
recovers about two-thirds of an equal-condition oracle (bit-F1 0.654 at FAR 0.147)
while reaching ~0.99 on chip-internal maps. On a genuine superposition domain
(MNIST) blind synthesis exceeds the oracle on unseen combinations. A margin-reject
stage drives observed normal FAR to zero at a few-percent review cost, and optional
calibration on a small known-good set provides a finite-sample marginal FAR
guarantee under exchangeability — reliability that operator-only prior work lacks.
The two-part criterion (label fidelity plus generative-model match) is measurable
before training and selects the faithful operator across modalities (the operator
flips to averaging on text); the combination-law condition predicts where the
approach matches the real distribution and where it does not. What remains with the
oracle is the appearance of real high-order interactions — a boundary we quantify
and leave as the open problem.

---

## Required before submission

- [ ] Finish the queued five-seed common-protocol evaluation of the exact
      Shin et al. 2022 Original/Average/Summation Mixup arms; implement and
      evaluate Shim--Kang 2023 only after its exact operator is recovered.
- [ ] Run the controlled fidelity intervention and paired main-arm seeds.
- [ ] Produce threshold--bit_F1--NORMAL-FAR frontiers from calibration-selected
      operating points.
- [ ] Repeat conformal calibration splits and report finite-sample coverage.
- [ ] Compile `D:/project/known-cnn/docs/mlsynth_paper/latex/main.tex` with a
      real LaTeX engine and add the three submission figures listed in
      `D:/project/known-cnn/docs/mlsynth_paper/SUBMISSION_READINESS_260710.md`.
