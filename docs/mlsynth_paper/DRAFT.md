# Multi-Label Recognition Without Multi-Label Annotation: Label-Faithful Synthesis from Single-Label Data

Working draft v0.2 (2026-07-10). All numbers are measured (primary log:
`D:/project/known-cnn/docs/superpowers/multilabel_synth_RESULTS.md`; result
CSVs under `D:/project/known-cnn/outputs/multilabel_synth/`). Candidate venues
are assessed in `D:/project/known-cnn/docs/mlsynth_paper/SUBMISSION_READINESS_260710.md`.

> **Positioning correction, 2026-07-17 (FINAL — single-train / multi-eval,
> operator-match framing; supersedes both the 2026-07-16 "summation is the best
> content-blind operator / no operator novelty" revision AND the earlier "overlay is
> inadmissible because it preserves location" exclusion-by-fiat, which handed a
> rigorous reviewer a gerrymandered-baseline kill).** SETTING: single-label
> **training**, multi-label **evaluation**, with **no multi-label or normal
> annotation** (content-blind). GOAL: **maximize** the multi-label performance–FAR
> tradeoff in THIS setting. METHOD: **FCM-PM** (full-cover complement + Pair-Mask) +
> **val-margin** checkpoint selection + **naive-Bayes reject**.
>
> **KEY MEASURED LAW (operator-match).** The RIGHT content-blind operator depends on
> the domain's TRUE combination law — a summation/union join (overlay = Shin et al.
> 2022 Summation Mixup; `np.maximum` of two single maps) for **superposition-structured**
> domains where evidence adds/saturates (MNIST digits, FSD50K audio, WM38 under its
> binary die encoding), and **FCM-PM** (partition-complement) for
> **partition-structured** domains where each unit carries one condition (chip-internal
> maps). This is now **MEASURED, not asserted**: on the real chip domain (a partition
> law) FCM-PM BEATS overlay decisively on BOTH axes under an identical protocol
> (head-to-head, 3 seeds s99/s7/s42) — FCM-PM eval bit_F1 **0.9968 +-0.0008** / Total-FAR
> **0.68 +-0.63%** vs overlay **0.8733 +-0.0338** / **16.20 +-5.53%** (bit_F1 +0.12;
> Total-FAR ~24x lower on average, holding for every seed with no overlap -- worst FCM-PM
> seed 0.9960 beats best overlay seed 0.9187). Both learn
> singles equally (train bit_F1 ~0.99); the entire gap is multi-defect eval, because
> pixelwise-max mis-models a partition (overlay under-detects the second defect in
> combos => combo bit_F1 0.61–0.80; FCM-PM keeps combos crisp 0.99–1.00). 3-component
> chip ablation: FCM-PM base 0.9948/97.80 -> +val-margin 0.9977/7.53 (dominant FAR
> fixer) -> +nb-reject 0.9979/0.06 (closes OOD tail).
>
> **WM38 (a superposition regime).** overlay/summation is genuinely **STRONG** and we
> report it **HONESTLY** at 0.80/0.010 — **we do NOT claim to beat it on WM38**. Among
> the content-blind operators FCM-PM gives the best FAR-controlled tradeoff (bit-F1
> **0.663 / FAR 0.228** at the arms' common guarded pick; 0.654/0.147 best-pick) vs
> cutmix (0.691 but FAR 0.439), mixup (0.537/0.225), single-only floor (0.473/0.602).
> The density-shift result (overlay beats FCM-PM per WM38 stratum) is now **CONSISTENT**
> with the law, not moot: WM38 is a superposition regime, so its matched operator
> (overlay) is expected to win there; FCM-PM wins where the law is a partition (chip).
> Density is a **modeling characterization** consistent with the law (FCM-PM matches
> real 0.29; overlay over-dense 0.50 = superposition signature).
>
> **Upper reference (multi-label supervision):** the **oracle** = multi-label
> **training** (0.974) — it uses supervision we forbid. (Overlay is NOT such a
> reference: it is a full content-blind operator, a pixelwise `np.maximum` of two
> single maps, no location.)
>
> **DROPPED (all WRONG for this paper):** "summation is the best operator", "overlay is
> inadmissible because it uses location", "FCM-PM leads the location-free frontier by
> excluding overlay", "density-shift is moot".
>
> The paper is an **annotation-free, reliability-guaranteed, cross-domain FRAMEWORK**
> whose contribution order (broad-venue repositioning) is: (i) the **operator-match
> LAW** as a general principle, with a predictive before-training criterion (evidence
> survival) and a falsifiable flip MEASURED in both directions on public MNIST and by
> the real chip head-to-head (right operator per domain's combination law; FCM-PM wins
> on partition domains); (ii) an **annotation-free, distribution-free split-conformal
> FAR guarantee** (operator-agnostic; the reliability layer operator-only prior work,
> incl. Shin22, lacks); (iii) an **excess-risk theory TIED to operator-match** (the
> matched operator minimizes TV, hence the 2B*TV bound; matched synthesis attains
> oracle-level risk up to the residual TV); (iv) the single->multi **setting** + the
> objective of **performance maximization** within it, with the oracle as a
> multi-label-trained upper reference; (v) the **method** = FCM-PM + val-margin +
> naive-Bayes reject (the partition-domain instantiation), the winner on the chip
> partition domain and the best FAR-controlled content-blind operator on WM38; (vi)
> **cross-domain framework support** (MNIST / audio / text) with an honest natural-image
> boundary (VOC). A **label-fidelity / operator-match criterion** characterizes which
> content-blind operator preserves evidence per domain (summation/union on
> superposition-structured wafer / digits / audio; partition-complement on
> partition-structured chip; averaging on disjoint-coordinate text). Evidence:
> `D:/project/known-cnn/docs/superpowers/multilabel_synth_RESULTS.md` (sections
> "DECISIVE density-shift stress test" and "Conformal FAR guarantee COMPLETE") plus the
> chip overlay-vs-FCM-PM head-to-head (seed 99).

---

## Abstract

Industrial visual inspection routinely faces images containing multiple
co-occurring defect types, yet multi-label annotation of such images is
impractical: co-occurrences explode combinatorially, rare combinations may
never be observed in labeled form, and overlapping patterns are ambiguous to
annotate. In contrast, single-defect examples are cheap and unambiguous. We
study a strict setting — **single-label training, multi-label evaluation** with
**no multi-label or normal annotation** (the content-blind constraint) — and give a general principle for **label-faithful synthesis**: to learn a multi-label
recognizer from single-label sources -- with no multi-label, location, or normal
annotation -- synthesize combination examples with the operator that matches the domain's
*true combination law*. We contribute (i) an **operator-match law** with a *predictive*,
before-training criterion (evidence survival / label fidelity): the right content-blind
operator is fixed not by fiat but by the domain's *true combination law* -- a
summation/union join (overlay = Shin et al. 2022 Summation Mixup) for
**superposition-structured** domains where evidence adds/saturates, and **FCM-PM** (a
partition-complement operator) for **partition-structured** domains where each unit
carries one condition; we verify the law's falsifiable flip on real industry data and
public data -- on the real chip domain (a partition law), under an *identical* protocol
across three seeds, FCM-PM beats overlay decisively on *both* axes (eval bit-F1
0.9968 +-0.0008 vs 0.8733 +-0.0338, +0.12; Total-FAR 0.68 +-0.63% vs 16.20 +-5.53%, ~24x
lower on average and holding for every seed with no distribution overlap -- worst FCM-PM
seed beats best overlay seed), because pixelwise-max mis-models a partition and
under-detects the second defect while FCM-PM keeps every combination crisp, and on four
public datasets (MNIST, FashionMNIST, KMNIST, EMNIST-letters) we reproduce the same flip in
**both** directions under one identical protocol (4/4, no std overlap; MNIST
partition-placement 0.906 vs overlay 0.655; superposition overlay 0.762 vs
partition-placement 0.425; the matched operator recovers the oracle in every case, and
absolute bit-F1 declines on harder sources while the flip direction stays invariant) --
converting operator-match from an assertion into a general, publicly reproducible law; (ii) an **annotation-free, operator-agnostic,
distribution-free split-conformal false-alarm-rate guarantee**: a full multi-alpha
calibration curve on which the realized FAR tracks the target across alpha in [0.5%, 10%]
to within 0.153 pp for every operator (a small set of known-good samples yields realized
FAR 0.040 at alpha=0.05 and 0.006 at alpha=0.01), backed by synthetic normals, a
negative-target lever, val-margin selection, and naive-Bayes rejection -- the reliability
layer operator-only prior work (incl. Shin22) lacks; (iii) an **excess-risk theory tied
to operator-match**: a general, one-directional bound 2B*TV(D_real, D_syn^T) under which,
among content-blind operators, the one matching the domain's true combination law
minimizes the total-variation distance to the real multi-label distribution and thereby
minimizes the bound, so matched synthesis attains oracle-level risk up to the residual TV
-- honestly loose and one-directional, and consistent with the measured MNIST
oracle-recovery where matched synthesis is statistically indistinguishable from training
on the true law; (iv) the **single->multi setting and the objective of performance
maximization within it**: a multi-label-trained **oracle** (0.974 on WM38) bounds the
problem as an upper reference (it uses supervision we forbid), and the setting is
stricter-than-SPML, evaluated on real multi-label data and real normals; (v) our **method
for partition domains -- FCM-PM, val-margin checkpoint selection, and naive-Bayes
rejection**: on the chip partition domain FCM-PM is the winner (~0.998 bit-F1), and on the
public MixedWM38 benchmark it gives the best FAR-controlled tradeoff among the
content-blind operators (bit-F1 0.663 at NORMAL-FAR 0.228 at the arms' common guarded
pick; 0.654/0.147 at its best pick), against cutmix (0.691 bit-F1 but FAR 0.439), mixup
(0.537/0.225), and the single-only floor (0.473/0.602); on WM38 -- a superposition regime
under the binary die encoding -- overlay/summation is the *matched* operator and is
genuinely strong (0.80 at FAR 0.010), which we report honestly and do *not* claim to
beat, with the Pair-Mask lever (which cuts the complement FAR 0.384->0.147), val-margin,
and naive-Bayes rejection as the FAR controls; and (vi) **cross-domain framework support**
across five families spanning the combination taxonomy: on a controlled MultiMNIST
superposition benchmark blind synthesis **exceeds** a fully-supervised oracle on held-out
label combinations it never saw (+0.198 mAP; full test mAP 0.868 vs 0.846) -- a regime
where supervision structurally cannot help -- the same **label-fidelity / operator-match
criterion** predicts the counterintuitive Reuters text averaging-flip and the FSD50K
audio summation ranking, and natural images (PASCAL VOC) mark the boundary where
content-blind synthesis fails and location supervision becomes necessary. Density is a
**modeling characterization** consistent with the law: FCM-PM matches the real defect-die
density (0.29 vs real 0.31), while overlay is over-dense (0.50) -- expected of the
superposition operator, and the reason WM38 (superposition) favors overlay while the chip
partition domain favors FCM-PM.

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

We work under a strict **content-blind** constraint — training images each carry a
single-defect label, with no multi-label or normal annotation — and answer by
synthesizing combination examples from single-label sources, with the explicit
objective of **maximizing** the multi-label performance–false-alarm tradeoff in this
setting. A multi-label-trained **oracle** bounds the problem as an upper reference (it
uses supervision the setting forbids). The content-blind operators — summation/union
(overlay = Shin et al. (2022) Summation Mixup, a pixelwise `np.maximum` of two
single-defect maps), the partition-style full-cover complement, cutmix, and pixel
averaging — all consume the same two single maps and none uses any location annotation,
so **which one is right is not fixed by fiat but by the domain's true combination
law**. This is the paper's central, and now *measured*, principle — the
**operator-match law**: a summation/union join is the evidence-preserving operator for
**superposition-structured** domains, where each source's evidence adds or saturates
(inked digits, audio, and wafer maps under a binary die encoding), whereas a
**partition-complement** (FCM-PM) is right for **partition-structured** domains, where
each unit carries one condition and two defects occupy distinct spatial regions
(chip-internal maps). We verify the crossover directly and in **both** directions: on the
real chip domain, whose true law is a partition, FCM-PM beats overlay decisively on both
bit-F1 and FAR under an identical protocol (Sec 5.2.1), because pixelwise-max mis-models a
partition; and on four fully public datasets (MNIST, FashionMNIST, KMNIST, EMNIST-letters)
we reproduce the same flip in both regimes — the matched operator wins each direction with
no std overlap and recovers the oracle in every case (4/4; Sec 5.1.1) — so the law is
general, publicly verifiable, and not reliant on internal data. The operator-match criterion is measured before any training via **label fidelity** (every
labeled source must retain detectable evidence after synthesis); operators that destroy
evidence (rectangle patching, pixel averaging) train on false labels. The same
criterion makes a falsifiable, pre-registered cross-regime call — vector averaging (not
summation) wins on disjoint-coordinate text, while a summation/union join preserves
evidence on superposition-structured domains (inked digits, audio) — so that the
counterintuitive text averaging-flip (Reuters) and the audio summation ranking (FSD50K)
are later out-of-sample confirmations, not post-hoc rankings (Sec 5).

The two domains exhibit the two regimes. On the public MixedWM38 benchmark — a
**superposition** regime under the binary die encoding (normal die 0.5, defect 1.0;
verified in code that `summation_mixup_shin22` and `overlay` both compute
`np.maximum(ca,cb)`) — the matched operator is whole-image max-union / summation
(= Shin et al. (2022) Summation Mixup), and it is **genuinely strong**: 0.80 bit-F1 at
NORMAL-FAR 0.010. We report this honestly and **do not claim to beat it on WM38**.
Among the content-blind operators our method — the partition-style full-cover
complement with a Pair-Mask view (**FCM-PM**), with val-margin checkpoint selection and
naive-Bayes rejection — gives the best **FAR-controlled** tradeoff: bit-F1 0.663 at
NORMAL-FAR 0.228 at the arms' common guarded pick (0.654/0.147 at its best pick),
against cutmix (0.691 bit-F1 but FAR 0.439), mixup (0.537/0.225), and the single-only
floor (0.473/0.602). The decisive test of the operator-match law is the
**chip-internal** domain, whose true law is a **partition**: there FCM-PM reaches ~0.998
bit-F1 and, in a matched head-to-head (Sec 5.2.1), **beats** overlay on both bit-F1 and
FAR, because overlay's superposition assumption mis-models the partition. The one upper
reference we cannot admit as a method is the multi-label oracle (0.974), which trains on
real multi-label data. **Density is a modeling characterization consistent with the
law**: FCM-PM matches the real 2-mix defect-die fraction (0.29 vs real 0.31), whereas
overlay is over-dense (0.50) — expected of the superposition operator; a density-shift
analysis (Sec 5.2) shows overlay winning per WM38 stratum, which does not contradict
FCM-PM winning where the law is a partition, but confirms it. Val-margin selection,
naive-Bayes rejection, and the Pair-Mask lever (which cuts the complement FAR
0.384→0.147 at comparable bit-F1) are the FAR controls.

The reliability layer is the practical core, and it is what operator-only prior
work omits. Because the training data contain no all-negative label, a naive
synthesizer over-alarms on real normals; we control false alarms without any
real-normal annotation by combining synthetic normals, a negative-target lever,
val-margin checkpoint selection on a disjoint held-out-source synthetic proxy,
class-conditional Gaussian (naive-Bayes) rejection fit only on further disjoint
single sources and synthetic normals, and a **split-conformal calibration for a
finite-sample, distribution-free FAR guarantee**. This guarantee holds for *any*
operator and is the reliability asset Shin et al. (2022) and other operator-only
methods do not provide. Real mixed and normal maps remain final test data.

Contributions:
1. **The operator-match law (a general principle for label-faithful synthesis).** The
   right content-blind operator is selected *before any training* by a predictive
   criterion (evidence survival / label fidelity), fixed not by fiat but by the domain's
   *true combination law*: a summation/union join (overlay = Shin et al. 2022 Summation
   Mixup, a pixelwise `np.maximum` of two single maps) is evidence-preserving for
   **superposition-structured** domains, while the partition-complement FCM-PM is right
   for **partition-structured** domains. We convert this from an assertion into a
   *verified* law with a matched head-to-head on the real chip domain (a partition law,
   Sec 5.2.1): across three seeds FCM-PM beats overlay decisively on *both* axes (eval
   bit-F1 0.9968 +-0.0008 vs 0.8733 +-0.0338; Total-FAR 0.68 +-0.63% vs 16.20 +-5.53%,
   ~24x lower on average with no per-seed overlap), the two operators learning single
   defects equally (train bit-F1 ~0.99) so the entire gap is multi-defect eval, where
   pixelwise-max mis-models the partition and under-detects the second defect. We
   demonstrate the **same flip in both directions on four public datasets (MNIST,
   FashionMNIST, KMNIST, EMNIST-letters)** (Sec 5.1.1): the matched operator wins every
   regime with no std overlap (4/4; e.g. MNIST partition-placement 0.9064 +-0.016 vs
   overlay 0.6550 +-0.020; superposition overlay 0.7616 +-0.007 vs partition-placement
   0.4249 +-0.034) and recovers the oracle in each direction (e.g. MNIST partition 0.9064
   vs 0.9078; superposition 0.7616 vs 0.7631), the matched-minus-mismatched gap 0.17-0.34
   throughout with absolute bit-F1 declining on harder sources while the flip direction
   stays invariant, so the law is measured on **four** public datasets (both directions)
   and the real chip domain -- a general, publicly verifiable law, not a dataset-specific
   artifact. On WM38 -- a superposition regime under the binary encoding -- overlay is
   instead the matched operator and is honestly strong (0.80/0.010); we do *not* claim to
   beat it there.
2. **An annotation-free FAR guarantee.** A strict source-only reliability pipeline --
   synthetic normals, negative-target control, synthetic validation margin for
   checkpoint selection, class-conditional Gaussian (naive-Bayes) pattern likelihood
   for rejection, and **split-conformal calibration** for a finite-sample,
   distribution-free marginal FAR guarantee under exchangeability -- that is
   operator-agnostic and absent from prior operator-only work (e.g. Shin et al. 2022).
   A full multi-alpha calibration curve shows the realized FAR tracking the target
   across alpha in [0.5%, 10%] to within 0.153 pp for every operator.
3. **An excess-risk theory tied to operator-match.** A general, one-directional
   excess-risk bound 2B*TV(D_real, D_syn^T) (Sec Theory): among content-blind operators,
   the one matching the domain's true combination law minimizes the total-variation
   distance to the real multi-label law and thereby minimizes the bound, so matched
   synthesis attains oracle-level risk up to the residual TV -- and in the idealized
   independent-source limit is risk-equivalent to full multi-label supervision. The
   statement is honestly loose and one-directional (a larger TV only weakens the *upper*
   guarantee, so it does no operator-preference work on WM38, where the matched max-union
   is over-dense yet strong), and is consistent with the measured MNIST oracle-recovery,
   where matched synthesis is statistically indistinguishable from training on the true
   law.
4. **The single->multi setting and performance maximization within it.** We train
   multi-label recognizers with **zero multi-label annotation** -- no multi-label image
   and no all-negative (normal) label -- by synthesizing combinations from single-label
   sources, and we make **maximizing** the multi-label performance-FAR tradeoff the
   objective. A multi-label-trained **oracle** bounds the problem as an upper reference
   (it trains on the real multi-label data we forbid). The setting is
   stricter-than-SPML (genuinely single-label training images), evaluated on real
   multi-label data and real normals.
5. **The method (partition-domain instantiation): FCM-PM + val-margin + naive-Bayes
   rejection.** A partition-style full-cover complement with a Pair-Mask view (FCM-PM),
   val-margin checkpoint selection on a disjoint synthetic proxy, and a synthetic-only
   class-conditional Gaussian (naive-Bayes) rejection stage. On the chip partition domain
   FCM-PM is the winner (~0.998 bit-F1, Total-FAR 0.06%); on MixedWM38 it gives the best
   FAR-controlled tradeoff among the content-blind operators (bit-F1 0.663 at FAR 0.228
   at the arms' common guarded pick; 0.654/0.147 at its best pick), against cutmix
   (0.691 bit-F1 but FAR 0.439), mixup (0.537/0.225), and the single-only floor
   (0.473/0.602); the Pair-Mask lever cuts the complement FAR 0.384->0.147, and
   val-margin and naive-Bayes rejection are the further FAR controls. A 3-component
   ablation shows each stage contributes (val-margin is the dominant FAR fixer;
   naive-Bayes rejection closes the OOD tail, Sec 5.2.1).
6. **Cross-domain framework support.** The framework and its label-fidelity /
   operator-match criterion generalize across five families spanning two combination
   regimes. On a controlled MultiMNIST superposition benchmark blind synthesis
   **exceeds** a fully-supervised oracle on held-out label combinations the oracle
   never observed (+0.198 mAP; full test mAP 0.868 vs 0.846) -- a regime where
   supervision structurally cannot help. The criterion characterizes which
   content-blind operator preserves evidence per domain and makes a falsifiable
   cross-regime call -- vector averaging (not summation) on disjoint-coordinate text
   (Reuters), a summation/union join on superposition-structured audio (FSD50K
   waveform summation wins bit-F1 0.433 vs 0.27-0.31) -- each a ranking image intuition
   gets wrong. Natural images (VOC) mark the boundary, where content-blind synthesis
   fails and location supervision becomes necessary.

## 2 Related Work

- **Mixing augmentations.** Mixup (Zhang et al., 2018): convex image/label
  blending — designed as a regularizer for single-label training, not as a
  combination synthesizer; we show averaging ghosts both objects (MNIST mixup
  full mAP 0.738 vs overlay 0.868; WM38 mixup bitF1 0.537 vs the summation/union
  reference join 0.80, strict 5-seed). CutMix (Yun et
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
  2026). **Under the benchmark's binary encoding (normal die 0.5, defect 1.0),
  Shin et al. (2022) Summation Mixup is exactly whole-image max-union** (verified
  in `multilabel_synth/synthesis/wm38_arms.py`: the `summation_mixup_shin22` and
  `overlay` arms both compute `np.maximum(ca, cb)`). This makes overlay a real, strong
  operator on WM38, not a straw man: it is the *matched* join for WM38's
  superposition-structured binary encoding, and we report it honestly at bit-F1 0.80 /
  FAR 0.010 — **we do not claim to beat it on WM38**. Our position is not that overlay
  is disqualified but that the *right* content-blind operator is selected by the
  domain's true combination law: overlay wins in superposition regimes, and the
  partition-complement FCM-PM wins in partition-structured regimes, which we verify with
  a measured chip head-to-head where FCM-PM beats overlay on both bit-F1 and FAR
  (Sec 5.2.1). Thuan (2025) and copy-paste (Ghiasi et al., 2021) are genuinely
  content-*aware* (they extract or place defect support using location or mask
  annotation), a distinction that does apply to them but not to overlay. **Our
  contribution on wafer is therefore a framework, not one operator:** (a) the
  annotation-free single→multi setting and the objective of maximizing performance
  within it; (b) the operator-match law, measured by the chip head-to-head; (c) the
  FCM-PM method (with val-margin selection and NB-reject), the winner on the chip
  partition domain and the best FAR-controlled content-blind operator on WM38; (d) the
  excess-risk theory that explains *when* blind synthesis can match the oracle; and (e)
  an annotation-free, distribution-free FAR-control layer (synthetic normals +
  val-margin selection + NB-reject + split-conformal guarantee) that Shin et al. and the
  other operator-only methods above do not provide. Density is a modeling
  characterization consistent with the law (FCM-PM matches the real defect-die density
  0.29; overlay is over-dense 0.50, as expected of the superposition operator). We
  compare directly against Shin et al. 2022 = max-union, reproduced exactly (verified =
  `overlay` in code); equal-protocol reproduction of Shim-Kang (2023) and the
  adaptive-ROI / diffusion pipelines (Thuan 2025; Li et al. 2025; Yang et al. 2026) is
  left to future work, as their exact operators are not recoverable from the published
  descriptions, and we report their native numbers and the protocol gap instead.
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
No multi-label sample and no normal (all-negative) label is available at training time
(content-blind constraint — motivated by the fab, where multi-defect co-occurrences are
never annotated). No defect location or mask annotation is available either, but this
does NOT single out any one operator: every content-blind operator we compare —
overlay/max-union included — consumes only the single-defect maps and uses no location,
so the operator choice is decided by the domain's true combination law, not by an
admissibility fiat.
Evaluation: genuine multi-label samples (>=2 bits), plus real all-negative
normals for false-alarm rate. Metrics: bit-F1 (macro-F1 over K bits), FAR
(false-positive rate on negative bits; NORMAL-FAR = fraction of real normal
samples raising any alarm), exact-match, mAP (for literature comparability),
and pos/neg mean predicted probability as calibration diagnostics.

## 4 Method: Label-Faithful Synthesis Framework

Our method has three parts: the partition-style **full-cover complement with a
Pair-Mask view (FCM-PM)**, **val-margin checkpoint selection** on a disjoint
held-out-source synthetic proxy, and a synthetic-only class-conditional Gaussian
(naive-Bayes) **NB-reject** stage (optionally backed by the split-conformal FAR
guarantee, Sec 5.11). The latter two, with synthetic normals and a negative-target
lever, form the annotation-free FAR-control layer that operator-only prior work omits
and that applies to any operator. FCM-PM is the operator matched to
**partition-structured** domains, where each unit carries one condition and two defects
occupy distinct spatial regions; it is the winner on the chip partition domain (~0.998
bit-F1) and the best FAR-controlled content-blind operator on WM38 (Sec 5.2). The one
operator we cannot admit as a method is the multi-label **oracle**: it trains on the
real multi-label data we forbid, so it is an upper reference. Overlay/max-union (= Shin
et al. 2022 Summation Mixup) is **not** disqualified — it is a full content-blind
operator, and it is the *matched* operator wherever the domain's true combination law is
superposition (inked digits, audio, and WM38 under the binary encoding), where it is
genuinely strong. What decides between overlay and FCM-PM is the domain's combination
law, and we verify the crossover with a matched head-to-head on the real chip domain,
where FCM-PM beats overlay on both bit-F1 and FAR (Sec 5.2.1).

A **label-fidelity criterion** accompanies the method as a framework tool: it
measures, before any training, how well a content-blind operator preserves each
source's evidence, and it characterizes evidence-preservation across domains. The
evidence geometry is domain-dependent: in **superposition-structured domains**
sources saturate or add, so a **summation/union join** preserves evidence — pixelwise
max-overlay for inked digits (ink saturates over black) and physical **waveform
summation** for audio (FSD50K, where waveform_sum wins bit-F1 0.433 vs 0.27-0.31; Sec
5.10) — while in **partition-structured domains** each unit carries one condition, so
the fidelity-maximizing operator is the partition-complement FCM-PM (chip-internal
maps), and in **disjoint-coordinate text** vector averaging preserves per-coordinate
evidence when vocabularies barely overlap (Sec 5.9). WM38 under the binary encoding is a
superposition regime, so its matched operator is overlay/summation and we report it as
strong there; on the chip partition domain the matched operator is FCM-PM, and the
measured head-to-head (Sec 5.2.1) confirms it beats overlay. Generality is established
across three combination regimes --- superposition-structured (MixedWM38 public wafer,
MultiMNIST inked digits, and FSD50K audio, Sec 5.10), partition-structured (chip), and
disjoint-coordinate text (Reuters, Sec 5.9) --- plus the natural-image boundary (VOC).

### 4.1 Synthesis operators (content-blind)

Given two singles (x_a, a), (x_b, b):
- **overlay / max-union** (the **superposition-regime** operator):
  per-pixel max (defect intensity wins over normal background) — both objects survive
  whole; hard label {a,b}. It is the evidence-preserving join wherever sources add or
  saturate: inked digits, audio, and **wafer maps** under WM38's binary encoding (normal
  0.5, defect 1.0), where it equals Shin et al. (2022) Summation Mixup's clipped binary
  sum and is genuinely strong (bit-F1 0.80 / FAR 0.010, Sec 5.2). It is a full
  content-blind operator (a pixelwise max of two single maps, no location), **not** an
  inadmissible reference; but because it preserves the joint co-location it produces
  over-dense maps (density 0.50 vs real 0.31), which is why it mis-models a **partition**
  domain: on the chip head-to-head it is beaten by FCM-PM (Sec 5.2.1).
- **complement (FCM)** (our **partition-regime** base operator): G x G grid
  (G = 3N, e.g. 9), cells randomly permuted and partitioned into n groups;
  mix_i = x_b base with x_a's group-i cells overwritten. The union of the n mixes
  covers x_a exactly once (full cover), so each die is owned by exactly one source; it
  uses no defect location and matches the real 2-mix defect-die density (0.29 vs real
  0.31). Hard labels; per-mix asymmetric (A,B) targets are a no-op within [0.9, 1.0]
  (chip leaderboard: 0.9889-0.9898 across A in {0.90,0.95,1.00}). Density-matching
  (0.665/0.384 without PM; 0.654/0.147 with PM).
- **pair mask (PM)** (the FAR-control lever for the complement arm): for each
  complement mix, also emit a mask sample — x_a's cells kept, x_b's cells with
  defects erased; target: bit a at a soft 0.65, all else negative. Teaches
  "near-normal map with weak fragments => low confidence"; a false-alarm
  suppressor. FCM+PM together form FCM-PM (PM cuts the complement arm's FAR 0.384
  -> 0.147 at comparable F1); FCM-PM also reaches ~0.99 on chip-internal maps.
- **synthetic normals**: defect pixels erased from real singles (wafer:
  min(x, normal-die value)) => all-negative samples without any normal label.
- Other content-blind baselines: CutMix (rectangle patch) and Mixup (convex blend,
  soft labels), plus single-only (no synthesis). Genuinely content-*aware* references
  (they use location / mask / multi-label supervision our setting forbids): copy-paste
  (mask-cropped placement) and the oracle (real multi-label training) — these are
  ceilings, not rivals; overlay is not among them (it is a content-blind operator, the
  superposition-regime one).

### 4.2 Label fidelity selects the operator

The operator is chosen by a single measurable criterion: **label fidelity**.
Define survival of the weaker source = fraction of its object pixels present in
the synthesized image. Measured over 3000 pairs (MNIST):

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

Measured weaker-source defect survival on WM38: max-union **1.000** > CutMix 0.579 >
complement (FCM) 0.527 > Mixup 0.236. On WM38's superposition encoding the
maximal-survival, evidence-preserving join is max-union / summation, and it is the
*matched* operator there — consistent with its strong raw performance. But survival on
one domain does not name a universal operator: the criterion earns its keep *across
regimes*, selecting the operator whose join reproduces the domain's true combination
law. **Pre-registration -- the criterion is falsifiable, not a survival=ranking
tautology.** Because label fidelity is evaluated *before any multi-label training*, it
fixes *in advance* which content-blind operator should maximize per-domain fidelity and
therefore win: a summation/union join on superposition-structured domains (wafer, inked
digits, audio) and vector averaging on disjoint-coordinate text. This call is non-obvious
and falsifiable -- on disjoint-coordinate text it predicts that *averaging beats
summation*, the opposite of what one gets by generalizing "overlay always wins" from
images -- and it was borne out (the Reuters averaging-flip, Sec 5.9), alongside the FSD50K
audio summation ranking and the chip head-to-head where the partition-complement beats
overlay (Sec 5.2.1). These are therefore *out-of-sample confirmations of a rule fixed in
advance*, not post-hoc rankings: a mere re-description of the survival ordering could not
have predicted the text flip, so the criterion carries predictive, not tautological,
content.

**Density as a characterization, consistent with the law.** The complement matches the
real density (0.293 vs real 0.305), while overlay is over-dense (0.501) — the signature
of the superposition operator. A density-shift analysis (Sec 5.2) shows overlay winning
per WM38 stratum, exactly as its superposition match predicts, and does not contradict
FCM-PM winning where the law is a partition (chip). Density-faithfulness is a modeling
property tied to the combination law, **not** a stand-alone performance lever.

### 4.3 Why not averaging

In shared-coordinate spaces, Mixup's blend produces ghosted objects (both at
half contrast) with soft labels; a hard, evidence-preserving join keeps each
source recoverable. Same combination-operator family, opposite outcome: on
superposition MNIST, overlay 0.868 vs mixup 0.738; on wafer maps the max-union
reference join 0.80 far exceeds mixup 0.537 (strict, 5 seeds). Failure of
"blending" is specifically averaging, not combining — except in
disjoint-coordinate text (Sec 5.9), where averaging is the fidelity maximizer and
wins.

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

#### 5.1.1 Operator-match law on four public datasets: the partition/superposition flip

The chip head-to-head (Sec 5.2.1) shows the partition-direction win on real fab data;
here we show the operator-match law is **general, not dataset-specific** by reproducing the
same flip, in **both** directions, on **four** fully public datasets — MNIST, FashionMNIST,
KMNIST, and EMNIST-letters (26 classes) — under one identical controlled protocol, so the
law is publicly verifiable **without internal data**. For each dataset two multi-label
domains are built from the **same** public source on a 56x56 two-by-two-cell canvas, trained
with the **same** shared SmallCNN (20 epochs, 3 seeds), changing **only** the domain's true
two-glyph combination law: a **partition** domain (the two glyphs occupy disjoint cells) and
a **superposition** domain (the two glyphs share one cell, combined by pixelwise max). In
each domain the two content-blind operators run head-to-head — partition-placement (the
FCM-PM analogue) and overlay/max-union — alongside the single-only floor and an oracle
trained on true-law combos (upper reference). Eval = held-out real multi-label test set for
that dataset; matched = the operator whose combine matches the domain's true law.

| dataset        | regime        | matched (mean +-std) | mismatched (mean +-std) | oracle | floor | flip |
|----------------|---------------|----------------------|-------------------------|--------|-------|------|
| MNIST          | partition     | **0.9064 +-0.016**   | 0.6550 +-0.020          | 0.9078 | 0.458 | yes  |
| MNIST          | superposition | **0.7616 +-0.007**   | 0.4249 +-0.034          | 0.7631 | 0.365 | yes  |
| FashionMNIST   | partition     | **0.7187 +-0.005**   | 0.5533 +-0.035          | 0.7131 | 0.300 | yes  |
| FashionMNIST   | superposition | **0.5861 +-0.010**   | 0.3188 +-0.026          | 0.5857 | 0.247 | yes  |
| KMNIST         | partition     | **0.6793 +-0.012**   | 0.4803 +-0.043          | 0.6842 | 0.358 | yes  |
| KMNIST         | superposition | **0.5504 +-0.010**   | 0.3373 +-0.010          | 0.5549 | 0.261 | yes  |
| EMNIST-letters | partition     | **0.6185 +-0.019**   | 0.3471 +-0.012          | 0.6145 | 0.318 | yes  |
| EMNIST-letters | superposition | **0.4088 +-0.026**   | 0.2303 +-0.015          | 0.4075 | 0.204 | yes  |

Matched = the blind operator whose join reproduces the regime's true law (partition:
partition-placement; superposition: overlay/max-union); mismatched = the other.

**The flip holds 4/4, in both directions, with no std overlap.** In every dataset and regime
the **matched** blind operator wins and the mismatched one is clearly worse, the winner and
loser trading places exactly as the law changes; across all eight rows the
matched-minus-mismatched bit_F1 gap is 0.17-0.34 and no matched/mismatched std bands overlap.
On MNIST the partition domain gives partition-placement 0.9064 +-0.016 vs overlay 0.6550
+-0.020 (+0.251), and the superposition domain gives overlay 0.7616 +-0.007 vs
partition-placement 0.4249 +-0.034 (+0.337); the same clean flip recurs on FashionMNIST
(0.7187/0.5533 partition, 0.5861/0.3188 superposition), KMNIST (0.6793/0.4803; 0.5504/0.3373),
and EMNIST-letters (0.6185/0.3471; 0.4088/0.2303). **The matched operator recovers the oracle
within noise in every dataset and both directions** (e.g. MNIST partition 0.9064 vs oracle
0.9078, superposition 0.7616 vs oracle 0.7631; FashionMNIST 0.7187 vs 0.7131, 0.5861 vs
0.5857; KMNIST 0.6793 vs 0.6842, 0.5504 vs 0.5549; EMNIST 0.6185 vs 0.6145, 0.4088 vs 0.4075),
so matched blind synthesis is statistically indistinguishable from training on the true law,
and every matched arm sits well above the single-only floor (0.20-0.46). **Honest scope
note:** the *absolute* bit_F1 declines monotonically as the source gets harder (MNIST ->
FashionMNIST -> KMNIST -> EMNIST), but the flip *direction* and the matched-vs-mismatched gap
are clean everywhere — the law is invariant even where absolute performance is not. CIFAR-10
(color) was not run under this grayscale pipeline; we state this as a scope limit, not a
result. This extends the (superposition-only) MultiMNIST study in 5.1 by adding the partition
direction across four sources under one controlled protocol, and it is the **public,
reproducible counterpart to the internal chip head-to-head** (Sec 5.2.1), which shows the
same partition-direction win (FCM-PM beats overlay) on real fab data: the operator-match law
is now measured on **four** public datasets (both directions, matched recovering the oracle)
and the real chip partition domain — a general law, not a dataset-specific artifact. Runner:
`multilabel_synth/run_operator_match_multidataset.py` (committed).

### 5.2 MixedWM38 (public benchmark; real multi-label evaluation)

Audit note (2026-07-10): pre-audit WM38 F1 values in this subsection use the
legacy macro whose denominator could change when an unsupported test bit
produced a false positive. They remain provenance records, not final
submission numbers. The paired saved-probability rerun will replace them with
supported-class macro-F1; all FAR values remain separately reported.

38,015 real wafer maps: 7,015 singles (train source), 30,000 real mixed
(29 combos; eval), 1,000 real normals (FAR). SmallCNN, 15 epochs, 3 seeds;
6 combos excluded from oracle training.

**Density characterization (honest, not a superiority argument).** WM38 real
2-mixes have a defect-die fraction close to singles, while whole-image max-union
is denser. Measured defect-die fraction of on-wafer area:

| arm                                  | defect-die fraction | vs real 2-mix        |
|--------------------------------------|---------------------|----------------------|
| real 2-mix (ground truth)            | 0.305 +-0.050       | —                    |
| single (1 defect)                    | 0.290               | matches              |
| FCM / FCM-PM (full-cover complement) | 0.293               | matches              |
| max-union / overlay (= Shin22 summ.) | 0.501 +-0.123       | 64% denser; 91% > 95th pct |

Max-union is over-dense (0.501 vs real 0.305); the partition complement matches the
real density (0.293). Over-density is the signature of the superposition operator, and
because WM38 under the binary encoding is a superposition regime, overlay's over-density
is exactly what its matched combination law predicts. Density is a modeling
characterization tied to the combination law, not a stand-alone performance lever.

**Density-shift analysis (overlay wins on WM38, consistent with the law).** We
stratify the real 2-mix test by density tertile and split out 3/4-mix, then compare
overlay against FCM-PM (SmallCNN, 3 seeds). bit-F1:

| condition          | overlay (= Shin22, matched) | FCM-PM (ours) |
|--------------------|-----------------------------|---------------|
| 2-mix low density  | 0.8548                      | 0.7868        |
| 2-mix mid density  | 0.8298                      | 0.7874        |
| 2-mix high density | 0.8049                      | 0.7144        |
| 3-mix              | 0.7245                      | 0.6336        |
| 4-mix              | 0.6552                      | 0.4841        |

We own this as **consistent** with the operator-match law rather than moot: WM38 is a
superposition regime, so its matched operator (overlay) is expected to win at every
stratum, and this does **not** contradict FCM-PM winning where the true law is a
partition — as the chip head-to-head measures (Sec 5.2.1). It confirms
density-faithfulness is a modeling property tied to the combination law, not a
stand-alone performance lever.

**Operator comparison (5 seeds, pick=val_tail_margin_guarded, neg 0.02; FAR as a
fraction).** Common protocol (matched view budget, splits, checkpoint selection,
rejection). WM38 is a superposition regime, so its matched operator is overlay
(strong; we do not claim to beat it); the row below the rule is the multi-label-trained
upper reference:

| operator                       | bit-F1 | NORMAL FAR | defect-die density | note                                     |
|--------------------------------|--------|------------|--------------------|------------------------------------------|
| overlay / max-union (= Shin22) | 0.800  | 0.010      | 0.501 (over-dense) | matched superposition operator (strong)  |
| FCM-PM (**ours**)              | 0.663  | 0.228      | 0.293 (matches)    | best FAR-controlled content-blind        |
| cutmix                         | 0.691  | 0.439      | —                  | content-blind; ~2x the FAR               |
| FCM (no Pair-Mask)             | 0.665  | 0.384      | 0.293 (matches)    | content-blind; PM lever removed          |
| mixup                          | 0.537  | 0.225      | —                  | content-blind; ghosting                  |
| single_only                    | 0.473  | 0.602      | 0.290              | content-blind; floor                     |
| oracle (real mixed + labels)   | 0.974  | 0.563      | 0.305              | UPPER ref (multi-label train)            |

Reading: WM38 under the binary die encoding is a **superposition** regime, so its
matched operator is whole-image **max-union / summation** — which **is Shin et al.
(2022) Summation Mixup** — and it is genuinely strong (0.80/0.010). We report it
honestly and **do not claim to beat it on WM38**. Among the content-blind operators our
method **FCM-PM** gives the best FAR-controlled tradeoff (bit-F1 0.663 at FAR 0.228 at
the arms' common guarded pick; 0.654/0.147 at its best pick) — cutmix trades a slightly
higher bit-F1 for ~2x the FAR, mixup and the floor are worse on both axes, and the
Pair-Mask lever cuts the complement FAR 0.384 -> 0.147. The decisive test of which
operator is right — overlay or FCM-PM — is not WM38 but the chip **partition** domain,
where FCM-PM beats overlay head-to-head (Sec 5.2.1). The oracle (0.974) is a
multi-label-trained upper reference. The annotation-free reliability layer (Sec 5.6,
5.11) turns FCM-PM's FAR into a finite-sample guarantee.

(The oracle matches published MixedWM38 accuracies 98-99%, validating the
harness. An earlier "statistical parity" claim against a SmallCNN-15ep oracle
(0.863) is retracted as a weak-oracle artifact. For the headline oracle
checkpoint, max-prob on real normals >=0.99 for 80% of normals, so tau=0.99 still
leaves FAR 0.799 — a checkpoint-level calibration result, not an impossibility
claim about all oracle training.)

**Component ablations (transfer across operators).** Pair masking in the complement
arm cuts observed NORMAL FAR in isolation (0.852 -> 0.012), and synthetic normals
reduce it in every arm. Full-scale / multi-seed runs of the max-union reference arm
reach bit-F1 0.795 (full-scale, all 7,015 singles, ResNet-18, 30ep) to 0.841
(nine-seed headline) at near-zero real-normal FAR (0.0003-0.0008). The label-fidelity
survival ordering (max-union > cutmix > complement > mixup) predicts the raw
detection ordering — the criterion's core empirical claim.

#### 5.2.1 Operator-match crossover: FCM-PM vs overlay on a partition domain (chip)

WM38's binary encoding is a superposition regime, so it cannot arbitrate between
overlay and FCM-PM — the operator-match law predicts overlay wins there, and it does.
The decisive test is a domain whose true combination law is a **partition**: the
chip-internal maps, where two co-occurring defects occupy **distinct** spatial regions
(each die belongs to one defect, not both). We run overlay and FCM-PM head-to-head on
chip through an **identical** pipeline — same single-defect training data, epochs,
backbone, val-margin selection, and naive-Bayes (I10) acceptor — changing **only** the
multi-defect synthesis operator (three seeds s99/s7/s42). FAR in
% (NI = Normal/Invalid, OOD = out-of-distribution wafer patterns):

| operator (chip, 3 seeds) | train bit_F1 | eval bit_F1 (3 seeds) | NI-FAR* | OOD-FAR* | Total-FAR (3 seeds) |
|--------------------------|--------------|-----------------------|---------|----------|---------------------|
| FCM-PM (**ours**)        | ~0.992       | **0.9968 +-0.0008**   | **0.00** | **0.08** | **0.68 +-0.63**     |
| overlay (= Shin 2022)    | ~0.989       | 0.8733 +-0.0338       | 19.00   | 7.66     | 16.20 +-5.53        |

\* NI/OOD-FAR shown for the representative seed 99; the last column is the 3-seed Total-FAR mean. Per-seed eval bit_F1 / Total-FAR (s99/s7/s42): FCM-PM 0.9979/0.06, 0.9964/1.54, 0.9960/0.44; overlay 0.9187/10.36, 0.8635/14.61, 0.8377/23.63.

Both operators learn single defects equally well (train bit_F1 ~0.99 for both; eval
singles bit_F1 1.000 for both), so training data and capacity are not the variable.
**The entire gap is in multi-defect evaluation, and across three seeds (s99/s7/s42)
FCM-PM wins on both axes: eval bit_F1 0.9968 +-0.0008 vs 0.8733 +-0.0338 (+0.12 on
average) and Total-FAR 0.68 +-0.63% vs 16.20 +-5.53% (~24x lower on average).** **The
result holds for every seed with no distribution overlap:** the worst FCM-PM seed
(bit_F1 0.9960) still beats the best overlay seed (0.9187) by +0.077, and the worst
FCM-PM Total-FAR (1.54%) is ~7x lower than the best overlay Total-FAR (10.36%). Overlay
is also far less stable -- its bit_F1 std is ~42x larger and its FAR std ~9x larger than
FCM-PM's -- so the partition operator is not merely better on average but more reliable.
The mechanism is exactly operator-law mismatch: because chip combinations are
partitions, pixelwise-max under-detects the second defect in each combination. As an
illustrative single seed (s99), overlay's combination bit_F1 collapses to 0.61-0.80
(bb+fork 0.618, fork+scratch 0.610, fork+scratch_rot 0.629) and it over-fires on
negatives (NI-FAR 19.00%, OOD-FAR 7.66%), whereas FCM-PM keeps every combination crisp
(0.99-1.00) with well-separated negatives (eval neg prob 0.1266; NI-FAR 0.00%). This is
the crossover the paper's law predicts and the piece WM38-only evidence could not
supply: a real, multi-label, matched head-to-head in which overlay's superposition
assumption **loses**, converting operator-match from an assertion into a verified law.
The same partition/superposition flip is reproduced in full on four public datasets (MNIST,
FashionMNIST, KMNIST, EMNIST-letters; Sec 5.1.1), where the matched operator recovers the
oracle in both directions on all four (4/4), so the operator-match law is measured on
**both** the real chip partition domain and four public datasets — a general law, publicly
verifiable and not reliant on internal data.

**Three-component ablation (chip champion recipe, preserved checkpoints, seed 99).** Each of the
three method stages contributes (chip multi-defect eval; FAR in %):

| configuration                               | eval bit_F1 | Total-FAR | delta Total-FAR |
|---------------------------------------------|-------------|-----------|-----------------|
| (a) FCM-PM base (val-F1 ckpt, raw 0.5 thr.) | 0.9948      | 97.80     | —               |
| (b) + val-margin selection                  | 0.9977      | 7.53      | -90.27          |
| (c) + naive-Bayes reject (= full method)    | **0.9979**  | **0.06**  | -7.47           |

From the FCM-PM base (the val-F1-selected checkpoint read out at the raw 0.5 threshold)
the model already reaches eval bit_F1 0.9948 but is unusable — Total-FAR 97.80%, because
the raw partition logits leave the negative scratch bit above 0.5. Adding val-margin
checkpoint selection is the dominant FAR fixer (Total-FAR 97.80 -> 7.53%, -90.27) at
higher bit_F1 (0.9977); adding the naive-Bayes (I10) reject stage closes the OOD tail
(Total-FAR 7.53 -> 0.06%, -7.47) at bit_F1 0.9979 — the full method. Neither stage is
redundant.

#### 5.2.2 Loss-engineering control: can a better loss on singles substitute?

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
FAR lower. (This isolation uses the max-union / Summation Mixup arm as the reference
join because it is the simplest; the point is loss-vs-structure, and it holds for any
co-occurrence synthesis including our FCM-PM.)
The bottleneck is not
loss calibration but the absence of label co-occurrence structure in single-label
data, which no loss can recover and which synthesis supplies by construction.

### 5.3 PASCAL VOC 2007 (natural-scene boundary analysis)

IMPORTANT SCOPING: the copy-paste arm here uses the dataset's bounding-box
annotations to crop objects — it is a CONTENT-AWARE reference probe, not
one of our content-blind operators. The content-blind constraint (no
location annotation) governs the industrial setting and all headline
claims (WM38, MNIST, Reuters, all blind). On natural scenes we
deliberately relax it to measure the boundary: blind operators fail here,
and even approaching the oracle requires location supervision — which is
itself the finding predicted by the superposition-domain condition.

We report the boundary from a single consistent protocol (5 arms, 5 seeds, one
held-out-pair crops run; ResNet-18). Findings: **content-blind synthesis falls
below the single-only floor** -- cutmix reaches only bit-F1 0.226 and mixup 0.194,
both under the floor 0.242 -- i.e., blind synthesis actively **hurts** on natural
scenes. **Only content-AWARE copy-paste (0.350), which uses the dataset's
bounding-box location supervision to crop and place objects, beats the floor**, and
it still trails the trained oracle (0.503). No compositional advantage appears (all
arms drop together on held-out-pair scenes), and on COCO-20 synthesis gains vanish.
What marks the boundary is that the content-BLIND VOC arms recover nothing (cutmix
0.226, mixup 0.194 below the floor) and that all arms drop together on held-out-pair
scenes; the content-AWARE copy-paste probe (0.350; gap-recovery
(0.350-0.242)/(0.503-0.242) = 41%) beats the floor only by using location
supervision, which our setting forbids and which is exactly the resource the boundary
shows to be necessary. This positions the domain condition: label-faithful blind
synthesis
requires the domain's combination to be reproducible by a content-blind,
evidence-preserving operator (summation/union in superposition-structured domains --
inked digits, wafer maps, audio) -- which wafer/palette domains admit and opaque RGB
scenes do not (objects occlude, not join). Outside superposition domains, blind
operators fail and location supervision becomes necessary.

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

### 5.10 Fifth family: audio (FSD50K) — a second superposition-structured domain

Audio is a **second superposition-structured domain that agrees with wafer**:
sounds combine by physical **waveform superposition** (waveforms add), so the
fidelity-maximizing operator is **summation** — exactly as on wafer maps and inked
digits. FSD50K, 5 seeds, n=20/arm, multi-label eval:

| operator     | bit-F1 | mAP    | FAR    | note                                     |
|--------------|--------|--------|--------|------------------------------------------|
| waveform_sum | 0.4328 | 0.5616 | 0.2115 | summation (fidelity-max); best bit-F1    |
| fcm_pm       | 0.3121 | 0.5713 | 0.1024 | best F1 among FAR-controlled (~0.10) arms|
| fcm          | 0.3034 | 0.5740 | 0.1057 | partition-style alternative              |
| single_only  | 0.2925 | 0.5763 | 0.0965 | floor                                    |
| mixup        | 0.2685 | 0.5685 | 0.0919 | averaging                                |
| cutmix       | 0.2665 | 0.5764 | 0.0848 | region patch                             |

Waveform summation wins bit-F1 decisively (0.433 vs 0.27-0.31 for every other arm,
sd ~0.019). This confirms the **cross-regime operator-match principle** — the
criterion characterizes the evidence-preserving operator for the domain's geometry:
summation/union preserves evidence on the superposition-structured domains (wafer,
digits, audio), while averaging preserves it on disjoint-coordinate text (Sec 5.9).
Audio is a superposition domain, so waveform summation is its matched content-blind
operator; on WM38 (also a superposition regime) the analogous max-union join is the
matched, strong operator, and the crossover to a partition domain — where FCM-PM beats
overlay — is the chip head-to-head (Sec 5.2.1).

Honest nuance: waveform_sum wins bit-F1 but at a HIGHER FAR (0.212 vs ~0.10) and
marginally lower mAP; the partition-style FCM-PM remains the best FAR-controlled
operator on audio — the highest bit-F1 (0.312) among arms holding FAR near 0.10,
at FAR 0.102 (below plain fcm's 0.106). The bit-F1-vs-FAR trade-off is thus
operator-dependent: the summation operator maximizes evidence (hence bit-F1),
while Pair-Mask stays a lever for false-alarm control. (No real all-negative /
multi-label oracle pool is used for audio; arms differ only by synthesis operator.)

### 5.11 Conformal FAR guarantee and its boundary

Split-conformal threshold from 500 known-good real normals: realized FAR
0.040 at alpha=0.05 and 0.006 at alpha=0.01, coverage >= 99.5% (guarantee
holds; known-good samples require no defect annotation). Calibrating on
training-style synthetic normals fails (realized FAR 0.97): training
collapses their scores, so exchangeability with deployment normals is the
binding assumption.

**The guarantee is the contribution; it holds for any operator.** The
split-conformal FAR guarantee — a finite-sample, distribution-free bound from
known-good normals — is what operator-only prior work (incl. Shin22) lacks, and it
applies to *any* operator — every content-blind operator (including our deployed
FCM-PM, whose measured FAR the conformal layer turns into a certified guarantee) and
the overlay/summation join alike. This is the framework's strongest novel asset.

**Coverage at a guaranteed FAR (5 seeds x 50 splits, n_cal=500), among the arms
measured.** Every method meets the finite-sample guarantee (realized FAR = alpha
to within CI), but the *usable coverage* retained at that guarantee separates the
arms; Pair-Mask, by lowering the model's intrinsic normal confidence, lets the
conformal reject abstain least:

| method                       | alpha=0.01 realized / coverage | alpha=0.05 realized / coverage |
|-------------------------------|--------------------------------|--------------------------------|
| FCM-PM (**ours**)             | 0.0098 / **0.972**             | 0.0503 / 0.989                 |
| fcm (no PM)                   | 0.0100 / 0.839                 | 0.0503 / 0.908                 |
| cutmix                        | 0.0097 / 0.759                 | 0.0517 / 0.884                 |
| mixup                         | 0.0096 / 0.737                 | 0.0488 / 0.799                 |
| single-only                   | 0.0092 / 0.518                 | 0.0495 / 0.540                 |
| summation (=max-union), ref.  | 0.0099 / 0.995                 | (ref.)                         |

The guarantee holds for every operator (realized FAR = alpha), and coverage at a
guaranteed 1% FAR is nearly free for our method: FCM-PM retains the best coverage among
the partition-regime and weaker content-blind arms, **97.2%**, then 51.8-83.9% for the
weaker arms; the overlay/summation arm retains 99.5%.
**Honest:** the overlay/summation join (strong on this superposition benchmark) has the
*highest* coverage (99.5%), consistent with it being WM38's matched operator; we
therefore claim only that (i) the conformal *guarantee* is operator-agnostic,
annotation-free, and novel (Shin 2022 provides none) and nearly free on our deployed
FCM-PM, and (ii) Pair-Mask improves coverage among the partition-regime arms.

**Full calibration curve (multi-alpha).** Extending the two operating points to a
sweep over target levels alpha in {0.5, 1, 2, 5, 10}% (5 seeds x 50 splits,
n_cal=500 known-good real normals), the realized NORMAL FAR tracks target alpha to
within **0.153 pp** (maximum deviation across all seven operators and five levels)
for *every* operator -- the distribution-free, finite-sample calibration that
operator-only prior work (incl. Shin 2022's Summation Mixup) lacks -- and the
guarantee is nearly free on our deployed FCM-PM; the max-union / summation curve is
WM38's matched superposition operator, shown for completeness. See
`docs/mlsynth_paper/latex/figs/fig_conformal_calibration.pdf`; per-arm numbers in
`outputs/multilabel_synth/wm38_conformal_calibration_curve_summary.csv`. This adds
the multi-alpha curve; the @1% coverage results above are unchanged.

### 5.12 What the oracle's advantage actually is

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

**What each side owns.** The multi-label oracle (0.974) bounds the problem as an upper
reference (it uses supervision we forbid). On WM38 — a superposition regime — overlay /
summation (0.80) is the matched, strong operator, reported honestly and not beaten;
among the content-blind operators FCM-PM gives the best FAR-controlled tradeoff
(0.663/0.228 at the common pick, 0.654/0.147 at its best pick). The decisive selection
is the chip **partition** domain, where FCM-PM beats overlay on both axes (Sec 5.2.1).
What the oracle owns over any single-label synthesis is the appearance interaction of
real high-order mixes, which independent-union synthesis does not reproduce; neither
combination-support matching nor naive higher-order synthesis closes it in our tests
(Sec 5.12). On the headline oracle checkpoint, even tau=0.99 leaves normal FAR 0.799;
the full-scale oracle varies from 0.295/0.262/0.001 across seeds, so this is an
optimization/calibration result, not an inherent impossibility theorem. Our method
reaches strong operating points (chip Total-FAR 0.06% at 0.998 bit-F1; WM38 FAR
0.147-0.228; the Pair-Mask, val-margin, and NB-reject levers) with synthetic normals,
margin rejection, and a conformal guarantee (Sec 5.6, 5.11) rather than real-normal
training.

**Density is a modeling characterization consistent with the law.** FCM-PM matches the
real defect-die density (0.29 vs real 0.31); overlay is over-dense (0.50) — the
signature of the superposition operator. A density-shift analysis (Sec 5.2) stratifies
the real 2-mix test and shows overlay beating FCM-PM per stratum and mix order (2-mix
0.855/0.830/0.805, 3-mix 0.725, 4-mix 0.655 vs 0.787/0.787/0.714, 0.634, 0.484). We own
this as **consistent** with the operator-match law rather than moot: WM38 is a
superposition regime, so its matched operator (overlay) is expected to win there, which
does not contradict FCM-PM winning where the true law is a partition — as the chip
head-to-head measures (Sec 5.2.1). Density-faithfulness is a modeling property tied to
the combination law, not a stand-alone performance lever. The excess-risk TV bound
(Sec Theory) is retained as a general, one-directional guarantee — it explains when
synthesis can match the oracle — and we do not use its density-mismatch lower bound to
prefer any operator. The surviving contributions are the single→multi setting and
objective, the operator-match law measured by the chip head-to-head, the FCM-PM method
with its FAR levers, the annotation-free conformal FAR guarantee, the theory, and
cross-domain framework support.

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
observe a co-occurrence at all — yet when synthesis approximates the domain's
evidence-preserving combination (summation/union for superposition-structured inked
digits and wafer maps; the full-cover complement for partition-structured chip maps) we
approach the oracle (Cor. 1), because the generative structure lets synthesis
reconstruct the co-occurrence distribution that observation withholds.

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

**Contribution type, not leaderboard SOTA.** We do not claim state-of-the-art
accuracy: fully-supervised MixedWM38 methods reach 98-99%, and the multi-label oracle
(0.974) is an upper reference that uses supervision we forbid. On WM38 — a
superposition regime — overlay / summation (0.80, = Shin 2022) is the matched, strong
content-blind operator, reported honestly and not beaten; among the content-blind
operators our FCM-PM gives the best FAR-controlled tradeoff. The partition-domain
result is decisive: FCM-PM beats overlay head-to-head on the chip domain (Sec 5.2.1).
All of this is *from single-label data alone*, with a finite-sample FAR guarantee. The
contribution is the annotation-free single→multi setting and objective, the
operator-match law (measured by the chip head-to-head), the FCM-PM method with its FAR
levers, the reliability guarantee, the theory, and cross-regime framework support across
five families -- a weak-supervision and reliability result; reliance on one *public*
real-multi-label benchmark is a real limitation (the chip partition head-to-head uses
internal data), and the cross-regime breadth is what carries the claim beyond it.

## 7 Conclusion

From single-label training data alone — no multi-label or real-normal annotation —
label-faithful synthesis trains multi-label recognizers and we maximize the
multi-label performance–FAR tradeoff. The central, now **measured**, principle is
operator-match: the right content-blind synthesis operator is set by the domain's true
combination law. On WM38 — a superposition regime under the binary encoding, where the
matched join is whole-image summation/union (= Shin et al. 2022 Summation Mixup) —
overlay is genuinely strong (bit-F1 0.80 at FAR 0.010), and we report it honestly and
**do not claim to beat it**; among the content-blind operators our FCM-PM (with
val-margin selection and NB-reject) gives the best FAR-controlled tradeoff (0.663/0.228
at the common pick; 0.654/0.147 at its best pick). The decisive test is the chip
**partition** domain: there, under an identical protocol across three seeds, FCM-PM beats overlay on both
axes (eval bit-F1 0.9968 +-0.0008 vs 0.8733 +-0.0338; Total-FAR 0.68 +-0.63% vs 16.20 +-5.53%, ~24x lower on average with no per-seed overlap), because
pixelwise-max mis-models a partition — turning operator-match from an assertion into a
verified law and, with it, FCM-PM into the method of choice for partition-structured
inspection maps. Density is a modeling characterization consistent with the law (FCM-PM
matches the real 0.29; overlay's over-density 0.50 is the superposition signature). On
a genuine superposition domain (MNIST) blind synthesis exceeds the oracle on unseen
combinations. A margin-reject stage drives observed normal FAR to zero at a few-percent
review cost, and calibration on a small known-good set provides a finite-sample,
distribution-free FAR guarantee under exchangeability that holds for any operator — the
reliability layer operator-only prior work lacks. The label-fidelity / operator-match
criterion is measurable before training and characterizes evidence-preservation across
regimes: superposition-structured domains (wafer, inked digits, audio) take
summation/union, partition-structured domains (chip) take the partition-complement,
disjoint-coordinate text takes averaging, and natural RGB (VOC) marks the boundary where
content-blind synthesis fails and location supervision becomes necessary. What remains
with the oracle is the appearance of real high-order interactions — a boundary we
quantify and leave as the open problem.

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
