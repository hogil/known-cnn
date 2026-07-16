# Multi-Label Recognition Without Multi-Label Annotation: Label-Faithful Synthesis from Single-Label Data

Working draft v0.2 (2026-07-10). All numbers are measured (primary log:
`D:/project/known-cnn/docs/superpowers/multilabel_synth_RESULTS.md`; result
CSVs under `D:/project/known-cnn/outputs/multilabel_synth/`). Candidate venues
are assessed in `D:/project/known-cnn/docs/mlsynth_paper/SUBMISSION_READINESS_260710.md`.

> **Positioning correction, 2026-07-16 (FINAL — density-shift refutation;
> supersedes the die-budget-partition revision).** A decisive **density-shift
> stress test** refuted the die-budget / faithful-operator thesis. On WM38,
> whole-image **max-union** — which under the binary encoding (normal die 0.5,
> defect 1.0) equals **Shin et al. (2022) Summation Mixup** (verified:
> `summation_mixup_shin22` and `overlay` both compute `np.maximum(ca,cb)`) — is
> the **best content-blind operator**, beating the partition-style full-cover
> complement (FCM-PM) on **EVERY** real-mix density stratum AND mix order (2-mix
> low/mid/high density 0.855/0.830/0.805 vs 0.787/0.787/0.714; 3-mix 0.725 vs
> 0.634; 4-mix 0.655 vs 0.484), *including* the low-density / high-order regimes
> where over-density was predicted to hurt, and at both higher bit-F1 (0.80 vs
> 0.65) and lower FAR (0.010 vs 0.147). **"Over-density hurts" is therefore
> FALSE**: max-union is over-dense (defect-die fraction 0.50 vs real 0.31) yet
> strictly better on wafer. We consequently **DROP** the earlier claims that
> FCM-PM is "the faithful WM38 method", that max-union is "distributionally
> mismatched / excluded", and that a die-budget partition makes complement beat
> summation. **On WM38 the best content-blind operator is summation/union
> (= max-overlay = Shin 2022 Summation Mixup); we do NOT beat prior art on the
> wafer operator and state this plainly.** The density facts are kept only as an
> honest characterization (max-union over-dense 0.50; FCM matches real 0.29) with
> the explicit caveat that this over-density is *empirically harmless*, so
> density-faithfulness is a **modeling property, not a performance advantage**;
> the TV lower bound survives as a general theory bound, **not a preference
> argument**.
>
> The paper is repositioned as an **annotation-free, reliability-guaranteed,
> cross-domain FRAMEWORK** (not a new/better operator): (i) the single→multi
> setting and method (zero multi-label / normal / location annotation); (ii) a
> **label-fidelity / operator-match criterion** that selects the right
> content-blind operator per domain — summation/union on superposition-structured
> domains (wafer / digits / audio), averaging on disjoint-coordinate text — and
> which **correctly selects summation on wafer**, consistent with the refutation;
> (iii) an **annotation-free, distribution-free split-conformal FAR guarantee**
> (the reliability layer operator-only prior work, incl. Shin22, lacks) — the
> strongest novel asset; (iv) an excess-risk theory as a general, honestly loose /
> one-directional bound, not a superiority proof; (v) cross-domain validation
> (wafer + audio + text) with an honest natural-image boundary (VOC). **FCM-PM is
> reported honestly as an alternative operator** that does NOT outperform
> summation on wafer (0.654/0.147 vs 0.80/0.010), but genuinely helps on
> chip-internal maps (~0.99) and controls FAR well relative to the weaker faithful
> arms (cutmix 0.691/0.439, FCM-without-PM 0.665/0.384, mixup 0.537/0.225;
> single-only 0.473/0.602). Evidence:
> `D:/project/known-cnn/docs/superpowers/multilabel_synth_RESULTS.md` (sections
> "DECISIVE density-shift stress test" and "Conformal FAR guarantee COMPLETE").

---

## Abstract

Industrial visual inspection routinely faces images containing multiple
co-occurring defect types, yet multi-label annotation of such images is
impractical: co-occurrences explode combinatorially, rare combinations may
never be observed in labeled form, and overlapping patterns are ambiguous to
annotate. In contrast, single-defect examples are cheap and unambiguous. We
study how to train a multi-label classifier from single-label supervision alone,
by synthesizing combination examples from single-label sources — with zero
multi-label, normal, or location annotation. Our contribution is a **framework,
not a new mixing operator**. We contribute (i) the single→multi setting and
method; (ii) a **label-fidelity / operator-match criterion** that selects, before
any training, the content-blind synthesis operator that best preserves each
source's evidence — summation/union on superposition-structured domains (inked
digits, wafer maps, audio), averaging on disjoint-coordinate text — and which
**correctly selects summation on wafer**; (iii) an **annotation-free,
distribution-free split-conformal false-alarm-rate guarantee** (plus synthetic
normals, negative-target control, val-margin selection, naive-Bayes rejection)
that operator-only prior work lacks; (iv) an excess-risk theory — a general upper
bound 2B·TV(D_real, D_syn^T), honestly one-directional (it does *not* prove an
operator preference) — explaining when blind synthesis can match a
fully-supervised oracle, with a natural-image (PASCAL VOC) boundary corollary;
and (v) cross-domain validation. On a controlled MultiMNIST benchmark — a genuine
superposition domain — blind max-overlay synthesis from single-label data exceeds
a fully-supervised oracle on the full test (mAP 0.868 vs 0.846) and by +0.198 mAP
on held-out label combinations the oracle never observed. On the public MixedWM38
wafer-map benchmark the **best content-blind operator is whole-image max-union /
summation**, which under the binary encoding (normal die 0.5, defect 1.0) equals
Shin et al. (2022) Summation Mixup (verified: `summation_mixup_shin22` and
`overlay` both compute `np.maximum`): it reaches bit-F1 0.80 at NORMAL-FAR 0.010,
recovering ~82% of an equal-condition oracle (0.974) from single-label data
alone, and it **wins on every real-mix density stratum and every mix order** in a
density-shift stress test (2-mix 0.855/0.830/0.805, 3-mix 0.725, 4-mix 0.655 vs
the partition arm's 0.787/0.787/0.714, 0.634, 0.484). **We do not claim a better
wafer operator than this prior art.** The partition-style full-cover complement
(FCM-PM) matches the real defect-die density (0.29 vs 0.50 for max-union), but
this density-faithfulness is **empirically harmless** — over-dense max-union is
strictly better — so we report FCM-PM honestly as an **alternative operator**
(bit-F1 0.654 / FAR 0.147; strong FAR control relative to cutmix 0.439, mixup
0.225, and FCM-without-Pair-Mask 0.384, and ~0.99 on chip-internal maps), **not
as the winner**. A small set of known-good samples yields a finite-sample
conformal FAR guarantee (realized 0.040 at alpha=0.05, 0.006 at alpha=0.01) that
holds for **any** operator — the reliability layer operator-only prior work (incl.
Shin22) omits. The same **operator-match principle** — synthesis should apply the
operator that maximizes label fidelity in the domain — is validated across
regimes: superposition-structured domains (wafer + inked digits + audio: FSD50K
waveform summation wins bit-F1 0.433 vs 0.27–0.31, consistent with wafer and
digits) all take summation/union, while disjoint-coordinate text (Reuters) takes
vector averaging (72% oracle recovery). We characterize the natural-image
boundary (PASCAL VOC), where blind synthesis fails and location supervision
becomes necessary, and release the full harness.

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
show that the choice of synthesis operator is governed by a single measurable
criterion: **label fidelity** — every labeled source must retain detectable
evidence in the synthesized example. Operators that destroy evidence (rectangle
patching, pixel averaging) train on false labels, and the measured survival
ordering predicts the raw detection ordering within every image family we test.
The fidelity-maximizing operator is domain-dependent: on **superposition-structured
domains** — inked digits, wafer maps, and audio, whose signal saturates or adds
when sources co-occur — the best-preserving operator is **summation/union**, and
on **disjoint-coordinate** text it is **vector averaging** (averaging preserves
per-coordinate evidence when vocabularies barely overlap). This label-fidelity /
operator-match criterion selects the operator *before any training*, and it
correctly selects summation on wafer maps.

On the public MixedWM38 benchmark the operator the criterion selects — whole-image
**max-union / summation** — is also the empirical winner. Under the benchmark's
binary encoding (normal die 0.5, defect 1.0) it coincides exactly with **Shin et
al. (2022) Summation Mixup** (verified in code: `summation_mixup_shin22` and
`overlay` both compute `np.maximum(ca,cb)`), so **we claim no new or better wafer
operator** and say so plainly. It reaches bit-F1 0.80 at NORMAL-FAR 0.010,
recovering ~82% of an equal-condition oracle (0.974) from single-label data alone.
A **density-shift stress test** settles a prior worry: max-union produces
over-dense wafers (defect-die fraction 0.50 vs the real 0.31) but this
over-density is **empirically harmless** — max-union beats the density-matching
partition operator (FCM-PM) on every real-mix density stratum and every mix order
(2-mix 0.855/0.830/0.805, 3-mix 0.725, 4-mix 0.655 vs 0.787/0.787/0.714, 0.634,
0.484). We therefore treat density-faithfulness as a **modeling property, not a
performance advantage**, and do not use it to prefer any operator. The
partition-style full-cover-complement operator with a Pair-Mask false-alarm view
(**FCM-PM**) is reported honestly as an **alternative** that does *not* beat
summation on wafer (0.654 bit-F1 / 0.147 FAR), but that is genuinely useful on
chip-internal maps (~0.99) and controls FAR well relative to the weaker arms
(cutmix 0.439, FCM-without-Pair-Mask 0.384, mixup 0.225).

The reliability layer is the practical core, and it is what operator-only prior
work omits. Because the training data contain no all-negative label, a naive
synthesizer over-alarms on real normals; we control false alarms without any
real-normal annotation by combining synthetic normals, a negative-target lever,
val-margin checkpoint selection on a disjoint held-out-source synthetic proxy,
class-conditional Gaussian (naive-Bayes) rejection fit only on further disjoint
single sources and synthetic normals, and a **split-conformal calibration for a
finite-sample, distribution-free FAR guarantee**. This guarantee holds for *any*
operator (including the winning summation) and is the reliability asset Shin et
al. (2022) and other operator-only methods do not provide. Real mixed and normal
maps remain final test data.

Contributions:
1. **The single→multi framework (setting + method).** We train multi-label
   recognizers with **zero multi-label annotation** — no multi-label image, no
   all-negative (normal) label, no location/mask annotation — by synthesizing
   combinations from single-label sources. The setting is stricter-than-SPML
   (genuinely single-label training images), evaluated on real multi-label data
   and real normals, with an equal-condition oracle protocol validated against
   published benchmark numbers.
2. **A label-fidelity / operator-match criterion.** A measurable, pre-training
   criterion (weaker-source evidence survival) that selects the content-blind
   synthesis operator per domain: summation/union on superposition-structured
   domains (wafer / digits / audio), averaging on disjoint-coordinate text. On
   wafer it correctly selects summation — the empirical winner and the closest
   prior art (Shin 2022) — so the framework's operator choice is principled, not a
   claim of operator novelty.
3. **Theory — a general excess-risk bound.** For ANY content-blind operator T,
   training on operator-T synthesis costs at most 2B·TV(D_real, D_syn^T) over the
   real-mix oracle, and the bound is minimized by the operator whose synthetic-mix
   law is closest to real (Cor. 1 gives risk-equivalence to full supervision under
   a matched law). We are explicit that this is a **loose, one-directional**
   guarantee: a large TV weakens the *upper bound* but does **not** prove larger
   real risk — indeed the over-dense max-union attains the *highest* bit-F1 — so
   the theory explains *when synthesis can match the oracle*, and does **not**
   certify a preferred operator. The density gap is retained only as a general TV
   lower-bound illustration, not a preference argument.
4. **Annotation-free FAR guarantee.** A strict source-only reliability pipeline —
   synthetic normals, negative-target control, synthetic validation margin for
   checkpoint selection, class-conditional Gaussian pattern likelihood for
   rejection, and split-conformal calibration for a finite-sample marginal FAR
   guarantee under exchangeability — operator-agnostic, and absent from prior
   operator-only work (e.g. Shin et al. 2022). This is the framework's strongest
   novel asset.
5. **Cross-domain generality.** The operator-match criterion is validated across
   regimes: superposition-structured domains — wafer maps, inked digits, and audio
   (FSD50K physical **waveform summation** wins bit-F1 0.433 vs 0.27-0.31,
   consistent with wafer and digits) — all take summation/union, while
   disjoint-coordinate text (Reuters) takes vector averaging (72% oracle recovery).
   Natural images (VOC) mark the boundary, where blind synthesis fails. Honest
   nuance: on audio, waveform summation wins bit-F1 at higher FAR (0.21 vs ~0.10),
   with FCM-PM the best FAR-controlled arm there (0.312 / 0.102).

## 2 Related Work

- **Mixing augmentations.** Mixup (Zhang et al., 2018): convex image/label
  blending — designed as a regularizer for single-label training, not as a
  combination synthesizer; we show averaging ghosts both objects (MNIST mixup
  full mAP 0.738 vs overlay 0.868; WM38 mixup bitF1 0.537 vs the winning
  summation/union join 0.80, strict 5-seed). CutMix (Yun et
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
  `overlay` arms both compute `np.maximum(ca, cb)`), and in our strict 5-seed
  comparison this is **the best content-blind operator** on WM38 (bit-F1 0.80,
  FAR 0.010). **Our label-fidelity criterion selects exactly this operator on
  wafer maps, so we claim no new or better wafer operator — we coincide with the
  closest prior work and report it plainly.** We do note that max-union produces
  over-dense wafers (defect-die density 0.50 vs the real 0.31, matching singles
  0.29), and that a partition-style full-cover complement matches the real density
  (0.29); but a **density-shift stress test shows this over-density is empirically
  harmless** (max-union wins on every density stratum and mix order), so density is
  reported only as a characterization, not a reason to prefer any operator. Our
  contributions relative to this prior line are therefore **not** an operator but
  (a) the annotation-free single→multi framework, (b) the label-fidelity /
  operator-match criterion that *selects* the correct operator per domain (and
  correctly picks summation on wafer), (c) the excess-risk theory that explains
  *when* blind synthesis can match the oracle, and (d) an annotation-free,
  distribution-free FAR-control layer (synthetic normals + val-margin selection +
  NB-reject + split-conformal conformal guarantee) that Shin et al. and the other
  operator-only methods above do not provide. Thuan is the closest direct synthesis
  competitor but explicitly extracts defect support (content-aware); our operators
  remain location-agnostic. A submission-grade version must reproduce every prior
  whose exact specification is available under our common 8-bit/FAR protocol —
  reporting both bit-F1 and FAR — and report native results plus protocol gaps for
  inaccessible methods; this is a required comparison, not an optional paragraph.
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
**label-fidelity criterion**. It has three parts: (i) the **label-fidelity /
operator-match criterion** (Sec 4.2) that selects, per modality and before any
training, the content-blind synthesis operator that best preserves each source's
evidence; (ii) **val-margin checkpoint selection** on a disjoint held-out-source
synthetic proxy; and (iii) **NB-reject** -- a synthetic-only class-conditional
Gaussian (naive-Bayes) pattern-likelihood reject/decode stage, optionally backed
by the split-conformal FAR guarantee (Sec 5.11). Parts (ii)-(iii), together with
synthetic normals and a negative-target lever, form the annotation-free
FAR-control layer that operator-only prior work omits and that applies to any
operator.

The operator the criterion selects is domain-dependent, because the evidence
geometry is. In **superposition-structured domains** co-occurring sources saturate
or add, so evidence is best preserved by a **summation/union join**: pixelwise
max-overlay for inked digits (ink saturates over black), whole-image max-union for
wafer maps (defect dies win over normal dies), and physical **waveform summation**
for audio (FSD50K, where waveform_sum wins bit-F1 0.433 vs 0.27-0.31; Sec 5.10).
On wafer maps this selected operator coincides, under the binary encoding, with
Shin et al. (2022) Summation Mixup (Sec 5.2) and is the empirical winner (bit-F1
0.80 / FAR 0.010) — so the criterion is correct there and **we claim no operator
novelty on wafer**. In **disjoint-coordinate text** the criterion selects vector
averaging (Sec 5.9), because averaging preserves per-coordinate evidence when
vocabularies barely overlap. The remaining operators in 4.1 (the partition-style
full-cover complement FCM / FCM-PM, CutMix, Mixup, single-only, oracle) are
content-blind reference/alternative arms evaluated under the identical gate; FCM-PM
is a useful alternative operator (chip-internal ~0.99; good FAR control among the
weaker arms) but does **not** beat summation on wafer (Sec 5.2). Generality is
established across three combination regimes --- superposition-structured
(MixedWM38 public wafer, MultiMNIST inked digits, and FSD50K audio, Sec 5.10),
disjoint-coordinate text (Reuters, Sec 5.9), and the natural-image boundary (VOC)
--- each with its own fidelity-maximizing operator selected by the same criterion.

### 4.1 Synthesis operators (content-blind)

Given two singles (x_a, a), (x_b, b):
- **overlay / max-union** (the fidelity-maximizing operator on
  superposition-structured domains): per-pixel max (defect intensity wins over
  normal background) — both objects survive whole; hard label {a,b}. It is the
  criterion's selection for inked digits, wafer maps, and (via waveform summation)
  audio (Sec 5.10). On **wafer maps** it is, under WM38's binary encoding (normal
  0.5, defect 1.0), exactly Shin et al. (2022) Summation Mixup's clipped binary
  sum, and it is the **best content-blind operator** (bit-F1 0.80 / FAR 0.010, Sec
  5.2). It produces over-dense maps (density 0.50 vs real 0.31), but a density-shift
  stress test shows this is empirically harmless (Sec 5.2). Domain analog: chip
  min-blend.
- **complement (FCM)** (a partition-style alternative operator): G x G grid
  (G = 3N, e.g. 9), cells randomly permuted and partitioned into n groups;
  mix_i = x_b base with x_a's group-i cells overwritten. The union of the n mixes
  covers x_a exactly once (full cover), so each die is owned by exactly one source;
  this matches the real 2-mix defect-die density (0.29 vs real 0.31). Hard labels;
  per-mix asymmetric (A,B) targets are a no-op within [0.9, 1.0] (chip
  leaderboard: 0.9889-0.9898 across A in {0.90,0.95,1.00}). Density-matching, but
  **not** the wafer winner (0.665/0.384 without PM; 0.654/0.147 with PM) — summation
  beats it (Sec 5.2).
- **pair mask (PM)** (the FAR-control lever for the complement arm): for each
  complement mix, also emit a mask sample — x_a's cells kept, x_b's cells with
  defects erased; target: bit a at a soft 0.65, all else negative. Teaches
  "near-normal map with weak fragments => low confidence"; a false-alarm
  suppressor. FCM+PM together form FCM-PM (PM cuts the complement arm's FAR 0.384
  -> 0.147 at comparable F1); FCM-PM also reaches ~0.99 on chip-internal maps.
- **synthetic normals**: defect pixels erased from real singles (wafer:
  min(x, normal-die value)) => all-negative samples without any normal label.
- Baselines / alternatives: CutMix (rectangle patch), Mixup (convex blend, soft
  labels), FCM / FCM-PM (partition-style complement, above), single-only (no
  synthesis), oracle (real multi-label training — ceiling, not a rival).

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

On superposition-structured domains, the survival-maximal operator is
summation/union. Measured weaker-source defect survival on WM38: max-union
**1.000** > CutMix 0.579 > complement (FCM) 0.527 > Mixup 0.236 — and the strict
bit-F1 ranking follows it (max-union 0.80 > CutMix 0.69 > FCM 0.665 > Mixup 0.537).
So the criterion **correctly selects max-union / summation on wafer**, which is
also the empirical winner (Sec 5.2). On disjoint-coordinate text the fidelity
maximizer flips to averaging (Sec 5.9), because averaging preserves per-coordinate
evidence there. The criterion is thus a per-domain operator selector, not a fixed
operator.

**Density as a characterization (not a selection criterion).** Max-union produces
over-dense wafers (defect-die density 0.501 vs real 0.305), while the partition
complement matches the real density (0.293). We report this honestly, but a
density-shift stress test (Sec 5.2) shows over-density is **empirically harmless**:
max-union wins across all density strata and mix orders. Density-faithfulness is
therefore a modeling property, **not** a performance advantage, and we do **not**
use it to prefer any operator.

### 4.3 Why not averaging

In shared-coordinate spaces, Mixup's blend produces ghosted objects (both at
half contrast) with soft labels; a hard, evidence-preserving join keeps each
source recoverable. Same combination-operator family, opposite outcome: on
superposition MNIST, overlay 0.868 vs mixup 0.738; on wafer maps the winning
max-union join 0.80 far exceeds mixup 0.537 (strict, 5 seeds). Failure of
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

Max-union is over-dense (0.501 vs real 0.305); the partition complement matches
the real density (0.293). We initially hypothesized this over-density would hurt
(a "die-budget" argument for preferring the complement), but a **density-shift
stress test refutes it** (below). Density is therefore reported only as a
characterization; it does **not** justify preferring any operator.

**Density-shift stress test (decisive; refutes "over-density hurts").** We
stratify the real 2-mix test by density tertile and split out 3/4-mix, then
compare the over-dense max-union operator against the density-matching FCM-PM
(SmallCNN, 3 seeds). bit-F1:

| condition          | max-union (= Shin22) | FCM-PM |
|--------------------|----------------------|--------|
| 2-mix low density  | 0.8548               | 0.7868 |
| 2-mix mid density  | 0.8298               | 0.7874 |
| 2-mix high density | 0.8049               | 0.7144 |
| 3-mix              | 0.7245               | 0.6336 |
| 4-mix              | 0.6552               | 0.4841 |

Max-union **wins on every density stratum and every mix order**, including the
low-density / high-order regimes where over-dense training was predicted to hurt
(the gap in fact grows with order, +0.17 at 4-mix). "Over-density hurts" is
refuted: over-dense max-union is simply the better wafer operator. This is the
empirical basis for dropping the earlier faithful-operator / die-budget preference
for FCM-PM.

**Strict operator comparison (5 seeds, pick=val_tail_margin_guarded, neg 0.02).**
Common protocol (matched view budget, splits, checkpoint selection, rejection):

| operator                       | bit-F1 | NORMAL FAR | defect-die density | note                              |
|--------------------------------|--------|------------|--------------------|-----------------------------------|
| max-union / overlay (= Shin22) | 0.800  | 0.010      | 0.501 (over-dense) | best operator on wafer (= Shin22) |
| cutmix                         | 0.691  | 0.439      | —                  | evidence-destroying, FAR high     |
| FCM (no Pair-Mask)             | 0.665  | 0.384      | 0.293 (matches)    | partition-style alternative       |
| FCM-PM                         | 0.654  | 0.147      | 0.293 (matches)    | alt; PM adds FAR control          |
| mixup                          | 0.537  | 0.225      | —                  | ghosting                          |
| single_only                    | 0.473  | 0.602      | 0.290              | floor                             |

Reading, stated honestly: whole-image **max-union / summation** — which under the
binary encoding **is Shin et al. (2022) Summation Mixup** — is the best
content-blind operator on both axes (bit-F1 0.80, FAR 0.010). Our label-fidelity
criterion selects exactly this operator (max-union has survival 1.000), so **the
framework's operator choice is correct on wafer and we claim no operator novelty vs
Shin22**. The partition-style FCM-PM is an alternative that does **not** beat it
(0.654/0.147); its value is (i) matching the real density — a modeling property
that the density-shift test shows carries no F1 benefit — and (ii) FAR control
relative to the weaker arms (Pair-Mask cuts the complement FAR 0.384 -> 0.147; both
CutMix 0.439 and FCM-without-PM 0.384 are higher). The framework, using its selected
operator (summation), **recovers ~82% of the literature-grade oracle** (0.800 vs
0.974) at FAR 0.010; the annotation-free reliability layer (Sec 5.6, 5.11) turns
that FAR into a finite-sample guarantee.

(The oracle matches published MixedWM38 accuracies 98-99%, validating the
harness. An earlier "statistical parity" claim against a SmallCNN-15ep oracle
(0.863) is retracted as a weak-oracle artifact. For the headline oracle
checkpoint, max-prob on real normals >=0.99 for 80% of normals, so tau=0.99 still
leaves FAR 0.799 — a checkpoint-level calibration result, not an impossibility
claim about all oracle training.)

**Component ablations (transfer across operators).** Pair masking in the complement
arm cuts observed NORMAL FAR in isolation (0.852 -> 0.012), and synthetic normals
reduce it in every arm. Full-scale / multi-seed runs of the winning max-union arm
reach bit-F1 0.795 (full-scale, all 7,015 singles, ResNet-18, 30ep) to 0.841
(nine-seed headline) at near-zero real-normal FAR (0.0003-0.0008). The label-fidelity
survival ordering (max-union > cutmix > complement > mixup) predicts the raw
detection ordering — the criterion's core empirical claim.

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
FAR lower. (This isolation uses the max-union / Summation Mixup arm — our selected
wafer operator — because it is the simplest join; the point is loss-vs-structure,
and it holds for any co-occurrence synthesis including the partition-style FCM-PM.)
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
and matching the oracle requires location supervision — which is itself
the finding predicted by the superposition-domain condition.

Natural single-category (56%) vs multi-category (44%) split — no label
masking. ResNet-18, subsampled protocol. Findings: (i) copy-paste (crops onto
scene backgrounds) is the only synthesis matching the oracle's bit-F1 (0.357
vs 0.355) — natural scenes need object/context scale alignment; (ii) tight
crops alone regress (scale mismatch); (iii) region > blend ordering persists.
Positions the domain condition: label-faithful blind synthesis requires the
domain's combination to be reproducible by a content-blind, evidence-preserving
operator (summation/union in superposition-structured domains — inked digits,
wafer maps, audio) — which wafer/palette domains admit and opaque RGB scenes do
not (objects occlude, not join).

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
sd ~0.019) — **the same result as WM38**, where the summation operator (max-union
= Shin22) also wins. Wafer maps, inked digits, and audio are thus all
superposition-structured domains in which **summation/union is the
fidelity-maximizing operator**, and the same label-fidelity criterion selects it
in each (it flips to averaging only on disjoint-coordinate text, Sec 5.9). This is
the general **operator-match principle** — the criterion picks the
evidence-preserving operator for the domain's geometry — validated across wafer +
audio + text, and its wafer selection (summation) is consistent with the
density-shift refutation (Sec 5.2).

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
applies to *any* operator, including the winning summation arm (whose measured FAR
0.010 the conformal layer turns into a certified guarantee). This is the
framework's strongest novel asset.

**Coverage at a guaranteed FAR (5 seeds x 50 splits, n_cal=500), among the arms
measured.** Every method meets the finite-sample guarantee (realized FAR = alpha
to within CI), but the *usable coverage* retained at that guarantee separates the
arms; Pair-Mask, by lowering the model's intrinsic normal confidence, lets the
conformal reject abstain least:

| method       | alpha=0.01 realized / coverage | alpha=0.05 realized / coverage |
|--------------|--------------------------------|--------------------------------|
| FCM-PM       | 0.0098 / **0.972**             | 0.0503 / **0.989**             |
| fcm (no PM)  | 0.0100 / 0.839                 | 0.0503 / 0.908                 |
| cutmix       | 0.0097 / 0.759                 | 0.0517 / 0.884                 |
| mixup        | 0.0096 / 0.737                 | 0.0488 / 0.799                 |
| single-only  | 0.0092 / 0.518                 | 0.0495 / 0.540                 |

Among these arms, FCM-PM keeps **97.2%** of wafers usable at a guaranteed 1%
false-alarm rate versus 51.8-83.9% for the others (1.2-1.9x more decisions at the
same certified safety) — Pair-Mask is the lever. **Honest caveat:** this sweep
did not include the winning max-union / summation arm, whose own intrinsic FAR
(0.010, near the 1% target) would also yield high coverage; we therefore claim
only that (i) the conformal *guarantee* is operator-agnostic and novel, and (ii)
Pair-Mask improves coverage among the arms shown — **not** that FCM-PM's coverage
beats summation.

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

**What each side owns.** Under equal conditions the oracle keeps a bit-F1 lead
over the framework's best content-blind operator (0.974 vs max-union / summation's
0.80, i.e. ~82% recovery): real high-order mixes contain appearance interactions
that independent-union synthesis does not reproduce; neither combination-support
matching nor naive higher-order synthesis closes it in our tests (Sec 5.12). On the
headline oracle checkpoint, even tau=0.99 leaves normal FAR 0.799; the full-scale
oracle varies from 0.295/0.262/0.001 across seeds, so this is an
optimization/calibration result, not an inherent impossibility theorem. The
framework reaches a strong operating point (FAR 0.010 for the summation arm; 0.147
for the FCM-PM alternative) with synthetic normals, margin rejection, and a
conformal guarantee (Sec 5.6, 5.11) rather than real-normal training.

**Density faithfulness does not buy performance (refuting our own earlier
hypothesis).** We had hypothesized that max-union's over-density (0.50 vs real
0.31) would hurt, motivating a preference for the density-matching complement. A
**density-shift stress test refutes this** (Sec 5.2): the over-dense max-union
operator beats the density-matching FCM-PM on *every* real-mix density stratum and
*every* mix order (2-mix 0.855/0.830/0.805, 3-mix 0.725, 4-mix 0.655 vs
0.787/0.787/0.714, 0.634, 0.484), including the low-density / high-order regimes
where over-density was predicted to hurt. We therefore **drop** the
"faithful-operator / die-budget" preference for FCM-PM and state plainly that on
WM38 the best content-blind operator is summation (= Shin 2022 Summation Mixup) and
we do **not** beat it. Density-faithfulness is a modeling property, not a
performance advantage. The excess-risk TV bound (Sec Theory) is retained as a
general, one-directional guarantee — it explains when synthesis can match the
oracle — but we no longer use its density-mismatch lower bound as a reason to
prefer any operator, and we note that the bound is consistent with the refutation
(a large TV weakens the upper bound but never implies larger real risk, and here
the higher-TV operator is in fact better). The surviving contributions are the
framework, the label-fidelity / operator-match criterion (which correctly selects
summation on wafer), the annotation-free conformal FAR guarantee, and cross-domain
generality.

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
observe a co-occurrence at all — yet when synthesis reproduces the domain's
evidence-preserving combination (summation/union for inked digits and wafer maps)
we approach the oracle (Cor. 1), because the generative structure lets synthesis
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
accuracy, nor a new/better mixing operator: fully-supervised MixedWM38 methods
reach 98-99%, and on the wafer operator our best content-blind arm coincides with
prior art (Shin 2022 Summation Mixup). Using that operator the framework recovers
~82% of an equal-condition oracle *from single-label data alone*, with a
finite-sample FAR guarantee. The contribution is the annotation-free single→multi
framework, the operator-match criterion, the reliability guarantee, and
cross-regime generality across five families -- a weak-supervision and reliability
result, not a single-benchmark accuracy or operator-novelty win; reliance on one
public real-multi-label benchmark is a real limitation, and the cross-regime
breadth is what carries the claim beyond it.

## 7 Conclusion

From single-label training data alone — no multi-label, real-normal, or location
annotation — label-faithful synthesis trains multi-label recognizers by applying
the operator that best preserves each source's evidence. On wafer maps that
operator is whole-image summation/union, which under the binary encoding **is**
Shin et al. (2022) Summation Mixup: our label-fidelity criterion selects it, and it
is the empirical winner (bit-F1 0.80 at FAR 0.010, ~82% of an equal-condition
oracle) — so we claim no new or better wafer operator. A density-shift stress test
shows this operator's over-density is harmless, refuting our own earlier "die-budget
faithful-operator" hypothesis; density-faithfulness is a modeling property, not a
performance advantage. On a genuine superposition domain (MNIST) blind synthesis
exceeds the oracle on unseen combinations. A margin-reject stage drives observed
normal FAR to zero at a few-percent review cost, and calibration on a small
known-good set provides a finite-sample, distribution-free FAR guarantee under
exchangeability that holds for any operator — the reliability layer operator-only
prior work lacks. The label-fidelity / operator-match criterion is measurable
before training and selects the right operator across regimes:
superposition-structured domains (wafer, inked digits, audio) all take
summation/union, and disjoint-coordinate text takes averaging; the criterion
predicts where the approach matches the real distribution (superposition domains)
and where it does not (natural RGB, VOC). The partition-style FCM-PM is reported
honestly as an alternative operator — useful on chip-internal maps (~0.99) and for
FAR control, but not a winner over summation on wafer. What remains with the oracle
is the appearance of real high-order interactions — a boundary we quantify and
leave as the open problem.

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
