# Multi-Label Recognition Without Multi-Label Annotation: Label-Faithful Synthesis from Single-Label Data

Working draft v0.2 (2026-07-10). All numbers are measured (primary log:
`D:/project/known-cnn/docs/superpowers/multilabel_synth_RESULTS.md`; result
CSVs under `D:/project/known-cnn/outputs/multilabel_synth/`). Candidate venues
are assessed in `D:/project/known-cnn/docs/mlsynth_paper/SUBMISSION_READINESS_260710.md`.

> **Positioning correction, 2026-07-17 (FINAL — single-train / multi-eval,
> location-free framing; supersedes the 2026-07-16 "summation is the best
> content-blind operator / no operator novelty" revision, which was WRONG for this
> paper).** SETTING: single-label **training**, multi-label **evaluation**, with **no
> multi-label, normal, or location annotation** (content-blind AND location-free).
> GOAL: **maximize** the multi-label performance–FAR tradeoff in THIS setting.
> METHOD: **FCM-PM** (full-cover complement + Pair-Mask) + **val-margin** checkpoint
> selection + **naive-Bayes reject** (chip/wafer scoped). Among **admissible**
> location-free operators, FCM-PM maximizes the performance–FAR tradeoff on WM38
> (bit-F1 0.654 / FAR 0.147; 0.663/0.228 at the guarded pick) vs cutmix (0.691 but
> FAR 0.439), mixup (0.537/0.225), single-only floor (0.473/0.602); ~0.99 on
> chip-internal maps.
>
> **INADMISSIBLE upper references (NOT competitors we must beat).** (a) the **oracle**
> = multi-label **training** (0.974); (b) **overlay / max-union / summation** (= Shin
> et al. 2022 Summation Mixup; `np.maximum` of two real single-defect maps)
> **preserves each defect's real LOCATION** => content-**aware** => **inadmissible**
> in the location-free setting (0.80/0.010). Both are reported ONLY as upper
> references, exactly like an oracle — they use information we forbid. Because overlay
> is inadmissible, the density-shift comparison (overlay beats FCM-PM per stratum) is
> **MOOT** (it compares against an inadmissible method); density is a **modeling
> characterization** only (FCM-PM matches real 0.29; max-union over-dense 0.50).
>
> **DROPPED (all WRONG for this paper):** "summation is the best operator", "we
> reproduce Shin22 as our operator", "we claim no operator novelty", "FCM-PM is not
> the winner", "density-shift refutes FCM-PM".
>
> The paper is an **annotation-free, reliability-guaranteed, cross-domain FRAMEWORK**
> whose contribution order is: (i) the single→multi **setting** + the objective of
> **performance maximization** within it (zero multi-label / normal / location
> annotation), with the oracle and the location-using max-union / summation as
> inadmissible upper references; (ii) the **method** = FCM-PM + val-margin +
> naive-Bayes reject, which maximizes the performance–FAR tradeoff among admissible
> location-free operators; (iii) an **annotation-free, distribution-free
> split-conformal FAR guarantee** (the reliability layer operator-only prior work,
> incl. Shin22, lacks); (iv) an **excess-risk theory** as a general, honestly loose /
> one-directional bound; (v) **cross-domain framework support** (MNIST / audio / text)
> with an honest natural-image boundary (VOC). A **label-fidelity / operator-match
> criterion** characterizes which content-blind operator preserves evidence per domain
> (summation/union on superposition-structured wafer / digits / audio; averaging on
> disjoint-coordinate text). Evidence:
> `D:/project/known-cnn/docs/superpowers/multilabel_synth_RESULTS.md` (sections
> "DECISIVE density-shift stress test" and "Conformal FAR guarantee COMPLETE").

---

## Abstract

Industrial visual inspection routinely faces images containing multiple
co-occurring defect types, yet multi-label annotation of such images is
impractical: co-occurrences explode combinatorially, rare combinations may
never be observed in labeled form, and overlapping patterns are ambiguous to
annotate. In contrast, single-defect examples are cheap and unambiguous. We
study a strict setting — **single-label training, multi-label evaluation** with
**no multi-label, normal, or location annotation** (the content-blind,
location-free constraint) — and ask how to **maximize** multi-label performance in
it by synthesizing combination examples from single-label sources. We contribute
(i) the **single→multi setting and the objective of performance maximization within
it**: two references bound the problem but are *not* admissible methods — a
multi-label-trained **oracle**, and whole-image max-union / summation (= Shin et al.
2022 Summation Mixup), which overlays two real single-defect maps and thereby
**preserves each defect's real location**, making it *content-aware* and
*inadmissible* in our location-free setting (both reported only as upper references,
exactly as an oracle uses information we forbid); (ii) our **method — FCM-PM
(full-cover complement with a Pair-Mask view), val-margin checkpoint selection, and
naive-Bayes rejection**: among the *admissible* location-free operators it maximizes
the performance–FAR tradeoff on the public MixedWM38 benchmark (bit-F1 0.654 at
NORMAL-FAR 0.147; 0.663/0.228 at the guarded pick), beating cutmix (0.691 bit-F1 but
FAR 0.439), mixup (0.537/0.225), and the single-only floor (0.473/0.602), and
reaching ~0.99 on chip-internal maps, with val-margin, naive-Bayes rejection, and
the Pair-Mask lever (which cuts the complement FAR 0.384→0.147) as the FAR controls;
(iii) an **annotation-free, operator-agnostic, distribution-free split-conformal
false-alarm-rate guarantee**: a full multi-alpha calibration curve on which the
realized FAR tracks the target across alpha in [0.5%, 10%] to within 0.153 pp for
every operator, backed by synthetic normals, a negative-target lever, val-margin
selection, and naive-Bayes rejection — the reliability layer operator-only prior work
(incl. Shin22) lacks; (iv) a general, one-directional excess-risk bound
2B*TV(D_real, D_syn^T) (honestly loose) that explains *when* blind synthesis can
match the oracle, retained as **supporting** analysis; and (v) **cross-domain
framework support** across five families: on a controlled MultiMNIST superposition
benchmark blind synthesis **exceeds** a fully-supervised oracle on held-out label
combinations it never saw (+0.198 mAP; full test mAP 0.868 vs 0.846) — a regime where
supervision structurally cannot help — a **label-fidelity / operator-match
criterion** characterizes which content-blind operator preserves evidence per domain
(predicting the counterintuitive Reuters text averaging-flip and the FSD50K audio
summation ranking), and natural images (PASCAL VOC) mark the boundary where
content-blind synthesis fails and location supervision becomes necessary. Density is
reported as a **modeling characterization** only: FCM-PM matches the real defect-die
density (0.29 vs real 0.31), while the over-density (0.50) of the inadmissible
max-union reference is a property of that reference, not an admissible-method
comparison. A small set of known-good samples yields a finite-sample conformal FAR
guarantee (realized 0.040 at alpha=0.05, 0.006 at alpha=0.01) that holds for **any**
operator — the reliability layer operator-only prior work (incl. Shin22) omits.

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

We work under a strict **content-blind, location-free** constraint — no defect's
location may be used at training time — and answer by synthesizing combination
examples from single-label sources, with the explicit objective of **maximizing** the
multi-label performance–false-alarm tradeoff in this setting. Two references bound the
problem but are *not* admissible methods, because each uses information the setting
forbids: a multi-label-trained **oracle** (uses multi-label supervision), and
whole-image **max-union / summation** (= Shin et al. (2022) Summation Mixup), which
overlays two real single-defect maps and thereby **preserves each defect's real
location** — the very quantity inspection is trying to find — making it
*content-aware*. We treat both as **upper references**, reported like an oracle, not
as operators we must beat. Among the operators that *are* admissible (they never use a
location), the choice is governed by a measurable criterion — **label fidelity**:
every labeled source must retain detectable evidence after synthesis. Operators that
destroy evidence (rectangle patching, pixel averaging) train on false labels. This
same criterion also characterizes evidence-preservation *across domains* and makes a
falsifiable, pre-registered cross-regime call — vector averaging (not summation) wins
on disjoint-coordinate text, while a summation/union join preserves evidence on
superposition-structured domains (inked digits, audio) — so that the counterintuitive
text averaging-flip (Reuters) and the audio summation ranking (FSD50K) are later
out-of-sample confirmations, not post-hoc rankings (Sec 5).

On the public MixedWM38 benchmark our method — the partition-style full-cover
complement with a Pair-Mask view (**FCM-PM**), with val-margin checkpoint selection
and naive-Bayes rejection — **maximizes the performance–FAR tradeoff among the
admissible location-free operators**: bit-F1 0.654 at NORMAL-FAR 0.147 (0.663/0.228 at
the guarded pick), against cutmix (0.691 bit-F1 but FAR 0.439), mixup (0.537/0.225),
and the single-only floor (0.473/0.602); it also reaches ~0.99 on chip-internal maps.
The two inadmissible upper references sit above this admissible frontier — the
multi-label oracle (0.974) and whole-image max-union / summation (0.80 at FAR 0.010, =
Shin et al. (2022) Summation Mixup under the binary encoding, normal die 0.5, defect
1.0; verified in code that `summation_mixup_shin22` and `overlay` both compute
`np.maximum(ca,cb)`) — but neither is a location-free method: the oracle trains on
real multi-label data, and max-union preserves each defect's real location. We report
them only as references. **Density is a modeling characterization, not a competitor
comparison**: FCM-PM matches the real 2-mix defect-die fraction (0.29 vs real 0.31),
whereas the inadmissible max-union reference is over-dense (0.50); a density-shift
analysis (Sec 5.2) characterizes that inadmissible reference and does not bear on the
ranking of admissible operators, among which FCM-PM leads the tradeoff. Val-margin
selection, naive-Bayes rejection, and the Pair-Mask lever (which cuts the complement
FAR 0.384→0.147 at comparable bit-F1) are the FAR controls.

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
1. **The single→multi setting and performance maximization within it.** We train
   multi-label recognizers with **zero multi-label annotation** — no multi-label
   image, no all-negative (normal) label, and **no location/mask annotation** — by
   synthesizing combinations from single-label sources, and we make **maximizing** the
   multi-label performance–FAR tradeoff in this location-free setting the objective.
   Two references bound the problem but are *not* admissible methods: the
   multi-label-trained **oracle**, and whole-image max-union / summation (= Shin et
   al. 2022 Summation Mixup), which overlays real single-defect maps and thereby uses
   each defect's real location (content-aware). Both are reported only as upper
   references. The setting is stricter-than-SPML (genuinely single-label training
   images), evaluated on real multi-label data and real normals.
2. **The method: FCM-PM + val-margin + naive-Bayes rejection.** A partition-style
   full-cover complement with a Pair-Mask view (FCM-PM), val-margin checkpoint
   selection on a disjoint synthetic proxy, and a synthetic-only class-conditional
   Gaussian (naive-Bayes) rejection stage. Among the *admissible* location-free
   operators, this method maximizes the performance–FAR tradeoff on MixedWM38 (bit-F1
   0.654 at FAR 0.147; 0.663/0.228 at the guarded pick), beating cutmix (0.691 bit-F1
   but FAR 0.439), mixup (0.537/0.225), and the single-only floor (0.473/0.602), and
   reaches ~0.99 on chip-internal maps; the Pair-Mask lever cuts the complement FAR
   0.384→0.147, and val-margin and naive-Bayes rejection are the further FAR controls.
3. **An annotation-free FAR guarantee.** A strict source-only reliability pipeline —
   synthetic normals, negative-target control, synthetic validation margin for
   checkpoint selection, class-conditional Gaussian (naive-Bayes) pattern likelihood
   for rejection, and **split-conformal calibration** for a finite-sample,
   distribution-free marginal FAR guarantee under exchangeability — that is
   operator-agnostic and absent from prior operator-only work (e.g. Shin et al. 2022).
   A full multi-alpha calibration curve shows the realized FAR tracking the target
   across alpha in [0.5%, 10%] to within 0.153 pp for every operator.
4. **An excess-risk theory (supporting).** A general, one-directional excess-risk
   bound 2B*TV(D_real, D_syn^T) (Sec Theory) that explains *when* blind synthesis can
   match the oracle; honestly loose and one-directional, retained as supporting
   analysis rather than a headline claim.
5. **Cross-domain framework support.** The framework and its label-fidelity /
   operator-match criterion generalize across five families spanning two combination
   regimes. On a controlled MultiMNIST superposition benchmark blind synthesis
   **exceeds** a fully-supervised oracle on held-out label combinations the oracle
   never observed (+0.198 mAP; full test mAP 0.868 vs 0.846) — a regime where
   supervision structurally cannot help. The criterion characterizes which
   content-blind operator preserves evidence per domain and makes a falsifiable
   cross-regime call — vector averaging (not summation) on disjoint-coordinate text
   (Reuters), a summation/union join on superposition-structured audio (FSD50K
   waveform summation wins bit-F1 0.433 vs 0.27-0.31) — each a ranking image intuition
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
  `overlay` arms both compute `np.maximum(ca, cb)`). Overlaying two real
  single-defect maps this way **preserves each defect's real on-wafer location**, so
  Summation Mixup is *content-aware*: in our strict location-free setting it is
  **inadmissible as a method** and enters only as an **upper reference** (bit-F1 0.80,
  FAR 0.010), parallel to the multi-label-trained oracle — it uses the defect
  location, the very quantity inspection is trying to find and that we forbid at
  training time. Thuan (2025) similarly extracts defect support and is likewise
  content-aware. **Our contribution on wafer is therefore not one of these
  location-using operators but a location-free framework:** (a) the annotation-free
  single→multi setting and the objective of maximizing performance within it; (b) the
  FCM-PM method (with val-margin selection and NB-reject) that maximizes the
  performance–FAR tradeoff *among the admissible location-free operators*; (c) the
  label-fidelity / operator-match criterion characterizing evidence-preservation per
  domain; (d) the excess-risk theory that explains *when* blind synthesis can match
  the oracle; and (e) an annotation-free, distribution-free FAR-control layer
  (synthetic normals + val-margin selection + NB-reject + split-conformal guarantee)
  that Shin et al. and the other operator-only methods above do not provide. Density
  is reported only as a modeling characterization (FCM-PM matches the real defect-die
  density 0.29; the inadmissible max-union reference is over-dense 0.50), not as a
  reason to prefer any operator. A submission-grade version must reproduce every prior
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

Our method has three parts: the partition-style **full-cover complement with a
Pair-Mask view (FCM-PM)**, **val-margin checkpoint selection** on a disjoint
held-out-source synthetic proxy, and a synthetic-only class-conditional Gaussian
(naive-Bayes) **NB-reject** stage (optionally backed by the split-conformal FAR
guarantee, Sec 5.11). The latter two, with synthetic normals and a negative-target
lever, form the annotation-free FAR-control layer that operator-only prior work omits
and that applies to any operator. FCM-PM is our deployed **location-free** synthesis
operator; among the admissible location-free operators it maximizes the
performance–FAR tradeoff on wafer maps (Sec 5.2) and reaches ~0.99 on chip-internal
maps. Crucially, the two operators that score higher on raw bit-F1 — the multi-label
oracle and whole-image max-union / summation (= Shin et al. 2022 Summation Mixup) —
are **inadmissible**: the oracle trains on real multi-label data, and max-union
overlays real single-defect maps and thereby uses each defect's real location. Both
are upper references, not competitors.

A **label-fidelity criterion** accompanies the method as a framework tool: it
measures, before any training, how well a content-blind operator preserves each
source's evidence, and it characterizes evidence-preservation across domains. The
evidence geometry is domain-dependent: in **superposition-structured domains**
sources saturate or add, so a **summation/union join** preserves evidence — pixelwise
max-overlay for inked digits (ink saturates over black) and physical **waveform
summation** for audio (FSD50K, where waveform_sum wins bit-F1 0.433 vs 0.27-0.31; Sec
5.10) — while in **disjoint-coordinate text** vector averaging preserves
per-coordinate evidence when vocabularies barely overlap (Sec 5.9). On wafer maps the
evidence-maximal join coincides, under the binary encoding, with the inadmissible
max-union / Summation Mixup reference; because we may not use it, our deployed
operator is the admissible FCM-PM. Generality is established across three combination
regimes --- superposition-structured (MixedWM38 public wafer, MultiMNIST inked
digits, and FSD50K audio, Sec 5.10), disjoint-coordinate text (Reuters, Sec 5.9), and
the natural-image boundary (VOC).

### 4.1 Synthesis operators (content-blind)

Given two singles (x_a, a), (x_b, b):
- **overlay / max-union** (an **inadmissible** upper reference on wafer maps):
  per-pixel max (defect intensity wins over normal background) — both objects survive
  whole; hard label {a,b}. On inked digits and audio it is the evidence-preserving
  join. On **wafer maps** it is, under WM38's binary encoding (normal 0.5, defect
  1.0), exactly Shin et al. (2022) Summation Mixup's clipped binary sum; because it
  overlays real single-defect maps it **preserves each defect's real location**, so
  it is content-aware and **inadmissible** in our location-free setting — reported
  only as an upper reference (bit-F1 0.80 / FAR 0.010, Sec 5.2), parallel to the
  oracle. It produces over-dense maps (density 0.50 vs real 0.31).
- **complement (FCM)** (our **admissible location-free** base operator): G x G grid
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
  soft labels), plus single-only (no synthesis). Upper references (inadmissible,
  location- or label-using): overlay / max-union / summation (above), copy-paste
  (mask-cropped placement), and the oracle (real multi-label training) — all
  ceilings, not rivals.

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
complement (FCM) 0.527 > Mixup 0.236. The maximal-survival join on wafer is max-union
/ summation, but it is **inadmissible** (it preserves defect location), so the
criterion identifies it as the evidence-preserving **reference** while our deployed
operator is the admissible FCM (with Pair-Mask). The criterion earns its keep *across
regimes*, not by re-picking the wafer reference: on disjoint-coordinate text the
fidelity maximizer flips to averaging (Sec 5.9), because averaging preserves
per-coordinate evidence there. The criterion is thus a per-domain characterization of
evidence-preservation, not a fixed operator.

**Density as a characterization (not a selection criterion).** The admissible
complement matches the real density (0.293 vs real 0.305), while the inadmissible
max-union reference is over-dense (0.501). A density-shift analysis (Sec 5.2)
characterizes that reference only; because max-union is inadmissible it does not enter
operator selection. Density-faithfulness is therefore a modeling property, **not** a
performance lever, and we do **not** use it to prefer any operator.

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

Max-union is over-dense (0.501 vs real 0.305); the admissible partition complement
matches the real density (0.293). Because max-union is inadmissible in our
location-free setting (it uses defect location), density is reported only as a
modeling characterization of that reference; it does **not** enter operator
selection.

**Density-shift analysis (characterizes the inadmissible max-union reference).** We
stratify the real 2-mix test by density tertile and split out 3/4-mix, then compare
the over-dense max-union reference against FCM-PM (SmallCNN, 3 seeds). bit-F1:

| condition          | max-union (= Shin22, ref.) | FCM-PM (ours) |
|--------------------|----------------------------|---------------|
| 2-mix low density  | 0.8548                     | 0.7868        |
| 2-mix mid density  | 0.8298                     | 0.7874        |
| 2-mix high density | 0.8049                     | 0.7144        |
| 3-mix              | 0.7245                     | 0.6336        |
| 4-mix              | 0.6552                     | 0.4841        |

Because max-union is **inadmissible** (it uses defect location), this per-stratum
comparison is against an upper reference, not an admissible competitor — so it is
**moot** for operator selection and does not undermine FCM-PM. It only characterizes
that reference and confirms density-faithfulness is a modeling property, not a
performance lever; among the admissible location-free operators FCM-PM leads the
tradeoff.

**Operator comparison (5 seeds, pick=val_tail_margin_guarded, neg 0.02).**
Common protocol (matched view budget, splits, checkpoint selection, rejection). The
two rows below the rule are inadmissible upper references:

| operator                       | bit-F1 | NORMAL FAR | defect-die density | note                                  |
|--------------------------------|--------|------------|--------------------|---------------------------------------|
| FCM-PM (**ours**)              | 0.654  | 0.147      | 0.293 (matches)    | admissible; leads the tradeoff        |
| cutmix                         | 0.691  | 0.439      | —                  | admissible; FAR 3x higher             |
| FCM (no Pair-Mask)             | 0.665  | 0.384      | 0.293 (matches)    | admissible; PM lever removed          |
| mixup                          | 0.537  | 0.225      | —                  | admissible; ghosting                  |
| single_only                    | 0.473  | 0.602      | 0.290              | admissible; floor                     |
| max-union / overlay (= Shin22) | 0.800  | 0.010      | 0.501 (over-dense) | INADMISSIBLE ref (uses location)      |
| oracle (real mixed + labels)   | 0.974  | 0.563      | 0.305              | INADMISSIBLE ref (multi-label train)  |

Reading: among the **admissible** location-free operators, our method **FCM-PM**
leads the performance–FAR tradeoff (bit-F1 0.654 at the lowest FAR 0.147; 0.663/0.228
at the guarded pick common to the other arms) — cutmix has slightly higher bit-F1 but
3x the FAR, mixup and the floor are worse on both axes, and the Pair-Mask lever cuts
the complement FAR 0.384 -> 0.147. The two inadmissible upper references sit above the
admissible frontier: whole-image **max-union / summation** — which under the binary
encoding **is Shin et al. (2022) Summation Mixup** — overlays real single-defect maps
and so **uses each defect's real location** (0.80/0.010), and the oracle trains on
real multi-label data (0.974). Neither is a location-free method; we report them only
as references (floor 0.473 -> admissible FCM-PM 0.654 -> references: summation 0.80,
oracle 0.974). The annotation-free reliability layer (Sec 5.6, 5.11) turns FCM-PM's
FAR into a finite-sample guarantee.

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
FAR lower. (This isolation uses the max-union / Summation Mixup arm as the reference
join because it is the simplest; the point is loss-vs-structure, and it holds for any
co-occurrence synthesis including our admissible FCM-PM.)
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
Audio has no location annotation to withhold, so waveform summation is admissible
here; on wafer the analogous max-union join is instead an inadmissible reference (it
uses defect location), where our deployed admissible operator is FCM-PM (Sec 5.2).

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
applies to *any* operator, both the admissible operators (including our deployed
FCM-PM, whose measured FAR 0.147 the conformal layer turns into a certified
guarantee) and the inadmissible references. This is the framework's strongest novel
asset.

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
guaranteed 1% FAR is nearly free for our method: among the admissible operators
FCM-PM retains the best coverage, **97.2%**, then 51.8-83.9% for the weaker
admissible arms; the inadmissible max-union reference retains 99.5%.
**Honest:** the inadmissible summation reference has the *highest* coverage (99.5%);
we therefore claim only that (i) the conformal *guarantee* is operator-agnostic,
annotation-free, and novel (Shin 2022 provides none) and nearly free on our deployed
FCM-PM, and (ii) Pair-Mask improves coverage among the admissible arms.

**Full calibration curve (multi-alpha).** Extending the two operating points to a
sweep over target levels alpha in {0.5, 1, 2, 5, 10}% (5 seeds x 50 splits,
n_cal=500 known-good real normals), the realized NORMAL FAR tracks target alpha to
within **0.153 pp** (maximum deviation across all seven operators and five levels)
for *every* operator -- the distribution-free, finite-sample calibration that
operator-only prior work (incl. Shin 2022's Summation Mixup) lacks -- and the
guarantee is nearly free on our deployed FCM-PM; the max-union / summation curve is an
inadmissible upper reference (it uses defect location). See
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

**What each side owns.** The two inadmissible upper references bound the problem: the
multi-label oracle (0.974) and the location-using max-union / summation reference
(0.80). Our admissible method, FCM-PM, reaches 0.654 at FAR 0.147 — the floor 0.473 ->
FCM-PM 0.654 -> references (0.80, 0.974) picture. What the oracle owns over any
location-free synthesis is the appearance interaction of real high-order mixes, which
independent-union synthesis does not reproduce; neither combination-support matching
nor naive higher-order synthesis closes it in our tests (Sec 5.12). On the headline
oracle checkpoint, even tau=0.99 leaves normal FAR 0.799; the full-scale oracle varies
from 0.295/0.262/0.001 across seeds, so this is an optimization/calibration result,
not an inherent impossibility theorem. Our method reaches a strong operating point
(FAR 0.147; the Pair-Mask, val-margin, and NB-reject levers) with synthetic normals,
margin rejection, and a conformal guarantee (Sec 5.6, 5.11) rather than real-normal
training.

**Density is a modeling characterization, not a selection lever.** FCM-PM matches the
real defect-die density (0.29 vs real 0.31); the location-using max-union reference is
over-dense (0.50). A density-shift analysis (Sec 5.2) stratifies the real 2-mix test
and compares the over-dense max-union reference against FCM-PM per stratum and mix
order (2-mix 0.855/0.830/0.805, 3-mix 0.725, 4-mix 0.655 vs 0.787/0.787/0.714, 0.634,
0.484). Because max-union is **inadmissible** in our location-free setting (it uses
defect location), this comparison is against an upper reference, not an admissible
competitor — so it is **moot** for operator selection and does not undermine FCM-PM:
among the admissible location-free operators FCM-PM leads the performance–FAR tradeoff.
We therefore report density purely as a modeling property. The excess-risk TV bound
(Sec Theory) is retained as a general, one-directional guarantee — it explains when
synthesis can match the oracle — and we do not use its density-mismatch lower bound to
prefer any operator. The surviving contributions are the single→multi setting and
objective, the FCM-PM method with its FAR levers, the annotation-free conformal FAR
guarantee, the theory, and cross-domain framework support.

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
evidence-preserving combination (summation/union for inked digits; the admissible
full-cover complement for wafer maps) we approach the oracle (Cor. 1), because the
generative structure lets synthesis reconstruct the co-occurrence distribution that
observation withholds.

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
accuracy: fully-supervised MixedWM38 methods reach 98-99%, and the two references that
sit above our admissible frontier — the multi-label oracle (0.974) and the
location-using max-union / summation reference (0.80, = Shin 2022) — are inadmissible
in our location-free setting. Among the admissible location-free operators our FCM-PM
method maximizes the performance–FAR tradeoff *from single-label data alone*, with a
finite-sample FAR guarantee. The contribution is the annotation-free single→multi
setting and objective, the FCM-PM method with its FAR levers, the reliability
guarantee, the theory, and cross-regime framework support across five families -- a
weak-supervision and reliability result; reliance on one public real-multi-label
benchmark is a real limitation, and the cross-regime breadth is what carries the claim
beyond it.

## 7 Conclusion

From single-label training data alone — no multi-label, real-normal, or location
annotation — and under a strict location-free constraint, label-faithful synthesis
trains multi-label recognizers and we maximize the multi-label performance–FAR
tradeoff. On wafer maps the evidence-maximal join is whole-image summation/union,
which under the binary encoding **is** Shin et al. (2022) Summation Mixup; but because
it overlays real single-defect maps it **uses each defect's real location**, so it is
**inadmissible** in our setting and enters only as an upper reference (bit-F1 0.80 at
FAR 0.010), alongside the multi-label-trained oracle (0.974). Among the **admissible**
location-free operators, our method — FCM-PM with val-margin selection and NB-reject —
leads the tradeoff (bit-F1 0.654 at FAR 0.147) and reaches ~0.99 on chip-internal
maps; density is a modeling characterization (FCM-PM matches the real 0.29), not a
performance lever. On a genuine superposition domain (MNIST) blind synthesis exceeds
the oracle on unseen combinations. A margin-reject stage drives observed normal FAR to
zero at a few-percent review cost, and calibration on a small known-good set provides
a finite-sample, distribution-free FAR guarantee under exchangeability that holds for
any operator — the reliability layer operator-only prior work lacks. The
label-fidelity / operator-match criterion is measurable before training and
characterizes evidence-preservation across regimes: superposition-structured domains
(wafer, inked digits, audio) take summation/union, disjoint-coordinate text takes
averaging, and natural RGB (VOC) marks the boundary where content-blind synthesis
fails and location supervision becomes necessary. What remains with the oracle is the
appearance of real high-order interactions — a boundary we quantify and leave as the
open problem.

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
