# A 1.16M-Parameter Object-Grid Network Beats an 88M Pixel CNN for Wafer Map Classification: Categorical Signals Must Not Be Interpolated

Paper C draft v0.1 (target: IEEE Trans. Semiconductor Manufacturing / TII).
All numbers measured (sources: docs/wafer-ensemble/DISCOVERY.md, RESULTS.md,
docs/chipgrid/RESULTS.md; runs under logs_chipgrid/).

## Abstract (skeleton)

Wafer-map defect classification is dominated by large pixel CNNs applied to
high-resolution fail-bit images. We show that a two-stage object-aware
pipeline — a chip-level classifier that labels each die region with an
object type, followed by a 1.16M-parameter CNN over the native 32x32
chip-grid of one-hot object labels — reaches val macro-F1 0.9946 on a
33-class synthetic wafer benchmark, beating an 88M ImageNet-pretrained
ConvNeXtV2 on raw pixels (0.9851) with a 76x smaller model and ~750x faster
training, and exceeding the oracle ensemble ceiling of the constituent
models (0.9919). The enabling factor is representational, not architectural:
object-identity maps are CATEGORICAL, and any interpolating resize (bicubic
or nearest at non-integer scale) corrupts them — the same information
presented as an interpolated 384px 3-channel compound image caps at 0.9784.
Block-integer expansion or native-resolution processing removes this
ceiling. The pipeline is robust to 10% simulated chip-classifier error
(0.9870) and, counter-intuitively, DROPS the raw pixel channel: the chip
classifier has already consumed the pixel evidence, and re-presenting it
adds noise (obj-only 0.9946 vs obj+pixels 0.9707).

## 1 Introduction

A wafer map is not a natural image. It is a lattice of dies, each of which
either carries a defect object of some type or does not; the wafer-level
defect class is defined by WHICH object types occupy WHICH regions of that
lattice. The dominant approach nevertheless treats the rendered map as a
photograph: a large ImageNet-pretrained CNN consumes a high-resolution
fail-bit image and is asked to rediscover, through tens of millions of
parameters, structure that the manufacturing process already made explicit.

We propose to make the structure explicit instead. A two-stage pipeline
first classifies each die-region crop with a small chip-level object
classifier, producing a 32x32 grid of categorical object labels; a
1.16M-parameter CNN then classifies the wafer directly from the one-hot
encoding of that native grid. On a 33-class benchmark under a fixed
fair-evaluation protocol, this reaches val macro-F1 0.9946 — beating the
88M pixel baseline (0.9851) with a 76x smaller model and ~750x faster
training (<1 minute), and exceeding the 0.9919 oracle ceiling of the
constituent pixel and grid models, which rules out an ensemble explanation:
the REPRESENTATION is the contribution, not model capacity.

Two findings explain why. First, object-identity maps are categorical, and
any interpolating resize corrupts them — the same information presented as
a bicubic-resized 384px compound image caps at 0.9784, and we show
interpolation manufactures fractional object identities on 6.5% of pixels
in a typical map. We formulate a simple rule (block-integer expansion or
native-resolution processing; never interpolation) that removes this
ceiling. Second, once stage 1 has consumed the pixel evidence, feeding
pixels to stage 2 again is harmful (0.9707 with pixels vs 0.9946 without):
the grid is a sufficient statistic and the raw channel is noise. A
10-seed experiment on real public WM-811K maps supports the categorical
principle (+0.060 macro-F1 for categorical treatment over standard
interpolated-grayscale input).

Contributions: (1) a two-stage object-grid representation with which a
1.16M model surpasses an 88M pixel CNN and the two-model oracle ceiling;
(2) the categorical-resize rule and its measured violation costs, on both
synthetic and real public data; (3) robustness and encoding ablations
showing the effect is representational (one-hot > integer-id > single-object
> pixels-only; robust to 10% stage-1 error).

## 2 Method

**Stage 1 — chip object classifier.** A CNN classifies each 200x200 die
crop into 5 object classes (bank_boundary, fork, scratch, scratch_rot,
invalid_main). Crops are saved inline during wafer generation with true
per-chip object labels (75% primary + 25% mixed environments), avoiding
post-hoc folder labeling errors. Stage-1 inference over a wafer yields a
32x32 object-identity map (one id per die region).

**Stage 2 — ChipGridCNN over the native grid.** The wafer classifier
consumes the 32x32xK one-hot encoding of the object map at NATIVE grid
resolution — zero interpolation anywhere in the pipeline. Architecture
(measured 1.16M params): six 3x3 conv layers (64-64-128/s2-128-256/s2-256,
BatchNorm+GELU), global average pooling, dropout 0.1, linear head. No
pretraining; trains in under a minute.

**Categorical-resize rule (block_expand).** When a categorical map must be
spatially resized (e.g., for a pixel-CNN consumer), the only
category-preserving operation is block-integer expansion: each source cell
maps to an integer block of output pixels (target_size // grid px/cell),
with the remainder rows/columns distributed evenly among cells — never
bicubic (creates fractional identities; measured 6.5% fractional pixels on
a typical map) and never naive nearest at non-integer scale (duplicates/
drops cells unevenly). Reference implementation: `block_expand_2d(map,
H, W)` — integer division for the base block, remainder spread by even
striding; output contains exactly the input's value set (0% fractional).

## 3 Experiments (all measured)

Fair-eval protocol: same split (0.8/0.1/0.1 stratified, seed 42), same
per-class sample, 30 epochs, no TTA (angle-bearing classes forbid it).

| Model                          | Input                 | Params | val_f1 | test_f1 |
|--------------------------------|-----------------------|--------|--------|---------|
| R-only ConvNeXtV2 (pretrained) | 1024px fail-bit pixel | 88M    | 0.9851 | -       |
| compound 3ch BICUBIC 384       | R+obj_id interpolated | 88M    | 0.9784 | 0.9736  |
| obj-only 4-layer               | 32x32 one-hot         | 0.4M   | 0.9844 | -       |
| V3 R+obj (6ch)                 | pixels + one-hot      | 1.16M  | 0.9707 | -       |
| V3 obj-only (ours)             | 32x32x5 one-hot       | 1.16M  | 0.9946 | 0.9872  |
| Tier-1 logit ensemble          | pixel + grid models   | -      | 0.9886 | -       |
| Oracle ceiling (either-correct)| -                     | -      | 0.9919 | -       |

Key findings:
1. **Interpolation is the ceiling**: same information, interpolated to 384px
   -> 0.9784; native grid -> 0.9946 (errors -75%).
2. **Pixels become noise after stage 1**: adding the R channel to the grid
   CNN HURTS (0.9707 vs 0.9946) — the chip classifier already consumed the
   pixel evidence.
3. **Beats the oracle ceiling**: V3 alone (0.9946) > best-possible ensemble
   of pixel+grid baselines (0.9919) — not an ensemble effect, a
   representation effect.
4. **Robust to stage-1 error**: 10% synthetic chip-label noise -> 0.9870
   (the 6ch variant degrades to 0.9707 baseline even without noise).
5. Encoding ablation (n=100/class, seed 42; measured):

| Variant | Encoding             | in_ch | val_f1 | test_f1 | Reading                              |
|---------|----------------------|-------|--------|---------|---------------------------------------|
| V0      | fail-bit R only      | 1     | 43.59% | 43.85%  | no object info: ~41% structural cap   |
| V2      | single-object binary | 2     | 65.43% | 64.79%  | one object channel frees one group    |
| V1      | integer id / 5       | 2     | 95.05% | 97.26%  | integer channel nearly saturates      |
| V3      | one-hot 5ch          | 6     | 96.89% | 98.79%  | zero information loss — best          |

   V0->V1 +51pp (the object-identity channel carries the task); V1->V3
   +1.84pp (one-hot separation beats ordinal integer compression — integer
   ids impose a false order on categories). V2 sub-variants (any single
   object: 51-66%) confirm the gain is the full identity map, not any one
   object. Seed stability: V3 across seeds {1,7,42,100,234} = 96.9-99.4% val.
   At n=220/class V3 reaches 99.45% val / 98.66% test; with 10% chip-label
   noise 99.17%.

## 4 Discussion

**Why the native categorical grid wins.** The wafer class is a function of
the object-occupancy pattern on the die lattice — nothing else. The 32x32
one-hot tensor is therefore a sufficient statistic for the label, at a
thousandth of the pixel input's dimensionality; everything a pixel CNN must
learn to extract is already present, losslessly, in 5,120 binary values.
Interpolation breaks exactly this sufficiency: a bicubic kernel averages
adjacent identities into fractional values that correspond to no physical
object, and the downstream network must then spend capacity separating
manufactured ambiguity from real signal. Nearest-neighbor resampling at
non-integer scale is not safe either — it duplicates and drops cells
unevenly. Block-integer expansion is the only resize that preserves the
statistic, and processing at native resolution makes the question moot.

**Why pixels become noise.** Stage 1 already converted pixel evidence into
object identity; re-presenting the raw channel to stage 2 offers no new
information but adds palette variation, sensor noise, and rendering detail
that the small network must learn to ignore — measured as a 2.4pp penalty
(0.9707 vs 0.9946). Information should cross the stage boundary exactly
once.

**Operational consequence.** A 1.16M model that trains in under a minute on
CPU-class hardware turns wafer-classifier maintenance from a scheduled
retraining event into an interactive operation: per-line, per-product, even
per-recipe models become affordable, and stage-1/stage-2 can be retrained
independently (stage 2 tolerates 10% stage-1 error).

**Composition with annotation-free multi-label training.** Stage 1 is
trained from single-defect chip crops — the same single-label supervision
studied in our companion work on multi-label synthesis. Together they form
an annotation-light stack: single-defect crops are the only labeled input,
from which chip-object labels, wafer-grid representations, and multi-defect
recognition are all manufactured.

### 3.1 Real-data anchor: WM-811K (public)

On 1,779 real WM-811K maps (cca 8-class, categorical pixel values), the same
small CNN over 10 seeds: nearest+one-hot 0.7633 +-0.052 > nearest-gray
0.7248 +-0.049 > bicubic-gray 0.7036 +-0.055. Treating the map as
CATEGORICAL (one-hot after category-preserving resize) beats the standard
interpolated-grayscale input by +0.060 macro-F1 (7/10 paired seeds), with
the gain driven primarily by categorical encoding (+0.039, 8/10). Honest
scoping: the pure nearest-vs-bicubic component is within noise on these
already-nearest-rendered 224px images; the dramatic interpolation ceiling is
the synthetic 33-class result (0.9784 -> 0.9946). Real public data supports
the thesis's core: categorical signals must be treated categorically.

## 5 References (to compile into bibtex)

- WM-811K dataset: Wu, Jang, Chen \& Lee, "Wafer Map Failure Pattern
  Recognition and Similarity Ranking for Large-Scale Data Sets," IEEE
  Trans. Semiconductor Manufacturing, 2015. (TODO-verify exact citation)
- MixedWM38: Wang et al., IEEE TSM 2020 (deformable conv, mixed-type).
- Representative wafer-map CNN lines on WM-811K (2-3 papers, TODO-verify:
  recent TSM/JIM CNN classifiers reporting 96-99% on 9-class).
- ConvNeXtV2: Woo et al., CVPR 2023 (FCMAE pretraining) — pixel baseline.
- Two-stage detection analogy: R-CNN family (Girshick et al.) — regional
  object hypotheses feeding a downstream classifier.
- Companion work: multi-label from single-label synthesis (paper A).

## 6 Status / TODO before submission

- [x] Core experiments (V0-V3, robustness, fair-eval table) — in repo.
- [x] WM-811K real-data anchor (Sec 3.1, 10 seeds, honest scoping).
- [x] V0-V3 encoding table ported; V3 seed stability included.
- [x] Corruption figure (BICUBIC 6.5% fractional vs block_expand 0%).
- [x] Intro / Method / Discussion full prose.
- [ ] Bibtex compile + TODO-verify citations against actual papers.
- [ ] LaTeX port (after venue choice).
- [ ] USER DECISION: rerun fair-eval at active-class-27 (class set grew
      since the 33-class experiments) or scope claims to the measured
      33-class protocol — either is defensible; rerun costs GPU/CPU time.
