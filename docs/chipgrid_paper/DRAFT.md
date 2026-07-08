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

- Wafer maps are grids of dies, not natural images; pixel CNNs ignore this.
- Two-stage: (1) per-chip object classifier (5 classes) on 200x200 crops;
  (2) wafer classifier on the 32x32 grid of chip-object labels (one-hot).
- Headline: 76x smaller, 750x faster, +0.95pp over the 88M pixel baseline,
  exceeding the two-model oracle ceiling — the grid representation is the
  contribution, not model scale.

## 2 Method

- Stage 1: chip object CNN (bank_boundary / particle-type / scratch types /
  invalid), trained on inline crops with true per-chip labels.
- Stage 2: ChipGridCNN over 32x32xK one-hot object maps at NATIVE grid
  resolution — zero interpolation anywhere.
- **Categorical-resize rule**: any spatial resize of categorical maps uses
  block-integer expansion (block_expand: integer px/cell, remainder spread),
  never bicubic/nearest interpolation. TODO: formal statement + failure
  example figure.

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
5. Encoding ablation (V0-V6): one-hot > integer-id (ordinal artifact) >
   interpolated variants. TODO: pull full V0-V6 table from
   docs/chipgrid/RESULTS.md.

## 4 Discussion

- Why native categorical grids win: wafer classes are defined by WHICH
  object occupies WHICH region of the die lattice; the 32x32 one-hot tensor
  is the sufficient statistic, and interpolation manufactures classes that
  do not exist (fractional object identities).
- Practical: 1.16M model trains in <1 min — enables per-line retraining.
- Relation to paper A: stage-1 chip labels come from single-defect
  supervision; the two papers compose into an annotation-light wafer stack.

## 5 Limitations / TODO before submission

- [ ] WM-811K real-data anchor: reproduce the pipeline (or its grid-native
      principle) on the public real benchmark — REQUIRED, currently all
      benchmark numbers are synthetic-data internal.
- [ ] V0-V6 full encoding table + per-class breakdown port.
- [ ] Formal block_expand statement + corruption figure (BICUBIC on one-hot).
- [ ] Baseline citations: wafer-map CNN literature (WM-811K lines), 2-stage
      detection analogies.
- [ ] Fair-eval rerun at active-class-27 if class set changed since.
