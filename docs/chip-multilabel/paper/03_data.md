# 3. Data

## 3.1 Chip image format

A *chip* is a 200×200 image cropped from a wafer fail-bit map. Pixels
are encoded by a fixed palette where grade 0 = white (no failure),
grade 1 = grey (mild failure), grades 2 and 3 = saturated colours
(severe). In the current iteration the source distribution is
heavily concentrated at grades 0 and 1; iterations introducing grade
2/3 elevation are queued for follow-up (§9).

## 3.2 Data synthesis pipeline

All training and evaluation data is *synthesised* in the sister repo
`D:/project/known-cnn`, file `_sample_gen.py`. The pipeline uses
WM-811K wafer maps grouped by class as the source distribution and
then composes 200×200 chips using one of three rules:

1. **Single-defect chips** (TRAIN_CLASSES): for each of
   `bank_boundary`, `fork`, `scratch`, `scratch_rot`, sample a
   class-conditional chip-coordinate distribution and stamp the
   corresponding fail-bit pattern onto a blank chip.
2. **Combo chips** (5 entries): `min`-blend two single-defect chips
   from distinct TRAIN_CLASSES (excluding `scratch+scratch_rot`,
   which is ill-defined as the rotation makes the two defects
   pixel-overlapping on the rotated stamp).
3. **`Normal` chips** (`_make_normal_chip`, sister-repo
   `_sample_gen.py:151`): sample a `BASELINE` background — low-grade
   speckle that mimics no-defect wafer noise.
4. **`Invalid` chips** (sister-repo `_sample_gen.py:915`): stamp the
   QC orange border (RGB ≈ (240, 160, 0)) on a near-white chip; the
   inference-side detector for these is a deterministic colour
   heuristic (`chip_multilabel/decision_tree.py:36`, `detect_invalid`),
   not a learned head.

## 3.3 Train / val / eval splits

| split | n chips | source                                                        |
|-------|--------:|---------------------------------------------------------------|
| train |     327 | sister repo `classification_chips/`, single-label, 5 classes  |
| val   |      82 | same source, 4:1 split with train                             |
| eval  |    2200 | `D:/project/data/wm-811k/chip_multilabel_eval_full/`          |

The **training data is single-label**: each training chip has exactly
one of the 4 defect classes (or `invalid_main`) as its ground truth,
and there are no `Normal` chips in train. The val set is also
single-label and is used purely for threshold tuning and temperature
scaling (no model selection beyond loss curves).

The **eval set is multi-label** by construction. It contains 11
logical classes:

| Group            | Class                       | n eval |
|------------------|-----------------------------|-------:|
| Single defect    | `bank_boundary`             |    240 |
|                  | `fork`                      |    240 |
|                  | `scratch`                   |    160 |
|                  | `scratch_rot`               |    160 |
| Combo (2 defect) | `bank_boundary+fork`        |    160 |
|                  | `bank_boundary+scratch`     |    160 |
|                  | `bank_boundary+scratch_rot` |    160 |
|                  | `fork+scratch`              |    160 |
|                  | `fork+scratch_rot`          |    160 |
| Other            | `Normal`                    |    160 |
|                  | `Invalid`                   |     40 |
| **Total**        |                             | **2200** |

Combo classes are encoded as multi-hot labels (e.g. `bank_boundary+fork`
sets both `bank_boundary` and `fork` bits to 1). `Normal` has all
defect bits 0. `Invalid` is a special class whose ground truth is
established by the QC border heuristic (chip excluded from the
defect-class bitmap entirely).

The combo `scratch + scratch_rot` is **excluded from the eval set** —
the same rotation invariance that makes `scratch` and `scratch_rot`
distinguishable (when present alone) makes the combo ill-defined: a
rotated scratch stamped on a non-rotated scratch overlaps pixel-wise.

## 3.4 Sanity checks (sister repo)

The synthesis pipeline runs three sanity checks before publishing
chips:

- **Per-class fail-bit density.** Each chip's grade-1+ pixel ratio
  must lie in the per-class histogram window measured from real
  WM-811K samples.
- **Combo orthogonality.** For a 2-defect combo, each contributing
  defect's pixel set must overlap by at most 30% with the other —
  we are simulating co-occurrence, not duplication.
- **Border purity.** `Invalid` chips must satisfy the
  `detect_invalid` heuristic (white-area ratio ≥0.95 + ≥3 of 4
  borders containing orange pixels within tolerance).

## 3.5 Backbone (T0)

The reference checkpoint under test, henceforth **T0**, is

```
D:/project/known-cnn/outputs/logs_chip/chip5_round4_v14_260505_061558_running/best_model.pth
```

a `convnextv2_base.fcmae_ft_in22k_in1k_384` initialised from
ImageNet FCMAE pretrain → ImageNet supervised → TAPT (task-aligned
pretraining) on the same synthetic chip distribution → final
single-label CE on 5 classes. Val 5-class accuracy is 1.0000 at
epoch 1; we view the multi-label benchmark as the *only*
discriminative test of the model and treat val accuracy as a
hyperparameter-selection signal rather than a quality signal
(§6 documents that single-label val accuracy is a poor predictor
of multi-label macro-F1: T1_LS25 hits val 1.0 but only 0.8663
multi-label, while T1_LS20 hits val 0.9756 and 0.9268 multi-label).

**Why TAPT instead of pure ImageNet?** The chip distribution is far
from natural images; pretrain on the same synthetic distribution
gives the backbone several percent of multi-label headroom on the
eval set. We retain TAPT throughout this paper and treat it as part
of T0. Re-pretraining experiments are deferred.

## 3.6 Limitations of the synthesis pipeline

The synthesis pipeline has two known limitations that bound the
upper macro-F1 we can achieve:

1. **Combo difficulty.** `min`-blend produces combo chips whose
   per-class fail-bit pattern is *weaker* than the source single
   chips (because `min` zeroes overlapping cells). Phase B+ work
   plans a `--source-strength-pct` filter to use only top-strength
   source chips when blending, which we hypothesise will lift the
   combo-class macro-F1 by up to 0.03.
2. **Grade variation.** Source chips are concentrated at grades 0–1.
   Generating chips with elevated grade-2/3 pixel populations
   (`--grade-mode {default, elevated_2, elevated_3}`) is queued; we
   expect this to test scratch vs scratch_rot under saturated
   colour conditions, where the two are visually most distinct.

These two are deferred until Phase A (this paper) is closed.

## 3.7 Train-time synthesis: multi-source CutMix (iter 6 / T7)

The data described above (`classification_chips/`, single-label
sources for train/val; `chip_multilabel_eval_full/`, multi-label
synthesised eval) is *fixed* across iters 1–5. Iter 6 introduces a
distinct mode of synthesis that operates **at training time** rather
than at dataset-construction time: multi-source CutMix
(Yun et al. 2019, arXiv:1905.04899; Wang et al. 2024 SpliceMix,
arXiv:2311.15200; Wang et al. 2024, arXiv:2405.13451 — multi-label
label propagation under CutMix; Wightman et al. 2021 ResNet strikes
back, arXiv:2110.00476 — BCE + mixup-style augmentation in the
multi-label recipe).

This subsection clarifies the distinction between the two synthesis
modes because the paper relies on both:

- **Eval-set synthesis** (§3.2 above) generates 2200 multi-label
  chips offline by `min`-blending pairs of single-defect chips.
  These are the gold-standard ground-truth combos; the model never
  sees them during training.
- **Train-time synthesis** (T7) generates multi-positive *training*
  samples on the fly during each forward pass. The chip-level data
  pipeline still loads single-label sources from
  `classification_chips/`; CutMix is applied as a *batch-time
  augmentation* between two single-label samples drawn from the
  current batch.

### 3.7.1 CutMix mechanics

Given two single-label chips `(x_A, y_A)` and `(x_B, y_B)` drawn
from the current batch with `y_A ≠ y_B` (distinct TRAIN_CLASSES):

1. Sample patch area fraction `λ ∼ U[0, 1]` (uniform).
2. Sample a random rectangular patch `(rx, ry, rw, rh)` of area
   `λ · 200²` from `x_A`.
3. Replace that patch in `x_B` with the same patch from `x_A`.
4. The mixed sample's **multi-hot target** is
   `y_mix_c = λ · 1[c=y_A] + (1−λ) · 1[c=y_B]`,
   i.e. a soft *multi-positive* target proportional to patch area.
5. The loss becomes BCE on the multi-hot soft target (CE cannot be
   used because the target now has two non-zero classes).

CutMix is applied per-batch with probability `p`. T7's sweep
(§5.6.3) finds a sharp peak at `p=0.5`. Mixing only operates between
defect classes (mixing with `Normal` would defeat the purpose;
mixing with `invalid_main` is excluded by class-mask).

### 3.7.2 Why it differs from eval-set min-blend

The eval-set `min`-blend (§3.2.2) and CutMix differ on three
fundamental axes:

| dimension                    | eval-set `min`-blend                       | T7 CutMix                                     |
|------------------------------|--------------------------------------------|-----------------------------------------------|
| **applied at**               | dataset construction (offline)             | each training batch (online)                  |
| **mixing operator**          | pixel-wise `min` over both chips           | rectangular patch replacement                 |
| **defect signal**            | weakened (`min` zeroes overlap)            | preserved (each patch carries full intensity) |
| **target encoding**          | strict multi-hot {0, 1}                    | soft multi-hot ∈ [0, 1] proportional to area  |
| **purpose**                  | benchmark eval combos                      | training supervision for combo capability     |

The two modes are complementary: the eval-set defines the
performance target (multi-label combo recall on `min`-blend chips),
while T7 CutMix *trains* the model to handle multi-hot outputs
under any mixing operator. Empirically T7c's bb+sr recall on
`min`-blend eval chips lifts 0.3250 → 0.9562, suggesting that
patch-CutMix-trained models generalise to `min`-blend evaluation —
the model learns *combo capability* in general, not the specific
pixel-mix operator used at training.

### 3.7.3 Effective combo training examples

With train_n = 327, batch_size = 32, ≈10 batches per epoch over 8
epochs, `p = 0.5` gives roughly:

```
total batches:   8 epochs × 10 batches = 80
mix batches:     80 × 0.5 = 40
mixed chips per batch: ≈16 (half of batch is mixed per pair)
total mix chips: 40 × 16 ≈ 640
```

Plus the model still sees ≈2616 clean single-defect chips
(327 chips × 8 epochs). The 640 mix samples cover all 6 ordered
defect pairs roughly uniformly (≈100/pair), giving the model
enough combo gradient signal to develop multi-positive output
capability without losing single-class identity.

### 3.7.4 Why CutMix and not a pre-synthesised combo train set

A natural alternative is to add `min`-blended combo chips to the
training set (matching the eval-set construction). T7's online
CutMix is preferred because:
1. **It is hyperparameter-cheap.** No new dataset directory, no
   new label files, no rebuild step. `p` is the only knob.
2. **It exposes the model to fresh combos every batch.** A fixed
   combo train set would be re-shown each epoch; CutMix samples a
   new (λ, patch position, source pair) every step.
3. **It generalises across mixing operators.** As shown above,
   patch-mix training transfers cleanly to `min`-blend eval.

A direct comparison (T7-CutMix vs T7-pre-synthesised-combos) is
queued for Phase G.
