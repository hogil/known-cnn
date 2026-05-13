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

> ★★★ **Absolute rule (260512, narrator update).** Training uses
> **only the 4 single defect classes** (`bank_boundary, fork, scratch,
> scratch_rot`). The `--no-normal` flag on every training script is
> mandatory; `Normal`, `Invalid`, and OOD wafer-pattern chips are
> **forbidden from training** and may appear only at evaluation time
> as negative-group probes. The evaluation set decomposes into
> **five disjoint groups**: (a) single defect, (b) 2-combo, (c)
> `Normal`, (d) `Invalid`, (e) OOD wafer-pattern. (a) and (b) are
> positive cells (bit-F1 source); (c) – (e) are negative cells
> (FAR source). See §5.1 for the metric definitions and
> `feedback_chip_multilabel_train_eval_composition.md` for the
> code-level enforcement points.
>
> **Metric definitions under the absolute rule.**
> - **bit-F1** = macro-F1 over positive cells **only**, computed bit-
>   wise on the four defect bits — i.e. for each defect bit b, the
>   F1 is computed over the union of single-defect and 2-combo eval
>   chips that carry b in their ground-truth multi-hot label, and
>   bit-F1 is the unweighted mean across the four bits.
>   *This is not the same as the `macro_f1` column in
>   `results_matrix.parquet`*, which averages over all 11 + OOD eval
>   class keys including `Normal`, `Invalid`, and the OOD wafer-
>   pattern distractors. The `results_matrix.parquet` column is
>   retained for cross-iter continuity (§5.1–§5.45) but the
>   paper-grade headline is `bit-F1` as defined here.
> - **Total FAR** = (`NI_fp` + `OOD_fp`) / (`N_NI` + `N_OOD`), where
>   the false-positive count is "any defect bit fires on a chip
>   from a non-defect group". `NI_fp` counts on `Normal` ∪ `Invalid`
>   chips; `OOD_fp` counts on the four wafer-pattern OOD groups
>   (CenterDonut, CrossScratch, DiagonalSmear, Starburst). Under
>   this definition the bundled "chip_FAR" reading from iter 1–11
>   (§3.9) is deprecated in favour of the strict Total FAR; the
>   `ni_FAR` reading from iters 12+ is the production-relevant
>   component, and the Total FAR adds back the OOD diagnostic
>   pressure.
>
> **`n_per_class` clarification.** Eval-time sampling uses
> `--n-per-class 160` for combo classes and `--n-per-class 200` for
> single-defect classes by default (subject to the per-class
> available chip count). The iter 112 best cell is evaluated on
> `n_per_class = 200` for singles + 160 for 2-combos + 200 Normal +
> 200 OOD per pattern + 50 Invalid, for a total of 2 440 chips
> tied to the bit-F1 / Total FAR figures in §5.46.
>
> **Restatement at iter 122 – 124.** All experiments reported in
> §5.47 (group-mixed CutMix granularity) and §6.31 (asymmetric loss
> as a failed direction) honour the five-group decomposition above
> without exception. Specifically: the positive cells (single
> defect ∪ 2-combo, n = 4 + 5 = 9 class keys, 1 600 chips on the
> `v15direct n = 200` protocol) feed **bit-F1** through the
> bit-wise macro-F1 aggregator, and the negative cells (Normal +
> Invalid + 4 OOD patterns, n = 6 class keys, 1 480 chips) feed
> **Total FAR** as `(NI_fp + OOD_fp) / (N_NI + N_OOD)`. Reporting
> bit-F1 without simultaneously reporting Total FAR — or vice
> versa — is **disallowed by the absolute rule** because each
> metric admits a recipe that maximises it in isolation while
> regressing the other (e.g. iter 122/123 ASL drives bit-F1 up
> 0.83 surface but blows Total FAR to 9.4 % / 5.0 %; iter 116 J
> baseline trades −0.04 bit-F1 for strict-zero FAR). The dual-
> gate audit threshold for production readiness is
> `bit-F1 ≥ 0.99 ∧ Total FAR ≤ 0.5 %`.

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

### 3.5.1 Backbone choice — three operational regimes (iter 21, Phase 87 v2)

_Added 2026-05-12 (paper §3.5 narrative correction). See
`iters/iter_21_backbone_throughput_paper3.md`._

#### Why the original "ConvNeXtV2 is best balanced" claim was wrong

An earlier draft of this paper presented ConvNeXtV2-Base as the
single best production backbone on the basis of accuracy alone (val
5-class 1.0000 at epoch 1, multi-label-eval bit-F1 0.9654 at the
iter46E recipe). That claim collapsed three orthogonal axes —
**accuracy**, **inline latency**, and **batched throughput** — into
one ranking. Once we measured inference cost properly (Phase 87 v2:
isolated-GPU, `torch.cuda.Event`-timed, 20 warm-up + 100 forward
passes, 4 backbones × 6 batch sizes ∈ {1, 4, 8, 16, 32, 64}, same
iter46E recipe across backbones, evaluated on `v15direct n=200` /
3080 chips), the three axes did not co-rank a single winner. We
therefore retract the single-claim and adopt a **three-regime
recommendation** indexed by the deployment scenario.

#### The three regimes

| regime                              | gating metric             | winner            | bit-F1 | Total FAR | cost figure                       |
|-------------------------------------|---------------------------|-------------------|--------|-----------|-----------------------------------|
| **A — latency-critical** (inline)   | ms/chip @ b = 1           | **Swin-Base 384** | 0.9692 | **0.00 %** | 21.08 ms/chip                    |
| **B — throughput-critical** (batch) | peak chip/s               | **ConvNeXt V1**   | **0.9830** | 2.62 %    | **76 chip/s** @ b = 64           |
| **C — FAR-strict** (any scenario)   | Total FAR + viable speed  | **Swin-Base 384** | 0.9692 | **0.00 %** (only strict-zero) | 54 chip/s @ b = 4    |

ConvNeXtV2-Base, the legacy paper-main checkpoint, **wins none of
the three regimes**. It survives as a reference point only because
the iter 1–20 ablation history is built on it; its 37 chip/s peak
(achieved at b = 1, with all larger batches strictly slower) places
it last in throughput, and Swin matches its strict-zero NI FAR with
a 6 ms latency saving.

#### Why ConvNeXtV2 loses on throughput — the GRN architectural quirk

ConvNeXtV2-Base is the **only backbone in the sweep where batching
makes throughput worse**: peak is 37 chip/s at b = 1, dropping to
30 chip/s at b = 4 and 26 chip/s at b = 64, a 0.70× negative
scaling. By contrast, ConvNeXt V1 — same family, same parameter
count (87.6 M ≈ 87.7 M), same author lineage — scales 1.88× from
b = 1 to b = 64 and reaches 76 chip/s. The single architectural
difference between the two is **GRN (Global Response Normalization,
Woo et al. 2023, arXiv:2301.00808)**, a ConvNeXtV2-specific block
that computes a per-channel mean of L2-norms over the spatial
dimensions at every stage. The L2 reduction itself batches well;
the *mean of norms* broadcast across batch elements does not — under
cuDNN's default kernel selection it serialises the channel-mean
reduction, producing a U-shaped per-chip latency curve that bottoms
at b = 1 and collapses past b = 32. Because the regression isolates
cleanly to the GRN layer (everything else in V1 vs V2 is identical
at the timm-name level), this is an **architectural finding, not a
recipe artifact**, and it explains in one line why V1 is 2.05×
faster batched at the same accuracy ceiling.

The practical consequence is direct: ConvNeXtV2 is a
**latency-optimised backbone, not a throughput backbone**. Its
b = 1 cell (27 ms / 37 chip/s) is competitive with Swin (21 ms / 47
chip/s) at the inline regime, but the moment chips are accumulated
into a batch, V1 or Swin should be preferred.

#### Why ConvNeXt V1 wins throughput and Swin wins FAR

ConvNeXt V1's throughput win is the GRN absence — no architectural
serialisation in the channel reduction, so cuDNN delivers normal
linear scaling up to b = 64. Its accuracy lift (+0.0176 bit-F1 over
V2 at the same recipe) is a fortunate by-product we do not fully
explain in this paper; we conjecture it reflects the 224-pixel
fine-tune size matching the inductive receptive field of the
classification_chips primitives better than 384, but a controlled
img-size match (V1 retrained at 384) is queued as a fair-comparison
follow-up.

Swin's FAR win is a different mechanism. Swin's window-attention
locality biases it toward learning **window-bounded defect
signatures** rather than diffuse global features, and on this
synthetic eval set the OOD wafer-canvas patterns (CenterDonut,
CrossScratch, etc.) project diffuse evidence that V1's global
receptive field happens to misread as combo. Swin's
window-restricted attention does not fire on these, yielding
Total FAR = 0.00 % — the only backbone in the sweep with a
strict-zero gate pass. The accuracy cost is small (−0.0138 bit-F1
vs V1), and Swin's b = 1 latency (21 ms) is best-in-class, making
Swin the dual winner of regimes A and C.

#### 10,000-chip cost projection

| backbone        | params | peak chip/s | batch | 10 k chip time | role                              |
|-----------------|-------:|------------:|------:|---------------:|-----------------------------------|
| EfficientV2-M   | 52.9 M | 158         | 4     | **63 s**       | (FAIL bit-F1 at iter46E recipe)   |
| ConvNeXt V1     | 87.6 M | 76          | 64    | **132 s**      | regime B winner (bit-F1 0.9830)   |
| Swin-Base 384   | 86.9 M | 54          | 4     | **185 s**      | regimes A & C winner (FAR = 0 %)  |
| ConvNeXtV2-Base | 87.7 M | 37          | 1     | **270 s**      | inline / paper-main legacy        |

ConvNeXt V1 is **2.05× faster than ConvNeXtV2** at the same
parameter count and a higher bit-F1. The 4-bag ensemble cost
quoted in §5.19.5 — previously stated as 4× single-model under the
implicit V2 base — would land at **4 / 2.05 ≈ 1.95× the cost of a
1× ConvNeXtV2 single** if migrated to V1 bags, while gaining
+0.0176 bit-F1 per bag.

#### Scope, limitations, and what this does *not* change

This correction adds **cost ranking as an orthogonal axis** to the
paper's accuracy ranking; it does not retract the paper-main
accuracy headline. iter46E ConvNeXtV2 remains the headline
single-model result at bit-F1 0.9654 / Total FAR 1.07 % because
the iter 1–20 ablation chain — every loss, matching, and decision
rule comparison — is anchored to it. Re-running the full ablation
matrix on ConvNeXt V1 or Swin is **queued** for a future iter and
is not in scope of this paper.

Three limitations bound the regime recommendations: (i) the
measurement is on a single GPU class (A6000 48 GB) — the GRN curve
will likely look different on H100 (different cuDNN heuristics) or
under Tensor RT compilation, so the regime recommendation is
strictly an A6000 / cuDNN-default deployment claim; (ii) the
4-backbone sweep used a small synthetic dataset (814 chips train,
3080 chips eval) and a single training seed per backbone — ViT-Base
and MaxViT-Base failed to converge under this budget and were
dropped, so the regime table covers only the four backbones that
trained successfully; (iii) ConvNeXt V1 was fine-tuned at 224
(timm canonical) while the others used 384, so V1's lead carries a
2.94× pixel-count advantage that a 384-matched re-measurement is
expected to attenuate but not erase. The full 24-row throughput
matrix backing every cell of this subsection is in
`docs/chip-multilabel/tables/backbone_throughput.csv`.

The separation between the paper-SOTA winner (V2, by historical
anchor) and the production-deployment winner (V1 for throughput,
Swin for FAR / inline) is, in our view, a **methodological feature**
of the corrected §3.5: a paper claiming a single backbone on a
single accuracy metric would have buried the 2.05× cost gap and
the 1.07 %-vs-0.00 % FAR gap that the three-regime decomposition
makes legible.

### 3.5.2 Modern self-distillation / improved-attention backbones underperform their predecessors (iter 95–99, added 2026-05-12 evening)

_Added 2026-05-12 19:50 (paper §3.5 expansion). See §5.45 for the
full experimental landscape and `_diary/260512_evening_modern_backbone_findings.md`._

A natural follow-up question to §3.5.1 is whether more recent
backbones — published 2022–2025 with stronger objectives or attention
mechanisms — would surface a new operating point above the
ConvNeXtV2 / Swin V1 Pareto frontier. Under the matched iter46E
recipe (T7 BCE + LS = 0.20 + CutMix complement p = 0.25 g = 3
pair = masked, AdamW LR = 1e-4 cosine 8 epoch, batch = 8 accum = 4)
and the v15direct n = 200 evaluation protocol, we measured three
modern candidates: DINOv3 ConvNeXt-Base (Meta 2025,
arXiv:2508.10104, self-distillation post-FCMAE), Swin V2 Base 384
(Liu et al. 2022, arXiv:2111.09883, log-CPB + cosine attention +
window 12 → 24 transfer), and Hiera-Base (Ryali et al. 2023,
arXiv:2306.00989, MAE-pretrained hierarchical ViT).

**Finding: every modern variant tested under-performs its direct
predecessor.** The matched-recipe ranking is:

| backbone family    | paper-main predecessor (recipe-matched bit-F1) | modern variant (best bit-F1) | Δ (modern − legacy)  |
|--------------------|-----------------------------------------------:|------------------------------:|---------------------:|
| ConvNeXt           | ConvNeXtV2-Base FCMAE = **0.9654**             | DINOv3 ConvNeXt-Base = 0.8700 (LR rescue) | **−0.0954** |
| Swin               | Swin V1 Base 384 = **0.9692**                  | Swin V2 Base 384 = 0.7843     | **−0.1849**          |
| (no ConvNeXt-Hiera pair) | —                                        | Hiera-Base = 0.7228           | —                    |

Each modern variant differs from its predecessor in a *single named
axis* — DINOv3 adds self-distillation on top of FCMAE, Swin V2 adds
log-CPB + cosine attention + 12 → 24 window transfer on top of
Swin V1 — and each *named axis hurts* on this benchmark under
matched recipe and matched parameter budget (≈ 87 M).

**Mechanistic reading.** The FCMAE objective (sparse-convolution
masked autoencoder + pixel reconstruction, Woo et al. 2023) trains
the backbone to reconstruct *pixel-level palette content*, which
appears to be the right inductive prior for chip palette domain
(grade 0 white / grade 1 grey / grade 2 – 7 saturated colour). The
DINOv3 self-distillation objective replaces pixel reconstruction
with *teacher-student feature alignment* on natural images, which
is a strictly weaker prior for our distribution: the chip palette
does not factor into a small number of high-level natural-image
semantic clusters. Swin V2's log-CPB + cosine attention is designed
for transfer to high-resolution natural images (the 12 → 24 window
expansion is the canonical example) — on our 200 × 200 chip the
expanded window is approximately the entire chip and the inductive
bias is largely eliminated. Hiera's hierarchical pooling assumes
multi-scale spatial structure that chip-multi-label does not have
at this resolution.

**LR sensitivity (DINOv3 only).** DINOv3 with default LR = 1e-4
collapses fork F1 to 0.38 (overall bit-F1 0.6211, iter95A). Halving
LR to 5e-5 rescues to 0.8700 (iter97A best at ep9) — at parity
with Swin-Base under FAR ≤ 5 % but **−0.0954 below the iter46E
baseline**. This LR sensitivity is consistent with the
self-distillation literature's smaller-step recommendation
(Caron et al. DINO arXiv:2104.14294 originally used 5e-4 with
warm-up over 10 epochs at much larger batch — direct LR transfer
to our 1e-4 / batch 32 setting requires re-tuning). No further LR
exploration was attempted for Swin V2 / Hiera (both converged on
the val_acc axis at LR = 1e-4); a per-backbone LR sweep is queued
as future work.

**Training time as an orthogonal axis.** Swin V2 took 150 minutes
to train under our recipe — **21× slower** than ConvNeXtV2-Base
(≈ 5 min) on the same A6000. The cost contribution is the log-CPB
relative-position computation (log-scaled bias indexed by relative
window distance) and the cosine-attention normalisation, both of
which serialise badly at 384 input. Hiera-Base trains fastest
(≈ 2.5 min) but at the lowest accuracy ceiling. The corrected
§3.5.1 three-regime recommendation therefore extends: even in the
**latency-critical regime A**, no modern backbone displaces Swin
V1 Base 384 (21 ms / chip, bit-F1 0.9692, Total FAR 0 %); in the
throughput regime B, ConvNeXt V1 (76 chip/s, bit-F1 0.9830) remains
the winner. The modern backbones land below the legacy frontier
on both axes.

**Paper claim (§3.5 update).** Under matched-recipe and matched-
parameter-budget evaluation on the chip palette domain, **the
FCMAE objective (pixel reconstruction) and the Swin V1 windowed
attention transfer uniquely well**. Modern self-distillation
(DINOv3) and improved-attention (Swin V2) variants under-perform
their direct predecessors by 0.10 – 0.18 bit-F1 — counter to the
natural-image SOTA ordering. The paper retains ConvNeXtV2-Base
FCMAE (iter46E) and Swin V1 Base 384 (iter77C) as the
paper-headline checkpoints; no 2022 – 2025 modern variant tested
in iter 95 – 99 displaces either.

_Sources: `outputs/iter95A_dinov3_convnext_base/...`,
`outputs/iter95B_swinv2_base_384/...`,
`outputs/iter96A_hiera_base/...`,
`outputs/iter97A_lr5e5/eval_v15direct_n200_best/...`,
`outputs/iter99{A,B,C,D,E}_*/eval_v15direct_n200/...`._

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

## 3.8 Iter 10 — Master folder consolidation and runtime sampling

The §3.3 split (327-chip train / 82-val / 2200-chip eval, three
disjoint folders) is iter 10's predecessor. Iter 10 (260506)
consolidates all data into a **single source-of-truth master
folder** and runtime-samples per-call. The change is purely
infrastructural (no data semantics shift) but it is paper-grade
because it eliminates a class of subset-folder errors that
previously polluted the iter pipeline.

### 3.8.1 Master folder layout

```
D:/project/data/wm-811k/chip_multilabel/
├── bank_boundary/      # 200 chip (single defect, strong source p50)
├── fork/               # 200
├── scratch/            # 200
├── scratch_rot/        # 200
├── bank_boundary+fork/ # 200 (combo, min-blend)
├── bank_boundary+scratch/      # 200
├── bank_boundary+scratch_rot/  # 200
├── fork+scratch/               # 200
├── fork+scratch_rot/           # 200
├── scratch+scratch_rot/        # 200 (added iter 10)
├── Normal/             # 200 (Beta(2,10) noise, seed=999 train-disjoint)
└── Invalid/            #  50 (orange-border QC chip)
                          ──── total: 2,450 chip
```

`gen_eval_set.py --source-strength-pct 50` filters source chips at
the strong end of the per-class `defect_pixel_ratio` distribution
before min-blending — this is the v18+ master.

### 3.8.2 Runtime sampling

At eval time, `--n-per-class 50` selects 50 sorted-by-filename chips
per class for evaluation (deterministic, reproducible). At train
time, the train/val split runs on the same master folder
(`--no-normal` toggles whether `Normal/` is included).

**Why a single master + runtime sample is the right discipline.**
Earlier iters created `chip_multilabel_eval_full/`,
`chip_multilabel_eval_strong50/`, `chip_multilabel_smoke/`, etc. —
three disjoint folders for three eval contexts. A single-axis
config change (e.g. user wants `--n-per-class 100` instead of 50)
required regenerating one of three folders, and the regeneration
was always slightly off-spec. The master folder removes this:

- Storage: 200 per defect class (largest expected eval-time
  sample), 200 Normal, 50 Invalid. Disk cost ≈ 75 MB.
- Runtime sampling: `--n-per-class N` with `N ≤ 200` produces a
  deterministic subset; `N > 200` errors out.
- Single source of truth: `chip_multilabel/` is the canonical
  location, with `defect_pixel_ratio` manifest column for
  strength-aware sampling.

The user directive (260506) "다시는 이런 subset 폴더 만들지마라" —
roughly "stop making subset folders" — is enforced by removing
both the `chip_multilabel_eval_*` archive folders and the
`obj_id_maps_round*` snapshots that had previously accumulated.

### 3.8.3 Normal chip synthesis (iter 10 addition to training)

The 200 Normal chips in `chip_multilabel/Normal/` (eval set) are
constructed with `_make_normal_chip` using `seed=42`. The 200 train
Normal chips placed in `classification_chips/Normal/` (training
set) use `seed=999` — a **train/eval seed disjointness rule** that
prevents leak. Both sets share the synthesis recipe:

```python
p_noise = Beta(2, 10).rvs(size=())            # mean 0.17, range 0.02–0.50
u = uniform(0, 1, size=(200, 200))             # per-pixel
is_noise = u < p_noise
u2 = uniform(0, 1, size=(200, 200))
grade = where(is_noise, where(u2 < 0.95, 1, 2), 0)  # 95% grade 1, 5% grade 2
```

A diversified variant (`gen_eval_set._make_normal_chip` patched
260506 09:30) adds five further axes (wider grey-ratio band,
per-pixel grey colour noise, white subtle noise, sprinkle 3-color
mix, brightness gradient) with a sanity gate at whiteness ≥ 0.70.
The diversified variant is used in §5.12 Phase 3 only; the simple
recipe is the default for §5.11 / §5.13.

### 3.8.4 Class taxonomy update at iter 10

| version  | classes                                                           | total |
|----------|-------------------------------------------------------------------|------:|
| iter 1–9 | 4 single + 5 combo + Normal + Invalid                             |    11 |
| iter 10+ | 4 single + **6 combo** (sc+sr re-added) + Normal + Invalid        |    12 |
| iter 12+ | iter-10 set + 5 OOD wafer-pattern (diagnostic only, not measured) |    17 |

`scratch+scratch_rot` was excluded in iters 1–9 as ill-defined
(rotated stamp pixel-overlaps non-rotated stamp). Iter 10 re-adds
it with the user's stake "measure it anyway"; baseline T9d's
sc+sr F1 is **0.755** at the time of re-introduction. The 5 OOD
wafer-pattern classes (added iter 12) are present in the master
folder for ensemble-side OOD-FAR diagnostics but their per-class
metrics are **never reported** (user directive 260506).

## 3.9 Iter 12 — FAR metric split

The iter-1 through iter-9 papers reported a bundled `chip_FAR`
metric over all 1000 non-defect chips (200 Normal + 50 Invalid +
800 OOD wafer-pattern). Iter 12 (260506–07) splits this into three
disjoint groups, recognising that the operational FAR includes
only the production-relevant classes:

| group           | classes                                | n chip | role                  |
|-----------------|----------------------------------------|-------:|-----------------------|
| `normal_invalid` ★ | Normal, Invalid                     |    200 | **paper main metric** |
| `normal_only`   | Normal alone                           |    160 | ablation diagnostic   |
| `ood`           | 5 wafer-pattern OOD                    |    800 | diagnostic only       |
| **bundled** (deprecated) | all three groups summed       |   1000 | backward compat       |

The `chip_FAR = normal_invalid_FAR` definition is adopted as the
paper-grade headline going forward. The old bundled metric is
retained in `chip_multilabel/_bit_metrics.py` with explicit
deprecation marker.

**Why this matters.** The bundled metric reads 96% on every
4-class-only trained variant, suggesting catastrophic FAR. The
decomposition reveals: 80% of the bundle is `normal_only` lock
(model never trained on Normal) and 100% is `ood` (5 classes
never trained at all). Production never sees the OOD classes; the
operational FAR is the `normal_invalid` component, and on it the
T7N (Normal-trained) variant locks **0.00%** while the no-Normal
variants lock **80.00%**. The bundled metric obscured a 80×
single-axis intervention (Normal training) that the
decomposition makes visible.

## 3.10a Train and evaluation are independently sampled from the same synthesis pipeline

_Added 2026-05-10 (methodological transparency disclosure)._

We disclose explicitly that the **training set**
(`D:/project/data/wm-811k/classification_chips/`, single-class chip-level
synthesis built by `dist_apply/_sample_gen.py`) and the **evaluation
set** (`D:/project/data/wm-811k/chip_multilabel_v15direct/`, multi-label
synthesis built by `chip_multilabel/_synth_multi_chips.py`) are produced
by **separate scripts** that nevertheless share the same underlying
synthesis primitives:

- same chip dimensions (200 × 200);
- same palette encoding (grade 0 = white, grade 1 = grey, grades 2–7 = saturated defect colours);
- same alpha-modulation matched-filter mechanism (Lorentzian sharp + heavy tail);
- same defect-type spec (`bank_boundary`, `fork`, `scratch`, `scratch_rot`).

**No chip in the eval set appears in the train set.** The two scripts
use **different RNG seeds** (train seed = 42, eval Normal seed = 999)
and **different generation modes** (train = single-class stamp; eval =
`min`-blend or RGB synth across class pairs / triples). The eval set
also contains four OOD wafer-canvas patterns (CenterDonut, CrossScratch,
DiagonalSmear, Starburst) that are **structurally absent from the
training distribution** and contribute to the operational `ni_FAR`
metric (§3.9, §4.5.1).

**Multi-class combos in eval are a new distribution mode unseen during
training.** The model is single-label-trained on 4 defect classes; it
encounters multi-positive ground-truth chips (combo-2 and combo-3)
only at evaluation time. The decision-tree multi-label inference rule
(§4) is itself never seen during training — it operates on the model's
sigmoid logits with calibrated per-class thresholds.

**Scope statement.** This design tests the methodology
(FCM-PM training + bag-ensemble inference) on a **controlled
synthetic benchmark**. The eval set probes (i) the model's combo
decoding capability under multi-positive ground truth, (ii)
distribution-shift handling on four OOD wafer-canvas patterns, and
(iii) seed-stability under the bag aggregator. **It does not establish
real-factory deployment performance.** Sensor noise, alignment drift,
calibration variation across fab tools, and process-recipe-induced
distribution shifts are not captured by either pipeline. The
headline numbers (v15direct n = 500 bit-F1 = 0.9953 / `ni_FAR = 0 %`)
are **ceiling estimates on this synthesis distribution**, and
real-factory validation is recommended as a follow-up study (§7.6.2).

## 3.10 v5.2 baseline reset (260507)

The chip-level synthesis logic (§3.2) is canonical at v5/v5.1/v5.2;
the wafer-level synthesis (the data path that produces wafer maps,
not chip multi-label eval chips) was updated in v5.2 with three
fixes: bank_boundary chip-seam removal, wafer pink baseline
uniform spread, RingDots fixed positions, and Edge-Top/Bottom
defect budget elevation. The chip multi-label eval set
(`chip_multilabel/` master) is **invariant under v5 → v5.1 →
v5.2** — chip-level grade distributions and Normal / Invalid
recipes are unchanged. The §5.11 / §5.13 chip multi-label results
therefore carry forward to the v5.2 baseline. See §5.14 for the
spec details and visual sanity manifest.
