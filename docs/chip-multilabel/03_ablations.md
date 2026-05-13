# 03 — Ablations: what worked / what didn't

All deltas measured on the same 2200-chip 11-class eval set. Pull
quotes use 4-decimal numbers from the canonical sources.

## ★★★ 2026-05-12 ablation update — Backbone scaling (iter 88-89, 12 backbones)

Companion to the iter21 throughput axis (Phase 87 v2): the iter88 batch
explored **8 candidate backbones across 4 families** + iter89 retuned
Swin-Base hyperparameters. Full per-cell numbers in `05_backbone_landscape.md`;
the ablation-relevant deltas here are:

### What did NOT work — scaling up Swin V1 from Base → Large

| change                                  | before (iter77C Swin-Base 86.9 M) | after (iter88E Swin-Large 195.2 M) | finding                                                                       |
|-----------------------------------------|-----------------------------------|------------------------------------|-------------------------------------------------------------------------------|
| Swin-Base 86.9 M → Swin-Large 195.2 M   | bF1 0.9692 / **Total FAR 0.00%** | bF1 **0.9192** / Total FAR 0.12%   | **−0.0500 bF1 at 2.25× params** under unchanged recipe — pure capacity scaling loses |

The same recipe (T7 / g=3 / LS=0.20 / 8 ep / LR=1e-4) on the larger backbone
underperforms. Likely deeper-net pathology — needs longer warmup + smaller LR;
re-tune queued for iter95.

### What did NOT work — scaling up ConvNeXt V1 from Base → Large

| change                                                  | before (iter77A ConvNeXt-Base V1 87.6 M)   | after (iter88F ConvNeXt-Large V1 196.2 M) | finding                                                                       |
|---------------------------------------------------------|--------------------------------------------|--------------------------------------------|-------------------------------------------------------------------------------|
| ConvNeXt-Base V1 → ConvNeXt-Large V1                     | bF1 0.9830 / Total FAR 2.62%               | bF1 **0.8720 (safe I13 only)** / Total FAR 10.60% | **−0.1110 bF1 + +7.98 pp FAR** — Large variant's I3/I7 cells score bF1 0.9919-0.9931 but at **100% Total FAR** (unusable); only I13 is safely under 11% FAR |

ConvNeXt V1's "everything is foreground" prior gets stronger with depth.
**Smaller is better** on this 4-class small-data multi-label task for ConvNeXt V1.

### What did NOT work — recipe transfer to non-ConvNeXt families

| backbone                          | best safe cell bF1_4def         | best Total FAR | notes                                                                          |
|-----------------------------------|--------------------------------:|---------------:|--------------------------------------------------------------------------------|
| iter88D EfficientNetV2-L 117.75 M | 0.7695 (no safe cell ≤ 5% FAR)  | 79.05%         | All four cells unsafe; predicts defect on every input                          |
| iter88B TinyViT-21M-384 20.67 M   | 0.6906 (I10 safe)                | 0.00%          | Smallest backbone tested — low accuracy ceiling                                |
| iter88G ResNet-50 23.57 M         | 0.6861 (I7 unsafe @ 95.24% FAR) | —              | I10/I13 degenerate to empty-set predictions (bF1=0)                            |
| iter88A EfficientFormerV2-L 25.75 M | EVAL_FAIL (NaN logits)        | —              | Train converged val_acc=0.9877 but eval forward produces NaN                   |
| iter88C MobileNetV4-Conv-L          | (no checkpoint written)        | —              | Process died early; re-train queued                                            |
| iter88H ResNet-152 58.30 M          | TRAIN_ONLY (no eval yet)       | —              | Train done, eval pending                                                       |

**Verdict**: the ConvNeXtV2-tuned T7 recipe transfers **only within the Swin V1
family** at small data scale. Every other family (EfficientNet, ResNet,
EfficientFormer, TinyViT, MobileNet) needs per-family recipe re-tuning — single
LR=1e-4 + 8 epoch + CutMix-complement does not generalise. Per-family recipe
sweeps queued for iter95.

### What worked — Swin-Base CutMix-group + label-smoothing co-tune

Within the Swin-Base recipe, iter89 isolates two ablation knobs:

| change                                   | before                         | after                          | delta bF1_4def | Total FAR |
|------------------------------------------|--------------------------------|--------------------------------|---------------:|----------:|
| LS=0.50 g=3 (iter77C) → LS=0.30 g=3 (iter89_LR14_LS3_g3) | 0.9692 (I10)         | 0.9434 (I10)                  | **−0.0258**    | 0.00%     |
| LS=0.30 g=3 (iter89_LR14_LS3_g3) → LS=0.30 g=2 (iter89_LR14_LS3_g2) | 0.9434 (I10) | 0.9278 (I10)                  | **−0.0156**    | 0.00%     |
| LS=0.30 g=2 → LS=0.50 g=2 (iter89_LR14_LS5_g2) | 0.9278 (I10)           | TRAIN_ONLY (eval pending)      | —              | —         |

Both knobs are **monotonic** — higher LS and higher group count help — re-confirming
the Phase A1 finding (iter05) that **LS=0.50 is the sweet spot** for the CutMix-complement
recipe. The combination iter77C (LS=0.50 g=3) is **+0.0414 bF1 vs iter89_LR14_LS3_g2
(LS=0.30 g=2)** at identical Total FAR=0.00% — a clear single-iter co-tune win.

## ★★★ 2026-05-12 ablation update — Backbone throughput axis (iter 21, Phase 87 v2)

Fourth axis added to the paper ablation table (alongside loss / matching / decision-rule): **backbone choice** has both an accuracy and a cost dimension, and these dimensions **do not co-rank**.

### What worked — ConvNeXt V1 vs ConvNeXtV2 at same param count

| change                                              | before                          | after                                | finding                                                                                            |
|-----------------------------------------------------|---------------------------------|--------------------------------------|----------------------------------------------------------------------------------------------------|
| ConvNeXtV2 (87.7 M, GRN) → ConvNeXt V1 (87.6 M, no GRN) | b=1 27 ms / 37 chip/s + bF1 0.9654 + FAR 1.07% | b=64 **13.21 ms / 76 chip/s + bF1 0.9830** + FAR 2.62% | **2.05× throughput + +0.0176 bF1** at the same parameter count; **GRN absence enables normal batching scaling** |
| ConvNeXtV2 → Swin-Base 384 (86.9 M)                  | b=1 27 ms / 37 chip/s + FAR 1.07% | **b=1 21.08 ms / 47 chip/s + FAR 0.00%** | best single-chip latency **AND** strict-zero Total FAR (only backbone in sweep with FAR=0) |

**Verdict**: GRN is an accuracy-regularising architectural prior (ConvNeXtV2 wins on small-data smaller-bF1-gap evaluation against same-recipe peers in earlier iters), but it **costs throughput**. The right call depends on operational regime, not a single-axis "best".

### What did not work — EfficientV2-M at iter46E recipe

| change                          | before (ConvNeXtV2 iter46E)     | after (EfficientV2-M iter46E)  | finding                                                                          |
|---------------------------------|---------------------------------|---------------------------------|----------------------------------------------------------------------------------|
| ConvNeXtV2 → EfficientV2-M      | bF1 0.9654 / 37 chip/s          | **bF1 FAIL / 158 chip/s**       | **3.77× peak GPU throughput** but recipe transfer fails — EffV2 needs longer schedule + warmup, not single-LR 1e-4 8 ep |

**Verdict**: hardware speed alone doesn't make a backbone deployable; **recipe-backbone coupling is real**. EfficientV2-M is queued for a separate recipe-tuning iter before re-evaluation as production candidate.

### Negative scaling — ConvNeXtV2 GRN batching quirk

| batch | ms/chip | chip/s | scaling vs b=1 |
|------:|--------:|-------:|---------------:|
| 1     | 26.92   | **37** | 1.00           |
| 4     | 32.83   | 30     | **0.82** ⛔     |
| 8     | 35.20   | 28     | **0.76** ⛔     |
| 32    | 28.95   | 35     | 0.93           |
| 64    | 38.18   | 26     | **0.70** ⛔     |

ConvNeXtV2 is the **only backbone in the sweep** where batching makes things slower. Mechanism: GRN's per-channel mean of L2-norms broadcasts a batch-wise reduction at every block (Woo et al. 2023 arXiv:2301.00808), which cuDNN does not vectorise across batch elements as cleanly as a plain channel-wise norm. **ConvNeXt V1 (same param count, no GRN) shows healthy 1.85× scaling b=1 → b=64**, confirming the regression is GRN-specific.

**Paper claim**: When citing inference cost, ConvNeXtV2 must be reported at b=1 only; b=64 numbers exist but are dominated by every other tested backbone.

Source: `iters/iter_21_backbone_throughput_paper3.md` + `tables/backbone_throughput.csv` (24 rows). Raw measurement `_phase87_precise_speed.py` torch.cuda.Event with 20 warm-up + 100 iter on isolated A6000.

## ★★★ 2026-05-12 ablation update — Total FAR re-score, rect-flag no-op, seed-fragility

Three findings from Phase 83 / 85 (iter18) + iter79 / iter80 (iter19) +
260512 trainer patch (iter20):

### Total FAR vs ni_FAR (Phase 83 / 85)

| change vs baseline                            | from cell / metric            | to metric             | finding                                                                                  |
|-----------------------------------------------|-------------------------------|-----------------------|------------------------------------------------------------------------------------------|
| ni_FAR (4 Normal sources) → Total FAR (+ OOD) | every "ni_FAR=0%" SOTA        | Total FAR             | **most prior SOTA cells silently held Total FAR 12–48%** on OOD wafer-patterns           |
| iter69 ep=12 KD                               | bF1=0.9941 / ni_FAR=0%        | Total FAR=36.67%      | masked OOD blow-up — paper revocation                                                    |
| iter50B paper KD                              | bF1=0.9872 / ni_FAR=0%        | Total FAR=12.86%      | KD trades NI-safety for OOD over-fire — paper revocation                                  |
| iter46E vanilla ★                             | bF1=0.9654                    | **Total FAR=1.07%**  | only single model with bF1≥0.96 AND Total FAR≤5% — **new single-model paper headline**    |
| 4-bag ensemble quorum k=2 → k=3              | bF1=0.9962 / Total FAR=2.86%  | bF1=**0.9909 / 0.00%** | stricter quorum buys true zero for −0.0053 bF1 — **new ensemble paper headline**          |

**Verdict**: `Total FAR = (NI + OOD) / total` is the only honest FAR metric for
a v15direct eval that intentionally includes OOD wafer-patterns. All paper
tables must move to `(bit_F1, Total FAR)` pairs.

### `--cutmix-rect` is a no-op under `cutmix_mode=complement`

| change                          | observation                                                              | source                              |
|---------------------------------|--------------------------------------------------------------------------|-------------------------------------|
| iter46E (`rect=0.3`)            | bit-identical macro_f1 to iter42F at seeds 7/13/21                       | iter80 log, 12 cells (iter19)        |
| iter42F (`rect=0.5`)            | bit-identical macro_f1 to iter46E at all 3 shared seeds                  | iter80 log                          |
| Δ(42F vs 46E) at all 3 seeds    | **0.0** (0.8142 / 0.7998 / 0.7750 each)                                  | iter80                               |

**Verdict**: the rect-region logic only fires under `cutmix_mode=single`. Folder
names like `iter46E_g3LS050_rect03` should not be cited as "rect=0.3 variant" —
they are the **same model family** as their rect=0.5 sibling. iter46E's real
recipe axes are `n-groups=3 / complete-label-scale=0.5 / pair=masked /
pair-fill=corner / LS=0.20 / mode=complement / p=0.25`.

### iter26H recipe is seed-fragile but ensemble-rescuable

| seed | bit_F1 (v14) | ni_FAR pass     |
|-----:|-------------:|:----------------|
|    1 | (in pass set)| ✓ (5/8 pass)    |
|    7 | **0.9961** (peak) | ✓          |
|   13 | (in pass set)| ✓               |
|   19 | (in pass set)| ✓               |
|   21 | (in pass set)| ✓               |
|   22 | (in pass set)| ✓               |
|   42 | (catastrophic)| **FAIL — ni_FAR=100%** ⛔ |
|  100 | (in pass set)| ✓               |

| ensemble                  | composition                       | bit_F1 | ni_FAR  |
|---------------------------|-----------------------------------|-------:|--------:|
| same-recipe 3-bag (vanilla) | iter79 `{s7 + s13 + s21}` I10 k=2 | 0.9955 | **0.00%** |

**Verdict**: vanilla single-seed iter26H mean 0.9695 ± 0.033 with 1 catastrophic
seed in 8 → 3-bag majority restores 0.9955 / 0% (NI-only; Total FAR pending
iter83). Same-recipe seed-diversity is sufficient — no need for cross-recipe.
Note Total FAR not measured for any iter79 cell yet; iter83 deferred.

### 260512 trainer patch — `--cutmix-other-label`

| flag                       | default | effect when default                | effect when > 0                                                                                       |
|----------------------------|--------:|------------------------------------|-------------------------------------------------------------------------------------------------------|
| `--cutmix-other-label`     |     0.0 | identical to pre-patch behavior    | off-class bits on mix chip get value `other_label` (e.g., 0.1 = soft uncertain) instead of hard zero |

Motivation: Phase 84b iter46E prob-dist diagnostic shows TRAIN defect own-prob
= 0.84–0.92 but EVAL OOD max-prob = ~0.55 (right at threshold). Hypothesis:
hard-zero off-class labels on mix chips teach an over-confident absent-class
prior that fails to generalize to OOD with mid-range max-prob.

**Verdict**: hypothesis-driven patch, **no result yet**. iter83 5-cell sweep
∈ {0, 0.05, 0.10, 0.15, 0.20} at iter46E base recipe is the planned next
ablation, gated by `bit_F1 within −0.005 of 83A AND Total FAR ≤ 5%`. If even
0.05 violates, the patch is paper-negative.

## Inference-side ablations (fixed model = T0)

| change vs baseline               | from cell        | to cell          | Δ macro_f1 | Δ top1_11 | verdict        |
|----------------------------------|------------------|------------------|-----------:|----------:|----------------|
| argmax → F1-max thresholds       | T0__I0 (0.7302)  | T0__I1 (0.8444)  |    +0.1142 |   +0.1852 | huge win       |
| F1-max → top-K=2 alone           | T0__I1 (0.8444)  | T0__I2 (0.7673)  |    -0.0771 |   -0.0585 | regression     |
| F1-max + top-K rescue (I1+I2)    | T0__I1 (0.8444)  | T0__I3 (0.8466)  |    +0.0022 |   -0.0307 | tiny win on F1 |
| I3 + temperature scaling         | T0__I3 (0.8466)  | T0__I4 (0.8466)  |    +0.0000 |   +0.0000 | no-op on F1    |
| I3 + TTA (rotation 4×)           | T0__I3 (0.8466)  | T0__I5 (0.8287)  |    -0.0179 |   -0.0006 | **DISALLOWED** |
| F1-max + min-floor 0.30          | T0__I3 (0.8466)  | T0__I6 (0.8177)  |    -0.0289 |   -0.0136 | regression     |
| F1-max + step-search Δ=0.02      | T0__I3 (0.8466)  | T0__I7 (0.8485)  |    +0.0019 |   +0.0193 | small win      |
| I3 + top-K=1 fallback            | T0__I3 (0.8466)  | T0__I8 (0.8456)  |    -0.0010 |   +0.0000 | flat           |
| I1 + temperature only            | T0__I1 (0.8444)  | T0__I9 (0.7741)  |    -0.0703 |   -0.0983 | regression     |
| I7 + entropy Normal gate (I10)   | T0__I7 (0.8485)  | T0__I10 (0.8542) |    +0.0057 |   +0.0307 | win, durable   |

_Source: outputs/stage1_260505_162842, _165400, _170827 results_matrix.parquet._

### Verdicts

- **F1-max thresholds (I1) is the single biggest single inference change**
  (+0.1142 macro-F1 over argmax). Most of the climb away from baseline is
  this one trick.
- **Step-search (I7)** is a clean micro-improvement on I3 with no extra
  pipeline complexity (Δ=0.02 grid).
- **Entropy Normal gate (I10)** is the only inference idea that *survives*
  retraining — every other variant either ties or trails I7 once the
  model is fine-tuned.
- **TTA (I5) is permanently disallowed** because rotation conflates
  scratch / scratch_rot. Even though it sometimes nudges precision, the
  semantic damage is unacceptable.
- **Temperature scaling (I4, I9)** does not help macro-F1 because the
  threshold sweep already absorbs whatever calibration shift T provides;
  it does help ECE (0.0778 → 0.0129 on I4), so keep it for any
  probability-honest downstream.
- **Min-floor 0.30 (I6)** hurts because the val-tuned fork threshold is
  ~0.12 — clipping it to 0.30 throws away most of fork's recall.

## Training-side ablations (best inference = I10)

| variant | loss            | best cell    | macro_f1 | top1_11 | Δ vs T0__I10 | verdict            |
|---------|-----------------|--------------|---------:|--------:|-------------:|--------------------|
| T0      | none (frozen)   | T0__I10      |   0.8542 |  0.6517 |       (ref)  | baseline           |
| T1      | CE + LS 0.10    | T1__I10      |   0.8634 |  0.7006 |      +0.0092 | win                |
| T4      | ASL             | T4__I10      |   0.7759 |  0.5830 |      -0.0783 | regression         |
| T5      | BCE             | T5__I10      |   0.7589 |  0.5432 |      -0.0953 | regression         |
| T6      | BCE → ASL       | T6__I10      |   0.8193 |  0.6256 |      -0.0349 | regression         |

_Source: outputs/stage2_260505_170121/results_matrix.parquet for T1/T4/T5/T6 ×
I0..I9 grid; outputs/stage1_260505_{173649,173829,173955,174123}/results_matrix.parquet
for the post-hoc I10 inference rows._

### Verdicts

- **T1 (CE + LS 0.10) is the only training intervention that helped**
  on the multi-label benchmark. The single-label CE pretrain provides a
  decent base and label smoothing softens the softmax peak so the
  runner-up class still has a usable score.
- **T4 (ASL), T5 (BCE), T6 (BCE→ASL) all regress** despite being the
  "obvious" multi-label choices. Hypothesis: these losses change the
  distribution of activations enough that the F1-max thresholds tuned
  on val don't transfer cleanly. Specifically T4 and T5 over-suppress
  bank_boundary, dropping its F1 from ~0.96 to ~0.85.
- T6 (BCE→ASL) is the worst hybrid: BCE collapses the softmax structure,
  then ASL doesn't have time to rebuild useful asymmetry in 4 epochs.
- T1 is also the cheapest (~330s on RTX 4090).

## LS sweep (iter 5, T1 only)

| LS    | best inference | macro_f1 | top1_11 | Δ vs LS=0.10 best (T1__I10 = 0.8634) |
|------:|----------------|---------:|--------:|-------------------------------------:|
|  0.05 | I7             |   0.7964 |  0.5591 |                              -0.0670 |
|  0.10 | I3             |   0.8363 |  0.6261 |                              -0.0271 |
|  0.15 | I3             |   0.8961 |  0.7517 |                              +0.0327 |
|  0.20 | **I7**         | **0.9268** | **0.8449** |                          **+0.0634** |
|  0.25 | I3             |   0.8663 |  0.6989 |                              +0.0029 |
|  0.30 | I3             |   0.8185 |  0.6466 |                              -0.0449 |

_Source: outputs/phase_a_260505_175105/sweep_log.csv,
outputs/phase_a_260505_182044/sweep_log.csv._

### Verdicts

- **LS=0.20 is the sweet spot** — too little smoothing leaves the
  single-label collapse intact; too much smoothing erases informative
  margin between classes.
- The curve is **non-monotonic and sharp**: 0.20 → 0.9268 vs 0.15 →
  0.8961 vs 0.25 → 0.8663. ±0.05 around the optimum costs ~0.03 macro-F1.
- The optimum cell is **`T1_LS20 + I7`**, not + I10. The entropy gate
  (I10) helps frozen / mildly-trained models because their Normal logit
  is poor; once LS=0.20 has trained the model into a more
  well-calibrated multi-label state, the explicit Normal gate becomes
  redundant and slightly hurts (0.9268 → 0.8841).

## Phase F — anomaly-detection BKM transfer (iter 7, negative)

Phase F tested two structural BKMs imported from anomaly-detection
literature on top of the Phase A winner (T1_LS20_ep8). Both regressed.

| variant | recipe                                            | best cell | macro_f1 | top1_11 | Δ vs T1_LS20__I7 (0.9268) | verdict     |
|---------|---------------------------------------------------|-----------|---------:|--------:|--------------------------:|-------------|
| F1      | LR warmup 2ep (start_factor=0.05, eta_min=1e-6)   | F1__I10   |   0.8181 |  0.5540 |                  **−0.1087** | regression  |
| F2      | EMA decay=0.95 + dynamic decay warmup             | F2__I10   |   0.8377 |  0.6602 |                  **−0.0891** | regression  |

_Source: outputs/stage1_260505_192541/results_matrix.parquet (F1),
outputs/stage1_260505_194014/results_matrix.parquet (F2)._

### Verdicts

- **F1 (warmup 2ep) regresses −0.109** because epoch-1 LR ≈ 5e-6 is far
  below the working LR for an 8-epoch CE+LS finetune on a
  TAPT-initialized backbone. Warmup helps when the model is far from a
  reasonable basin (cold-start, large LR, anomaly binary chart) — none
  of which apply here. The warmup eats two epochs that the LS=0.20
  recipe was using to actually converge.
- **F2 (EMA 0.95) regresses −0.089** because EMA's bias correction needs
  many effective steps before the running average becomes useful. With
  ~12 effective steps (8 epochs × small loader), EMA is averaging
  partially-trained weights into the final checkpoint and damping the
  late-epoch sharpening that LS=0.20 was relying on.
- **Paper-worthy negative result**: anomaly-detection BKMs (warmup, EMA)
  do not transfer cleanly to small-data multi-label chip classification
  with a TAPT init. The structural assumption gap (cold-start vs warm
  TAPT, many-step vs few-step training) breaks the transfer.

## Phase F band-aid — I11 pair-aware threshold (iter 7, rejected)

I11 is a pure-inference variant that adds a pair-aware additional
threshold: declare bank_boundary+scratch_rot together when both class
logits exceed an extra co-occurrence threshold. Tested on T1_LS20__I7
without retraining.

| metric             | T1_LS20__I7 | T1_LS20__I11 | Δ        |
|--------------------|------------:|-------------:|---------:|
| macro_f1           |      0.9268 |       0.9199 | −0.0069  |
| top1_11            |      0.8449 |       0.8432 | −0.0017  |
| bb+sr combo recall |      ~0.325 |       ~0.481 | +25 chips |
| bb+fork FP         |    baseline |       +31 FP | over-trigger |

_Source: outputs/stage1_260505_194443/results_matrix.parquet._

**Verdict**: rejected. The bb+sr recall gain is real (+25 chips on the
specific combo), but the pair-aware threshold also over-triggers on
bb+fork (31 false positives) because the bb logit is already at the
co-occurrence cutoff for many bb-only chips. Net macro-F1 −0.007. The
right fix is at the training stage (CutMix, see Phase F T7), not an
inference band-aid.

## Phase F — T7 atomic decomposition (iter 7)

Goal: separate the contributions of two simultaneous changes — switching
loss CE → BCE and adding CutMix p=0.5 — by running each step in
isolation. Train recipe held at LS=0.20, LR=1e-4, ep=8.

| step  | loss | cutmix-p | best cell | macro_f1 | top1_11 | Δ from prev | Δ vs T1_LS20 |
|-------|------|---------:|-----------|---------:|--------:|------------:|-------------:|
| T1    | CE   | 0.0      | T1__I7    |   0.9268 |  0.8449 |       (ref) |       (ref)  |
| T7a   | BCE  | 0.0      | T7a__I3   |   0.8577 |  0.5534 |     **−0.0691** |     **−0.0691**  |
| T7c ★ | BCE  | **0.5**  | T7c__I10  |   **0.9271** |  0.8307 |     **+0.0694** |     **+0.0003**  |

_Source: outputs/phase_a_260505_175105/sweep_log.csv (T1),
outputs/stage1_260505_200523/results_matrix.parquet (T7a),
outputs/stage1_260505_195730/results_matrix.parquet (T7c)._

### Verdicts

- **CE → BCE alone (T1 → T7a) costs −0.0691** macro-F1, confirming the
  iter 4 finding that BCE drops the useful softmax shape. T7a's best cell
  is I3 (not I7), and top1_11 drops to 0.5534 — the model has no Normal
  gate signal because the softmax-style entropy doesn't apply.
- **+ CutMix p=0.5 (T7a → T7c) recovers +0.0694**, almost exactly
  cancelling the BCE penalty in macro-F1 terms. CutMix on multi-hot
  targets directly teaches the model that bb+sr can co-occur in pixel
  space, repairing the precision-recall trade that BCE alone breaks.
- **Net (T1 → T7c) is +0.0003 on macro-F1 — statistically tied** — but
  the operational profile flips: bb+sr combo recall **0.32 → 0.96**
  (+0.63 absolute), `scratch_rot` per-class F1 reaches 1.0000, ECE_post
  drops 4× (0.1788 → 0.0446). The trade-off is paid by bank_boundary F1
  (0.8974 → 0.8885) and scratch F1 (0.9725 → 0.9554) — both still
  excellent. The macro-F1 tie hides a large operational improvement.

## Phase F — CutMix-p sweep (iter 7, BCE+LS=0.20)

Same recipe as T7a (BCE+LS=0.20, ep=8, LR=1e-4) varying only `cutmix_p`.

| cutmix-p | best cell | macro_f1 | top1_11 | bb+sr recall | verdict          |
|---------:|-----------|---------:|--------:|-------------:|------------------|
| 0.0      | T7a__I3   |   0.8577 |  0.5534 | (low)        | BCE-only floor   |
| 0.3      | T7b__I10  |   0.8626 |  0.5511 | 0.7312       | partial recovery |
| **0.5 ★**| T7c__I10  | **0.9271** | **0.8307** | **0.9562** | **peak**         |
| 0.7      | T7d__I10  |   0.9038 |  0.7432 | (high)       | over-mixing      |

_Source: outputs/stage1_260505_200523, _203340, _195730, _201706/results_matrix.parquet._

### Verdicts

- **cutmix_p=0.5 is the sharp peak** — both lower (0.0/0.3) and higher
  (0.7) values lose ~0.03–0.07 macro-F1. This mirrors the LS sweep
  shape from iter 5: a non-monotonic optimum surrounded by penalty.
- The bb+sr recall axis is monotonically increasing in cutmix_p up to
  0.5 (then we don't have data above 0.5 except T7d=0.7 macro_f1=0.9038),
  but **macro-F1 turns over at 0.7** because too much mixing breaks
  single-class chips: the model starts hallucinating combos when only
  one defect is present.
- **CutMix is the first training-side intervention since LS=0.20 to
  produce a paper-headline gain** — not on the macro_f1 axis (tie) but
  on the bb+sr operational axis (+0.63 recall).

## LS sweep under BCE+CutMix (iter 8, T9 family)

Re-sweeping the LS axis on top of the iter-7 BCE+CutMix(p=0.5) recipe.
Recipe held: BCE, CutMix p=0.5, ep=8, LR=1e-4. Only `label_smoothing`
(and seed for T9g) varies.

| LS    | seed | run  | best cell  | macro_f1 | top1_11 | mAP    | ECE_post | Δ vs T7c=0.9271 |
|------:|-----:|------|------------|---------:|--------:|-------:|---------:|----------------:|
|  0.00 |   42 | T9c  | T9c__I10   |   0.8609 |  0.6443 | 0.8384 |   0.0114 |        −0.0662  |
|  0.05 |   42 | T9b  | T9b__I7    |   0.9449 |  0.8670 | 0.9378 |   0.0060 |        +0.0178  |
|  0.06 |   42 | T9f  | T9f__I3    |   0.9401 |  0.8648 | 0.9521 |   0.0088 |        +0.0130  |
| **0.07 ★**| 42 | T9d | T9d__I7  | **0.9705** | **0.9267** | **0.9864** | **0.0106** |    **+0.0434** |
|  0.07 |   43 | T9g  | T9g__I7    |   0.9408 |  0.8307 | 0.9468 |   0.0079 |        +0.0137  |
|  0.08 |   42 | T9e  | T9e__I3    |   0.8085 |  0.4449 | 0.8362 |   0.0425 |        −0.1186  |
|  0.10 |   42 | T9a  | T9a__I10   |   0.9364 |  0.8489 | 0.9451 |   0.0143 |        +0.0093  |

_Source: outputs/stage1_260505_{210059,210535,210932,211334,211752,212153,212557}/results_matrix.parquet._

### Verdicts

- **The CE-era LS=0.20 optimum does not transfer.** Under BCE+CutMix the
  optimum shifts low to **LS=0.07**. Diagnosis: BCE already softens
  hard targets via independent sigmoids, and CutMix interpolates
  multi-hot labels; LS=0.20 stacked on top over-softens.
- **The curve is non-smooth — knife-edge cliff at LS=0.08** (0.8085, a
  0.16-macro-F1 drop over a 0.01 step from LS=0.07's 0.9705 and a 0.13
  drop from LS=0.10's 0.9364). LS=0.05 / 0.06 / 0.10 all sit in a 0.94
  band; only 0.08 falls off. Hypothesis: a phase-transition where
  BCE+CutMix runner-up gradient signal collapses around 0.08 effective
  positive-target value.
- **Single-seed variance ±0.030 at the optimum.** T9d (LS=0.07,
  seed=42) = 0.9705; T9g (LS=0.07, seed=43) = 0.9408. The seed-driven
  spread is concentrated in **fork F1** (0.9448 vs 0.8149) — the
  diffuse longest-tail class is the variance carrier; bank_boundary /
  scratch / scratch_rot are stable across seeds.
- **Reporting policy**: T9d 0.9705 = "best observed", T9g 0.9408 =
  "realistic point estimate". Neither alone is paper-honest; both
  together with the variance caveat are.

## Negative axes — atomic-failed (iter 9, on top of T9 LS=0.07 recipe)

Three orthogonal axes probed on top of the iter-8 BCE+LS=0.07+CutMix(p=0.5)
recipe to test whether they can lift the realistic baseline (T9g=0.9408)
above seed noise. **All regress.**

| run  | axis change           | seed | best cell  | macro_f1 | top1_11 | Δ vs T9d (0.9705) | verdict   |
|------|-----------------------|-----:|------------|---------:|--------:|------------------:|-----------|
| T10a | drop_path 0.05        |   42 | T10a__I3   |   0.9160 |  0.7335 |          −0.0545  | regress   |
| T10b | drop_path 0.05        |   43 | T10b__I11  |   0.8918 |  0.7511 |          −0.0787  | regress   |
| T11a | cutmix-rect 0.25      |   42 | T11a__I7   |   0.8646 |  0.6551 |          −0.1059  | regress   |
| T12a | two-LR backbone/head  |   42 | T12a__I10  |   0.8862 |  0.6511 |          −0.0843  | regress   |

_Source: outputs/stage1_260505_{213423,213817,214222,214634}/results_matrix.parquet._

### Verdicts

- **drop_path 0.05 (T10a/b, n=2 seeds)**: −0.054 / −0.049. Two-seed
  consistency rules out a seed unluck. Same diagnosis as iter 7
  warmup/EMA: stochastic-depth regularizers need many effective steps
  before the expectation settles, and 8-epoch + small data + TAPT init
  doesn't provide them.
- **cutmix-rect 0.25 (T11a)**: −0.106. Confounded with a 0.5→0.25
  CutMix-ratio drop. T11a's 0.8646 is essentially identical to iter 7's
  T7b (cutmix p=0.3) = 0.8626, so the **rect-vs-square mask shape
  contributes ~zero signal** at chip-grid resolution; the regression is
  driven by the ratio drop, which is the iter-7 result re-confirmed.
  No separate iter-9 lesson from this run.
- **two-LR backbone/head (T12a)**: −0.084 macro-F1 *and* −0.27 top1_11.
  The combo-prediction collapse (top1_11 0.9267 → 0.6511) confirms the
  diagnosis: BCE+CutMix asks for sharper independent-sigmoid
  discrimination per class, which requires the backbone to update; a
  lowered backbone LR starves that update. Two-LR is again a long-
  training-regime BKM that doesn't transfer to 8-epoch budgets.

### The growing structural-mismatch catalogue

iter 7 atomic-failed warmup and EMA on the same diagnosis. Iter 9 adds
drop_path, two-LR, and (implicitly) the CutMix ratio drop. The pattern:
**any BKM that needs many effective gradient steps to stabilize loses**
in this 8-epoch + small-data + TAPT-init regime.

## Things that didn't work — short list

1. **TTA (I5)** — rotation breaks scratch / scratch_rot.
2. **ASL (T4)** — over-suppresses bank_boundary.
3. **BCE (T5)** — drops the softmax shape that was actually useful.
4. **BCE → ASL (T6)** — neither phase converges far enough in 4 epochs.
5. **Min-threshold floor (I6)** — fork needs a low threshold; floor kills it.
6. **Temperature alone (I9)** — without rescue, top-K combo recovery drops.
7. **LR warmup 2ep (F1, iter 7)** — −0.109 macro-F1 on TAPT-init small data.
8. **EMA 0.95 (F2, iter 7)** — −0.089 macro-F1; not enough effective steps.
9. **I11 pair-aware threshold (iter 7)** — −0.007 macro-F1 net; bb+sr +25 chips offset by bb+fork +31 FP.
10. **CutMix p=0.7 (T7d, iter 7)** — over-mixes and starts hallucinating combos; loses 0.023 vs p=0.5.
11. **LS=0.08 under BCE+CutMix (T9e, iter 8)** — knife-edge cliff −0.119 vs T7c; LS=0.07 (0.9705) and LS=0.10 (0.9364) are both fine, only 0.08 falls off.
12. **drop_path 0.05 (T10a/b, iter 9, n=2 seeds)** — −0.054 / −0.049; long-training-regime regularizer doesn't transfer to 8-epoch budgets.
13. **cutmix-rect 0.25 (T11a, iter 9)** — −0.106; rect mask shape carries no signal at chip-grid resolution; regression driven by 0.5→0.25 ratio drop (= iter-7 result re-confirmed).
14. **two-LR backbone/head (T12a, iter 9)** — −0.084 macro-F1, −0.27 top1_11; starves backbone of BCE+CutMix-driven updates.

## Iter 12 — v19zpp 21-class master ablation (no Normal training)

8-variant matrix on v19zpp tier chip data + master 21-class eval (4 single +
6 2-combo + 4 3-combo + Normal + Invalid + 5 OOD), all with `--no-normal`.

| variant   | CF1    | F1_fk  | bit_FAR | chip_FAR | verdict                       |
|-----------|-------:|-------:|--------:|---------:|-------------------------------|
| T7 ★      | 0.8490 | 0.5248 |  24.90% |  96.00%  | best single (LS=0.20+CutMix)  |
| T5        | 0.8349 | 0.5236 |  26.62% |  96.00%  |                               |
| T9        | 0.8258 | 0.5209 |  25.35% |  96.00%  |                               |
| T0        | 0.7645 | 0.5453 |  28.65% |  96.00%  | pure CE — fork prob 평탄      |
| T4        | 0.7642 | 0.5185 |  28.68% |  96.00%  | ASL γ=4 over-aggressive       |
| T3        | 0.7604 | 0.5240 |  48.60% |  96.00%  | Focal — fork over-fire        |
| T1        | 0.7403 | 0.5601 |  31.47% |  96.00%  | CE+LS=0.10 — multi-label 부적합|
| T6        | 0.6531 | 0.5403 |  36.28% |  96.00%  | BCE→ASL — worst               |

★ **모든 8 variant chip_FAR = 96.00%** (Normal training OFF 의 본질적 한계).
ni_chip_FAR 80% (Normal 200 mis-fire) + ood_chip_FAR 100% (OOD 800 mis-fire)
가 합쳐서 96% bundle. **paper finding**: bundled `chip_FAR` 단일 metric 폐기 →
split metric 필수 (`normal_invalid_chip_FAR` + `ood_chip_FAR` 분리).

_Source: outputs/T*_v19zpp_seed42_*/eval_I3/bit_metrics_split.json (8 runs),
notes.md `## iter 12 v19z++ on stable master` section._

## Iter 13 — Normal training as the single FAR lever (★ paper main)

**Cycle A** Normal training ON 의 단일 효과 측정 (T7N single):

| metric | T7-no-Normal (v19zpp) | T7N (with Normal) | Δ                |
|---|---:|---:|---:|
| CF1                 | 0.8490 | 0.9042 | **+0.0552**        |
| F1_fork             | 0.5248 | 0.7796 | **+0.2548**        |
| ni_chip_FAR         | 80.00% | 0.00%  | **-80pp** ★        |
| ood_chip_FAR        | 100.00%| 16.38% | **-83.62pp** ★     |
| F1_sc               | 0.9066 | 0.8676 | -0.0390 (trade-off)|

★ Normal training 단일 lever 가 chip_FAR 96% → 13.1% 단독 해결. iter 10
finding 재확인 + paper main result.

**Cycle A logit-avg ensembles** (9 cells):

| ensemble | weights | CF1 | F1_fork | ni_FAR | verdict |
|---|---|---:|---:|---:|---|
| **T7N+T5** ★ | 70:30 | **0.9083** | 0.7656 | 0.50% | overall winner |
| T7N+T7  | 60:40 | 0.9043 | 0.6988 | 0.00% | sc/sr ceiling |
| T7N single | n/a | 0.9042 | 0.7796 | 0.00% | base |
| T7N+T5  | 60:40 | 0.9018 | 0.7389 | 2.00% | |
| T7N+T9  | 60:40 | 0.9001 | 0.7281 | 13.00% | |

★ T7N anchor (≥60% weight) 가 ni_FAR lock-in. T5 minority weight 가 sc 에서
lift. iter 10 H ensemble winner mechanism (complementary diversity)
재현. logit-avg ensemble = **single 모델 + threshold tuning 의 한계 깸**.

**Cycle B** CutMix variant grid (T7N base):

| variant | CF1 | F1_fk | ni_FAR | ood_FAR | verdict |
|---|---:|---:|---:|---:|---|
| **random_rect** ★ | **0.9188** | 0.8436 | 20.00% | 0.94% | Cycle B winner |
| no_cutmix | 0.9162 | 0.8324 | 20.00% | 14.69% | CutMix 자체 marginal |
| grid50 | 0.8967 | 0.7635 | 20.00% | 0.31% | |
| grid25 | 0.8849 | 0.7450 | 20.00% | 3.12% | |
| grid12 | 0.8596 | 0.7778 | 20.00% | 12.03% | small patch 약함 |
| scattered | 0.8423 | 0.6912 | 20.00% | 23.44% | worst — multi-patch HURT |

★ Cycle B 의 모든 cell ni_FAR 20% lock — Cycle A T7N single 0% 보다 후퇴
(CutMix variant 가 Normal 신호 약화). random_rect = Cycle B winner CF1
0.9188 > Cycle A 0.9042 + 0.0146.

_Source: outputs/T7_T7_with_normal_v19zpp_seed42_v2_260507_002217 (Cycle A),
outputs/T7_T7N_*_seed42_260507_07*-08* (Cycle B 6 cells)._

## Iter 14 — v20 chip data fork sigma raised (atomic chip-data version)

fork sigma 1.0~1.5 → 1.8~2.5 (두께 ↑) 단일 변경:

| metric | Cycle B baseline | T7N v20 | Δ |
|---|---:|---:|---:|
| CF1 | 0.9188 | 0.9226 | +0.0038 |
| F1_fork | 0.8436 | 0.8591 | +0.0155 ✓ |
| F1_sc | 0.8658 | 0.8658 | 0 |
| F1_sr | 0.9937 | 0.9937 | 0 |
| **ni_chip_FAR** | 20.00% | **0.00%** | **-20pp** ★ |
| ood_chip_FAR | 0.94% | 0.94% | 0 |

★ fork single recall **1.0000** (이전 weak recall 깨짐). `fork+scratch_rot`
recall **0.625 → 0.7188** (+9.4pp partial fix). ni_chip_FAR 20% 도 같이
0% 로 복구 (chip data 신선화 효과).

**잔존 약점**: `fork+scratch_rot+ood_CrossScratch` 0.5687 — sr+CrossScratch
overlap 의 본질적 어려움 (둘 다 회전 패턴, augment / loss 변경 필요).

_Source: outputs/T7_T7N_v20_seed42_260507_063032/eval_I3/bit_metrics_split.json._

## Iter 15 — paper-style 4-class only ablation (Normal OFF, post-v5 chip data)

iter 11 의 paper figure 재공급 — 7 cell LS sweep + 2 alternative loss + P0
baseline. **모든 cell `--no-normal`** (Normal training OFF, 4-class only).

### LS sweep on T7 (BCE+LS, no CutMix, no Normal)

| LS | CF1 | F1_fk | ni_FAR | ood_FAR | verdict |
|---:|---:|---:|---:|---:|---|
| 0.025 | 0.8890 | 0.8308 | 45.00% | 5.94% | |
| **0.05** ★ | **0.9088** | 0.8351 | 36.00% | 5.94% | **paper baseline winner** |
| 0.075 | 0.8976 | 0.8679 | 26.50% | 3.59% | F1_fk peak |
| 0.10 | 0.8784 | 0.8806 | 31.50% | 15.47% | F1_fk peak (LS=0.10) |
| 0.15 | 0.8643 | 0.8159 | 22.00% | 9.69% | |
| 0.20 | 0.8648 | 0.8145 | 20.50% | 25.16% | |
| 0.25 | 0.8625 | 0.8465 | 29.50% | 23.75% | |

★ **LS=0.05 sweet spot** — iter 8 (T9 LS sweep on cutmix-base) 의 LS=0.07
peak 과 일관 신호. 다만 모든 cell ni_FAR ≥ 20% (Normal training 없으면
real-env Normal 잡기 불가).

### Alternative loss

| variant | CF1 | F1_fk | ni_FAR | ood_FAR | verdict |
|---|---:|---:|---:|---:|---|
| T9 sigfocal | 0.8273 | 0.7169 | 46.50% | 5.16% | sigmoid_focal moderate |
| T3 Focal | 0.7768 | 0.5717 | **100%** | **100%** | Focal worst — re-confirms iter 11 |

★ **Focal 이 ni/ood FAR 모두 100%** — iter 11 finding 일관 (Focal+cutmix
P3=0.513 보다 더 나쁨). post-v5 chip data 에서도 Focal 은 fork over-fire 를
극단적으로 누르며 Normal/OOD prob 도 평탄화 → 모든 chip 어떤 signal 한 개라도
나오면 declare.

### P0 baseline (T5 BCE pure, no LS, no CutMix)

| metric | T5 P0 | 비고 |
|---|---:|---|
| CF1 | 0.8583 | BCE pure |
| F1_fk | 0.7756 | |
| ni_chip_FAR | 24.50% | Normal not learned |
| ood_chip_FAR | 1.25% | unexpected low |
| ood_overlay 2bit_recall | 0.3906 | weak combo |

★ **iter 15 conclusion**: 4-class only environment 에서 LS=0.05 가 paper
baseline winner (CF1 0.9088). 그러나 ni_FAR 36% 로 operational 부적합 →
**iter 13 Cycle A T7N (Normal training ON, ni_FAR 0.50%) 가 paper main
result**. iter 15 는 paper baseline counter-example 으로 사용.

_Source: outputs/T5_P0_pure_baseline_seed42_260507_094228 + 7 T7_P1A_LS*
+ T3_P1A_T3_focal + T9_P1A_T9_sigfocal (10 runs), all
eval_I3/bit_metrics_split.json._

## Updated "things that didn't work" list (iter 12-15)

15. **No-Normal training on master 21-class (iter 12)** — bundled chip_FAR
    96% catastrophic. Normal training 누락 = 8 variant 전부 동일 96% 실패.
    paper finding: split metric 필수 + Normal training 필수.
16. **CutMix scattered/grid12 patches (iter 13 Cycle B)** — multi-patch
    분산 mask 가 single random_rect 보다 약함. ood_chip_FAR 23.44%/12.03%
    spike. iter 12 Phase 4 의 scattered finding 일관.
17. **CutMix variant axis (iter 13 Cycle B)** — Cycle A T7N single (0% ni_FAR)
    의 우위를 깸 (모든 variant 20% ni_FAR). CutMix variant 자체가 Normal
    학습 신호 약화시킴 — Normal training 환경에선 CutMix 줄여야.
18. **Focal loss on post-v5 chip data (iter 15, T3)** — ni/ood FAR 100%
    동시 mis-fire. iter 11 finding 재확인.
19. **LS ≥ 0.10 under T7N+CutMix (iter 15 P1A)** — LS=0.05 sweet spot 위
    monotonic 하락. ood_FAR 또한 LS 강할수록 spike (LS=0.10 → 15.47%,
    LS=0.20 → 25.16%).

## Iter 21 findings (clean baseline, dual-eval no-leak protocol)

Eval = disjoint v14class (800) + v15direct (1000, +4 OOD wafer-canvas).
Source: `iters/iter_21_clean_baseline.md`,
`tables/iter21_paper_headline.csv`.

### Positive (worked)

20. **★ Complement CutMix g=2 LS=1.0 FCM-PM (E, 19C repeat)** — first single
    model to clear both eval gates simultaneously: v14 bit_F1=0.9913 ni_FAR=0.00%,
    v15direct bit_F1=0.9691 ni_FAR=3.75%. Per-class F1 ≥ 0.94 on all 4
    defect bits. Confirms iter 19B (single-seed) was not a fluke.
21. **Soft labels in CutMix paste regions are the N/I gate switch.** C
    (T7N + std CutMix LS=1.0) → ni_FAR=100% on both evals. D (same recipe,
    label-scale 0.5) → ni_FAR=1.25% v14 / 2.50% v15. Same train data,
    same Normal sentinel, only paste-region LS differs.
22. **Complement CutMix > std/grid CutMix on far-OOD.** v15direct
    wafer-canvas chips: std CutMix C bit_F1=0.8457, grid D 0.9252,
    complement E 0.9691. Complement structure (paired-bit constraint)
    inoculates the model against unseen patterns better than dense paste
    grids.

### Negative (didn't work / collapsed)

23. **T5 baseline (no-Normal, no-CutMix) is misleadingly strong on v14.**
    bit_F1=0.9745 — but ni_FAR=100% (every Normal/Invalid fires defect).
    v15 collapses to 0.7872. Single-label-collapse signature; reaffirms
    Normal-training necessity from iter 10 / 13.
24. **T7N pure (no CutMix) — Normal training alone insufficient.** B:
    v14 bit_F1=0.8609 ni_FAR=100%, fork F1 only 0.6420 on v15. Sentinel
    target zeroing without compositional augmentation under-calibrates
    the rejection boundary.
25. **g=4 LS=0.25 (G, 19G repeat) — over-paste under-soft.** Best v15
    bit_F1 (0.9716) but ni_FAR=100% on v15direct — heavy paste with
    too-soft labels destroys N/I gate. Confirms small-g + high-LS
    (E: g=2, LS=1.0) is the operating point, not the apparent bit_F1
    optimum.

## Iter 28 — Mixup α sweep (paper §5 evidence: pixel α-blend palette destruction)

Source: `iters/iter_28_29_paper_ablation.md`. 6 trains all share iter21E
base recipe (T7N, BCE+LS=0.20, 8ep, AdamW 1e-4, RandomAffine, seed=1)
with **CutMix replaced by Mixup α-blend** at the listed α.

| tag | spec                                   | v14 bF1 | v14 ni% | v15 bF1 | v15 ni% | dual-pass? |
|:---:|:---------------------------------------|--------:|--------:|--------:|--------:|:----------:|
| 28A | Mixup α=0.2 (Zhang 2018 default)       |  0.9875 |   5.00% |  0.9834 | 100.00% | ✗ |
| 28B | Mixup α=1.0                            |  0.9092 | 100.00% |  0.8924 | 100.00% | ✗ |
| 28C | Mixup α=0.1                            |  0.9098 | 100.00% |  0.8627 | 100.00% | ✗ |
| 28D | Mixup α=0.4                            |  0.9753 | 100.00% |  0.9141 | 100.00% | ✗ |
| 28E | Mixup α=2.0                            |  0.9783 | 100.00% |  0.9671 | 100.00% | ✗ |
| 28F | Mixup α=0.4 + cutmix-p=0.5 combo       |  0.9091 | 100.00% |  0.8984 | 100.00% | ✗ |

26. **★ ALL 6 Mixup variants fail v15 ni_FAR (100%) — categorical CutMix
    vs Mixup design difference.** Every α tested fails the v15
    Normal/Invalid gate. Only α=0.2 holds v14 ni at 5% — a fragile
    coincidence that explodes to 100% under v15 OOD pressure.
    **Mechanism (paper §5 narrative)**: chip images are palette-graded
    PNGs (pixel value 0 = Normal, 1–7 = defect intensity grade). Mixup
    `λ·x_A + (1−λ)·x_B` synthesizes invalid intermediate grades (e.g.
    grade 0 + grade 5 at λ=0.5 → quantized grade 3, an unrelated defect
    intensity). Training on out-of-palette pixels with mixed labels
    destroys Normal-vs-defect calibration at the rejection boundary.
    CutMix preserves every pixel's palette grade — this is the
    structural reason CutMix > Mixup on palette-graded multi-label,
    **not a tuning question**.
27. **Mixup+CutMix combo (28F) does NOT rescue Mixup.** α=0.4 +
    cutmix-p=0.5 still hits v15 ni_FAR=100% on both eval sets. The
    pixel-level α-blend contamination cannot be diluted by adding
    CutMix — they don't cancel.

_Source: outputs/iter28A..F/{eval_v14class,eval_v15direct}/preds_chip.parquet._

## Iter 29 — label × spatial isolation (paper §5 evidence: 4 designs all necessary)

Source: `iters/iter_28_29_paper_ablation.md`,
`tables/paper_section5_ablation.csv`. Decomposes iter21E ★ winner into
4 atomic design axes: **region paste vs pixel α-blend** + **full grid
cover vs single rect** + **pair mask vs random partner** + **hard label
vs soft λ-mix**. Three new trains complete the 6-cell label×spatial
matrix (other 3 cells covered by iter21C/21D/21E).

| tag | spec                                              | v14 bF1 | v14 ni% | v15 bF1 | v15 ni% | dual-pass? |
|:---:|:--------------------------------------------------|--------:|--------:|--------:|--------:|:----------:|
| 29A | std box-cut (single rect) + hard label            |  0.7381 | 100.00% |  0.7616 | 100.00% | ✗ |
| 29B | complement g=2 + pair mask + soft LS=0.5          |  0.9921 | 100.00% |  0.9953 | 100.00% | ✗ (highest bF1, FAR fail) |
| 29C | grid_complete g=2 + no pair mask + hard LS=1.0    |  0.9369 |   2.50% |  0.9248 | 100.00% | ✗ |

### Paper §5 — 6-cell label × spatial matrix (final form)

| spatial \ label | soft (λ-mix) | hard (both [A=1, B=1]) |
|:---|:---|:---|
| std box-cut (Yun 2019) | iter21C: v15 0.85/100% ✗ | iter29A: v15 0.76/100% ✗ |
| grid_complete (no pair mask) | iter21D 18F1: v15 0.93/2.5% ✓ | iter29C: v15 0.92/100% ✗ |
| complement + pair mask | iter29B: v15 0.99/100% ✗ | **iter21E ★: v15 0.97/3.75% ✓** |

28. **★ Only iter21E ★ (complement + pair mask + hard label + full-cover)
    clears both gates — every single-axis removal breaks the model.**
    Region paste (vs pixel α-blend, iter28) + full cover (vs single rect,
    iter29A) + pair mask (vs none, iter29C) + hard label (vs soft, iter29B
    and iter21D) all four are necessary. **No single design choice is
    dispensable.**
29. **iter29B is the F1-trap warning — highest single-model v15 bit_F1
    in entire chip-multilabel history (0.9953) but ni_FAR=100%.** Combining
    full-cover complement + pair mask + soft λ-mix label perfectly fits
    in-distribution paste signal but leaks Normal probability mass into
    both defect bins. **F1-only winner ≠ deployable** — the cleanest
    documented case that ni_FAR-blind metric optimization is a trap. Any
    paper or report quoting only bit_F1 (without dual-eval ni_FAR) on
    this design would mis-rank it as the new SOTA.
30. **iter29A (std box-cut + hard label) collapses bit_F1 to 0.74–0.76
    on both eval sets, ni_FAR=100%.** Single rectangular paste leaves
    majority of chip un-touched; hard label says [A=1, B=1] but only
    A's region was actually pasted — the calibration mismatch destroys
    bit_F1 AND ni_FAR simultaneously. Hard label without full cover is
    the worst combination.
31. **iter29C (grid_complete + hard LS=1.0 NO pair mask) v14 ni passes
    at 2.5% but v15 ni explodes to 100%.** Removing pair mask while
    keeping hard label creates ambiguous mixed-class regions that confuse
    the OOD-rejection boundary. Pair mask is the OOD-stability anchor —
    not just a bit_F1 lever.

_Source: outputs/iter29A_box_hard, iter29B_compl_g2_softLS05,
iter29C_grid_hard_LS10/{eval_v14class,eval_v15direct}/._

---

## Best-from-epoch policy ablation (iter99, 260512)

iter99 (5 cells × 4 inference points = 20 cells, all `v15direct n_per_class=200`,
T7 recipe, BCE+LS=0.20 / CutMix complement p=0.25 / g=3 / rect=0.5) tests
whether **forcing `--best-from-epoch=6` on a 10-epoch budget improves model
selection** relative to the iter88/95-97 free policy (any-epoch val_acc winner).

### Setup

| variable | iter95-97 | iter99 |
|----------|-----------|--------|
| epoch budget | 8 (iter95-96) / 20 (iter97) | **10** |
| best_from_epoch | free (typically picks ep 1-3) | **>= 6** |
| LR | 1e-4 (iter95-96) / {5e-5, 2e-5, 1e-5} (iter97) | 1e-4 (A,B,C,D) / 5e-5 (E) |
| backbones | 1 per iter | **5 simultaneously** (ConvNeXtV2 / Swin V1 / DINOv3 / Hiera / ConvNeXtV2 LR-rescue) |

### Per-backbone best-epoch result

| backbone (iter99 tag) | free-policy best ep (iter88/95-97) | forced ep >= 6 (iter99) | I10 macro_f1 free | I10 macro_f1 forced | I10 Total FAR free | I10 Total FAR forced | verdict |
|------------------------|----------------------------------:|-----------------------:|------------------:|--------------------:|--------------------:|---------------------:|---------|
| ConvNeXtV2-B-384 (99A)  | iter46E recipe ~ep3 ~0.9654 / 1.07% | ep 6, 0.8367 | 0.9654 (ref) | 0.8367 | 1.07% | 12.14% | **mild loss** (ep>=6 not optimal) |
| Swin-V1-B-384 (99B)     | iter89 ep=1, 0.9278 / 0.00% | ep 6, **0.8030 / 100% FAR** | 0.9278 | 0.8030 | 0.00% | **100%** | **★ breaks**: peak is ep 1, ep>=6 over-trains |
| DINOv3-ConvNeXt-B (99C) | iter95A ep=3, 0.6211 / 34% | ep 6, **0.7423 / 86.8% FAR** | 0.6211 | 0.7423 (+0.12) | 34.05% | 86.79% | mild rescue but still **below iter97A LR=5e-5 ep=9 ceiling 0.87 / 1.4%** |
| Hiera-B (99D)           | iter96A ep=1, **0.0000 / 0.0%** (degenerate) | ep 6, **0.7039 / 4.76%** | 0.0000 | 0.7039 (+0.70) | 0.0% (reject all) | **4.76%** | **★ full rescue** — degenerate -> useful |
| ConvNeXtV2-B-384 LR=5e-5 (99E) | (no free baseline at LR=5e-5) | ep 8, 0.8282 | — | 0.8282 | — | 10.83% | LR=5e-5 vs LR=1e-4 99A: -0.0085 bF1 (default LR wins) |

### Findings

32. **Best-from-epoch=6 is NOT a global improvement.** Of 5 backbones tested
    at iter99, **2 break** (Swin V1 loses 0.13 bF1 + 100% FAR; ConvNeXtV2
    loses 0.13 bF1 + 11% FAR), **1 is unchanged** (ConvNeXtV2 LR=5e-5 ep~=8
    by chance), **1 is partially rescued** (DINOv3 +0.12 but still -0.13
    vs iter97A LR=5e-5 ceiling), **1 is fully rescued** (Hiera +0.70 from
    a degenerate baseline). Net effect across the 5 backbones is mildly
    negative on bF1 and strongly negative on Total FAR.

33. **The per-backbone best-epoch surface is bimodal**: FCMAE / Swin V1
    backbones peak at **ep 1-3** (matched by 8-epoch budget iter88/95);
    SSL-pretrained backbones (DINOv3) and ViT-with-MAE backbones (Hiera)
    peak at **ep 6-9**. The bimodality is consistent with the
    pretraining-objective hypothesis: backbones whose pretrained features
    already cover most of the chip-palette manifold (FCMAE was supervised
    on a curated 22k-class image set with channel statistics close to the
    chip palette) need only a head-warmup phase, while backbones whose
    pretraining features are far from the chip palette (DINOv3 SSL on
    natural images, Hiera MAE on natural images) need additional
    feature-adaptation epochs.

34. **The cleanest case is Swin V1 base**: iter89 reaches 0.9278 bF1 at
    ep=1, iter99B at the same recipe but forced ep>=6 reaches only
    0.8030 with 100% Total FAR. The 1-epoch checkpoint is **structurally
    better** for this domain — adding 5 more epochs strictly hurts.

35. **The cleanest counter-case is Hiera-B**: iter96A's ep=1 checkpoint
    produces 0.7228 bF1 at I3 (100% FAR) and 0.0000 at I10 (degenerate
    all-reject); iter99D ep=6 produces 0.7039 at I10 with 4.76% FAR
    (deployable). The free policy gives Hiera no useful inference cell;
    forced ep>=6 gives one. **Without iter99, Hiera-B would be ranked as
    fully unusable in `backbone_landscape.csv`; with iter99, it is ranked
    as competitive-but-recipe-mismatched.**

36. **DINOv3 needs both LR rescue AND late-epoch.** iter97A LR=5e-5 ep=9
    free-policy reaches 0.8700 / 1.43%. iter99C LR=1e-4 (iter88 default)
    forced ep=6 reaches 0.7423 / 86.79%. **Reducing LR by 2x has more
    leverage on DINOv3 than forcing late-epoch selection.** The two
    knobs do not combine cleanly — the iter99 budget cannot replicate the
    iter97A ceiling without also dropping LR.

37. **Operational recommendation (paper section 5)**: replace global
    `--best-from-epoch` floor with **per-backbone early-stop with
    backbone-aware patience**:
    | backbone family | patience | rationale |
    |------------------|---------:|-----------|
    | FCMAE supervised (ConvNeXtV2, Swin V1) | 2 epochs | peak at ep 1-3 |
    | SSL (DINOv3) | 5 epochs | peak at ep 8-9 |
    | MAE-pretrained ViT (Hiera) | 5 epochs + LR warmup >= 3 | needs feature adaptation phase |
    Implement as `--early-stop-patience` arg with per-backbone defaults
    pulled from a small mapping in `_train_chip_variant.py`. The current
    floor-based mechanism is the wrong abstraction.

_Source: `outputs/iter99{A,B,C,D,E}_*/T7_*/eval_v15direct_n200/stage1_*/`,
training history `outputs/iter99*/T7_*/history.json`,
log `outputs/_iter99_backbone_ep10_bestfrom6.log`,
delta vs free policy via `outputs/iter89_LR14_LS3_g{2,3}/`,
`outputs/iter95A_*/`, `outputs/iter96A_hiera_base/`,
`outputs/iter97A_lr5e5/`._

## Cosine T_max sweep (ep=10 vs ep=20) — iter111 vs iter112 (260512 night)

**Question**: holding the recipe and seed fixed (ConvNeXtV2-B FCMAE 384
+ T7 BCE+LS=0.20 + FCM-PM CutMix p=0.25 n_groups=3 + `--no-normal`,
seed=1, lr=1e-4 / batch=2 / accum=8), does doubling the cosine LR period
(T_max 10 → 20, also doubling epochs 10 → 20) move the bF1 / Total-FAR
Pareto front?

**Method**: train both with `--save-every-epoch`, dump every epoch
checkpoint (12 ckpts for iter111, 22 ckpts for iter112), then run
`chip_multilabel.run_stage1` against the v15direct n=200 eval set on
every checkpoint × 4 inference cells (I3, I7, I10, I13). Compute the
**absolute-rule bF1** (positive group = single + 2-combo, **3-combo
excluded**) and **Total FAR** (FP rate on Normal + Invalid + OOD
wafer-pattern chips). 12 × 4 + 22 × 4 = 136 (ckpt × cell) measurements,
joined with `outputs/_reeval_absolute_rule.csv` and history.json
val metrics.

**Outcome**:

| run | epochs | T_max | val_f1-selected ckpt | bF1 (best) | Total FAR (best) | Δ vs iter111 |
|-----|------:|------:|---------------------:|-----------:|-----------------:|-------------:|
| iter111 | 10 | 10 | ep08 (val_f1=0.9907 tie) | 0.9963 | 1.31% | — |
| **iter112** | **20** | **20** | **ep06 (val_f1=0.9907 first)** | **0.9964** | **0.83%** | **+0.0001 bF1, −0.48pp FAR** |

**Interpretation**: the dominant gain is on the **FAR axis** (1.31% → 0.83%,
a 37% relative drop in false-alarm rate) not on bF1 (already near
ceiling). The mechanism is that with T_max=10, LR(ep08) = 0.0000095 (1%
of peak) — the model has effectively stopped learning. With T_max=20,
LR(ep06) = 0.0000794 (79% of peak), so the FCM-PM CutMix compositional
signal is still actively shaping the decision boundary at ep06.

**Negative observations**:
- **Total FAR is non-monotone with epoch** under T_max=20 (iter112): ep02
  0.95%, ep05 1.79%, ep06 0.83%, ep08 6.07%, ep10 4.40%, ep16 91.7%, ep20
  0.95%. The volatility argues against any "longer is always better"
  rule — checkpoint selection by val_f1 is essential.
- **val_f1 plateau is multi-modal** in iter112: ep04 (0.9878), ep06/08/10
  (0.9907) are all near-ties on val. Eval-time bF1/FAR varies widely
  across these (ep06 = 0.83% FAR; ep08 = 6.07% FAR). Even within the
  val_f1 plateau, the **first-hit epoch** (ep06) is empirically the
  best — possibly because later plateaus are sampling from the
  decision-boundary collapse phase visible at ep16.
- **iter111 final_epoch and iter101A final_epoch tie at bF1 = 0.9964**
  (the ceiling on this metric), but at 4.52% FAR — **higher bF1 alone
  cannot pick the SOTA cell on the FAR-constrained operating point.**

**Carry-forward**: T_max=20 + `--val-criterion f1` becomes the trainer
default for chip-multilabel paper-replication runs. T_max=ep is no
longer recommended.

_Source: `outputs/iter111_seed1_reproduce_now/T7_*/{history.json,
eval_v15direct_n200_*}`,
`outputs/iter112_ep20/T7_*/{history.json, eval_v15direct_n200_*}`,
combined per-epoch reeval `outputs/_reeval_absolute_rule.csv` (492 rows),
training logs `outputs/_iter111_now.log`, `outputs/_iter112_ep20.log`.
Per-epoch table mirrored at
`docs/chip-multilabel/tables/iter111_112_per_epoch_eval.csv`._

## Validation-criterion ablation (`--val-criterion`: acc / f1 / auroc / aggregate) — iter111-112 (260512 night)

**Question**: which validation-time metric — `val_acc`, `val_f1`,
`val_auroc`, or aggregates of (val_f1, val_auroc) — should drive the
`best_model.pth` selection rule that ships in the paper?

**Method**: with iter101A, iter111, iter112 all training with
`--save-every-epoch`, retroactively re-pick the "best" checkpoint by
each candidate criterion on the stored history.json, then look up the
absolute-rule bF1 / Total FAR for that checkpoint in
`outputs/_reeval_absolute_rule.csv`. Compute Spearman ρ between each
val-criterion and bF1 / Total FAR across all (epoch, run) tuples.

**Outcome on iter112 (cleanest cosine T_max=20 run)**:

| criterion | argmax epoch | bF1 (eval) | Total FAR (eval) | comment |
|-----------|------------:|-----------:|-----------------:|---------|
| val_acc max (=0.9877) | ep01, ep02, ep04, ep10 (tie) | 0.9875 | 1.90% (at ep01) | anti-correlated with bF1 (ρ ≈ −0.52); picks too-early ckpt |
| **val_f1 max (=0.9907)** | **ep06 (first), ep08, ep10** | **0.9964** | **0.83% (at ep06)** | **★ optimal — picks the SOTA cell** |
| val_auroc max (=1.0000) | ep16 (only) | 0.9965 | **91.7%** | saturate-vulnerable; ties at 1.0 push pick toward late epoch with catastrophic FAR |
| arith-mean(vf1, vauroc) | ep06 (tied with ep10) | 0.9964 | 0.83% | matches val_f1 pick |
| geo-mean(vf1, vauroc) | ep06 | 0.9964 | 0.83% | matches val_f1 pick |
| harm-mean(vf1, vauroc) | ep06 | 0.9964 | 0.83% | matches val_f1 pick |

**Interpretation**:
- **val_acc anti-correlates** with the eval metric. On chip-multilabel,
  val_acc measures single-label argmax correctness on a tiny in-distribution
  val split (n=163, no Normal, no OOD). The eval metric probes 2-combo
  compositional correctness + OOD false-alarm rate, which is a *different
  surface*. ep01-02 have peak val_acc because the model has memorized the
  4-class single-label decision boundary; eval-time bF1 + FAR continue
  improving long after val_acc plateaus.
- **val_auroc saturates to 1.0** several times in iter112 (ep03 and ep16
  both hit 1.0000). The arg-max policy picks ep16 — which has 91.7%
  Total FAR (a single-label decision boundary has collapsed onto the
  positive side, scoring every chip as defect). **val_auroc is dangerous
  as a stand-alone selection criterion.**
- **val_f1** picks ep06 cleanly (first-hit of the 0.9907 plateau) and
  recovers the eval SOTA. Aggregate criteria (arith / geo / harm mean of
  val_f1 and val_auroc) all collapse to the val_f1 pick because val_auroc
  ties at 1.0 don't tip the aggregate beyond val_f1's first-hit epoch.

**Recommendation**: ship `--val-criterion f1` as the default and document
val_auroc as **never use stand-alone** in the trainer help text.

**Negative observation**: even val_f1 is not a perfect oracle. **At
val_f1 = 0.9907 plateau (3 ties: ep06 / ep08 / ep10)**, the eval bF1
and Total FAR vary substantially (FAR 0.83% / 6.07% / 4.40%). The
first-hit policy (current implementation) lands on ep06 = SOTA, but
this could be luck — iter113+ multi-seed sweep will check whether the
first-hit-of-val_f1-plateau policy is robust across seeds, or whether a
**val_f1 + low-Total-FAR-on-held-out-OOD** combined criterion is needed
for full robustness.

_Source: `outputs/iter101A_convnextv2_perep/T7_*/history.json`,
`outputs/iter111_seed1_reproduce_now/T7_*/history.json`,
`outputs/iter112_ep20/T7_*/history.json`,
joined with `outputs/_reeval_absolute_rule.csv` per-epoch bF1/FAR rows._

## ★★★ 2026-05-13 ablation update — iter122 + iter123: T6 (BCE → ASL) family, loss-axis dead-end

**Hypothesis (analyst)**: hold iter116J recipe (T7 BCE+LS=0.30, FCM-PM CutMix
complement g=3, val_margin, save-every-epoch) **frozen** and swap the loss to
T6 (BCE warmup → ASL γ_neg=4 with clip), to test whether asymmetric focal
gradient on negatives can amplify the relatively weak partner-bit signal in
2-combos (`bb+sr` partner = sr, `fork+sr` partner = sr). Two clip values
were tested as **a 1-atomic axis sweep on the same recipe**:

| iter | switch ep (BCE → ASL) | ASL clip | val_pick | ep selected | best ckpt eval cell |
|------|---------------------:|---------:|---------:|------------:|--------------------:|
| iter122 | ep6  | 0.05 | val_margin | ep3 (BCE phase) | T0__I10 |
| iter123 | ep4  | 0.10 | val_margin | ep3 (BCE phase) | T0__I10 |

### Eval (T0__I10, n=200/class, seed=42, `chip_multilabel_v15direct`)

| run | ep | bF1 (4 single + 5 combo) | Total FAR | NI FAR | OOD FAR | bb+sr→sr | fork+sr→sr | fork+sr→fork |
|-----|---:|---:|---:|---:|---:|---:|---:|---:|
| iter116J T7 BCE+LS0.30 (val_f1 sel, baseline) | 1 | 0.7911 | 0.00% | 0.00% | 0.00% | 0.831 | **1.000** | 0.919 |
| iter122 T6 ep3 (val_margin BCE pick) | 3 | 0.8122 | **84.20%** | 76.50% | 86.60% | 0.869 | 0.988 | 0.912 |
| iter122 T6 ep6 (first ASL ckpt, clip=0.05) | 6 | 0.8298 | 74.20% | 79.00% | 72.70% | 0.900 | 0.787 | 0.775 |
| iter122 T6 ep10 (final ASL, clip=0.05) | 10 | 0.8297 | 9.40% | 29.00% | 3.30% | **0.981** | 0.750 | 0.819 |
| **iter123 T6 ep3** (val_margin BCE pick) | 3 | 0.7132 | (BCE — same as iter122) | — | — | 0.869 | 0.988 | 0.912 |
| **iter123 T6 ep10** (final ASL, clip=**0.10**) | 10 | 0.8297 | **5.00%** | 16.00% | 1.60% | **0.988** | 0.838 | 0.838 |

### Findings

**clip 0.05 → 0.10 single-atomic delta (iter122 ep10 → iter123 ep10)**:
- bF1: 0.8297 → 0.8297 (Δ = 0.0000)
- Total FAR: 9.40% → **5.00%** (−4.40pp)
- NI FAR: 29.00% → **16.00%** (−13.00pp)
- OOD FAR: 3.30% → **1.60%** (−1.70pp)
- fork+sr partner recall: 0.750 → **0.838** (+0.088 — iter122 trade-off partially recovered)

**vs iter112 paper SOTA (bF1 0.9964 / FAR 0.83%)**: iter122 / iter123 are
regressions on **both axes**. ASL γ_neg=4 amplifies partner-bit gradient
for `bb+sr → sr` (0.831 → 0.981, +0.150) but the same mechanism rebalances
the per-class auto-tuned thresholds so low (e.g. fork=0.02, scratch=0.06)
that Normal/Invalid/OOD chips with weak scratch-like or fork-like noise
flip into the defect class. **Total FAR ≥ 5% under any clip value** — at
least an order of magnitude above iter112's 0.83% FAR. The fork+sr partner
recall (sr-bit on a fork-paired chip) is also fragile under clip=0.05
(0.750), only partially recovered to 0.838 at clip=0.10.

**Why val_margin selection blocks the ablation**: the BCE warmup phase
(ep1-ep3 for iter122, ep1-ep3 for iter123) saturates val_margin > 0.97
early. The ASL phase (ep4-10) produces slightly lower val_margin
(~0.96) because ASL deliberately pushes negative logits below zero by
the clip amount. The result: `--val-criterion margin_max` always
selects an **early BCE ckpt** that has *no ASL signal at all* —
identical between iter122 and iter123 (both pick ep3, both get
`bF1=0.7132` from the unrelated full eval and `bF1=0.8122` from the
preliminary eval). The val_margin criterion cannot distinguish loss
ablations whose effect kicks in only after the switch epoch.

**Verdict — DEAD-END for the T6 loss axis under current recipe**:
1. clip=0.05 over-shifts the prob distribution; FAR catastrophic (9.4%)
2. clip=0.10 less aggressive but still ≥5% FAR
3. clip controls FAR severity but does **not** unlock a bF1 above 0.83
4. partner-recall trade-off (sr-on-bb vs sr-on-fork) is intrinsic to
   γ_neg=4 — increasing γ_neg helps weak-positive recall on one combo
   but suppresses fork-class auto-threshold causing FN on the other
5. val_margin selection cannot pick the ASL phase, so the *full* ASL
   recipe is never compared against val_f1-selected iter112 SOTA at
   the same selection criterion. Even with a hypothetical phase-aware
   selector (`val_f1 + min_ep=switch_ep`), ep10 still yields FAR=5%,
   so the axis is dead end.

**Next axes to try** (1-atomic, retain iter112 recipe as base):
- γ_neg=2 (mild ASL) instead of γ_neg=4 — milder probability shift
- BCE → ASL switch at ep=8 (last 2 ep ASL) — preserves BCE-phase
  conservative thresholds, only briefly applies ASL on near-converged
  weights
- abandon T6/ASL entirely and pivot to a different axis (e.g. spatial
  granularity, see iter124 below)

_Source: `outputs/iter122_T6_asl_gn4/T6_iter122_T6_asl_gn4_260513_085714/eval_v15direct_n200/`,
`outputs/iter122_T6_asl_gn4/T6_iter122_T6_asl_gn4_260513_085714/eval_ep06/`,
`outputs/iter122_T6_asl_gn4/T6_iter122_T6_asl_gn4_260513_085714/eval_ep10/`,
`outputs/iter123_T6_asl_clip01/T6_iter123_T6_asl_clip01_260513_091520/eval_v15direct_n200/`,
`outputs/iter123_T6_asl_clip01/T6_iter123_T6_asl_clip01_260513_091520/eval_ep10/`,
`chip_multilabel/notes.md` (iter 122 / iter 123 entries)._

## ★★★ 2026-05-13 ablation update — iter124: FCM-PM spatial granularity sweep

**Hypothesis (analyst)**: hold iter116J recipe frozen (T7 BCE+LS=0.30,
FCM-PM CutMix `mode=complement` `pair=masked` `label_scale=0.5`,
`val_criterion=margin`, `save-every-epoch`, `--no-normal`), and sweep the
**spatial granularity of the FCM-PM mask** along two axes:

- **g** = `--cutmix-n-groups` (partition count of GRID×GRID cells across A and B)
- **n** = grid multiplier such that `GRID = g · n` (so total cell count =
  `(g·n)² = g²·n²` and `cells_per_group = (g·n)²/g = g·n²`)

This parameterization yields **clean integer GRID values where each group
gets exactly `n²` cells (square sub-blocks)** — the analyst's "clean GRID
= g × n" formulation that avoids irregular partition shapes that earlier
sweeps had hit.

| iter sub | g | n | GRID | total cells | cells/group | cutmix_mode | best ep | total ep | elapsed s |
|----------|--:|--:|-----:|------------:|------------:|-------------|--------:|---------:|----------:|
| 124a | 2 | 1 | 2 | 4   | 2  | complement | 6  | 10 | 407 |
| 124b | 2 | 2 | 4 | 16  | 8  | complement | 10 | 10 | 414 |
| 124c | 2 | 3 | 6 | 36  | 18 | complement | 4  | 10 | 407 |
| 124d | 2 | 4 | 8 | 64  | 32 | complement | 4  | 10 | 403 |
| 124e | 3 | 1 | 3 | 9   | 3  | complement | 10 | 10 | 466 |
| 124f | 3 | 2 | 6 | 36  | 12 | complement | 4  | 10 | 469 |
| 124g | 3 | 3 | 9 | 81  | 27 | complement | 10 | 10 | 463 |
| 124h | — | — | — | bisect_h | — | bisect_h | 6 | 10 | 339 |
| 124i | — | — | — | bisect_v | — | bisect_v | 6 | 10 | 341 |

### Eval status (as of recording)

Only **iter124a** has a completed full-eval (`eval_v15direct/`, 11-class 200
samples × 11 = 2200 chips). The other 8 sub-runs have **empty
`eval_v15direct_n200/`** folders (analyst re-eval was dispatched via
`_iter124_reeval.sh` and is in progress at the time of logging). The 124a
full-eval results (T0__I10 best cell) are summarised here; the remaining
sub-runs are scheduled with rows reserved in the CSV (`bit_F1 = PENDING`)
and will be back-filled when the reeval completes.

### iter124a best-cell eval (T0__I10, full 11-class 200/class)

| metric | value |
|---|---:|
| macro_f1 (4-class, single-label sense) | 0.7699 |
| **bF1 (4 single + 5 combo macro)** | **0.8705** |
| Total FAR | **1.07%** |
| NI FAR | 0.00% |
| OOD FAR | 1.41% |
| top1_11class | 0.5354 |

Per positive-class F1 (T0__I10): bank_boundary=0.6926, fork=0.9496,
scratch=0.9581, scratch_rot=0.9014, bb+fork=0.8626, bb+scratch=0.7619,
bb+scratch_rot=0.8305, fork+scratch=0.9423, fork+scratch_rot=0.9354.

### Findings (partial — single completed sub-run)

- **vs iter112 paper SOTA** (bF1 0.9964 / FAR 0.83%, GRID=g=3 via
  default): iter124a (GRID=2, total 4 cells) yields **bF1 0.8705 / FAR 1.07%**
  — a large bF1 regression (−0.1259), small FAR delta (+0.24pp).
- **bF1 is severely degraded at GRID=2** because at only 4 spatial cells
  per chip image, the FCM-PM complement mask covers exactly half the
  image per "group" — too coarse to learn fine-grained spatial
  composition between the 4 defect bits.
- bank_boundary F1 = 0.6926 (vs iter112's ~0.998) is the dominant
  failure mode — at coarse granularity, the BB perimeter signal gets
  mixed with the paired chip's central defect signal, causing
  position-confused FN. This validates that **the FCM-PM mechanism
  requires GRID ≥ 6** (i.e. cell size ≤ 64 px at 384 input).

### Outstanding cells (analyst re-eval in progress)

The remaining 8 sub-runs (b-i) will reveal:
1. **GRID monotonicity**: does bF1 increase monotonically with GRID
   (2 → 4 → 6 → 8 → 9)? Or is there an intermediate optimum?
2. **g=2 vs g=3 at matched GRID=6** (124c vs 124f): does partition
   count matter when total spatial resolution is held constant?
3. **bisect_h vs bisect_v** (124h vs 124i): horizontal vs vertical
   half-mask — does scratch_rot (top-tilted scratch) benefit from
   horizontal slicing more than vertical?
4. Whether **any granularity beats iter112 paper SOTA** (bF1 0.9964,
   GRID=g=3 via current default).

_Source: `outputs/iter124_a_g2_n1/T7_*/eval_v15direct/stage1_260513_104129/`,
`outputs/iter124_{b,c,d,e,f,g,h,i}_*/T7_*/` (train_summary.json,
history.json — eval pending),
`outputs/_iter124_grid_size_sweep_summary.log` (sub-run mapping),
`_iter124_reeval.sh` (reeval dispatch script)._
