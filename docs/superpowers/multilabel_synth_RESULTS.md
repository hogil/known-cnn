# multilabel_synth — results log

CSV artifacts live under `outputs/multilabel_synth/` (gitignored); key numbers
recorded here.

## MultiMNIST — FCM-PM tuning sweep (3 seeds, ResNet-free SmallCNN 0.62M, CPU)

Protocol: single pool + synthesized combos (arm-specific) + 100/class single
augment; train SmallCNN (spatial-preserving) 20 epochs; evaluate on full-multi
test (all 45 pairs) and held-out-combo test (9 pairs the oracle never trains on).
`outputs/multilabel_synth/fcm_sweep_3seed.csv`.

```
| config          | full mAP        | holdout mAP     | exact | pos   | neg   |
|-----------------|-----------------|-----------------|-------|-------|-------|
| fcm_fill        | 0.7645 +-0.0022 | 0.7625 +-0.0074 | 0.262 | 0.597 | 0.091 |
| oracle          | 0.7591 +-0.0049 | 0.6328 +-0.0068 | 0.253 | 0.593 | 0.091 |
| cutmix_f0.25    | 0.6980 +-0.0022 | 0.6780 +-0.0089 | 0.142 | 0.496 | 0.102 |
| copy_paste      | 0.6606 +-0.0086 | 0.6573 +-0.0147 | 0.150 | 0.488 | 0.094 |
| fcm_checker_g4  | 0.6485 +-0.0073 | 0.6564 +-0.0069 | 0.085 | 0.415 | 0.094 |
| fcm_checker_g3  | 0.6443 +-0.0073 | 0.6405 +-0.0084 | 0.092 | 0.414 | 0.083 |
| fcm_checker_g2  | 0.6303 +-0.0047 | 0.6378 +-0.0139 | 0.098 | 0.435 | 0.094 |
| fcm_strip_g3    | 0.6203 +-0.0055 | 0.6230 +-0.0086 | 0.086 | 0.403 | 0.099 |
| fcm_strip_g2    | 0.6142 +-0.0127 | 0.5987 +-0.0149 | 0.106 | 0.427 | 0.086 |
| cutmix_f0.50    | 0.6066 +-0.0096 | 0.6181 +-0.0193 | 0.085 | 0.421 | 0.115 |
| single_only     | 0.5990 +-0.0098 | 0.5994 +-0.0039 | 0.043 | 0.320 | 0.060 |
| fcm_strip_g4    | 0.5918 +-0.0162 | 0.5922 +-0.0038 | 0.069 | 0.405 | 0.114 |
```

### Findings

1. **fcm_fill (filling-complement: keep base digit whole, fill partner into
   empty background) wins every column.** It matches/exceeds the fully-supervised
   oracle on full mAP (0.7645 vs 0.7591) using single-label data only, and beats
   every other synthesis method.
2. **Compositional generalization confirmed.** On held-out combinations the
   oracle collapses 0.759 -> 0.633 (it never trained on those pairs), while
   fcm_fill stays 0.7645 -> 0.7625 — a +0.130 margin over the oracle at std
   ~0.007 (~18 sigma). Synthesis recovers unseen label combinations that real
   multi-label training cannot.
3. **Mechanism: region-preservation is the lever.** Whole-object preservation
   (fill) >> fragmentation (checker/strip, 0.59-0.65). A naive checkerboard port
   of FCM-PM lost; the faithful filling-complement wins. Confirms "keep the
   object whole, hard-label" as the essential ingredient — and that the
   synthesis must match object structure (compact digits vs extended textures).
4. **Baseline hyperparameters matter too.** cutmix improves markedly with a
   smaller patch (frac 0.25 = 0.698 vs frac 0.50 = 0.607) — rankings from a
   single untuned config are unreliable.

### Caveats

- MultiMNIST is a controlled sanity check; the real generality test is VOC/COCO
  (natural single-vs-multi split). Central-overlap test prior favors
  whole-object-preserving synthesis — a property that also holds for compact
  real objects but must be re-examined on scene images.
- mixup arm omitted from this sweep (focused on region methods); earlier 1-seed
  run had mixup full ~0.678 with soft labels. To be added for completeness.

## MultiMNIST — content-aware (fill) vs content-blind (checker grid) — 3 seeds

Motivation: `fill` needs to know the background (MNIST: pixel==0; chips: palette
grade) — content-aware. `checker` needs no such knowledge — content-blind, so it
is the variant that transfers to natural images where the background is unknown.
Sweep checker grid granularity. `outputs/multilabel_synth/_checker_sweep.log`.

```
| config       | full mAP        | holdout mAP     | exact | pos   | neg   |
|--------------|-----------------|-----------------|-------|-------|-------|
| fcm_fill     | 0.7645 +-0.0022 | 0.7625 +-0.0074 | 0.262 | 0.597 | 0.091 |
| oracle       | 0.7591 +-0.0049 | 0.6328 +-0.0068 | 0.253 | 0.593 | 0.091 |
| cutmix_f0.25 | 0.6980 +-0.0022 | 0.6780 +-0.0089 | 0.142 | 0.496 | 0.102 |
| chk_g20      | 0.6737 +-0.0087 | 0.6821 +-0.0040 | 0.061 | 0.352 | 0.065 |
| chk_g10      | 0.6645 +-0.0106 | 0.6758 +-0.0008 | 0.063 | 0.365 | 0.063 |
| chk_g6       | 0.6566 +-0.0042 | 0.6677 +-0.0083 | 0.075 | 0.388 | 0.068 |
| chk_g4       | 0.6485 +-0.0073 | 0.6564 +-0.0069 | 0.085 | 0.415 | 0.094 |
| chk_g8       | 0.6455 +-0.0101 | 0.6501 +-0.0124 | 0.065 | 0.356 | 0.063 |
| chk_g2       | 0.6303 +-0.0047 | 0.6378 +-0.0139 | 0.098 | 0.435 | 0.094 |
```

### Findings

1. **Finer checker climbs** (full g2 0.630 -> g20 0.674; holdout 0.638 -> 0.682):
   as cells shrink toward pixel dither, both digits survive interleaved.
2. **Content-blind synthesis also beats the oracle on held-out combos**
   (chk_g20 holdout 0.682 > oracle 0.633). Compositional generalization does NOT
   require knowing the background — encouraging for natural images.
3. **Content-aware fill still wins** (0.76 vs blind 0.67-0.68, ~0.08 edge), with
   much better exact-match / confidence. Fine checker lifts per-class mAP but
   lowers exact-match and pos_prob (dither halves each object's density:
   detectable but less confident, worse joint accuracy).

Domain implication: where the background is known (MNIST pixel==0; chip palette
grade) fill dominates; where it is unknown (natural scenes) a content-blind
method still yields compositional generalization above the oracle.

## MultiMNIST — fill is disqualified as cheating; blind max-overlay wins

User ruling: `fill` requires knowing the object/defect location in advance
(where the empty background is), which is exactly what detection tries to find —
so it is cheating and disqualified. Only content-blind methods are legitimate.

Label-fidelity measurement (weaker source's surviving-ink ratio over 3000 synth
pairs; "defect not attached" = survival < 15%):

```
| arm            | mean survival | P(lost<15%) | P(<30%) |
|----------------|---------------|-------------|---------|
| cutmix f0.50   |         0.100 |       0.714 |   0.893 |
| copy_paste     |         0.320 |       0.310 |   0.533 |
| cutmix f0.25   |         0.329 |       0.206 |   0.462 |
| checker g2     |         0.396 |       0.030 |   0.231 |
| checker g4     |         0.441 |       0.004 |   0.054 |
| checker g20    |         0.491 |       0.000 |   0.000 |
| fill           |         0.703 |       0.000 |   0.008 |
```

cutmix/copy_paste frequently drop a labeled object (label noise); fine checker
and fill reach zero — but fill cheats. Fine checker is blind AND label-honest.

Blind max-overlay (max of two whole single digits = the chip min-blend analog;
stronger signal wins per pixel; no location knowledge) — 3 seeds:

```
| config          | full mAP        | holdout mAP     | exact | pos   | neg   |
|-----------------|-----------------|-----------------|-------|-------|-------|
| overlay (blind) | 0.7730 +-0.0030 | 0.7755 +-0.0059 | 0.270 | 0.604 | 0.092 |
| oracle (ref)    | 0.7591 +-0.0049 | 0.6328 +-0.0068 | 0.253 | 0.593 | 0.091 |
| cutmix_f0.25    | 0.6980 +-0.0022 | 0.6780 +-0.0089 | 0.142 | 0.496 | 0.102 |
| chk_g20 (blind) | 0.6737 +-0.0087 | 0.6821 +-0.0040 | 0.061 | 0.352 | 0.065 |
| single_only     | 0.5990 +-0.0098 | 0.5994 +-0.0039 | 0.043 | 0.320 | 0.060 |
```

### Findings (final MultiMNIST picture)

1. **Blind max-overlay wins everything** (full 0.773, holdout 0.776) — beats the
   oracle and the cheating fill, using only single-label data and no location
   knowledge. This is the MNIST analog of the chip min-blend.
2. **max, not average.** overlay (max, hard label) 0.773 vs mixup (average, soft)
   ~0.678 — the failure of blending is specifically averaging/ghosting, not
   combining per se. Right recipe: per-pixel max of the stronger signal + hard
   labels.
3. **Compositional generalization, honestly.** overlay holdout 0.776 vs oracle
   0.633: synthesis generates any combination (incl. held-out) whereas real data
   cannot. Caveat: overlay's full-test edge over oracle partly reflects overlay
   training on held-out pairs too; the clean claims are holdout generalization +
   blind legitimacy + beating fragment/average methods.
4. **Domain unification.** chip min-blend (defect wins over normal) and MNIST
   max-overlay (digit wins over black) are the same "stronger signal wins per
   pixel, no location needed" operation.

## VOC 2007 — natural single vs multi category split (feasibility)

Parsed VOCdevkit annotations directly (pjreddie mirror; the torchvision default
Oxford mirror stalled at ~80 KB/s). No label dropping — the split is the natural
per-image category count.

```
| split    | total | single-category | multi-category |
|----------|-------|-----------------|----------------|
| trainval |  5011 |    2808 (56%)   |    2203 (44%)  |
| test     |  4952 |    2841 (57%)   |    2111 (43%)  |
```

Per-class single-category pool is imbalanced: person 408 ... diningtable 5.
Objects that naturally co-occur (diningtable, tvmonitor, sofa, chair) have few
single-category images relative to their total presence — exactly the classes
where synthesis from singles matters most. The design premise (natural single ->
multi split, no SPML label masking) is feasible on VOC.

Next: build ResNet-50 + mAP training on this split (GPU needed; CPU impractical
for a multi-arm sweep). Adapt the synthesis arms to natural images (content-blind
overlay/cutmix; content-aware needs masks -> COCO).

## VOC 2007 — first real run (ResNet-18, bit_F1/FAR, 8 epochs, 1 seed, subsampled/CPU)

Single pool 765 (cap 40/class), oracle multi 800, test 300, size 128.

```
| arm         | TRAIN bitF1 | EVAL bitF1 | EVAL FAR | exact | ev_pos | ev_neg | ev_mAP |
|-------------|-------------|------------|----------|-------|--------|--------|--------|
| oracle      |       1.000 |      0.355 |    0.021 | 0.293 |  0.615 |  0.043 |  0.511 |
| cutmix      |       0.997 |      0.285 |    0.010 | 0.013 |  0.243 |  0.024 |  0.474 |
| single_only |       1.000 |      0.262 |    0.002 | 0.000 |  0.230 |  0.016 |  0.511 |
| mixup       |       0.919 |      0.162 |    0.001 | 0.000 |  0.174 |  0.011 |  0.482 |
```

Findings: (1) ordering oracle > cutmix > single_only > mixup — synthesis beats
the single-only floor, region (cutmix) beats blend (mixup), consistent with
MNIST. (2) But synthesis does NOT match oracle here (0.285 vs 0.355), unlike
MNIST. Two causes: heavy overfitting on 800-image subsample (train bitF1 ~1.0,
eval ~0.16-0.36) and cutmix label-noise on natural RGB (random rectangle rarely
captures a whole object -> underconfident, pos 0.243). (3) bit_F1/FAR
discriminate arms; mAP did not (all ~0.49).

Next: Mosaic arm (tile whole single images in a 2x2 grid = label-honest blind
synthesis, natural-image analog of MNIST overlay) to test the label-fidelity
mechanism. Absolute VOC performance (overfitting) needs more data -> GPU.
MNIST metric note: exact-match / bit_F1 discriminate arms far better than mAP.

## VOC 2007 — crop-based single pool (negative result: scale mismatch)

To fix the thin single coverage (diningtable 5 natural-single images vs 237
multi-test appearances), switched the single pool to per-object bbox crops
(all 20 classes cap-balanced at 60). Result: every arm got WORSE.

```
| arm         | crop-single bitF1 | natural-single bitF1 | delta  |
|-------------|-------------------|----------------------|--------|
| oracle      |            0.355  |               0.355  |  0.000 |
| cutmix      |            0.234  |               0.285  | -0.051 |
| single_only |            0.227  |               0.262  | -0.035 |
| mixup       |            0.156  |               0.162  | -0.006 |
```

Cause: scale/context mismatch. Crops are tight zoomed single objects; the multi
test is full scenes -> training on crops fails to transfer (crop cutmix pos_prob
0.181 vs natural 0.243). The balanced-coverage gain is outweighed by the
crop->scene domain gap. VOC has no clean single pool: natural-single is
scene-scale but thin for co-occurring classes; crops are balanced but
scale-mismatched. Confirms VOC is a hard testbed for the single->multi paradigm.

Proper fix = Copy-Paste (paste object crops onto scene backgrounds): coverage
(crops) + scale/context (background). Also: overfitting is severe (train bitF1
~1.0, eval ~0.23-0.36 vs literature ~0.8) — credible absolute numbers need
GPU + full data.
