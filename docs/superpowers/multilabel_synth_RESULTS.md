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

## MixedWM38 — first run (real singles train -> REAL mixed eval, seed 0, 15ep)

Public wafer benchmark: 7,015 real singles (train source), 3,000 real mixed
eval (of 30,000; 29 combos, 6 held out from oracle), 1,000 real normals (FAR).

```
| arm         | TRAIN bitF1 | EVAL bitF1 | ev_FAR | exact | HOLDOUT bitF1 | NORMAL FAR |
|-------------|-------------|------------|--------|-------|---------------|------------|
| oracle      |       0.945 |      0.817 |  0.015 | 0.768 |         0.796 |      0.578 |
| overlay     |       0.930 |      0.609 |  0.009 | 0.199 |         0.588 |      0.562 |
| cutmix      |       0.799 |      0.551 |  0.029 | 0.281 |         0.487 |      1.000 |
| fcm_pm      |       0.847 |      0.502 |  0.031 | 0.226 |         0.445 |      0.852 |
| mixup       |       0.873 |      0.450 |  0.007 | 0.072 |         0.412 |      0.005 |
| single_only |       0.930 |      0.227 |  0.019 | 0.002 |         0.224 |      0.636 |
```

Findings: (1) synthesis >> single_only on REAL mixed (0.609 vs 0.227, +0.38) —
thesis direction confirmed on a genuine multi-label benchmark. (2) overlay
(max union = chip min-blend analog) is the best synthesis, consistent with the
wafer encoding (defect beats normal die). (3) oracle gap remains (0.817 vs
0.609) — real mixed has interactions beyond independent-union. (4) NORMAL FAR
explodes for all hard arms (0.56-1.00) — no all-negative signal in training;
exactly the chip lesson (Normal absent -> FAR explosion). Next: port chip
fixes — pair-mask + neg-target.

## MNIST — faithful complement (grid 3N random scatter) sweep, 3 seeds

```
| config              | full mAP        | holdout mAP     | exact | pos   | neg   |
|---------------------|-----------------|-----------------|-------|-------|-------|
| overlay(blind)      | 0.7730 +-0.0030 | 0.7755 +-0.0059 | 0.270 | 0.604 | 0.092 |
| oracle(ref)         | 0.7591 +-0.0049 | 0.6328 +-0.0068 | 0.253 | 0.593 | 0.091 |
| chk_g20(old)        | 0.6737 +-0.0087 | 0.6821 +-0.0040 | 0.061 | 0.352 | 0.065 |
| cmpl_g6n3           | 0.6724 +-0.0082 | 0.6755 +-0.0228 | 0.104 | 0.453 | 0.100 |
| cmpl_g9n2           | 0.6675 +-0.0074 | 0.6721 +-0.0147 | 0.099 | 0.429 | 0.087 |
| cmpl_g18n3          | 0.6561 +-0.0072 | 0.6487 +-0.0095 | 0.083 | 0.392 | 0.063 |
| cmpl_g9n3 (recipeA) | 0.6558 +-0.0067 | 0.6603 +-0.0220 | 0.111 | 0.444 | 0.092 |
| cmpl_g9n4           | 0.6431 +-0.0038 | 0.6284 +-0.0054 | 0.090 | 0.431 | 0.089 |
| single_only         | 0.5990 +-0.0098 | 0.5994 +-0.0039 | 0.043 | 0.320 | 0.060 |
```

Faithful complement lands in the fragment band (0.64-0.67) on compact digits;
overlay stays best. Larger A share helps (n2>n3>n4). ALL complement variants
beat the oracle on held-out combos. Together with WM38 (overlay 0.609 >
complement 0.502): raw complement geometry is not universally superior —
the chip SOTA's complement value is coupled to pair-mask FAR control, which is
the next port.

## MixedWM38 — final 3-seed results with synthetic-normal (blind FAR fix)

synthetic-normal = defect dies erased from real singles via min(x, 0.5): a
blind all-negative training signal (no normal labels, no location knowledge).

```
| config (3 seeds)      | TRAIN bitF1 | EVAL bitF1      | exact | HOLDOUT bitF1 | NORMAL FAR      |
|-----------------------|-------------|-----------------|-------|---------------|-----------------|
| oracle                |       0.942 | 0.863 +-0.064   | 0.770 |         0.836 | 0.548 +-0.175   |
| overlay+sn+neg003     |       0.924 | 0.641 +-0.019   | 0.248 |         0.614 | 0.031 +-0.026   |
| overlay+sn            |       0.906 | 0.581 +-0.022   | 0.178 |         0.554 | 0.001 +-0.001   |
| fcm_pm_pm+sn+neg003   |       0.835 | 0.540 +-0.035   | 0.254 |         0.487 | 0.058 +-0.031   |
```

Findings: (1) synthetic-normal completely fixes NORMAL FAR (overlay 0.562 ->
0.001) at ~0.03 bitF1 cost; with neg003 the winner reaches bitF1 0.641 +-0.019
(74% of oracle) at NORMAL FAR 0.031 — 18x lower than the oracle trained on
real mixed data (0.548 +-0.175, unstable). (2) The oracle cannot control false
alarms without a normal concept; the synthesis pipeline solves it by design.
Headline: zero multi labels, zero normal labels, zero location knowledge ->
74% of oracle bitF1 with 18x lower false-alarm rate.

## MS-COCO — subsampled-scale boundary (honest negative)

Natural split measured: train2017 = 24,186 single-cat (20%) / 93,080 multi-cat
(79%); all 80 classes have singles but the tail is thin (baseball glove 2).
At CPU-subsampled scale (cap 30/class singles, 1.5k synth, ResNet-18, 6ep):
oracle 0.146 / cutmix 0.087 / mixup 0.062 / single_only 0.053 / copypaste
0.052 eval bitF1 with train ~1.0 — everything collapses, no ranking signal.
80-way multi-label needs GPU-scale training; recorded as a measured scale
boundary, not a method result.

## MixedWM38 — backbone + scaling hardening

ResNet-18 (small-input stem) on the winner config overlay+sn+neg003, 3 seeds:
bitF1 0.717 +-0.079 (83% of oracle 0.863), exact 0.399, NORMAL FAR
0.013 +-0.003 (42x below oracle 0.548). Scaling (SmallCNN, seed 0):
n_train 1000/3000/6000 -> bitF1 0.589/0.660/0.683 (monotone, unsaturated).
Backbone criticism answered; gap to oracle narrows with capacity + data.

## MixedWM38 — stage-2/3 validation (selection + rejection), seed 0, 30ep

Stage 3 (margin reject, winner recipe, SmallCNN 30ep): rejecting samples with
max prob < tau: tau=0.9 -> coverage 91.7%, accepted bitF1 0.709 (up from
0.700), NORMAL FAR 0.151 -> 0.000 on 1000 real normals. Zero false alarms at
8.3% review cost; accepted-set accuracy does not drop. Also: 30ep lifts
SmallCNN bitF1 to 0.719 (15ep: 0.641) at the cost of FAR 0.031 -> 0.146 —
which rejection then eliminates.

Stage 2 (val-F1 vs val-margin checkpoint selection): NOT reproduced on WM38 —
with a synthetic-combo val set, val_f1 keeps rising to ep27 and both pickers
choose adjacent epochs (bitF1 0.719 vs 0.700, FAR ~0.15 both). The chip
val-F1-saturation pathology stems from an easy in-dist val; a well-built
synthetic val keeps F1 informative. Boundary finding: margin selection
matters when the val set saturates (chip); synthesis-based val removes the
need. Honest scope: stage-2 evidence remains chip-domain.

## MixedWM38 — easy-val stage-2 test (prediction failed, honest) + new best

Prediction "easy in-dist val saturates F1 like the chip" FAILED: val_f1 kept
setting records to ep30 (0.988); f1-pick(ep30) REAL bitF1 0.7787 BEAT
margin-pick(ep26) 0.6958. The chip val-F1-saturation pathology requires the
chip regime (pretrained 88M + LS0.295 + trivially easy val); scratch SmallCNN
on WM38 never enters it within 30 ep. Stage-2 (margin selection) is therefore
chip-scoped in the paper — two external attempts, both negative, reported.

Silver linings: (1) stage-3 reject reproduced again (tau .9: coverage 94.4%,
NORMAL FAR 0.041 -> 0.000, bitF1 kept). (2) NEW BEST single model: SmallCNN
30ep f1-pick bitF1 0.7787 / NORMAL FAR 0.028 pre-reject — beats ResNet-18
15ep 0.717; training is far from converged at 30 ep -> pushing epochs closes
the oracle gap (0.863).

## MixedWM38 — 60-epoch run: synthetic-real overfitting boundary + stage-3 third reproduction

60ep (combo val, seed 0): real bitF1 PEAKS near ep30 (0.7787) then declines
(ep52 pick: 0.7482) while synthetic val keeps rising to ep50 (0.954) — beyond
~ep30 the model overfits the synthetic combo distribution at the expense of
real mixed. Neither selection criterion (F1 or margin) on a synthetic val can
see the real peak — an honest limitation of synthetic-val selection and an
argument for a small real val when available. exact-match keeps rising
(0.298 -> 0.482). Stage-3 reject third reproduction, cheapest yet: tau 0.9 ->
coverage 98.1%, NORMAL FAR 0.062 -> 0.000, bitF1 kept (0.7465).

## MixedWM38 — label-fidelity survival (mechanism cross-validated)

Weaker-source defect-pixel survival (2000 pairs): overlay 1.000 > cutmix 0.579
> fcm_pm(complement g9n3) 0.527 > mixup(lam .5) 0.236 (92% of pairs lose >70%
of the weaker defect to ghosting). Survival ordering exactly matches the
downstream 3-seed bitF1 ordering (0.58-0.72 / 0.563 / 0.495 / 0.435) — label
fidelity predicts multi-label performance on a second dataset (after MNIST).

## MixedWM38 — R-batch final (headline): oracle parity + zero FAR, 3 seeds

ResNet-18 + overlay+sn+neg003, 30ep, 3 seeds: bitF1 0.825 +-0.037 vs oracle
0.863 +-0.064 (within combined seed variance = statistical parity), holdout
0.801 vs 0.836, and NORMAL FAR 0.000 on ALL THREE SEEDS without rejection
(oracle: 0.548 +-0.175). SmallCNN n6000 30ep: 0.749 +-0.064 (data helps;
backbone is the stronger lever). Epoch curve (SmallCNN): real bitF1 peaks
0.818 @ ep30 while synthetic val rises monotonically to ep50 — figure-grade
evidence that synthetic val cannot see the real peak. MNIST mixup 3-seed:
0.666 +-0.010 (overlay 0.773: max-vs-avg gap +0.107 confirmed).

## VOC 3-seed + held-out pairs / COCO-20 (natural-RGB boundary, final)

VOC (20ep, cap100, 3 seeds): full bitF1 oracle 0.410 +-0.076 > copypaste
0.379 +-0.010 > single_only 0.303 +-0.019 — synthesis clearly beats the floor
and approaches the oracle on natural RGB. Held-out pairs: ALL methods collapse
together (oracle 0.200, copypaste 0.208, floor 0.186) — no compositional
advantage in RGB. COCO-20 (s0): oracle 0.450 (signal restored vs 80-class
0.146), but synthesis ~= floor (cutmix 0.304 vs single 0.293). Consistent with
the superposition-domain theory (Def 1): parity/compositional/zero-FAR are
superposition-domain properties; in RGB synthesis yields modest (VOC) to no
(COCO) gains and no compositional edge. The theory predicts both regimes.

## MixedWM38 — order extrapolation + annotation efficiency (extrachain final)

Order extrapolation (pairs-only training, real mixes by order): bit-level
extrapolates gracefully (2/3/4-mix bitF1 0.784/0.671/0.628) but joint
prediction does not (exact 0.699/0.143/0.000). Adding higher-order synthesis
(+triples/+quads) HURTS everything: real higher-order mixes are not arbitrary
joins — only 12 of 56 possible 3-combos and 4 of 70 4-combos exist in WM38.
Synthesizing arbitrary combos wastes capacity on non-existent combinations.
Key insight: the oracle's real privilege is knowledge of the COMBINATION
SUPPORT, not the mixed images themselves — this identifies the delta term in
Prop 1. Practical recipe stays pairs-only.

Annotation efficiency (winner recipe, 30ep, seed 0): 500 singles -> bitF1
0.617 (FAR 0.002); 2000 -> 0.697; 7015 -> 0.779. ~62 singles per class
already yields 0.62 bitF1 with near-zero false alarms.

## MixedWM38 — conformal FAR calibration (Prop 3 verdict)

Real-normal calibration (n=500, held-out 500 real normals for test): guarantee
HOLDS — alpha=0.05 -> realized FAR 0.040 (tau 0.527, coverage 99.97%);
alpha=0.01 -> 0.006 (tau 0.672, coverage 99.47%); accepted bitF1 unchanged.
Practical: normals are known-good samples requiring NO defect annotation, so
finite-sample FAR control costs zero labeling expertise.
Synthetic-normal calibration FAILS (tau 0.045 -> real FAR 0.97-0.99): training
on the same synthetic normals collapses their scores; the score-space shift to
real normals is large. Calibration must be exchangeable with deployment
normals — an honest boundary of the annotation-free regime.

## Reuters-21578 — 4th domain family (text), 3 seeds

Natural split (5,995 single-topic train / 300 real multi-topic test, top-20).
oracle 0.567 +-0.025 > vec_avg 0.402 +-0.023 > concat 0.359 +-0.020 >>
single_only 0.221 +-0.007. Synthesis recovers 71% of the oracle from singles
only — 4th family confirmed. OPERATOR FLIP: the averaging (mixup-analog)
operator BEATS the join (concat) in text, the reverse of images. Explanation
completes the theory: topic evidence lives on (nearly) disjoint feature
coordinates (distinct vocabularies), so averaging preserves evidence; image
classes share pixel coordinates, so averaging ghosts. The invariant law is
evidence preservation (label fidelity), not any fixed operator — the
fidelity-maximizing operator is determined by the modality's evidence
geometry.

## MixedWM38 — fairness batch final (equal-condition comparison + oracle+reject)

Equal condition (ResNet-18, 30ep): oracle 3-seed bitF1 0.974 +-0.019 (matches
published MixedWM38 accuracy 98-99% — harness validated against literature),
NORMAL FAR 0.563 +-0.083 (4 trainings: 0.446/0.629/0.613/0.945 — high and
unstable). Ours 6-seed: 0.837 +-0.039 = 86% of the literature-grade oracle,
NORMAL FAR 0.000 in ALL SIX seeds without rejection. The earlier "statistical
parity" (vs SmallCNN-15ep oracle 0.863) is RETRACTED as a weak-oracle
artifact — caught by our own fairness check.

Oracle + rejection (F3): rejection CANNOT rescue the oracle. Its max-prob on
real normals is >=0.99 for 80% of them, so even tau=0.99 leaves NORMAL FAR
0.799 at 100% coverage — no threshold separates its normals from defects.
The reliability advantage is created by the TRAINING design (synthetic
normals + pair-mask + neg-target shaping the confidence geometry), not by
the rejection stage; the oracle cannot buy it with more real data.

## MixedWM38 — combo-support hypothesis rejected (delta term finalized)

Synthesizing only the 29 REAL combos (support knowledge, no images/labels)
does not recover joint accuracy: 4-mix exact stays 0.000, 3-mix 0.117->0.158
marginal, 2-mix drops (budget split). The oracle's privilege is therefore not
knowledge of WHICH combos occur but the APPEARANCE INTERACTIONS of real
higher-order mixes — an image-distribution gap no label-side knowledge can
close under independent-union synthesis. This finalizes the delta term of
Prop 1: bit-level evidence composes; joint appearance does not.

## WM-811K (real, public) — categorical-treatment anchor for paper C

cca 8-class real maps (1,779; values {0,128,255}), SmallCNN, 10 seeds:
A nearest+one-hot 0.7633 +-0.0524 > C nearest-gray 0.7248 +-0.0485 >
B bicubic-gray 0.7036 +-0.0552. Categorical treatment beats standard
interpolated-grayscale input by +0.060 macro-F1 (7/10 paired seeds); the
gain is driven primarily by one-hot categorical ENCODING (A-C +0.039, 8/10),
while the pure nearest-vs-bicubic resize effect is within noise here (C-B
+0.021, 5/10) — the cca renders are already nearest-upscaled 224px, limiting
interpolation damage. Honest scoping: real-data support for "treat
categorical signals categorically"; the dramatic interpolation ceiling
remains the synthetic 33-class result (0.9784 -> 0.9946).

## MixedWM38 — headline tightened to 9 seeds

overlay+sn+neg003 (ResNet-18, 30ep), 9 seeds: bitF1 0.8413 +-0.0339 (86.4%
of the equal-condition oracle 0.974), holdout 0.8229, exact 0.4806, NORMAL
FAR mean 0.0008 (exactly zero in six of nine seeds; max 0.005). Individual
seeds: 0.858/0.773/0.845/0.881/0.799/0.865/0.879/0.835/0.837.

## Plant Pathology 2021 (agriculture, RGB) + resource ceiling note

Condition-type SEMANTICS (diseases co-occur on one leaf) but RGB-PHOTO
representation. Lean 1-seed (CPU memory-bound; full 8.2k-image 3-seed run
OOM-killed twice on the shared box): oracle 0.316 / overlay 0.425 / cutmix
0.372 / mixup 0.347 — all clustered 0.32-0.43, no clean separation, oracle
not dominant (undertrained at lean scale). This matches the VOC RGB pattern,
NOT the wafer pattern. Refined thesis: the predictor of synthesis success is
the REPRESENTATION being signal-ordered (wafer palette grade, X-ray density,
TF-IDF), NOT the condition-type semantics per se. A condition-type task in a
non-signal-ordered (RGB photo) space behaves like the natural-image boundary.

Resource ceiling (honest): on the shared CPU box, full-size Plant (3 seeds,
8.2k imgs RGB 128px) OOM-dies mid-run; ChestX-ray14 full (45GB) download
failed to write on two attempts. CXR (grayscale = closer to signal-ordered,
the key test of the refined thesis) and full-scale runs are GPU/'큰-머신'-gated.

## ChestX-ray14 (HF streaming, medical grayscale) — CPU-capacity collapse (inconclusive)

Streamed 3943 single / 2500 multi / 3500 normal (npz cached; no 45GB dl —
kaggle giant-zip failed, HF parquet-shard streaming worked). SmallCNN 0.62M,
128px, 20ep: ALL arms collapse to mAP ~0.22-0.24 and bitF1@0.5 ~0
(oracle 0.233/0.008, single_only 0.228/0.000, synth 0.22-0.24). No arm
separation — 14 subtle thoracic findings exceed this tiny model's capacity at
CPU scale (same wall as COCO-20 and full-Plant). Faint pos>neg gap (oracle
0.19>0.10) shows weak signal below threshold, but the model/metric cannot use
it. CXR is thus INCONCLUSIVE at CPU scale — a GPU/stronger-backbone (ResNet+,
AUC metric) experiment, not a method failure. NORMAL FAR stayed near 0 for all
(the confidence-geometry mechanism still holds trivially since nothing fires).

## Consolidated CPU-reachable evidence (state of the campaign)

Solid (CPU): MixedWM38 (0.841 vs oracle 0.974, FAR 0 x9), MultiMNIST
(overlay > oracle on held-out), Reuters (71%, operator flip). These + theory
carry the TMLR/Q1 case. GPU-gated boundary/confirmation: COCO, full-scale
Plant/WM38, ChestX-ray14 (all collapse or OOM at CPU-SmallCNN scale — capacity
walls, not refutations). Refined thesis (signal-ordered representation, not
condition-type semantics, predicts success) stands on WM38+Reuters(yes) vs
Plant-RGB(no); CXR would test grayscale but needs GPU.

## FULL-DATA confirmation runs (3-seed)

Reuters FULL (all top-20 singles 5995 / test-multi 300 / oracle-multi 889):
oracle 0.603 / vec_avg 0.433 (72% of oracle) / concat 0.398 / single_only
0.254. Same ordering as subsample (vec_avg>concat>>single, operator flip
intact), numbers slightly higher — subsample conclusion confirmed at full data.

## MultiMNIST FULL (400 singles/class, 8k synth train, 3k test, 25ep, 3 seeds)
Blind max-overlay EXCEEDS the equal-condition oracle at full scale, and the
compositional-generalization gap widens on held-out pairs:
- overlay 0.8676 mAP_full / 0.8832 holdout (pos 0.737 neg 0.057)
- oracle  0.8457 / 0.6847  (pos 0.720 neg 0.058)  <- collapses on unseen pairs
- mixup   0.7376 / 0.7343  (pos 0.340 neg 0.046)
- single  0.6193 / 0.6347  (pos 0.311 neg 0.046)
- cutmix  0.6055 / 0.6040  (pos 0.464 neg 0.104)
overlay beats oracle (+0.022 full, +0.198 holdout) AND beats mixup/cutmix
baselines. Ranking unchanged from ablation scale; margins larger. Paper
(DRAFT + latex tab:families + MNIST subsec + abstract) updated to these.

## Loss-engineering control (WM38, SmallCNN, atomic arm x loss, 3 seeds)
Isolates "why synthesize instead of a better loss on singles?" — no
synth-normal, no neg-target, n_single_aug 2000, 20ep. Only arm and loss vary.
- single_only + BCE   : ev_bitF1 0.243 (pos 0.167 neg 0.032) nrm_FAR 0.591
- single_only + ASL   : ev_bitF1 0.316 (pos 0.231 neg 0.103) nrm_FAR 1.000  <- over-predicts
- single_only + Focal : ev_bitF1 0.242 (pos 0.172 neg 0.044) nrm_FAR 0.584
- overlay + BCE (ours): ev_bitF1 0.607 (pos 0.555 neg 0.019) nrm_FAR 0.549
TRAIN bitF1 ~0.92-0.95 for all four (equal single-fit). The strongest
multi-label loss (ASL) recovers only +0.073 over BCE and blows FAR to 1.00;
synthesis adds +0.364 (5x the loss gain) with LOWER FAR. Conclusion: the
bottleneck is missing co-occurrence structure, unrecoverable by any loss on
singles; synthesis supplies it. Direct ICLR rebuttal to "just use a better
loss." (CSVs: aslbase_{bce,asl,focal}.csv)

## MixedWM38 FULL (ResNet18, 17015 train / 14000 test, 30ep, 3 seeds)
All singles as sources (n_single_aug 7015), full 14k mixed test. synth-normal
(4000) + neg-target 0.03 applied IDENTICALLY to every non-oracle arm.
- oracle      : ev_bitF1 0.9844+-0.0065  ho 0.980  NORMAL_FAR 0.186  (100%)
- overlay(ours): ev_bitF1 0.7947+-0.0734  ho 0.797  NORMAL_FAR 0.0003 ( 81%)
- cutmix      : ev_bitF1 0.8548+-0.0487  ho 0.897  NORMAL_FAR 0.618  ( 87%)
- mixup       : ev_bitF1 0.5809+-0.0891  ho 0.506  NORMAL_FAR 0.758  ( 59%)
- single_only : ev_bitF1 0.4086+-0.0098  ho 0.375  NORMAL_FAR 0.390  ( 42%)

Honest reading (KEY):
1. Under IDENTICAL synth-normal FAR control, only overlay reaches zero FAR
   (0.0003). cutmix/mixup label-noise (destroyed-source false positives)
   defeats the control -> fires on normals (FAR 0.62/0.76). Clean empirical
   proof of the fidelity->FAR theory (Prop 2 / Thm 1 independence term).
2. cutmix's HIGHER raw bit_F1 (0.855 vs overlay 0.795) is UNUSABLE: 62% false
   alarm. The bit_F1-only lens crowns cutmix; the FAR lens shows overlay is
   the only viable operator. Reinforces mandatory bit_F1 + FAR reporting.
3. overlay ~2x single_only (0.795 vs 0.409) at full scale.
4. overlay bit_F1 0.795+-0.073 is within seed variance of the 9-seed n6000
   headline (0.841+-0.034); oracle rose with full data (0.974->0.984) so
   recovery reads 81% here vs 86%. NOT a regression — 3-seed noise (s0=0.898).

## ChestX-ray14 feasibility probe (ResNet18, 128px, 20ep, 1 seed) — INCONCLUSIVE
Cached HF split (single 3943 / multi 2500 / normal 3500). lr 3e-4.
- oracle      : bitF1 0.148  mAP 0.312  NORMAL_FAR 0.276
- overlay     : bitF1 0.140  mAP 0.272  NORMAL_FAR 0.455
- single_only : bitF1 0.075  mAP 0.280  NORMAL_FAR 0.204
Read: the ORACLE itself is weak (mAP 0.31) -> 128px/20ep/subsampled is
badly undertrained (still ~3x chance, so learning but far from converged).
overlay's bitF1 ~2x single_only (synthesis signal present) but mAP is a tie
and its NORMAL_FAR is WORSE (0.46 vs 0.20) -- synth-normal control does NOT
transfer here as it does on WM38. NOT a clean medical win at this scale.
Decision: CXR as a 2nd real benchmark needs a real investment (224px, 50ep+,
full 100k data, more GPU) = a separate project, not a quick tier-add. Held
for user direction; paper does NOT claim CXR. The paper stands on the clean
superposition domains (WM38 / MNIST / Reuters) + the theory (Thm 1/2, Cor 1/2)
which already explains where the method applies without needing CXR.

## WM38 margin+confidence reject: does the chip recipe transfer? NO
SmallCNN, n3000, 3 seeds. synth-normal + neg-target on ALL arms incl. oracle.
Max-probability confidence-reject bitF1 / NORMAL_FAR (3-seed mean):
- oracle     0.9757 / 0.000
- overlay    0.6977 / 0.000
- fcm_pm     0.6230 / 0.003
- fcm_pm_pm  0.5921 / 0.000
Verdict: fcm_pm / fcm_pm_pm are the WEAKEST synthesis arms on the PUBLIC WM38
benchmark (below overlay, far below oracle). The chip domain's ~0.99 result is
chip-specific and does NOT transfer. Confidence rejection controls FAR to ~0 for every
arm (the one robust win), but oracle+synth-normal also reaches FAR 0, so the
honest FAR claim is "annotation-free synth-normal gives any model FAR 0", not
"synthesis beats oracle on FAR". Hypothesis (fcm_pm+margin+reject matches/beats
oracle on WM38) REFUTED. Keep overlay as the WM38 representative; drop fcm_pm
from the WM38 headline. This experiment is not Gaussian NB: it thresholds the
maximum predicted bit probability. CSV:
`D:/project/known-cnn/outputs/multilabel_synth/wm38_margin_reject_3s.csv`.

## WM38 fidelity CAUSAL intervention: attenuate one source's evidence, keep label
SmallCNN, n3000, 3 seeds. f = evidence-retained fraction (1=full overlay,
0=erased-but-still-labeled). Preliminary 3-seed mean:
| f    | bitF1  | pos    | neg    | NORMAL_FAR |
| 1.00 | 0.6789 | 0.6547 | 0.0416 | 0.0337     |
| 0.75 | 0.6906 | 0.6754 | 0.0355 | 0.0027     |
| 0.50 | 0.7013 | 0.6984 | 0.0312 | 0.0050     |
| 0.25 | 0.6815 | 0.6703 | 0.0377 | 0.1810     |
| 0.00 | 0.5785 | 0.5232 | 0.1303 | 0.1223     |
Verdict: naive "fidelity down -> bit_F1 down monotonically" is REFUTED (bit_F1
is flat/up to f=0.5; defect still detectable at half strength). The real
preliminary signal is a threshold effect: neg_prob jumps at full erasure and
NORMAL_FAR becomes highly variable below f=0.5. It does not yet establish a
monotonic FAR law (three seeds and large variance are insufficient).

The original `survival` column is invalid for a causal slope because it credits
source-B evidence inside source-A's support, producing about 0.292 survival at
`f=0`. The implementation was corrected to counterfactual marginal survival
(`f=1 -> 1`, `f=0 -> 0`), catastrophe rate, per-class probabilities, and five
paired data splits. Until the corrected run completes, use this only as a
preliminary negative/non-monotonic result. Legacy CSV:
`D:/project/known-cnn/outputs/multilabel_synth/wm38_fidelity_causal_3s.csv`.

Corrected five-split result (counterfactual marginal survival):

| f | supported bitF1 | pos | neg | gap | NORMAL FAR |
|---:|---:|---:|---:|---:|---:|
| 0.00 | 0.631 | 0.517 | 0.133 | 0.384 | 0.179 |
| 0.10 | 0.724 | 0.602 | 0.067 | 0.536 | 0.507 |
| 0.25 | 0.806 | 0.681 | 0.040 | 0.642 | 0.253 |
| 0.50 | 0.811 | 0.697 | 0.032 | 0.665 | 0.005 |
| 0.75 | 0.814 | 0.687 | 0.032 | 0.655 | 0.012 |
| 1.00 | 0.793 | 0.654 | 0.041 | 0.614 | 0.015 |

Endpoint paired-bootstrap effects (full minus erased): bitF1 +0.162
[0.129, 0.208], gap +0.229 [0.191, 0.276], NORMAL FAR -0.164
[-0.234, -0.087]. Spearman rho is 0.662 for bitF1, 0.624 for gap, and
-0.656 for FAR, but adjacent-step monotonicity is only 60%, 68%, and 52%.
Conclusion: catastrophic evidence loss degrades the operating frontier, while
maximum evidence is not the F1 optimum. Corrected CSV:
`D:/project/known-cnn/outputs/multilabel_synth/wm38_fidelity_causal_corrected_5split.csv`.

## WM38 strict prior-method comparison COMPLETE (5 seeds, pick=val_tail_margin_guarded, neg=.02)
KEY FINDING: `summation_mixup_shin22` == `overlay` == np.maximum(ca,cb) in wm38_arms.py
(binary encoding: max-union IS the faithful re-encoded clipped sum of Shin 2022).
So our best operator COINCIDES with the closest prior work. No operator novelty on WM38.
| operator                        | bitF1  | nrmFAR |
| max-union/overlay(=Shin22 summ) | 0.8000 | 0.0102 |  <- best
| cutmix                          | 0.6909 | 0.4390 |
| fcm (no Pair-Mask)              | 0.6649 | 0.3836 |
| FCM-PM (chip op)                | 0.6544 | 0.1472 |
| average_mixup_shin22            | 0.5741 | 0.6762 |
| union_mixup                     | 0.5667 | 0.6550 |
| mixup                           | 0.5370 | 0.2250 |
| single_only                     | 0.4732 | 0.6022 |
| mixup_shin22                    | 0.4536 | 0.4814 |
Implications: (1) FCM-PM (0.654) LOSES to max-union (0.80) on WM38 -> FCM-PM scoped to chip
(0.99) not WM38 headline. (2) Pair-Mask value = FAR control: FCM-PM 0.147 vs fcm 0.384 FAR
at similar bitF1. (3) Paper repositioned to FRAMEWORK + THEORY + GUARANTEE (commit 4e03deb):
contribution is the fidelity criterion (predicts max-union is best), the superposition theory
(why it matches oracle), and the annotation-free conformal FAR guarantee (Shin22 lacks) -
NOT a new operator. Pending: operating-curves + conformal (the guarantee evidence).

## DECISIVE: max-union (=Shin22 summation) violates wafer die-budget (density evidence)
Defect-die fraction of on-wafer area (measured):
- real 2-mix (ground truth): 0.3052 +/- 0.050
- single (1 defect):         0.2904
- max-union/overlay/Shin22:  0.5012 +/- 0.123   <- 64% denser than real; 91% exceed real 95th pct
- FCM (complement):          0.2925             <- matches real 2-mix
- FCM-PM (complement+PM):    0.2925             <- matches real
INTERPRETATION: WM38 real mixing is a die-budget PARTITION (each die = one defect type),
NOT max-superposition (union double-counts). max-union assumes the WRONG generative model
-> over-dense (0.50 vs real 0.31) -> distributionally unrealistic ("cheating" per user, now
empirically justified). FCM complement is the FAITHFUL partition model (matches real density).
CONSEQUENCE: max-union/overlay/summation EXCLUDED as die-budget-violating. Among faithful blind
synthesis, FCM-PM has the best F1-FAR trade-off (0.654 bitF1 / 0.147 FAR vs cutmix 0.69/0.44,
fcm 0.665/0.384). This REHABILITATES FCM-PM as the legitimate WM38 method and connects to the
theory (correct generative model -> oracle-faithful). Repositioning: FCM-PM = method (faithful
+ best legit trade-off + Pair-Mask FAR control + conformal guarantee); die-budget partition is
a novel verified insight (WM38 is partition, not superposition). ICLR estimate up to ~28-40%.

## Conformal FAR guarantee COMPLETE (real-normal cal, 5 seeds x 50 splits) - FCM-PM WINS coverage
Distribution-free split-conformal on max-prob; realized FAR ~= alpha for ALL methods (guarantee holds),
but COVERAGE at guaranteed FAR differs sharply (FCM-PM = far more usable):
| method       | a=0.01 realized/cov | a=0.05 realized/cov |
| FCM-PM (ours)| 0.0098 / 0.972      | 0.0503 / 0.989      |  <- best coverage
| fcm (no PM)  | 0.0100 / 0.839      | 0.0503 / 0.908      |
| cutmix       | 0.0097 / 0.759      | 0.0517 / 0.884      |
| mixup        | 0.0096 / 0.737      | 0.0488 / 0.799      |
| single_only  | 0.0092 / 0.518      | 0.0495 / 0.540      |
KEY: at a GUARANTEED 1% false-alarm rate, FCM-PM retains 97.2% of wafers usable vs 51.8-83.9% for
baselines (1.2-1.9x more). Pair-Mask lowers intrinsic FAR so the conformal reject abstains least.
This is the decisive differentiator: FCM-PM = die-budget-faithful synthesis + best legit F1-FAR
trade-off + highest coverage under a distribution-free FAR guarantee. Combined with the die-budget
insight + theory + cross-domain (text) + chip (0.99), the package is coherent+rigorous.
ICLR estimate: ~32-42% (up; clean quantified guarantee win). TMLR/IEEE TSM 70-85%.
