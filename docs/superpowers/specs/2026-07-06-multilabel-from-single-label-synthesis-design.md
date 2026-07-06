# Multi-label from single-label supervision via region-preserving synthesis

Design spec — 2026-07-06

## 1. Motivation & thesis

In real industrial inspection (semiconductor defect maps), **multi-label
annotation is infeasible**: co-occurring defects explode combinatorially, many
combinations are rare, and overlapping patterns are ambiguous to label. What is
cheap and unambiguous is **single-label** data — an isolated instance of one
defect type.

**Thesis (title-level claim):** *We recover unseen label combinations from
single-label supervision alone.* A region-preserving, hard-label synthesis of
combinations from single-label sources (FCM-PM) (a) dominates other ways of
manufacturing multi-label training data from singles, (b) approaches the
fully-supervised multi-label oracle, and (c) **exceeds the oracle on held-out
combinations** — because a model trained on real multi-label data cannot learn
a combination that is absent from that data, whereas synthesis can generate it.

The value is **zero multi-label annotation**, not zero supervision. Single-label
supervision is used; no human ever labels a multi-defect image. This is NOT
fully unsupervised learning (that is a different method and out of scope).

**Why this is paper-worthy (not just SOTA):** the contribution is a mechanism
(why hard-label region synthesis beats blending), a problem framing
(single-label field constraint), honest gap-closing to an oracle, and one
genuine oracle-beating result — compositional generalization. SOTA numbers are
not the currency; the transferable insight is.

## 2. Positioning & target venues

- **Stretch target:** top-tier CV/ML — CVPR / ICCV / ECCV / NeurIPS, or IEEE
  TPAMI / TIP (journal equivalents). Reaching this hinges on **elevating
  compositional generalization to the central thesis**, standard-protocol SPML
  comparison, mechanism analysis of why blending fails, and crisp
  differentiation from Copy-Paste (Ghiasi CVPR'21), CutMix, and SPML.
- **Solid landing:** strong domain/applied venues — IEEE TII, IEEE T-ASE, IEEE
  T-Semiconductor Manufacturing, Pattern Recognition, WACV, BMVC. High
  probability with the design below even without clearing the top-tier bar.
- Explicitly **not** targeting IEEE Access (low selectivity).

The single biggest lever for the ceiling: **held-out-combo compositional
generalization is the main experiment, not an appendix.**

## 3. Problem formulation

- Label space: `L` categories. A sample's target is a multi-hot vector over `L`.
- **Single-label pool** `S`: samples where exactly one category is present
  (multiple instances of that one category allowed — this is defect *type*, not
  count, matching the chip setting).
- **Multi-label test** `M`: samples where >= 2 categories are present.
- Where a dataset is naturally single-only (MNIST), both the combos and the test
  set are synthesized. Where a dataset is naturally mixed (VOC/COCO), the
  single/multi split is taken from the **natural** label counts — no artificial
  label dropping (this is cleaner and more honest than SPML label-masking).

## 4. Datasets & roles

```
| Dataset      | single pool           | multi test              | size        | role                     |
|--------------|-----------------------|-------------------------|-------------|--------------------------|
| MultiMNIST   | MNIST single digit    | synthesized 2-digit     | synth (inf) | controlled mechanism     |
| VOC 2007     | natural 1-category    | natural >=2-category    | ~10k        | standard bench (mAP)     |
| MS-COCO      | natural 1-category    | natural >=2-category    | ~120k       | scaling (heaviest, last) |
```

At build time, verify VOC/COCO have enough natural single-category images to
form a usable single pool; report the counts. (VOC is expected to have plenty.)

## 5. Synthesis arms (fixed backbone; only the data-manufacturing method varies)

```
| Arm          | training data              | expected position     |
|--------------|----------------------------|-----------------------|
| oracle       | real multi-label           | ceiling (not a rival) |
| FCM-PM       | single + FCM-PM combo      | closest to ceiling    |
| CutMix synth | single + CutMix combo      | middle                |
| Mixup synth  | single + Mixup combo       | low (blend corrupts)  |
| copy-paste   | single + pasted combo      | middle-low            |
| single-only  | single only                | floor                 |
```

FCM-PM essentials carried over from the chip work: complement-fill grid layout,
pair-mask, hard (region) labels rather than blended soft labels. These are the
mechanism under test; ablations isolate each.

## 6. Main experiment — held-out combo (compositional generalization)

For each dataset, choose a set of held-out category combinations `H`.

- **Oracle** is trained on real multi-label data with combinations in `H`
  **removed** (it never sees them).
- **Synthesis arms** are trained on single-label data and synthesize
  combinations, including those in `H`.
- All arms are evaluated on a test set of `H` combinations.

Headline table per dataset: full-multi mAP (gap-closing) **and** held-out-combo
mAP (where FCM-PM is expected to exceed the oracle). This is the paper's core
figure.

## 7. Backbone & metrics

- VOC / COCO: **ResNet-50** (literature-standard for multi-label; mAP directly
  comparable to published SPML numbers). MultiMNIST: small CNN.
- Metrics: **mAP** (headline for VOC/COCO), **exact-match accuracy** + per-digit
  F1 (MultiMNIST). Diagnostic: per-class pos/neg probability (train + eval),
  consistent with project reporting conventions.
- Standard-protocol SPML baselines (ROLE, EM, Hill loss, etc.) compared where
  feasible to establish external credibility.

## 8. Experiment matrix & compute

- Matrix: 3 datasets x 6 arms x 3 seeds ~= 54 runs (36 without COCO).
- Compute reality: GPU is shared (~30-40% occupied by other work); plan for
  ~40-50 GB free on an H100.
  - MultiMNIST: minutes/run. VOC: cheap (few min/epoch, ResNet-50). COCO:
    heavy — deferred to last, optionally subsampled.
- Seeds: 3 (report mean +- std). A 1-seed fast draft is acceptable for the very
  first mechanism check.

## 9. Build order (phased)

1. **Harness** — dataset loaders (MultiMNIST synth, VOC/COCO natural split),
   the 6 synthesis arms behind one interface, ResNet-50 / small-CNN trainer,
   mAP + exact-match + pos/neg-prob eval, held-out-combo protocol.
2. **MultiMNIST full matrix** — cheapest; proves mechanism + compositional
   generalization immediately.
3. **VOC** — natural single->multi split + held-out combo.
4. **COCO** — scaling, last (optionally subsampled).

Each phase gates the next: read results before committing to the next dataset.

## 10. Deliverables

- Reusable harness under a new top-level dir (e.g. `multilabel_synth/`), separate
  from the chip-specific `chip_multilabel/`.
- Per-dataset headline tables (full-multi mAP + held-out-combo mAP), synthesis
  showdown table, ablation of FCM-PM essentials, mechanism analysis (why blend
  fails), SPML-baseline comparison.
- All result artifacts under `outputs/` (gitignored); large data under `data/`
  or `E:/data/images/` (gitignored) — never committed.

## 11. Out of scope

- Fully unsupervised (zero-label) multi-label learning.
- The same-domain wafer dataset (MixedWM38) — deliberately excluded; the point
  is generality on datasets whose objects we do not need domain knowledge of.
- Beating the oracle on the *full* multi-label test (not claimed; oracle is a
  ceiling except on held-out combos).

## 12. Key risks

- **Novelty vs Copy-Paste / CutMix / SPML.** Mitigation: make the pair-mask +
  complement-fill + hard-label mechanism and the compositional-generalization
  result the thesis, with rigorous ablation and (ideally) analysis of why
  blending corrupts the label.
- **Non-standard VOC/COCO single->multi split.** Mitigation: also run the
  standard SPML protocol so numbers are comparable to prior work; report the
  natural-split counts transparently.
- **COCO cost.** Mitigation: defer; subsample if needed; log any coverage caps.
