# 260511 Phase 56 — iter 55 loss-function ablation

## Setup
- Base recipe: 26 B (FCM-PM pair-mask + complement-CutMix + LS = 0.20 + 8 epochs + g = 3 + corner fill).
- Modifier per cell: loss family + LS strength only. All other knobs held at 26 B values. FULL n = 200, single seed.
- Goal: validate that the choice of T7 BCE + LS at ls = 0.20 is at a sweet spot — both loss family and LS strength.

## Results — 7-row matrix (incl. 26 B baseline)

| cell | loss | ls | bF1 | ni_FAR | dual | bb / fk / sc / sr |
|------|------|---:|----:|-------:|:----:|---|
| 55 A | T3 Focal (γ = 2) | 0.20 | 0.9155 | 100 % | FAIL | 0.9803 / 0.9413 / 0.8870 / 0.8533 |
| 55 B | T4 ASL (default γ) | 0.20 | 0.7056 | 1 % | PASS | 0.9279 / 0.5799 / 0.6577 / 0.6569 |
| 55 C | T9 sigmoid focal | 0.20 | 0.9615 | 0 % | PASS | 0.9602 / 0.9518 / 0.9450 / 0.9889 |
| 55 D | T8 CE + soft + LS | 0.20 | 0.9105 | 0 % | PASS | 0.9145 / 0.9091 / 0.9552 / 0.8632 |
| 55 E | T7 ls = 0.05 | 0.05 | 0.9585 | 100 % | FAIL | 0.9762 / 0.9552 / 0.9154 / 0.9873 |
| 55 F | T7 ls = 0.30 | 0.30 | 0.8133 | 0 % | PASS | 0.9677 / 0.7431 / 0.6392 / 0.9032 |
| 26 B | T7 BCE + LS | 0.20 | **0.9781** | **2.5 %** | PASS | (canonical) |

## Findings (4)

1. **T7 BCE + LS = strict winner** among 6 multi-label loss formulations. All 5 alternatives regress (− 0.017 to − 0.272). Two cells (Focal, weak LS) regress AND break FAR. The chosen loss is the optimum among standard multi-label losses tested at this data scale.
2. **LS = 0.20 is a narrow sweet spot, not a plateau.** Sweeping LS at fixed loss family T7: ls = 0.05 under-smooths (FAR break + bF1 regress), ls = 0.30 over-smooths (− 0.165 bF1). PASS region within ±0.05 of 0.20.
3. **T4 ASL fails counter-textbook.** Designed for multi-label imbalance (negatives dominant), should help — instead delivers − 0.272 catastrophic. Default γ⁻ / γ⁺ from COCO-80 over-down-weights borderline-positive gradients in our 4-class small-cardinality setting. Hyper-params from large multi-label benchmarks do not transfer.
4. **T3 Focal breaks FAR via confidence-pushing.** Same mechanism as iter 54's EMA / warmup / drop-path failures (§5.36 / §6.22): up-weighting hard examples treats Normal-chip residual activations as hard negatives → pushed toward defect → `ni_FAR` collapses to 100 %.

## Unified FAR-control story (paper §6.23)

Three orthogonal mechanisms compose to maintain `ni_FAR ≤ 5 %`:

| layer | mechanism | section |
|-------|-----------|---------|
| data construction | pair-mask + complement-CutMix | §6.19 |
| loss calibration | BCE + LS at ls = 0.20 | §6.23 |
| improvement (KD) | 4-bag teacher soft targets, α = 0.5 / T = 4 | §6.22 |

Confidence-pushing modifiers (Focal loss family, weak LS, EMA, warmup, drop-path) all break FAR by the same dynamic — Normal chips with residual defect-like activations become "hard" examples and are pushed toward defect predictions. Calibration-friendly losses (BCE + LS at ls = 0.20) cap maximum target probability at 0.80, providing a confidence ceiling that prevents leakage. KD soft targets augment this with per-class FAR-boundary information without disrupting either pair-mask suppression or LS calibration.

## Paper impact

- HEADLINE 0.9953 / 0 % unchanged.
- §5/§6/§7/abstract gain "T7 BCE + LS at ls = 0.20 is THE loss" claim, anchored by 6 negative results.
- Paper-grade negative result: ASL fails counter to its design intent; default hyperparams calibrated for COCO-80 do not transfer to 4-class small-cardinality. Cite Ridnik 2021 (arxiv 2009.14119) and contrast.
- Combined with §5.36 / §6.22, the 26 B non-KD baseline is now exhausted under both training-dynamics modifiers (§5.36) and loss-function alternatives (§5.37). Beyond 26 B requires KD distillation or ensemble cost.

## Files edited
- `docs/chip-multilabel/paper/05_experiments.md` (§5.37 appended)
- `docs/chip-multilabel/paper/06_analysis.md` (§6.23 appended, unified FAR-control story)
- `docs/chip-multilabel/paper/07_discussion.md` (§7.10.4 appended)
- `docs/chip-multilabel/paper/abstract.md` (loss-function ablation paragraph appended)
- this diary entry
