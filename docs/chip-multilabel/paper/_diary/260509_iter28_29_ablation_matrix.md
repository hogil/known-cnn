# 260509 — iter 28 / iter 29 paper ablation matrix integration

## What landed

- **iter 28 (Mixup α sweep)** — six α values (0.1, 0.2, 0.4, 0.8, 1.0,
  2.0) all show v15 `ni_FAR = 100 %`. Zhang 2018 default α = 0.2 has
  best v14 (5.0 % ni_FAR) but unusable v15. Palette-discreteness
  violation at the data-manifold level — hyperparameter-independent.
- **iter 29 (label × spatial 6-cell isolation)** — 2 × 3 ablation
  matrix isolates label axis ({soft λ-mix, hard both}) × spatial axis
  ({std box-cut Yun 2019, grid_complete no pair mask, complement +
  pair mask}). Six cells:
  - 29A (hard + std box-cut) → bit-F1 0.76 / v15 100 % — **worst**
  - 29C (hard + grid_complete) → 0.92 / 100 % — broken
  - 29B (soft + complement + pair mask) → **0.99** / 100 % — surprise
  - 21E (★ FCM-PM, hard + complement + pair mask) → 0.97 / pass — **winner**
  - 21C (soft + std box-cut) — broken (already documented §5.15)
  - 18F1 (soft + grid_complete) — passes (already documented §5.13)

## Paper updates

1. `01_introduction.md` — added §1.1 (Mixup α sweep motivation,
   palette-discreteness violation) + §1.2 (four-design necessity
   teaser).
2. `04_methods.md` — added §4.6.6 (component-decomposition rationale
   table, four-axis necessity, soft/hard label trade-off teaser).
3. `05_experiments.md` — added §5.18 (Mixup α sweep §5.18.1, label ×
   spatial 6-cell matrix §5.18.2, what-§5.18-changes §5.18.3).
4. `06_analysis.md` — added §6.13 (cell 29B surprise mechanism,
   recall–FAR trade-off, generalisation to BCE-sigmoid +
   co-occurrence + FAR pressure regimes).

## Source-of-truth references

- `docs/chip-multilabel/iters/iter_28_29_paper_ablation.md`
  (forthcoming logger artefact)
- `docs/chip-multilabel/tables/paper_section5_ablation.csv`
  (logger working file)

## Next narrator action

- When logger finalises numerical bit-F1 / ni_FAR digits in CSV,
  back-fill into §5.18.1 / §5.18.2 tables (mark with `_Numerical
  digits to be re-imported verbatim_` placeholders currently in
  place).
- §7 discussion may want a paragraph on the four-axis necessity
  proof as a methodological contribution beyond the iter-21 single
  cell narrative.
