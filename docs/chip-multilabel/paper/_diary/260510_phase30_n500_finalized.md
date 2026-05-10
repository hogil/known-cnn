# 260510 — Phase 28 n = 500 finalisation + hard + KD ties pure-hard

## What

Phase 28 v15direct **n = 500** cross-eval over 7 080-chip
intersection (merged 9 model `preds_chip.parquet`) — the
most reliable evaluation in the paper.

## Headline results

- ★ pure-hard MAIN {24_LS030_seed42 + 26 B + 26 D + 26 H}:
  v15 bit-F1 = **0.9953** / `ni_FAR = 0.00 %`.
- ★ **hard + KD ablation {24_LS030_seed42 + 26 B + 26 H +
  33 D}: TIES at 0.9953 / 0.00 %** — replacing 26 D
  (g = 4 LS = 0.40 hard) with 33 D (KD α = 0.5 T = 8)
  produces **identical bit-F1 and per-class numbers**
  (max delta 0.0003 on bb).
- alt seed 7 (24_LS030_seed7 + 26 B/D/H): 0.9963 / 4.50 %
  — highest bit-F1 of all configs but FAR borderline
  (above 5 % gate by one chip's worth).
- iter-33 alt (21 H replacement): 0.9935 / 0 % (PASS, 0.002
  below MAIN).
- iter-34 KD + asym (33 A + 37 E): 0.9922 / 0 % (PASS,
  0.003 below MAIN).

## Cross-scale stability

- pure-hard MAIN: 0.9992 (n = 50) → 0.9955 (n = 200) →
  **0.9953 (n = 500)**, Δ_n200→n500 = 0.0002.
- hard + KD: 0.9984 → 0.9953 → **0.9953**, Δ = 0.0000.

The headline is stable at **0.9953 ± 0.0002**; further
re-evaluation is not headline-priority.

## Single-model FAR fragility (n = 500)

24_LS030_seed42 fails dual-gate alone with **ni_FAR =
22.5 %** (worse than n = 200 reading of ≈ 21 %), yet inside
both 4-bag configurations the ensemble lands at 0 % FAR.
The 22.5-pp absorption is the **strongest paper instance**
of ensemble-from-fragility; the phenomenon **strengthens
with eval-set size** as the non-overlapping over-firing
patterns separate more cleanly across more chips.

## Thesis confirmations

1. **"Pure-hard composition wins" thesis fully falsified at
   n = 500.** Pure-hard 0.9953 = hard + KD 0.9953,
   per-class delta ≤ 0.0003, ni_FAR = 0 % for both. The KD
   axis is a **free substitution slot**, neither penalty
   nor lift.
2. **§6.17 revised reading validated.** All four 4-bag
   composition types (pure-hard, hard + KD, iter-33 alt,
   KD + asym) PASS dual-gate at n = 500; spread 0.0031,
   same envelope as at n = 200.
3. **Ensemble-from-fragility (§6.17.2) strengthened.**
   24_LS030 single-FAIL → 4-bag PASS at all three eval
   scales (n = 50 / 200 / 500), with absorption rate
   *increasing* at n = 500 (22.5 → 0 pp).

## Files updated

- `paper/abstract.md` — final headline 0.9955 → 0.9953
  (n = 500), added "hard + KD ties pure-hard" claim, KD
  axis is a free substitution slot.
- `paper/05_experiments.md` — appended §5.26 with 9-row
  single-model + 5-row ensemble cross-eval (n = 50 / 200 /
  500) tables, n = 200 → n = 500 agreement noted, hard +
  KD ties finding called out.
- `paper/06_analysis.md` — §6.17 strengthened with
  "n = 500 confirmation" subsection (table + per-class +
  fragility-strengthens-with-scale).
- `paper/07_discussion.md` — §7 cost frontier updated to
  n = 500: 4× MAIN row split into pure-hard + hard + KD
  TIE rows; KD-substitution called out as free axis swap.
- `paper/09_conclusion.md` — final headline updated
  0.9955 → 0.9953 (n = 500), TIE configurations
  highlighted.

## Source

- Phase 28 cross-eval block in
  `docs/chip-multilabel/iters/iter_39_purehard_4bag.md`;
- 7 080-chip intersection across 9 model
  `preds_chip.parquet` files;
- aggregated table:
  `docs/chip-multilabel/tables/paper_main_headline.csv`.
