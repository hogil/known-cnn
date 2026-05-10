# 260509 — iter 21 FCM-PM (19C) narrative landed

## What happened

iter 21 ran the 8-model dual-eval panel (v14 min-blend ∪ v15 direct
synth). Logger owns the headline table
`docs/chip-multilabel/tables/iter21_paper_headline.csv` and the iter
log `docs/chip-multilabel/iters/iter_21_clean_baseline.md`; this
narrator pass added the paper-grade reasoning into 04/05/06.

## Quantitative anchors (cited from logger table)

- 19C (FCM-PM): v14 bit_F1 = 0.9913, v15 bit_F1 = 0.9691,
  ni_FAR = 0% / 3.75%, F1bb = 1.000 on both.
- 12-T5 baseline (iter 11): v14 = 0.9745, v15 = 0.7872 (drop
  −0.182 → fails dual-eval).
- F1_scratch v15: 12-T5 = 0.5841 → 19C = 0.9439 (+0.36 abs / +62%
  rel).
- F1_sr v15: 12-T5 = 0.7739 → 19C = 0.9776 (+0.20 / +26%).
- 21C (standard CutMix Yun 2019, λ-mix label): ni_FAR = 100% on
  both eval sets — broken.
- 19E / 19F (Complement): ni_FAR ≤ 5% on both eval sets — partial
  pass.

## What I wrote

- **04_methods.md §4.6 FCM-PM** — full method spec with cell
  partitioning (g=4, `(i+j) mod g` sub-lattice, no quadrant), Mix
  / Mask construction with pseudocode, hard union target on mix
  and A-only target on mask, novelty argument vs Yun 2019 / Zhang
  2018 / Kim 2020 / Chong 2024, 19C training recipe table.
- **05_experiments.md §5.15** — iter 21 narrative with the
  dual-eval protocol motivation (v14 vs v15 covariate), 4-cell
  cite table (12-T5 / 19C / 21C / 19E-F), the four quantitative
  claims pointing at the logger headline table, mechanism for why
  standard CutMix collapses Normal decision boundary
  (sigmoid+λ-mix mismatch), mechanism for why FCM-PM works
  (full-cover, pair-grounded mask, hard union under sigmoid head),
  paper claim sentence.
- **06_analysis.md §6.10** — three sub-sections: (1) why dual-eval
  is the right protocol with the three-tier robustness
  classification (robust / v14-overfit / broken); (2) 19C residual
  analysis on the 3 v15 false-alarm chips (root cause = same
  fork-overfiring of §6.3 at lower amplitude in the bright-pink
  band); (3) distribution-shift robustness paper claim with the
  scope-narrowing caveat (synthesis-pipeline level, not
  deployment level).

## What I deliberately did not write

- Did not duplicate the headline numerics from the logger table.
  Cited cells, gave reference. Logger owns numbers, narrator owns
  reasoning. (Per separation-of-concerns brief.)
- Did not add an iter-21 entry to abstract.md / 09_conclusion.md /
  07_discussion.md — those are higher-level docs and require a
  separate pass once the iter-21 follow-up (multi-seed 19C +
  ensemble (19C, C_44)) lands.
- Did not touch 02_related_work.md. Yun 2019 / Zhang 2018 / Kim
  2020 / Chong 2024 are now cited in §4.6 but not yet enumerated
  in §2.3 (multi-label losses) — queued for the next narrator
  pass after the iter-22 ensemble experiment confirms the recipe
  generalises beyond 19C single-model.

## Files touched (this session)

- `docs/chip-multilabel/paper/04_methods.md` (+§4.6, ~140 lines)
- `docs/chip-multilabel/paper/05_experiments.md` (+§5.15, ~140 lines)
- `docs/chip-multilabel/paper/06_analysis.md` (+§6.10, ~95 lines)
- `docs/chip-multilabel/paper/_diary/260509_iter21_fcm_pm.md` (this
  file, new)

## Next narrator pass triggers

- iter 22 ensemble (19C ⊕ C_44) — if positive, append to §4.3
  ensemble subsection and §5.16, update §6.10.3 robustness claim.
- 3-seed 19C variance run — append variance bars to §5.15
  outcome table, update H1 evidence strength.
- g sweep `{2, 4, 8}` and FCM-PM mix probability `p` sweep —
  append §5.16 sensitivity analysis to §5.15, possibly §4.6.6
  hyperparameter sweep table.
