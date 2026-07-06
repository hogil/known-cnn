# 260519 09:06 — cron #210 — NEW CHAMPION E22 (KD-T-temperature diversity)

## TL;DR

**E22 = 5-way {iter116J_s1 + iter116J_clone_s77 + iter116J_g3_ls20_s77_v17 + KD_v7 + KD_v12} per-bit majority k=2 I10 at v15direct_n2000 POS9 strict.**

- bit_F1 = 0.9956
- NI-FAR = 0.00 % (0/2000)
- OOD-FAR = 0.00 % (0/640)
- Total FAR = 0.00 % (0/2640)
- vs E21 (prev champion 0.9953 / 0% / 0/2640): **+0.0003 bit_F1** at identical 0-FP gate
- vs E7 (chain v7 champion 0.9941 / 0%): +0.0015 bit_F1
- vs single SOTA iter116J_s1 (0.9927 / 0%): +0.0029 bit_F1

## What changed

The marginal axis added on top of E21 is **KD_v12** — the second KD student
trained with the **same α=0.30 base recipe but T=3** (vs KD_v7 T=2). This
introduces a **KD-T-temperature diversity axis** that did not exist in the
prior pools:

| Pool axis           | E21 (prev champion)                  | E22 (new champion)                                  |
|---------------------|--------------------------------------|------------------------------------------------------|
| LS=0.30 seeds       | s1, s77                              | s1, s77                                              |
| LS=0.20 seeds       | s77                                  | s77                                                  |
| KD α                | 0.30                                 | 0.30                                                 |
| **KD T**            | **T=2 only (KD_v7)**                 | **T=2 + T=3 (KD_v7 + KD_v12)**                       |
| Total members       | 4                                    | 5                                                    |
| Majority threshold  | k=2                                  | k=2                                                  |

## Why it works (hypothesis)

KD_v7 (T=2) and KD_v12 (T=3) share teacher and α but differ only in
softening temperature. T=3 produces softer teacher probabilities → the
student learns a marginally more entropy-rich decision boundary. At per-bit
majority k=2, the two KD students vote independently on hard chips where
hard-label students (s1, s77, LS20_s77) might over-confidently miss
weak-signal bits. The 0/2640 FP gate is preserved because both KD students
still inherit conservative OOD/NI rejection from the shared teacher; the
+0.0003 bit_F1 is purely from positive bit recall on borderline chips.

## Sweep context

- Source: `outputs/_kd_swap_sweep/C5_5way_v7v12.json`
- Full table: 14 candidates C1-C14 covering all combinations of {KD α ∈
  {0.20, 0.30, 0.40, 0.50}, T ∈ {2, 3, 4}, skip-cutmix toggle} added to
  the E21 4-way base
- C5 (E22) = the only candidate achieving **0% FAR AND bit_F1 > 0.9953**
- All other 5-way candidates either (a) regress bit_F1 below E21, or
  (b) lift bit_F1 marginally but break the 0% FAR gate

## Pareto context

- E22 k=1 I10 = 0.9957 / 6.93% Total FAR — peak F1 but FAR fail (loose-FAR
  variant if 5% gate)
- E22 k=2 I10 = **0.9956 / 0.00%** — NEW CHAMPION (strict)
- E22 k=3 I10 = 0.9935 / 0.00% — over-conservative (-0.0021 vs k=2)

## Files updated

- `docs/chip-multilabel/tables/paper_main_headline.csv` — E22 row appended
  (`ens_5way_E22_KD_Tdiversity_NEW_CHAMPION`); prior E21 row marked
  SUPERSEDED 260519
- `docs/chip-multilabel/02_results.md` — champion timeline new top section
  `## 2026-05-19 NEW CHAMPION — E22 5-way KD-T-temperature-diversity bit-vote ensemble`
- `docs/chip-multilabel/paper/_diary/260519_0906_cron210_E22_new_champion_KD_T_diversity.md`
  — this entry

## Not touched (per directive)

- `09_conclusion.md` — frozen-final lock; requires user unfreeze
- `outputs/` — read-only
- `chip_multilabel/` source — no code change

## Open questions / next probes

1. Does a third KD student at T=4 or T=5 lift further? (6-way C-axis sweep)
2. Does swapping KD_v7 (T=2) for a T=2.5 student preserve gain while
   shrinking T-axis spread? (axis-position ablation)
3. Symmetric ablation: does **removing** KD_v12 from a 6-way (E22 + LS30_s11)
   pool drop bit_F1 back to E21 level? (KD_v12 isolated contribution test)

These are pure eval-only probes — no training risk.
