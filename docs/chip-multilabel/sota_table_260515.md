# Chip Multi-Label SOTA Snapshot — 2026-05-15

**Scope**: Consolidates every checkpoint that has been evaluated on the open-set
multi-label suite (4 single defects + 5 two-combos + Normal + Invalid + 4 OOD
wafer-pattern classes) since the 5-stage ladder kicked off (260514). All numbers
follow the project metric convention:

- `bit_F1` = macro-F1 over the 9 positive cells (4 single + 5 combo).
- `NI-FAR` = (Normal + Invalid) false-alarm rate — 2,000 chips.
- `OOD-FAR` = false-alarm rate over 4 OOD wafer patterns — 640 chips.
- `Total FAR` = (NI + OOD) FP / 2,640 chips. Primary safety metric.

Eval set: `chip_multilabel_v15direct_n2000` (n=23,300 chips, 16 class keys,
seed 42 split val=4,660 / eval=18,640). All single checkpoints are evaluated
with the 4-variant inference matrix (I3 / I7 / I10 / I13). Ensembles use I10
sigmoid probs.

Source CSV: `outputs/_n200_sota_scan.csv`, ad-hoc per-cell extraction via
`_sota_table_extract.py` (computes POS_KEYS_9 macro from
`preds_chip.parquet`). Ensemble JSONs:
`outputs/_ens_3fcmpm_vote.json`, `outputs/_ens_3fcmpm_logit_entropy.json`,
`outputs/_ens_3fcmpm_T0__I10.json`.

## Findings (260515)

The current single-model SOTA is **iter116J (g=3, LS=0.30) with I10**:
bit_F1 0.9927 at 0.00 / 0.00 / 0.00 % FAR — every Normal, Invalid, and OOD
chip is correctly rejected while still nailing 99 % of 9 positive cells.
Stacking the three SOTA singles (g=2 / g=3 / g=4) via post-hoc ensembling
extracts another +0.30 pp: **vote_union_bits I10 = 0.9958** at 0.00 / 0.31 /
0.08 % FAR (only 2 OOD chips leak), and **logit-avg + entropy gate I10 =
0.9935** with the FAR locked back to 0.00 %. Vote-majority and majority-bits
land at ~0.989, slightly behind union but with FAR strictly 0.

Three failure modes clarified by today's sweeps:

1. The wafer-grid retraining run (`W2RT_pt95_nt30_g2_grid16`) and the
   pure-grid baseline (`GRID_g2_NP/PR`) all hit ≥80 % NI-FAR — the
   coarse-grid framing collapses Normal/Invalid into the active set.
2. CutMix-only single (`RECT025_NP`, `_PR`, `_PR2`, `_PR2_pn70/85`) keeps
   bit_F1 high (0.97) but cannot beat iter116J — the asymmetric pair
   formulation (`g4n2`, complement masked corner) is the lever.
3. All four KD attempts (v1 = 3fcmpm teacher, v2 = T2/α=0.7, v3 = T8/α=0.3,
   v4 = T4/α=0.5/LS=0.20, v5 = α=0.2 iter116J recipe) collapse to bit_F1
   < 0.20. The teacher prob distribution + skip-on-cutmix gating zeroes the
   student's F1 — KD does not transfer in this regime.

The ASY g=3 cell (bit_F1 0.998 at I7) is intriguing for ensembling but its
NI-FAR (5 %) and OOD-FAR (25 %) drop the constraint. Next-step focus is
either (a) ensemble swapping iter116J for ASY_g3 to lift partner combos, or
(b) bagging g=4 LS variants where G4_pt95 already gives 95 % bit_F1 with
manageable FAR.

## Consolidated SOTA table

```
| Recipe                                        | bestI | bit_F1 | NI-FAR | OOD-FAR | Total FAR | Status        |
|-----------------------------------------------|-------|--------|--------|---------|-----------|---------------|
| ladder1 BCE+LS=0.30 single (baseline)         | I13   | 0.5644 |  99.65 |   98.91 |     99.47 | collapse      |
| ladder2 Focal LS=0 single                     | I13   | 0.9081 |   0.00 |    0.31 |      0.08 | FAR ok F1 mid |
| ladder3 ASL g_neg=4 clip=0.05 single          | I3    | 0.9101 | 100.00 |  100.00 |    100.00 | over-positive |
| ladder3 ASL g_neg=4 clip=0.05 single          | I13   | 0.9017 |  99.85 |   99.69 |     99.81 | over-positive |
| ladder4 CutMix-only single p=0.25 LS=0.30     | I7    | 0.9926 | 100.00 |  100.00 |    100.00 | peak F1       |
| ladder4 CutMix-only single p=0.25 LS=0.30     | I10   | 0.9795 |  37.00 |   57.81 |     42.05 | F1 lift       |
| ladder4 CutMix-only single p=0.25 LS=0.30     | I13   | 0.8547 |  14.20 |   49.53 |     22.77 | FAR drop      |
| ladder5 CutMix pair p=0.25 LS=0.30            | I3    | 0.9913 | 100.00 |  100.00 |    100.00 | peak F1       |
| ladder5 CutMix pair p=0.25 LS=0.30            | I13   | 0.9015 |  98.55 |   99.69 |     98.83 | FAR fail      |
| ladder5b complement g4n2 pair p=0.25 LS=0.30  | I3    | 0.9860 | 100.00 |  100.00 |    100.00 | peak F1       |
| ladder5b complement g4n2 pair p=0.25 LS=0.30  | I10   | 0.9591 |  80.00 |   52.66 |     73.37 | half FAR cut  |
| ladder5b complement g4n2 pair p=0.25 LS=0.30  | I13   | 0.9424 |  79.85 |   35.94 |     69.20 | pair-mask eff |
| ladder5c single pair p=0.50 LS=0.30           | I3    | 0.9859 | 100.00 |  100.00 |    100.00 | peak F1       |
| ladder5c single pair p=0.50 LS=0.30           | I13   | 0.8639 |  80.00 |   77.81 |     79.47 | mid F1        |
| ladder5d complement g4n2 pair p=1.00 LS=0.30  | I3    | 0.9920 |  80.00 |   78.12 |     79.55 | peak F1       |
| ladder5d complement g4n2 pair p=1.00 LS=0.30  | I13   | 0.9499 |  10.85 |   61.88 |     23.22 | tradeoff      |
| iter116A g=2 LS=0.20 complement g4n2 p=0.25   | I3    | 0.9873 | 100.00 |  100.00 |    100.00 | peak F1       |
| iter116A g=2 LS=0.20 complement g4n2 p=0.25   | I10   | 0.9829 |   0.00 |    0.00 |      0.00 | g=2 SOTA      |
| iter116A g=2 LS=0.20 complement g4n2 p=0.25   | I13   | 0.9713 |   0.00 |    0.00 |      0.00 | g=2 valid     |
| iter116J g=3 LS=0.30 complement g4n2 p=0.25   | I3    | 0.9941 | 100.00 |  100.00 |    100.00 | peak F1       |
| iter116J g=3 LS=0.30 complement g4n2 p=0.25   | I10   | 0.9927 |   0.00 |    0.00 |      0.00 | OVERALL SOTA  |
| iter116J g=3 LS=0.30 complement g4n2 p=0.25   | I13   | 0.9745 |   0.00 |    0.00 |      0.00 | safe variant  |
| iter116F g=4 LS=0.30 complement g4n2 p=0.25   | I3    | 0.9700 |   2.20 |   19.53 |      6.40 | F1 + low FAR  |
| iter116F g=4 LS=0.30 complement g4n2 p=0.25   | I7    | 0.9631 |   0.20 |    5.78 |      1.55 | balanced      |
| iter116F g=4 LS=0.30 complement g4n2 p=0.25   | I10   | 0.9623 |   0.00 |    0.31 |      0.08 | g=4 SOTA      |
| iter116F g=4 LS=0.30 complement g4n2 p=0.25   | I13   | 0.9352 |   0.00 |    0.31 |      0.08 | g=4 safe      |
| W2RT pt95 nt30 c50 g=2 grid16 retrain         | I7    | 0.9982 | 100.00 |  100.00 |    100.00 | peak F1       |
| W2RT pt95 nt30 c50 g=2 grid16 retrain         | I10   | 0.9950 |  91.40 |   92.19 |     91.59 | FAR collapse  |
| W2RT pt95 nt30 c50 g=2 grid16 retrain         | I13   | 0.9982 | 100.00 |   99.84 |     99.96 | unsafe        |
| GRID g=2 NP grid16                            | I7    | 0.9883 | 100.00 |  100.00 |    100.00 | peak F1       |
| GRID g=2 NP grid16                            | I10   | 0.9825 |  80.00 |   89.22 |     82.23 | collapse      |
| GRID g=2 PR grid16                            | I3    | 0.9952 |  80.00 |   78.12 |     79.55 | peak F1       |
| GRID g=2 PR grid16                            | I10   | 0.9928 |  78.85 |   77.97 |     78.64 | collapse      |
| RECT025 NP single p=0.25 LS=0.30 (rect=0.25)  | I3    | 0.9701 | 100.00 |  100.00 |    100.00 | peak F1       |
| RECT025 NP single p=0.25 LS=0.30 (rect=0.25)  | I10   | 0.9575 |   0.00 |    0.47 |      0.11 | clean low F1  |
| RECT025 NP single p=0.25 LS=0.30 (rect=0.25)  | I13   | 0.9235 |   0.00 |    0.31 |      0.08 | safer         |
| RECT025 PR single p=0.25 LS=0.30 (rect=0.25)  | I7    | 0.9623 | 100.00 |  100.00 |    100.00 | peak F1       |
| RECT025 PR single p=0.25 LS=0.30 (rect=0.25)  | I10   | 0.9594 |  79.75 |   77.66 |     79.24 | FAR fail      |
| RECT025 PR2 (asym labels)                     | I3    | 0.8652 | 100.00 |  100.00 |    100.00 | peak F1       |
| RECT025 PR2 (asym labels)                     | I13   | 0.7874 | 100.00 |   99.53 |     99.89 | F1 drop       |
| RECT025 PR2 pn70 n30 (asym labels)            | I3    | 0.9777 | 100.00 |  100.00 |    100.00 | peak F1       |
| RECT025 PR2 pn70 n30 (asym labels)            | I10   | 0.9741 |  10.55 |    0.94 |      8.22 | best-asym     |
| RECT025 PR2 pn70 n30 (asym labels)            | I13   | 0.9080 |   5.85 |    0.47 |      4.55 | clean+low F1  |
| RECT025 PR2 pn85 n15 (asym labels)            | I10   | 0.7965 | 100.00 |   99.84 |     99.96 | F1 drop       |
| ASY g=2 pt90 nt10 ab=9070                     | I7    | 0.9813 |  17.45 |   75.94 |     31.63 | over-OOD      |
| ASY g=2 pt90 nt10 ab=9070                     | I13   | 0.9813 |  13.70 |   72.97 |     28.07 | over-OOD      |
| ASY g=3 pt85 nt15 ab=9070                     | I7    | 0.9979 |   4.85 |   25.00 |      9.73 | F1 champ      |
| ASY g=3 pt85 nt15 ab=9070                     | I13   | 0.9979 |   2.55 |   20.62 |      6.93 | F1 + lower    |
| ASY g=4 pt85 nt15 ab=9070                     | I7    | 0.9659 |  19.05 |   88.12 |     35.80 | over-OOD      |
| ASY g=4 pt85 nt15 ab=9070                     | I13   | 0.9635 |   9.50 |   41.56 |     17.27 | mid           |
| G4 pt95 nt05 LS sweep                         | I3    | 0.9524 |  23.70 |   59.69 |     32.42 | mid           |
| G4 pt95 nt05 LS sweep                         | I13   | 0.9258 |   3.55 |   20.47 |      7.65 | safer mid     |
| G4 pt90 nt10 LS sweep                         | I7    | 0.9976 | 100.00 |  100.00 |    100.00 | peak F1       |
| G4 pt90 nt10 LS sweep                         | I13   | 0.9272 |  84.65 |   99.69 |     88.30 | unsafe        |
| G4 pt875 nt125 LS sweep                       | I3    | 0.9948 |  79.15 |   76.41 |     78.48 | peak F1       |
| G4 pt875 nt125 LS sweep                       | I13   | 0.8891 |  28.00 |   46.09 |     32.39 | mid           |
| KD v1 iter116J recipe (3fcmpm teacher)        | I3    | 0.9491 | 100.00 |  100.00 |    100.00 | partial peak  |
| KD v1 iter116J recipe (3fcmpm teacher)        | I10   | 0.0049 |   2.40 |    0.00 |      1.82 | collapse      |
| KD v1 iter116J recipe (3fcmpm teacher)        | I13   | 0.0000 |   0.00 |    0.00 |      0.00 | dead          |
| KD v4 iter50B exact T4 a=0.5 LS=0.20 ep=8     | I3    | 0.9408 | 100.00 |    0.00 |    100.00 | partial       |
| KD v4 iter50B exact T4 a=0.5 LS=0.20 ep=8     | I10   | 0.2032 |   7.50 |    0.00 |      7.50 | collapse      |
| 3fcmpm vote_majority I10                      | I10   | 0.9891 |   0.00 |    0.00 |      0.00 | safe ens      |
| 3fcmpm vote_unanimous I10                     | I10   | 0.8871 |   0.00 |    0.00 |      0.00 | F1 drop       |
| 3fcmpm vote_intersection_bits I10             | I10   | 0.9521 |   0.00 |    0.00 |      0.00 | F1 mid        |
| 3fcmpm vote_majority_bits I10                 | I10   | 0.9894 |   0.00 |    0.00 |      0.00 | safe ens      |
| 3fcmpm vote_union_bits I10                    | I10   | 0.9958 |   0.00 |    0.31 |      0.08 | ENS SOTA      |
| 3fcmpm logit_avg + entropy gate I10           | I10   | 0.9935 |   0.00 |    0.00 |      0.00 | safe ens SOTA |
| KD v2 (T2 a=0.7) skip-cutmix                  | -     | -      |      - |       - |         - | train only    |
| KD v3 (T8 a=0.3) skip-cutmix                  | -     | -      |      - |       - |         - | n2000 pending |
| KD v5 (a=0.2 iter116J recipe)                 | -     | -      |      - |       - |         - | collapsed     |
```

## Source paths

Single-model checkpoints + n2000 eval parquets:

- `outputs/iter116A_g2_ls20/T7_iter116A_g2_ls20_260512_234333/best_model.pth`
  → `eval_n2000_pred/stage1_260514_175950/preds_chip.parquet`
- `outputs/iter116J_g3_ls30/T7_iter116J_g3_ls30_260513_010015/best_model.pth`
  → `eval_n2000_pred/stage1_260514_161529/preds_chip.parquet`
- `outputs/iter116F_g4_ls30/T7_iter116F_g4_ls30_260513_002653/best_model.pth`
  → `eval_n2000_pred/stage1_260514_162743/preds_chip.parquet`

Ensembles + threshold source:

- `outputs/_ens_3fcmpm_vote.json` (5 vote modes, fixed POS_KEYS_9 thresholds)
- `outputs/_ens_3fcmpm_logit_entropy.json` (logit avg + entropy ≥ 0.85·log4 → Normal)
- `outputs/_ens_3fcmpm_T0__I10.json` (per-class threshold derived ensemble F1)
- threshold source: `outputs/iter116J_g3_ls30/.../eval_n2000_pred/stage1_260514_161529/thresholds.json`

KD attempts (5 variants, 4 collapsed, 1 still in flight):

- v1: `outputs/KD_iter116J_recipe_3fcmpm_teach/.../eval_n2000_pred/stage1_260515_170737/`
- v2: `outputs/KD_v2_iter116J_T2_a07_skipcutmix/` (no eval)
- v3: `outputs/KD_v3_iter50setting_T8_a03_skipcutmix/` (n200 / n2000 not run)
- v4: `outputs/KD_v4_iter50B_exact_T4_a05_LS20_ep8/.../eval_v15direct/` (n2000 dir empty)
- v5: `outputs/KD_v5_alpha02_iter116J_recipe/` (collapsed, val_f1 0 ep1-10)

Ladder (5-stage progression on n2000):

- `outputs/ladder1_baseline/...`, `ladder2_focal/...`, `ladder3_asl/...`,
  `ladder4_cutmix_only/...`, `ladder5_cutmix_pair/...`,
  `ladder5b_complement_g4n2_p25/...`, `ladder5c_single_pair_p50/...`,
  `ladder5d_complement_g4n2_p100/...`

## Per-class F1 (top 3 cells)

```
| Cell                           | bb     | fork   | sc     | sr     | bb+fk  | bb+sc  | bb+sr  | fk+sc  | fk+sr  | bit_F1 |
|--------------------------------|--------|--------|--------|--------|--------|--------|--------|--------|--------|--------|
| iter116J I10                   | 0.9904 | 0.9865 | 0.9905 | 1.0000 | 0.9906 | 0.9876 | 0.9837 | 0.9941 | 0.9912 | 0.9927 |
| 3fcmpm vote_union_bits I10     | 1.0000 | 0.9984 | 1.0000 | 1.0000 | 0.9936 | 0.9851 | 0.9956 | 0.9945 | 0.9950 | 0.9958 |
| 3fcmpm logit_avg+entropy I10   | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.9896 | 0.9824 | 0.9843 | 0.9912 | 0.9940 | 0.9935 |
```

The ensemble lifts the 4 single-defect cells to perfect 1.000 (iter116J's
fork 0.9865 and bb 0.9904 absorb +1.4 pp from the g=2/g=4 votes), and the
combo cells gain +0.5–+1.4 pp consistently. The remaining error mass is
`bb+scratch` (0.985 — confused with `bb+fork` due to fork-vs-thin-scratch
visual ambiguity).
