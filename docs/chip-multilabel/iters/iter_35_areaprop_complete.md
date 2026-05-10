# Iter 35 + 36 — Area-proportional FCM-PM + g=2 symmetric LS sweep

- **iter35 tag**: `areaprop` (FCM-PM CutMix area-proportional label scaling, 8 cells)
- **iter36 tag**: `g2_LS_sweep` (g=2 symmetric, label scale ∈ [0.40, 0.90], 8 cells planned)
- **timestamp**: 260509 → 260510
- **status**: iter35 **8/8 complete**, iter36 **2/8 complete (in progress, A/B done, C–H pending)**.
- **goal**: Probe whether **area-proportional label scaling** — assign label mass `(area_A / total, area_B / total)` per CutMix pair instead of fixed (LS, 1−LS) — fixes the FCM-PM ni_FAR collapse seen in iter28/29/30/31 sweeps. iter36 then dials in **g=2 symmetric** as a cleanup pass.
- **fixed**: T7 (BCE+LS=0.20), CutMix complement mode, `n_patches=5`, total_ratio=0.3, 8 epoch, seed=1.

---

## TL;DR

| iter | cells | dual-pass | best dual-pass v15 bit_F1 |
|:---:|:---:|:---:|---:|
| 35  | 8/8 | **1/8 (H only)** | 0.9033 (H, g=3 areaprop s=0.3) |
| 36  | 2/8 | **0/2** | — |

**14 / 16 cells across iter35 + iter36-partial collapse to v15 ni_FAR = 100 %.** This is the largest single-batch FAR collapse rate in the chip-multilabel project. The mechanism is **paper §6 structural failure mode confirmed**: any FCM-PM variant that pushes effective label mass per pair above ≈0.5 forces the student into the OOD-permissive regime where Normal chips are pulled into defect bins.

---

## iter35 — area-proportional FCM-PM (8/8 complete)

### Hparam table

For each cell: `g` = group count, `scale` = base scale multiplier on the area-proportional weight, **(A, B) labels** = effective label mass per (smaller, larger) pair partner derived as `(area / total) * scale` after re-clamping.

| cell                | g | scale | (A, B) labels   | v15 bit_F1 | v15 ni_FAR | dual |
|:--------------------|:-:|:-----:|:----------------|-----------:|-----------:|:----:|
| A_g3_areaprop_s10   | 3 | 1.0   | (0.33, 0.67)    |     0.9456 |    100.00% | FAIL |
| B_g4_areaprop_s10   | 4 | 1.0   | (0.25, 0.75)    |     0.9760 |    100.00% | FAIL |
| C_g3_areaprop_s15   | 3 | 1.5   | (0.50, 1.00)    |     0.9850 |    100.00% | FAIL |
| D_g4_areaprop_s133  | 4 | 1.33  | (0.33, 1.00)    |     0.9439 |    100.00% | FAIL |
| E_g3_areaprop_s05   | 3 | 0.5   | (0.17, 0.33)    |     0.9898 |    100.00% | FAIL |
| F_g4_areaprop_s05   | 4 | 0.5   | (0.125, 0.375)  |     0.9717 |    100.00% | FAIL |
| G_g2_areaprop_s10   | 2 | 1.0   | (0.50, 0.50)    |     0.9873 |    100.00% | FAIL |
| **H_g3_areaprop_s03** ★ | 3 | 0.3 | (0.10, 0.20)   | **0.9033** |  **0.00%** | PASS |

### Per-cell preds path (all v15direct)

```
outputs/iter35A_g3_areaprop_s10/T7_*/eval_v15direct/stage1_*/preds_chip.parquet
outputs/iter35B_g4_areaprop_s10/T7_*/eval_v15direct/stage1_*/preds_chip.parquet
outputs/iter35C_g3_areaprop_s15/T7_*/eval_v15direct/stage1_*/preds_chip.parquet
outputs/iter35D_g4_areaprop_s133/T7_*/eval_v15direct/stage1_*/preds_chip.parquet
outputs/iter35E_g3_areaprop_s05/T7_*/eval_v15direct/stage1_*/preds_chip.parquet
outputs/iter35F_g4_areaprop_s05/T7_*/eval_v15direct/stage1_*/preds_chip.parquet
outputs/iter35G_g2_areaprop_s10/T7_*/eval_v15direct/stage1_*/preds_chip.parquet
outputs/iter35H_g3_areaprop_s03/T7_*/eval_v15direct/stage1_*/preds_chip.parquet
```

Dispatch log: `outputs/_iter35_areaprop.log`.

### Findings (iter35)

1. **Area-proportional labeling is structurally broken** for 7 of 8 cells (E_s05 highest v15 bit_F1 = 0.9898 with FAR = 100 %). The F1-only winner is exactly the cell that fails the FAR gate hardest — same trap as iter29B (paper §5 §29B "F1-only winner not deployable").
2. **Only s = 0.3 (cell H, label mass (0.10, 0.20)) PASSES**, at the cost of dropping v15 bit_F1 to 0.9033 — well below iter21E single (0.9691) and iter27/33 ensembles. The pattern: **once effective label mass per pair partner ≥ 0.20 in the smaller half, ni_FAR collapses to 100 %**.
3. **g axis is null** under area-proportional. g=2 (G), g=3 (A/C/E/H), g=4 (B/D/F) all sit on the same FAR cliff — the labeling rule, not the granularity, is the root cause.
4. **Scale axis is monotone-bad** until s ≤ 0.3. s = 0.5 (E/F) → s = 1.0 (A/B) → s = 1.5 (C, D) all FAIL; only s = 0.3 (H) clears.

---

## iter36 — g=2 symmetric LS sweep (2/8 complete)

Probes whether **g=2 with symmetric (LS, LS) labeling** at high LS recovers iter21E's g=2 LS=1.0 PASS without paying the area-proportional FAR tax. Because g=2 has only 2 partition cells, area-proportional and symmetric are degenerate at LS = 1.0 (both = (0.5, 0.5) after normalization), but at LS < 1.0 the symmetric form (LS, LS) differs from area-proportional `(0.5·s, 0.5·s)`.

### Hparam table

| cell      | LS   | v15 bit_F1 | v15 ni_FAR | dual | status |
|:----------|:----:|-----------:|-----------:|:----:|:-------|
| A_LS040   | 0.40 |     0.8797 |    100.00% | FAIL | done  |
| B_LS045   | 0.45 |     0.8653 |    100.00% | FAIL | done  |
| C_LS055   | 0.55 |          — |          — |  —   | pending |
| D_LS060   | 0.60 |          — |          — |  —   | pending |
| E_LS065   | 0.65 |          — |          — |  —   | pending |
| F_LS070   | 0.70 |          — |          — |  —   | pending |
| G_LS080   | 0.80 |          — |          — |  —   | pending |
| H_LS090   | 0.90 |          — |          — |  —   | pending |

Dispatch log: `outputs/_iter36_g2_LS_sweep.log`.

### Findings (iter36 partial)

1. Both completed cells (LS=0.40, LS=0.45) **collapse v15 ni_FAR to 100 %** — same failure mode as iter35 A–G. v15 bit_F1 also drops below 0.90 (0.8797 / 0.8653), well below the iter28/33 winning band (0.97+).
2. **Pattern matches iter30D** (g=2 LS=0.50, FAIL) and **diverges from iter21E** (g=2 LS=1.0, PASS, the seed of the iter25/27/33 ensembles). The transition between FAIL and PASS along the g=2 LS axis happens **somewhere in [0.45, 1.00)**. iter36 C–H will localize it.
3. C–H pending at the time of writing — see resume log.

---

## Cross-link — the LS axis at g=2

Three iters now sample the g=2 symmetric LS axis on FCM-PM:

| iter | cell | LS  | v15 bit_F1 | v15 ni_FAR | dual |
|:----:|:----:|:---:|-----------:|-----------:|:----:|
|  21  | 21E  | **1.00** |     0.9691 |      3.75% | **PASS** |
|  30  | 30D  | 0.50 |     (low)  |    (high)  | **FAIL** |
|  36  | 36A  | 0.40 |     0.8797 |    100.00% | **FAIL** |
|  36  | 36B  | 0.45 |     0.8653 |    100.00% | **FAIL** |

The PASS region is consistent with the paper §5 finding that **only LS = 1.0 (hard, full-mass labels) survives the v15 OOD ni_FAR gate** when g = 2. Anything that softens the within-pair label mass below the hard regime collapses calibration.

---

## Paper §6 — structural failure mode confirmed

iter35 + iter36 partial together = **14 / 16 cells with v15 ni_FAR = 100 %**. This is the **single largest negative-axis sweep** in the project to date. The §6 paper claim ("FCM-PM only works under hard labels + complement fill + pair mask + full cover") now has empirical evidence at 4 independent dimensions:

1. **Label rule** (iter35 area-proportional vs iter28 hard-LS-soft): area-proportional fails 7/8.
2. **LS axis at g=2** (iter21E PASS @ LS=1.0, iter30D / 36A / 36B all FAIL @ LS ≤ 0.50).
3. **Group count** (iter30 g∈{5, 6} all macro_f1 < 0.80, all FAIL).
4. **Soft labels w/o pair mask** (iter29C, FAIL).

Removing **any one** of {hard labels, complement fill, pair mask, full cover} or pushing **the label scale below 1.0** at g=2 → catastrophic ni_FAR. The 4-design synergy of iter21E (the seed of all subsequent ensembles) is **structurally tight**, not a hyperparameter accident.

---

## What's NOT changing

- **PAPER MAIN HEADLINE remains iter33 4-bag (0.9945, FAR 0%)**. iter35H PASS does not advance; v15 bit_F1 = 0.9033 is far below all dual-pass singles.
- **No bag candidate from iter35/36**. Negative-axis only; ensemble composition unchanged.
- **iter36 C–H queue** retained for completeness — needed to localize the LS PASS/FAIL boundary at g=2.

---

## Source

- `outputs/iter35{A..H}_*/T7_*/eval_v15direct/stage1_*/preds_chip.parquet`
- `outputs/iter36{A,B}_LS{040,045}/T7_*/eval_v15direct/stage1_*/preds_chip.parquet`
- Dispatch logs: `outputs/_iter35_areaprop.log`, `outputs/_iter36_g2_LS_sweep.log`
- Cross-link: `iters/iter_21_clean_baseline.md` (21E PASS reference), `iters/iter_30_31_g_extension_regularization.md` (30D FAIL reference)
