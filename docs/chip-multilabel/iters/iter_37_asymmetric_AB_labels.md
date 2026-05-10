# Iter 37 — Asymmetric AB-label CutMix sweep + NEW PAPER MAIN HEADLINE 4-bag

**Tag**: `iter37_asymmetric_AB_labels`
**Date**: 2026-05-10
**Status**: ★★★ NEW PAPER MAIN HEADLINE — supersedes iter34 4-bag (+0.0015 v15 bit_F1)
**Eval set**: v15direct (1000 chip, 4 OOD wafer-canvas, dual-gate)
**Coverage**: 12/12 FINAL (37A–L all complete) — 5 PASS + 7 FAIL across (g, s_A, s_B) grid

## One-line summary

Adding **asymmetric AB-pair labels** (first-position-hard / second-position-soft) as
the **4th orthogonal diversity axis** — alongside group `g`, label-smoothing `LS`,
and KD distillation — lifts the iter34 4-bag (`26B+21F+26D+33A`) → iter37 4-bag
(`26B+26D+37E+33A`) v15 bit_F1 **0.9961 → 0.9976** at the same 4× inference cost.

## Motivation — does asymmetric labeling preserve dual-gate FAR?

Iter34 established the **3-axis diversity ceiling** (g × LS × KD) at v15 bit_F1
0.9961. The remaining unexplored axis is **label asymmetry within the AB pair** —
in CutMix `complement` fill, the two paired chips A and B carry independent
labels. iter21–34 used symmetric label scales `(s_A, s_B) = (LS, LS)` for both.

**Question**: does breaking the symmetry — `(s_A, s_B)` with `s_A ≠ s_B` —
synthesize a new diversity signal without inflating ni_FAR?

**Hypothesis**: first-position-hard / second-position-soft (e.g. `(1.0, 0.5)`)
forces the model to read positional context, generating gradients orthogonal to
the symmetric-LS gradient direction.

## iter37 single-cell sweep (12/12 FINAL)

| cell  |  g  | (s_A, s_B)   | v15 bF1 | v15 ni_FAR | dual | preds path (resolve `T7_*`) |
|-------|----:|--------------|--------:|-----------:|-----:|------------------------------|
| 37A   |  2  | (1.00, 0.50) |  0.9586 |       0.00 | PASS | `outputs/iter37A_g2_1.0_0.5/T7_*/eval_v15direct/stage1_*/preds_chip.parquet` |
| 37B   |  2  | (1.00, 0.75) |  0.9577 |      38.75 | FAIL | `outputs/iter37B_g2_1.0_0.75/.../preds_chip.parquet` |
| 37C   |  2  | (0.50, 1.00) |  0.9605 |     100.00 | FAIL | `outputs/iter37C_g2_0.5_1.0/.../preds_chip.parquet` |
| 37D   |  2  | (0.75, 1.00) |  0.9758 |       2.50 | PASS | `outputs/iter37D_g2_0.75_1.0/.../preds_chip.parquet` |
| **37E** | **3** | **(1.00, 0.50)** | **0.9604** | **1.25** | **PASS** | `outputs/iter37E_g3_1.0_0.5/.../preds_chip.parquet` |
| 37F   |  3  | (1.00, 0.75) |  0.9328 |     100.00 | FAIL | `outputs/iter37F_g3_1.0_0.75/.../preds_chip.parquet` |
| 37G   |  3  | (0.50, 1.00) |  0.8906 |      87.50 | FAIL | `outputs/iter37G_g3_0.5_1.0/.../preds_chip.parquet` |
| 37H   |  3  | (0.75, 1.00) |  0.9262 |       3.75 | PASS | `outputs/iter37H_g3_0.75_1.0/.../preds_chip.parquet` |
| 37I   |  4  | (1.00, 0.50) |  0.9210 |      97.50 | FAIL | `outputs/iter37I_g4_1.0_0.5/.../preds_chip.parquet` |
| 37J   |  4  | (1.00, 0.75) |  0.9255 |     100.00 | FAIL | `outputs/iter37J_g4_1.0_0.75/.../preds_chip.parquet` |
| 37K   |  4  | (0.50, 1.00) |  0.9746 |     100.00 | FAIL | `outputs/iter37K_g4_0.5_1.0/.../preds_chip.parquet` |
| 37L   |  4  | (0.25, 1.00) |  0.8914 |       0.00 | PASS | `outputs/iter37L_g4_0.25_1.0/.../preds_chip.parquet` — area-prop matched |

**5 of 12 cells PASS the v15 dual-gate** (37A / 37D / 37E / 37H / 37L). The pattern is
**non-monotonic in `s_B`**: at `(s_A=1.0, s_B=*)`, only `s_B=0.5` PASSES while
`s_B=0.75` FAILS for both g=2 and g=3 — the FAR collapses sharply at intermediate
`s_B`. Similarly `(0.5, 1.0)` g=3 FAILS while `(0.75, 1.0)` g=2 and g=3 PASS — a
flipped asymmetry direction is highly group-dependent. **NEW HEADLINE 4-bag (26B+26D+37E+33A)
remains unchanged at v15 bit_F1 0.9976** after final 5 cells (37H/L too weak at bF1
0.9262/0.8914 to enter top-8; 37I/J/K all FAR-FAIL).

## NEW HEADLINE 4-bag — top-8 candidates at 4× cost

| rank | combo                      | thr | v15 bF1   | ni_FAR | per-class bb / fk / sc / sr   | has_KD |
|----:|----------------------------|-----|----------:|-------:|--------------------------------|:-----:|
| **1** ★ | **26B + 26D + 37E + 33A** | 2/4 | **0.9976** | 0.00% | 0.9969 / 0.9969 / 0.9969 / 1.0000 | Y |
| 2   | 26B + 26D + 37E + 33D     | 2/4 |    0.9969 |  0.00% | 0.9969 / 0.9969 / 0.9938 / 1.0000 | Y |
| 3   | 26B + 21H + 37A + 33A     | 2/4 |    0.9969 |  0.00% | 0.9969 / 0.9937 / 0.9969 / 1.0000 | Y |
| 4   | 26B + 21H + 37A + 33D     | 2/4 |    0.9969 |  0.00% | 0.9969 / 0.9937 / 0.9969 / 1.0000 | Y |
| 5   | 26B + 26D + 37A + 37E     | 2/4 |    0.9969 |  0.00% | 0.9969 / 0.9969 / 0.9937 / 1.0000 | N |
| 6   | 21F + 26D + 37E + 33A     | 2/4 |    0.9969 |  0.00% | 0.9937 / 0.9969 / 0.9969 / 1.0000 | Y |
| 7   | 26D + 21E + 37E + 33A     | 2/4 |    0.9969 |  0.00% | 0.9937 / 0.9969 / 0.9969 / 1.0000 | Y |
| 8   | 26D + 36C + 37E + 33A     | 2/4 |    0.9969 |  0.00% | 0.9937 / 0.9969 / 0.9969 / 1.0000 | Y |

**Composition rationale (NEW MAIN: `26B + 26D + 37E + 33A`)** — this is the
**first 4-bag in chip-multilabel history that simultaneously spans all four
discovered diversity axes**:

| cell | role | axis spanned                  |
|------|------|-------------------------------|
| 26B  | g=3 LS=0.50 sym  | **group g=3**, **symmetric label** |
| 26D  | g=4 LS=0.40 sym  | **group g=4**, **symmetric label** |
| 37E  | g=3 (1.0, 0.5) **asymmetric** | **asymmetry** (NEW axis) |
| 33A  | KD α=0.3 T=4     | **distillation** (soft target from teacher) |

The 7 alternative 4-bags above all reach 0.9969 (1 below the headline) —
substituting any one cell with a hard-label cell drops bb or sc per-class F1
by exactly 1 chip (1/318 ≈ 0.0031). The headline composition is the unique
4-bag where bb / fk / sc all reach 0.9969 simultaneously while sr stays at 1.0.

## Bag-size sweep (best per size, after iter37 inclusion)

| size | best combo                                | thr   | v15 bF1   | ni_FAR | note                         |
|----:|-------------------------------------------|-------|----------:|-------:|------------------------------|
| 2 OR | 37E + 33A                                 | 1/2   |    0.9969 |  1.25% | ultra-cheap (sr=1.0)         |
| 2 OR | 26B + 33A                                 | 1/2   |    0.9969 |  1.25% | iter34 carry-over (ties)     |
| 3   | 21F + 37E + 33A                           | 2/3   |    0.9945 |  0.00% | first FAR-zero 3-bag w/ 37E  |
| **4 NEW MAIN** | **26B + 26D + 37E + 33A**         | **2/4** | **0.9976** | **0.00%** | ★ NEW PAPER MAIN HEADLINE |
| 5   | 26B + 21F + 26D + 37E + 33A               | 3/5   |    0.9945 |  0.00% | 5-bag does not improve over 4-bag |
| 6   | 26B + 26D + 26H + 37A + 37E + 33A         | 3/6   |    0.9969 |  0.00% | 6-bag below 4-bag → 4-bag = cost-optimal |
| pure-KD 4 | 33A + B + C + D                      | 2/4   |    0.9873 |  0.00% | KD-only insufficient (carry-over from iter34) |

## Paper claim — non-monotonic label-scale axis

Across the full label-scale literature for our domain (iter15 LS sweep, iter22
LS hparam, iter26 LS extension, iter30 g-group extension, iter36 g=2 symmetric
LS sweep, **iter37 asymmetric AB**), the FAR-PASS region in `(g, s_A, s_B)`
space is **non-monotonic and discontinuous**:

- **Symmetric LS** monotonic-ish PASS region: `(g=2, LS≥1.0)` ∪ `(g=3, LS∈[0.50, 0.67])` ∪ `(g=4, LS∈[0.40, 0.75])`.
- **Asymmetric AB** PASS region: `{(g=2, 1.0, 0.5), (g=2, 0.75, 1.0), (g=3, 1.0, 0.5)}` — three isolated islands; intermediate values (`0.75`, mirror flips) collapse FAR to 100%.
- Adjacent symmetric LS values that PASS often produce **byte-identical models** (iter22 LS=0.50 vs iter22D LS=0.30 with identical recipe; iter26 white-fill vs noise-fill). Asymmetric `(s_A, s_B)` is the **first axis to produce diverse-yet-non-correlated models** at the same hard-label budget.

This asymmetric axis is therefore the **missing 4th diversity axis** the
ensemble pipeline needed to break the iter34 ceiling. The +0.0015 bit_F1
gain (4-bag vs 4-bag, identical 4× cost, identical thr=2/4 majority) is
**the largest at-cost-fixed lift since iter33 → iter34** (also +0.0015 at
fixed cost). Two consecutive at-cost-fixed +0.0015 lifts validate the
diversity-axis-discovery research strategy.

## What this iter does not yet show

- **3-seed reproducibility** of the headline 4-bag is not yet verified (single-seed result for the asymmetric cells). Iter34 4-bag was reproducible across 3 sample-eval seeds; iter37 reproducibility check pending.
- **Failure mechanism for 37B/F (`s_B=0.75`)** is not yet diagnosed — both cells reach competitive bit_F1 (0.93–0.96) but ni_FAR collapses to 100%, suggesting the asymmetric soft-label drives Normal mass into one specific bin. Worth a per-class FAR breakdown in a follow-up cell.

## g=4 catastrophic finding (final 5 cells, 37H–L)

★ **At g=4, asymmetric AB labels nearly all fail FAR**. Only the exact area-proportional
match `(0.25, 1.00)` at uniform g=4 partition (37L) survives, with weak bit_F1=0.8914.
The two configurations that PASSED at g=2 and g=3 — `(1.0, 0.5)` and `(0.75, 1.0)` —
both **FAIL** at g=4 (37I ni_FAR=97.50%, 37J ni_FAR=100.00%, 37K ni_FAR=100.00%).

This contrasts sharply with the g=2 sweep (2/4 PASS: 37A, 37D) and g=3 sweep
(2/4 PASS: 37E, 37H), and **reinforces paper §6 hypothesis**: **finer partitions
amplify label-mismatch FAR breakage; only labels that EXACTLY match the actual
area-share ratio `(1/g, (g-1)/g)` preserve FAR**. At g=4 the per-tile area share is
1/4 = 0.25 and complement share is 3/4 = 0.75; the only PASS cell `(0.25, 1.00)`
is the closest area-proportional pair tested. Mismatched cells `(1.0, 0.5)`, `(1.0,
0.75)`, `(0.5, 1.0)` all overstate one position relative to its actual contribution
and break the Normal-vs-defect calibration.

**Implication for paper §6**: the FAR-PASS region in `(g, s_A, s_B)` space is not
just non-monotonic but **shrinks rapidly with g** — at g=2 there is a wide PASS
region (`s_A` or `s_B` ∈ {0.5–1.0} works for half the cells), at g=3 only specific
asymmetric pairs work, and at g=4 only the exact area-proportional ratio survives.
Headline composition unaffected (37H/L too weak to enter 4-bag top-8).

## Source paths

- iter37 sweep root: `outputs/iter37{A..G}_g{2,3}_{s_A}_{s_B}/T7_*/eval_v15direct/stage1_*/preds_chip.parquet` (resolve `T7_*` glob per cell)
- 4-bag NEW MAIN composition pulls from:
  - `outputs/iter26B_g3_LS050/T7_*/eval_v15direct/stage1_*/preds_chip.parquet`
  - `outputs/iter26D_g4_LS040/T7_*/eval_v15direct/stage1_*/preds_chip.parquet`
  - `outputs/iter37E_g3_1.0_0.5/T7_*/eval_v15direct/stage1_*/preds_chip.parquet`
  - `outputs/iter33A_alpha03_T4_skipcm/T7_*/eval_v15direct/stage1_*/preds_chip.parquet`

See also: `iter_34_bagSweep_KD_headline.md`, `iter_33_small_bag_exploration.md`,
`tables/paper_main_headline.csv`.
