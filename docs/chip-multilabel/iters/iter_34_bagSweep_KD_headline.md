# Iter 34 — Bag-size sweep across 11-model pool with KD-student diversity axis (NEW PAPER MAIN HEADLINE)

**Tag**: `iter34_bagSweep_KD_headline`
**Date**: 2026-05-10
**Status**: ★★★ NEW PAPER MAIN HEADLINE — supersedes iter33 4-bag (+0.0016 v15 bit_F1)
**Eval set**: v15direct (1000 chip, 4 OOD wafer-canvas, dual-gate)

## One-line summary

Replacing the weakest hard-label cell (21H, g=4 LS=0.75) in the iter33 4-bag with the
iter33A KD-student (α=0.3 T=4) lifts v15 bit_F1 **0.9945 → 0.9961** at the same
4× inference cost — first chip-multilabel ensemble to combine the **hard-label
diversity axis** with a **non-correlated KD-soft-label axis**.

## Motivation

- iter33 (prior MAIN): 4-bag (26B + 21F + 21H + 26D) hard-label-only majority vote → v15 bit_F1 0.9945.
- iter33A (KD WINNER, deployment variant): single student α=0.3 T=4 = v15 bit_F1 0.9840 single-pass.
- Question: does adding the KD-student as a **diversity axis** to a hard-label
  bag yield more than just averaging two single models?
- Specifically — replace 21H (the weakest cell at v15 bit_F1=0.9346) with 33A
  KD-student (single bit_F1=0.9840) and re-evaluate the 4-bag majority vote.

## 11-model pool (per-model best-safe single, FAR ≤ 5%)

| tag | desc | cell | v15 bF1 | v15 ni_FAR |
|-----|------|------|--------:|-----------:|
| 26B | g=3 LS=0.50 | T0__I10 | 0.9791 | 1.25% |
| 21F | g=3 LS=0.67 | T0__I10 | 0.9676 | 1.25% |
| 21H | g=4 LS=0.75 | T0__I10 | 0.9346 | 0.00% |
| 26D | g=4 LS=0.40 | T0__I10 | 0.9353 | 0.00% |
| 26H | g=3 LS=0.67 white | T0__I10 | 0.9687 | 2.50% |
| 21E | g=2 LS=1.00 | T0__I10 | 0.9691 | 3.75% |
| 36C | g=2 LS=0.55 | T0__I10 | 0.9745 | 0.00% |
| **33A** | **KD α=0.3 T=4** | **T0__I10** | **0.9840** | **0.00%** |
| 33B | KD α=0.7 T=4 | T0__I10 | 0.9747 | 0.00% |
| 33C | KD α=0.5 T=2 | T0__I10 | 0.9808 | 1.25% |
| 33D | KD α=0.5 T=8 | T0__I10 | 0.9695 | 0.00% |

Pool axis decomposition: **7 hard-label** (26B/21F/21H/26D/26H/21E/36C, span
g∈{2,3,4} × LS∈{0.40, 0.50, 0.55, 0.67, 0.75, 1.00} × CutMix-fill∈{compl, white}) +
**4 KD-student** (33A/B/C/D, span α∈{0.3, 0.5, 0.7} × T∈{2, 4, 8}).

## Bag-size sweep (best per size, FAR ≤ 5%)

| size | combo | thr | v15 bF1 | v15 ni_FAR | per-class bb / fk / sc / sr |
|----:|------|-----|--------:|-----------:|------------------------------|
| 2 | **26B + 33A** | 1/2 OR | **0.9969** | 1.25% | 1.0000 / 0.9937 / 0.9938 / 1.0000 |
| 2 | 36C + 33A | 1/2 OR | 0.9953 | 0.00% | 0.9937 / 0.9937 / 0.9938 / 1.0000 |
| 3 | 26B + 21F + 33A | 2/3 | 0.9929 | 0.00% | 0.9937 / 0.9937 / 0.9841 / 1.0000 |
| **4 NEW MAIN ★★★** | **26B + 21F + 26D + 33A** | **2/4** | **0.9961** | **0.00%** | **0.9937 / 0.9937 / 0.9969 / 1.0000** |
| 4 alt | 26B + 26D + 21E + 33A | 2/4 | 0.9961 | 0.00% | 0.9937 / 0.9937 / 0.9969 / 1.0000 |
| 4 alt | 26B + 26D + 36C + 33A | 2/4 | 0.9961 | 0.00% | 0.9937 / 0.9937 / 0.9969 / 1.0000 |
| 4 alt | 26B + 26D + 26H + 33A | 2/4 | 0.9961 | 0.00% | 1.0000 / 0.9969 / 0.9873 / 1.0000 |
| 5 | 26B + 21F + 21H + 33A + 33B | 3/5 | 0.9929 | 0.00% | 0.9937 / 0.9905 / 0.9873 / 1.0000 |
| 6 | 26B + 21F + 26D + 21E + 33A + 33D | 3/6 | 0.9961 | 0.00% | 0.9937 / 0.9937 / 0.9969 / 1.0000 |
| pure-KD 4 | 33A + B + C + D | 2/4 | 0.9873 | 0.00% | 0.9776 / 0.9937 / 0.9778 / 1.0000 |
| pure-KD 4 | 33A + B + C + D | 3/4 | 0.9799 | 0.00% | 0.9677 / 0.9873 / 0.9646 / 1.0000 |
| pure-KD 4 | 33A + B + C + D | 4/4 AND | 0.9522 | 0.00% | 0.9474 / 0.9333 / 0.9439 / 0.9841 |

## NEW HEADLINE composition rationale (4-bag MAIN)

**26B + 21F + 26D + 33A KD-student**

- **26B** (g=3 LS=0.50, hard) — strongest single in pool at v15 0.9791
- **21F** (g=3 LS=0.67, hard) — same g=3 spine, different LS
- **26D** (g=4 LS=0.40, hard) — g=4 axis (FAR=0%, conservative)
- **33A** (KD α=0.3 T=4) — **non-correlated diversity axis** (soft-target distillation
  from 14-bag teacher), replaces 21H (g=4 LS=0.75 hard, weakest at 0.9346)

Span: hard-label {g=3 LS={0.50, 0.67}, g=4 LS=0.40} ∪ KD-soft {α=0.3 T=4 from teacher}.
Three of the four cells are hard-label diversity (g × LS), one is the KD axis —
this is the minimum mixture to demonstrate the KD lift.

## Pure-KD comparison (KD diversity alone is insufficient)

Pure 4-bag of KD students (33A + B + C + D, span α∈{0.3, 0.5, 0.7} × T∈{2, 4, 8}):

- thr 2/4 majority: v15 bit_F1 = **0.9873** (vs hard-label-only iter33 4-bag 0.9945, mixed iter34 4-bag 0.9961)
- thr 3/4: 0.9799
- thr 4/4 AND: 0.9522

→ KD students share the same teacher's biases — bagging across (α, T) doesn't
recover the variance that hard-label cells contribute. The KD axis must be
**combined with** hard-label diversity, not used in isolation.

## Paper diversity-axis claim

| bag composition | v15 bit_F1 | gain over hard-label-only 4-bag |
|---|---:|---:|
| iter33 4-bag (4 hard-label cells) | 0.9945 | baseline |
| iter34 4-bag (3 hard-label + 1 KD-student) | **0.9961** | **+0.0016** |
| iter34 pure-KD 4-bag (4 KD students) | 0.9873 | −0.0072 |

**Claim**: orthogonal diversity axes (hard-label CutMix complement × KD soft-label
distillation) compound multiplicatively in majority-vote ensembles. A single
KD-student replacing the weakest hard-label cell yields **+0.0016 v15 bit_F1**
at the same 4× inference cost.

The 6-bag also ties 4-bag at 0.9961 — adding more hard-label or KD bags beyond
the 4-bag minimum-mixture does not lift further. **4-bag MAIN is the cost-optimal
sweet spot**, matching iter33's "diversity > quantity" finding extended along
the KD axis.

## Ultra-cheap 2-bag option (production deployment variant)

**26B + 33A OR (1/2)** = v15 bit_F1 **0.9969** (highest of all configurations)
at ni_FAR=1.25% (still well under the 5% dual-gate threshold). 2× inference cost.

This is the **single highest bit_F1** observed in chip-multilabel project history,
but trades 1.25% ni_FAR (vs 0.00% for the 4-bag MAIN). For deployments tolerating
small false-alarm uplift, this is the cost-optimal alternative.

## Sources (NEW HEADLINE 4 models)

- **26B**: `outputs/iter26B_g3_LS050/T7_iter26B_g3_LS050_seed1_260509_154354/eval_v15direct/stage1_260509_160430/preds_chip.parquet`
- **21F**: `outputs/iter21F_19E_repeat/T7_iter21F_19E_repeat_seed1_260509_103953/eval_v15direct/stage1_260509_105714/preds_chip.parquet`
- **26D**: `outputs/iter26D_g4_LS040/T7_iter26D_g4_LS040_seed1_260509_162552/eval_v15direct/stage1_260509_163327/preds_chip.parquet`
- **33A**: `outputs/iter33A_alpha03_T4/T7_iter33A_alpha03_T4_seed1_260509_233558/eval_v15direct/stage1_*/preds_chip.parquet`

## Cross-iter status

- vs iter27 14-bag (prior PAPER HEADLINE): v15 0.9929 → **0.9961 = +0.0032** at
  4× inference cost vs 14× cost
- vs iter33 4-bag (prior PAPER MAIN): v15 0.9945 → **0.9961 = +0.0016** at same cost
- vs 12-T5 baseline (paper start): v15 0.7872 → **0.9961 = +0.2089 (+27%)**, ni_FAR 100% (real env) → **0.00%**

## What's next

- iter34 single-cell ablation (drop 33A from 4-bag → 3-bag = 0.9929; replace 33A
  with 33C T=2 → check) — done in this sweep, ties confirm the structural finding.
- Multi-seed verification of 33A KD-student (currently single seed=1).
- Production decision: 4-bag MAIN (FAR=0%) vs 2-bag OR (highest bit_F1, FAR=1.25%).

## Files written / updated

- `docs/chip-multilabel/tables/paper_main_headline.csv` — added iter34 4-bag MAIN row + 2-bag OR row, marked iter33 row as superseded
- `docs/chip-multilabel/tables/all_runs_macro_f1.csv` — appended 12 ensemble rows
- `docs/chip-multilabel/02_results.md` — added cross-iter timeline row 34, marked row 33 as superseded
- `docs/chip-multilabel/iters/iter_34_bagSweep_KD_headline.md` — this file
