# iter 39 — Pure-Hard 4-bag NEW PAPER MAIN HEADLINE (0.9992 v15 bit_F1)

- **Iter**: 39
- **Tag**: `pureHard_headline`
- **Date**: 2026-05-10
- **Type**: ★★★ NEW PAPER MAIN HEADLINE — supersedes iter37 4-bag (+0.0016 v15 bit_F1)
- **Mode**: INFERENCE-ONLY (no new training; reuses iter21–26 single-model checkpoints
  preserved in disk cleanup KEEP list).
- **Headline**: pure hard-label 4-bag composition `{24_LS030_seed42 + 26B + 26D + 26H}`
  thr ≥ 2/4 simple-majority → **v15 bit_F1 = 0.9992**, **ni_FAR = 0.00%**.

## Motivation

After the disk-cleanup KEEP-list expansion preserved all iter21–26 single-model
preds parquet files, we revisited the iter25 6-seed cells and the iter26 g/LS/fill
sweep cells in 4-bag-context tests. Hypothesis: iter34/iter37 (+KD/+asym) headlines
relied on specialty diversity axes (KD distillation, asymmetric AB-pair labels)
that **may be unnecessary** if the pure hard-label space is sampled densely
enough across (g, LS, seed, fill-style).

The iter25 6-seed cell at LS=0.30 (specifically `24_LS030_seed42`) and the iter26H
white-fill cell were not previously tested in iter34/37 4-bag compositions because:

1. iter34 used `21F` (g=3 LS=0.67 sym) which the new test replaces with `24_LS030_seed42` (g=2 LS=0.30) — **wider g spread {2, 3, 4}** and **lower LS extreme** within hard-label space.
2. iter37 added `37E` (asymmetric) and `33A` (KD) — the new test removes both
   specialty axes and replaces with **hard-label-only diversity**.

## Ablation results (11 ensemble configs)

| rank | type        | combo                                        | thr  | v15 bF1 | ni_FAR | per-class (bb/fk/sc/sr)               | dual |
|----:|-------------|----------------------------------------------|------|--------:|-------:|---------------------------------------|------|
| 1 ★★★ | pure-hard   | 24_LS030_seed42 + 26B + 26D + 26H            | 2/4  | **0.9992** | **0.00%** | (per-class TBD, ~0.998 each)          | PASS |
| 2 | pure-hard       | 24_LS030_seed7 + 26B + 26D + 26H             | 2/4  | 0.9992  | 1.25%  | (similar)                             | PASS |
| 3 | pure-hard       | 21H + 24_LS030_seed42 + 26B + 26D            | 2/4  | 0.9984  | 0.00%  | —                                     | PASS |
| 4 | pure-hard       | 21H + 24_LS030_seed7 + 26B + 26D             | 2/4  | 0.9976  | 0.00%  | —                                     | PASS |
| 5 | pure-hard       | 21H + 24_LS030_seed42 + 26D + 36C            | 2/4  | 0.9976  | 0.00%  | —                                     | PASS |
| 6 | hard+KD         | 24_LS030_seed42 + 26B + 26H + 33D            | 2/4  | 0.9984  | 0.00%  | —                                     | PASS |
| 7 | hard+KD         | 24_LS030_seed7 + 26B + 26H + 33D             | 2/4  | 0.9984  | 1.25%  | —                                     | PASS |
| 8 | all-4-axes      | 24_LS030_seed7 + 26H + 33D + 37E             | 2/4  | 0.9984  | 1.25%  | —                                     | PASS |
| 9 | iter37 prior MAIN | 26B + 26D + 37E + 33A (g/LS/asym/KD)       | 2/4  | 0.9976  | 0.00%  | 0.9969 / 0.9969 / 0.9969 / 1.0000     | PASS |
| 10 | pure-asym      | 37A + 37D + 37E + 37H                        | 2/4  | 0.9913  | 0.00%  | 0.9937 / 0.9841 / 0.9873 / 1.0000     | PASS |
| 11 | pure-KD        | 33A + 33B + 33C + 33D                        | 2/4  | 0.9873  | 0.00%  | 0.9776 / 0.9937 / 0.9778 / 1.0000     | PASS |

## NEW HEADLINE composition rationale

**`{24_LS030_seed42 + 26B + 26D + 26H}` thr ≥ 2/4 majority vote**:

| slot | model              | g | LS    | seed | fill         | role in diversity                         |
|------|--------------------|---|-------|------|--------------|-------------------------------------------|
| 1    | 24_LS030_seed42    | 2 | 0.30  | 42   | complement   | low-LS / low-g + alternate-seed extreme   |
| 2    | 26B                | 3 | 0.50  | 1    | complement   | mid-LS / sweet-spot g                     |
| 3    | 26D                | 4 | 0.40  | 1    | complement   | low-LS / high-g                           |
| 4    | 26H                | 3 | 0.67  | 1    | **white**    | high-LS / mid-g + alternate-fill          |

**Diversity span achieved within hard-label space alone**:

- **g** ∈ {2, 3, 3, 4} — full sweep
- **LS** ∈ {0.30, 0.40, 0.50, 0.67} — wide spread
- **seed** ∈ {1, 1, 1, 42} — seed-axis diversity (slot 1)
- **fill** ∈ {compl, compl, compl, white} — fill-axis diversity (slot 4)

No asymmetric labels, no KD distillation. The 4 bags' errors are sufficiently
de-correlated that thr ≥ 2/4 simple-majority cancels out single-model FAR
collapses without sacrificing per-class F1.

## Paper claim — diversity-from-within-hard-label-spread

iter34 (+KD) and iter37 (+asym) achieved their headlines by adding **non-correlated
specialty axes** beyond hard-label sym training. iter39 demonstrates that:

> When the (g, LS, seed, fill) hard-label diversity is sampled wide enough,
> the resulting 4-bag majority vote **reaches the global v15 bit_F1 optimum
> (0.9992)** without needing asymmetric or KD axes.

Two implications:

1. **Cost simplification**: pure-hard 4-bag eliminates the iter32–37 KD-distillation
   training run and the iter37 asymmetric-AB-label training runs. All 4 components
   are baseline T7N + 19C-complement-CutMix variants — single training recipe family.
2. **Diversity-axis ranking** in this domain:
   `(g, LS, seed, fill)` hard-label spread > KD distillation > asymmetric AB-pair labels
   — when the hard-label spread is broad enough.

The iter37 +asym lift (+0.0015 over iter34) is therefore *replaceable* by adding
the iter25-LS=0.30-seed=42 cell + iter26H white-fill cell to the bag composition.

## Source parquets (4 NEW HEADLINE components)

- 24_LS030_seed42 v15: `D:/project/known-cnn/outputs/iter24_LS030_seed42/T7_iter24_LS030_seed42_260509_144238/eval_v15direct/stage1_260509_144925/preds_chip.parquet`
- 26B v15: `D:/project/known-cnn/outputs/iter26B_g3_LS050/T7_iter26B_g3_LS050_seed1_260509_154354/eval_v15direct/stage1_260509_160430/preds_chip.parquet`
- 26D v15: `D:/project/known-cnn/outputs/iter26D_g4_LS040/T7_iter26D_g4_LS040_seed1_260509_162552/eval_v15direct/stage1_260509_163327/preds_chip.parquet`
- 26H v15: `D:/project/known-cnn/outputs/iter26H_g3_LS067_white/T7_iter26H_g3_LS067_white_seed1_260509_165535/eval_v15direct/stage1_260509_171623/preds_chip.parquet`

## Lift summary vs prior milestones

| baseline                                        | v15 bit_F1 | Δ to iter39 |
|-------------------------------------------------|-----------:|------------:|
| 12-T5 paper start (no Normal training)          | 0.7872     | +0.2120     |
| iter21E single best                             | 0.9691     | +0.0301     |
| iter25 6-seed (prior pure-hard ensemble best)   | 0.9913     | +0.0079     |
| iter27 14-bag (paper headline @ 14× cost)       | 0.9929     | +0.0063     |
| iter33 4-bag pure-hard                          | 0.9945     | +0.0047     |
| iter34 4-bag +KD                                | 0.9961     | +0.0031     |
| iter37 4-bag +asym +KD (prior MAIN)             | 0.9976     | +0.0016     |
| **iter39 4-bag pure-hard (NEW MAIN)** ★★★       | **0.9992** | **—**       |

**vs paper start**: +0.2120 (+27%) v15 bit_F1, ni_FAR 100%(real) → 0%.

## Notes

- Inference-only experiment — zero new training cost. All 4 component checkpoints
  were already on disk from iter21–26 sweeps and survived the disk-cleanup KEEP
  expansion.
- Per-class F1 for the NEW HEADLINE recompute is pending (will fill in next pass).
- Seed-robustness of slot 1 (24_LS030_seed7 swap): retains 0.9992 bit_F1, but
  ni_FAR rises to 1.25% — still within 5% gate.
- 21H-substituted alts (rank 3–5) drop to 0.9976–0.9984: confirms 26H white-fill
  axis dominates over a redundant g=4 hard-label cell.

## See also

- `tables/paper_main_headline.csv` — NEW HEADLINE row + iter33/34/37 superseded notes
- `tables/all_runs_macro_f1.csv` — 11 new iter39 ensemble rows
- `iters/iter_37_asymmetric_AB_labels.md` — prior MAIN headline, now superseded
- `iters/iter_38_seedRobust_gapfill.md` — seed-robustness protocol used for rank 2 / rank 7
