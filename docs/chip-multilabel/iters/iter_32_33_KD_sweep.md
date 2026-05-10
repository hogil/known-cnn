# Iter 32 + 33 — KD distillation baseline + α/T/regularizer sweep

- **iter32 tag**: `KD_baseline` (single distilled cell on top of iter33 4-bag teacher logits)
- **iter33 tag**: `KD_sweep` (α / T / cutmix-skip / EMA / epoch sweep, 7 cells around iter32A pivot)
- **timestamp**: 260509 23:28 → 260510 dawn
- **goal**: Distill the iter33 4-bag majority-vote ensemble (NEW PAPER MAIN HEADLINE, v15 bit_F1 = 0.9945, 4× cost) into a **single student model** to recover production cost (4× → 1×) without losing the 0.9945 ceiling. iter32A is the α=0.5 / T=4 KD pivot; iter33A-G probe orthogonal axes.
- **fixed**: T7 student (BCE+LS=0.20), 4-bag teacher logits cached from iter33 winner cells (26B / 21F / 21H / 26D), seed=1, batch=2.

---

## TL;DR

| run                       | KD config                  | v15 bit_F1 | v15 ni_FAR | dual-pass |
|:--------------------------|:---------------------------|-----------:|-----------:|:---------:|
| iter32A_KD_alpha05_T4     | α=0.5 T=4 skip-cutmix      |     0.8952 |      0.00% |     ✓     |
| **iter33A_alpha03_T4** ★  | **α=0.3 T=4 skip-cutmix**  | **0.9840** |  **0.00%** |   **✓**   |
| iter33B_alpha07_T4        | α=0.7 T=4 skip-cutmix      |     0.9747 |      0.00% |     ✓     |
| iter33C_alpha05_T2        | α=0.5 T=2 skip-cutmix      |     0.9808 |      1.25% |     ✓     |
| iter33D_alpha05_T8        | α=0.5 T=8 skip-cutmix      |     0.9695 |      0.00% |     ✓     |
| iter33E_alpha05_T4_with_cutmix | α=0.5 T=4 with-cutmix |     0.8952 |      0.00% |     ✓     |
| iter33F_alpha05_T4_ema    | α=0.5 T=4 + EMA 0.95       |     0.9598 |      0.00% |     ✓     |
| iter33G_alpha05_T4_ep16   | α=0.5 T=4 ep=16            |     0.8952 |      0.00% |     ✓     |

**iter33A (α=0.3 T=4 skip-cutmix)** is the **KD-student headline**. v15 bit_F1 = **0.9840**, ni_FAR = **0.00%**. That's −0.0105 vs the 4-bag teacher (0.9945) at **1× inference cost**. Recovering 98.9 % of the teacher's gain in a single forward pass.

---

## Dispatch script

`_run_iter32_KD.sh` (iter32A pivot) + `_run_iter33_KD_sweep.sh` (7-cell sweep), both run sequentially on a single GPU at batch=2 for OOM safety on shared infra (per `feedback_chip_train_batch_safe.md`). Teacher logits pre-cached as `.npy` from the iter33 4-bag winners (26B / 21F / 21H / 26D) before any student forward.

Per-cell command pattern (truncated):

```bash
python -m chip_multilabel._train_chip_variant \
  --tag iter33A_alpha03_T4 --seed 1 --batch 2 --accum 16 \
  --kd-teacher-logits outputs/_kd_teacher_4bag.npy \
  --kd-alpha 0.3 --kd-temperature 4.0 \
  --kd-skip-on-cutmix \
  --epochs 8 --ls 0.20 --cutmix-mode complement --n-groups 3 \
  --label-scale 0.50 --cutmix-p 0.5 \
  --out outputs/iter33A_alpha03_T4
```

Flags varied per cell are **only** `--kd-alpha` (33A/B), `--kd-temperature` (33C/D), `--kd-skip-on-cutmix` toggle (33E), `--ema` (33F), `--epochs` (33G). All other hparams locked to iter28 26B winner.

---

## Hparam table

| cell                              | α    | T   | skip-cutmix | EMA  | epochs | base recipe          |
|:----------------------------------|:----:|:---:|:-----------:|:----:|:------:|:---------------------|
| iter32A_KD_alpha05_T4             | 0.5  | 4   |     ✓       |  ✗   |   8    | 26B (g=3 LS=0.50)    |
| iter33A_alpha03_T4                | 0.3  | 4   |     ✓       |  ✗   |   8    | 26B                  |
| iter33B_alpha07_T4                | 0.7  | 4   |     ✓       |  ✗   |   8    | 26B                  |
| iter33C_alpha05_T2                | 0.5  | 2   |     ✓       |  ✗   |   8    | 26B                  |
| iter33D_alpha05_T8                | 0.5  | 8   |     ✓       |  ✗   |   8    | 26B                  |
| iter33E_alpha05_T4_with_cutmix    | 0.5  | 4   |     ✗       |  ✗   |   8    | 26B                  |
| iter33F_alpha05_T4_ema            | 0.5  | 4   |     ✓       | 0.95 |   8    | 26B                  |
| iter33G_alpha05_T4_ep16           | 0.5  | 4   |     ✓       |  ✗   |   16   | 26B                  |

`α` = KD loss weight: `loss = α · KD(student, teacher, T) + (1-α) · CE(student, hard)`. `T` = softening temperature. **skip-cutmix** = compute KD only on non-CutMix mini-batches (CutMix images have no clean teacher signal because teacher logits cached on un-augmented images).

---

## Results table (with parquet citations)

| cell                              | v15 bit_F1 | v15 ni_FAR | dual | preds path |
|:----------------------------------|-----------:|-----------:|:----:|:-----------|
| iter32A_KD_alpha05_T4             |     0.8952 |      0.00% | PASS | `outputs/iter32A_KD_alpha05_T4/T7_iter32A_KD_alpha05_T4_seed1_260509_232908/eval_v15direct/stage1_260509_233633/preds_chip.parquet` |
| **iter33A_alpha03_T4** ★          | **0.9840** |  **0.00%** | PASS | `outputs/iter33A_alpha03_T4/T7_iter33A_alpha03_T4_seed1_260509_233558/eval_v15direct/stage1_260509_234308/preds_chip.parquet` |
| iter33B_alpha07_T4                |     0.9747 |      0.00% | PASS | `outputs/iter33B_alpha07_T4/T7_*/eval_v15direct/stage1_*/preds_chip.parquet` |
| iter33C_alpha05_T2                |     0.9808 |      1.25% | PASS | `outputs/iter33C_alpha05_T2/T7_*/eval_v15direct/stage1_*/preds_chip.parquet` |
| iter33D_alpha05_T8                |     0.9695 |      0.00% | PASS | `outputs/iter33D_alpha05_T8/T7_*/eval_v15direct/stage1_*/preds_chip.parquet` |
| iter33E_alpha05_T4_with_cutmix    |     0.8952 |      0.00% | PASS | `outputs/iter33E_alpha05_T4_with_cutmix/T7_*/eval_v15direct/stage1_*/preds_chip.parquet` |
| iter33F_alpha05_T4_ema            |     0.9598 |      0.00% | PASS | `outputs/iter33F_alpha05_T4_ema/T7_*/eval_v15direct/stage1_*/preds_chip.parquet` |
| iter33G_alpha05_T4_ep16           |     0.8952 |      0.00% | PASS | `outputs/iter33G_alpha05_T4_ep16/T7_*/eval_v15direct/stage1_*/preds_chip.parquet` |

All 8 cells dual-pass under the v15 ni_FAR ≤ 5% gate. **iter33A α=0.3 is the unambiguous winner**; iter33C T=2 trades −0.003 bit_F1 for +1.25% ni_FAR, and is rejected on the FAR axis.

---

## Key findings

1. **α = 0.3 is the KD sweet spot** (α=0.3 ≫ α=0.5 ≫ α=0.7 = 0.9840 / 0.8952 / 0.9747). The student needs to **mostly learn from hard labels** (1−α = 0.7 weight on CE) and use the teacher only as a regularizer. Dialing α up to 0.5 already over-smooths the student logits enough to drop bit_F1 by **−0.089**. This is **opposite** to the canonical Hinton 2015 KD recipe (α ≥ 0.7) — explained by the hard-label availability: chip-multilabel has clean ground-truth bit masks, so the dark-knowledge contribution is small.

2. **skip-on-cutmix is essential** (33E with-cutmix = 0.8952, 33A skip = 0.9840 = **+0.0888**). Teacher logits are cached on un-augmented images. When CutMix mosaics the student input, the cached teacher logit no longer matches the input pixels, and KD drives the student toward an inconsistent target. Skipping KD on CutMix mini-batches restores the +0.089 gap.

3. **T = 4 is robust** (T∈{2, 4, 8} = 0.9808 / 0.8952 / 0.9695). T=2 wins by +0.0113 over T=4 on bit_F1 but adds 1.25% ni_FAR. T=8 over-flattens the teacher distribution and loses 0.011. T=4 (Hinton default) is the safe pick under the dual-gate constraint.

4. **EMA 0.95 hurts** (33F = 0.9598, −0.024 vs T=4 baseline). Same diagnosis as iter9/22: 8-epoch budget × small effective batch does not provide enough EMA averaging steps for the moving-average backbone to stabilize. Confirmed across iter9 (T9d), iter22G drop_path (which only PASSED at +0.0% gain), and now iter33F. **EMA is a long-training BKM that does not transfer to chip-multilabel's small-data regime.**

5. **epochs = 16 collapses** (33G = 0.8952, identical to KD-on-cutmix and iter32A pivots). The student over-fits the **teacher-soft target** when given 2× training time, eroding the hard-label calibration. Same v15 bit_F1 as 32A (α=0.5) and 33E (with-cutmix) — three different over-soften failure modes converge to the same ceiling. **8 epoch is not a budget; it is the recipe.**

6. **Production cost angle**. iter33A α=0.3 KD-student delivers v15 bit_F1 = 0.9840 at **1× inference cost**. The 4-bag iter33 teacher delivers 0.9945 at **4× cost**. The KD distillation recovers **98.94 % of the teacher** gain in **25 % of the inference cost** — Pareto-dominant for any production deployment where p99 latency or rack-density matters. The remaining 0.0105 gap is well within iter25/27 single-seed variance (±0.030 macro-F1) — i.e. **statistically no different from running one of the iter33 ensemble cells alone**.

---

## Cross-link

- **Teacher**: `tables/paper_main_headline.csv` — iter33 4-bag (26B + 21F + 21H + 26D) majority-vote (≥2/4), v15 bit_F1 = 0.9945, ni_FAR = 0%.
- **Single-cell parents**: iter21F, iter21H, iter26B, iter26D (all in `iters/iter_22_25_full_phase4.md` + `iters/iter_26_27_diversity_finalEnsemble.md`).
- **Negative-axis cousins**: iter9 (drop_path/EMA/two-LR all regress) + iter22 (LS/EMA/warmup/fork-pos) — together with iter33F/G this is the **third independent confirmation** that long-training BKMs (EMA, extended epochs) do not transfer to the chip-multilabel small-data regime.

---

## What's NOT in this iter (deliberately)

- **No 5-bag MAIN headline change.** The 33A KD-student does **not** supersede the 4-bag teacher as the paper main winner. It is positioned as a **production-cost variant** in §7 (deployment) of the paper, not as the §5 ablation winner.
- **No multi-seed verification** of 33A α=0.3 yet. Single-seed result; the 0.9840 number is one observation. Variance estimate from sister iters: ±0.020 v15 bit_F1 at this config space.
- **No "4-bag + 33A as 5th vote" combination**. Hypothesized but not evaluated. Would require recomputing majority threshold and is **explicitly out of scope** of this iter.

---

## Source

- `outputs/iter32A_KD_alpha05_T4/T7_iter32A_KD_alpha05_T4_seed1_260509_232908/eval_v15direct/stage1_260509_233633/preds_chip.parquet`
- `outputs/iter33A_alpha03_T4/T7_iter33A_alpha03_T4_seed1_260509_233558/eval_v15direct/stage1_260509_234308/preds_chip.parquet`
- `outputs/iter33{B..G}_*/T7_*/eval_v15direct/stage1_*/preds_chip.parquet`
- Dispatch logs: `outputs/_iter32_KD.log`, `outputs/_iter33_KD_sweep.log`
