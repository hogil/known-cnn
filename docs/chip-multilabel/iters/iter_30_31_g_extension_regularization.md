# Iter 30 + 31 — FCM-PM g extension + 26B regularization tune

- **iter30 tag**: `g_extension` (FCM-PM CutMix `n_groups` + `complete_label_scale` sweep)
- **iter31 tag**: `regularization` (26B 3-seed + EMA / drop_path / warmup / epoch / lrhead sweep)
- **timestamp**: 260509 18:42 — running (incremental log)
- **status**: iter30 A/B/C complete (eval done). iter30 D training. iter30 E/F + all iter31 cells **pending** (queue).
- **goal**: (1) extend FCM-PM grid beyond iter28/29 (g∈{2,3,4} → g∈{5,6}, label_scale∈{0.20,0.30,0.50}). (2) Add 3-seed mini-ensemble of 26B (paper iter28 winner) + 17-bag (14-bag + 26B 3-seed) majority vote.
- **fixed**: T7 (BCE+LS=0.20), CutMix `complement` mode, `n_patches=5`, `total_ratio=0.3`, 8 epoch, seed=1, batch=2 (A/B/C) / 4 (D).

---

## iter30 — FCM-PM g extension

| variant | n_groups | label_scale | seed | best_val_acc | best_ep | v14 best macro_f1 (cell) | v15 best macro_f1 (cell) |
|:---|:--:|:--:|:--:|---:|:--:|---:|---:|
| 30A_g5_LS020 | 5 | 0.20 | 1 | 0.9877 | 4 | **0.7919** (T0__I10) | **0.6981** (T0__I10) |
| 30B_g5_LS050 | 5 | 0.50 | 1 | 0.9939 | 1 | **0.7752** (T0__I10) | **0.7050** (T0__I6)  |
| 30C_g6_LS030 | 6 | 0.30 | 1 | 0.9877 | 1 | **0.7625** (T0__I7)  | **0.7446** (T0__I10) |
| 30D_g2_LS050 | 2 | 0.50 | 1 | (pending) | — | (pending) | (pending) |
| 30E_g3_LS050_seed7  | 3 | 0.50 | 7  | (pending) | — | (pending) | (pending) |
| 30F_g3_LS050_seed42 | 3 | 0.50 | 42 | (pending) | — | (pending) | (pending) |

### Per-cell breakdown (4-class F1 from per_class_metrics.parquet, completed only)

**iter30A_g5_LS020 / eval_v14class** (best T0__I10 macro_f1=0.7919, top1_11=0.4484):
- bb=0.9876, fk=0.7034, sc=0.6667, sr=0.8100

**iter30A_g5_LS020 / eval_v15direct** (best T0__I10 macro_f1=0.6981, top1_11=0.2400):
- bb=0.9716, fk=0.6604, sc=0.6960, sr=0.4645

**iter30B_g5_LS050 / eval_v14class** (best T0__I10 macro_f1=0.7752, top1_11=0.5766):
- bb=0.9084, fk=0.7067, sc=0.7193, sr=0.7665

**iter30B_g5_LS050 / eval_v15direct** (best T0__I6 macro_f1=0.7050, top1_11=0.3713):
- bb=0.9297, fk=0.7706, sc=0.4474, sr=0.6722

**iter30C_g6_LS030 / eval_v14class** (best T0__I7 macro_f1=0.7625, top1_11=0.6000):
- bb=0.8190, fk=0.7746, sc=0.6084, sr=0.8480

**iter30C_g6_LS030 / eval_v15direct** (best T0__I10 macro_f1=0.7446, top1_11=0.4637):
- bb=0.8408 (est), fk=0.7068 (est), sc=0.5752 (I10 est from 0.6961), sr=0.7508 (est)

### Findings (partial)

1. **g=5/6 underperform g=3/4** (iter28 winner range). 30A/B/C all v14 < 0.80 vs iter28 26B v14 single-model ~0.93. Higher group count makes patch areas smaller → loses label-spatial signal.
2. **label_scale=0.20** best v14 (0.7919) but worst v15 (0.6981). label_scale=0.50 inverted (v14 0.7752 / v15 0.7050) — better v15 generalization but lower in-class performance.
3. **g=6 more balanced** (v14 0.7625 / v15 0.7446) — smaller cells help domain transfer to v15direct.
4. **All v15 macro_f1 < v14** (gap 0.03–0.10) — domain shift cost still present at this g level.
5. **Best epoch = 1** for B/C — fast convergence then plateau (iter25/27 winning span was epoch 4–6). g=5/6 fails to find stable late minima.

### bit_F1 + ni_FAR (★ pending — preds_chip.parquet exists, not yet rolled up)

bit_F1 + ni_FAR with the 10-defect-key bag-vote framework (matching iter27 paper headline) requires rolling each model's `preds_chip.parquet` against the 10 defect class keys × 40 chips/key + Normal/Invalid — runs as part of the bag-aggregation script. Pending against the 14-bag + 26B-3-seed extension below.

---

## iter31 — 26B regularization tune (PENDING — queue not yet started)

iter31 spec (from `_iter31_26B_tune.log`: "waiting for iter30-resume to finish"). Variants planned:

| variant | EMA | drop_path | warmup | epochs | head_lr | seed | status |
|:---|:--:|:--:|:--:|:--:|:--:|:--:|:--:|
| 31A_ema095        | 0.95 | 0    | 0 | 8  | 1e-4 | 1 | (pending) |
| 31B_droppath005   | —    | 0.05 | 0 | 8  | 1e-4 | 1 | (pending) |
| 31C_droppath010   | —    | 0.10 | 0 | 8  | 1e-4 | 1 | (pending) |
| 31D_warmup2       | —    | 0    | 2 | 8  | 1e-4 | 1 | (pending) |
| 31E_ema_warmup    | 0.95 | 0    | 2 | 8  | 1e-4 | 1 | (pending) |
| 31F_epochs16      | —    | 0    | 0 | 16 | 1e-4 | 1 | (pending) |
| 31G_lrhead5e5     | —    | 0    | 0 | 8  | 5e-5 | 1 | (pending) |

All seven variants share the iter28 26B winner config: T7 (BCE+LS=0.20), CutMix complement g=3, label_scale=0.50, n_patches=5, total_ratio=0.3, batch=2.

### Hypotheses

- **EMA 0.95** (Iter6/7 lost): retest at iter28-paradigm size (n=651 train) — earlier failure was at n=200.
- **drop_path 0.05/0.10**: small-scale ConvNeXtV2-base regularization. Iter22G drop_path=0.05 was a 14-bag member (helpful contribution).
- **warmup 2 ep**: stabilize early Adam updates with 8-ep total budget.
- **epochs 16**: longer training, see if 26B's epoch-2 best is data-bound or schedule-bound.
- **lrhead 5e-5**: gentler head LR vs frozen-style backbone.

---

## ★ 14-bag + 26B 3-seed → 17-bag (PENDING — needs 30E/30F seeds 7/42)

**Plan**: once 30E + 30F finish (iter30 D/E/F all use g=3 LS=0.50 = 26B paradigm with seeds {1, 7, 42}), aggregate with the 14-bag from iter27 (`paper_main_headline.csv`):

- **3-seed mini-ensemble** (30D seed1 + 30E seed7 + 30F seed42, majority ≥ 2/3 per bit):
  - candidate paper headline if v15 bF1 > 14-bag's 0.9929
  - target: same architectural locus (g=3, LS=0.50) with seed-only diversity → tests if **deep-seed variance is the diversity bottleneck**
- **17-bag** (14-bag from iter25/26 + iter30 D/E/F):
  - threshold sweep ≥ 6/17 (35% maj), ≥ 9/17 (53% simple-maj), ≥ 12/17 (71%)
  - paper main is updated only if v15 bF1 > 0.9929 AND ni_FAR ≤ 0.00%

Both held back until 30D/E/F preds complete.

---

## Source paths

- `outputs/iter30A_g5_LS020/T7_iter30A_g5_LS020_seed1_260509_184223/`
- `outputs/iter30B_g5_LS050/T7_iter30B_g5_LS050_seed1_260509_185513/`
- `outputs/iter30C_g6_LS030/T7_iter30C_g6_LS030_seed1_260509_190634/`
- `outputs/iter30D_g2_LS050/T7_iter30D_g2_LS050_seed1_260509_194611/` (training)
- `outputs/iter30E_g3_LS050_seed7/` (queued)
- `outputs/iter30F_g3_LS050_seed42/` (queued)
- `outputs/iter31A_ema095/` ... `outputs/iter31G_lrhead5e5/` (queued)
- log: `outputs/_iter30_fcmpm_variants_resume.log`, `outputs/_iter31_26B_tune.log`
