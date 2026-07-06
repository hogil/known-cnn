# 260517 chain v6 종합 — 1h cron tick (4 phase 완료 후 summary)

## What ran (entire chain v6, 4 phase)

- Phase 1: `iter116J_clone_s11` (seed=11, T7 BCE+LS=0.30 + FCM-PM CutMix g=3 corner)
  - train TS 260517_082231, eval TS 260517_084417
  - ckpt margin_max -> ep1 (under-trained, val_acc 0.9876 plateau)
- Phase 2: `iter116J_clone_s23` (seed=23, same recipe family)
  - train TS 260517_085059, eval TS 260517_090654
  - ckpt margin_max -> ep9, train val_f1 0.9969 by ep10 but synthetic-eval collapse
- Phase 3: `iter116J_clone_s77` (seed=77, same recipe family)
  - train TS 260517_091330, eval TS 260517_092932
  - ckpt margin_max -> ep8, train val_f1 0.9969 by ep10
- Phase 6: `KD_v7_iter116J_a03_T2_skipcutmix` (KD via new --kd-skip-on-cutmix)
  - train TS 260517_095713, eval TS 260517_101336
  - teacher = `outputs/iter116J_g3_ls30/T7_iter116J_g3_ls30_260513_010015`
  - alpha=0.3, T=2, seed=1, ckpt margin_max -> ep7

(Note: there was no Phase 4 / Phase 5 in this chain — phase ids 4 and 5
were reserved during planning but not dispatched in this tick window.
Naming kept as v6.01/02/03/04 in iters/ for sequence; Phase 6 label
preserved for KD experiment provenance.)

## Headline summary (best variant per row, POS9 strict + 4 OOD strict)

```
| chain | iter | tag                              | seed | LS   | ep | best | bit_F1 | NI-FAR | OOD-FAR | Total FAR |
|-------|------|----------------------------------|------|------|----|------|--------|--------|---------|-----------|
| past  | 116J | T7 LS=0.30 g=3 seed=1 SOTA       |    1 | 0.30 |  6 | I13  | 0.9927 |   0.00 |    0.00 |      0.00 |
| v6    |    1 | iter116J_clone_s11               |   11 | 0.30 |  1 | I10  | 0.8456 |  72.65 |   63.91 |     70.53 |
| v6    |    2 | iter116J_clone_s23               |   23 | 0.30 |  9 | I10  | 0.4738 |  63.20 |   76.56 |     66.44 |
| v6    |    3 | iter116J_clone_s77               |   77 | 0.30 |  8 | I10  | 0.9786 |   0.40 |    1.88 |      0.76 |
| v6    |    4 | KD_v7_iter116J_a03_T2_skipcutmix |    1 | 0.30 |  7 | I10  | 0.9265 |   0.00 |    0.00 |      0.00 |
```

## 3 핵심 finding (cross-chain interpretation)

### Finding 1 — s=77 micro-win FAR trade-off (Phase 3)

- First non-baseline seed in scan {1, 7, 11, 23, 42, 77, 99} that exceeds
  iter116J SOTA on I10 bit_F1: +0.0038 (0.9748 -> 0.9786)
- Cost: +0.76 pp Total FAR (mostly NI-FAR 0.40% + OOD-FAR 1.88%)
- Per-defect I3 breakdown isolates scratch as the unique weak class:
  bb 0.997 / fork 0.980 / scratch 0.847 / scratch_rot 0.999
- Combined product (bit_F1 - 0.01 * FAR): 0.9710 vs 0.9927 -> does NOT
  replace iter116J as headline. Logged as informational, not SOTA.

### Finding 2 — KD_v7 collapse fix via --skip-on-cutmix (Phase 6)

- 6 prior KD attempts all collapsed (bit_F1 < 0.5 or NaN). Root cause
  hypothesis: teacher prob computed on clean chip vs student forward on
  CutMix-complement-active chip (25% of batches under g=3 corner) ->
  KL gradient mismatch.
- New flag --kd-skip-on-cutmix disables KD loss on those 25% batches,
  preserves regularisation on the other 75%.
- Result: bit_F1 0.9265 / Total FAR 0.00% — no collapse. But also no
  ceiling-break: -0.0483 bit_F1 vs teacher (iter116J g3_ls30 I10 0.9748).
- KD acts as a clean regulariser (lowers variance, lowers ceiling).
- Collateral effect: KD I3 lands at FAR 3.75% vs no-KD seed-clone I3 at
  87-100%, so KD calibrates simpler gates enough to make I3 viable
  without the entropy gate (I10) or invalid-score gate (I13).
- Next experiment: multi-teacher KD (bag of {s1, s77, future seeds})
  may add diversity to beat single-teacher ceiling.

### Finding 3 — margin_max selector variance ±0.13 bit_F1 within recipe

- chain v6 seed clones (s=11, s=23, s=77) all use margin_max selector
  by default; selected epochs were ep1 / ep9 / ep8 respectively.
- Train trajectory was similar in all three (val_acc 0.9876 plateau,
  single-epoch peaks <= 0.991).
- I10 bit_F1 spread across {s11, s23, s77} = [0.4738, 0.8456, 0.9786],
  std ±0.21 — far exceeding the chain v5 (3-seed val_f1-selector)
  envelope of ±0.046.
- Same recipe family with val_f1 selector at s=1 -> 0.9927 (iter116J).
- Conclusion: selector decision (which epoch is "best") carries more
  variance than the seed itself within this recipe.
- Action: emit ep_by_val_f1 + ep_by_margin_max + last_epoch as 3 ckpt
  candidates per training run -> disentangle selector vs seed in a
  single train.

## Versus iter116J SOTA — cross-iter delta

```
| iter | dbit_F1 (best) | dTotal_FAR_pp (best) | net product (bit_F1 - 0.01 * FAR) |
|------|----------------|----------------------|-----------------------------------|
| 116J |          0.000 |                 0.00 |                            0.9927 |
| v6.1 |        -0.1471 |               +70.53 |                            0.1404 |
| v6.2 |        -0.5189 |               +66.44 |                           -0.1906 |
| v6.3 |        -0.0141 |                +0.76 |                            0.9710 |
| v6.4 |        -0.0662 |                 0.00 |                            0.9265 |
```

iter116J remains headline. v6.3 (s=77) closest non-baseline at 0.9710
product.

## Files recorded this tick

- `docs/chip-multilabel/iters/iter_v6_0[1-4]_*.md` (4 per-iter files,
  written by master agent during phase dispatch — not by this tick)
- `docs/chip-multilabel/02_results.md` — chain v6 종합 분석 section
  prepended above existing chain v6 iter 1 section (this tick)
- `docs/chip-multilabel/paper/05_experiments.md` — ### chain v6 종합
  subsection appended at file end (this tick)
- `docs/chip-multilabel/paper/_diary/260517_chain_v6_summary_cron_tick.md`
  — this entry (this tick)
- `docs/chip-multilabel/tables/all_runs_n2000.csv` — already contains
  v6,1..4 rows (16 rows) from per-iter dispatches; not modified this tick

## Next tick suggestions (1h later)

1. Re-evaluate s11, s23, s77 best_model at ep_by_val_f1 (instead of
   margin_max) — single eval call each, isolates selector axis.
2. Multi-teacher KD: build 2-teacher bag {iter116J g3_ls30 s=1,
   iter116J_clone_s77} -> single KD run, alpha=0.3, T=2,
   --kd-skip-on-cutmix. Test diversity hypothesis vs single-teacher
   ceiling.
3. If selector hypothesis confirmed, retire margin_max default in
   runner config; switch to val_f1 + emit-all-three pattern.
