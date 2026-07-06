# 260517 chain v6 1h tick 1 — Phase 1 complete (s=11)

## What ran
- `iter116J_clone_s11` (T7 BCE+LS=0.30 + FCM-PM CutMix g=3 corner, seed=11, 10 ep)
- Train TS: 260517_082231
- Eval TS: 260517_084417 (n=2000)
- Best ckpt by margin_max selector = **ep1** (val_acc 0.9876)
- ep7 hit 0.9907 (true peak) but selector picked ep1

## Headline (POS9 strict + 4 OOD strict)

```
| Variant | bit_F1 | NI-FAR | OOD-FAR | Total FAR |
|---------|--------|--------|---------|-----------|
| I3      | 0.8582 | 100.00 |  100.00 |    100.00 |
| I7      | 0.8420 | 100.00 |  100.00 |    100.00 |
| I10     | 0.8456 |  72.65 |   63.91 |     70.53 |
| I13     | 0.7469 |  68.15 |   46.41 |     62.88 |
```

## Versus iter 116 J SOTA
- bit_F1: 0.9927 -> 0.8456 (best variant) = **-0.147 swing**
- Total FAR: 0.00% -> 70.53% (best variant) = **+70.5 pp swing**
- This is the largest single-iter regression in chain v5/v6 so far.

## Interpretation
- Not a recipe / hparam regression — same code path, same recipe family.
- **Ckpt-selection variance**: margin_max selector picked ep1 (under-trained)
  while the per-epoch trajectory had a clear ep7 peak at val_acc 0.9907.
- chain v5 implicitly varied seed + selector together; chain v6 Phase 1 has
  now isolated the selector axis on a single seed.

## Phase 2 (queued, not run this tick)
1. Re-eval s=11 best_model selected by val_f1 (= ep7), single eval call.
2. Generalise runner to emit multiple ckpt candidates per training run
   (ep_by_val_f1, ep_by_margin_max, last_epoch) so future iters
   disentangle seed effect from selector effect in 1 train.

## Recorded files
- `docs/chip-multilabel/iters/iter_v6_01_s11.md` (new)
- `docs/chip-multilabel/02_results.md` (chain v6 section prepended above v5)
- `docs/chip-multilabel/tables/all_runs_n2000.csv` (+4 rows: v6,1)
- `docs/chip-multilabel/paper/05_experiments.md` (### iter v6.01 appended)
- this diary entry

## Next tick (1h later)
- expect Phase 2 (s=11 ep7 re-eval) to be done — record delta vs ep1.
- if ep7 lands within chain v5 3-seed envelope (mean 0.9049 +/- 0.0464),
  confirm selector-variance hypothesis.
- if ep7 also collapses, escalate to "seed=11 is adverse" and trigger
  4th-seed dispatch.
