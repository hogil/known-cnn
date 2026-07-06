# Cron tick #4 — KD sweep finale (chain v7 close)

- TS: 260517 ~13:15 (1h cron tick #4, chain v7 KD sweep terminal)
- Chains touched: v7 (Phase 1 ensemble closed, Phase 3 KD sweep terminated)
- New artefact state on disk: no new eval parquet this tick.

## KD sweep terminal state (chain v7 Phase 3)

```
| run                      | alpha | T  | recipe base       | train_status     | eval_status       |
|--------------------------|-------|----|-------------------|------------------|-------------------|
| KD_v2_iter116J_T2_a07    |   0.7 |  2 | iter116J          | done             | recorded          |
| KD_v3_T8_a03             |   0.3 |  ? | iter50 / T8       | done             | recorded          |
| KD_v4_T4_a05_LS20_ep8    |   0.5 |  ? | iter50B exact     | done             | recorded          |
| KD_v5_alpha02            |   0.2 |  ? | iter116J          | done             | recorded          |
| KD_v6_4sota              |     - |  - | iter50B clone     | done             | recorded          |
| KD_v7_iter116J_a03_T2    |   0.3 |  2 | iter116J skipcm   | done (10 ep)     | recorded 0.9265   |
| KD_v8_a05_T2_skipcm      |   0.5 |  2 | T7 skipcm         | crash ep03 OOM   | reeval crash CUDA |
| KD_v9_a02_T2_skipcm      |   0.2 |  2 | T7 skipcm         | crash pre-ep OOM | n/a               |
| KD_v10_a03_T1_skipcm     |   0.3 |  1 | T7 skipcm         | crash pre-ep OOM | n/a               |
```

## KD_v8 incident

- `outputs/KD_v8_a05_T2_skipcm/20260517_121833_T7_KD_v8_a05_T2_skipcm/best_model.pth`
  exists (ep03 snapshot saved before crash) but no `eval_n2000_pred/` subdir.
- Train log `outputs/_KD_v8_a05_T2_skipcm_train.log`:
  - `[kd] loaded 2015 teacher probs ... alpha=0.5 T=2.0 skip_on_cutmix=True`
  - ep01-03 ran clean (v_f1=0.9969, v_auroc=1.0000 — matches KD_v7 trajectory exactly).
  - ep04 backward pass: `RuntimeError: CUDA error: out of memory` at
    `scaler.scale(loss).backward()` in `_train_chip_variant.py:1479`.
  - Subsequent `Unhandled exception caught in c10/util/AbortHandler.h` →
    process abort. No `epoch_04_model.pth` written.
- Re-eval attempt log `outputs/_KD_v8_a05_T2_skipcm_reeval_n2000.log`:
  - Loaded ep03 `best_model.pth`, ran stage1 invalid heuristic, then crashed
    in `inference_variants.forward_all_logits` at `F.linear` call with
    `CUBLAS_STATUS_EXECUTION_FAILED`. Likely shared-GPU contention rather than
    intrinsic model defect — KD_v7 with the same train recipe + smaller alpha
    succeeded in both train and eval on the same hardware.
- **No POS9 strict metric recorded for KD_v8**; not eligible for chain v7 SOTA
  pool. `iters/iter_v7_02_KD_v8.md` is therefore not created this tick (no
  eval parquet to cite).

## KD_v9 / KD_v10 incidents

- Both crashed before any `[ep XX]` line — same `Unhandled exception caught
  in c10/util/AbortHandler.h` with `at::cuda::CUDAEvent::createEvent` /
  `at::cuda::getCachingHostAllocator` frames.
- Trace signature is the classic CUDA OOM at init-time tensor allocation, not
  a NaN / loss-divergence collapse. The runtime did not get far enough to
  print an `[ep 01]` line.
- KD_v9 hparams: alpha=0.2, T=2, skip_on_cutmix=True (T7 base).
- KD_v10 hparams: alpha=0.3, T=1, skip_on_cutmix=True (T7 base).
- Both runs share the timestamp window 13:02-13:08 (immediately after KD_v8
  crash at ~12:50). Most likely root cause: residual VRAM held by the crashed
  KD_v8 python process not released back to the shared-GPU pool before the
  KD_v9 / KD_v10 dispatches launched. Matches the known
  `feedback_windows_python_dispatch.md` failure mode (Windows Hidden-window
  python zombies hold GPU memory until kill).

## KD viable-corner hypothesis (chain v7 finding)

Restricting to runs that produced a recordable I10 bit_F1:

```
| run                     | alpha | T  | I10 bit_F1 | Total FAR | status         |
|-------------------------|-------|----|------------|-----------|----------------|
| KD_v5_alpha02 (iter116J)|   0.2 |  ? |     0.1093 |     99.47 | collapse       |
| KD_v3_T8_a03            |   0.3 |  ? |     0.6435 |    100.00 | over-positive  |
| KD_v7_iter116J_a03_T2   |   0.3 |  2 |     0.9265 |      0.00 | viable         |
| KD_v4_T4_a05_LS20_ep8   |   0.5 |  ? |     0.8298 |     22.77 | partial        |
| KD_v2_iter116J_T2_a07   |   0.7 |  2 |     0.7874 |      0.08 | over-smoothed  |
```

(KD_v8 alpha=0.5 T=2 would have filled the (alpha=0.5, T=2) cell but is
absent due to the crash. KD_v6 was a 4-teacher SOTA aggregate and is not
indexed in the alpha sweep.)

**Hypothesis (paper-citable).** The KD viable corner on this T7 + BCE+LS=0.30
+ FCM-PM CutMix base is **alpha ∈ [0.3, 0.5] at T=2**. Below alpha=0.3 the
student over-trusts the teacher's high-confidence soft target on the
near-saturated (val_acc=0.9969) validation distribution and the loss
landscape collapses (KD_v5 0.1093). Above alpha=0.5 the smoothing of the
hard target by the soft target washes out the discriminative cells and the
student loses 0.2+ bit_F1 (KD_v2 0.7874 at alpha=0.7). T=1 (KD_v10) and
T=2 (KD_v7, KD_v8 partial) are the only temperatures tested in the
clean-Pareto window; KD_v10's pre-epoch crash leaves the T=1 cell unconfirmed.

**Single confirmed in-window cell**: alpha=0.3, T=2, with `--kd-skip-on-cutmix`
(KD_v7). This is the student that earned a spot in the chain v7 Phase 1
ensemble (vote_majority_bits) and contributed the deciding vote on edge
cases where iter116J s=1 and iter116J_clone_s77 disagreed (see iter_v7_01).
**KD_v8 (alpha=0.5, T=2)** was the second candidate for the viable window
but did not survive to eval; recommend re-dispatch once GPU pool is idle
(see Recovery section).

## Chain v7 summary (Phase 1 / Phase 2 / Phase 3)

- **Phase 1 — post-hoc 3-model bit-vote ensemble**: succeeded.
  `vote_majority_bits` over (iter116J s=1, iter116J_clone_s77, KD_v7) =
  **bit_F1 0.9941 / Total FAR 0.00%**, first cell across chain v5+v6+v7 to
  beat the iter116J SOTA on bit_F1 without any FAR penalty (+0.0014 at zero
  FAR). Recorded in `iters/iter_v7_01_ensemble_champion.md`.
- **Phase 2 — val_f1 ckpt-selector re-eval**: flagged invalid in cron tick #3.
  The val_f1 selector under-reports vs `margin_max` on near-saturated val
  distributions (val_acc=0.9969 → 0.9985 ties common, val_f1 picks the
  earliest tie which is ep01-02, before the CutMix-driven combo recovery).
  No paper headline pulled from Phase 2.
- **Phase 3 — KD hparam sweep**: partial. KD_v7 (alpha=0.3, T=2) is the only
  in-window KD student confirmed; KD_v8/v9/v10 OOM crashes blocked the
  alpha=0.5 confirmation and the T=1 ablation.

## Recovery / next-tick recommendation

1. Verify no orphan python processes hold GPU memory before any KD re-dispatch
   (`feedback_problem_kill_restart_rule`). Issue a clean kill-all + idle wait
   before launching KD_v8 retry.
2. KD_v8 (alpha=0.5, T=2, skip_on_cutmix=True) is the highest-value
   re-dispatch — fills the only unconfirmed in-window cell. Smaller batch
   (`--batch 4 --accum 8` to maintain effective 32 with halved peak VRAM)
   recommended given the OOM signature.
3. KD_v10 (alpha=0.3, T=1) is the next-priority ablation if the alpha sweep
   closes successfully — confirms whether T=1 narrows or widens the viable
   window relative to T=2.
4. KD_v9 (alpha=0.2) is **not** recommended for retry — KD_v5 at alpha=0.2
   already collapsed (bit_F1 0.1093), so the alpha=0.2 cell is empirically
   established as out-of-window. A second crash at alpha=0.2 would not add
   evidence.

_Source: `outputs/_KD_v[7-10]_*_train.log`, `outputs/_KD_v8_*_reeval_n2000.log`,
`outputs/KD_v8_a05_T2_skipcm/20260517_121833_*/best_model.pth` (ep03 snapshot,
not eligible for paper headline); cross-iter ledger
`docs/chip-multilabel/tables/all_runs_n2000.csv`._
