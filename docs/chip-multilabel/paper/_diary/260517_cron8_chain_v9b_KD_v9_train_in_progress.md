# 2026-05-17 cron 8 (18:07) — chain v9b KD_v9 train in progress, eval pending

## Event

Chain v9b (`_run_chain_v9b.sh`) phase 1 entry — KD_v9 retry with relaxed
GPU thresholds (train 70%, eval 70%) — dispatched at 17:05:06 after the
foreign GPU dropped to 55% (below the 70% train threshold).

```
| Time     | Phase            | Event                                          |
|----------|------------------|------------------------------------------------|
| 17:05:06 | v9b summary log  | gpu free (used=55%, <70%) — proceeding         |
| 17:05:06 | v9b summary log  | TRAIN KD_v9_a02_T2_skipcm dispatched           |
| 17:05:06 | v9b runner       | bash _run_chain_v9b.sh pid 527961 alive        |
| 17:05:06 | python           | _train_chip_variant pid 527968 alive           |
| 17:24:31 | filesystem       | best_model.pth first write (350 MB)            |
| 18:07    | cron 8 check     | gpu 14833/16380 MiB 100% util — train ongoing  |
```

## Status

```
| Component                | State                                              |
|--------------------------|----------------------------------------------------|
| chain v9b bash pid       | 527961 alive (uptime 1h 02m)                       |
| python train pid         | 527968 alive (CPU/GPU active)                      |
| best_model.pth           | exists 350MB, mtime 17:24:31 (one best update)     |
| train log file size      | 0 bytes (buffered stdout, written on finish)       |
| eval_n2000_pred dir      | absent (eval not yet dispatched)                   |
| preds_chip.parquet       | absent                                             |
| GPU memory               | 14833 / 16380 MiB                                  |
| GPU util                 | 100%                                               |
```

Train command (from `ps -ef`):

```
python -X utf8 -m chip_multilabel._train_chip_variant \
  --variant T7 --ls 0.30 --batch 2 --accum 8 --seed 1 --lr 1e-4 --epochs 10 \
  --data-root E:/data/images/classification_chips \
  --backbone-timm convnextv2_base.fcmae_ft_in22k_in1k_384 \
  --img-size 384 --backbone-timm-weights models/...base_384.pth \
  --no-normal --cutmix-mode complement --cutmix-pair masked \
  --cutmix-pair-fill corner --cutmix-p 0.25 --cutmix-n-groups 3 \
  --cutmix-complete-label-scale 0.5 \
  --tag KD_v9_a02_T2_skipcm --out-root outputs/KD_v9_a02_T2_skipcm \
  --kd-teacher-probs outputs/_teacher_probs_iter116J_e.parquet \
  --kd-alpha 0.2 --kd-temperature 2.0 --kd-skip-on-cutmix
```

## Why eval has not started

`_run_chain_v9b.sh::train_eval_gated` is sequential: it waits for the
python train subprocess to exit (return ckpt) before calling
`run_stage1`.  best_model.pth at 17:24 was a best-on-val checkpoint
write during training, not the train-end marker.  At batch=2 accum=8
effective bs=16 with grad-checkpointing on a single chip dataset
(~28 chips/class × 4 = 112 train chips per epoch, but each chip
augmented with cutmix), 10 epochs is ~1-1.5h.  62 min into the run is
consistent with mid-late training.  Eval will dispatch as soon as
the python pid 527968 returns and the next phase of v9b loops to
the eval-gated path.

## Action

Do not interfere.  Cron 9 will re-check; if best_model.pth mtime
advances past current 17:24 (= late epoch best), train is still
producing better epochs.  If python pid 527968 has exited and
preds_chip.parquet appears, record cron 9 will compute POS9 strict
+ Total FAR for variants I3/I7/I10/I13 and add the KD_v9 row to
RESULTS_TIMELINE.md section C and tables/all_runs_n2000.csv.

## No changes this tick

- iter doc: not written (no eval results)
- RESULTS_TIMELINE.md: not updated (no new metrics)
- tables/all_runs_n2000.csv: not appended (no new metrics)
- 05_experiments.md: not appended

## Files touched this tick

- `docs/chip-multilabel/paper/_diary/260517_cron8_chain_v9b_KD_v9_train_in_progress.md` (this file)

## Source

- Train python: pid 527968, started 17:05:06, alive at 18:07
- Train ckpt: `outputs/KD_v9_a02_T2_skipcm/20260517_170518_T7_KD_v9_a02_T2_skipcm/best_model.pth` (mtime 17:24:31)
- Train log: `outputs/_KD_v9_a02_T2_skipcm_train.log` (0 bytes, buffered)
- Supervisor / runner: `outputs/_chain_v9b_summary.log`, `outputs/_chain_v9b_runner.log`
