# Cron tick #3 — KD_v8 pending

- TS: 260517 ~12:30 (1h cron tick #3, chain v7 KD sweep mid-state)
- Tag: `KD_v8_a05_T2_skipcm`
- Recipe: T7 KD with alpha=0.5, T=2, `--skip-on-cutmix`
- State: TRAIN IN PROGRESS, no eval yet.

## Snapshot

- Run dir: `outputs/KD_v8_a05_T2_skipcm/20260517_121833_T7_KD_v8_a05_T2_skipcm/`
- Checkpoints present: `best_model.pth`, `epoch_01_model.pth` (12:20), `epoch_02_model.pth` (12:23)
- Train log: `outputs/_KD_v8_a05_T2_skipcm_train.log` (0 bytes — buffered, not flushed yet)
- Eval dir: not created (`eval_n2000_pred/` absent)
- preds_chip.parquet: not found

## ETA (rough)

- ~3 min/epoch observed (12:20 → 12:23 per checkpoint).
- Likely total 8 epochs (matches sibling `KD_v4` `_ep8` suffix pattern) → train end ~12:42.
- Eval n2000 (4 variants × stage1) typically +8-12 min → results ~12:55.
- Next cron tick (#4) should be able to record final KD_v8 numbers.

## KD sweep context (chain v7)

| run                      | alpha | T  | recipe base       | state                |
|--------------------------|-------|----|-------------------|----------------------|
| KD_v2_iter116J_T2_a07    | 0.7   | 2  | iter116J          | done (recorded)      |
| KD_v3_T8_a03             | 0.3   | ?  | iter50 / T8       | done                 |
| KD_v4_T4_a05_LS20_ep8    | 0.5   | ?  | iter50B exact     | done                 |
| KD_v5_alpha02            | 0.2   | ?  | iter116J          | done                 |
| KD_v6_4sota              | -     | -  | iter50B clone     | done                 |
| KD_v7_iter116J_a03_T2    | 0.3   | 2  | iter116J skipcm   | done                 |
| KD_v8_a05_T2_skipcm      | 0.5   | 2  | T7 skipcm         | TRAIN (ep2/~8)       |

## No action

Per recorder policy: do not dispatch train/eval. Just record pending state.
