# 2026-05-16 — chain v5 iter 1-4 backlog recorded

Recorded the 4 chain-v5 iters in one batch (recorder agents restarted
after a hang, backlog cleanup per user directive).

## Iters processed

```
| iter | tag                          | seed | LS   | best | bit_F1 | Total FAR |
|------|------------------------------|------|------|------|--------|-----------|
|    1 | iter50_clone_seed99_v3       |   99 | 0.30 | I10  | 0.8778 |      0.04 |
|    2 | iter50_clone_seed42_v4       |   42 | 0.30 | I10  | 0.9583 |      0.30 |
|    3 | iter50_clone_seed07_v4       |    7 | 0.30 | I10  | 0.8787 |     18.22 |
|    4 | iter50_clone_LS025_s1_v4     |    1 | 0.25 | I10  | 0.9121 |      0.42 |
| 116J | (past reference)             |    1 | 0.30 | I13  | 0.9927 |      0.00 |
```

Source parquets all under `outputs/iter50_clone_*/<TS>/eval_n2000_pred/stage1_*/preds_chip.parquet`.

## Headline finding

iter 116 J is a **seed outlier** (+1.9σ above the 3-seed mean at I10), not
a robust recipe optimum. Gate (I10/I13) is seed-dependent — works fully
at 3 of 4 seeds, fails partially at seed=7 (Total FAR 18-22%). LS=0.25
strictly regresses vs LS=0.30 at same seed.

Detail in `paper/05_experiments.md` §5.48 (5 subsections).

## Files updated

- `docs/chip-multilabel/iters/iter_v5_0[1-4]_*.md` (new, 4 files)
- `docs/chip-multilabel/tables/all_runs_n2000.csv` (new, 16 rows)
- `docs/chip-multilabel/02_results.md` (prepended 2026-05-16 update section)
- `docs/chip-multilabel/paper/05_experiments.md` (appended §5.48)
- `docs/chip-multilabel/paper/_diary/260516_chain_v5_iter_1_4_backlog.md` (this)
