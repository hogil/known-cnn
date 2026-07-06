# 2026-05-17 cron 11 (20:45) — Model Soup 3-way landing + KD_v10 train FAIL

## Headline

Two events landed at cron 11.  Order of operational priority:

1. **Model Soup 3-way (Wortsman 2022 uniform mean) eval n2000 COMPLETE.**
   Soup ckpt `outputs/soup_v1_3way/best_model.pth` over 3 members
   {iter116J s=1, iter116J_clone_s77, KD_v7}; eval at
   `outputs/soup_v1_3way/eval_n2000_pred/stage1_260517_201557/`.
   Soup I10 = **bit_F1 0.9748 / Total FAR 0.00 %**.  Champion
   `vote_majority_bits` (same 3-way pool) remains **0.9941 / 0.00 %** at I10.
   **Champion not updated.**  Headline regression: -0.0193 bit_F1 at I10
   (POS9 strict) vs the chain v7/v8 champion.

2. **KD_v10 (alpha=0.3 T=1 skip-cm) train FAIL.**  OOM within 9 min when
   external Python processes pushed sys_ram to 91 %.  No checkpoint
   persisted, no parquet.  T=1 corner at alpha=0.3 remains operationally
   unmapped.  Status note only — no metric to record.

## Soup result detail (POS9 strict)

```
| Variant | bit_F1 | NI-FAR | OOD-FAR | Total FAR | Δ vs vote_majority_bits |
|---------|--------|--------|---------|-----------|-------------------------|
| I3      | 0.9274 | 100.00 |  100.00 |    100.00 |                       - |
| I7      | 0.9263 | 100.00 |  100.00 |    100.00 |                       - |
| I10     | 0.9748 |   0.00 |    0.00 |      0.00 |                 -0.0193 |
| I13     | 0.9564 |   0.00 |    0.00 |      0.00 |                 -0.0036 |
```

## Hypothesis evaluation

**Pre-cron-11 hypothesis (from chain v9 closure notes).**  Weight-space
averaging would recover the discretization loss that vote_majority_bits
suffers — continuous prediction preserved through averaging vs hard
2-of-3 thresholded vote.  Predicted Soup I10 ≥ 0.9941.

**Result.**  FALSIFIED.  Soup I10 landed -0.0193 below vote_majority_bits.

**Operative reasons.**

1. Wortsman 2022 boundary condition violated.  Soup gains documented in
   the paper are for same-recipe pools with varied LR / WD / seed.  This
   pool mixes 2 in-basin same-recipe members + 1 cross-basin KD member.
   KD_v7 carries a regulariser-shifted weight surface from the alpha=0.3
   KL pressure, and the 3-way mean drifts away from the iter116J basin.

2. Per-bit ceiling lock.  Three of four single cells already at F1=1.0 in
   vote_majority_bits.  Weight averaging on a ceiling cell is strictly
   downward; per-cell decomposition shows -0.04 to -0.06 on the four
   single cells where vote=1.0.  The structural ceiling caps soup gain
   at ≈ +0.005 (best case in-basin same-recipe soup over 4 seeds), too
   small to offset the single-cell drift cost.

## KD_v10 train FAIL detail

- Trigger: external Python processes (chain v10 phase 2 cutmix-p sweep
  at seed=42 dispatched simultaneously per supervisor scheduler) pushed
  sys_ram to 91 %, exceeding the chain v10 dispatcher's 85 % guard
  (which was set conservatively post-chain-v9 KD_v9 stall, but did not
  trigger a pre-emptive pause on the KD_v10 train because the spike
  came from a separate process tree).
- Duration to fail: ~9 min from dispatch to OOM (CUDA out-of-memory on
  the first batch after the host process page-faulted into swap).
- Recovery: no checkpoint, no parquet, no train log retained.
- Effect on the chain v10 plan: KD_v10 will need re-dispatch under a
  proper GPU + sys_ram joint gate.  Not within cron 11 budget.

## Champion stability table

```
| cron tick | chain  | champion cell                     | bit_F1 | Total FAR | status                  |
|-----------|--------|-----------------------------------|--------|-----------|-------------------------|
| cron 3    | v6.3   | iter116J_clone_s77 I10            | 0.9786 |      0.76 | micro-win               |
| cron 4    | v7.1   | ensemble vote_majority_bits I10   | 0.9941 |      0.00 | NEW CHAMPION            |
| cron 5    | v8.1   | ensemble vote_majority_bits I10   | 0.9941 |      0.00 | re-confirmed            |
| cron 6    | v9.0   | (re-eval in flight, no new cell)  |      - |         - | gpu-gated               |
| cron 7    | v9.0   | (KD_v8 landed, not pool-eligible) | 0.9941 |      0.00 | champion unchanged      |
| cron 11   | v10.0  | Model Soup 3-way I10              | 0.9941 |      0.00 | champion unchanged      |
```

## Cron 11 progress on chain v10 phases

- Phase 0: Model Soup 3-way uniform — **DONE this cron**, result above.
- Phase 1: cmp015 (cutmix_p=0.15 LS=0.5 seed=42) — train DONE 13 min,
  eval in progress, parquet not yet landed at cron 11 read time.
- Phase 2: KD_v10 (alpha=0.3 T=1 skip-cm) — **FAIL this cron**, OOM at
  9 min into train.  Will need re-dispatch with sys_ram joint gate.

## Recorded artifacts

- Iter file: `iters/iter_v10_01_model_soup.md` (NEW)
- Results timeline: `RESULTS_TIMELINE.md` B-table E13 + E14 rows (Soup)
- 02_results timeline: `02_results.md` cron 11 entry prepended
- CSV: `tables/all_runs_n2000.csv` rows v10/I3,I7,I10,I13 (4 rows)
- Paper narrative: `paper/05_experiments.md` chain v10 cron 11 subsection (NEW)

## Lessons recorded

1. Model soup over heterogeneous (mixed in-basin + cross-basin) members
   underperforms vote_majority_bits on this pool — Wortsman 2022 boundary
   condition is binding for this method.
2. Do not re-spend on soup over the same 3 students.  If pursuing soup,
   use same-recipe pool (e.g. iter116J s=1/11/23/77 from chain v6).
3. Even best-case in-basin soup is capped at ≈ max(member_I10) + 0.005
   by per-bit ceiling lock, below the chain v8 ensemble champion 0.9941.
   Model soup is a dominated aggregator for this pool size and ceiling
   regime.
4. KD_v10 train guard: future KD train dispatches need joint
   (GPU-memory + sys_ram) gating, not GPU-only.  The chain v10
   sys_ram-85 % guard exists but did not pre-empt mid-train OOM from a
   separate-process-tree RAM spike.
