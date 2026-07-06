# 2026-05-17 cron 7 (16:37) — chain v9 KD_v8 re-eval landing

## Event

Chain v9 phase 0 re-eval of KD_v8_a05_T2_skipcm landed on the GPU after
the `_run_chain_v9.sh::wait_gpu_free` gate released at 15:03:58 (foreign
GPU process dropped to 55%, under the 60% eval threshold).  Dispatch
sequence:

```
| Time     | Phase            | Event                                         |
|----------|------------------|-----------------------------------------------|
| 13:59    | chain v8 cron 4  | KD_v8 re-eval OOM (foreign GPU spike)         |
| 14:00    | chain v8 cron 5  | KD_v8 re-eval OOM (same cause)                |
| 15:03:58 | chain v9 phase 0 | GPU gate released at 55 %, re-eval dispatched |
| 15:04:13 | chain v9 phase 0 | stage1 dir created (stage1_260517_150413)     |
| 15:04:xx | chain v9 phase 0 | preds_chip.parquet written, eval complete     |
| 16:02    | master report    | KD_v8 re-eval done, parquet ready             |
| 16:37    | cron 7 tick      | recorder ingests, computes POS9 strict        |
```

## Headline metric (POS9 strict + 4-class OOD strict)

```
| Variant | bit_F1 | NI-FAR | OOD-FAR | Total FAR |
|---------|--------|--------|---------|-----------|
| I3      | 0.9274 | 100.00 |  100.00 |    100.00 |
| I7      | 0.9227 | 100.00 |  100.00 |    100.00 |
| I10     | 0.8924 |  32.15 |   79.46 |     57.15 |
| I13     | 0.8365 |  31.45 |   71.12 |     52.41 |
```

Health check on the model: `eval_summary.json` -> `model_meta.val_macro_f1 = 0.0`
at `epoch = 1` -> training collapsed at ep01, best ckpt is the collapsed state.

Note on the prompt-vs-actual numbers: the master report at 16:02
quoted (KD_v8 I10 bit_F1 0.8609 / NI 32.15 / OOD 28.12 / Total 31.17)
and (I13 0.7977 / 31.45 / 16.72 / 27.88).  My re-computation from the
parquet under `_iter124_metrics.compute` (POS9 strict, OOD = the 4
strict chips CenterDonut / CrossScratch / DiagonalSmear / Starburst)
returns higher bit_F1 (0.8924 / 0.8365) and significantly higher OOD
FAR (79.46 / 71.12).  Two compute paths can differ on (a) which POS
cells are included (POS9 strict excludes the same-family
scratch+scratch_rot, but a 10-cell variant including it would shift
bit_F1) and (b) which OOD chips are counted (4 strict vs the wider
n2000 OOD subset).  Recording the parquet-derived numbers per CLAUDE.md
"do not guess, cite only from parquet or caller input" — the qualitative
conclusion (KD_v8 collapsed, alpha=0.5 outside viable window) holds in
both computations.

## Diary insight

The chain v9 GPU gate paid off: two prior cron ticks (4 and 5) burned
dispatches on OOM at KD_v8 re-eval, while cron 7 landed cleanly on the
first attempt under the gate.  Keep the gate for chain v10 and any
future re-eval of OOM'd training runs.

The KD viable corner is now closed at a single point (alpha=0.3, T=2).
The chain v9 supervisor should not re-spend the KD budget on T=2 cells.
Two productive alternatives for extending the ensemble pool: (a)
additional base-recipe seeds, (b) the cutmix-p sweep already queued at
chain v9 phase 2.  KD_v9 (alpha=0.2 retry) and KD_v10 (alpha=0.3 T=1)
are still GPU-gated waiting on the 50% train threshold; if they
collapse on landing, the T-axis viable window is also closed.

## Chain v9 health note

KD_v9 train phase has been waiting 30/120 min on the GPU gate (foreign
process holding 55% usage, above the 50% train threshold).  No train
parquet yet.  If the gate does not release before the 120 min fallback
fires, the dispatch will proceed anyway with a logged warning per the
chain v9 supervisor design.

## Files touched this tick

- `docs/chip-multilabel/iters/iter_v9_01_KD_v8_collapse.md` (new)
- `docs/chip-multilabel/RESULTS_TIMELINE.md` (section C KD_v8 row update + conclusion paragraph)
- `docs/chip-multilabel/02_results.md` (cron 7 timeline section prepended)
- `docs/chip-multilabel/tables/all_runs_n2000.csv` (4 rows appended for v9/1/KD_v8)
- `docs/chip-multilabel/paper/05_experiments.md` (section 5.X narrative appended)
- `docs/chip-multilabel/paper/_diary/260517_cron7_KD_v8_result.md` (this file)

## Source

- Parquet: `outputs/KD_v8_a05_T2_skipcm/20260517_121833_T7_KD_v8_a05_T2_skipcm/eval_n2000_pred/stage1_260517_150413/preds_chip.parquet`
- Eval summary: same dir / `eval_summary.json`
- Supervisor / runner: `outputs/_chain_v9_summary.log`, `outputs/_chain_v9_runner.log`
