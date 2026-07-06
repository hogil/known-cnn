# 260517 21:38 — chain v12 dispatch + unified findings synthesis

**Trigger.** Chain v12 Phase 1 BCE_ls00_baseline `best_model.pth` saved
(training done); n2000 eval queued. Phases 2-8 in serial dispatcher
queue. Champions unchanged: single iter116J s=1 / I10 = 0.9927 /
0.00 % FAR; ensemble chain v8 vote_majority_bits 3-way = 0.9941 /
0.00 % FAR.

**Chain v12 phase queue (linear, 1 GPU, 8 phases).**

```
| Phase | Tag                       | Axis                              | Status         |
|-------|---------------------------|-----------------------------------|----------------|
| 1     | BCE_ls00_baseline         | LS = 0, multi-label BCE baseline  | trained, eval q|
| 2     | BCE_ls02                  | LS = 0.20                         | queued         |
| 3     | s33                       | seed = 33 (in-basin)              | queued         |
| 4     | s55                       | seed = 55 (in-basin)              | queued         |
| 5     | g2_ls030                  | FCM-PM g = 2 (cross-basin)        | queued         |
| 6     | KD_v11                    | alpha = 0.25, T = 2               | queued         |
| 7     | KD_v12                    | alpha = 0.30, T = 3               | queued         |
| 8     | KD_v13 / v14              | alpha = 0.30 T=4 / 0.35 T=2.5     | queued         |
```

**Paper sections updated.**

1. `paper/05_experiments.md` — appended `### chain v12 — BCE
   baseline, ensemble member diversity, KD alpha corner sweep
   (260517, in progress)` after the chain v10 Model Soup
   subsection. Includes: phase plan motivation, WHY each phase
   (closing 3 ablation gaps left by v5-v10), expected outcomes
   prior, and the headline-metric snapshot table (5-row code
   block, CLAUDE.md 260515 format compliant).

2. `paper/06_analysis.md` — appended `## §6.32 Chain v6-v12
   unified findings — what works, what doesn't, and why (260517)`
   after §6.31. Seven subsections:
   - §6.32.1 motivation (saturation regime characterisation)
   - §6.32.2 baseline progression (chain v5 → v8 → 0.9927 / 0.9941)
   - §6.32.3 what works (vote_majority_bits, KD alpha=0.3,
     FCM-PM g=3 + pair-mask)
   - §6.32.4 what doesn't (Model Soup, KD out of [0.25, 0.35],
     ASL, top-2 truncate)
   - §6.32.5 KD alpha corner sweep narrowness
   - §6.32.6 chain v12 systematic ablation closing 3 gaps
   - §6.32.7 connection to paper §5/§6/§7 narrative

**Metric snapshot at time of paper update (CLAUDE.md 260515
format, POS9 strict, Total FAR 260512 rule).**

```
| Rank | Recipe                                            | Variant | bit_F1 | NI-FAR | OOD-FAR | Total FAR | Status         |
|------|---------------------------------------------------|---------|--------|--------|---------|-----------|----------------|
| 1    | chain v8 3-way vote_majority_bits (s1+s77+KDv7)   | I10     | 0.9941 |   0.00 |    0.00 |      0.00 | ensemble champ |
| 2    | iter116J single (s=1, T7 LS=0.30 g=3, FCM-PM)     | I10     | 0.9927 |   0.00 |    0.00 |      0.00 | single champ   |
| 3    | chain v8 logit_avg (s1+s77+KDv7) + entropy gate   | I10     | 0.9935 |   0.00 |    0.00 |      0.00 | continuous     |
| 4    | chain v10 soup uniform mean 3-way                 | I10     | 0.9748 |   0.00 |    0.00 |      0.00 | soup regress   |
| 5    | chain v6 KD_v7 (alpha=0.3 T=2)                    | I10     | 0.9786 |   0.00 |    0.00 |      0.00 | KD viable point|
| 6    | chain v12 Phase 1 BCE_ls00_baseline               | -       | -      |      - |       - |         - | eval pending   |
| 7    | chain v12 Phase 2 BCE_ls02                        | -       | -      |      - |       - |         - | queued         |
| 8    | chain v12 Phase 3-5 s33 / s55 / g2_ls030          | -       | -      |      - |       - |         - | queued         |
| 9    | chain v12 Phase 6-8 KD_v11-v14                    | -       | -      |      - |       - |         - | queued         |
```

**Next paper update trigger.** Phase 1 BCE_ls00_baseline n2000 eval
completion → recorder appends row to `tables/all_runs_n2000.csv` →
narrator appends 1-paragraph "what we learned" footnote to the
chain v12 subsection in §5 (LS=0 multi-label drop quantified vs
T7 LS=0.30 baseline; whether the §6.1 LS curve transfers to
multi-label per-bit BCE).

**Files written.**

- `D:/project/known-cnn/docs/chip-multilabel/paper/05_experiments.md`
  (chain v12 subsection appended)
- `D:/project/known-cnn/docs/chip-multilabel/paper/06_analysis.md`
  (§6.32 unified findings appended)
- `D:/project/known-cnn/docs/chip-multilabel/paper/_diary/260517_chain_v12_dispatch.md`
  (this entry)
