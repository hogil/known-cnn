# 260518 05:20 — KD_v12 v15 (α=0.30, T=3) new KD best + α/T window finding

**Trigger.** paper-recorder cron #44 at 05:20. Chain v15 Phase 4
KD_v12 eval completed and was recorded into
`docs/chip-multilabel/tables/paper_main_ablation.csv` (line 13) and
`docs/chip-multilabel/RESULTS_TIMELINE.md` (line 110) prior to this
narrative tick.

**Headline.** KD_v12 (α=0.30, T=3, `--kd-skip-on-cutmix`,
grad-checkpointing) reaches POS9 bit_F1 **0.9470 / Total FAR 0.00 %**
on n=2000 POS9 strict — new single-student KD best, +0.0205 vs §5.49
row 9 (KD_v7 at α=0.30, T=2 = 0.9265). NI-FAR and OOD-FAR both 0.00 %.

**Paper section updated.** `paper/05_experiments.md` §5.49.1 appended
after the §5.49 main ablation table.

**Why (one sentence per design decision).**

1. **WHY append rather than overwrite §5.49 row 9.** The main ablation
   table is sealed at the (α=0.30, T=2) KD entry to preserve
   reproducibility against the recorded CSV; §5.49.1 documents the
   refinement above the canonical row so §5.49 stays citable.
2. **WHY call this an "α/T window" rather than a "new corner".** Two
   neighbouring cells (KD_v11 at T=2 non-collapse, KD_v12 at T=3
   +0.0205) confirm that the viable region is contiguous along the T
   axis at fixed α=0.30, not a single point as chain v7/v8 concluded.
3. **WHY the +0.0205 is attributable to T not to grad-checkpointing.**
   KD_v11 (also grad-checkpointing, same trainer code, T=2) reproduced
   KD_v7's value within ±0.0073, so the checkpointing factor is bounded
   to that noise band; the +0.0205 lift therefore tracks the T axis.
4. **WHY T=3 helps 2-combo cells specifically.** Higher T softens the
   teacher's mode-collapse onto the dominant member of each 2-combo
   pair (e.g. fork+scratch), distributing soft-target mass more evenly
   so the student supervises both members.

**Negative-result branch (still pending).** KD_v13 (α=0.30, T=4) and
KD_v14 (α=0.35, T=2.5) train next; their evals will define the upper T
boundary and the α-side boundary of the window. If T=4 collapses, T=3
is the upper edge and KD_v12 becomes the canonical KD entry replacing
row 9.

**Citation provenance.**

```
| Item            | Path                                                                                                                          |
|-----------------|-------------------------------------------------------------------------------------------------------------------------------|
| KD_v12 preds    | outputs/KD_v12_a030_T3_skipcm_v15/T7_KD_v12_a030_T3_skipcm_260518_044903/eval_n2000_pred/stage1_260518_045541/preds_chip.parquet |
| recorder CSV    | docs/chip-multilabel/tables/paper_main_ablation.csv (line 13)                                                                  |
| timeline row    | docs/chip-multilabel/RESULTS_TIMELINE.md (line 110)                                                                            |
| KD_v11 baseline | line 109 of RESULTS_TIMELINE.md (POS9 bit_F1 = 0.9192 at I10)                                                                  |
| §5.49 row 9 src | outputs/iter116J_g3_ls30/T7_iter116J_g3_ls30_260513_010015/eval_n2000_pred/stage1_260514_161529/preds_chip.parquet (KD_v7)     |
```

**Champion summary (unchanged).** Single iter116J s=1 / I10 = 0.9927
POS9 / 0.00 % FAR; ensemble chain v8 vote_majority_bits 3-way = 0.9941
POS9 / 0.00 % FAR. KD_v12 at 0.9470 sits below both — the
contribution is methodological (α/T window characterisation), not a
headline displacement.
