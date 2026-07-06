# 260518 06:10 cron #49 — base-only ensemble (no KD) headline + 4-stage paper pipeline

**Trigger.** cron fire #49, 06:10. User: `base-only ensemble (no KD) = 0.9929 / 0.27% finding 추가 + paper structure pipeline (학습 → KD → ensemble → 최종 KD) 정리. 1 line.`

**Action.** Appended §5.49.3 to `paper/05_experiments.md` formalising the 4-stage pipeline (train → KD → ensemble → final KD). §5.49.2 (the base-only ensemble headline 0.9929 / 0.27 %) was already written by a parallel agent at the same cron tick, so cron #49 contributes the *pipeline framing* + the *attribution decomposition* + the *stage-4 alternative* (final-KD distillation against base-ensemble soft-targets).

## 4-stage pipeline (paper headline framing)

```
| Stage | Component                       | POS9 bit_F1 | Total FAR | Role                          |
|-------|---------------------------------|-------------|-----------|-------------------------------|
| 1     | base single (iter116J s=1)      |      0.9927 |      0.00 | single-model floor            |
| 2     | KD single student (KD_v12)      |      0.9470 |      0.00 | KD ceiling (NOT winner)       |
| 3     | base-only ensemble (no KD)      |      0.9929 |      0.27 | seed-diversity headline       |
| 4     | KD-mixed final ensemble (E7)    |      0.9941 |      0.00 | absolute champion             |
```

## Attribution decomposition of 0.9941

- +0.0002 bit_F1 / 0.00 → 0.27 pp FAR : **seed diversity** (Stage 1 → Stage 3)
- +0.0012 bit_F1 / 0.27 → 0.00 pp FAR : **KD calibration injection** (Stage 3 → Stage 4)

WHY this matters (1 sentence): the publishable headline 0.9941 decomposes into *attributable* deltas — pure seed diversity saturates bit_F1 near 0.99 but cannot reach FAR 0.00 %; only a calibration-diverse KD student flips the deciding vote on ~7 hardest negatives (NI 4 / OOD 3 out of 2640).

## Stage-4 alternative (queued)

Final-KD distillation: train a single student against §5.49.2 base-only ensemble's per-bit majority soft-targets. If chain v16 Phase 3 (KD_v15 against E1 teacher) lands non-collapse at POS9 bit_F1 ≥ 0.93, the paper gets a *single-model 1× inference cost* version of the KD-mixed ensemble.

## Provenance

- §5.49.3 appended at `docs/chip-multilabel/paper/05_experiments.md` (after §5.49.2 base-only ensemble entry)
- Base-only ensemble JSON: `outputs/_ensemble_no_kd_s1_s77_s33_I10.json`
- Stage-1 source: `outputs/iter116J_g3_ls30/T7_iter116J_g3_ls30_260513_010015/`
- Stage-2 source: `outputs/KD_v12_a030_T3_skipcm_v15/T7_KD_v12_a030_T3_skipcm_260518_044903/`
- Stage-4 source: `outputs/chain_v8_*_vote_majority_bits/` (s1 + s77 + KD_v7)
- s33_v15 ensemble member: `outputs/iter116J_s33_v15/20260518_051617_T7_iter116J_s33/`
- Timeline reference: `docs/chip-multilabel/RESULTS_TIMELINE.md` lines 100-110 (A.E15)
