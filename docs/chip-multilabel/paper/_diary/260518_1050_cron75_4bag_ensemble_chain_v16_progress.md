# cron #75 — 4-way base-only ensemble + chain v16 in-progress

**Time:** 2026-05-18 10:50

## Result

```
| Mode (4-way s1+s77+s33_v15+s99) | bit_F1 | NI-FAR | OOD-FAR | Total FAR |
|---------------------------------|--------|--------|---------|-----------|
| k>=2 (majority-bits floor)      | 0.9937 |   0.35 |    0.16 |      0.27 |
| k>=3 (strict-AND tightened)     | 0.9863 |   0.00 |    0.00 |      0.00 |
```

## Reading

- 4-way k>=2 essentially ties §5.49.2 3-way 0.9929/0.27 % (+0.0008 bit_F1, same FAR floor). Adding the s99 fourth base seed widens diversity but does not break the calibration ceiling.
- 4-way k>=3 reaches Total FAR 0.00 % via pure seed diversity (no KD) at −0.0074 bit_F1 vs k>=2. This matches the §5.49 chain-v8 KD-mixed E7 0.9941/0.00 % FAR-side outcome with no KD member required — isolates KD-calibration contribution at the ensemble stage as **+0.0078 bit_F1 lift at FAR-parity** (vs §5.49.3 estimate +0.0012 which was 3-way vs E7 mixed-population).
- WHY: refines the cron #49 4-stage pipeline decomposition — Stage 3 with 4 base seeds + strict-AND now matches Stage 4's FAR contribution, attributing the residual +0.0078 bit_F1 specifically to KD's soft-target calibration rather than seed diversity quantity.

## Chain v16 progress

- Phase 1 (E1 teacher train, iter116J recipe seed=1): complete.
- Phase 2 (KD_v15 student, α=0.30, T=3, `--kd-skip-on-cutmix`, grad-checkpointing, E1 teacher): in progress.
- Phase 3 (eval n=2000 POS9 strict): queued.
- Tests whether KD ceiling tracks single-teacher quality (iter116J s=1 = 0.9927) vs the chain v8 3-run averaged teacher (cap ~0.9470 at KD_v12).

## Next

- On chain v16 Phase 3 land: cross-reference KD_v15 single-student F1 vs KD_v12 0.9470 to confirm/refute the teacher-quality hypothesis.
- If KD_v15 ≥ 0.94, the §5.49.3 Stage-4 alternative single-model path (base-ensemble → final-KD) becomes the publishable one-third-cost reformulation of the KD-mixed ensemble.

_Source: 4-way ensemble at `outputs/_ensemble_4bag_*`; chain v16 phases at `outputs/chain_v16_01_E1_teacher_iter116J_s1/` and `outputs/chain_v16_02_KD_v15_E1_a030_T3/`; appended to §5.49.3 in `docs/chip-multilabel/paper/05_experiments.md`._
