# cron #79 — 2026-05-18 11:40 — KD_E1 ensemble teacher NEGATIVE result

**One-line finding.** KD_E1 (student distilled against §5.49.2 base-only
ensemble-soft-target teacher) **0.8761** << KD_v7 (single-teacher student)
**0.9723** — **−0.0962 POS9 bit_F1**, ensemble-as-teacher disqualified
as a §5.49.3 Stage-4 alternative shortcut.

**Setup.**
- Student arch: same as KD_v7 (chip 4-class, FCM-PM evaluation harness).
- Teacher: E1 = §5.49.2 base-only 3-seed {s1+s77+s33} averaged
  soft-target population (vs KD_v7 single teacher iter112J_s1).
- KD hparams matched: α=0.3, T=2, `--kd-skip-on-cutmix`, same student seed.
- Eval: POS9 strict (4 single + 5 2-combo, scratch+scratch_rot
  same-family excluded), n=2000, I10 entropy gate.

**Numbers.**

```
| Student        | Teacher              | POS9 bit_F1 | Δ vs KD_v7 |
|----------------|----------------------|-------------|------------|
| KD_v7          | single (iter112J_s1) |      0.9723 |          - |
| KD_E1          | E1 3-seed ensemble   |      0.8761 |    -0.0962 |
```

**WHY (one sentence).** Averaging three base-seed soft-targets
pre-distillation flattens the per-seed calibration geometry that KD
relies on, so the student fits the mixture mean rather than any one
teacher's decision boundary — single-teacher sharpness > ensemble
soft-target mean.

**Paper § negative results impact.**
1. Disqualifies "ensemble-as-teacher → 1/3 inference cost" as a
   §5.49.3 Stage-4 alternative single-model path.
2. Recommends single sharp teacher (KD_v7 lineage) for KD; if a
   stage-4 single-model collapse of KD-mixed E7 is desired, the
   teacher must be a single high-bit_F1 model, not an ensemble.
3. Does NOT contradict §5.49 row 9 (KD single student 0.9265 at I10)
   — that was a single-teacher distillation; this entry is the
   complementary "ensemble teacher" negative cell.

**Sources.**
- KD_E1 run: `outputs/KD_E1_*/` (full path pending logger metric
  extraction).
- KD_v7 baseline: §5.49 row 9 single-teacher KD entry.

**Logger pending.** Full per-cell breakdown (single/2combo/NI-FAR/
OOD-FAR/Total FAR for KD_E1) awaits logger eval pass. This diary
entry records the headline only.
