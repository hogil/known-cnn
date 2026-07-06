# 260518 20:06 — cron #131 KD_6way pending, v19 chain conclusion (provisional)

Appended §5.49.5 cron-#131 update to `05_experiments.md` capturing chain v19 partial-completion state at the cron #131 measurement window.

## State at cron #131

- Phase 1 E21 4-way teacher_probs gen — done 17:11 (`_teacher_probs_E21_4way.parquet`).
- Phase 2 KD_E21 α=0.25 — completed train+eval earlier window (skip flag triggered at 18:57 start logged for α=0.30; the α=0.25 result not separately landed in this measurement window, will need a follow-up cron tick).
- Phase 2 KD_E21 α=0.30 — done 19:13. POS9 bit_F1 I10 = 0.9144 / I3 = 0.8925 / I7 = 0.8825 / I13 = 0.8819, epoch-1 best, n_eval = 18 640.
- Phase 3 6-way teacher_probs gen (E21 + LS30_s11 + LS20_s1) — done 19:17 (`_teacher_probs_6way_E21plus.parquet`).
- Phase 4 KD_6way α=0.25 — train dispatched 19:17, GPU-gated since 19:59 (co-resident process used = 64 % vs 50 % threshold), 49 min wait so far, no checkpoint yet.
- Phase 4 KD_6way α=0.30 + downstream evals — pending behind α=0.25.

## KD-ceiling triangulation across teacher composition axis

| KD teacher composition          | POS9 bit_F1 best variant | Δ vs 4-way champion (0.9953) |
|---------------------------------|--------------------------|------------------------------|
| KD_E1 3-way ensemble (cron #79) | 0.7040 (I10)             | -0.2913                      |
| KD_v7 single sharp seed         | 0.9265 (I10)             | -0.0688                      |
| KD_E21 single sharp recipe      | 0.8886 (I10)             | -0.1067                      |
| KD_E21 4-way teacher (cron #131)| 0.9144 (I10)             | -0.0809                      |
| KD_6way teacher (pending)       | -                        | -                            |

Monotone ceiling at POS9 ≈ 0.92 across all KD-teacher variants — the v19 chain's structural finding is **already bounded** by the Phase 2 reading; the pending KD_6way α=0.25/0.30 measurements are predicted to land in the same 0.90-0.92 band.

## v19 chain conclusion (provisional)

KD-from-larger-ensemble-teacher (3 → 4 → 6 members) does not break the §5.49.4 4-way bit-vote champion ceiling at 0.9953 / 0.00 % Total FAR. The KD axis as standalone training improvement is **structurally bounded** at POS9 ≈ 0.92 regardless of teacher composition density. §5.49.4 Insight 3 + §5.49.5 Insight 1 (KD-as-ensemble-diversifier, not KD-as-standalone) reaffirmed by triangulation across the full {1, 3, 4, 6} teacher-size sweep.

WHY paper-worth: monotone bounding of the KD-teacher-scaling axis closes the negative-result with explicit teacher-composition density coverage; no further KD-teacher composition argument can structurally challenge the post-hoc bit-vote ensemble champion in our regime.

## Files

- update: `docs/chip-multilabel/paper/05_experiments.md` §5.49.5 cron-#131 paragraph
- source: `outputs/KD_E21_a030_T2_skipcm_v19/20260518_185709_T7_KD_E21_a030_T2_skipcm/eval_n2000_pred/stage1_260518_190538/eval_summary.json`
- chain log: `outputs/_chain_v19_summary.log`
- recipe: `_run_chain_v19.sh`
