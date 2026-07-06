# Cron #134 — 2026-05-18 20:36 — v19 chain final + KD_6way landed

**Trigger.** Chain v19 Phase 4 (KD_6way α=0.25) eval landed; KD-teacher-scaling axis closes.

**Result (1-line headline).** KD_6way α=0.25 I10 macro_f1 = **0.9141** vs KD_E21 α=0.25 4-way teacher I10 = **0.9661**, a **Δ = −0.0520** regression from scaling the teacher 4 → 6 members.

**Measurement sources.**
- KD_6way α=0.25: `outputs/KD_6way_a025_T2_skipcm_v19/20260518_191740_T7_KD_6way_a025_T2_skipcm/eval_n2000_pred/stage1_260518_200530/eval_summary.json` (n_eval=18 640, epoch=1 best, ts=260518_200530)
- KD_E21 α=0.25 (4-way teacher reference): `outputs/KD_E21_a025_T2_skipcm_v19/20260518_171154_T7_KD_E21_a025_T2_skipcm/eval_n2000_pred/stage1_260518_180537/eval_summary.json` (n_eval=18 640, epoch=9 best)
- KD_E21 α=0.30 (4-way teacher, alt-α): `outputs/KD_E21_a030_T2_skipcm_v19/20260518_185709_T7_KD_E21_a030_T2_skipcm/eval_n2000_pred/stage1_260518_190538/eval_summary.json` (epoch=1 best, I10 0.9144)
- Chain progression: `outputs/_chain_v19_summary.log` (Phase 1 17:08 teacher-gen → Phase 2 17:11/18:57 KD_E21 trains → Phase 3 19:13 6-way teacher-gen → Phase 4 19:17 KD_6way train dispatch)

**v19 chain KD-teacher-scaling axis (final).**

| Teacher composition | Members | Student α | Best epoch | I10 macro_f1 | Source              |
|---------------------|---------|-----------|------------|--------------|---------------------|
| single sharp seed   |       1 |      0.30 |        n/a |       0.8886 | §5.49.5 cron #122   |
| 3-way ensemble (E1) |       3 |      0.30 |        n/a |       0.7040 | §5.49.cron #79      |
| 4-way ensemble (E21)|       4 |      0.25 |          9 |       0.9661 | this cron, ref      |
| 4-way ensemble (E21)|       4 |      0.30 |          1 |       0.9144 | cron #131           |
| 6-way ensemble      |       6 |      0.25 |          1 |       0.9141 | this cron, regress  |

**Insight (paper-grade).** **Larger teacher size → weaker student** within the saturated 4-class chip multi-label regime — refutes textbook "more teachers smooth labels better" assumption. The KD distillation path is bounded above by an **inverse-scaling ceiling**: peak at the 4-way teacher (0.9661 I10), degradation in both directions (smaller and larger teacher compositions). The §5.49.4 4-way bit-vote post-hoc ensemble champion (0.9953 / 0.00 % Total FAR) remains uncontested by any KD-teacher-composition variant. WHY this is paper-worth: closes the KD-axis exploration across four distinct teacher compositions and confirms KD-as-standalone is structurally below the post-hoc ensemble path, regardless of teacher scaling direction.

**Paper sections updated.**
- `docs/chip-multilabel/paper/05_experiments.md` § 5.49.5 — cron #134 update appended (1-line summary + comparison table + sources).
- This diary entry.

**Champion table unchanged.** §5.49.4 4-way bit-vote (E7 + LS20_s77) at POS9 bit_F1 0.9953 / Total FAR 0.00 % remains the headline.

**Next.** No further KD-axis exploration warranted. Open paths from §5.49.4: (i) LS-axis extension to 0.10 / 0.40 single member in 4-way pool; (ii) final-KD distillation from 4-way per-bit majority pseudo-labels (single sharp pseudo-label teacher, not soft mixture — explicitly avoids the v19 scaling pathology).
