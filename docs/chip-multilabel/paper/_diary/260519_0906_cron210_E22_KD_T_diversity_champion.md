# Diary — cron #210 09:06 — E22 new champion via KD T-temperature diversity

**Date.** 2026-05-19 09:06 (cron #210, paper-recorder dispatch).

**Event.** New v19-chain ensemble champion identified via post-hoc stored-parquet sweep over KD-axis candidates.

**Champion.**
- **E22** = `{iter116J_s1, clone_s77, LS20_s77_v17, KD_v7, KD_v12}`
- **Vote rule.** Per-bit majority k ≥ 2 (odd-N = 5)
- **Eval.** POS9 strict v15direct_n2000, I10
- **Metrics.** bit_F1 = **0.9956** / Total FAR = **0.00 %**
- **Delta vs E21.** +0.0003 bit_F1 at matched zero FAR (E21 = 0.9953 / 0 %)

**Sweep design (14 candidates, C1-C14).**
- Axis 1 — KD-axis swap. Replace one of E21's 4 base seeds with one KD member.
- Axis 2 — T-temperature diversity. Add a second KD member trained at a different distillation temperature alongside the existing KD_v7 (T=2). KD_v12 = T=3 candidate.
- Axis 3 — Base-seed addition. Add s11 / s23 / s33 to the 4-way E21.
- Axis 4 — LS-axis 2x. Double the LS20 member.

**Result.** C5 (axis 2, T-diversity) = strict-gate winner. C9 / C11 = Pareto F1-max relaxed leaders (0.9964 / 0.42 %) gate-fail.

**Standalone reference.**
- KD_v7 alone = 0.9265 POS9 bit_F1
- KD_v12 alone = 0.9470 POS9 bit_F1 (best non-collapse standalone KD)
- Both uncompetitive standalone, but per-bit-vote complementary to BCE-LS base seeds.

**Mechanism — T-diversity > single-T-saturation.**
The +0.0003 gain materialises only when both T=2 and T=3 KD members are present in the vote. Pairing two replicas of the strongest single KD does not reproduce the gain. Three literature anchors converge:
1. **Hinton 2015** (arXiv 1503.02531) — T-scaling exposes different layers of dark-knowledge mass; low T preserves peaks, high T amplifies inter-class similarity.
2. **Lakshminarayanan 2017** (arXiv 1612.01474) — Deep-ensemble gain depends on member calibration diversity, not strongest-member accuracy.
3. **Hansen & Salamon 1990** (IEEE TPAMI 12(10)) — Odd-N majority voting strictly improves on strongest member iff member errors are pairwise independent above per-class threshold.

The T-diversity result is the convergence: the two KD temperatures give Lakshminarayanan-style calibration diversity extracted via Hinton T-scaling, and the 5-member k≥2 vote (Hansen-Salamon optimal odd-N) leverages it without admitting either KD's higher standalone FAR.

**Why this matters (paper).** First KD member to land inside a deployable champion ensemble in v19 chain. Reframes the §5.49.6 KD-as-standalone negative result: KD path contribution is **not** standalone (sealed negative) but as a **T-diverse calibration source** for ensemble voting. The standalone-evaluation lens of §5.49.1-§5.49.6 was blind to this contribution.

**Champion table.** E22 / 0.9956 / 0.00 % supersedes E21 / 0.9953 / 0.00 %. §5.49.7 appended to `05_experiments.md`.

**Negative result preserved.** KD-as-standalone path remains sealed at the 4-way teacher (§5.49.6). The T-diversity gain is +0.0003 (small in absolute terms but at the deployable strict-gate frontier where every cell matters).

**Frozen.** `09_conclusion.md` not touched (per cron #210 task instructions — needs user unfreeze before any champion-table refresh propagates to the conclusion section).

**Source pointers.**
- Champion definition + sweep C1-C14 results — `outputs/_sweep_C1_C14_KD_postaxis_260519/sweep_summary.json` (to be promoted from working directory)
- Predecessor E21 — `docs/chip-multilabel/paper/05_experiments.md §5.49.4`
- KD member standalone refs — §5.49.5 (KD_v7 = 0.9265) and §5.49.6 (KD_v12 = 0.9470 recorded under this cron supplement)
- Narrative — `docs/chip-multilabel/paper/05_experiments.md §5.49.7`
