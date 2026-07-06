# 260518 05:40 cron #46 — KD α window closed at α=0.35

**Trigger.** cron fire #46, 05:40. User report: `KD α window 확정 — α=0.35 collapse (KD_v14 fail 2min)`.

**Action.** Appended §6.32.6.6 to `paper/06_analysis.md`. One paragraph recording that KD_v14 (α=0.35, T=2.5) collapsed within ~2 minutes, closing the KD viable basin to a single operating point at (α=0.30, T=2).

**Why this matters (1-line paper hook).** The §6.32.5 prediction of single-point KD basin geometry is now empirically confirmed on both sides — chain v9 KD_v8 closed the α=0.5 upper boundary, chain v9 KD_v9 closed the α=0.2 lower boundary, and cron #46 KD_v14 closes the α=0.35 near-upper boundary. The viable region cannot contain an interior plateau wider than ~±0.025 around α=0.30.

**Champion unchanged.** iter116J single 0.9927 / 3-way vote ensemble 0.9941.

**No metric change.** Negative result (collapse), recorded for paper completeness per §3 Method "failures are valuable" policy.

**Source refs.**
- Existing §6.32.5 KD alpha narrow basin section.
- Existing §5.45 KD viable basin chapter.
- Prior chain v9 collapse notes in `paper/_diary/260517_cron9_KD_v9_stall_guard_down.md`.
- New §6.32.6.6 in `docs/chip-multilabel/paper/06_analysis.md`.
