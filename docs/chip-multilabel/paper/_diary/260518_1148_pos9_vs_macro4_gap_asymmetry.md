# 260518 11:48 cron #80 — POS9 vs macro_4 gap asymmetry recorded

## Recorder input
- `POS9 gap = -0.2225` (headroom on strict positive macro over single + 2-combo cells)
- `macro_4 gap = -0.0504` (headroom on 4 single-defect cells)
- Ratio: **4.41×** — 2-combo cells carry ~4× the residual error mass of single cells on the single-teacher per-seed pool.

## Narrative claim
Single-teacher per-seed calibration is the binding constraint on compositional learning in the 4-class chip multi-label saturated regime. Per-seed members trained from the same iter116J basin saturate the single cells but produce *correlated* per-bit errors on 2-combo cells, because every seed inherits the same teacher's per-bit calibration. `vote_majority_bits` extracts only the diversity present in the pool, so it cannot lift 2-combo cells past the teacher's per-bit ceiling — which is consistent with the §6.32.2 +0.0014 bit_F1 ensemble lift being a small fraction of the -0.2225 POS9 gap.

## WHY this matters for chain v12
- §6.32.6 item 2 in-basin seed additions (Phase 3-5 s33, s55) predicted to saturate at the σ ≈ 0.030 seed band, not at the -0.2225 POS9 gap.
- Cross-teacher members (chain v7 KD_v7 cross-basin distillation, chain v12 g2_ls030 cross-FCM-PM-gain perturbation) are the **only** path to reduce 2-combo cell error correlation.
- Consistent with §6.32.4 Model Soup falsification (-0.0193 bit_F1) and §6.32.6.1 BCE multi-label single-point LS viability: both fail because they operate on members sharing teacher calibration.

## Negative result framing (paper-worthy)
The 4× gap asymmetry is itself a paper finding: it explains why the chain v6-v12 ensemble headroom is bounded by teacher-calibration correlation rather than by seed variance. The §7 discussion now frames the 4-class chip multi-label benchmark as a **cross-teacher diversity limited regime** rather than a per-seed sampling limited regime.

## Append location
- §6.32.6.7 in `docs/chip-multilabel/paper/06_analysis.md` (inserted before §6.32.7 n=200 vs n=2000 artifact section to preserve chain v12 chronological narrative).

## Status
- Champion unchanged: iter116J single 0.9927 / 3-way vote ensemble 0.9941.
- No new metric on headline cells — finding is mechanistic explanation only.
- Cites: §6.32.2 (ensemble champion), §6.32.4 (Model Soup falsification), §6.32.5 (KD α basin narrowness), §6.32.6.1 (BCE multi-label LS single-point viability), §6.7.5 (σ ≈ 0.030 seed band calibration).
