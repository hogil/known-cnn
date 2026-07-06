# cron #87 — E21 4-way bit-vote champion FROZEN FINAL (260518 12:46)

## Status
Champion **E21** (`ens_4way_3strong_KDv7_LS20s77_FINAL_CHAMPION`) frozen as paper final headline. Pool `{iter116J_s1, iter116J_clone_s77, iter116J_g3_ls20_s77_v17, KD_v7}`, aggregator `vote_majority_bits`, k = 2 / 4, cell I10, eval v15direct n = 2000.

## Final numbers (locked)
- POS9 bit_F1 = **0.9953**
- NI-FAR = 0.00 %
- OOD-FAR = 0.00 %
- Total FAR = **0.00 %**
- Inference cost = 4 ×
- Δ vs E7 chain v7 champ (0.9941 / 0 %): +0.0012 bit_F1 at matched zero FAR
- Δ vs single-model SOTA iter116J s = 1 (0.9927 / 0 %): +0.0026 bit_F1
- Δ vs revoked iter39 4-bag n = 2000 reverify (0.9555 / 4.05 %): +0.0398 bit_F1, -4.05 pp FAR

## Verification of consistency across paper artefacts
- `abstract.md` §9.7 pointer block (lines 633+ region) — already at 0.9953 / 0 % cron #85 headline (consistent).
- `09_conclusion.md` §9.7 — added FROZEN FINAL annotation (cron #87 12:46) confirming downstream pointers.
- `05_experiments.md` §5.49.4 — already at 0.9953 / 0 % NEW CHAMPION cell (consistent).
- `06_analysis.md` §6.32.9 — bit-vote vs logit-avg mechanism analysis (consistent).
- `tables/paper_main_headline.csv` — row `ens_4way_3strong_KDv7_LS20s77_FINAL_CHAMPION` already labelled "FINAL CHAMPION (frozen 260518 12:46)" (consistent).
- `RESULTS_TIMELINE.md` row E21 — already at 0.9953 / 0 % NEW CHAMPION (consistent).

## Action this cron
- Single edit to §9.7 (`09_conclusion.md`): one-line FROZEN FINAL lock paragraph timestamping cron #87 12:46.
- No new narrative section.
- No new metric.
- Abstract pointer is already consistent with the frozen headline — no further edits.

## WHY (single-sentence)
The 0.9953 / 0.00 % number has been the unchanged headline across crons #85 / #86 / #87 (16 + 16 min of stability); freezing it as the paper's final operating point provides downstream sections an explicit lock anchor without changing any numeric content.
