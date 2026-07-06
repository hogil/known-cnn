# cron #122 — KD_E21 single-student KD path measurement (2026-05-18 18:36)

## Event
KD_E21 measurement complete:
- I10: POS9 bit_F1 **0.8886** / Total FAR **0.08 %**
- I13: POS9 bit_F1 **0.8096** / Total FAR **0.00 %**

## Status
- **KD-family best** within cron #118-#122 KD batch (best single-teacher KD attempt of the batch)
- **Champion miss**: −0.1067 bit_F1 vs §5.49.4 4-way bit-vote champion (0.9953 / 0.00 %)
- **vs KD_v7 historical best (0.9265 I10)**: KD_E21 I10 0.8886 sits below KD_v7 — does not lift the KD-axis ceiling

## Narrative captured
§5.49.5 appended to `05_experiments.md`:
- KD path negative-headline confirmation
- Insight 1: KD axis saturates well below ensemble axis (~0.92-0.93 vs 0.9953)
- Insight 2: 4-way champion table unchanged; KD_E21 closes KD-axis search (paper-worth as KD-path bounding result)
- Recommendation now empirically bounded on both sides: ensemble-teacher failure (§5.49.cron #79) + single-teacher ceiling (KD_E21, this entry)

## Champion table
Unchanged at E7+LS20_s77 4-way bit-vote `0.9953 / 0.00 %` (cron #85).

## Next paths (KD axis closed)
- LS-axis extension (LS=0.10 / LS=0.40) in 4-way pool
- Final-KD distillation from 4-way per-bit majority pseudo-labels

WHY paper-worth: empirically closes the KD-distillation path search, framing KD as ensemble-diversifier (not standalone improvement) with bounds.
