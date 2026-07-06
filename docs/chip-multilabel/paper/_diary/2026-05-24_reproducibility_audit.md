# 2026-05-24 — mega_matrix reproducibility audit note

## Trigger
Cron #959 (10min auto re-trigger). Active topic:
mega_matrix github reproducibility analysis.

## New data (since last narrative update)
- Variant I13 was the missing piece in
  `run_single_sota.sh` (which dispatched I10 only).
- v5 OOD chip distribution diverged from v12 generator
  default → +1.97% Total FAR delta on otherwise
  identical recipe.
- I10 + v5 OOD reproduces single-var/single-recipe
  bit-F1 0.9919 / Total FAR 0.00%.
- I13 eval in progress, expected to confirm exact
  0.9927 / 0.00% headline record.

## Paper change
Appended `§7.14 Reproducibility hazards beyond
hyperparameters and seed` to
`docs/chip-multilabel/paper/07_discussion.md`. WHY:
the two hazards (decision-rule variant ID drift,
synthetic generator version drift) are first-class
reproducibility hazards that the standard checklist
(seed + hyperparameter pinning) does not cover, and
both were *required* to recover the published
single-model record. The lesson generalises beyond
this dataset and belongs in Discussion, not in an
iter-level §5/§6 entry.

## Cited references (added to paper)
- Pineau et al. 2021 arXiv:2003.12206
  (reproducibility checklists)
- Bouthillier et al. 2021 arXiv:2103.03098
  (seed-and-data variance)

## Open items (not paper-side)
- I13 eval confirmation pending — once 0.9927/0.00%
  exact match lands, §7.14 can drop "in progress"
  qualifier and cite the eval log path.
- Generator version field in `eval_manifest.json`
  is a code-side change, owned by runner/logger
  agents, not narrator.

## Skipped
- No §5 / §6 iter-level append — the audit is a
  meta-finding, not a new iter result.
- No abstract / §1 / §3 / §4 edit — discussion-only
  scope.
