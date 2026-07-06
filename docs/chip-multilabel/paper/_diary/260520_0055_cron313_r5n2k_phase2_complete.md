# 260520 00:55 cron #313 — r5n2k Phase 2 COMPLETE

**Trigger.** cron #313 00:55. r5n2k (Row 5 n=2000) Phase 2 complete with 22-variant sweep across pair-mask CutMix family and CutMix-only no-pair family. Paper narrative appended to §5.50.

## Sweep composition

- **11 pair-mask CutMix variants** (Row 5-pair A through K) — LS × cutmix_p × rect × other_label_strength
- **11 no-pair CutMix-only variants** (Row 5-nopair A through K) — same axes, no paired-label coupling
- All 22 evaluated on POS9 strict n=2000

## Key cells

| Cell                                     | bestI | bit_F1 | Total FAR | Reading                       |
|------------------------------------------|-------|--------|-----------|-------------------------------|
| pair `sweep_C` (LS=0.20 p=0.30 rect=0.5) | I3    | 0.9943 |   100.00  | peak F1, I3 collapse          |
| pair Variant A (LS=0.30 p=0.20 rect=0.5) | I10   | 0.9520 |    29.20  | deployable pair               |
| nopair Variant I (LS=0.30 p=0.15)        | I10   | 0.9420 |     5.00  | lowest FAR no-pair, Row 5 pub |
| nopair `sweep_B` (LS=0.50 p=0.15)        | -     | 0.0000 |     0.00  | degenerate F1=0               |
| iter116J_s1 (frozen single)              | -     | 0.9927 |     0.00  | unbeaten past best            |
| E22 5-member ensemble (frozen)           | -     | 0.9956 |     0.00  | champion unbeaten             |

## Conclusions logged in §5.50

1. **Design rationale** — pair vs no-pair bracketed in one matrix; LS × p × rect × other_label resolved at 22 cells.
2. **22-variant landscape** — table dumped in §5.50 with family, axis values, bestI, bit_F1, FAR triple, status.
3. **Decision** — Row 5 single-model entry = no-pair Variant I (0.942 / 5.0 % Total FAR). pair-mask sweep_C I3 (0.9943 / 100 %) recorded as I3 inference-rule pathology, not table entry. Champions (E22 0.9956 / 0 %, iter116J_s1 0.9927 / 0 %) frozen — both unbeaten across full 22-variant sweep.

## Sections added

- `§5.50 Row 5 CutMix+Pair vs CutMix-only paper-grade sweep (22 variants × n=2000 POS9 strict)` — new top-level subsection in `docs/chip-multilabel/paper/05_experiments.md`

No champion-table change. Row 5 paper-grade established. Chain can proceed to next iter without re-sweep dependency.
