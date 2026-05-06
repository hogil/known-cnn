# Iter 6 — Phase A3: epochs sweep at LS=0.20

**Run dir**: `outputs/phase_a_260505_185805/`
**Date**: 2026-05-05 18:58 – 19:15
**Per-epochs train dirs**:
`outputs/logs_chip_multilabel/T1_LS20_LR04_ep{3,5,12}_<TS>/`

## Goal

Phase A1 (iter 5) found the LS optimum at α=0.20 with epochs=8. Phase A3
sweeps epochs ∈ {3, 5, 12} at the fixed (LS=0.20, LR=1e-4) recipe to
verify that 8 is the right number of epochs and to map the effect of
training duration on inference-variant choice.

## Sweep design

- **Epochs** ∈ {3, 5, 12}  (8 already covered by iter 5)
- **LS** = 0.20  (held)
- **LR** = 1e-4  (held)
- **Inference** = I3 / I7 / I10  per train run  → 9 cells total
- Strictly sequential.

## Sweep log

| epochs | inference | macro_f1 | top1_11 | T      | elapsed_train_sec |
|-------:|-----------|---------:|--------:|-------:|------------------:|
|      3 | I3        |   0.8467 |  0.5909 | 1.0000 |             132.1 |
|      3 | I7        |   0.8500 |  0.6023 | 1.0000 |             132.1 |
|      3 | **I10**   | **0.8763** | **0.6881** | 1.0000 |         132.1 |
|      5 | I3        |   0.8254 |  0.5920 | 1.0000 |             220.0 |
|      5 | I7        |   0.8236 |  0.6165 | 1.0000 |             220.0 |
|      5 | **I10**   | **0.8567** | **0.7074** | 1.0000 |         220.0 |
|      8 | I3        |   0.9239 |  0.8324 | 1.0000 |   ↳ iter 5 (A1)   |
|      8 | **I7**    | **0.9268** | **0.8449** | 1.0000 |   ↳ iter 5 (A1)   |
|      8 | I10       |   0.8841 |  0.8136 | 1.0000 |   ↳ iter 5 (A1)   |
|     12 | **I3**    | **0.8926** | **0.7273** | 1.0000 |             515.9 |
|     12 | I7        |   0.8872 |  0.7233 | 1.0000 |             515.9 |
|     12 | I10       |   0.8351 |  0.6716 | 1.0000 |             515.9 |

_Source: outputs/phase_a_260505_185805/sweep_log.csv (rows for ep ∈ {3,5,12}),
outputs/phase_a_260505_175105/sweep_log.csv (ep=8 reference)._

## Phase A3 best per-epochs

| epochs | best inference | macro_f1 | top1_11 |
|-------:|----------------|---------:|--------:|
|      3 | I10            |   0.8763 |  0.6881 |
|      5 | I10            |   0.8567 |  0.7074 |
|      8 | I7             |   0.9268 |  0.8449 |
|     12 | I3             |   0.8926 |  0.7273 |

**ep=8 remains the global Phase A winner** — 0.9268 macro-F1, +0.0342
above the next-best ep (12 → 0.8926) and +0.0505 above ep=3 (0.8763).

## The regime change — inference variant flips with epochs

| epochs | I3     | I7     | I10    | winner | regime hypothesis                               |
|-------:|-------:|-------:|-------:|--------|-------------------------------------------------|
|      3 | 0.8467 | 0.8500 | 0.8763 | I10    | logit still flat → entropy gate decides Normal  |
|      5 | 0.8254 | 0.8236 | 0.8567 | I10    | logit flat-ish → entropy gate still helps       |
|      8 | 0.9239 | 0.9268 | 0.8841 | **I7** | logit sharp → step-search F1-max wins, gate hurts |
|     12 | 0.8926 | 0.8872 | 0.8351 | I3     | logit very sharp → top-K rescue + thresholds win |

The progression is consistent and tells a clean story:

1. **Under-trained model (ep≤5)** — sigmoid distributions are still
   broad. The entropy Normal gate (I10) carries large macro-F1 gains by
   correctly identifying Normal chips via low overall confidence.
2. **Sweet-spot model (ep=8)** — logit margins are sharp enough that
   step-search F1-max thresholds (I7) directly select the right defects,
   while the entropy gate over-fires on real-defect chips that now have
   intermediate (not high) confidence.
3. **Over-trained model (ep=12)** — logits are sharp enough that even
   the simpler I3 (F1-max + top-K rescue) ties or wins; the more elaborate
   step-search of I7 starts to over-fit val.

This is the most surprising finding of Phase A: **the best inference
variant is a function of training duration**, and choosing I10 by default
(as iter 3 / 4 winners suggested) costs ~0.04 macro-F1 once the model is
properly trained.

## Phase A — final winner (across A1/A2/A3)

**LS=0.20, LR=1e-4, epochs=8, inference=I7  →  macro_f1 = 0.9268, top1_11 = 0.8449**

This is the iter 5 cell `T1_LS20__I7`; Phase A3 confirms it is the best
along the epochs axis as well. **Phase A is closed at this winner.**

## Decision for next phases

- Phase B: T4 (ASL) hparam sweep (γ_+, γ_-, clip).
- Phase C: T3 (Focal) hparam sweep (γ).
- Phase D: T5 (BCE) pos_weight + LS sweep.
- Phase E: T6 (BCE→ASL) warmup + ASL γ sweep.
- Phase F: best-known-method combination across phases.
- Phase G: extended metrics (ECE, mAP, per-class breakdown) on top cells.

## Files

- `outputs/phase_a_260505_185805/sweep_log.csv` — A3 raw rows.
- `outputs/logs_chip_multilabel/T1_LS20_LR04_ep{3,5,12}_*/` — per-epoch trains.
