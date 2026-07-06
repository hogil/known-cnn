# chain v6 iter 1 — iter116J_clone_s11 (seed=11 ckpt-selection variance)

- TS: 260517_082231 (train) / 260517_084417 (eval n2000)
- Source: `outputs/iter116J_clone_s11/20260517_082231_T7_iter116J_clone_s11/eval_n2000_pred/stage1_260517_084417/preds_chip.parquet`
- Recipe: T7 BCE+LS=0.30 + FCM-PM CutMix g=3 corner, seed=11
  (vs iter116J seed=1 — same recipe family, new seed point)
- Baseline (iter116J): bit_F1 0.9927 / Total FAR 0.00% (T7 LS=0.30 g=3 seed=1, ep6 by val_f1)
- Selected ckpt: **ep1** by margin_max criterion (val_acc 0.9876, while ep7 had 0.9907)

## Eval n2000 (POS9 strict + 4 OOD strict)

```
| Variant | bit_F1 | NI-FAR | OOD-FAR | Total FAR |
|---------|--------|--------|---------|-----------|
| I3      | 0.8582 | 100.00 |  100.00 |    100.00 |
| I7      | 0.8420 | 100.00 |  100.00 |    100.00 |
| I10     | 0.8456 |  72.65 |   63.91 |     70.53 |
| I13     | 0.7469 |  68.15 |   46.41 |     62.88 |
```

best variant: I10 — bit_F1 0.8456, Total FAR 70.53% (1862 FP / 2640)

## Delta vs iter116J baseline (same recipe family, atomic seed delta 1 -> 11)

```
| Variant | dbit_F1 | dTotal_FAR_pp |
|---------|---------|---------------|
| I3      | -0.1345 |       +100.00 |
| I7      | -0.1507 |       +100.00 |
| I10     | -0.1471 |        +70.53 |
| I13     | -0.2458 |        +62.88 |
```

## Insights / hypotheses

- **This is a checkpoint-selection variance event, not a hparam variance
  event.** The training run completed 10 epochs with stable val_acc
  trajectory: ep1-6 plateau at 0.9876, ep7 jumps to 0.9907, ep8-10 fall
  back to 0.9876. The margin_max criterion picked **ep1** as best.
- **ep1 is severely under-trained at the inference-gate level**: all 4
  variants show catastrophic FAR (I3/I7 at 100%, I10 70.5%, I13 62.9%).
  The model has learned positive-class logits well enough for top1
  accuracy 0.78-0.78 (matching macro_f1 0.92 on multi-label decode), but
  the negative-rejection margins (max_prob gate in I10, invalid_score
  gate in I13) are not yet calibrated.
- **Magnitude of the variance**: chain v5 (3 seeds at LS=0.30) saw
  Total FAR spread [0.04, 0.30, 18.22] at I10. s11 ep1 adds a fourth
  data point at 70.53% — pushing the 4-seed mean from 6.19% to 22.27%
  and exposing that the seed=7 18.22% was **not** a worst case.
- **The 0.069 bit_F1 swing iter116J ep6 (0.9927) vs s11 ep1 (0.8456)**
  is dominated by **ckpt selection criterion**, not seed. We do not yet
  have an s11 ep6/ep7 eval point to disentangle seed-only effect.
- This iter is **not interpretable as a recipe regression** until s11 is
  re-evaluated at a properly selected ckpt (val_f1 max = ep7, or
  Phase-2 multi-criterion comparison).

## Action items for chain v6 Phase 2+

1. Re-evaluate s11 best_model selected by **val_f1** (= ep7) instead of
   margin_max (= ep1) — single line ckpt-criterion swap.
2. Add an alternative ckpt selector flag to the runner so that one
   training run produces multiple ckpt candidates (ep_by_val_f1 /
   ep_by_margin_max / last_epoch) and all three are evaluated.
3. Hold off declaring "seed=11 is bad" until step 1 is done — the recipe
   may be fine, the selector clearly is not.

## chain v6 verdict (running)

```
| iter | seed | LS   | ckpt_sel    | ep_sel | best | bit_F1 | Total FAR | comment            |
|------|------|------|-------------|--------|------|--------|-----------|--------------------|
| v6.1 |   11 | 0.30 | margin_max  |      1 | I10  | 0.8456 |     70.53 | under-trained ckpt |
| 116J |    1 | 0.30 | val_f1      |      6 | I13  | 0.9927 |      0.00 | recorded SOTA      |
```
