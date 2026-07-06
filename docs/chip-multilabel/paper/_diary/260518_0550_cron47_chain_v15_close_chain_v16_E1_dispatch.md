# 260518 05:50 cron #47 — chain v15 KD viable corner closed; chain v16 (E1 teacher) dispatched

**Trigger.** cron fire #47, 05:50. User report: `chain v15 KD viable corner finding 정리 + chain v16 (E1 teacher) dispatch 흐름`.

**Action.** Appended a single update paragraph to `paper/05_experiments.md` §5.49.1 (the KD α/T corner refinement section). No other section touched.

## Chain v15 closure (KD viable corner finding)

| Cell  | α    | T   | POS9 bit_F1 | Total FAR | Status                |
|-------|------|-----|-------------|-----------|-----------------------|
| KD_v7 | 0.30 | 2   |      0.9265 |    0.00 % | sealed §5.49 row 9    |
| KD_v11| 0.25 | 2   |      0.9192 |    0.00 % | non-collapse, in band |
| KD_v12| 0.30 | 3   |      0.9470 |    0.00 % | new KD champion       |
| KD_v13| 0.30 | 4   |      0.9347 |    0.00 % | upper T edge          |
| KD_v14| 0.35 | 2.5 |           - |         - | collapse ~05:22       |

**Geometric finding.** KD viable region = **L-shaped plateau {(α=0.25, T=2), (α=0.30, T∈[2,3])}** — characterised, not a single point. +0.0205 lift from T=2→T=3 at α=0.30 attributable to temperature axis (soft-target entropy widens, 2-combo cells benefit).

## Chain v16 (E1 teacher swap) dispatch flow

```
Phase 1: train E1 teacher = iter116J recipe @ seed=1   (running, ETA ~06:30)
         outputs/chain_v16_01_E1_teacher_iter116J_s1/
Phase 2: KD_v15 = student vs E1 teacher (α=0.30, T=3,  (queued, GPU share)
         --kd-skip-on-cutmix, grad-checkpointing)
         outputs/chain_v16_02_KD_v15_E1_a030_T3/
Phase 3: eval n=2000 POS9 strict
```

**WHY E1 teacher (1 sentence).** Chain v8 averaged-teacher (3-run mean) embeds run-noise into soft targets and may cap student F1 at 0.9470; iter116J s=1 single-teacher reached 0.9927 — chain v16 tests whether KD ceiling tracks teacher F1 or saturates from soft-target geometry alone.

## Champion (unchanged)

- single: iter116J s=1 / I10 = **0.9927** POS9 / 0.00 % FAR
- ensemble: chain v8 3-way vote_majority_bits = **0.9941** POS9 / 0.00 % FAR
- KD methodological side-corner: KD_v12 = 0.9470 (best KD); chain v16 pending.

## Provenance

- §5.49.1 update appended at line ~5920 of `docs/chip-multilabel/paper/05_experiments.md`
- chain v15 KD_v14 collapse already recorded at `paper/06_analysis.md` §6.32.6.6 (cron #46)
- chain v16 outputs not yet on disk metric-wise — dispatch flow only, no metric claims
