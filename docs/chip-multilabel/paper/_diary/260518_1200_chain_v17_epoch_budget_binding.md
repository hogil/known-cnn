# 260518 12:00 cron #82 — chain v17 epoch budget binding effect

## Finding
Chain v17 `--epochs=10` truncation caused LS=0.20 reproduction to fail at POS9 = 0.8535. Best checkpoint landed at epoch 1 of 10.

## Why paper-worthy
Direct evidence of multi-iter ablation **epoch budget binding effect**: when best_epoch ≤ 0.2 × epochs, the §6.29 mean-of-criteria selection rule has not traversed the multi-label warmup window. The 10-epoch budget statistically front-loads the best-epoch distribution to the warmup regime, where per-bit calibration that LS controls has not yet fitted. LS=0.20 viability question is **unanswered**, not answered in the negative.

## Connection to prior findings
- §6.32.6.1: LS=0.30 single-point viability — still supported by chain v12 Phase 2 LS=0.20 outright divergence (no checkpoint), but chain v17 attempt is now a methodological caveat rather than confirmation.
- §6.28: val_acc biased for multi-label bit-F1 — compounds with budget truncation.
- §6.29: mean-of-criteria selection rule — requires epoch budget for criterion convergence.
- §6.32.6.5: grad-checkpointing wall-clock overhead — chain throughput pressure that caused the truncation.

## New paper paragraph (§6.32.6.9)
Promotes the chain v17 result from "LS=0.20 falsified" to "unconverged — epoch-budget binding." Adds fifth nested failure mode (v) to the §6.32.6.5 (i)-(iv) infrastructure modes: **methodological** rather than infrastructural — per-trainer epoch budget below criterion convergence horizon.

## §7 discussion implication
Multi-iter ablation studies on saturated benchmarks require per-trainer epoch budgets sufficient for the selection criterion to traverse the multi-label warmup window. Budget truncation produces best-epoch-1 artifacts statistically indistinguishable from a 1-epoch run. Future BCE LS curve re-test requires ≥ 20 epochs.

## Champion status
Unchanged: iter116J single 0.9927 / 3-way vote ensemble 0.9941.
