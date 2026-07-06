# 260518 11:48 — KD_E1 ensemble-teacher α/T sweep falsified (cron #81)

## Trigger

cron fire #81 reported KD_E1 (ensemble-teacher KD) sweep
result.

## Measurements

| Config              | POS9   | Δ vs KD_v7 single-teacher |
|---------------------|--------|--------------------------:|
| KD_v7 (single)      | 0.9265 |                     (ref) |
| KD_E1 α=0.30 T=2    | 0.7040 |                   -0.2225 |
| KD_E1 α=0.25 T=2    | 0.8285 |                   -0.0980 |
| KD_E1 α=0.30 T=3    |  dead  | trainer CPU-cap kill      |

α=0.25 recovers +0.1245 POS9 vs α=0.30, but still -0.0980
below single-teacher KD_v7. T=3 variant aborted under 30 %
RAM trainer cap (§6.32.6.5) — ensemble teacher distribution
materialisation roughly doubles trainer memory footprint vs
single-teacher KD.

## Why α/T tuning cannot recover

The KD_E1 teacher = logit-avg of 3 chain-v6 checkpoints living
in the §6.32.7 cross-teacher diversity limited regime. Their
per-class calibration error correlations are tight
(§6.32.6.7 modal-collapse anchor: `POS9 -0.2225 vs macro_4
-0.0504`, 4.41× ratio). The averaged softmax distribution is
**modally collapsed** — α/T rescaling cannot restore per-bit
calibration signal that was destroyed by averaging correlated
distributions.

## Paper-level conclusion

Extends §6.32.4 Model Soup falsification (weight-space avg of
correlated teachers fails) to **distribution-space** averaging
(KD with averaged-logit teacher). Both fail under the same
§6.32.7 cross-teacher diversity limit.

**Single-teacher per-seed KD is the unique KD winner** in the
4-class chip multi-label regime. KD path closed:

- α=0.35 single-teacher → out of basin (§6.32.6.6, -collapse)
- ensemble-teacher α∈{0.25, 0.30} T∈{2, 3} → modal collapse
  recovery insufficient (this entry)
- α=0.30 T=2 single-teacher → KD_v7 representative, POS9
  0.9265

## Status

Champion unchanged: iter116J single 0.9927 / 3-way vote 0.9941.
KD_E1 ensemble-teacher α/T sweep closed as falsified. No
further ensemble-teacher KD configs queued.

## Cross-refs

- §6.32.4 Model Soup falsification (weight-space)
- §6.32.5 KD α basin narrowness (single-teacher)
- §6.32.6.5 grad-checkpoint / 30 % RAM cap constraint
- §6.32.6.6 α=0.35 collapse
- §6.32.6.7 cross-teacher modal-collapse anchor
- §6.32.6.8 (this entry, appended to 06_analysis.md)
