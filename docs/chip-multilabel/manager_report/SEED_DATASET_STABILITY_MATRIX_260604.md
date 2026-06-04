# FCMPM Seed/Dataset Stability Matrix

이 문서는 단일 최고 row가 아니라 seed와 dataset 반복에서 살아남는 조건을 찾기 위한 stability matrix다.

Stable 판정:

```text
n >= 2, min(bit_F1) >= 0.990, max(FAR) <= 2.0, min(gap) >= 0.10
```

## Summary

| axis | value | n | datasets | seeds | F1 mean | F1 min | F1 std | FAR max | gap mean | gap min | gap std | decision |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| cutmix_p | p0575 | 1 | 1 | 1 | 0.9975 | 0.9975 | 0.0000 | 0.40 | 0.285 | 0.285 | 0.000 | repeat/prune-check |
| cutmix_p | p080 | 1 | 1 | 1 | 0.9971 | 0.9971 | 0.0000 | 1.96 | 0.234 | 0.234 | 0.000 | repeat/prune-check |
| cutmix_p | p030 | 1 | 1 | 1 | 0.9964 | 0.9964 | 0.0000 | 1.75 | 0.233 | 0.233 | 0.000 | repeat/prune-check |
| neg_target | neg002 | 1 | 1 | 1 | 0.9964 | 0.9964 | 0.0000 | 0.04 | 0.212 | 0.212 | 0.000 | repeat/prune-check |
| cutmix_p | p065 | 1 | 1 | 1 | 0.9964 | 0.9964 | 0.0000 | 1.54 | 0.171 | 0.171 | 0.000 | repeat/prune-check |
| cutmix_p | p055 | 1 | 1 | 1 | 0.9963 | 0.9963 | 0.0000 | 0.49 | 0.203 | 0.203 | 0.000 | repeat/prune-check |
| neg_target | neg0015 | 1 | 1 | 1 | 0.9960 | 0.9960 | 0.0000 | 0.11 | 0.180 | 0.180 | 0.000 | repeat/prune-check |
| cutmix_p | p070 | 1 | 1 | 1 | 0.9957 | 0.9957 | 0.0000 | 1.01 | 0.226 | 0.226 | 0.000 | repeat/prune-check |
| neg_target | neg005 | 1 | 1 | 1 | 0.9954 | 0.9954 | 0.0000 | 0.06 | 0.204 | 0.204 | 0.000 | repeat/prune-check |
| seed_repeat_neg | neg002 | 3 | 1 | 3 | 0.9961 | 0.9952 | 0.0009 | 42.66 | 0.108 | -0.010 | 0.105 | repeat/prune-check |
| cutmix_p | p060 | 1 | 1 | 1 | 0.9944 | 0.9944 | 0.0000 | 0.08 | 0.183 | 0.183 | 0.000 | repeat/prune-check |
| neg_target | neg0025 | 1 | 1 | 1 | 0.9944 | 0.9944 | 0.0000 | 3.32 | 0.145 | 0.145 | 0.000 | repeat/prune-check |
| cutmix_p | p0625 | 1 | 1 | 1 | 0.9943 | 0.9943 | 0.0000 | 0.25 | 0.113 | 0.113 | 0.000 | repeat/prune-check |
| neg_target | neg003 | 1 | 1 | 1 | 0.9931 | 0.9931 | 0.0000 | 0.08 | 0.138 | 0.138 | 0.000 | repeat/prune-check |
| cutmix_p | p040 | 1 | 1 | 1 | 0.9929 | 0.9929 | 0.0000 | 3.14 | 0.034 | 0.034 | 0.000 | repeat/prune-check |
| seed_repeat_p | p0575 | 3 | 1 | 3 | 0.9938 | 0.9924 | 0.0014 | 52.98 | 0.118 | -0.084 | 0.175 | repeat/prune-check |
| baseline | A100_B100_neg000_p050 | 5 | 2 | 4 | 0.9946 | 0.9904 | 0.0026 | 68.91 | 0.132 | -0.093 | 0.130 | repeat/prune-check |
| seed_repeat_p | p060 | 3 | 1 | 3 | 0.9928 | 0.9874 | 0.0047 | 3.94 | 0.159 | 0.017 | 0.123 | repeat/prune-check |
| cutmix_p | p020 | 1 | 1 | 1 | 0.9865 | 0.9865 | 0.0000 | 1.01 | 0.098 | 0.098 | 0.000 | repeat/prune-check |
| seed_repeat_neg | neg005 | 3 | 1 | 3 | 0.9903 | 0.9863 | 0.0046 | 16.11 | 0.096 | 0.024 | 0.105 | repeat/prune-check |
| neg_target | neg010 | 1 | 1 | 1 | 0.9847 | 0.9847 | 0.0000 | 0.37 | 0.028 | 0.028 | 0.000 | repeat/prune-check |

## Interpretation

- High single-run F1 is not enough. Seed-specific OOD tails can destroy gap and FAR.
- Promote only rows that keep both POS min and NEG max separated across seeds/datasets.
- Rows with high F1 but high FAR are calibration/tail failures, not champions.
