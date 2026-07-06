# FCMPM Seed/Dataset Stability Matrix

이 문서는 단일 최고 row가 아니라 seed와 dataset 반복에서 살아남는 조건을 찾기 위한 stability matrix다.

Stable 판정:

```text
n >= 2, min(bit_F1) >= 0.990, max(FAR) <= 2.0, min(gap) >= 0.10
```

## Summary

| axis | value | n | datasets | seeds | F1 mean | F1 min | F1 std | FAR max | gap mean | gap min | gap std | decision |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| seed_repeat_abpos_Avar_B100 | A080_B100 | 3 | 1 | 3 | 0.9957 | 0.9945 | 0.0010 | 1.69 | 0.210 | 0.164 | 0.047 | stable-promote |
| seed_repeat_p | p065 | 3 | 1 | 3 | 0.9961 | 0.9943 | 0.0018 | 0.44 | 0.225 | 0.180 | 0.055 | stable-promote |
| seed_repeat_p | p070 | 2 | 1 | 2 | 0.9973 | 0.9969 | 0.0006 | 74.91 | 0.035 | -0.139 | 0.247 | repeat/prune-check |
| cutmix_p | p030 | 1 | 1 | 1 | 0.9964 | 0.9964 | 0.0000 | 1.75 | 0.233 | 0.233 | 0.000 | repeat/prune-check |
| neg_target | neg0015 | 1 | 1 | 1 | 0.9960 | 0.9960 | 0.0000 | 0.11 | 0.180 | 0.180 | 0.000 | repeat/prune-check |
| neg_target | neg005 | 3 | 3 | 1 | 0.9955 | 0.9954 | 0.0002 | 16.95 | 0.189 | 0.141 | 0.043 | repeat/prune-check |
| seed_repeat_neg | neg002 | 3 | 1 | 3 | 0.9961 | 0.9952 | 0.0009 | 42.66 | 0.108 | -0.010 | 0.105 | repeat/prune-check |
| neg_target | neg0025 | 1 | 1 | 1 | 0.9944 | 0.9944 | 0.0000 | 3.32 | 0.145 | 0.145 | 0.000 | repeat/prune-check |
| cutmix_p | p0625 | 5 | 5 | 1 | 0.9959 | 0.9943 | 0.0015 | 76.75 | 0.082 | -0.155 | 0.156 | repeat/prune-check |
| cutmix_p | p050 | 5 | 5 | 1 | 0.9961 | 0.9936 | 0.0016 | 74.09 | 0.087 | -0.113 | 0.121 | repeat/prune-check |
| neg_target | neg003 | 1 | 1 | 1 | 0.9931 | 0.9931 | 0.0000 | 0.08 | 0.138 | 0.138 | 0.000 | repeat/prune-check |
| seed_repeat_p | p090 | 2 | 1 | 2 | 0.9945 | 0.9929 | 0.0022 | 19.97 | 0.146 | 0.101 | 0.063 | repeat/prune-check |
| cutmix_p | p070 | 5 | 5 | 1 | 0.9947 | 0.9927 | 0.0014 | 62.19 | 0.134 | -0.145 | 0.162 | repeat/prune-check |
| cutmix_p | p065 | 5 | 5 | 1 | 0.9961 | 0.9927 | 0.0020 | 76.18 | 0.138 | -0.163 | 0.172 | repeat/prune-check |
| seed_repeat_p | p0575 | 5 | 3 | 3 | 0.9943 | 0.9924 | 0.0012 | 52.98 | 0.093 | -0.084 | 0.129 | repeat/prune-check |
| cutmix_p | p090 | 5 | 5 | 1 | 0.9954 | 0.9921 | 0.0021 | 76.30 | 0.167 | -0.064 | 0.155 | repeat/prune-check |
| seed_repeat_neg | neg0015 | 3 | 1 | 3 | 0.9945 | 0.9921 | 0.0021 | 57.15 | 0.084 | -0.074 | 0.137 | repeat/prune-check |
| seed_repeat_p | p020 | 2 | 1 | 2 | 0.9939 | 0.9909 | 0.0042 | 4.06 | 0.045 | 0.002 | 0.061 | repeat/prune-check |
| cutmix_p | p055 | 5 | 5 | 1 | 0.9956 | 0.9907 | 0.0028 | 76.24 | 0.095 | -0.287 | 0.220 | repeat/prune-check |
| baseline | A100_B100_neg000_p050 | 5 | 2 | 4 | 0.9946 | 0.9904 | 0.0026 | 68.91 | 0.132 | -0.093 | 0.130 | repeat/prune-check |
| seed_repeat_p | p030 | 2 | 1 | 2 | 0.9927 | 0.9903 | 0.0033 | 2.23 | 0.109 | 0.085 | 0.033 | repeat/prune-check |
| cutmix_p | p0575 | 5 | 5 | 1 | 0.9955 | 0.9902 | 0.0031 | 76.24 | 0.157 | -0.175 | 0.193 | repeat/prune-check |
| seed_repeat_neg | neg010 | 2 | 1 | 2 | 0.9921 | 0.9901 | 0.0029 | 75.66 | -0.088 | -0.212 | 0.175 | repeat/prune-check |
| seed_repeat_p | p050 | 5 | 3 | 3 | 0.9929 | 0.9899 | 0.0026 | 68.91 | 0.109 | -0.093 | 0.118 | repeat/prune-check |
| neg_target | neg002 | 3 | 3 | 1 | 0.9933 | 0.9898 | 0.0033 | 6.18 | 0.065 | -0.120 | 0.169 | repeat/prune-check |
| cutmix_p | p100 | 5 | 5 | 1 | 0.9949 | 0.9896 | 0.0031 | 64.00 | 0.088 | -0.115 | 0.139 | repeat/prune-check |
| cutmix_p | p040 | 5 | 5 | 1 | 0.9937 | 0.9896 | 0.0025 | 76.21 | -0.013 | -0.263 | 0.165 | repeat/prune-check |
| seed_repeat_p | p100 | 2 | 1 | 2 | 0.9925 | 0.9890 | 0.0049 | 32.06 | -0.004 | -0.030 | 0.037 | repeat/prune-check |
| seed_repeat_p | p080 | 2 | 1 | 2 | 0.9927 | 0.9888 | 0.0054 | 13.37 | 0.126 | 0.018 | 0.153 | repeat/prune-check |
| seed_repeat_neg | neg0025 | 3 | 1 | 3 | 0.9920 | 0.9874 | 0.0041 | 15.05 | 0.116 | 0.027 | 0.085 | repeat/prune-check |
| seed_repeat_p | p060 | 4 | 2 | 3 | 0.9933 | 0.9874 | 0.0040 | 21.17 | 0.160 | 0.017 | 0.100 | repeat/prune-check |
| cutmix_p | p020 | 1 | 1 | 1 | 0.9865 | 0.9865 | 0.0000 | 1.01 | 0.098 | 0.098 | 0.000 | repeat/prune-check |
| seed_repeat_neg | neg005 | 3 | 1 | 3 | 0.9903 | 0.9863 | 0.0046 | 16.11 | 0.096 | 0.024 | 0.105 | repeat/prune-check |
| cutmix_p | p060 | 5 | 5 | 1 | 0.9946 | 0.9863 | 0.0048 | 76.22 | 0.132 | -0.156 | 0.185 | repeat/prune-check |
| seed_repeat_neg | neg003 | 3 | 1 | 3 | 0.9921 | 0.9848 | 0.0063 | 11.17 | 0.161 | 0.042 | 0.105 | repeat/prune-check |
| neg_target | neg010 | 1 | 1 | 1 | 0.9847 | 0.9847 | 0.0000 | 0.37 | 0.028 | 0.028 | 0.000 | repeat/prune-check |
| seed_repeat_abpos_Avar_B100 | A070_B100 | 3 | 1 | 3 | 0.9874 | 0.9835 | 0.0059 | 1.30 | 0.133 | 0.105 | 0.025 | repeat/prune-check |
| cutmix_p | p010 | 1 | 1 | 1 | 0.9835 | 0.9835 | 0.0000 | 1.16 | 0.090 | 0.090 | 0.000 | repeat/prune-check |
| seed_repeat_p | p040 | 5 | 3 | 3 | 0.9912 | 0.9830 | 0.0057 | 41.78 | 0.077 | -0.058 | 0.130 | repeat/prune-check |
| seed_repeat_p | p055 | 5 | 3 | 3 | 0.9897 | 0.9823 | 0.0056 | 30.83 | 0.000 | -0.164 | 0.139 | repeat/prune-check |
| cutmix_p | p080 | 5 | 5 | 1 | 0.9932 | 0.9786 | 0.0082 | 76.19 | 0.057 | -0.130 | 0.174 | repeat/prune-check |
| seed_repeat_p | p0625 | 3 | 1 | 3 | 0.9883 | 0.9781 | 0.0095 | 7.68 | 0.125 | -0.003 | 0.122 | repeat/prune-check |
| seed_repeat_p | p010 | 2 | 1 | 2 | 0.9764 | 0.9580 | 0.0260 | 1.19 | 0.058 | -0.076 | 0.190 | repeat/prune-check |
| seed_repeat_abpos_Avar_B100 | A090_B100 | 3 | 1 | 3 | 0.9732 | 0.9322 | 0.0355 | 25.78 | 0.065 | -0.202 | 0.231 | repeat/prune-check |

## Interpretation

- High single-run F1 is not enough. Seed-specific OOD tails can destroy gap and FAR.
- Promote only rows that keep both POS min and NEG max separated across seeds/datasets.
- Rows with high F1 but high FAR are calibration/tail failures, not champions.
