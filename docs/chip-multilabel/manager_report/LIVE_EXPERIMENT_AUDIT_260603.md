# Live FCMPM Experiment Audit

updated: 2026-06-08 22:41:18

Purpose: keep training/eval running, record split effects, prune weak checkpoints, and identify next additions.

## Active Processes

- no active python experiment process found

## Split Summary

| axis | value | n | dataset_n | F1 mean | F1 std | FAR mean | gap mean | gap std | decision |
|---|---|---:|---:|---:|---:|---:|---:|---:|---|
| seed_repeat_p | p065_s99 | 1 | 1 | 0.9979 | 0.0000 | 0.06 | 0.286 | 0.000 | promote: use for 2-factor/seed repeat |
| seed_repeat_p | p070_s42 | 1 | 1 | 0.9978 | 0.0000 | 74.91 | -0.139 | 0.000 | prune: delete pth and avoid expansion |
| seed_repeat_neg | neg002_s42 | 1 | 1 | 0.9970 | 0.0000 | 42.66 | -0.010 | 0.000 | prune: delete pth and avoid expansion |
| seed_repeat_p | p070_s13 | 1 | 1 | 0.9969 | 0.0000 | 0.38 | 0.210 | 0.000 | promote: use for 2-factor/seed repeat |
| seed_repeat_p | p0625_s13 | 1 | 1 | 0.9968 | 0.0000 | 0.80 | 0.239 | 0.000 | promote: use for 2-factor/seed repeat |
| seed_repeat_p | p020_s13 | 1 | 1 | 0.9968 | 0.0000 | 1.46 | 0.088 | 0.000 | repeat: promising but dispersion unknown |
| baseline | A100_B100_neg000_p050_grid9_g3_cmp100 | 2 | 2 | 0.9966 | 0.0010 | 0.62 | 0.186 | 0.066 | promote: use for 2-factor/seed repeat |
| seed_repeat_p | p080_s13 | 1 | 1 | 0.9965 | 0.0000 | 0.20 | 0.234 | 0.000 | promote: use for 2-factor/seed repeat |
| seed_repeat_abpos_Avar_B100 | A080_B100_s42 | 1 | 1 | 0.9964 | 0.0000 | 0.17 | 0.164 | 0.000 | promote: use for 2-factor/seed repeat |
| cutmix_p | p030 | 1 | 1 | 0.9964 | 0.0000 | 1.75 | 0.233 | 0.000 | repeat: promising but dispersion unknown |
| seed_repeat_abpos_Avar_B100 | A080_B100_s13 | 1 | 1 | 0.9961 | 0.0000 | 0.00 | 0.258 | 0.000 | promote: use for 2-factor/seed repeat |
| cutmix_p | p050 | 5 | 5 | 0.9961 | 0.0016 | 16.94 | 0.087 | 0.121 | prune: delete pth and avoid expansion |
| cutmix_p | p065 | 5 | 5 | 0.9961 | 0.0020 | 18.49 | 0.138 | 0.172 | prune: delete pth and avoid expansion |
| neg_target | neg0015 | 1 | 1 | 0.9960 | 0.0000 | 0.11 | 0.180 | 0.000 | promote: use for 2-factor/seed repeat |
| seed_repeat_p | p065_s42 | 1 | 1 | 0.9960 | 0.0000 | 0.44 | 0.209 | 0.000 | promote: use for 2-factor/seed repeat |
| seed_repeat_neg | neg002_s13 | 1 | 1 | 0.9960 | 0.0000 | 0.60 | 0.143 | 0.000 | repeat: promising but dispersion unknown |
| seed_repeat_p | p090_s13 | 1 | 1 | 0.9960 | 0.0000 | 1.39 | 0.190 | 0.000 | repeat: promising but dispersion unknown |
| cutmix_p | p0625 | 5 | 5 | 0.9959 | 0.0015 | 21.33 | 0.082 | 0.156 | prune: delete pth and avoid expansion |
| seed_repeat_p | p100_s13 | 1 | 1 | 0.9959 | 0.0000 | 15.99 | -0.030 | 0.000 | prune: delete pth and avoid expansion |
| seed_repeat_neg | neg003_s13 | 1 | 1 | 0.9958 | 0.0000 | 0.71 | 0.198 | 0.000 | promote: use for 2-factor/seed repeat |
| seed_repeat_neg | neg0015_s13 | 1 | 1 | 0.9958 | 0.0000 | 2.62 | 0.163 | 0.000 | observe: keep evidence, no expansion yet |
| abpos_Avar_B100 | A090_B100 | 3 | 3 | 0.9958 | 0.0014 | 6.42 | 0.149 | 0.061 | prune: delete pth and avoid expansion |
| cutmix_p | p055 | 5 | 5 | 0.9956 | 0.0028 | 23.91 | 0.095 | 0.220 | prune: delete pth and avoid expansion |
| seed_repeat_neg | neg003_s99 | 1 | 1 | 0.9956 | 0.0000 | 0.30 | 0.242 | 0.000 | promote: use for 2-factor/seed repeat |
| neg_target | neg005 | 3 | 3 | 0.9955 | 0.0002 | 6.34 | 0.189 | 0.043 | prune: delete pth and avoid expansion |
| seed_repeat_neg | neg0015_s42 | 1 | 1 | 0.9955 | 0.0000 | 57.15 | -0.074 | 0.000 | prune: delete pth and avoid expansion |
| cutmix_p | p0575 | 5 | 5 | 0.9955 | 0.0031 | 17.92 | 0.157 | 0.193 | prune: delete pth and avoid expansion |
| seed_repeat_p | p060_s13 | 2 | 2 | 0.9954 | 0.0009 | 10.74 | 0.195 | 0.042 | prune: delete pth and avoid expansion |
| seed_repeat_abpos_Avar_B100 | A090_B100_s99 | 1 | 1 | 0.9954 | 0.0000 | 0.20 | 0.185 | 0.000 | promote: use for 2-factor/seed repeat |
| cutmix_p | p090 | 5 | 5 | 0.9954 | 0.0021 | 18.02 | 0.167 | 0.155 | prune: delete pth and avoid expansion |
| seed_repeat_baseline | s42 | 1 | 1 | 0.9954 | 0.0000 | 68.91 | -0.093 | 0.000 | prune: delete pth and avoid expansion |
| seed_repeat_p | p050_s42 | 1 | 1 | 0.9954 | 0.0000 | 68.91 | -0.093 | 0.000 | prune: delete pth and avoid expansion |
| seed_repeat_neg | neg005_s42 | 1 | 1 | 0.9953 | 0.0000 | 1.93 | 0.216 | 0.000 | repeat: promising but dispersion unknown |
| seed_repeat_neg | neg002_s99 | 1 | 1 | 0.9952 | 0.0000 | 0.08 | 0.192 | 0.000 | promote: use for 2-factor/seed repeat |
| seed_repeat_p | p0575_s13 | 3 | 3 | 0.9952 | 0.0002 | 28.43 | 0.105 | 0.086 | prune: delete pth and avoid expansion |
| seed_repeat_p | p030_s13 | 1 | 1 | 0.9950 | 0.0000 | 0.40 | 0.085 | 0.000 | repeat: promising but dispersion unknown |
| seed_repeat_p | p060_s99 | 1 | 1 | 0.9950 | 0.0000 | 0.77 | 0.234 | 0.000 | promote: use for 2-factor/seed repeat |
| seed_repeat_neg | neg0025_s99 | 1 | 1 | 0.9950 | 0.0000 | 1.93 | 0.196 | 0.000 | repeat: promising but dispersion unknown |
| cutmix_p | p100 | 5 | 5 | 0.9949 | 0.0031 | 26.65 | 0.088 | 0.139 | prune: delete pth and avoid expansion |
| cutmix_p | p070 | 5 | 5 | 0.9947 | 0.0014 | 17.22 | 0.134 | 0.162 | prune: delete pth and avoid expansion |
| seed_repeat_p | p010_s42 | 1 | 1 | 0.9947 | 0.0000 | 0.57 | 0.192 | 0.000 | promote: use for 2-factor/seed repeat |
| cutmix_p | p060 | 5 | 5 | 0.9946 | 0.0048 | 18.41 | 0.132 | 0.185 | prune: delete pth and avoid expansion |
| seed_repeat_abpos_Avar_B100 | A080_B100_s99 | 1 | 1 | 0.9945 | 0.0000 | 1.69 | 0.207 | 0.000 | repeat: promising but dispersion unknown |
| neg_target | neg0025 | 1 | 1 | 0.9944 | 0.0000 | 3.32 | 0.145 | 0.000 | observe: keep evidence, no expansion yet |
| seed_repeat_p | p065_s13 | 1 | 1 | 0.9943 | 0.0000 | 0.36 | 0.180 | 0.000 | promote: use for 2-factor/seed repeat |
| seed_repeat_p | p040_s99 | 1 | 1 | 0.9942 | 0.0000 | 0.25 | 0.221 | 0.000 | promote: use for 2-factor/seed repeat |
| seed_repeat_abpos_Avar_B100 | A070_B100_s99 | 1 | 1 | 0.9942 | 0.0000 | 0.42 | 0.138 | 0.000 | repeat: promising but dispersion unknown |
| seed_repeat_baseline | s13 | 1 | 1 | 0.9942 | 0.0000 | 2.00 | 0.190 | 0.000 | repeat: promising but dispersion unknown |
| seed_repeat_neg | neg010_s42 | 1 | 1 | 0.9942 | 0.0000 | 75.66 | -0.212 | 0.000 | prune: delete pth and avoid expansion |
| seed_repeat_p | p0575_s99 | 1 | 1 | 0.9938 | 0.0000 | 0.32 | 0.233 | 0.000 | promote: use for 2-factor/seed repeat |
| seed_repeat_neg | neg0025_s13 | 1 | 1 | 0.9937 | 0.0000 | 0.43 | 0.126 | 0.000 | repeat: promising but dispersion unknown |
| cutmix_p | p040 | 5 | 5 | 0.9937 | 0.0025 | 21.06 | -0.013 | 0.165 | prune: delete pth and avoid expansion |
| neg_target | neg002 | 3 | 3 | 0.9933 | 0.0033 | 3.94 | 0.065 | 0.169 | observe: keep evidence, no expansion yet |
| cutmix_p | p080 | 5 | 5 | 0.9932 | 0.0082 | 34.69 | 0.057 | 0.174 | prune: delete pth and avoid expansion |
| neg_target | neg003 | 1 | 1 | 0.9931 | 0.0000 | 0.08 | 0.138 | 0.000 | repeat: promising but dispersion unknown |
| seed_repeat_p | p055_s99 | 1 | 1 | 0.9930 | 0.0000 | 0.08 | 0.147 | 0.000 | repeat: promising but dispersion unknown |
| grid_g3 | grid12 | 3 | 3 | 0.9929 | 0.0021 | 8.96 | 0.183 | 0.057 | prune: delete pth and avoid expansion |
| seed_repeat_p | p050_s13 | 3 | 3 | 0.9929 | 0.0026 | 15.80 | 0.149 | 0.040 | prune: delete pth and avoid expansion |
| seed_repeat_p | p090_s42 | 1 | 1 | 0.9929 | 0.0000 | 19.97 | 0.101 | 0.000 | prune: delete pth and avoid expansion |
| seed_repeat_p | p0575_s42 | 1 | 1 | 0.9924 | 0.0000 | 52.98 | -0.084 | 0.000 | prune: delete pth and avoid expansion |
| seed_repeat_neg | neg0015_s99 | 1 | 1 | 0.9921 | 0.0000 | 0.31 | 0.162 | 0.000 | observe: keep evidence, no expansion yet |
| seed_repeat_abpos_Avar_B100 | A090_B100_s13 | 1 | 1 | 0.9920 | 0.0000 | 4.06 | 0.211 | 0.000 | observe: keep evidence, no expansion yet |
| abpos_Avar_B100 | A080_B100 | 3 | 3 | 0.9916 | 0.0068 | 10.23 | 0.049 | 0.255 | prune: delete pth and avoid expansion |
| seed_repeat_p | p040_s13 | 3 | 3 | 0.9914 | 0.0074 | 20.91 | 0.065 | 0.131 | prune: delete pth and avoid expansion |
| seed_repeat_p | p055_s13 | 3 | 3 | 0.9910 | 0.0052 | 21.52 | -0.008 | 0.143 | prune: delete pth and avoid expansion |
| seed_repeat_p | p020_s42 | 1 | 1 | 0.9909 | 0.0000 | 4.06 | 0.002 | 0.000 | observe: keep evidence, no expansion yet |
| seed_repeat_baseline | s99 | 1 | 1 | 0.9904 | 0.0000 | 1.51 | 0.192 | 0.000 | observe: keep evidence, no expansion yet |
| seed_repeat_p | p050_s99 | 1 | 1 | 0.9904 | 0.0000 | 1.51 | 0.192 | 0.000 | observe: keep evidence, no expansion yet |
| seed_repeat_p | p030_s42 | 1 | 1 | 0.9903 | 0.0000 | 2.23 | 0.132 | 0.000 | observe: keep evidence, no expansion yet |
| grid_g3 | grid6 | 1 | 1 | 0.9902 | 0.0000 | 1.32 | 0.111 | 0.000 | observe: keep evidence, no expansion yet |
| seed_repeat_neg | neg010_s13 | 1 | 1 | 0.9901 | 0.0000 | 0.35 | 0.036 | 0.000 | observe: keep evidence, no expansion yet |
| seed_repeat_p | p0625_s42 | 1 | 1 | 0.9901 | 0.0000 | 7.68 | 0.140 | 0.000 | prune: delete pth and avoid expansion |
| seed_repeat_neg | neg005_s99 | 1 | 1 | 0.9892 | 0.0000 | 6.15 | 0.024 | 0.000 | prune: delete pth and avoid expansion |
| seed_repeat_p | p100_s42 | 1 | 1 | 0.9890 | 0.0000 | 32.06 | 0.022 | 0.000 | prune: delete pth and avoid expansion |
| seed_repeat_p | p080_s42 | 1 | 1 | 0.9888 | 0.0000 | 13.37 | 0.018 | 0.000 | prune: delete pth and avoid expansion |
| seed_repeat_p | p040_s42 | 1 | 1 | 0.9876 | 0.0000 | 9.93 | -0.031 | 0.000 | prune: delete pth and avoid expansion |
| seed_repeat_p | p060_s42 | 1 | 1 | 0.9874 | 0.0000 | 3.94 | 0.017 | 0.000 | prune: delete pth and avoid expansion |
| seed_repeat_neg | neg0025_s42 | 1 | 1 | 0.9874 | 0.0000 | 15.05 | 0.027 | 0.000 | prune: delete pth and avoid expansion |
| cutmix_p | p020 | 1 | 1 | 0.9865 | 0.0000 | 1.01 | 0.098 | 0.000 | prune: delete pth and avoid expansion |
| seed_repeat_neg | neg005_s13 | 1 | 1 | 0.9863 | 0.0000 | 16.11 | 0.047 | 0.000 | prune: delete pth and avoid expansion |
| seed_repeat_neg | neg003_s42 | 1 | 1 | 0.9848 | 0.0000 | 11.17 | 0.042 | 0.000 | prune: delete pth and avoid expansion |
| neg_target | neg010 | 1 | 1 | 0.9847 | 0.0000 | 0.37 | 0.028 | 0.000 | prune: delete pth and avoid expansion |
| seed_repeat_abpos_Avar_B100 | A070_B100_s42 | 1 | 1 | 0.9844 | 0.0000 | 1.20 | 0.105 | 0.000 | prune: delete pth and avoid expansion |
| cutmix_p | p010 | 1 | 1 | 0.9835 | 0.0000 | 1.16 | 0.090 | 0.000 | prune: delete pth and avoid expansion |
| seed_repeat_abpos_Avar_B100 | A070_B100_s13 | 1 | 1 | 0.9835 | 0.0000 | 1.30 | 0.155 | 0.000 | prune: delete pth and avoid expansion |
| seed_repeat_p | p055_s42 | 1 | 1 | 0.9823 | 0.0000 | 24.37 | -0.120 | 0.000 | prune: delete pth and avoid expansion |
| abpos_Avar_B100 | A070_B100 | 1 | 1 | 0.9820 | 0.0000 | 0.54 | 0.044 | 0.000 | prune: delete pth and avoid expansion |
| grid_g3 | grid3 | 1 | 1 | 0.9813 | 0.0000 | 8.15 | -0.037 | 0.000 | prune: delete pth and avoid expansion |
| seed_repeat_p | p0625_s99 | 1 | 1 | 0.9781 | 0.0000 | 0.51 | -0.003 | 0.000 | prune: delete pth and avoid expansion |
| loss_variant | T6 | 1 | 1 | 0.9602 | 0.0000 | 8.48 | -0.337 | 0.000 | prune: delete pth and avoid expansion |
| seed_repeat_p | p010_s13 | 1 | 1 | 0.9580 | 0.0000 | 1.19 | -0.076 | 0.000 | prune: delete pth and avoid expansion |
| seed_repeat_abpos_Avar_B100 | A090_B100_s42 | 1 | 1 | 0.9322 | 0.0000 | 25.78 | -0.202 | 0.000 | prune: delete pth and avoid expansion |
| loss_variant | T10 | 1 | 1 | 0.0001 | 0.0000 | 0.00 | -0.089 | 0.000 | prune: delete pth and avoid expansion |
| loss_variant | T4 | 1 | 1 | 0.0001 | 0.0000 | 0.00 | -0.089 | 0.000 | prune: delete pth and avoid expansion |

## Queue / Prune Suggestions

- combine: neg002 x best cutmix_p and neg002 x T10
- combine: neg005 x best cutmix_p and neg005 x T10
- prune: A070_B100 A target down-weight is weak; prefer ASL/T10
- prune: A080_B100 A target down-weight is weak; prefer ASL/T10

## Response Curve Fit

Internal score: if `FAR<=1%`, `score = bit_F1 + 0.05*gap`; otherwise the score subtracts `0.02*(FAR-1)`.
This keeps high-F1 rows from winning if the negative tail leaks.

| axis | points | fit | R2 | observed best x | suggested x | note |
|---|---:|---|---:|---:|---:|---|
| abpos_Avar_B100 | 4 | quadratic | 0.887 | 1.0000 | 1.0000 | best observed |
| cutmix_p | 14 | linear | 0.614 | 0.3000 | 0.3000 | best observed |
| neg_target | 7 | quadratic | 0.464 | 0.0000 | 0.0000 | best observed |

## Operating Rule

- Delete low-value `.pth` only; keep CSV/MD/log/probability evidence.
- Expand only conditions with high mean F1, controlled FAR, and stable gap.
- Treat one-off high rows as candidates, not conclusions.
- Prefer ASL/T10 over lowering A target when A target down-weight reduces combo POS min.
