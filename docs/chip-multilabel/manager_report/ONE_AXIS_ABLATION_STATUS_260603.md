# FCM-PM One-Axis Ablation Status

이 문서는 FCM-PM 단일 변수 분리 평가 결과를 누적 정리한다.

고정 baseline:

```text
dataset=frozen_original
train=E:/data/images/classification_chips
eval=E:/data/images/chip_multilabel_v15direct_n2000
T7, LS=0.295, g=3, grid=9x9, cmp=1.0, cutmix_p=0.5
A/B target=1.00/1.00, neg target=0.0, mpos=0.65, seed=7
train=200/class, eval=2000/class
```

진행 원칙:

1. 단일 변수 분리 평가로 영향 인자를 찾는다.
2. 상위 성능 축 2개를 조합해 2-factor interaction을 본다.
3. 2축 조합에서 안정적인 상위 조건이 나오면 3축 조합으로 확장한다.
4. 모든 실험 row는 bit_F1, FAR, POS min / NEG max gap을 같이 본다.
5. 관리자용 표도 FAR를 포함한다. probability separation은 FAR와 함께 해석한다.

## Active / Planned Queue

| phase | axis | values | status |
|---|---|---|---|
| 1-axis | A/B positive target | A=0.90/0.80/0.70, B=1.00 fixed | running / queued |
| 1-axis | neg target | 0.015 / 0.02 / 0.025 / 0.03 / 0.05 / 0.10 | queued |
| 1-axis | cutmix_p | decile 0.10 / 0.20 / ... / 1.00 plus refine 0.55 / 0.575 / 0.625 / 0.65 | running / queued |
| 1-axis | loss variant | T10 / T4 / T6 completed on frozen_original; collapsed or leaked | pruned from transfer repeats |
| repeat | seed stability | baseline, A=0.90/0.80/0.70, neg=0.015/0.02/0.025/0.03/0.05/0.10, p decile 0.10-1.00 plus 0.55/0.575/0.625/0.65 at seed 13/42/99 | queued |
| 1-axis | grid, g=3 | 3x3 / 6x6 / 9x9 / 12x12 / 15x15 / 18x18 | running / queued |
| repeat | grid seed stability | g=3 grid 3x3/6x6/9x9/12x12/15x15/18x18 at seed 13/42/99 | queued |
| 1-axis | group-grid alignment | g=2 grid6 / g=4 grid12, baseline g=3 grid9 | queued |
| existing evidence | cmp | 0.5 / 0.7 / 0.8 / 1.0 | mined, not rerun |
| multi-dataset | transfer data | frozen_original, gapstress seed31/97, frozen snapshots | running after restart |
| 2-factor | top 1-axis pairs | p=0.575-centered neg/p, A=0.90/p, A=0.90/neg, grid/p | pending |
| 3-factor | top 2-factor neighborhood | compact A=0.90 + neg + p=0.575 candidates | pending |

## Completed Rows

| dataset | axis | value | bit_F1 | FAR | pos | neg | gap | worst POS min | worst NEG max |
|---|---|---|---:|---:|---:|---:|---:|---|---|
| frozen_original | abpos_Avar_B100 | A070_B100 | 0.9820 | 0.54 | 0.7125 | 0.2250 | 0.044 | bank_boundary+scratch/sc=0.536 | CrossScratch/bb=0.492 |
| frozen_original | abpos_Avar_B100 | A080_B100 | 0.9837 | 15.54 | 0.7169 | 0.2203 | -0.246 | bank_boundary+scratch/sc=0.257 | Starburst/sc=0.503 |
| frozen_original | abpos_Avar_B100 | A090_B100 | 0.9949 | 0.33 | 0.7706 | 0.2209 | 0.130 | bank_boundary+scratch/sc=0.652 | Invalid/sr=0.522 |
| frozen_iter116J_orig814_eval_n20000 | baseline | A100_B100_neg000_p050_grid9_g3_cmp100 | 0.9973 | 0.86 | 0.8072 | 0.2266 | 0.233 | bank_boundary+scratch/sc=0.741 | Invalid/sr=0.508 |
| frozen_original | baseline | A100_B100_neg000_p050_grid9_g3_cmp100 | 0.9959 | 0.38 | 0.7890 | 0.2202 | 0.140 | bank_boundary+scratch/sc=0.676 | Invalid/fk=0.536 |
| frozen_original | cutmix_p | p010 | 0.9835 | 1.16 | 0.7442 | 0.2288 | 0.090 | bank_boundary+scratch/sc=0.586 | Invalid/sr=0.496 |
| frozen_original | cutmix_p | p020 | 0.9865 | 1.01 | 0.7712 | 0.2393 | 0.098 | bank_boundary+scratch/sc=0.544 | Invalid/fk=0.446 |
| frozen_original | cutmix_p | p030 | 0.9964 | 1.75 | 0.7904 | 0.2304 | 0.233 | bank_boundary+scratch/sc=0.685 | Invalid/fk=0.452 |
| frozen_original | cutmix_p | p040 | 0.9929 | 3.14 | 0.7970 | 0.2225 | 0.034 | bank_boundary+scratch/sc=0.617 | CrossScratch/bb=0.583 |
| frozen_original | cutmix_p | p050 | 0.9959 | 0.38 | 0.7890 | 0.2202 | 0.140 | bank_boundary+scratch/sc=0.676 | Invalid/fk=0.536 |
| frozen_original | cutmix_p | p055 | 0.9963 | 0.49 | 0.8034 | 0.2225 | 0.203 | fork+scratch/fk=0.736 | Invalid/fk=0.533 |
| frozen_original | cutmix_p | p0575 | 0.9975 | 0.40 | 0.7861 | 0.2234 | 0.285 | fork+scratch_rot/sr=0.726 | Invalid/fk=0.441 |
| frozen_original | cutmix_p | p060 | 0.9944 | 0.08 | 0.7902 | 0.2212 | 0.183 | bank_boundary+scratch/sc=0.643 | CrossScratch/bb=0.460 |
| frozen_original | cutmix_p | p0625 | 0.9943 | 0.25 | 0.7937 | 0.2261 | 0.113 | bank_boundary+scratch/sc=0.675 | Normal/sc=0.562 |
| frozen_original | cutmix_p | p065 | 0.9964 | 1.54 | 0.8003 | 0.2032 | 0.171 | fork+scratch/fk=0.702 | Invalid/fk=0.531 |
| frozen_original | cutmix_p | p070 | 0.9957 | 1.01 | 0.7981 | 0.2200 | 0.226 | fork+scratch/fk=0.689 | Invalid/fk=0.463 |
| frozen_original | cutmix_p | p080 | 0.9971 | 1.96 | 0.8046 | 0.2235 | 0.234 | fork+scratch/sc=0.709 | Normal/sc=0.475 |
| frozen_original | cutmix_p | p090 | 0.9921 | 0.96 | 0.7886 | 0.2233 | 0.175 | bank_boundary+scratch/sc=0.640 | Normal/sc=0.465 |
| frozen_original | cutmix_p | p100 | 0.9961 | 1.43 | 0.7888 | 0.2261 | 0.253 | fork+scratch/fk=0.701 | CenterDonut/bb=0.448 |
| frozen_original | grid_g3 | grid12 | 0.9917 | 0.11 | 0.7720 | 0.2270 | 0.181 | fork+scratch/fk=0.634 | Invalid/fk=0.453 |
| frozen_original | grid_g3 | grid3 | 0.9813 | 8.15 | 0.8047 | 0.2247 | -0.037 | bank_boundary+scratch/sc=0.668 | Invalid/fk=0.705 |
| frozen_original | grid_g3 | grid6 | 0.9902 | 1.32 | 0.7921 | 0.2144 | 0.111 | bank_boundary+scratch/sc=0.640 | Invalid/fk=0.529 |
| frozen_original | loss_variant | T10_T10_LS029500_g3_grid9_cmp10000_p05000_mpos065_s7_ep10_tr200_ev02000 | 0.0001 | 0.00 | 0.7723 | 0.5879 | -0.089 | bank_boundary+scratch/sc=0.682 | Invalid/fk=0.771 |
| frozen_original | loss_variant | T4_T4_LS029500_g3_grid9_cmp10000_p05000_mpos065_s7_ep10_tr200_ev02000 | 0.0001 | 0.00 | 0.7723 | 0.5879 | -0.089 | bank_boundary+scratch/sc=0.682 | Invalid/fk=0.771 |
| frozen_original | loss_variant | T6_T6_LS029500_g3_grid9_cmp10000_p05000_mpos065_s7_ep10_tr200_ev02000 | 0.9602 | 8.48 | 0.9100 | 0.1123 | -0.337 | bank_boundary+scratch/sc=0.197 | Invalid/fk=0.534 |
| frozen_original | neg_target | neg0015 | 0.9960 | 0.11 | 0.7958 | 0.2309 | 0.180 | bank_boundary+scratch/sc=0.688 | Normal/sc=0.508 |
| frozen_original | neg_target | neg002 | 0.9964 | 0.04 | 0.7605 | 0.2226 | 0.212 | bank_boundary+scratch/sc=0.654 | Normal/sc=0.442 |
| frozen_original | neg_target | neg0025 | 0.9944 | 3.32 | 0.7838 | 0.2240 | 0.145 | bank_boundary+scratch/sc=0.656 | Normal/sc=0.511 |
| frozen_original | neg_target | neg003 | 0.9931 | 0.08 | 0.7835 | 0.2266 | 0.138 | bank_boundary+scratch/sc=0.631 | Normal/sc=0.493 |
| frozen_original | neg_target | neg005 | 0.9954 | 0.06 | 0.7753 | 0.2287 | 0.204 | scratch+scratch_rot/sc=0.656 | Normal/sc=0.452 |
| frozen_original | neg_target | neg010 | 0.9847 | 0.37 | 0.7769 | 0.2458 | 0.028 | bank_boundary+scratch/sc=0.594 | Invalid/fk=0.566 |
| frozen_original | other | oneaxis_seed_repeat_grid_g3_grid12_s13_T7_LS029500_g3_grid12_cmp10000_p05000_mpos065_s13_ep10_tr200_ev02000 | 0.9978 | 0.12 | 0.8098 | 0.2270 | 0.222 | bank_boundary+scratch/sc=0.716 | Invalid/sr=0.494 |
| frozen_original | other | oneaxis_seed_repeat_grid_g3_grid12_s42_T7_LS029500_g3_grid12_cmp10000_p05000_mpos065_s42_ep10_tr200_ev02000 | 0.7693 | 98.76 | 0.5966 | 0.2626 | -0.685 | bank_boundary+fork/fk=0.154 | CenterDonut/bb=0.839 |
| frozen_original | other | oneaxis_seed_repeat_grid_g3_grid15_s13_T7_LS029500_g3_grid15_cmp10000_p05000_mpos065_s13_ep10_tr200_ev02000 | 0.9957 | 0.60 | 0.7638 | 0.2183 | 0.109 | scratch+scratch_rot/sc=0.651 | Invalid/sr=0.542 |
| frozen_original | other | oneaxis_seed_repeat_grid_g3_grid15_s42_T7_LS029500_g3_grid15_cmp10000_p05000_mpos065_s42_ep10_tr200_ev02000 | 0.9921 | 32.01 | 0.7890 | 0.2301 | 0.014 | bank_boundary+scratch/sc=0.685 | CrossScratch/bb=0.671 |
| frozen_original | other | oneaxis_seed_repeat_grid_g3_grid18_s13_T7_LS029500_g3_grid18_cmp10000_p05000_mpos065_s13_ep10_tr200_ev02000 | 0.9975 | 0.19 | 0.7873 | 0.2200 | 0.283 | bank_boundary+scratch/sc=0.703 | Invalid/sr=0.420 |
| frozen_original | other | oneaxis_seed_repeat_grid_g3_grid18_s42_T7_LS029500_g3_grid18_cmp10000_p05000_mpos065_s42_ep10_tr200_ev02000 | 0.9911 | 9.23 | 0.7842 | 0.2266 | 0.039 | bank_boundary+scratch/sc=0.657 | CrossScratch/bb=0.618 |
| frozen_original | other | oneaxis_seed_repeat_grid_g3_grid3_s13_T7_LS029500_g3_grid3_cmp10000_p05000_mpos065_s13_ep10_tr200_ev02000 | 0.9919 | 5.57 | 0.8123 | 0.2290 | 0.123 | bank_boundary+fork/fk=0.698 | Invalid/sr=0.575 |
| frozen_original | other | oneaxis_seed_repeat_grid_g3_grid3_s42_T7_LS029500_g3_grid3_cmp10000_p05000_mpos065_s42_ep10_tr200_ev02000 | 0.9799 | 24.54 | 0.8148 | 0.2330 | -0.042 | bank_boundary+fork/fk=0.715 | Invalid/fk=0.757 |
| frozen_original | other | oneaxis_seed_repeat_grid_g3_grid6_s13_T7_LS029500_g3_grid6_cmp10000_p05000_mpos065_s13_ep10_tr200_ev02000 | 0.9928 | 1.27 | 0.7932 | 0.2136 | 0.143 | bank_boundary+fork/fk=0.643 | Invalid/sr=0.500 |
| frozen_original | other | oneaxis_seed_repeat_grid_g3_grid6_s42_T7_LS029500_g3_grid6_cmp10000_p05000_mpos065_s42_ep10_tr200_ev02000 | 0.9931 | 10.93 | 0.7950 | 0.2352 | 0.128 | bank_boundary+scratch/sc=0.707 | Invalid/fk=0.579 |
| frozen_original | other | oneaxis_seed_repeat_grid_g3_grid9_s13_T7_LS029500_g3_grid9_cmp10000_p05000_mpos065_s13_ep10_tr200_ev02000 | 0.9942 | 2.00 | 0.7887 | 0.2230 | 0.190 | scratch+scratch_rot/sr=0.641 | Invalid/sr=0.451 |
| frozen_original | other | oneaxis_seed_repeat_grid_g3_grid9_s42_T7_LS029500_g3_grid9_cmp10000_p05000_mpos065_s42_ep10_tr200_ev02000 | 0.9954 | 68.91 | 0.8080 | 0.2409 | -0.093 | bank_boundary+scratch/sc=0.743 | CrossScratch/bb=0.836 |
| frozen_original | seed_repeat_abpos_Avar_B100 | A070_B100_s13 | 0.9835 | 1.30 | 0.7207 | 0.1841 | 0.155 | bank_boundary+scratch/sc=0.547 | Normal/sc=0.392 |
| frozen_original | seed_repeat_abpos_Avar_B100 | A070_B100_s42 | 0.9844 | 1.20 | 0.7608 | 0.2106 | 0.105 | bank_boundary+scratch/sc=0.594 | DiagonalSmear/sc=0.489 |
| frozen_original | seed_repeat_abpos_Avar_B100 | A070_B100_s99 | 0.9942 | 0.42 | 0.7371 | 0.2264 | 0.138 | fork+scratch/fk=0.582 | CrossScratch/bb=0.444 |
| frozen_original | seed_repeat_abpos_Avar_B100 | A080_B100_s13 | 0.9961 | 0.00 | 0.7281 | 0.2277 | 0.258 | scratch+scratch_rot/sc=0.638 | Invalid/sr=0.380 |
| frozen_original | seed_repeat_abpos_Avar_B100 | A080_B100_s42 | 0.9964 | 0.17 | 0.7662 | 0.2211 | 0.164 | bank_boundary+scratch/sc=0.615 | DiagonalSmear/sc=0.451 |
| frozen_original | seed_repeat_abpos_Avar_B100 | A080_B100_s99 | 0.9945 | 1.69 | 0.7647 | 0.2278 | 0.207 | bank_boundary+scratch/sc=0.626 | CrossScratch/bb=0.419 |
| frozen_original | seed_repeat_abpos_Avar_B100 | A090_B100_s13 | 0.9920 | 4.06 | 0.7583 | 0.2248 | 0.211 | scratch+scratch_rot/sr=0.653 | Starburst/fk=0.442 |
| frozen_original | seed_repeat_abpos_Avar_B100 | A090_B100_s42 | 0.9322 | 25.78 | 0.7512 | 0.2170 | -0.202 | bank_boundary+scratch/sc=0.471 | CrossScratch/sr=0.673 |
| frozen_original | seed_repeat_abpos_Avar_B100 | A090_B100_s99 | 0.9954 | 0.20 | 0.7908 | 0.2270 | 0.185 | bank_boundary+scratch/sc=0.659 | CenterDonut/bb=0.474 |
| frozen_original | seed_repeat_baseline | s13 | 0.9942 | 2.00 | 0.7887 | 0.2230 | 0.190 | scratch+scratch_rot/sr=0.641 | Invalid/sr=0.451 |
| frozen_original | seed_repeat_baseline | s42 | 0.9954 | 68.91 | 0.8080 | 0.2409 | -0.093 | bank_boundary+scratch/sc=0.743 | CrossScratch/bb=0.836 |
| frozen_original | seed_repeat_baseline | s99 | 0.9904 | 1.51 | 0.8044 | 0.2261 | 0.192 | bank_boundary+scratch/sc=0.701 | Invalid/sc=0.509 |
| frozen_original | seed_repeat_neg | neg0015_s13 | 0.9958 | 2.62 | 0.7845 | 0.2243 | 0.163 | scratch+scratch_rot/sr=0.693 | Invalid/sr=0.530 |
| frozen_original | seed_repeat_neg | neg0015_s42 | 0.9955 | 57.15 | 0.7950 | 0.2443 | -0.074 | bank_boundary+scratch/sc=0.700 | CrossScratch/bb=0.774 |
| frozen_original | seed_repeat_neg | neg0015_s99 | 0.9921 | 0.31 | 0.7920 | 0.2418 | 0.162 | fork+scratch/sc=0.656 | Invalid/sc=0.494 |
| frozen_original | seed_repeat_neg | neg0025_s13 | 0.9937 | 0.43 | 0.7695 | 0.2265 | 0.126 | bank_boundary+scratch/sc=0.631 | Invalid/sr=0.505 |
| frozen_original | seed_repeat_neg | neg0025_s42 | 0.9874 | 15.05 | 0.8099 | 0.2371 | 0.027 | bank_boundary+scratch/sc=0.720 | CenterDonut/bb=0.693 |
| frozen_original | seed_repeat_neg | neg0025_s99 | 0.9950 | 1.93 | 0.7879 | 0.2331 | 0.196 | scratch+scratch_rot/sr=0.663 | Invalid/sc=0.467 |
| frozen_original | seed_repeat_neg | neg002_s13 | 0.9960 | 0.60 | 0.7843 | 0.2237 | 0.143 | scratch+scratch_rot/sr=0.647 | Invalid/sr=0.504 |
| frozen_original | seed_repeat_neg | neg002_s42 | 0.9970 | 42.66 | 0.8123 | 0.2371 | -0.010 | fork+scratch/sc=0.733 | CrossScratch/bb=0.743 |
| frozen_original | seed_repeat_neg | neg002_s99 | 0.9952 | 0.08 | 0.8043 | 0.2253 | 0.192 | bank_boundary+scratch/sc=0.713 | Invalid/sc=0.521 |
| frozen_original | seed_repeat_neg | neg003_s13 | 0.9958 | 0.71 | 0.7789 | 0.2254 | 0.198 | bank_boundary+scratch/sc=0.698 | Invalid/sr=0.500 |
| frozen_original | seed_repeat_neg | neg003_s42 | 0.9848 | 11.17 | 0.8114 | 0.2374 | 0.042 | bank_boundary+scratch/sc=0.717 | CrossScratch/bb=0.675 |
| frozen_original | seed_repeat_neg | neg003_s99 | 0.9956 | 0.30 | 0.8112 | 0.2291 | 0.242 | bank_boundary+scratch/sc=0.743 | CrossScratch/bb=0.501 |
| frozen_original | seed_repeat_neg | neg005_s13 | 0.9863 | 16.11 | 0.7622 | 0.2347 | 0.047 | bank_boundary+scratch/sc=0.637 | CenterDonut/bb=0.590 |
| frozen_original | seed_repeat_neg | neg005_s42 | 0.9953 | 1.93 | 0.8066 | 0.2268 | 0.216 | bank_boundary+scratch/sc=0.692 | Invalid/sr=0.476 |
| frozen_original | seed_repeat_neg | neg005_s99 | 0.9892 | 6.15 | 0.7855 | 0.2315 | 0.024 | bank_boundary+scratch/sc=0.559 | CrossScratch/bb=0.535 |
| frozen_original | seed_repeat_neg | neg010_s13 | 0.9901 | 0.35 | 0.7405 | 0.2236 | 0.036 | bank_boundary+scratch_rot/sr=0.601 | Invalid/sr=0.565 |
| frozen_original | seed_repeat_neg | neg010_s42 | 0.9942 | 75.66 | 0.7864 | 0.2602 | -0.212 | bank_boundary+scratch/sc=0.644 | CenterDonut/bb=0.856 |
| frozen_original | seed_repeat_p | p010_s13 | 0.9580 | 1.19 | 0.7124 | 0.2290 | -0.076 | fork+scratch_rot/fk=0.475 | Invalid/sr=0.551 |
| frozen_original | seed_repeat_p | p010_s42 | 0.9947 | 0.57 | 0.7985 | 0.2248 | 0.192 | bank_boundary+scratch/sc=0.668 | Invalid/sc=0.476 |
| frozen_original | seed_repeat_p | p020_s13 | 0.9968 | 1.46 | 0.7875 | 0.2191 | 0.088 | fork+scratch/sc=0.712 | Invalid/sr=0.624 |
| frozen_original | seed_repeat_p | p020_s42 | 0.9909 | 4.06 | 0.7545 | 0.2297 | 0.002 | bank_boundary+scratch/sc=0.582 | Invalid/fk=0.580 |
| frozen_original | seed_repeat_p | p030_s13 | 0.9950 | 0.40 | 0.7673 | 0.2243 | 0.085 | bank_boundary+scratch/sc=0.588 | Invalid/sr=0.503 |
| frozen_original | seed_repeat_p | p030_s42 | 0.9903 | 2.23 | 0.7949 | 0.2216 | 0.132 | bank_boundary+scratch/sc=0.630 | Invalid/fk=0.498 |
| frozen_original | seed_repeat_p | p040_s13 | 0.9943 | 0.94 | 0.7958 | 0.2361 | 0.203 | bank_boundary+scratch/sc=0.672 | CrossScratch/bb=0.469 |
| frozen_original | seed_repeat_p | p040_s42 | 0.9876 | 9.93 | 0.7777 | 0.2278 | -0.031 | bank_boundary+scratch/sc=0.546 | Invalid/sr=0.577 |
| frozen_original | seed_repeat_p | p050_s13 | 0.9942 | 2.00 | 0.7887 | 0.2230 | 0.190 | scratch+scratch_rot/sr=0.641 | Invalid/sr=0.451 |
| frozen_original | seed_repeat_p | p050_s42 | 0.9954 | 68.91 | 0.8080 | 0.2409 | -0.093 | bank_boundary+scratch/sc=0.743 | CrossScratch/bb=0.836 |
| frozen_original | seed_repeat_p | p055_s13 | 0.9940 | 11.89 | 0.7928 | 0.2226 | 0.117 | bank_boundary+scratch/sc=0.691 | Invalid/sr=0.574 |
| frozen_original | seed_repeat_p | p055_s42 | 0.9823 | 24.37 | 0.7791 | 0.2439 | -0.120 | bank_boundary+scratch/sc=0.579 | CrossScratch/bb=0.699 |
| frozen_original | seed_repeat_p | p0575_s13 | 0.9952 | 1.91 | 0.8011 | 0.2248 | 0.204 | fork+scratch/fk=0.731 | Invalid/sr=0.527 |
| frozen_original | seed_repeat_p | p0575_s42 | 0.9924 | 52.98 | 0.7928 | 0.2372 | -0.084 | bank_boundary+scratch/sc=0.679 | CenterDonut/bb=0.763 |
| frozen_original | seed_repeat_p | p0575_s99 | 0.9938 | 0.32 | 0.7919 | 0.2143 | 0.233 | bank_boundary+scratch/sc=0.685 | Invalid/sc=0.452 |
| frozen_original | seed_repeat_p | p060_s13 | 0.9961 | 0.31 | 0.7830 | 0.2281 | 0.225 | bank_boundary+scratch/sc=0.680 | Invalid/sr=0.455 |
| frozen_original | seed_repeat_p | p060_s42 | 0.9874 | 3.94 | 0.7987 | 0.2271 | 0.017 | bank_boundary+scratch/sc=0.638 | Normal/sr=0.621 |
| frozen_original | seed_repeat_p | p060_s99 | 0.9950 | 0.77 | 0.7959 | 0.2357 | 0.234 | bank_boundary+scratch/bb=0.706 | Invalid/fk=0.472 |
| frozen_original | seed_repeat_p | p0625_s13 | 0.9968 | 0.80 | 0.8052 | 0.2191 | 0.239 | bank_boundary+scratch/sc=0.670 | CrossScratch/bb=0.431 |
| frozen_original | seed_repeat_p | p0625_s42 | 0.9901 | 7.68 | 0.8096 | 0.2383 | 0.140 | bank_boundary+scratch/sc=0.741 | CrossScratch/bb=0.601 |
| frozen_original | seed_repeat_p | p065_s13 | 0.9943 | 0.36 | 0.8173 | 0.2458 | 0.180 | bank_boundary+scratch/sc=0.664 | CrossScratch/bb=0.484 |
| frozen_original | seed_repeat_p | p065_s42 | 0.9960 | 0.44 | 0.8250 | 0.2173 | 0.209 | bank_boundary+scratch/bb=0.747 | Invalid/sr=0.538 |
| frozen_original | seed_repeat_p | p070_s13 | 0.9969 | 0.38 | 0.7949 | 0.2349 | 0.210 | bank_boundary+scratch/sc=0.672 | Invalid/fk=0.462 |
| frozen_original | seed_repeat_p | p070_s42 | 0.9978 | 74.91 | 0.8028 | 0.2509 | -0.139 | bank_boundary+scratch/sc=0.722 | Starburst/bb=0.861 |
| frozen_original | seed_repeat_p | p080_s13 | 0.9965 | 0.20 | 0.8084 | 0.2250 | 0.234 | bank_boundary+scratch/sc=0.724 | Invalid/sr=0.490 |
| frozen_original | seed_repeat_p | p080_s42 | 0.9888 | 13.37 | 0.8038 | 0.2357 | 0.018 | bank_boundary+scratch/sc=0.714 | CrossScratch/bb=0.696 |
| frozen_original | seed_repeat_p | p090_s13 | 0.9960 | 1.39 | 0.8014 | 0.2240 | 0.190 | bank_boundary+scratch/sc=0.690 | Invalid/sr=0.500 |
| frozen_original | seed_repeat_p | p090_s42 | 0.9929 | 19.97 | 0.8108 | 0.2380 | 0.101 | bank_boundary+scratch/sc=0.736 | CrossScratch/bb=0.635 |
| frozen_original | seed_repeat_p | p100_s13 | 0.9959 | 15.99 | 0.7870 | 0.2227 | -0.030 | fork+scratch/sc=0.675 | Invalid/sc=0.705 |
| frozen_original | seed_repeat_p | p100_s42 | 0.9890 | 32.06 | 0.8174 | 0.2514 | 0.022 | bank_boundary+fork/fk=0.765 | CrossScratch/bb=0.743 |

## Mean / Dispersion by Split

| axis | value | n | dataset n | bit_F1 mean | bit_F1 std | FAR mean | FAR max | gap mean | gap std | pos mean | neg mean |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| abpos_Avar_B100 | A070_B100 | 1 | 1 | 0.9820 | 0.0000 | 0.54 | 0.54 | 0.044 | 0.000 | 0.7125 | 0.2250 |
| abpos_Avar_B100 | A080_B100 | 1 | 1 | 0.9837 | 0.0000 | 15.54 | 15.54 | -0.246 | 0.000 | 0.7169 | 0.2203 |
| abpos_Avar_B100 | A090_B100 | 1 | 1 | 0.9949 | 0.0000 | 0.33 | 0.33 | 0.130 | 0.000 | 0.7706 | 0.2209 |
| baseline | A100_B100_neg000_p050_grid9_g3_cmp100 | 2 | 2 | 0.9966 | 0.0010 | 0.62 | 0.86 | 0.186 | 0.066 | 0.7981 | 0.2234 |
| cutmix_p | p010 | 1 | 1 | 0.9835 | 0.0000 | 1.16 | 1.16 | 0.090 | 0.000 | 0.7442 | 0.2288 |
| cutmix_p | p020 | 1 | 1 | 0.9865 | 0.0000 | 1.01 | 1.01 | 0.098 | 0.000 | 0.7712 | 0.2393 |
| cutmix_p | p030 | 1 | 1 | 0.9964 | 0.0000 | 1.75 | 1.75 | 0.233 | 0.000 | 0.7904 | 0.2304 |
| cutmix_p | p040 | 1 | 1 | 0.9929 | 0.0000 | 3.14 | 3.14 | 0.034 | 0.000 | 0.7970 | 0.2225 |
| cutmix_p | p050 | 1 | 1 | 0.9959 | 0.0000 | 0.38 | 0.38 | 0.140 | 0.000 | 0.7890 | 0.2202 |
| cutmix_p | p055 | 1 | 1 | 0.9963 | 0.0000 | 0.49 | 0.49 | 0.203 | 0.000 | 0.8034 | 0.2225 |
| cutmix_p | p0575 | 1 | 1 | 0.9975 | 0.0000 | 0.40 | 0.40 | 0.285 | 0.000 | 0.7861 | 0.2234 |
| cutmix_p | p060 | 1 | 1 | 0.9944 | 0.0000 | 0.08 | 0.08 | 0.183 | 0.000 | 0.7902 | 0.2212 |
| cutmix_p | p0625 | 1 | 1 | 0.9943 | 0.0000 | 0.25 | 0.25 | 0.113 | 0.000 | 0.7937 | 0.2261 |
| cutmix_p | p065 | 1 | 1 | 0.9964 | 0.0000 | 1.54 | 1.54 | 0.171 | 0.000 | 0.8003 | 0.2032 |
| cutmix_p | p070 | 1 | 1 | 0.9957 | 0.0000 | 1.01 | 1.01 | 0.226 | 0.000 | 0.7981 | 0.2200 |
| cutmix_p | p080 | 1 | 1 | 0.9971 | 0.0000 | 1.96 | 1.96 | 0.234 | 0.000 | 0.8046 | 0.2235 |
| cutmix_p | p090 | 1 | 1 | 0.9921 | 0.0000 | 0.96 | 0.96 | 0.175 | 0.000 | 0.7886 | 0.2233 |
| cutmix_p | p100 | 1 | 1 | 0.9961 | 0.0000 | 1.43 | 1.43 | 0.253 | 0.000 | 0.7888 | 0.2261 |
| grid_g3 | grid12 | 1 | 1 | 0.9917 | 0.0000 | 0.11 | 0.11 | 0.181 | 0.000 | 0.7720 | 0.2270 |
| grid_g3 | grid3 | 1 | 1 | 0.9813 | 0.0000 | 8.15 | 8.15 | -0.037 | 0.000 | 0.8047 | 0.2247 |
| grid_g3 | grid6 | 1 | 1 | 0.9902 | 0.0000 | 1.32 | 1.32 | 0.111 | 0.000 | 0.7921 | 0.2144 |
| loss_variant | T10_T10_LS029500_g3_grid9_cmp10000_p05000_mpos065_s7_ep10_tr200_ev02000 | 1 | 1 | 0.0001 | 0.0000 | 0.00 | 0.00 | -0.089 | 0.000 | 0.7723 | 0.5879 |
| loss_variant | T4_T4_LS029500_g3_grid9_cmp10000_p05000_mpos065_s7_ep10_tr200_ev02000 | 1 | 1 | 0.0001 | 0.0000 | 0.00 | 0.00 | -0.089 | 0.000 | 0.7723 | 0.5879 |
| loss_variant | T6_T6_LS029500_g3_grid9_cmp10000_p05000_mpos065_s7_ep10_tr200_ev02000 | 1 | 1 | 0.9602 | 0.0000 | 8.48 | 8.48 | -0.337 | 0.000 | 0.9100 | 0.1123 |
| neg_target | neg0015 | 1 | 1 | 0.9960 | 0.0000 | 0.11 | 0.11 | 0.180 | 0.000 | 0.7958 | 0.2309 |
| neg_target | neg002 | 1 | 1 | 0.9964 | 0.0000 | 0.04 | 0.04 | 0.212 | 0.000 | 0.7605 | 0.2226 |
| neg_target | neg0025 | 1 | 1 | 0.9944 | 0.0000 | 3.32 | 3.32 | 0.145 | 0.000 | 0.7838 | 0.2240 |
| neg_target | neg003 | 1 | 1 | 0.9931 | 0.0000 | 0.08 | 0.08 | 0.138 | 0.000 | 0.7835 | 0.2266 |
| neg_target | neg005 | 1 | 1 | 0.9954 | 0.0000 | 0.06 | 0.06 | 0.204 | 0.000 | 0.7753 | 0.2287 |
| neg_target | neg010 | 1 | 1 | 0.9847 | 0.0000 | 0.37 | 0.37 | 0.028 | 0.000 | 0.7769 | 0.2458 |
| other | oneaxis_seed_repeat_grid_g3_grid12_s13_T7_LS029500_g3_grid12_cmp10000_p05000_mpos065_s13_ep10_tr200_ev02000 | 1 | 1 | 0.9978 | 0.0000 | 0.12 | 0.12 | 0.222 | 0.000 | 0.8098 | 0.2270 |
| other | oneaxis_seed_repeat_grid_g3_grid12_s42_T7_LS029500_g3_grid12_cmp10000_p05000_mpos065_s42_ep10_tr200_ev02000 | 1 | 1 | 0.7693 | 0.0000 | 98.76 | 98.76 | -0.685 | 0.000 | 0.5966 | 0.2626 |
| other | oneaxis_seed_repeat_grid_g3_grid15_s13_T7_LS029500_g3_grid15_cmp10000_p05000_mpos065_s13_ep10_tr200_ev02000 | 1 | 1 | 0.9957 | 0.0000 | 0.60 | 0.60 | 0.109 | 0.000 | 0.7638 | 0.2183 |
| other | oneaxis_seed_repeat_grid_g3_grid15_s42_T7_LS029500_g3_grid15_cmp10000_p05000_mpos065_s42_ep10_tr200_ev02000 | 1 | 1 | 0.9921 | 0.0000 | 32.01 | 32.01 | 0.014 | 0.000 | 0.7890 | 0.2301 |
| other | oneaxis_seed_repeat_grid_g3_grid18_s13_T7_LS029500_g3_grid18_cmp10000_p05000_mpos065_s13_ep10_tr200_ev02000 | 1 | 1 | 0.9975 | 0.0000 | 0.19 | 0.19 | 0.283 | 0.000 | 0.7873 | 0.2200 |
| other | oneaxis_seed_repeat_grid_g3_grid18_s42_T7_LS029500_g3_grid18_cmp10000_p05000_mpos065_s42_ep10_tr200_ev02000 | 1 | 1 | 0.9911 | 0.0000 | 9.23 | 9.23 | 0.039 | 0.000 | 0.7842 | 0.2266 |
| other | oneaxis_seed_repeat_grid_g3_grid3_s13_T7_LS029500_g3_grid3_cmp10000_p05000_mpos065_s13_ep10_tr200_ev02000 | 1 | 1 | 0.9919 | 0.0000 | 5.57 | 5.57 | 0.123 | 0.000 | 0.8123 | 0.2290 |
| other | oneaxis_seed_repeat_grid_g3_grid3_s42_T7_LS029500_g3_grid3_cmp10000_p05000_mpos065_s42_ep10_tr200_ev02000 | 1 | 1 | 0.9799 | 0.0000 | 24.54 | 24.54 | -0.042 | 0.000 | 0.8148 | 0.2330 |
| other | oneaxis_seed_repeat_grid_g3_grid6_s13_T7_LS029500_g3_grid6_cmp10000_p05000_mpos065_s13_ep10_tr200_ev02000 | 1 | 1 | 0.9928 | 0.0000 | 1.27 | 1.27 | 0.143 | 0.000 | 0.7932 | 0.2136 |
| other | oneaxis_seed_repeat_grid_g3_grid6_s42_T7_LS029500_g3_grid6_cmp10000_p05000_mpos065_s42_ep10_tr200_ev02000 | 1 | 1 | 0.9931 | 0.0000 | 10.93 | 10.93 | 0.128 | 0.000 | 0.7950 | 0.2352 |
| other | oneaxis_seed_repeat_grid_g3_grid9_s13_T7_LS029500_g3_grid9_cmp10000_p05000_mpos065_s13_ep10_tr200_ev02000 | 1 | 1 | 0.9942 | 0.0000 | 2.00 | 2.00 | 0.190 | 0.000 | 0.7887 | 0.2230 |
| other | oneaxis_seed_repeat_grid_g3_grid9_s42_T7_LS029500_g3_grid9_cmp10000_p05000_mpos065_s42_ep10_tr200_ev02000 | 1 | 1 | 0.9954 | 0.0000 | 68.91 | 68.91 | -0.093 | 0.000 | 0.8080 | 0.2409 |
| seed_repeat_abpos_Avar_B100 | A070_B100_s13 | 1 | 1 | 0.9835 | 0.0000 | 1.30 | 1.30 | 0.155 | 0.000 | 0.7207 | 0.1841 |
| seed_repeat_abpos_Avar_B100 | A070_B100_s42 | 1 | 1 | 0.9844 | 0.0000 | 1.20 | 1.20 | 0.105 | 0.000 | 0.7608 | 0.2106 |
| seed_repeat_abpos_Avar_B100 | A070_B100_s99 | 1 | 1 | 0.9942 | 0.0000 | 0.42 | 0.42 | 0.138 | 0.000 | 0.7371 | 0.2264 |
| seed_repeat_abpos_Avar_B100 | A080_B100_s13 | 1 | 1 | 0.9961 | 0.0000 | 0.00 | 0.00 | 0.258 | 0.000 | 0.7281 | 0.2277 |
| seed_repeat_abpos_Avar_B100 | A080_B100_s42 | 1 | 1 | 0.9964 | 0.0000 | 0.17 | 0.17 | 0.164 | 0.000 | 0.7662 | 0.2211 |
| seed_repeat_abpos_Avar_B100 | A080_B100_s99 | 1 | 1 | 0.9945 | 0.0000 | 1.69 | 1.69 | 0.207 | 0.000 | 0.7647 | 0.2278 |
| seed_repeat_abpos_Avar_B100 | A090_B100_s13 | 1 | 1 | 0.9920 | 0.0000 | 4.06 | 4.06 | 0.211 | 0.000 | 0.7583 | 0.2248 |
| seed_repeat_abpos_Avar_B100 | A090_B100_s42 | 1 | 1 | 0.9322 | 0.0000 | 25.78 | 25.78 | -0.202 | 0.000 | 0.7512 | 0.2170 |
| seed_repeat_abpos_Avar_B100 | A090_B100_s99 | 1 | 1 | 0.9954 | 0.0000 | 0.20 | 0.20 | 0.185 | 0.000 | 0.7908 | 0.2270 |
| seed_repeat_baseline | s13 | 1 | 1 | 0.9942 | 0.0000 | 2.00 | 2.00 | 0.190 | 0.000 | 0.7887 | 0.2230 |
| seed_repeat_baseline | s42 | 1 | 1 | 0.9954 | 0.0000 | 68.91 | 68.91 | -0.093 | 0.000 | 0.8080 | 0.2409 |
| seed_repeat_baseline | s99 | 1 | 1 | 0.9904 | 0.0000 | 1.51 | 1.51 | 0.192 | 0.000 | 0.8044 | 0.2261 |
| seed_repeat_neg | neg0015_s13 | 1 | 1 | 0.9958 | 0.0000 | 2.62 | 2.62 | 0.163 | 0.000 | 0.7845 | 0.2243 |
| seed_repeat_neg | neg0015_s42 | 1 | 1 | 0.9955 | 0.0000 | 57.15 | 57.15 | -0.074 | 0.000 | 0.7950 | 0.2443 |
| seed_repeat_neg | neg0015_s99 | 1 | 1 | 0.9921 | 0.0000 | 0.31 | 0.31 | 0.162 | 0.000 | 0.7920 | 0.2418 |
| seed_repeat_neg | neg0025_s13 | 1 | 1 | 0.9937 | 0.0000 | 0.43 | 0.43 | 0.126 | 0.000 | 0.7695 | 0.2265 |
| seed_repeat_neg | neg0025_s42 | 1 | 1 | 0.9874 | 0.0000 | 15.05 | 15.05 | 0.027 | 0.000 | 0.8099 | 0.2371 |
| seed_repeat_neg | neg0025_s99 | 1 | 1 | 0.9950 | 0.0000 | 1.93 | 1.93 | 0.196 | 0.000 | 0.7879 | 0.2331 |
| seed_repeat_neg | neg002_s13 | 1 | 1 | 0.9960 | 0.0000 | 0.60 | 0.60 | 0.143 | 0.000 | 0.7843 | 0.2237 |
| seed_repeat_neg | neg002_s42 | 1 | 1 | 0.9970 | 0.0000 | 42.66 | 42.66 | -0.010 | 0.000 | 0.8123 | 0.2371 |
| seed_repeat_neg | neg002_s99 | 1 | 1 | 0.9952 | 0.0000 | 0.08 | 0.08 | 0.192 | 0.000 | 0.8043 | 0.2253 |
| seed_repeat_neg | neg003_s13 | 1 | 1 | 0.9958 | 0.0000 | 0.71 | 0.71 | 0.198 | 0.000 | 0.7789 | 0.2254 |
| seed_repeat_neg | neg003_s42 | 1 | 1 | 0.9848 | 0.0000 | 11.17 | 11.17 | 0.042 | 0.000 | 0.8114 | 0.2374 |
| seed_repeat_neg | neg003_s99 | 1 | 1 | 0.9956 | 0.0000 | 0.30 | 0.30 | 0.242 | 0.000 | 0.8112 | 0.2291 |
| seed_repeat_neg | neg005_s13 | 1 | 1 | 0.9863 | 0.0000 | 16.11 | 16.11 | 0.047 | 0.000 | 0.7622 | 0.2347 |
| seed_repeat_neg | neg005_s42 | 1 | 1 | 0.9953 | 0.0000 | 1.93 | 1.93 | 0.216 | 0.000 | 0.8066 | 0.2268 |
| seed_repeat_neg | neg005_s99 | 1 | 1 | 0.9892 | 0.0000 | 6.15 | 6.15 | 0.024 | 0.000 | 0.7855 | 0.2315 |
| seed_repeat_neg | neg010_s13 | 1 | 1 | 0.9901 | 0.0000 | 0.35 | 0.35 | 0.036 | 0.000 | 0.7405 | 0.2236 |
| seed_repeat_neg | neg010_s42 | 1 | 1 | 0.9942 | 0.0000 | 75.66 | 75.66 | -0.212 | 0.000 | 0.7864 | 0.2602 |
| seed_repeat_p | p010_s13 | 1 | 1 | 0.9580 | 0.0000 | 1.19 | 1.19 | -0.076 | 0.000 | 0.7124 | 0.2290 |
| seed_repeat_p | p010_s42 | 1 | 1 | 0.9947 | 0.0000 | 0.57 | 0.57 | 0.192 | 0.000 | 0.7985 | 0.2248 |
| seed_repeat_p | p020_s13 | 1 | 1 | 0.9968 | 0.0000 | 1.46 | 1.46 | 0.088 | 0.000 | 0.7875 | 0.2191 |
| seed_repeat_p | p020_s42 | 1 | 1 | 0.9909 | 0.0000 | 4.06 | 4.06 | 0.002 | 0.000 | 0.7545 | 0.2297 |
| seed_repeat_p | p030_s13 | 1 | 1 | 0.9950 | 0.0000 | 0.40 | 0.40 | 0.085 | 0.000 | 0.7673 | 0.2243 |
| seed_repeat_p | p030_s42 | 1 | 1 | 0.9903 | 0.0000 | 2.23 | 2.23 | 0.132 | 0.000 | 0.7949 | 0.2216 |
| seed_repeat_p | p040_s13 | 1 | 1 | 0.9943 | 0.0000 | 0.94 | 0.94 | 0.203 | 0.000 | 0.7958 | 0.2361 |
| seed_repeat_p | p040_s42 | 1 | 1 | 0.9876 | 0.0000 | 9.93 | 9.93 | -0.031 | 0.000 | 0.7777 | 0.2278 |
| seed_repeat_p | p050_s13 | 1 | 1 | 0.9942 | 0.0000 | 2.00 | 2.00 | 0.190 | 0.000 | 0.7887 | 0.2230 |
| seed_repeat_p | p050_s42 | 1 | 1 | 0.9954 | 0.0000 | 68.91 | 68.91 | -0.093 | 0.000 | 0.8080 | 0.2409 |
| seed_repeat_p | p055_s13 | 1 | 1 | 0.9940 | 0.0000 | 11.89 | 11.89 | 0.117 | 0.000 | 0.7928 | 0.2226 |
| seed_repeat_p | p055_s42 | 1 | 1 | 0.9823 | 0.0000 | 24.37 | 24.37 | -0.120 | 0.000 | 0.7791 | 0.2439 |
| seed_repeat_p | p0575_s13 | 1 | 1 | 0.9952 | 0.0000 | 1.91 | 1.91 | 0.204 | 0.000 | 0.8011 | 0.2248 |
| seed_repeat_p | p0575_s42 | 1 | 1 | 0.9924 | 0.0000 | 52.98 | 52.98 | -0.084 | 0.000 | 0.7928 | 0.2372 |
| seed_repeat_p | p0575_s99 | 1 | 1 | 0.9938 | 0.0000 | 0.32 | 0.32 | 0.233 | 0.000 | 0.7919 | 0.2143 |
| seed_repeat_p | p060_s13 | 1 | 1 | 0.9961 | 0.0000 | 0.31 | 0.31 | 0.225 | 0.000 | 0.7830 | 0.2281 |
| seed_repeat_p | p060_s42 | 1 | 1 | 0.9874 | 0.0000 | 3.94 | 3.94 | 0.017 | 0.000 | 0.7987 | 0.2271 |
| seed_repeat_p | p060_s99 | 1 | 1 | 0.9950 | 0.0000 | 0.77 | 0.77 | 0.234 | 0.000 | 0.7959 | 0.2357 |
| seed_repeat_p | p0625_s13 | 1 | 1 | 0.9968 | 0.0000 | 0.80 | 0.80 | 0.239 | 0.000 | 0.8052 | 0.2191 |
| seed_repeat_p | p0625_s42 | 1 | 1 | 0.9901 | 0.0000 | 7.68 | 7.68 | 0.140 | 0.000 | 0.8096 | 0.2383 |
| seed_repeat_p | p065_s13 | 1 | 1 | 0.9943 | 0.0000 | 0.36 | 0.36 | 0.180 | 0.000 | 0.8173 | 0.2458 |
| seed_repeat_p | p065_s42 | 1 | 1 | 0.9960 | 0.0000 | 0.44 | 0.44 | 0.209 | 0.000 | 0.8250 | 0.2173 |
| seed_repeat_p | p070_s13 | 1 | 1 | 0.9969 | 0.0000 | 0.38 | 0.38 | 0.210 | 0.000 | 0.7949 | 0.2349 |
| seed_repeat_p | p070_s42 | 1 | 1 | 0.9978 | 0.0000 | 74.91 | 74.91 | -0.139 | 0.000 | 0.8028 | 0.2509 |
| seed_repeat_p | p080_s13 | 1 | 1 | 0.9965 | 0.0000 | 0.20 | 0.20 | 0.234 | 0.000 | 0.8084 | 0.2250 |
| seed_repeat_p | p080_s42 | 1 | 1 | 0.9888 | 0.0000 | 13.37 | 13.37 | 0.018 | 0.000 | 0.8038 | 0.2357 |
| seed_repeat_p | p090_s13 | 1 | 1 | 0.9960 | 0.0000 | 1.39 | 1.39 | 0.190 | 0.000 | 0.8014 | 0.2240 |
| seed_repeat_p | p090_s42 | 1 | 1 | 0.9929 | 0.0000 | 19.97 | 19.97 | 0.101 | 0.000 | 0.8108 | 0.2380 |
| seed_repeat_p | p100_s13 | 1 | 1 | 0.9959 | 0.0000 | 15.99 | 15.99 | -0.030 | 0.000 | 0.7870 | 0.2227 |
| seed_repeat_p | p100_s42 | 1 | 1 | 0.9890 | 0.0000 | 32.06 | 32.06 | 0.022 | 0.000 | 0.8174 | 0.2514 |

## Axis Best So Far

| axis | best value | bit_F1 | FAR | gap | reason |
|---|---|---:|---:|---:|---|
| abpos_Avar_B100 | A090_B100 | 0.9949 | 0.33 | 0.130 | worst POS bank_boundary+scratch/sc=0.652; worst NEG Invalid/sr=0.522 |
| baseline | A100_B100_neg000_p050_grid9_g3_cmp100 | 0.9973 | 0.86 | 0.233 | worst POS bank_boundary+scratch/sc=0.741; worst NEG Invalid/sr=0.508 |
| cutmix_p | p0575 | 0.9975 | 0.40 | 0.285 | worst POS fork+scratch_rot/sr=0.726; worst NEG Invalid/fk=0.441 |
| grid_g3 | grid12 | 0.9917 | 0.11 | 0.181 | worst POS fork+scratch/fk=0.634; worst NEG Invalid/fk=0.453 |
| loss_variant | T6_T6_LS029500_g3_grid9_cmp10000_p05000_mpos065_s7_ep10_tr200_ev02000 | 0.9602 | 8.48 | -0.337 | worst POS bank_boundary+scratch/sc=0.197; worst NEG Invalid/fk=0.534 |
| neg_target | neg002 | 0.9964 | 0.04 | 0.212 | worst POS bank_boundary+scratch/sc=0.654; worst NEG Normal/sc=0.442 |
| other | oneaxis_seed_repeat_grid_g3_grid12_s13_T7_LS029500_g3_grid12_cmp10000_p05000_mpos065_s13_ep10_tr200_ev02000 | 0.9978 | 0.12 | 0.222 | worst POS bank_boundary+scratch/sc=0.716; worst NEG Invalid/sr=0.494 |
| seed_repeat_abpos_Avar_B100 | A080_B100_s42 | 0.9964 | 0.17 | 0.164 | worst POS bank_boundary+scratch/sc=0.615; worst NEG DiagonalSmear/sc=0.451 |
| seed_repeat_baseline | s42 | 0.9954 | 68.91 | -0.093 | worst POS bank_boundary+scratch/sc=0.743; worst NEG CrossScratch/bb=0.836 |
| seed_repeat_neg | neg002_s42 | 0.9970 | 42.66 | -0.010 | worst POS fork+scratch/sc=0.733; worst NEG CrossScratch/bb=0.743 |
| seed_repeat_p | p070_s42 | 0.9978 | 74.91 | -0.139 | worst POS bank_boundary+scratch/sc=0.722; worst NEG Starburst/bb=0.861 |

## Next Stage Rule

- 모든 실험/관리자 표에는 FAR 컬럼을 넣는다. 내부 후보 gate도 `Total FAR <= 1%`를 같이 본다.
- 1축에서 `bit_F1 >= 0.993`, `Total FAR <= 1%`, `gap`이 baseline보다 개선되는 값을 후보로 둔다.
- 여러 데이터셋/seed에 걸친 평균과 표준편차를 같이 본다. 단일 row 최고값보다 `mean(bit_F1)`과 `std(gap)`이 더 중요하다.
- 후보가 2개 이상이면 2축 조합을 만든다. 예: `A/B target best` x `neg target best`.
- 2축 조합에서 다시 상위 조건이 안정되면 3축 조합으로 확장한다.
- 이미 충분히 결과가 많은 `cmp` 축은 새로 반복하지 않고 기존 evidence를 사용한다.
