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
4. 매 row는 bit_F1뿐 아니라 POS min / NEG max gap을 같이 본다.
5. 관리자용 표는 오탐률 컬럼을 빼고, probability separation 중심으로 표시한다.

## Active / Planned Queue

| phase | axis | values | status |
|---|---|---|---|
| 1-axis | A/B positive target | A=0.90/0.80/0.70, B=1.00 fixed | running / queued |
| 1-axis | neg target | 0.015 / 0.02 / 0.025 / 0.03 / 0.05 / 0.10 | queued |
| 1-axis | cutmix_p | 0.20 / 0.30 / 0.40 / 0.55 / 0.575 / 0.60 / 0.625 / 0.65 / 0.70 / 0.80 / 0.90 / 1.00 | running / queued |
| 1-axis | loss variant | T10 / T4 / T6 completed on frozen_original; collapsed or leaked | pruned from transfer repeats |
| repeat | seed stability | baseline, neg=0.02/0.05, p=0.55/0.575/0.60/0.65/0.70/0.80 at seed 13/42/99 | queued |
| 1-axis | grid, g=3 | 3x3 / 6x6 / 9x9 / 12x12 | running / queued |
| repeat | grid seed stability | g=3 grid 3x3/6x6/9x9/12x12 at seed 13/42/99 | queued |
| 1-axis | group-grid alignment | g=2 grid6 / g=4 grid12, baseline g=3 grid9 | queued |
| existing evidence | cmp | 0.5 / 0.7 / 0.8 / 1.0 | mined, not rerun |
| multi-dataset | transfer data | frozen_original, gapstress seed31/97, frozen snapshots | running after restart |
| 2-factor | top 1-axis pairs | p=0.575-centered neg/p, A=0.90/p, A=0.90/neg, grid/p | pending |
| 3-factor | top 2-factor neighborhood | compact A=0.90 + neg + p=0.575 candidates | pending |

## Completed Rows

| dataset | axis | value | bit_F1 | pos | neg | gap | worst POS min | worst NEG max |
|---|---|---|---:|---:|---:|---:|---|---|
| frozen_original | abpos_Avar_B100 | A070_B100 | 0.9820 | 0.7125 | 0.2250 | 0.044 | bank_boundary+scratch/sc=0.536 | CrossScratch/bb=0.492 |
| frozen_original | abpos_Avar_B100 | A080_B100 | 0.9837 | 0.7169 | 0.2203 | -0.246 | bank_boundary+scratch/sc=0.257 | Starburst/sc=0.503 |
| frozen_original | abpos_Avar_B100 | A090_B100 | 0.9949 | 0.7706 | 0.2209 | 0.130 | bank_boundary+scratch/sc=0.652 | Invalid/sr=0.522 |
| frozen_iter116J_orig814_eval_n20000 | baseline | A100_B100_neg000_p050_grid9_g3_cmp100 | 0.9973 | 0.8072 | 0.2266 | 0.233 | bank_boundary+scratch/sc=0.741 | Invalid/sr=0.508 |
| frozen_original | baseline | A100_B100_neg000_p050_grid9_g3_cmp100 | 0.9959 | 0.7890 | 0.2202 | 0.140 | bank_boundary+scratch/sc=0.676 | Invalid/fk=0.536 |
| frozen_original | cutmix_p | p020 | 0.9865 | 0.7712 | 0.2393 | 0.098 | bank_boundary+scratch/sc=0.544 | Invalid/fk=0.446 |
| frozen_original | cutmix_p | p030 | 0.9964 | 0.7904 | 0.2304 | 0.233 | bank_boundary+scratch/sc=0.685 | Invalid/fk=0.452 |
| frozen_original | cutmix_p | p040 | 0.9929 | 0.7970 | 0.2225 | 0.034 | bank_boundary+scratch/sc=0.617 | CrossScratch/bb=0.583 |
| frozen_original | cutmix_p | p055 | 0.9963 | 0.8034 | 0.2225 | 0.203 | fork+scratch/fk=0.736 | Invalid/fk=0.533 |
| frozen_original | cutmix_p | p0575 | 0.9975 | 0.7861 | 0.2234 | 0.285 | fork+scratch_rot/sr=0.726 | Invalid/fk=0.441 |
| frozen_original | cutmix_p | p060 | 0.9944 | 0.7902 | 0.2212 | 0.183 | bank_boundary+scratch/sc=0.643 | CrossScratch/bb=0.460 |
| frozen_original | cutmix_p | p0625 | 0.9943 | 0.7937 | 0.2261 | 0.113 | bank_boundary+scratch/sc=0.675 | Normal/sc=0.562 |
| frozen_original | cutmix_p | p065 | 0.9964 | 0.8003 | 0.2032 | 0.171 | fork+scratch/fk=0.702 | Invalid/fk=0.531 |
| frozen_original | cutmix_p | p070 | 0.9957 | 0.7981 | 0.2200 | 0.226 | fork+scratch/fk=0.689 | Invalid/fk=0.463 |
| frozen_original | cutmix_p | p080 | 0.9971 | 0.8046 | 0.2235 | 0.234 | fork+scratch/sc=0.709 | Normal/sc=0.475 |
| frozen_original | grid_g3 | grid12 | 0.9917 | 0.7720 | 0.2270 | 0.181 | fork+scratch/fk=0.634 | Invalid/fk=0.453 |
| frozen_original | grid_g3 | grid3 | 0.9813 | 0.8047 | 0.2247 | -0.037 | bank_boundary+scratch/sc=0.668 | Invalid/fk=0.705 |
| frozen_original | grid_g3 | grid6 | 0.9902 | 0.7921 | 0.2144 | 0.111 | bank_boundary+scratch/sc=0.640 | Invalid/fk=0.529 |
| frozen_original | loss_variant | T10_T10_LS029500_g3_grid9_cmp10000_p05000_mpos065_s7_ep10_tr200_ev02000 | 0.0001 | 0.7723 | 0.5879 | -0.089 | bank_boundary+scratch/sc=0.682 | Invalid/fk=0.771 |
| frozen_original | loss_variant | T4_T4_LS029500_g3_grid9_cmp10000_p05000_mpos065_s7_ep10_tr200_ev02000 | 0.0001 | 0.7723 | 0.5879 | -0.089 | bank_boundary+scratch/sc=0.682 | Invalid/fk=0.771 |
| frozen_original | loss_variant | T6_T6_LS029500_g3_grid9_cmp10000_p05000_mpos065_s7_ep10_tr200_ev02000 | 0.9602 | 0.9100 | 0.1123 | -0.337 | bank_boundary+scratch/sc=0.197 | Invalid/fk=0.534 |
| frozen_original | neg_target | neg0015 | 0.9960 | 0.7958 | 0.2309 | 0.180 | bank_boundary+scratch/sc=0.688 | Normal/sc=0.508 |
| frozen_original | neg_target | neg002 | 0.9964 | 0.7605 | 0.2226 | 0.212 | bank_boundary+scratch/sc=0.654 | Normal/sc=0.442 |
| frozen_original | neg_target | neg0025 | 0.9944 | 0.7838 | 0.2240 | 0.145 | bank_boundary+scratch/sc=0.656 | Normal/sc=0.511 |
| frozen_original | neg_target | neg003 | 0.9931 | 0.7835 | 0.2266 | 0.138 | bank_boundary+scratch/sc=0.631 | Normal/sc=0.493 |
| frozen_original | neg_target | neg005 | 0.9954 | 0.7753 | 0.2287 | 0.204 | scratch+scratch_rot/sc=0.656 | Normal/sc=0.452 |
| frozen_original | neg_target | neg010 | 0.9847 | 0.7769 | 0.2458 | 0.028 | bank_boundary+scratch/sc=0.594 | Invalid/fk=0.566 |
| frozen_original | seed_repeat_baseline | s13 | 0.9942 | 0.7887 | 0.2230 | 0.190 | scratch+scratch_rot/sr=0.641 | Invalid/sr=0.451 |
| frozen_original | seed_repeat_baseline | s42 | 0.9954 | 0.8080 | 0.2409 | -0.093 | bank_boundary+scratch/sc=0.743 | CrossScratch/bb=0.836 |
| frozen_original | seed_repeat_baseline | s99 | 0.9904 | 0.8044 | 0.2261 | 0.192 | bank_boundary+scratch/sc=0.701 | Invalid/sc=0.509 |
| frozen_original | seed_repeat_neg | neg002_s13 | 0.9960 | 0.7843 | 0.2237 | 0.143 | scratch+scratch_rot/sr=0.647 | Invalid/sr=0.504 |
| frozen_original | seed_repeat_neg | neg002_s42 | 0.9970 | 0.8123 | 0.2371 | -0.010 | fork+scratch/sc=0.733 | CrossScratch/bb=0.743 |
| frozen_original | seed_repeat_neg | neg002_s99 | 0.9952 | 0.8043 | 0.2253 | 0.192 | bank_boundary+scratch/sc=0.713 | Invalid/sc=0.521 |
| frozen_original | seed_repeat_neg | neg005_s13 | 0.9863 | 0.7622 | 0.2347 | 0.047 | bank_boundary+scratch/sc=0.637 | CenterDonut/bb=0.590 |
| frozen_original | seed_repeat_neg | neg005_s42 | 0.9953 | 0.8066 | 0.2268 | 0.216 | bank_boundary+scratch/sc=0.692 | Invalid/sr=0.476 |
| frozen_original | seed_repeat_neg | neg005_s99 | 0.9892 | 0.7855 | 0.2315 | 0.024 | bank_boundary+scratch/sc=0.559 | CrossScratch/bb=0.535 |
| frozen_original | seed_repeat_p | p0575_s13 | 0.9952 | 0.8011 | 0.2248 | 0.204 | fork+scratch/fk=0.731 | Invalid/sr=0.527 |
| frozen_original | seed_repeat_p | p0575_s42 | 0.9924 | 0.7928 | 0.2372 | -0.084 | bank_boundary+scratch/sc=0.679 | CenterDonut/bb=0.763 |
| frozen_original | seed_repeat_p | p0575_s99 | 0.9938 | 0.7919 | 0.2143 | 0.233 | bank_boundary+scratch/sc=0.685 | Invalid/sc=0.452 |
| frozen_original | seed_repeat_p | p060_s13 | 0.9961 | 0.7830 | 0.2281 | 0.225 | bank_boundary+scratch/sc=0.680 | Invalid/sr=0.455 |
| frozen_original | seed_repeat_p | p060_s42 | 0.9874 | 0.7987 | 0.2271 | 0.017 | bank_boundary+scratch/sc=0.638 | Normal/sr=0.621 |
| frozen_original | seed_repeat_p | p060_s99 | 0.9950 | 0.7959 | 0.2357 | 0.234 | bank_boundary+scratch/bb=0.706 | Invalid/fk=0.472 |

## Mean / Dispersion by Split

| axis | value | n | dataset n | bit_F1 mean | bit_F1 std | gap mean | gap std | pos mean | neg mean |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| abpos_Avar_B100 | A070_B100 | 1 | 1 | 0.9820 | 0.0000 | 0.044 | 0.000 | 0.7125 | 0.2250 |
| abpos_Avar_B100 | A080_B100 | 1 | 1 | 0.9837 | 0.0000 | -0.246 | 0.000 | 0.7169 | 0.2203 |
| abpos_Avar_B100 | A090_B100 | 1 | 1 | 0.9949 | 0.0000 | 0.130 | 0.000 | 0.7706 | 0.2209 |
| baseline | A100_B100_neg000_p050_grid9_g3_cmp100 | 2 | 2 | 0.9966 | 0.0010 | 0.186 | 0.066 | 0.7981 | 0.2234 |
| cutmix_p | p020 | 1 | 1 | 0.9865 | 0.0000 | 0.098 | 0.000 | 0.7712 | 0.2393 |
| cutmix_p | p030 | 1 | 1 | 0.9964 | 0.0000 | 0.233 | 0.000 | 0.7904 | 0.2304 |
| cutmix_p | p040 | 1 | 1 | 0.9929 | 0.0000 | 0.034 | 0.000 | 0.7970 | 0.2225 |
| cutmix_p | p055 | 1 | 1 | 0.9963 | 0.0000 | 0.203 | 0.000 | 0.8034 | 0.2225 |
| cutmix_p | p0575 | 1 | 1 | 0.9975 | 0.0000 | 0.285 | 0.000 | 0.7861 | 0.2234 |
| cutmix_p | p060 | 1 | 1 | 0.9944 | 0.0000 | 0.183 | 0.000 | 0.7902 | 0.2212 |
| cutmix_p | p0625 | 1 | 1 | 0.9943 | 0.0000 | 0.113 | 0.000 | 0.7937 | 0.2261 |
| cutmix_p | p065 | 1 | 1 | 0.9964 | 0.0000 | 0.171 | 0.000 | 0.8003 | 0.2032 |
| cutmix_p | p070 | 1 | 1 | 0.9957 | 0.0000 | 0.226 | 0.000 | 0.7981 | 0.2200 |
| cutmix_p | p080 | 1 | 1 | 0.9971 | 0.0000 | 0.234 | 0.000 | 0.8046 | 0.2235 |
| grid_g3 | grid12 | 1 | 1 | 0.9917 | 0.0000 | 0.181 | 0.000 | 0.7720 | 0.2270 |
| grid_g3 | grid3 | 1 | 1 | 0.9813 | 0.0000 | -0.037 | 0.000 | 0.8047 | 0.2247 |
| grid_g3 | grid6 | 1 | 1 | 0.9902 | 0.0000 | 0.111 | 0.000 | 0.7921 | 0.2144 |
| loss_variant | T10_T10_LS029500_g3_grid9_cmp10000_p05000_mpos065_s7_ep10_tr200_ev02000 | 1 | 1 | 0.0001 | 0.0000 | -0.089 | 0.000 | 0.7723 | 0.5879 |
| loss_variant | T4_T4_LS029500_g3_grid9_cmp10000_p05000_mpos065_s7_ep10_tr200_ev02000 | 1 | 1 | 0.0001 | 0.0000 | -0.089 | 0.000 | 0.7723 | 0.5879 |
| loss_variant | T6_T6_LS029500_g3_grid9_cmp10000_p05000_mpos065_s7_ep10_tr200_ev02000 | 1 | 1 | 0.9602 | 0.0000 | -0.337 | 0.000 | 0.9100 | 0.1123 |
| neg_target | neg0015 | 1 | 1 | 0.9960 | 0.0000 | 0.180 | 0.000 | 0.7958 | 0.2309 |
| neg_target | neg002 | 1 | 1 | 0.9964 | 0.0000 | 0.212 | 0.000 | 0.7605 | 0.2226 |
| neg_target | neg0025 | 1 | 1 | 0.9944 | 0.0000 | 0.145 | 0.000 | 0.7838 | 0.2240 |
| neg_target | neg003 | 1 | 1 | 0.9931 | 0.0000 | 0.138 | 0.000 | 0.7835 | 0.2266 |
| neg_target | neg005 | 1 | 1 | 0.9954 | 0.0000 | 0.204 | 0.000 | 0.7753 | 0.2287 |
| neg_target | neg010 | 1 | 1 | 0.9847 | 0.0000 | 0.028 | 0.000 | 0.7769 | 0.2458 |
| seed_repeat_baseline | s13 | 1 | 1 | 0.9942 | 0.0000 | 0.190 | 0.000 | 0.7887 | 0.2230 |
| seed_repeat_baseline | s42 | 1 | 1 | 0.9954 | 0.0000 | -0.093 | 0.000 | 0.8080 | 0.2409 |
| seed_repeat_baseline | s99 | 1 | 1 | 0.9904 | 0.0000 | 0.192 | 0.000 | 0.8044 | 0.2261 |
| seed_repeat_neg | neg002_s13 | 1 | 1 | 0.9960 | 0.0000 | 0.143 | 0.000 | 0.7843 | 0.2237 |
| seed_repeat_neg | neg002_s42 | 1 | 1 | 0.9970 | 0.0000 | -0.010 | 0.000 | 0.8123 | 0.2371 |
| seed_repeat_neg | neg002_s99 | 1 | 1 | 0.9952 | 0.0000 | 0.192 | 0.000 | 0.8043 | 0.2253 |
| seed_repeat_neg | neg005_s13 | 1 | 1 | 0.9863 | 0.0000 | 0.047 | 0.000 | 0.7622 | 0.2347 |
| seed_repeat_neg | neg005_s42 | 1 | 1 | 0.9953 | 0.0000 | 0.216 | 0.000 | 0.8066 | 0.2268 |
| seed_repeat_neg | neg005_s99 | 1 | 1 | 0.9892 | 0.0000 | 0.024 | 0.000 | 0.7855 | 0.2315 |
| seed_repeat_p | p0575_s13 | 1 | 1 | 0.9952 | 0.0000 | 0.204 | 0.000 | 0.8011 | 0.2248 |
| seed_repeat_p | p0575_s42 | 1 | 1 | 0.9924 | 0.0000 | -0.084 | 0.000 | 0.7928 | 0.2372 |
| seed_repeat_p | p0575_s99 | 1 | 1 | 0.9938 | 0.0000 | 0.233 | 0.000 | 0.7919 | 0.2143 |
| seed_repeat_p | p060_s13 | 1 | 1 | 0.9961 | 0.0000 | 0.225 | 0.000 | 0.7830 | 0.2281 |
| seed_repeat_p | p060_s42 | 1 | 1 | 0.9874 | 0.0000 | 0.017 | 0.000 | 0.7987 | 0.2271 |
| seed_repeat_p | p060_s99 | 1 | 1 | 0.9950 | 0.0000 | 0.234 | 0.000 | 0.7959 | 0.2357 |

## Axis Best So Far

| axis | best value | bit_F1 | gap | reason |
|---|---|---:|---:|---|
| abpos_Avar_B100 | A090_B100 | 0.9949 | 0.130 | worst POS bank_boundary+scratch/sc=0.652; worst NEG Invalid/sr=0.522 |
| baseline | A100_B100_neg000_p050_grid9_g3_cmp100 | 0.9973 | 0.233 | worst POS bank_boundary+scratch/sc=0.741; worst NEG Invalid/sr=0.508 |
| cutmix_p | p0575 | 0.9975 | 0.285 | worst POS fork+scratch_rot/sr=0.726; worst NEG Invalid/fk=0.441 |
| grid_g3 | grid12 | 0.9917 | 0.181 | worst POS fork+scratch/fk=0.634; worst NEG Invalid/fk=0.453 |
| loss_variant | T6_T6_LS029500_g3_grid9_cmp10000_p05000_mpos065_s7_ep10_tr200_ev02000 | 0.9602 | -0.337 | worst POS bank_boundary+scratch/sc=0.197; worst NEG Invalid/fk=0.534 |
| neg_target | neg002 | 0.9964 | 0.212 | worst POS bank_boundary+scratch/sc=0.654; worst NEG Normal/sc=0.442 |
| seed_repeat_baseline | s42 | 0.9954 | -0.093 | worst POS bank_boundary+scratch/sc=0.743; worst NEG CrossScratch/bb=0.836 |
| seed_repeat_neg | neg002_s42 | 0.9970 | -0.010 | worst POS fork+scratch/sc=0.733; worst NEG CrossScratch/bb=0.743 |
| seed_repeat_p | p060_s13 | 0.9961 | 0.225 | worst POS bank_boundary+scratch/sc=0.680; worst NEG Invalid/sr=0.455 |

## Next Stage Rule

- 관리자 표에는 오탐률 컬럼을 넣지 않는다. 단, 내부 후보 gate에서는 `Total FAR <= 1%`를 같이 본다.
- 1축에서 `bit_F1 >= 0.993`, `Total FAR <= 1%`, `gap`이 baseline보다 개선되는 값을 후보로 둔다.
- 여러 데이터셋/seed에 걸친 평균과 표준편차를 같이 본다. 단일 row 최고값보다 `mean(bit_F1)`과 `std(gap)`이 더 중요하다.
- 후보가 2개 이상이면 2축 조합을 만든다. 예: `A/B target best` x `neg target best`.
- 2축 조합에서 다시 상위 조건이 안정되면 3축 조합으로 확장한다.
- 이미 충분히 결과가 많은 `cmp` 축은 새로 반복하지 않고 기존 evidence를 사용한다.
