# FCM-PM Val Selection 관리자 보고서

작성일: 2026-06-03  
대상: chip multilabel defect classification  
목적: `val_f1` best model과 `val_margin` best model을 train/eval sample 수별로 비교하고, 관리자 보고용 대표 조건을 선정한다.

## 1. 결론

현재 matrix 기준 대표 조건은 다음이다.

| train/class | pick | eval/class | bit_F1 | pos_prob | neg_prob | gap | per-bit F1 (bb/fk/sc/sr) |
|---:|---|---:|---:|---:|---:|---:|---|
| 200 | val_margin | 20000 | 0.9958 | 0.7801 | 0.2292 | 0.5509 | 0.998 / 0.993 / 0.992 / 1.000 |

선정 이유:

- `val_f1` 기준은 validation F1이 빨리 포화되어 early checkpoint를 고르는 경향이 있다.
- `val_margin` 기준은 POS/NEG probability gap을 직접 키운 checkpoint를 고른다.
- train/class 200은 high bit_F1과 큰 probability gap이 동시에 유지된다.
- train/class 400은 POS 성능은 높지만 OOD negative class에서 `bank_boundary` tail이 올라가므로 현재 조건 그대로는 대표 조건으로 두지 않는다.

## 2. 고정 실험 조건

### Dataset

| 항목 | 경로 |
|---|---|
| train root | `E:/data/images/classification_chips_iter116J_orig814_260529` |
| validation root | `E:/data/images/chip_multilabel_v15direct_n2000` |
| eval root | `E:/data/images/eval_n20000` |

### Recipe

| 항목 | 값 |
|---|---|
| variant | `T7` |
| backbone | `convnextv2_base.fcmae_ft_in22k_in1k_384` |
| image size | `384` |
| epochs | `10` |
| batch / accum | `2 / 8` |
| learning rate | `1e-4` |
| label smoothing | `0.295` |
| seed | `7` |
| cutmix mode | `complement` |
| cutmix pair | `masked` |
| cutmix probability | `0.5` |
| FCM-PM group count | `3` |
| grid | `9x9` |
| complete label scale | `1.0` |
| A/B label | `0.90 / 1.00` |
| mask positive target | `0.65` |
| pair bias | `fork,scratch:2` |
| normal in train | excluded |

### Checkpoint Selection

| pick | 저장 파일 | 의미 |
|---|---|---|
| `val_f1` | `best_f1_model.pth` | validation bit F1 최대 epoch |
| `val_margin` | `best_model.pth` | validation POS/NEG probability gap 최대 epoch |

## 3. Train/Eval Matrix

관리자 보고용 matrix에서는 오탐률 컬럼을 제외하고, bit_F1과 probability separation 중심으로 표시한다.

| train/class | pick | eval/class | bit_F1 | pos_prob | neg_prob | gap | per-bit F1 (bb/fk/sc/sr) |
|---:|---|---:|---:|---:|---:|---:|---|
| 50 | val_f1 | 200 | 0.7568 | 0.6120 | 0.2217 | 0.3903 | 0.956 / 0.663 / 0.499 / 0.909 |
| 50 | val_f1 | 2000 | 0.7533 | 0.6123 | 0.2214 | 0.3909 | 0.949 / 0.653 / 0.503 / 0.908 |
| 50 | val_f1 | 20000 | 0.7537 | 0.6128 | 0.2213 | 0.3915 | 0.949 / 0.646 / 0.511 / 0.909 |
| 50 | val_margin | 200 | 0.9132 | 0.7194 | 0.2297 | 0.4896 | 0.999 / 0.959 / 0.696 / 0.999 |
| 50 | val_margin | 2000 | 0.9167 | 0.7199 | 0.2299 | 0.4901 | 0.998 / 0.963 / 0.706 / 1.000 |
| 50 | val_margin | 20000 | 0.9163 | 0.7201 | 0.2300 | 0.4901 | 0.999 / 0.962 / 0.705 / 1.000 |
| 100 | val_f1 | 200 | 0.9521 | 0.6523 | 0.2121 | 0.4402 | 0.956 / 0.936 / 0.941 / 0.975 |
| 100 | val_f1 | 2000 | 0.9462 | 0.6521 | 0.2118 | 0.4403 | 0.941 / 0.945 / 0.933 / 0.966 |
| 100 | val_f1 | 20000 | 0.9445 | 0.6522 | 0.2120 | 0.4402 | 0.940 / 0.944 / 0.929 / 0.966 |
| 100 | val_margin | 200 | 0.9933 | 0.7674 | 0.2146 | 0.5528 | 0.996 / 0.998 / 0.979 / 1.000 |
| 100 | val_margin | 2000 | 0.9925 | 0.7671 | 0.2145 | 0.5526 | 0.997 / 0.995 / 0.979 / 0.999 |
| 100 | val_margin | 20000 | 0.9925 | 0.7671 | 0.2146 | 0.5524 | 0.997 / 0.994 / 0.980 / 1.000 |
| 200 | val_f1 | 200 | 0.9935 | 0.7468 | 0.2348 | 0.5119 | 0.993 / 0.993 / 0.989 / 0.999 |
| 200 | val_f1 | 2000 | 0.9926 | 0.7461 | 0.2343 | 0.5118 | 0.994 / 0.991 / 0.987 / 0.999 |
| 200 | val_f1 | 20000 | 0.9928 | 0.7463 | 0.2343 | 0.5119 | 0.994 / 0.990 / 0.988 / 1.000 |
| 200 | val_margin | 200 | 0.9971 | 0.7815 | 0.2293 | 0.5521 | 0.998 / 0.998 / 0.992 / 1.000 |
| 200 | val_margin | 2000 | 0.9959 | 0.7801 | 0.2291 | 0.5510 | 0.998 / 0.994 / 0.992 / 1.000 |
| 200 | val_margin | 20000 | 0.9958 | 0.7801 | 0.2292 | 0.5509 | 0.998 / 0.993 / 0.992 / 1.000 |
| 400 | val_f1 | 200 | 0.9705 | 0.7292 | 0.2239 | 0.5053 | 0.991 / 0.984 / 0.919 / 0.988 |
| 400 | val_f1 | 2000 | 0.9702 | 0.7287 | 0.2240 | 0.5047 | 0.988 / 0.984 / 0.921 / 0.988 |
| 400 | val_f1 | 20000 | 0.9710 | 0.7289 | 0.2239 | 0.5050 | 0.989 / 0.983 / 0.924 / 0.988 |
| 400 | val_margin | 200 | 0.9967 | 0.7590 | 0.2270 | 0.5320 | 0.991 / 0.996 / 0.999 / 1.000 |
| 400 | val_margin | 2000 | 0.9944 | 0.7580 | 0.2270 | 0.5310 | 0.989 / 0.992 / 0.997 / 1.000 |
| 400 | val_margin | 20000 | 0.9947 | 0.7580 | 0.2270 | 0.5310 | 0.990 / 0.992 / 0.997 / 1.000 |

## 4. Probability Pattern

차트 파일은 이미지 출력 절대규칙에 따라 `E:/data/images/` 아래에 저장했다.

![FCM-PM probability bars all conditions](E:/data/images/chip_multilabel_reports/manager_260603/fcm_pm_prob_bars_all_conditions.png)

![FCM-PM OOD bb tail all conditions](E:/data/images/chip_multilabel_reports/manager_260603/fcm_pm_ood_bb_tail_all_conditions.png)

### Group Mean Probability

| condition | group | bb | fk | sc | sr |
|---|---|---:|---:|---:|---:|
| train=50 val_f1 | single POS | 0.362 | 0.304 | 0.279 | 0.348 |
| train=50 val_f1 | combo POS | 0.578 | 0.273 | 0.174 | 0.414 |
| train=50 val_f1 | OOD NEG | 0.133 | 0.343 | 0.293 | 0.200 |
| train=50 val_f1 | Normal/Invalid | 0.206 | 0.503 | 0.351 | 0.311 |
| train=50 val_margin | single POS | 0.328 | 0.315 | 0.330 | 0.325 |
| train=50 val_margin | combo POS | 0.473 | 0.368 | 0.328 | 0.419 |
| train=50 val_margin | OOD NEG | 0.174 | 0.335 | 0.556 | 0.177 |
| train=50 val_margin | Normal/Invalid | 0.197 | 0.428 | 0.557 | 0.252 |
| train=100 val_f1 | single POS | 0.328 | 0.308 | 0.324 | 0.307 |
| train=100 val_f1 | combo POS | 0.493 | 0.379 | 0.239 | 0.377 |
| train=100 val_f1 | OOD NEG | 0.442 | 0.178 | 0.312 | 0.114 |
| train=100 val_f1 | Normal/Invalid | 0.269 | 0.296 | 0.419 | 0.230 |
| train=100 val_margin | single POS | 0.322 | 0.323 | 0.325 | 0.328 |
| train=100 val_margin | combo POS | 0.453 | 0.408 | 0.395 | 0.438 |
| train=100 val_margin | OOD NEG | 0.375 | 0.152 | 0.461 | 0.140 |
| train=100 val_margin | Normal/Invalid | 0.261 | 0.290 | 0.483 | 0.323 |
| train=200 val_f1 | single POS | 0.316 | 0.314 | 0.336 | 0.320 |
| train=200 val_f1 | combo POS | 0.439 | 0.385 | 0.402 | 0.410 |
| train=200 val_f1 | OOD NEG | 0.451 | 0.280 | 0.318 | 0.284 |
| train=200 val_f1 | Normal/Invalid | 0.321 | 0.356 | 0.391 | 0.383 |
| train=200 val_margin | single POS | 0.323 | 0.324 | 0.324 | 0.325 |
| train=200 val_margin | combo POS | 0.465 | 0.428 | 0.426 | 0.430 |
| train=200 val_margin | OOD NEG | 0.281 | 0.227 | 0.402 | 0.332 |
| train=200 val_margin | Normal/Invalid | 0.287 | 0.310 | 0.415 | 0.407 |
| train=400 val_f1 | single POS | 0.323 | 0.352 | 0.327 | 0.327 |
| train=400 val_f1 | combo POS | 0.477 | 0.392 | 0.351 | 0.415 |
| train=400 val_f1 | OOD NEG | 0.170 | 0.274 | 0.401 | 0.324 |
| train=400 val_f1 | Normal/Invalid | 0.132 | 0.336 | 0.390 | 0.447 |
| train=400 val_margin | single POS | 0.321 | 0.322 | 0.329 | 0.326 |
| train=400 val_margin | combo POS | 0.447 | 0.403 | 0.401 | 0.416 |
| train=400 val_margin | OOD NEG | 0.390 | 0.238 | 0.346 | 0.307 |
| train=400 val_margin | Normal/Invalid | 0.263 | 0.296 | 0.379 | 0.416 |

읽는 법:

- single POS는 자기 bit 하나만 높고 나머지는 낮은 구조지만, group mean으로 평균내면 네 bit가 비슷하게 보인다. class별 표에서 실제 one-hot 구조를 확인해야 한다.
- combo POS는 두 bit가 동시에 올라가야 하며, combo POS 평균이 낮으면 multi-label recall이 약해진다.
- OOD NEG와 Normal/Invalid는 모든 bit가 낮아야 한다. 특정 bit 평균이 올라가면 해당 bit tail이 생긴 것이다.
- train=400 val_margin은 combo POS는 강하지만 OOD NEG의 `bb` tail이 커진다. train=200 val_margin은 그 tail이 낮아 대표 조건으로 적합하다.

## 5. Class Probability Diagnostic

아래 표는 모든 train/class 조건과 두 checkpoint 선택 기준(`val_f1`, `val_margin`)을 동일하게 보여준다. POS는 single+combo, NEG는 Normal/Invalid/OOD이다.

### train=50 val_f1, eval=20000

| class | GT | bb | fk | sc | sr | metric |
|---|---|---:|---:|---:|---:|---|
| bank_boundary | 1000 | 0.862 | 0.140 | 0.106 | 0.209 | bit_F1=1.000 |
| fork | 100 | 0.188 | 0.839 | 0.094 | 0.144 | bit_F1=0.985 |
| scratch | 10 | 0.215 | 0.103 | 0.798 | 0.155 | bit_F1=1.000 |
| scratch_rot | 1 | 0.184 | 0.135 | 0.120 | 0.883 | bit_F1=1.000 |
| bank_boundary+fork | 1100 | 0.831 | 0.264 | 0.100 | 0.184 | bit_F1=0.503 |
| bank_boundary+scratch | 1010 | 0.839 | 0.144 | 0.145 | 0.173 | bit_F1=0.496 |
| bank_boundary+scratch_rot | 1001 | 0.745 | 0.118 | 0.120 | 0.510 | bit_F1=0.635 |
| fork+scratch | 110 | 0.535 | 0.519 | 0.242 | 0.116 | bit_F1=0.263 |
| fork+scratch_rot | 101 | 0.338 | 0.489 | 0.098 | 0.680 | bit_F1=0.654 |
| scratch+scratch_rot | 11 | 0.179 | 0.103 | 0.338 | 0.820 | bit_F1=0.652 |
| Normal | 0 | 0.127 | 0.520 | 0.361 | 0.195 | FAR=0.222 |
| Invalid | 0 | 0.285 | 0.486 | 0.340 | 0.427 | FAR=0.000 |
| DiagonalSmear | 0 | 0.126 | 0.348 | 0.289 | 0.196 | FAR=0.008 |
| CenterDonut | 0 | 0.132 | 0.343 | 0.292 | 0.200 | FAR=0.005 |
| CrossScratch | 0 | 0.139 | 0.348 | 0.292 | 0.201 | FAR=0.005 |
| Starburst | 0 | 0.136 | 0.333 | 0.300 | 0.203 | FAR=0.011 |

### train=50 val_margin, eval=20000

| class | GT | bb | fk | sc | sr | metric |
|---|---|---:|---:|---:|---:|---|
| bank_boundary | 1000 | 0.856 | 0.152 | 0.148 | 0.144 | bit_F1=1.000 |
| fork | 100 | 0.153 | 0.846 | 0.167 | 0.142 | bit_F1=0.998 |
| scratch | 10 | 0.163 | 0.123 | 0.849 | 0.145 | bit_F1=1.000 |
| scratch_rot | 1 | 0.141 | 0.137 | 0.155 | 0.868 | bit_F1=1.000 |
| bank_boundary+fork | 1100 | 0.743 | 0.565 | 0.110 | 0.114 | bit_F1=0.904 |
| bank_boundary+scratch | 1010 | 0.811 | 0.099 | 0.489 | 0.109 | bit_F1=0.583 |
| bank_boundary+scratch_rot | 1001 | 0.773 | 0.105 | 0.093 | 0.787 | bit_F1=0.998 |
| fork+scratch | 110 | 0.182 | 0.604 | 0.690 | 0.112 | bit_F1=0.895 |
| fork+scratch_rot | 101 | 0.162 | 0.735 | 0.074 | 0.656 | bit_F1=0.991 |
| scratch+scratch_rot | 11 | 0.167 | 0.102 | 0.510 | 0.739 | bit_F1=0.567 |
| Normal | 0 | 0.166 | 0.384 | 0.600 | 0.152 | FAR=0.099 |
| Invalid | 0 | 0.227 | 0.473 | 0.514 | 0.351 | FAR=0.000 |
| DiagonalSmear | 0 | 0.164 | 0.336 | 0.556 | 0.175 | FAR=0.029 |
| CenterDonut | 0 | 0.174 | 0.335 | 0.554 | 0.177 | FAR=0.025 |
| CrossScratch | 0 | 0.179 | 0.338 | 0.554 | 0.178 | FAR=0.030 |
| Starburst | 0 | 0.181 | 0.329 | 0.560 | 0.180 | FAR=0.037 |

### train=100 val_f1, eval=20000

| class | GT | bb | fk | sc | sr | metric |
|---|---|---:|---:|---:|---:|---|
| bank_boundary | 1000 | 0.845 | 0.124 | 0.175 | 0.128 | bit_F1=1.000 |
| fork | 100 | 0.159 | 0.839 | 0.135 | 0.119 | bit_F1=0.993 |
| scratch | 10 | 0.117 | 0.116 | 0.865 | 0.117 | bit_F1=1.000 |
| scratch_rot | 1 | 0.192 | 0.152 | 0.120 | 0.865 | bit_F1=1.000 |
| bank_boundary+fork | 1100 | 0.606 | 0.510 | 0.163 | 0.106 | bit_F1=0.663 |
| bank_boundary+scratch | 1010 | 0.741 | 0.149 | 0.366 | 0.097 | bit_F1=0.974 |
| bank_boundary+scratch_rot | 1001 | 0.839 | 0.112 | 0.123 | 0.591 | bit_F1=0.954 |
| fork+scratch | 110 | 0.165 | 0.694 | 0.412 | 0.107 | bit_F1=0.978 |
| fork+scratch_rot | 101 | 0.196 | 0.635 | 0.102 | 0.562 | bit_F1=0.865 |
| scratch+scratch_rot | 11 | 0.411 | 0.175 | 0.265 | 0.799 | bit_F1=0.911 |
| Normal | 0 | 0.247 | 0.220 | 0.389 | 0.074 | FAR=0.308 |
| Invalid | 0 | 0.292 | 0.372 | 0.449 | 0.385 | FAR=0.000 |
| DiagonalSmear | 0 | 0.437 | 0.174 | 0.309 | 0.113 | FAR=0.608 |
| CenterDonut | 0 | 0.441 | 0.176 | 0.310 | 0.114 | FAR=0.615 |
| CrossScratch | 0 | 0.450 | 0.180 | 0.308 | 0.112 | FAR=0.649 |
| Starburst | 0 | 0.441 | 0.183 | 0.321 | 0.117 | FAR=0.557 |

### train=100 val_margin, eval=20000

| class | GT | bb | fk | sc | sr | metric |
|---|---|---:|---:|---:|---:|---|
| bank_boundary | 1000 | 0.851 | 0.149 | 0.144 | 0.147 | bit_F1=1.000 |
| fork | 100 | 0.150 | 0.852 | 0.153 | 0.150 | bit_F1=0.995 |
| scratch | 10 | 0.143 | 0.145 | 0.852 | 0.153 | bit_F1=1.000 |
| scratch_rot | 1 | 0.146 | 0.148 | 0.149 | 0.862 | bit_F1=1.000 |
| bank_boundary+fork | 1100 | 0.764 | 0.698 | 0.092 | 0.124 | bit_F1=0.992 |
| bank_boundary+scratch | 1010 | 0.763 | 0.098 | 0.651 | 0.112 | bit_F1=0.935 |
| bank_boundary+scratch_rot | 1001 | 0.808 | 0.116 | 0.098 | 0.734 | bit_F1=0.998 |
| fork+scratch | 110 | 0.126 | 0.696 | 0.734 | 0.105 | bit_F1=0.989 |
| fork+scratch_rot | 101 | 0.129 | 0.761 | 0.101 | 0.778 | bit_F1=0.993 |
| scratch+scratch_rot | 11 | 0.126 | 0.078 | 0.696 | 0.772 | bit_F1=0.979 |
| Normal | 0 | 0.298 | 0.216 | 0.515 | 0.179 | FAR=0.001 |
| Invalid | 0 | 0.224 | 0.364 | 0.451 | 0.467 | FAR=0.000 |
| DiagonalSmear | 0 | 0.362 | 0.149 | 0.463 | 0.139 | FAR=0.027 |
| CenterDonut | 0 | 0.374 | 0.151 | 0.459 | 0.140 | FAR=0.032 |
| CrossScratch | 0 | 0.379 | 0.154 | 0.457 | 0.138 | FAR=0.044 |
| Starburst | 0 | 0.385 | 0.153 | 0.463 | 0.143 | FAR=0.052 |

### train=200 val_f1, eval=20000

| class | GT | bb | fk | sc | sr | metric |
|---|---|---:|---:|---:|---:|---|
| bank_boundary | 1000 | 0.843 | 0.155 | 0.162 | 0.146 | bit_F1=1.000 |
| fork | 100 | 0.140 | 0.831 | 0.143 | 0.149 | bit_F1=1.000 |
| scratch | 10 | 0.137 | 0.138 | 0.879 | 0.129 | bit_F1=1.000 |
| scratch_rot | 1 | 0.143 | 0.132 | 0.159 | 0.854 | bit_F1=1.000 |
| bank_boundary+fork | 1100 | 0.782 | 0.634 | 0.115 | 0.115 | bit_F1=0.981 |
| bank_boundary+scratch | 1010 | 0.747 | 0.095 | 0.615 | 0.107 | bit_F1=0.942 |
| bank_boundary+scratch_rot | 1001 | 0.765 | 0.111 | 0.130 | 0.709 | bit_F1=0.992 |
| fork+scratch | 110 | 0.107 | 0.650 | 0.717 | 0.087 | bit_F1=0.982 |
| fork+scratch_rot | 101 | 0.117 | 0.733 | 0.093 | 0.705 | bit_F1=0.992 |
| scratch+scratch_rot | 11 | 0.117 | 0.087 | 0.741 | 0.736 | bit_F1=0.997 |
| Normal | 0 | 0.346 | 0.297 | 0.407 | 0.299 | FAR=0.000 |
| Invalid | 0 | 0.296 | 0.415 | 0.375 | 0.467 | FAR=0.000 |
| DiagonalSmear | 0 | 0.413 | 0.283 | 0.326 | 0.291 | FAR=0.105 |
| CenterDonut | 0 | 0.446 | 0.281 | 0.321 | 0.284 | FAR=0.170 |
| CrossScratch | 0 | 0.472 | 0.280 | 0.316 | 0.274 | FAR=0.231 |
| Starburst | 0 | 0.472 | 0.277 | 0.307 | 0.289 | FAR=0.209 |

### train=200 val_margin, eval=20000

| class | GT | bb | fk | sc | sr | metric |
|---|---|---:|---:|---:|---:|---|
| bank_boundary | 1000 | 0.852 | 0.150 | 0.147 | 0.148 | bit_F1=1.000 |
| fork | 100 | 0.149 | 0.853 | 0.149 | 0.147 | bit_F1=1.000 |
| scratch | 10 | 0.147 | 0.147 | 0.853 | 0.148 | bit_F1=1.000 |
| scratch_rot | 1 | 0.146 | 0.146 | 0.145 | 0.855 | bit_F1=1.000 |
| bank_boundary+fork | 1100 | 0.791 | 0.702 | 0.115 | 0.123 | bit_F1=0.989 |
| bank_boundary+scratch | 1010 | 0.772 | 0.102 | 0.674 | 0.111 | bit_F1=0.968 |
| bank_boundary+scratch_rot | 1001 | 0.806 | 0.126 | 0.117 | 0.762 | bit_F1=0.998 |
| fork+scratch | 110 | 0.144 | 0.737 | 0.773 | 0.104 | bit_F1=0.988 |
| fork+scratch_rot | 101 | 0.143 | 0.793 | 0.101 | 0.749 | bit_F1=0.993 |
| scratch+scratch_rot | 11 | 0.136 | 0.106 | 0.775 | 0.733 | bit_F1=1.000 |
| Normal | 0 | 0.298 | 0.246 | 0.431 | 0.343 | FAR=0.000 |
| Invalid | 0 | 0.276 | 0.375 | 0.400 | 0.472 | FAR=0.000 |
| DiagonalSmear | 0 | 0.273 | 0.225 | 0.404 | 0.331 | FAR=0.001 |
| CenterDonut | 0 | 0.280 | 0.226 | 0.403 | 0.331 | FAR=0.002 |
| CrossScratch | 0 | 0.289 | 0.228 | 0.403 | 0.326 | FAR=0.007 |
| Starburst | 0 | 0.281 | 0.228 | 0.398 | 0.341 | FAR=0.007 |

### train=400 val_f1, eval=20000

| class | GT | bb | fk | sc | sr | metric |
|---|---|---:|---:|---:|---:|---|
| bank_boundary | 1000 | 0.859 | 0.150 | 0.156 | 0.149 | bit_F1=1.000 |
| fork | 100 | 0.149 | 0.898 | 0.133 | 0.150 | bit_F1=1.000 |
| scratch | 10 | 0.142 | 0.195 | 0.869 | 0.161 | bit_F1=1.000 |
| scratch_rot | 1 | 0.142 | 0.164 | 0.150 | 0.848 | bit_F1=1.000 |
| bank_boundary+fork | 1100 | 0.719 | 0.687 | 0.110 | 0.117 | bit_F1=0.984 |
| bank_boundary+scratch | 1010 | 0.801 | 0.117 | 0.482 | 0.111 | bit_F1=0.791 |
| bank_boundary+scratch_rot | 1001 | 0.692 | 0.104 | 0.130 | 0.717 | bit_F1=0.972 |
| fork+scratch | 110 | 0.320 | 0.688 | 0.664 | 0.088 | bit_F1=0.923 |
| fork+scratch_rot | 101 | 0.185 | 0.648 | 0.084 | 0.731 | bit_F1=0.976 |
| scratch+scratch_rot | 11 | 0.148 | 0.108 | 0.633 | 0.726 | bit_F1=0.938 |
| Normal | 0 | 0.104 | 0.305 | 0.380 | 0.341 | FAR=0.000 |
| Invalid | 0 | 0.159 | 0.367 | 0.399 | 0.552 | FAR=0.107 |
| DiagonalSmear | 0 | 0.156 | 0.275 | 0.401 | 0.323 | FAR=0.004 |
| CenterDonut | 0 | 0.169 | 0.274 | 0.401 | 0.324 | FAR=0.011 |
| CrossScratch | 0 | 0.172 | 0.275 | 0.400 | 0.323 | FAR=0.015 |
| Starburst | 0 | 0.183 | 0.272 | 0.403 | 0.327 | FAR=0.016 |

### train=400 val_margin, eval=20000

| class | GT | bb | fk | sc | sr | metric |
|---|---|---:|---:|---:|---:|---|
| bank_boundary | 1000 | 0.849 | 0.139 | 0.151 | 0.154 | bit_F1=1.000 |
| fork | 100 | 0.143 | 0.856 | 0.154 | 0.148 | bit_F1=1.000 |
| scratch | 10 | 0.143 | 0.147 | 0.859 | 0.147 | bit_F1=1.000 |
| scratch_rot | 1 | 0.149 | 0.145 | 0.152 | 0.855 | bit_F1=1.000 |
| bank_boundary+fork | 1100 | 0.765 | 0.746 | 0.110 | 0.132 | bit_F1=0.977 |
| bank_boundary+scratch | 1010 | 0.772 | 0.081 | 0.687 | 0.103 | bit_F1=0.985 |
| bank_boundary+scratch_rot | 1001 | 0.758 | 0.106 | 0.113 | 0.714 | bit_F1=0.979 |
| fork+scratch | 110 | 0.154 | 0.682 | 0.719 | 0.087 | bit_F1=0.983 |
| fork+scratch_rot | 101 | 0.115 | 0.720 | 0.090 | 0.736 | bit_F1=0.992 |
| scratch+scratch_rot | 11 | 0.119 | 0.080 | 0.689 | 0.722 | bit_F1=0.999 |
| Normal | 0 | 0.245 | 0.262 | 0.383 | 0.364 | FAR=0.000 |
| Invalid | 0 | 0.282 | 0.331 | 0.375 | 0.469 | FAR=0.010 |
| DiagonalSmear | 0 | 0.355 | 0.244 | 0.356 | 0.311 | FAR=0.052 |
| CenterDonut | 0 | 0.388 | 0.236 | 0.346 | 0.309 | FAR=0.090 |
| CrossScratch | 0 | 0.402 | 0.234 | 0.340 | 0.306 | FAR=0.116 |
| Starburst | 0 | 0.413 | 0.237 | 0.342 | 0.303 | FAR=0.118 |

## 6. NB Reject Sidecar

NB reject는 model 자체를 다시 학습하지 않고, 이미 나온 4-bit probability vector 위에 별도 reject rule을 붙이는 sidecar다.

사용한 완료 case:

| 항목 | 값 |
|---|---|
| recipe family | `samplecap_T7_LS02950_g3_grid9_cmp10000_p05000_ab090_100_mpos065_s7_ep10_tr200` |
| calibration preds | `E:/data/images/chip_multilabel_v15direct_n2000` 평가 preds |
| eval preds | `E:/data/images/eval_n20000` 평가 preds |
| NB model | GaussianNB on defect probability vectors |
| selected threshold | `pos-q=0.0001` |
| tau | `-40.935299` |

### NB 결과

| mode | reject-empty bit_F1 | accepted-only bit_F1 | false reject POS | false accept NEG | pos coverage | neg coverage |
|---|---:|---:|---:|---:|---:|---:|
| raw model | 0.9962 | 0.9962 | 0 | small nonzero | 1.0000 | 1.0000 |
| NB reject, pos-q=0.0001 | 0.9954 | 0.9965 | 282 | 0 | 0.9982 | 0.0000 |

해석:

- raw model은 모든 sample에 대해 그대로 예측한다.
- NB reject는 probability vector가 defect class distribution과 충분히 닮은 경우만 accept한다.
- rejected sample은 empty prediction으로 바꾼다.
- accepted-only 기준 bit_F1은 `0.9962 -> 0.9965`로 소폭 상승한다.
- false accept NEG가 `0`이 되어 negative leakage를 제거한다.
- 대신 POS 282개가 reject되므로, reject-empty bit_F1은 `0.9954`로 약간 낮아진다.

즉 NB reject는 classifier 자체를 더 정확하게 만든다기보다, ambiguous probability vector를 운영상 reject로 보내서 accepted region의 신뢰도를 높이는 장치다.

## 7. NB Reject의 실제 의미: max-prob가 아니라 4-bit pattern likelihood

NB reject를 붙이는 이유는 raw classifier의 bit_F1을 크게 올리기 위해서가 아니다. 핵심은 **어떤 OOD sample에서 특정 bit probability가 높아도, 그 4개 확률의 전체 모양은 single 또는 2-combo defect와 다르다**는 점을 이용하는 것이다.

예를 들어 `train=400 val_margin`에서 OOD인 `CrossScratch`, `Starburst`는 `bank_boundary(bb)` probability가 `0.40` 근처까지 올라온다. 단순 max-prob 또는 threshold만 보면 "bb가 꽤 높다"라고 볼 수 있다. 하지만 실제 `bank_boundary` single 또는 `bank_boundary+X` 2-combo와 비교하면 4-bit vector 모양이 다르다.

![NB reject pattern](E:/data/images/chip_multilabel_reports/manager_260603/nb_reject_pattern_bank_boundary_ood.png)

| pattern | GT | bb | fk | sc | sr | likelihood interpretation |
|---|---|---:|---:|---:|---:|---|
| bank_boundary | 1000 | 0.849 | 0.139 | 0.151 | 0.154 | single bb: bb high, other bits low |
| bank_boundary+fork | 1100 | 0.765 | 0.746 | 0.110 | 0.132 | combo: bb and fk both high |
| bank_boundary+scratch | 1010 | 0.772 | 0.081 | 0.687 | 0.103 | combo: bb and sc both high |
| bank_boundary+scratch_rot | 1001 | 0.758 | 0.106 | 0.113 | 0.714 | combo: bb and sr both high |
| CrossScratch (OOD) | 0000 | 0.402 | 0.234 | 0.340 | 0.306 | OOD: bb is moderately high, but no valid single/combo pattern |
| Starburst (OOD) | 0000 | 0.413 | 0.237 | 0.342 | 0.303 | OOD: bb is moderately high, but no valid single/combo pattern |

즉 OOD의 `bb=0.40`은 단독으로 보면 커 보이지만:

- `bank_boundary` single이라면 `bb≈0.85`이고 `fk/sc/sr≈0.14~0.15`여야 한다.
- `bank_boundary+fork` combo라면 `bb≈0.76`, `fk≈0.75`가 같이 높아야 한다.
- `bank_boundary+scratch` combo라면 `bb≈0.77`, `sc≈0.69`가 같이 높아야 한다.
- `bank_boundary+scratch_rot` combo라면 `bb≈0.76`, `sr≈0.71`이 같이 높아야 한다.
- OOD는 `bb≈0.40`이 떠도 `fk/sc/sr`가 애매하게 중간값이고, 어떤 single/2-combo의 전형적인 모양에도 맞지 않는다.

GaussianNB는 이 차이를 다음처럼 본다.

```text
p = [p_bb, p_fk, p_sc, p_sr]

For each defect class c:
  L_c(p) = log P(p | c)
         = sum_j log Normal(p_j ; mu_{c,j}, sigma^2_{c,j})

score(p) = max_c L_c(p)

accept if score(p) >= tau
reject otherwise
```

여기서 중요한 점은 `max(p)`만 보는 것이 아니라 `[p_bb, p_fk, p_sc, p_sr]` 전체를 class별 Gaussian 분포에 넣는다는 것이다. 그래서 `bb` 하나가 높아도 나머지 bit 조합이 class distribution과 안 맞으면 likelihood가 낮아진다.

정리하면:

- max-prob threshold: "`bb`가 높으니 positive일 수 있다"까지만 본다.
- NB reject: "`bb`가 높긴 한데, 이 4-bit vector가 `bank_boundary` single 또는 `bank_boundary+X` combo처럼 생겼는가?"를 본다.
- 따라서 NB reject는 **bit threshold 보정**이 아니라 **probability-shape reject**다.

주의:

- NB reject는 raw model 학습 성능을 대체하지 않는다.
- raw 성능표와 NB-reject 성능표는 분리해서 제시해야 한다.
- 운영상 의미는 "불확실한 OOD/ambiguous vector를 reject하고 accepted region의 신뢰도를 높이는 것"이다.

## 8. 후속 실험

train/class 400을 살리려면 현재 조건을 그대로 유지하기보다 positive expansion을 줄이는 방향이 맞다.

| priority | change | reason |
|---:|---|---|
| 1 | `cmp=1.0 -> 0.7 or 0.8` | synthetic combo target을 완화해 OOD tail 상승 억제 |
| 2 | `cutmix_p=0.5 -> 0.25 or 0.35` | FCMPM 노출 비율을 낮춰 over-expansion 완화 |
| 3 | `A/B label=0.90/1.00 -> 0.85/1.00 or 0.90/0.95` | 강한 source와 약한 source의 target balance 조정 |
| 4 | checkpoint criterion에 OOD negative tail penalty 추가 | `val_margin`만으로는 OOD tail을 직접 보지 못함 |

## 9. 관리자용 요약

이번 실험에서 확인한 내용은 다음이다.

- `val_margin` checkpoint selection은 `val_f1` selection보다 probability separation이 좋다.
- 대표 조건은 `train=200 / val_margin / eval=20000`이다.
- train=400은 POS bit_F1은 높지만 OOD 쪽 `bb` tail이 올라가므로 현재 조건에서는 대표 조건으로 두지 않는다.
- NB reject는 ambiguous probability vector를 reject해 accepted region의 품질을 높인다.
- NB reject는 raw model 성능과 별도로 운영용 safety layer로 보고해야 한다.

최종 보고 기준:

| 목적 | 조건 |
|---|---|
| raw model 대표 성능 | `train=200 / val_margin / eval=20000` |
| 운영 safety layer | GaussianNB reject sidecar |
| 다음 개선 방향 | train=400에서 `cmp`, `cutmix_p`, A/B label 완화 |
