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

![FCM-PM probability bars](E:/data/images/chip_multilabel_reports/manager_260603/fcm_pm_prob_bars_tr200_vs_tr400.png)

### Group Mean Probability

| condition | group | bb | fk | sc | sr |
|---|---|---:|---:|---:|---:|
| train=200 val_margin | single POS | 0.324 | 0.325 | 0.324 | 0.325 |
| train=200 val_margin | combo POS | 0.466 | 0.426 | 0.424 | 0.427 |
| train=200 val_margin | OOD NEG | 0.281 | 0.227 | 0.402 | 0.332 |
| train=200 val_margin | Normal/Invalid | 0.287 | 0.310 | 0.416 | 0.408 |
| train=400 val_margin | single POS | 0.323 | 0.323 | 0.327 | 0.326 |
| train=400 val_margin | combo POS | 0.449 | 0.405 | 0.400 | 0.416 |
| train=400 val_margin | OOD NEG | 0.389 | 0.238 | 0.345 | 0.305 |
| train=400 val_margin | Normal/Invalid | 0.264 | 0.297 | 0.379 | 0.416 |

읽는 법:

- single POS는 자기 bit 하나만 강하고 나머지는 낮은 구조다.
- combo POS는 두 bit가 동시에 올라가며, 이 값이 충분히 높아야 combo recall이 산다.
- OOD NEG는 원래 모든 bit가 낮아야 한다.
- train=400에서는 OOD NEG의 `bb` 평균이 `0.281 -> 0.389`로 상승한다.
- 이 때문에 train=400은 bit_F1은 높아도 OOD 쪽 negative tail이 커진다.

## 5. Class Probability Diagnostic

### train=200, val_margin, eval=20000

| class | GT | bb | fk | sc | sr | note |
|---|---|---:|---:|---:|---:|---|
| bank_boundary | 1000 | 0.852 | 0.150 | 0.147 | 0.148 | clean single positive |
| fork | 0100 | 0.149 | 0.853 | 0.149 | 0.147 | clean single positive |
| scratch | 0010 | 0.147 | 0.147 | 0.853 | 0.148 | clean single positive |
| scratch_rot | 0001 | 0.146 | 0.146 | 0.145 | 0.855 | clean single positive |
| bank_boundary+fork | 1100 | 0.791 | 0.702 | 0.115 | 0.123 | strong combo |
| bank_boundary+scratch | 1010 | 0.772 | 0.102 | 0.674 | 0.111 | weakest combo but usable |
| bank_boundary+scratch_rot | 1001 | 0.806 | 0.126 | 0.117 | 0.762 | strong combo |
| fork+scratch | 0110 | 0.144 | 0.737 | 0.773 | 0.104 | strong combo |
| fork+scratch_rot | 0101 | 0.143 | 0.793 | 0.101 | 0.749 | strong combo |
| scratch+scratch_rot | 0011 | 0.136 | 0.106 | 0.775 | 0.733 | strong combo |
| Normal | 0000 | 0.298 | 0.246 | 0.431 | 0.343 | negative tail controlled |
| Invalid | 0000 | 0.276 | 0.375 | 0.400 | 0.472 | negative tail controlled |
| DiagonalSmear | 0000 | 0.273 | 0.225 | 0.404 | 0.331 | OOD tail controlled |
| CenterDonut | 0000 | 0.280 | 0.226 | 0.403 | 0.331 | OOD tail controlled |
| CrossScratch | 0000 | 0.289 | 0.228 | 0.403 | 0.326 | OOD tail controlled |
| Starburst | 0000 | 0.281 | 0.228 | 0.398 | 0.341 | OOD tail controlled |

### train=400, val_margin, eval=20000

| class | GT | bb | fk | sc | sr | note |
|---|---|---:|---:|---:|---:|---|
| bank_boundary | 1000 | 0.849 | 0.139 | 0.151 | 0.154 | clean single positive |
| fork | 0100 | 0.143 | 0.856 | 0.154 | 0.148 | clean single positive |
| scratch | 0010 | 0.143 | 0.147 | 0.859 | 0.147 | clean single positive |
| scratch_rot | 0001 | 0.149 | 0.145 | 0.152 | 0.855 | clean single positive |
| bank_boundary+fork | 1100 | 0.765 | 0.746 | 0.110 | 0.132 | strong combo |
| bank_boundary+scratch | 1010 | 0.772 | 0.081 | 0.687 | 0.103 | combo improved vs 200 |
| bank_boundary+scratch_rot | 1001 | 0.758 | 0.106 | 0.113 | 0.714 | usable combo |
| fork+scratch | 0110 | 0.154 | 0.682 | 0.719 | 0.087 | usable combo |
| fork+scratch_rot | 0101 | 0.115 | 0.720 | 0.090 | 0.736 | strong combo |
| scratch+scratch_rot | 0011 | 0.119 | 0.080 | 0.689 | 0.722 | strong combo |
| Normal | 0000 | 0.245 | 0.262 | 0.383 | 0.364 | negative tail mostly controlled |
| Invalid | 0000 | 0.282 | 0.331 | 0.375 | 0.469 | sr tail near boundary |
| DiagonalSmear | 0000 | 0.355 | 0.244 | 0.356 | 0.311 | OOD bb/sc tail elevated |
| CenterDonut | 0000 | 0.388 | 0.236 | 0.346 | 0.309 | OOD bb tail elevated |
| CrossScratch | 0000 | 0.402 | 0.234 | 0.340 | 0.306 | OOD bb tail elevated |
| Starburst | 0000 | 0.413 | 0.237 | 0.342 | 0.303 | OOD bb tail elevated |

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

## 7. 왜 NB Reject가 올라가는가

FCM-PM의 4-bit probability vector는 class별 모양이 다르다.

| group | probability shape |
|---|---|
| single POS | 하나의 bit만 높고 나머지 bit는 낮음 |
| combo POS | 정답 두 bit가 동시에 높고 나머지 bit는 낮음 |
| Normal/Invalid | 여러 bit가 중간값 근처에 머물며 뚜렷한 positive pattern이 약함 |
| OOD NEG | 특정 bit tail이 올라갈 수 있지만, defect class의 full pattern과는 다름 |

GaussianNB는 이 4-bit vector를 보고 "defect class probability pattern과 얼마나 닮았는지"를 log-likelihood로 계산한다. 그래서 단순 threshold에서 애매하게 positive로 보이던 OOD/Normal tail을 reject할 수 있다.

수식적으로는 다음과 같다.

```text
p = [p_bb, p_fk, p_sc, p_sr]
L_c(p) = log P(p | class=c)
score(p) = max_c L_c(p)
accept if score(p) >= tau
reject otherwise
```

이 구조의 장점:

- single과 combo는 class-conditional probability pattern이 뚜렷하다.
- OOD는 일부 bit가 올라가도 전체 4-bit vector가 defect class centroid와 다르다.
- 따라서 NB reject는 단일 max-prob threshold보다 "pattern-level reject"에 가깝다.

주의:

- NB reject는 본 모델의 학습 성능을 대체하지 않는다.
- 운영 단계에서 ambiguous case를 reject하는 selective prediction 장치다.
- 논문/보고서에서는 raw 성능과 NB-reject 성능을 분리해서 제시해야 한다.

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
