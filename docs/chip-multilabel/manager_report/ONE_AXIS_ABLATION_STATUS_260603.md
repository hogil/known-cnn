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
| 1-axis | neg target | 0.02 / 0.05 / 0.10 | queued |
| 1-axis | cutmix_p | 0.20 / 0.30 / 0.40 / 0.60 / 0.70 / 0.80 | running / queued |
| 1-axis | loss variant | T10 ASL+LS / T4 ASL / T6 BCE->ASL, A/B=1.00/1.00 fixed | queued |
| repeat | seed stability | baseline and neg=0.02/0.05 at seed 13/42/99 | queued |
| 1-axis | grid, g=3 | 3x3 / 6x6 / 12x12, baseline 9x9 | queued |
| 1-axis | group-grid alignment | g=2 grid6 / g=4 grid12, baseline g=3 grid9 | queued |
| existing evidence | cmp | 0.5 / 0.7 / 0.8 / 1.0 | mined, not rerun |
| 2-factor | top 1-axis pairs | neg/p/A-grid plus T10 loss interactions | pending |
| 3-factor | top 2-factor neighborhood | compact T10/neg/p and A/neg/p candidates | pending |

## Completed Rows

| axis | value | bit_F1 | pos | neg | gap | worst POS min | worst NEG max |
|---|---|---:|---:|---:|---:|---|---|
| abpos_Avar_B100 | A070_B100 | 0.9820 | 0.7125 | 0.2250 | 0.044 | bank_boundary+scratch/sc=0.536 | CrossScratch/bb=0.492 |
| abpos_Avar_B100 | A080_B100 | 0.9837 | 0.7169 | 0.2203 | -0.246 | bank_boundary+scratch/sc=0.257 | Starburst/sc=0.503 |
| abpos_Avar_B100 | A090_B100 | 0.9949 | 0.7706 | 0.2209 | 0.130 | bank_boundary+scratch/sc=0.652 | Invalid/sr=0.522 |
| baseline | A100_B100_neg000_p050_grid9_g3_cmp100 | 0.9959 | 0.7890 | 0.2202 | 0.140 | bank_boundary+scratch/sc=0.676 | Invalid/fk=0.536 |
| cutmix_p | p020 | 0.9865 | 0.7712 | 0.2393 | 0.098 | bank_boundary+scratch/sc=0.544 | Invalid/fk=0.446 |
| neg_target | neg002 | 0.9964 | 0.7605 | 0.2226 | 0.212 | bank_boundary+scratch/sc=0.654 | Normal/sc=0.442 |
| neg_target | neg005 | 0.9954 | 0.7753 | 0.2287 | 0.204 | scratch+scratch_rot/sc=0.656 | Normal/sc=0.452 |
| neg_target | neg010 | 0.9847 | 0.7769 | 0.2458 | 0.028 | bank_boundary+scratch/sc=0.594 | Invalid/fk=0.566 |

## Axis Best So Far

| axis | best value | bit_F1 | gap | reason |
|---|---|---:|---:|---|
| abpos_Avar_B100 | A090_B100 | 0.9949 | 0.130 | worst POS bank_boundary+scratch/sc=0.652; worst NEG Invalid/sr=0.522 |
| baseline | A100_B100_neg000_p050_grid9_g3_cmp100 | 0.9959 | 0.140 | worst POS bank_boundary+scratch/sc=0.676; worst NEG Invalid/fk=0.536 |
| cutmix_p | p020 | 0.9865 | 0.098 | worst POS bank_boundary+scratch/sc=0.544; worst NEG Invalid/fk=0.446 |
| neg_target | neg002 | 0.9964 | 0.212 | worst POS bank_boundary+scratch/sc=0.654; worst NEG Normal/sc=0.442 |

## Next Stage Rule

- 관리자 표에는 오탐률 컬럼을 넣지 않는다. 단, 내부 후보 gate에서는 `Total FAR <= 1%`를 같이 본다.
- 1축에서 `bit_F1 >= 0.993`, `Total FAR <= 1%`, `gap`이 baseline보다 개선되는 값을 후보로 둔다.
- 후보가 2개 이상이면 2축 조합을 만든다. 예: `A/B target best` x `neg target best`.
- 2축 조합에서 다시 상위 조건이 안정되면 3축 조합으로 확장한다.
- 이미 충분히 결과가 많은 `cmp` 축은 새로 반복하지 않고 기존 evidence를 사용한다.
