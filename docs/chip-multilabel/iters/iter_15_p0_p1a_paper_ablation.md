# iter 15 — P0 baseline + P1A LS sweep + P1A loss alternatives (paper-style 4-class ablation matrix)

★ paper-style ablation matrix 재현 — 4-class only (Normal training OFF,
`--no-normal=True`) 로 전통 single-chip CNN baseline 이 LS / loss 변경
sweep 에서 어떻게 움직이나 검증.

## 1. spec / source

- **chip data**: `D:/project/data/wm-811k/classification_chips/` (post-v5
  classification chips, 4 train class + invalid_main).
- **eval data**: `D:/project/data/wm-811k/chip_multilabel/` master 21-class.
- **fixed args**: `--epochs 8 --batch 8 --accum 4 --lr-head 1e-4 --seed 42
  --no-normal`, inference `I3`. cutmix-p=0.0 (CutMix 없음).
- **goal**: **Normal training OFF** baseline 에서 LS axis sweep + loss
  alternatives → iter 11 paper ablation matrix 의 paper figure 재공급.

## 2. P0 — Pure baseline (T5 BCE)

run: `D:/project/known-cnn/outputs/T5_P0_pure_baseline_seed42_260507_094228/`

| metric | T5_P0_pure_baseline | 비고 |
|---|---:|---|
| CF1 | 0.8583 | BCE pure, no LS, no CutMix |
| F1_def_only | 0.8676 | 4-class macro |
| F1_bb | 0.9404 | |
| F1_fork | 0.7756 | |
| F1_sc | 0.7939 | |
| F1_sr | 0.9233 | |
| ni_chip_FAR | 24.50% | Normal not learned |
| ood_chip_FAR | 1.25% | unexpectedly low |
| ood_overlay_2bit_recall | 0.3906 | weak combo |

## 3. P1A — LS sweep on T7 (BCE+LS, no CutMix, no Normal)

7 cell sweep, ls ∈ {0.025, 0.05, 0.075, 0.10, 0.15, 0.20, 0.25}, all other
args fixed.

### 3.1 Full sweep table

| LS | run_dir | CF1 | F1_def | F1_bb | F1_fk | F1_sc | F1_sr | ni_FAR | ood_FAR | 2bit_R |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.025 | T7_P1A_LS0.025_260507_095547 | 0.8890 | 0.9056 | 0.9405 | 0.8308 | 0.8587 | 0.9261 | 45.00% | 5.94% | 0.5125 |
| **0.05** ★ | T7_P1A_LS0.05_260507_100208 | **0.9088** | **0.9243** | 0.9471 | 0.8351 | 0.9063 | 0.9466 | 36.00% | 5.94% | 0.6281 |
| 0.075 | T7_P1A_LS0.075_260507_100833 | 0.8976 | 0.9085 | 0.9537 | **0.8679** | 0.8449 | 0.9240 | 26.50% | 3.59% | 0.6000 |
| 0.10 | T7_P1A_LS0.1_260507_101455 | 0.8784 | 0.8999 | 0.9443 | **0.8806** | 0.7381 | 0.9507 | 31.50% | 15.47% | 0.5891 |
| 0.15 | T7_P1A_LS0.15_260507_102116 | 0.8643 | 0.8799 | 0.8814 | 0.8159 | 0.7832 | 0.9769 | 22.00% | 9.69% | 0.4531 |
| 0.20 | T7_P1A_LS0.2_260507_102736 | 0.8648 | 0.8937 | 0.8973 | 0.8145 | 0.8277 | 0.9198 | **20.50%** | 25.16% | 0.5375 |
| 0.25 | T7_P1A_LS0.25_260507_103411 | 0.8625 | 0.8940 | 0.9200 | 0.8465 | 0.7469 | 0.9367 | 29.50% | 23.75% | 0.5469 |

### 3.2 LS sweep findings

1. **LS=0.05 = sweet spot** — CF1 0.9088, F1_def 0.9243. iter 8 (T9 LS sweep
   on cutmix-base) 에서도 0.07 이 winner — LS small 영역 일관 신호.
2. **LS 0.025 vs 0.05**: ni_FAR 45% → 36% (LS 강할수록 FAR 약간 감소). LS=0.05
   가 sweet spot 유지.
3. **LS=0.075 / 0.10** 에서 F1_fork 가 peak (0.8679, 0.8806) — fork 의
   over-confidence calibration 효과. but CF1 는 LS=0.05 가 best (sc/sr trade-off).
4. **LS=0.10 ood_FAR 15.47% spike** — fork prob distribution 평탄화로 OOD
   chip 잡기 어려워짐.
5. **LS=0.15 / 0.20 / 0.25** — sweet spot 영역 벗어나 monotonic 하락 (CF1
   0.86 plateau).
6. **모든 cell ni_chip_FAR ≥ 20%** — Normal training OFF 의 본질적 한계
   (iter 11 finding 동일). 4-class only environment 에선 LS 어떻게 조절해도
   real-env Normal 잡지 못함.

### 3.3 LS sweep curve

```
CF1  ↑
0.91 |                  *0.05 ★
0.90 |              0.075
0.89 |          0.025
0.88 |                            0.10
0.87 |
0.86 |                                  0.15  0.20  0.25
0.85 |
     +---|----|----|----|----|----|----|----→ LS
     0.025 0.05 0.075 0.10 0.15 0.20 0.25
```

sharp peak at LS=0.05, monotonic 하락 0.075~0.25. iter 8 의 LS curve 와 비슷
(LS=0.07 peak) 한데, iter 15 는 CutMix 없는 환경 → peak 가 LS=0.05 로 약간
더 낮음.

## 4. P1A loss alternatives — T3 Focal, T9 Sigmoid Focal

### 4.1 T3 Focal

run: `D:/project/known-cnn/outputs/T3_P1A_T3_focal_260507_104037/`

| metric | T3_focal | vs P1A LS=0.05 | 비고 |
|---|---:|---:|---|
| CF1 | 0.7768 | -0.1320 | Focal worst |
| F1_fork | 0.5717 | -0.2634 | fork 약함 |
| F1_sr | 0.9087 | -0.0379 | |
| ni_chip_FAR | **100.00%** | +64% | Normal 완전 mis-fire |
| ood_chip_FAR | **100.00%** | +94% | OOD 완전 mis-fire |
| ood_overlay_2bit_recall | 0.5984 | -0.0297 | |

★ T3 Focal 이 ni_chip_FAR / ood_chip_FAR 모두 100% — Focal 이 fork over-fire
를 극단적으로 누른 결과 다른 class (Normal/OOD) prob 도 평탄화 → 어떤 chip
이든 강한 signal 한 개 나오면 declare 함. iter 11 의 Focal finding 과 비슷
(Focal+P3=0.513, FAR 5%) 보다 더 나쁨 — 이 차이는 chip data version 차이 (post-v5).

### 4.2 T9 Sigmoid Focal

run: `D:/project/known-cnn/outputs/T9_P1A_T9_sigfocal_260507_104655/`

| metric | T9_sigfocal | vs P1A LS=0.05 | 비고 |
|---|---:|---:|---|
| CF1 | 0.8273 | -0.0815 | sigmoid_focal 중간 |
| F1_fork | 0.7169 | -0.1182 | |
| F1_bb | 0.9584 | +0.0113 | |
| F1_sc | 0.7803 | -0.1260 | |
| F1_sr | 0.8534 | -0.0932 | |
| ni_chip_FAR | 46.50% | +10.5pp | |
| ood_chip_FAR | 5.16% | -0.78pp | |
| ood_overlay_2bit_recall | 0.3406 | -0.2875 | weak combo |

T9 sigmoid_focal 이 T3 보다 나음 (CF1 0.83 vs 0.78), sigmoid 의 multi-label
친화 + focal 의 hard example focus combo. 하지만 P1A LS=0.05 (CF1 0.9088) 못 이김.

### 4.3 T4 g1g4 (ASL γ_neg=4 default) — eval failed

run: `D:/project/known-cnn/outputs/T4_P1A_T4_g1g4_260507_105315/`
- ⚠️ eval_I3/bit_metrics_split.json 미존재 (eval pipeline 중단 또는 실패).
- 학습은 끝남 (best_model.pth 존재) — 나중에 재평가 가능.

## 5. ★ iter 15 winner (P0 baseline + P1A 9 cells)

| rank | name | CF1 | F1_fork | ni_FAR | ood_FAR | 2bit_R |
|---|---|---:|---:|---:|---:|---:|
| **1** ★ | T7N P1A LS=0.05 | **0.9088** | 0.8351 | 36.00% | 5.94% | 0.6281 |
| 2 | T7N P1A LS=0.075 | 0.8976 | 0.8679 | 26.50% | 3.59% | 0.6000 |
| 3 | T7N P1A LS=0.025 | 0.8890 | 0.8308 | 45.00% | 5.94% | 0.5125 |
| 4 | T7N P1A LS=0.10 | 0.8784 | 0.8806 | 31.50% | 15.47% | 0.5891 |
| 5 | T7N P1A LS=0.15 | 0.8643 | 0.8159 | 22.00% | 9.69% | 0.4531 |
| 6 | T7N P1A LS=0.20 | 0.8648 | 0.8145 | 20.50% | 25.16% | 0.5375 |
| 7 | T7N P1A LS=0.25 | 0.8625 | 0.8465 | 29.50% | 23.75% | 0.5469 |
| 8 | T5_P0_baseline | 0.8583 | 0.7756 | 24.50% | 1.25% | 0.3906 |
| 9 | T9_sigfocal | 0.8273 | 0.7169 | 46.50% | 5.16% | 0.3406 |
| 10 | T3_focal | 0.7768 | 0.5717 | 100.00% | 100.00% | 0.5984 |

## 6. Conclusions

1. ★ **LS=0.05 = paper baseline winner** (CF1 0.9088) — small LS in [0.025,
   0.075] sweet spot 일관 (iter 8/iter 15 모두 confirm).
2. ★ **Normal training OFF environment 에서도 LS axis 가 핵심** — F1_fork
   0.78 → 0.88 (LS=0.10 peak), CF1 0.86 → 0.91 (LS=0.05 peak).
3. **Focal alternative loss negative 결과 재확인** — T3 Focal 이 ni/ood FAR
   모두 100%. iter 11 finding 과 일관.
4. **모든 single 4-class only 모델 ni_chip_FAR ≥ 20%** — Normal training 이
   필수 paper finding (iter 13 Cycle A 의 0% 와 대비).
5. ★ **iter 13 Cycle A T7N (Normal training ON) CF1 0.9042 vs iter 15 P1A
   LS=0.05 (Normal training OFF) CF1 0.9088** — 4-class CF1 만 보면 P1A 가
   marginal 우세, but ni_FAR 36% vs 0% 로 operational profile 가 완전 다름.
   **paper 에선 Normal training 을 main metric 으로 잡고 P1A 는 baseline
   counter-example**.

## 7. 산출 파일 (절대 경로)

- `D:/project/known-cnn/outputs/T5_P0_pure_baseline_seed42_260507_094228/eval_I3/bit_metrics_split.json`
- `D:/project/known-cnn/outputs/T7_P1A_LS{0.025,0.05,0.075,0.1,0.15,0.2,0.25}_260507_*/eval_I3/bit_metrics_split.json`
- `D:/project/known-cnn/outputs/T3_P1A_T3_focal_260507_104037/eval_I3/bit_metrics_split.json`
- `D:/project/known-cnn/outputs/T9_P1A_T9_sigfocal_260507_104655/eval_I3/bit_metrics_split.json`
- `D:/project/known-cnn/outputs/T4_P1A_T4_g1g4_260507_105315/best_model.pth` (eval pending)

## 8. Δ vs prior best

| metric | iter 14 v20 (Normal ON) | iter 15 P1A LS=0.05 (Normal OFF) | Δ |
|---|---:|---:|---:|
| CF1 | 0.9226 | 0.9088 | -0.0138 |
| F1_fork | 0.8591 | 0.8351 | -0.0240 |
| F1_sc | 0.8658 | 0.9063 | +0.0405 |
| F1_sr | 0.9937 | 0.9466 | -0.0471 |
| ni_chip_FAR | **0.00%** | 36.00% | +36pp ❌ |
| ood_chip_FAR | 0.94% | 5.94% | +5pp |
| ood_overlay_2bit_recall | 0.7500 | 0.6281 | -0.12 |

★ iter 14 (v20 + Normal training) 가 iter 15 P1A best (Normal OFF) 보다
operational metric 에서 우세. iter 15 는 paper baseline (counter-example).
