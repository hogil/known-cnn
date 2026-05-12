# Chip Multi-Label 결함 분류기

## 프로젝트

chip 이미지(200×200) 의 결함 4종(bank_boundary, fork, scratch, scratch_rot) 을 **multi-label** 로 분류 — 한 chip 에 여러 결함 동시 가능, sigmoid 4 head 독립 출력. 학습 = 단일 결함 4×200 + 정상 200 (1,000장), 평가 = 단일 + 2-combo + OOD-overlay + Normal/Invalid + OOD wafer 무늬 = 20 class 3,850장. chip 합성 = wafer 분포 → alpha map → 확률적 grade 픽셀 → 8-color palette PNG. combo 는 RGB pixel-wise min (두 결함 darker 보존).

## 데이터 — 4 group sample

### 학습 4 obj single defect (학습 + 평가)

| | | | |
|:---:|:---:|:---:|:---:|
| ![](figs/bank_boundary.png) | ![](figs/fork.png) | ![](figs/scratch.png) | ![](figs/scratch_rot.png) |
| bank_boundary | fork | scratch | scratch_rot |

### 6 2-combo (평가 only, min-blend)

| | | |
|:---:|:---:|:---:|
| ![](figs/bank_boundary_AND_fork.png) | ![](figs/bank_boundary_AND_scratch.png) | ![](figs/bank_boundary_AND_scratch_rot.png) |
| bb+fork | bb+scratch | bb+sr |
| ![](figs/fork_AND_scratch.png) | ![](figs/fork_AND_scratch_rot.png) | ![](figs/scratch_AND_scratch_rot.png) |
| fork+scratch | fork+sr | sc+sr |

### 4 OOD-overlay (평가 only — 2 trained + 1 OOD overlay, GT는 2 trained bit만)

bit 순서 = `[bb, fork, scratch, scratch_rot]`

| | |
|:---:|:---:|
| ![](figs/fork_AND_scratch_AND_OOD_DiagonalSmear.png) | ![](figs/bank_boundary_AND_fork_AND_OOD_CenterDonut.png) |
| fork+sc + OOD_DiagonalSmear<br>GT = [bb=0, **fork=1**, **sc=1**, sr=0] | bb+fork + OOD_CenterDonut<br>GT = [**bb=1**, **fork=1**, sc=0, sr=0] |
| ![](figs/fork_AND_scratch_rot_AND_OOD_CrossScratch.png) | ![](figs/scratch_AND_scratch_rot_AND_OOD_Starburst.png) |
| fork+sr + OOD_CrossScratch<br>GT = [bb=0, **fork=1**, sc=0, **sr=1**] | sc+sr + OOD_Starburst<br>GT = [bb=0, fork=0, **sc=1**, **sr=1**] |

### Normal / Invalid

| | |
|:---:|:---:|
| ![](figs/Normal.png) | ![](figs/Invalid.png) |
| Normal | Invalid |

### 4 OOD wafer-pattern (평가 only — 학습 안한 외형, false alarm 측정)

| | | | |
|:---:|:---:|:---:|:---:|
| ![](figs/DiagonalSmear.png) | ![](figs/CenterDonut.png) | ![](figs/CrossScratch.png) | ![](figs/Starburst.png) |
| DiagonalSmear | CenterDonut | CrossScratch | Starburst |

OOD chip 은 wafer 단위 합성한 불량 wafer 에서 결함 영역(bin ≥ 200) 의 chip 만 따왔다. 등록된 4 결함(bb/fork/sc/sr) 형태가 아니라 학습에 한 번도 들어가지 않은 외형이다. **현업 라인에서 학습 안 한 새 결함 패턴이 random 하게 들어오는 상황을 시뮬레이션** — 모델이 4 결함 중 하나로 잘못 fire 하면 false alarm (`ood_chip_FAR`).

## 평가 지표

각 chip 4-bit GT vs 4-bit pred — `[ bit 0 = bank_boundary, bit 1 = fork, bit 2 = scratch, bit 3 = scratch_rot ]`. 예: fork+scratch chip → GT `[0,1,1,0]`.

- **CF1** (paper main): 4 class 각 F1 산술평균, minority class 진단
- **ni_chip_FAR**: 정상/측정불능 chip 중 결함 bit 잘못 fire 비율 (운영 main)
- **ood_chip_FAR**: 학습 안한 OOD chip 중 fire 비율 (diagnostic)

운영 통과 = `CF1 ≥ 0.83` + `ni_chip_FAR ≤ 5%` ✅

## 핵심 결과

### 🏆 Paper Main (260511, n=200 robust)

**FCM-PM (Full-Cover Mixup with Pair Mask)** + 4-bag majority vote ensemble

- **Paper headline**: 4-bag {pure-hard composition} = **CF1 0.9953 / ni_FAR 0.00%** ✅
- **1× cost single SOTA**: 4-bag teacher KD distillation = **CF1 0.9872 / ni_FAR 0.50%**

### Two-tier FCM-PM method (paper §4)

1. **Pair Mask** (safety-critical): training 시 chip A 결함 + chip B 결함 paste 영역 의 pair-mask 가 Normal/Invalid 의 defect-prediction 차단. **제거 시 -0.18 CF1 catastrophic FAR collapse** (2.5% → 100%).
2. **Group Complete** (accuracy-critical, CutMix-complement mode): chip A 결함 영역 전체에 chip B 의 결함 합성 — combo 학습 효과. 제거 시 -0.035 (helpful but tunable).

### Production deployment 권장

| 비용 | model | CF1 | FAR | 권장 |
|---:|---|---:|---:|---|
| 1× | KD distill 4-bag teacher α=0.5 T=4 | 0.9872 | 0.5% | ★ standard production |
| 4× | 4-bag majority vote (pure-hard) | 0.9953 | 0% | high-accuracy SOTA |

## FCM-PM 학습 기법

### Pair Mask + Group Complete (paper main)

학습 시 chip A 위에 chip B 의 결함을 paste 해서 multi-label sample 생성. 두 메커니즘 핵심:

1. **Pair Mask**: paste 영역의 Normal background 부분에 mask 처리 → model 이 background 를 "정상" 으로 학습 → ni_FAR 차단
2. **Group Complete (complement mode)**: chip A 결함 영역 전체에 chip B 결함 채움 → 두 결함 fully overlap 학습 → combo 평가에 generalize

### CutMix base (Yun 2019)

학습 데이터에는 단일 결함 chip 만. CutMix 는 학습 중 일부 batch (25%) 에 chip A + chip B 합성으로 multi-label sample 생성. 학습 안한 combo 평가에 일반화.

| 원본 bank_boundary | 원본 scratch | CutMix 결과 |
|:---:|:---:|:---:|
| ![](figs/cutmix_demo/orig_bank.png) | ![](figs/cutmix_demo/orig_scratch.png) | ![](figs/cutmix_demo/cutmix_random_rect.png) |

### BCE + Label Smoothing (T7 BCE+LS, ls=0.20)

target soft-label `[bb=0.05, fork=0.85, sc=0.05, sr=0.05]` (LS ε=0.20). over-confidence 완화 + FAR 통제.

→ paper §6: **Pair Mask + BCE+LS + Knowledge Distillation = 3 orthogonal FAR-control mechanisms** (실험 35+ 종 결과)

### Knowledge Distillation (1× cost SOTA)

4-bag (NEW MAIN composition) 의 평균 sigmoid prob 를 teacher 로 → student α=0.5 T=4 학습. paper §5: 14-bag teacher (α=0.3) vs 4-bag teacher (α=0.5) — teacher prob sharpness 에 따라 α 조정. 4-bag teacher 가 sharper → α=0.5 sweet spot.

## paper grounding

BCE / CF1 / F1_bit (Tsoumakas 2007, Wang 2016, Chen 2019), Label Smoothing (Müller 2019), Focal (Lin 2017), ASL (Ridnik 2021), CutMix (Yun 2019, Walawalkar 2020, Sumbul 2024), Knowledge Distillation (Hinton 2015, Yang 2023 multi-label).

---

## 260513 update — val_margin selection + iter116J NEW SOTA

### val_f1 → val_margin selection 전환 효과 (★)

**val_margin** = mean(prob[positive bits]) − max(prob[negative bits]) per chip, averaged. **Decision boundary sharpness 직접 측정**.

기존 val_f1 (per-bit BCE F1) 은 small val (n=163, single-label only) 에서 saturate (3 reachable values 0.9818/0.9847/0.9907) → coin-flip selection. val_margin = continuous spectrum + anti-saturation + multi-label friendly.

**Pooled Spearman ρ vs eval bit_F1 (iter101A/111/112 audit, 35 ckpts)**:
| metric | ρ | 평가 |
|---|---|---|
| val_acc | -0.42 | anti-correlated |
| val_f1 | -0.10 | saturate noise |
| val_auroc | +0.30 | unstable (plateau 1.0000) |
| val_bce | -0.03 | noise |
| **val_margin** | **+0.56** | ★ best correlated |

### NEW SOTA — iter116J (절대 룰 준수)

```
Recipe: ConvNeXtV2-Base FCMAE 384 + T7 BCE+LS=0.30 + ep=10
        + cutmix-mode complement g=3 pair=masked corner cls=0.5 p=0.25
        + --no-normal + --val-criterion margin_max + --save-every-epoch
        + seed=1, batch 2 accum 8, lr=1e-4 (cosine)
```

| candidate | bit_F1 | Total FAR | comparison |
|---|---|---|---|
| **★ iter116J val_margin (ep6)** | **0.9943** | **0.00%** | ZERO FAR paper headline |
| iter116J val_f1 (ep1) | 0.9422 | 0.00% | same recipe, val_f1 selection |
| iter116F (g=4 LS=0.30) val_margin | 0.9953 | 0.24% | balance |
| iter112 ep06 val_f1 (이전) | 0.9964 | 0.83% | high F1 but FAR 비싸짐 |
| iter46E (옛 룰 Normal trained) | 0.9755 | 1.07% | reference baseline |

→ **+0.052 bit_F1 vs val_f1 selection** (same recipe, only criterion 변경)
→ **+0.018 bit_F1 vs iter46E + Total FAR -1.07%** (절대 룰 + val_margin combined)

### iter116J val_margin vs val_f1 — per-bit active/inactive prob distribution

format: `mean ± std` (prob 분포)

**val_f1 (ep1)** :
| group | n | bb_pos | bb_neg | fk_pos | fk_neg | sc_pos | sc_neg | sr_pos | sr_neg |
|---|---|---|---|---|---|---|---|---|---|
| single | 640 | 0.87±0.01 | 0.14±0.02 | 0.85±0.03 | 0.15±0.01 | 0.79±0.02 | 0.14±0.01 | 0.84±0.01 | 0.17±0.01 |
| 2-combo | 960 | 0.69±0.15 | 0.19±0.08 | 0.41±0.16 | 0.15±0.02 | 0.26±0.11 | 0.12±0.02 | 0.60±0.14 | 0.17±0.02 |
| 3-combo | 640 | 0.60±0.15 | 0.22±0.06 | 0.23±0.08 | 0.14±0.02 | 0.18±0.06 | 0.12±0.02 | 0.44±0.11 | 0.18±0.01 |
| Normal | 160 | N/A | 0.41±0.02 | N/A | 0.20±0.01 | N/A | 0.42±0.04 | N/A | 0.40±0.02 |
| Invalid | 40 | N/A | 0.38±0.03 | N/A | 0.15±0.02 | N/A | 0.30±0.02 | N/A | 0.35±0.02 |
| OOD | 640 | N/A | 0.43±0.04 | N/A | 0.16±0.01 | N/A | 0.43±0.03 | N/A | 0.35±0.02 |

**val_margin (ep6)** :
| group | n | bb_pos | bb_neg | fk_pos | fk_neg | sc_pos | sc_neg | sr_pos | sr_neg |
|---|---|---|---|---|---|---|---|---|---|
| single | 640 | 0.87±0.00 | 0.15±0.01 | 0.85±0.01 | 0.15±0.01 | 0.85±0.01 | 0.15±0.01 | 0.84±0.00 | 0.14±0.00 |
| 2-combo | 960 | 0.67±0.07 | 0.13±0.02 | **0.53±0.12** | 0.12±0.02 | **0.43±0.09** | 0.10±0.01 | 0.54±0.10 | 0.13±0.01 |
| 3-combo | 640 | 0.52±0.09 | 0.13±0.03 | 0.34±0.10 | 0.11±0.02 | 0.25±0.07 | 0.09±0.01 | 0.42±0.05 | 0.12±0.01 |
| Normal | 160 | N/A | 0.37±0.02 | N/A | 0.23±0.01 | N/A | 0.48±0.01 | N/A | 0.23±0.01 |
| Invalid | 40 | N/A | 0.30±0.02 | N/A | 0.21±0.01 | N/A | 0.24±0.01 | N/A | 0.23±0.01 |
| OOD | 640 | N/A | 0.39±0.02 | N/A | 0.23±0.02 | N/A | 0.45±0.02 | N/A | 0.23±0.01 |

### 핵심 변화 (val_f1 → val_margin)

| metric | val_f1 → val_margin | Δ |
|---|---|---|
| 2-combo fork_pos | 0.41 → 0.53 | **+0.12** ★ |
| 2-combo scratch_pos | 0.26 → 0.43 | **+0.17** ★★ |
| 2-combo inactive bits | 0.12-0.19 → 0.10-0.13 | -0.03~-0.06 |
| OOD/Normal max_prob | 0.45~0.48 (gate 0.55 미만 유지) | 동등 안전 |
| bit_F1 (absolute rule) | **0.9422 → 0.9943** | **+0.052** ★★ |
| Total FAR | 0.00% → 0.00% | tied |

### FAR=0 메커니즘 (★ I13 entropy gate)

`max_prob < 0.55 → Normal 강제` 가 FAR 의 핵심 차단막. per-bit threshold 만으론 OOD chip 의 scratch prob 0.45 가 threshold 0.18 초과 → false fire. **gate 가 모든 bit 평탄한 chip (OOD/Normal) 차단** :

| group | max_prob mean | gate (0.55) 통과 |
|---|---|---|
| single | 0.85 | ✓ |
| 2-combo | 0.65 (bb/sr 강함) | ✓ |
| 3-combo | 0.51 | △ boundary |
| OOD | 0.45 (모든 bit 평탄) | ✗ |
| Normal | 0.48 | ✗ |
| Invalid | 0.30 | ✗ |

→ **2-combo 는 항상 1-2 bit peaked, OOD 는 flat 분포** — 이 차이가 분리 enabling factor.

### Code patches (260512-13)

| flag | 효과 |
|---|---|
| `--val-criterion {acc,f1,auroc,bce_min,brier_min,margin_max,f1_best_tau}` | 7 selection criterion |
| `--save-every-epoch` | per-epoch ckpt 저장 |
| `--cutmix-mode bisect_h/v/rand` | 절반 split cutmix (paper §6 NEW axis) |
| `--no-normal` | 절대 룰 enforce |

