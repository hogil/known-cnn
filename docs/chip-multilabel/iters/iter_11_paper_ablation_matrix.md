# Iter 11 — Paper-style 4-row Ablation Matrix + p30 + Normal Diversity

**Date**: 2026-05-06
**Phases**: 1 (p50 simple), 2 (p30 simple), 3 (p50 diverse Normal) — 3 phases × 6 trains × 6 inferences = 108 cells
**Headline**: 4-class only training (no Normal) catastrophic on Normal handling. iter 10 ensemble (0.995) remains unbeaten by any single train×inference.

## 0. Motivation

After iter 10 reached 0.995 via baseline+C_44 ensemble, user requested paper-style 4-row ablation:
1. 전통 single-chip CNN train + multi sigmoid pred (Row 1)
2. Pred 불량 판정 방식 다양 (Row 2)
3. Loss 변경 (ASL/Focal etc) + multi sigmoid (Row 3)
4. Loss × pred full matrix (Row 4)

Plus distribution-shift verification:
- p50 (top 50% strong defect, current) → p30 (top 70%, weaker defects included, harder eval)
- Normal diversity (current too uniform → noise/grey/sprinkle/gradient variation)

## 1. Setup

### Train side (6 variants, all 4-class only — no Normal training)

| Variant | Loss | Setting |
|---|---|---|
| T1 | CE softmax + LS=0.1 | traditional baseline |
| T3 | Focal γ=2 | hard-negative focus |
| T4 | ASL γ_pos=1, γ_neg=4, clip=0.05 | asymmetric multi-hot |
| T5 | BCE multi-hot only | raw multi-label |
| T6 | BCE 5ep warmup → ASL | two-phase hybrid |
| T7 | BCE multi-hot + LS=0.1 | iter 8 family |

Common: ep=8, batch=8 accum=4, lr=1e-4, cutmix-p=0.25, cutmix-rect=0.5, seed=42, **`--no-normal`** flag.

### Inference side (6 variants)

I3 (sigmoid + per-class F1-max threshold), I7 (joint coord-descent threshold), I10 (I7 + entropy Normal gate), I11 (I7 + bb+sr pair rescue), I12 (I7 + sc+sr pair rescue), I13 (I7 + max-prob Normal gate).

### Eval side
- 3 phases × 36 cells = 108 measurements
- Master: defect 200 + Normal 200 + Invalid 50 = 2450 chip
- Runtime: --n-per-class 50, sample-seed=42 → 600 chip per inference (val 120 / eval 480)

## 2. Phase 1 (p50, simple Normal) — Full 6×6 matrix

format: `10-def F1 / sc+sr F1 / Normal F1 / FAR%`

| Train | I3 | I7 | I10 | I11 | I12 | I13 |
|---|---|---|---|---|---|---|
| T1 | 0.528/.96/.00/100 | 0.545/.92/.00/100 | 0.545/.92/.00/100 | **0.577**/.92/.00/100 | 0.545/.92/.00/100 | 0.545/.92/.00/100 |
| T3 | 0.484/.90/**1.00**/0 | 0.509/.89/.97/5 | 0.509/.89/.97/5 | **0.513**/.89/.97/5 | 0.507/.78/.97/5 | 0.509/.89/.97/5 |
| T4 | 0.738/.00/.00/100 | 0.779/.14/.00/100 | **0.803**/.14/.86/18 | 0.748/.14/.00/100 | 0.655/.06/.00/100 | 0.779/.14/.00/100 |
| T5 | 0.799/.99/.00/100 | 0.804/1.00/.00/100 | 0.804/1.00/.00/100 | **0.806**/1.00/.00/100 | 0.804/1.00/.00/100 | 0.804/1.00/.00/100 |
| **T6** | **0.905**/1.00/.00/100 | 0.895/.95/.00/100 | 0.864/.95/.00/100 | 0.895/.95/.00/100 | 0.794/.59/.00/100 | 0.895/.95/.00/100 |
| T7 | 0.758/.52/.00/100 | **0.851**/1.00/.00/100 | 0.851/1.00/.00/100 | 0.851/1.00/.00/100 | 0.851/1.00/.00/100 | 0.851/1.00/.00/100 |

### 4-Row tables

**Row 1** (T1+I3 traditional baseline): 10-def 0.528, Normal 0.000, FAR 100% — **disaster**
**Row 2** (T1 across inferences): max 0.577 (I11), Normal still 0
**Row 3** (loss × I3): T6=0.905 best macro but Normal=0
**Row 4** (overall best): T6+I3 = 0.905 macro, FAR=100%

## 3. Phase 2 (p30, simple Normal) — Distribution-shift robustness

p30 master = top 70% source filter (includes weaker defects, harder eval).

| Train | P1 p50 best | P2 p30 best | Δ macro |
|---|---|---|---:|
| T1 | I11 def=0.577 | I11 def=0.591 | +0.014 |
| T3 | I11 def=0.513 | I3 def=0.498 | -0.016 |
| T4 | I10 def=0.803 | I10 def=0.785 | -0.017 |
| T5 | I11 def=0.806 | I12 def=0.810 | +0.004 |
| T6 | I3 def=0.905 | I7 def=0.901 | -0.004 |
| T7 | I7 def=0.851 | I7 def=0.860 | +0.009 |

→ All Δ < 0.02 magnitude. **모든 model robust**. p30 (harder eval) 에서도 macro 거의 유지.

## 4. Phase 3 (p50, diverse Normal) — Normal diversity ablation

`_make_normal_chip` patched (260506) — 5 variation sources:
1. Wider grey ratio 5-22%
2. Per-pixel grey color noise ±15
3. White subtle noise -5~0
4. Sprinkle count 0-1.5%, 3-color mix
5. Brightness gradient 20% chance, ±8

Sanity ≥ 0.70 whiteness preserved.

| Train | Δ 10-def | Δ Normal F1 | Δ FAR |
|---|---:|---:|---:|
| T1 | +0.043 | 0 | 0 |
| T3 | -0.001 | -0.027 | +5% |
| **T4** | -0.049 | **+0.070** | **-12.5%** ✅ |
| T5 | +0.088 | 0 | 0 |
| T6 | 0 | 0 | 0 |
| T7 | -0.003 | 0 | 0 |

→ 대부분 marginal change. T4 (ASL) 만 Normal 다양화 효과 (+0.07 N, -12.5% FAR) — asymmetric loss 의 robustness.

## 5. Cross-phase summary

| Train | Best of all 3 phases | macro | Normal | FAR |
|---|---|---:|---:|---:|
| **T6 (BCE→ASL)** | P1+I3 / P3+I3 | **0.905** | 0.00 | 100% |
| T7 (BCE+LS) | P2+I7 | 0.860 | 0.00 | 100% |
| T5 (BCE) | P3+I11 | 0.894 | 0.00 | 100% |
| T4 (ASL) | P1+I10 | 0.803 | 0.86 | 18% |
| T1 (CE+LS) | P3+I11 | 0.620 | 0.00 | 100% |
| T3 (Focal) | P1+I11 | 0.513 | 0.97 | 5% |

**최고 single 4-class only**: T6+I3 (BCE→ASL) = 0.905 macro, **그러나 FAR 100%** = 운영 불가능.

## 6. Comparison with iter 10 ensemble (the FINAL winner)

| Method | 10-def macro | sc+sr | Normal | FAR | Operations? |
|---|---:|---:|---:|---:|---|
| iter 11 best single (T6+I3, p50 simple) | 0.905 | 1.000 | 0.000 | 100% | ❌ |
| iter 10 baseline T9d (with Normal data) | 0.927 | 0.769 | 0.974 | 5% | ⚠ |
| iter 10 C_44 single | 0.972 | 1.000 | 1.000 | 0% | ✅ |
| **iter 10 baseline + C_44 ensemble** ★ | **0.995** | 1.000 | 1.000 | **0%** | ✅✅ |

→ **iter 10 ensemble 이 모든 single iter 11 model 압도** by +0.09 macro.

## 7. Key insights (paper-worthy)

### 7.1 Normal training non-negotiable (사용자 directive 입증)

4-class only training (T1/T5/T6/T7) → BEST cell 에서도 Normal F1 = 0.000, FAR = 100%. Operationally unusable in real-env (Normal 80% prevalence → 800 false alarms per 1000-chip wafer).

### 7.2 Asymmetric / Focal loss → natural Normal generalization

T3 (Focal): Normal 1.00, FAR 0% — Focal 의 hard-negative focus 가 Normal 의 약한 신호 학습.
T4 (ASL): Normal 0.86, FAR 18% — ASL γ_neg=4 의 negative class confidence 억제.
**4-class only 학습 시 Asymmetric/Focal loss 가 BCE/CE 보다 Normal robust.**

### 7.3 Distribution-shift robustness 입증 (p50 → p30)

모든 6 trains × best inference = Δ < 0.02. 강한 defect (p50) 학습이 약한 defect (p30) 까지 generalize.

### 7.4 Normal diversity = T4 ASL 만 영향 받음

Normal 합성을 다양화해도 T1/T5/T6/T7 (defect-aggressive mode) 는 여전히 100% FAR. T4 ASL 만 +7% Normal F1 (asymmetric mechanism).

### 7.5 Ensemble (iter 10 H) 이 single 의 한계 깬다

어떤 single (loss × inference) cell 도 0.91 도달 못 함. baseline + C_44 logit averaging = 0.995 — paper "ensemble unique value" 입증.

## 8. Source

- Master folder: `D:/project/data/wm-811k/chip_multilabel/` (2450 chip, 3 phase 마다 in-place regen)
- Trained models: `outputs/T*_ablation_*_seed42_<TS>/` (6 dirs)
- Inference runs:
  - Phase 1 (p50 simple): `outputs/stage1_260506_092731~092930/` (6 runs)
  - Phase 2 (p30 simple): `outputs/stage1_260506_093205~093353/` (6 runs)
  - Phase 3 (p50 diverse Normal): `outputs/stage1_260506_0939*~0941*/` (6 runs)
- Code patches:
  - `_train_chip_variant.py`: `--no-normal` flag (260506 09:00)
  - `gen_eval_set.py`: `_make_normal_chip` diversified (260506 09:30)

## 9. Memory rules (carry-over from iter 10)
- `feedback_logit_ensemble_complementary.md` — H ensemble core finding
- `feedback_normal_training_open_set.md` — Normal training non-negotiable
- `feedback_cross_class_suppression.md` — fork combo prob collapse
- `feedback_master_storage_vs_runtime_sampling.md` — single SoT
- `feedback_chip_train_batch_safe.md` — shared GPU batch=8

## 10. Verdict

**Iter 11 ablation conclusion**: paper-style 4-class only ablation **confirms** that without Normal training, no single (loss × inference) configuration achieves operational-grade results. Best single = T6+I3 = 0.905 macro but FAR 100%. **iter 10 ensemble (baseline + C_44 logit avg) remains the optimal architecture** — combining a no-Normal model (preserves combo signal) with a Normal-trained model (locks Normal F1).

The paper narrative: train both models, ensemble at inference. Operational grade achieved at 0.9930 ± 0.005 across 5 sample seeds, FAR 0.0%.
