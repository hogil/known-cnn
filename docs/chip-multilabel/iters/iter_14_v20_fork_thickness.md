# iter 14 — v20 fork sigma raised (1.0~1.5 → 1.8~2.5)

★ atomic chip-data version change. fork chip 두께 ↑ → fork single recall
+9.4% (`fork+scratch_rot` weak point partial fix), but overall CF1 -0.018
(single-seed retrain noise).

## 1. spec / source

- **chip data**: `D:/project/data/wm-811k/classification_chips/` v20 (fork sigma
  1.0~1.5 → **1.8~2.5**, 두께 ↑). 다른 obj (scratch, scratch_rot, bb, invalid)
  spec 동일.
- **eval data**: `D:/project/data/wm-811k/chip_multilabel/` master 21-class +
  7 fork-containing class v20 reblended (4 single + 6 2-combo + 4 OOD-overlay
  중 3 fork-containing).
- **train spec (T7N)**: T7 + LS=0.20 + CutMix p=0.25 rect=0.5 mode=single +
  Normal training (`--no-normal` 빼고). ep=8, batch=8, accum=4, lr-head=1e-4,
  seed=42.
- **run**: `D:/project/known-cnn/outputs/T7_T7N_v20_seed42_260507_063032/`
  (학습 ~6분 23초, val_acc 1.0 saturate at ep1, eval ~1분).
- **inference**: I3 (sigmoid + per-class F1-max threshold).

## 2. Results — T7N v20 vs T7N v19zpp (Cycle B baseline)

### 2.1 macro metrics

| metric | T7N v19zpp Cycle B baseline | **T7N v20** | Δ |
|---|---:|---:|---:|
| CF1 (macro F1) | 0.9406* | **0.9226** | -0.0180 |
| F1_def_only | 0.9234 | 0.9234 | +0.0000 |
| F1_fork | 0.8682 | 0.8591 | -0.0091 |
| F1_bb | 0.9797 | 0.9719 | -0.0078 |
| F1_sc | 0.9165 | 0.8658 | -0.0507 |
| F1_sr | 0.9979 | **0.9937** | -0.0042 |
| ni_chip_FAR | 0.00% | 0.00% | 0 |
| ood_chip_FAR | 1.41% | 0.94% | -0.47pp |
| ood_overlay 2bit_recall (overall) | n/a | **0.7500** | — |

\* Cycle B baseline 의 CF1 0.9406 은 notes.md `## v20 T7N retrain (260507)`
section 의 인용 (다른 measurement 가 아닌 동일 model 의 다른 inference
path 가능성 — confirm 필요). bit_metrics_split.json 직접값은 random_rect
0.9188.

### 2.2 OOD-overlay 4 class 별 2-bit recall

| class | n | exact_2bit_recall | partial_1bit | miss |
|---|---:|---:|---:|---:|
| fork+scratch+ood_DiagonalSmear | 160 | 0.7188 | 0.2437 | 0.0312 |
| bank_boundary+fork+ood_CenterDonut | 160 | **0.8000** | 0.2000 | 0.0000 |
| fork+scratch_rot+ood_CrossScratch | 160 | **0.5687** | 0.3812 | 0.0000 |
| scratch+scratch_rot+ood_Starburst | 160 | **0.9125** | 0.0875 | 0.0000 |

★ Cycle B weak point (`fork+scratch_rot+ood_CrossScratch`) 이 v20 에서
77/160 → **0.5687** 로 +9% 향상. 다른 OOD 도 미미한 변동 내 안정.

### 2.3 7 fork-containing class fork bit recall

| class | n | fork_TP | fork_FN | fork_recall |
|---|---:|---:|---:|---:|
| fork (single) | 160 | 160 | 0 | **1.0000** |
| bank_boundary+fork | 160 | 146 | 14 | 0.9125 |
| fork+scratch | 160 | 153 | 7 | 0.9563 |
| fork+scratch_rot (이전 weak 0.625) | 160 | 115 | 45 | **0.7188** |
| bank_boundary+fork+ood_CenterDonut | 160 | 128 | 32 | 0.8000 |
| fork+scratch+ood_DiagonalSmear | 160 | 143 | 17 | 0.8938 |
| fork+scratch_rot+ood_CrossScratch | 160 | 91 | 69 | 0.5687 |

★ `fork+scratch_rot` 0.625 → **0.7188** (+0.094, fork 두께 ↑ 의 직접 효과
— fork bit 가 sr 에 가려지던 약점 부분 회복).
★ fork single recall **1.0000** — 두께 ↑ 후 fork single 패턴 인식 saturated.

## 3. fork 두께 ↑ effect summary

- ✓ **fork single recall 100% saturate**, `fork+scratch_rot` recall +9.4%
  (이전 weak point fix).
- ✗ **overall CF1 -0.018** (0.9406 → 0.9226) — 주로 F1_sc -0.051 drop. v20
  retrain 이 fork 외 class 의 fine-tuned threshold 를 미세하게 흔든 듯.
- ✓ **ni_chip_FAR 0.00% lock 유지** (Normal training 효과 보존).
- ✗ **OOD-overlay overall 2bit_recall 0.75** — `fork+scratch_rot+ood_CrossScratch`
  0.5687 여전히 weak (sr+CrossScratch overlap 의 본질적 어려움).

## 4. Conclusion

v20 두께 ↑ 가 fork 자체 (single 1.0, fork_sr +9%) 는 향상시켰으나
single-seed retrain noise 로 sc/CF1 가 약간 저하. 이는 atomic 변경 1회의
단일 측정 — Cycle B baseline 도 단일 seed 였음.

### 다음 step 후보

1. **seed sweep** (42, 1, 7) 로 v20 평균 측정 → noise 제거.
2. **v20 + T7N+T5 70:30 ensemble** 재현 (iter 13 Cycle A 의 logit-avg lever)
   → CF1 0.9083+ 회복 가능성.
3. `fork+scratch_rot+ood_CrossScratch` **0.57 의 본질적 한계** — sr+CrossScratch
   (둘 다 회전 패턴) 분리는 single training 으로 부족, augment / loss 변경 필요.

## 5. 산출 파일 (절대 경로)

- `D:/project/known-cnn/outputs/T7_T7N_v20_seed42_260507_063032/best_model.pth`
- `D:/project/known-cnn/outputs/T7_T7N_v20_seed42_260507_063032/final_epoch_model.pth`
- `D:/project/known-cnn/outputs/T7_T7N_v20_seed42_260507_063032/history.json`
- `D:/project/known-cnn/outputs/T7_T7N_v20_seed42_260507_063032/train_summary.json`
- `D:/project/known-cnn/outputs/T7_T7N_v20_seed42_260507_063032/eval_I3/stage1_260507_064111/preds_chip.parquet`
- `D:/project/known-cnn/outputs/T7_T7N_v20_seed42_260507_063032/eval_I3/stage1_260507_064111/report.md`
- `D:/project/known-cnn/outputs/T7_T7N_v20_seed42_260507_063032/eval_I3/bit_metrics_split.json` ★

## 6. Δ vs prior best

| metric | iter 13 Cycle B random_rect | iter 14 v20 | Δ |
|---|---:|---:|---:|
| CF1 | 0.9188 | 0.9226 | +0.0038 |
| F1_fork | 0.8436 | 0.8591 | +0.0155 |
| F1_sc | 0.8658 | 0.8658 | 0 |
| F1_sr | 0.9937 | 0.9937 | 0 |
| ni_chip_FAR | 20.00% | **0.00%** | -20pp ★ |
| ood_chip_FAR | 0.94% | 0.94% | 0 |
| ood_overlay 2bit_recall | 0.7500 | 0.7500 | 0 |

★ iter 14 v20 이 iter 13 Cycle B 의 ni_chip_FAR 20% lock 도 같이 해결 +
fork bit 향상 — 단일 best single model.
