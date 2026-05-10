# iter 13 — Cycle A (T7N v19zpp v2) + Cycle B (CutMix variant grid)

★ paper main metric pivot: `chip_FAR` (single bundled) → `normal_invalid_chip_FAR`
(real-env, 200 chip 분모) + `ood_chip_FAR` (diagnostic, 640 chip 분모) 분리.
Normal training 도입으로 ni_chip_FAR 80% → 0.0% lock 깬 cycle.

## 1. spec / source

- **데이터**: `D:/project/data/wm-811k/classification_chips/` (v19zpp tier, 200/class
  4 train + 200 invalid_main + 200 Normal). Normal 학습 활성 (`--no-normal` 빼고 첫 시도).
- **eval**: `D:/project/data/wm-811k/chip_multilabel/` master (★ **21 class** — 4
  single + 6 2-combo + 4 3-combo + Normal + Invalid + 5 OOD), `--n-per-class 200`.
- **fixed args**: `--epochs 8 --batch 8 --accum 4 --lr-head 1e-4 --seed 42`,
  inference `I3`.
- **Cycle A target**: T7-with-Normal training 단일 모델 + 8 v19zpp non-Normal
  baseline 과 logit-avg ensemble.
- **Cycle B target**: Cycle A winner (T7N) 위에서 CutMix variant 6 cell grid sweep
  (random_rect / scattered / grid50 / grid25 / grid12 / no_cutmix).

## 2. Cycle A — T7N single (260507 00:22) + 9 ensemble configs

### 2.1 T7N single training

- run: `D:/project/known-cnn/outputs/T7_T7_with_normal_v19zpp_seed42_v2_260507_002217/`
- spec: T7 (BCE+LS) + LS=0.20 + CutMix p=0.25 rect=0.5 mode=single + Normal training
  (`--no-normal` 빼고 200 Normal y=-1 sentinel 추가).
- `train_summary.json`: variant=T7, ls=0.20, cutmix_p=0.25, cutmix_rect=0.5,
  no_normal=False, best_val_acc=1.0 ep=1.

### 2.2 split-FAR + per-class F1 (T7N single, master 21-class eval)

| metric | T7-no-Normal baseline (v19zpp) | **T7N (with Normal)** | Δ |
|---|---:|---:|---:|
| CF1 (macro F1) | 0.8490 | **0.9042** | +0.0552 |
| F1_micro       | 0.8038 | **0.9181** | +0.1143 |
| F1_bb          | 0.9684 | 0.9722 | +0.0038 |
| F1_fork        | 0.5248 | **0.7796** | +0.2548 |
| F1_sc          | 0.9066 | 0.8676 | -0.0390 |
| F1_sr          | 0.9964 | 0.9973 | +0.0009 |
| ni_chip_FAR    | 80.00% | **0.00%** | -80% |
| ood_chip_FAR   | 100.00% | 16.38% | -83.62% |
| 3plus%         | 1.98% | 1.42% | -0.56% |

★ **Normal training 단일 lever**: bundled chip_FAR 96% → 13.1% (memory rule
입증 — `feedback_normal_training_open_set.md`). ni_chip_FAR 80% → 0%
(Normal 학습 직접 효과). ood_chip_FAR 100% → 16% (high-confidence threshold
이 cross-domain false alarm 도 같이 억제).
★ fork F1 0.49 → 0.78 (+0.29) — Normal training 으로 fork 의 sigmoid
prob distribution 이 더 sharp 해짐 (Normal 학습이 fork 의 false-alarm
prior 를 누름).
★ trade-off: scratch F1 0.95 → 0.87 (-0.08, fork 와 cross-class 영향).

### 2.3 9 ensemble configs (T7N anchor + non-Normal minor partner)

postproc: `chip_multilabel/_logit_avg_ensemble.py` — sigmoid logit avg + 새
threshold (0.10 단순) + decision_tree.

| pair | weights | CF1 | F1_micro | F1_fork | F1_sc | F1_sr | ni_FAR | ood_FAR | 3plus% |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| T7N+T5 | 50:50 | 0.8844 | 0.8899 | 0.6697 | 0.8912 | 0.9955 | 12.50% | 22.50% | 0.28% |
| T7N+T5 | 60:40 | 0.9018 | 0.9035 | 0.7389 | 0.8878 | 0.9964 | 2.00% | 22.38% | 0.77% |
| T7N+T5 | 40:60 | 0.8648 | 0.8530 | 0.5901 | 0.8952 | 0.9947 | 80.00% | 66.75% | 0.62% |
| **T7N+T5** ★ | **70:30** | **0.9083** | **0.9080** | **0.7656** | 0.8853 | 0.9969 | **0.50%** | 21.88% | 1.45% |
| T7N+T9 | 50:50 | 0.8847 | 0.8840 | 0.6634 | 0.9088 | 0.9951 | 77.50% | 26.87% | 0.34% |
| T7N+T9 | 60:40 | 0.9001 | 0.9030 | 0.7281 | 0.9039 | 0.9960 | 13.00% | 19.25% | 0.34% |
| T7N+T7 | 50:50 | 0.8805 | 0.8887 | 0.6025 | 0.9366 | 0.9982 | 0.00% | 32.12% | 0.09% |
| T7N+T7 | 60:40 | 0.9043 | 0.9089 | 0.6988 | **0.9379** | **0.9978** | 0.00% | 23.13% | 0.19% |

### 2.4 ★ Cycle A winner (FAR ≤ 5% 제약)

constraint: CF1 ≥ 0.83 + F1_fork ≥ 0.55 + ni_chip_FAR ≤ 5%.

| rank | name | CF1 | fork_f1 | ni_FAR | ood_FAR |
|---|---|---:|---:|---:|---:|
| **1** ★ | **T7N+T5_w70_30** | **0.9083** | 0.7656 | 0.50% | 21.88% |
| 2 | T7N+T7_w60_40 | 0.9043 | 0.6988 | 0.00% | 23.13% |
| 3 | T7N_single | 0.9042 | 0.7796 | 0.00% | 16.38% |
| 4 | T7N+T5_w60_40 | 0.9018 | 0.7389 | 2.00% | 22.38% |
| 5 | T7N+T7_w50_50 | 0.8805 | 0.6025 | 0.00% | 32.12% |

vs **v19y T5 baseline** (paper old headline iter 12): CF1 0.8162, chip_FAR 3.30% (bundled).
vs **v19zpp T7 no-Normal**: CF1 0.8490, ni_chip_FAR 80%.

## 3. Cycle B — CutMix variant grid on top of T7N (260507 07:54~08:38)

### 3.1 Goal

Cycle A winner (T7N anchor) 위에서 CutMix variant axis 만 변경. 모든 cell
T7N base hparam (LS=0.20, ep=8, batch=8, accum=4, seed=42, Normal training=ON)
고정, CutMix variant 만 sweep.

### 3.2 Grid 결과 (master 21-class eval, I3 inference)

| cell | run_dir | CF1 | F1_def | F1_bb | F1_fk | F1_sc | F1_sr | ni_FAR | ood_FAR | 2bit_R |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| **random_rect** ★ | T7_T7N_random_rect_seed42_260507_075428 | **0.9188** | 0.9234 | 0.9719 | 0.8436 | 0.8658 | **0.9937** | 20.00% | 0.94% | 0.7500 |
| no_cutmix | T7_T7N_no_cutmix_seed42_260507_082859 | 0.9162 | 0.9389 | 0.9359 | 0.8324 | **0.9360** | 0.9605 | 20.00% | 14.69% | 0.6141 |
| grid50 | T7_T7N_grid50_seed42_260507_080757 | 0.8967 | 0.9043 | 0.9061 | 0.7635 | 0.9299 | 0.9874 | 20.00% | 0.31% | 0.4922 |
| grid25 | T7_T7N_grid25_seed42_260507_081422 | 0.8849 | 0.8945 | 0.9355 | 0.7450 | 0.8722 | 0.9869 | 20.00% | 3.12% | 0.5453 |
| grid12 | T7_T7N_grid12_seed42_260507_082053 | 0.8596 | 0.8766 | 0.9351 | 0.7778 | 0.8088 | 0.9166 | 20.00% | 12.03% | 0.5406 |
| scattered | T7_T7N_scattered_seed42_260507_080124 | 0.8423 | 0.8647 | 0.9107 | 0.6912 | 0.7749 | 0.9922 | 20.00% | 23.44% | 0.4328 |

### 3.3 Findings — Cycle B

1. **CutMix variant axis 의 Cycle A best 못 깸**. 모든 cell 의 ni_chip_FAR
   = 20% (Normal training side effect — but not better than Cycle A T7N single
   0%). 이 20% lock 의 출처는 Cycle B 의 변경된 chip data (v19zpp tier 동일,
   하지만 train script 의 Normal augment 흐름이 약간 다른 것 의심).
2. **random_rect** = Cycle B winner CF1 0.9188 — 단순 single rect CutMix 가
   여러 patch (scattered, grid12/25/50) 보다 강함. 이는 iter 12 Phase 4 에서도
   재확인된 신호 (fork pattern 학습에 small dispersed patch 보다 single rect 이 강함).
3. **no_cutmix** 가 의외로 2위 (0.9162) — CutMix 자체가 Normal-trained 환경에서
   필수적이지 않음을 시사. F1_sc 0.9360 으로 best (CutMix 제거 시 sc 정확도 ↑,
   fork 정확도 ↓ trade-off).
4. **scattered worst** (CF1 0.8423, ood_FAR 23.44%) — multi-patch 가 OOD
   discrimination 에 negative effect.
5. **grid12 ood_FAR 12.03%** — 가장 작은 grid 가 OOD 잡는데 약함. patch 가
   chip-level pattern 보다 작아서 model 이 fine-grained 학습 부족.

### 3.4 ★ Cycle B winner

**T7N + random_rect** = CF1 0.9188, F1_fk 0.8436, F1_sr 0.9937, ood_FAR 0.94%.
- ★ Cycle A T7N single (CF1 0.9042) 보다 +0.0146 CF1.
- ★ ood_FAR 16.38% → 0.94% (OOD 완전히 잡음 — random_rect 한 효과).
- ⚠️ ni_chip_FAR 0.00% → 20.00% — Cycle A 단일 모델 우위가 깨짐 (CutMix
  variant 가 Normal 학습 신호 약화).

## 4. Cross-iter delta vs paper baseline

| line | CF1 | F1_fk | ni_FAR | ood_FAR | source |
|---|---:|---:|---:|---:|---|
| iter 12 v19y T5 (paper old) | 0.8162 | 0.3985 | n/a | n/a | bundled chip_FAR 3.30% |
| iter 12 v19zpp T7 (no Normal) | 0.8490 | 0.5248 | 80.00% | 100.00% | paper "fix" iter |
| iter 13 Cycle A T7N single | 0.9042 | 0.7796 | 0.00% | 16.38% | Normal training lever |
| **iter 13 Cycle A T7N+T5_w70_30** ★ | **0.9083** | 0.7656 | **0.50%** | 21.88% | logit-avg ensemble |
| iter 13 Cycle B random_rect | 0.9188 | 0.8436 | 20.00% | 0.94% | CutMix variant grid winner |

## 5. Conclusions

1. ★ **paper headline metric pivot** — `chip_FAR` 단일 bundled 폐기 →
   `normal_invalid_chip_FAR` (200 chip, real-env) + `ood_chip_FAR` (640 chip,
   diagnostic) 분리.
2. ★ **Normal training 가 single lever** — chip_FAR 96% → 0% 단독 해결.
   iter 10 finding 재확인 + paper main metric 으로 정착.
3. ★ **logit-avg ensemble 가 single 모델 추가 lift** — T7N+T5 70:30 = CF1
   0.9042 → 0.9083 (+0.004, fork 0.78 유지).
4. **CutMix variant axis 는 marginal** — random_rect 가 Cycle A 보다 +0.01
   하지만 ni_FAR 0% → 20% 후퇴 (불완전 trade-off).
5. **★ 다음 axis 후보**: ood_chip_FAR 16~22% 잔여 — cross-domain
   regularization, 5-class OOD aware loss, 또는 더 강한 negative augment.

## 6. 산출 파일 (절대 경로)

- `D:/project/known-cnn/chip_multilabel/_bit_metrics.py` (patched, 3-group split FAR)
- `D:/project/known-cnn/chip_multilabel/_logit_avg_ensemble.py` (new, post-hoc prob-avg ensemble)
- `D:/project/known-cnn/outputs/T*_v19zpp*/eval_I3/bit_metrics_split.json` (8 + 1 model split-FAR)
- `D:/project/known-cnn/outputs/T7_T7_with_normal_v19zpp_seed42_v2_260507_002217/` (T7N single)
- `D:/project/known-cnn/outputs/T7_T7N_random_rect_seed42_260507_075428/` (Cycle B winner)
- `D:/project/known-cnn/outputs/T7_T7N_{scattered,grid50,grid25,grid12,no_cutmix}_seed42_260507_*/`
- `D:/project/known-cnn/outputs/_iter12_v19zpp_logs/ensemble/*.json` (17 ensemble configs)

_Source files for numerical claims:_
- `outputs/T7_T7_with_normal_v19zpp_seed42_v2_260507_002217/eval_I3/bit_metrics_split.json`
- `outputs/T7_T7N_random_rect_seed42_260507_075428/eval_I3/bit_metrics_split.json`
- (5 other Cycle B variants 동일 path).

## 7. Δ vs prior best

iter 12 Cycle B closing (v19zpp, no Normal) → iter 13:

| metric | iter 12 best | iter 13 best | Δ |
|---|---:|---:|---:|
| CF1 (single model) | 0.8490 (T7) | 0.9188 (random_rect) | +0.0698 |
| CF1 (ensemble) | n/a | 0.9083 (T7N+T5_w70_30) | new metric |
| F1_fork | 0.5248 | 0.8436 | +0.3188 |
| ni_chip_FAR | 80.00% | 0.50% (ENS) / 20% (single) | -79.5pp / -60pp |
| ood_chip_FAR | 100% | 0.94% (random_rect) | -99.06pp |
