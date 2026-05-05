# Intra-distribution per-class breakdown

V3 obj-only 의 진짜 weak point 측정. distribution-level 정확도와 각 distribution
안에서 obj 식별 정확도 분리.

## 측정 방법

1. wafer class name → (distribution, object) parse:
   - `Donut_scratch_21deg` → (`Donut`, `scratch_21deg`)
   - `Starburst` → (`Starburst`, None)
2. distribution accuracy = predicted distribution == GT distribution
3. object accuracy = predicted object == GT object (GT 가 obj 있는 sample 만)
4. **per-distribution obj_acc** = 그 distribution 안에서 obj 식별 정확도

산출 스크립트: `_intra_dist_eval.py`. 결과: `results_intra_dist/summary.json`.

## 핵심 표 — per-distribution object accuracy

| Distribution | R-only ConvNeXt 88M | obj-only 4-layer 0.4M | Tier1 ens α=0.35 | V3 obj-only 1.16M |
|---|---|---|---|---|
| Center | 1.0000 | 0.9954 | 1.0000 | 1.0000 |
| Donut | 1.0000 | 1.0000 | 1.0000 | 0.9954 |
| **Edge-Bottom** | 0.9630 | 0.9537 | 0.9583 | **0.9907** ★ |
| Edge-Ring | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| **Edge-Top** | 0.9630 | 0.9722 | 0.9722 | **0.9954** ★ |
| Full | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| Normal | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| Thick-Edge | 1.0000 | 1.0000 | 1.0000 | 1.0000 |

★ = V3 의 best 이지만 weak point. Edge-Bottom + Edge-Top 의 chip 6개 안에서
obj 식별이 V3 의 진짜 ceiling.

## 해석

### Distribution-level
- 모든 모델이 distribution 분류 (8-class) 100% 또는 거의 100% — wafer pattern
  자체는 쉬운 task.
- intra-distribution obj 식별이 진짜 어려운 부분 — 같은 distribution 안에서
  5 chip object 어느 것인지 결정.

### Why Edge-Bottom / Edge-Top 어려운지
- `_sample_gen.py` 의 `DEFECT_BUDGET`: Edge-Top, Edge-Bottom 등 spatial 한정
  class → defect chip 6개만 (전체 1024 중)
- 6 chip 안에서 obj 종류 구분 → 통계량 너무 작음
- 다른 distribution (Center, Full, Donut) 은 defect chip 더 많아 obj 분포 명확

### V3 가 Edge-Bottom 0.9907 로 best 인 이유
- 32×32 native + one-hot 5ch → chip CNN 의 forced-choice obj_id 손상 없이 입력
- R-only 의 raw RGB 보다 chip CNN 의 distilled obj label 이 6 chip 안에서
  더 discriminative
- but 0.9907 = 9/10 매번 100% 가 아닌 한계 — 합성 spec 의 의도된 ambiguity 때문

## V3 의 약점이 안 풀리는 이유

1. **Mid-fusion 으로도 못 잡음** — R 도 같은 6 chip RGB 보고 같은 confusion
2. **Cross-attention 도 같음** — 두 stream 의 정보가 본질적으로 같음
3. **MoE 도 같음** — Edge-Bottom expert 도 6 chip 만 본다
4. **Knowledge Distillation 도 teacher 의 한계 그대로**

→ Tier 3-6 모두 expected gain 거의 0. V3 가 oracle 0.9919 도 추월.

## 합성 spec 변경 후보 (priority 낮음)

1. `_sample_gen.py:739` `DEFECT_BUDGET` ↑ (Edge-Bottom 6 → 12 chip) — but
   spec 변경 큼, WM-811K 분포 학습 결과와 충돌
2. obj-only 학습 시 inter-class margin loss 추가 — 사용자 진행 중
   (`cnn_train_chipgrid_fusion.py` + `_chipgrid_kde_gmm.py` Phase 3)
3. 데이터 증가 (220 → 500/class) — diminishing returns

## Other weak class (낮은 F1)

V3 baseline (n=220, seed 42) 의 per-class F1 < 1.0 (val split):

| class | F1 | Sup | 메모 |
|---|---|---|---|
| Edge-Bottom_bank_boundary | 0.800 | 11 | weak (특히) |
| Edge-Bottom_invalid_main | 0.919 | 17 | |
| Edge-Bottom_scratch | 0.778 | 9 | weak (특히) |
| Edge-Bottom_scratch_21deg | 0.909 | 5 | small sup |
| Edge-Top_bank_boundary | 0.933 | 7 | |
| Edge-Top_invalid_main | 0.966 | 15 | |
| Edge-Top_particle_blast | 0.909 | 12 | |
| Edge-Top_scratch | 0.800 | 6 | weak (작은 sup) |
| Edge-Top_scratch_21deg | 0.929 | 13 | |

→ active_classes_20.yaml 에 Edge-Bottom × 5, Edge-Top × 5 모두 보존된 이유.

## 산출

- `_intra_dist_eval.py` — distribution / object 분리 측정 entry
- `results_intra_dist/summary.json` — JSON output (위 표 + per-dist breakdown)
- `_v3_fair_eval.py` — V3 ckpt 를 우리 0.8/0.2 val 1420 위 fair eval
- `results_v3_eval/v3_logits.npy` — V3 inference logits on our val
- `results_ensemble_ep10/r_logits.npy`, `obj_logits.npy`, `val_y.npy` — ep10 fair compare base

## Cross-link

- V3 best result 종합 → `RESULTS.md`
- 7 핵심 발견 (Edge weak point = #6) → `DISCOVERY.md`
- archive 결정 (Center/Full/등 100% saturated 라 archive) → `ACTIVE_CLASSES.md`
