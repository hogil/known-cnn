# Discoveries — Wafer Classifier (V3 chipgrid)

V3 발견 round 의 7 핵심 발견. 모든 수치는 fair comparison (same data,
0.8/0.1/0.1 또는 0.8/0.2 split, seed 42).

## D1. chip CNN forced-choice obj_id mask = wafer 분류의 sufficient statistic

### Mechanism
chip 5-class CNN 은 forced-choice softmax — 모든 chip 에 1 obj label 강제 부여.
Starburst / CommaCluster / Normal wafer 처럼 chip-level true obj 가 없는 wafer
도 inference 결과 obj_id 가 모든 cell 에 채워짐.

### Empirical evidence
- obj_id 32×32 spatial 분포가 wafer class 의 sufficient statistic
- 예: Donut wafer → 중앙은 "none" (id=0), edge ring 만 obj 채워짐 → spatial pattern 자체가 fingerprint

### Why it matters
- obj_id mask = wafer pattern 의 distillation (chip CNN 이 spatial filter)
- 작은 모델 (0.4M) 이 spatial pattern 분류만 하면 됨
- chip CNN 의 noise (false obj label) 도 spatial 일관성 안에서 흡수

## D2. palette idx (R) 는 chip CNN obj label 에 redundant

### Empirical
- V3 R+5obj (in_chans=6): val_f1 0.9707
- V3 obj-only (in_chans=5, R 빼고): val_f1 0.9946 (★ best)

→ **R 채널 빼는 게 오히려 더 좋음**. chip CNN 이 이미 palette grade 정보
사용해서 obj label 결정 → R 다시 보는 것이 noise.

### Why it matters
- production deploy: R 입력 안 받아도 됨 → IO 작아짐
- chip CNN forced-choice 의 distillation 효과가 raw palette 보다 더 단순/강력

## D3. chip-grid 32×32 native > BICUBIC 384

### Comparison
| 입력 | encoding | params | val_f1 | test_f1 |
|---|---|---|---|---|
| compound 384 BICUBIC | R+obj+0 (3ch) | 88M | 0.9784 | 0.9736 |
| **V3 32 native** | one-hot 5ch | **1.16M** | **0.9946** | 0.9872 |

→ V3 가 +1.62%p 절대 (val), errors 75% 감소.

### Why
- compound 의 obj_id BICUBIC 보간이 categorical 신호 깨뜨림 (1.3, 2.7 같은
  의미 없는 실수)
- V3 는 obj_id 의 자연 해상도 (32×32) 그대로 → 보간 0
- block_expand_2d 정책 (categorical preserving) 이 V3 의 enabling factor.
  자세한 건 `feedback_block_expand_only.md`.

## D4. one-hot 5ch encoding > 정수 1ch

### chipgrid V0~V4 비교 (n=100/class, seed=42)

| variant | encoding | val_f1 | test_f1 |
|---|---|---|---|
| V0 | R only | 0.4359 | 0.4385 |
| V1 | argmax/5 1ch | 0.9505 | 0.9726 |
| V2 | binary single class | 0.6543 | 0.6479 |
| **V3** | **one-hot 5ch** | **0.9689** | **0.9879** |

→ V0 → V1: +51%p (obj_id 채널이 dominant)
→ V1 → V3: +1.84%p (one-hot binary > 정수 압축)

n=220/class V3: val_f1 0.9946 (full data effect +2.56%p)

### Why
- 정수 1ch (id/5): chip CNN class 간 ordinal 가정 (1=bank_boundary <
  2=invalid_main < 3=particle_blast)이 잘못됨
- one-hot: 각 obj class 가 독립 channel → CNN 이 자연스럽게 per-class spatial
  filter 학습

## D5. 작은 model (1M) > 큰 model (88M) — 단순 task 에 over-parameterize 무의미

### Fair comparison (same data 5680 train, ep10)
| Model | params | val_f1 | 학습 |
|---|---|---|---|
| R-only ConvNeXtV2 base | 88M | 0.9851 | 12.5h |
| obj-only 4-layer | 0.4M | 0.9844 | 2 min |
| **V3 chipgrid** | **1.16M** | **0.9946** | **<1 min** |

→ V3 가 R-only ConvNeXt 보다 +0.95pp val, **76× 작은 모델로 750× 빠른 학습**.

### Why
- chip CNN 이 이미 hard problem (chip-level obj 식별) 풀어줌
- wafer 분류 = 32×32 categorical map → spatial pattern → 작은 CNN 충분
- ConvNeXt 88M 의 RGB-pixel high-frequency 능력은 wafer task 에서 over-kill

## D6. Edge row (chip 6개) 의 obj 식별 = V3 의 진짜 weak point

### per-distribution obj_acc (intra-distribution → INTRA_DIST.md)

| Distribution | R-only | obj-only | V3 |
|---|---|---|---|
| Donut | 1.0000 | 1.0000 | 0.9954 |
| **Edge-Bottom** | 0.9630 | 0.9537 | **0.9907** ★ |
| **Edge-Top** | 0.9630 | 0.9722 | **0.9954** ★ |
| (다른 5 dist) | 1.0000 | ≥0.9954 | 1.0000 |

V3 가 모든 distribution 에서 best 이지만 Edge-Bottom 0.9907 / Edge-Top 0.9954 가 진짜 ceiling.

### Root cause
`_sample_gen.py:739` 의 `DEFECT_BUDGET`:
- Edge-Top, Edge-Bottom, Edge-Ring, Donut, Center 등 spatial 한정 class →
  defect chip 6개만 (전체 1024 중)
- 6 chip 안에서 obj 종류 구분 → 통계량 너무 작음

### Why R-only / obj-only / V3 모두 같은 weak class
- 동일분포 다른오브젝트 → 6 chip 의 RGB 또는 chip CNN obj label 만 보고
  결정해야 함
- 어떤 fusion 으로도 정보 추가 X → mid-fusion / cross-attention / MoE 모두
  expected gain 0
- 합성 spec 의 의도된 결과 — 알고리즘 한계 X 데이터 한계 O

### 해결 후보 (priority 낮음)
1. DEFECT_BUDGET ↑ (Edge-Bottom 6 → 12 chip) — but spec 변경 큼
2. obj-only 학습 시 inter-class margin loss 추가 — 사용자 진행 중
   (`cnn_train_chipgrid_fusion.py` + `_chipgrid_kde_gmm.py` Phase 3 GMM hybrid)
3. ensemble 가중치 per-class adapt — Edge-Bottom subgroup 만 R-only 위주

## D7. EMA 효과 +0.55pp test (single seed, multi-seed 검증 필요)

### Empirical
V3 학습 시 EMA (decay=0.95, warmup ep 3) 적용:
- single seed (42) test_f1 +0.55pp 측정
- but single seed 분산 ±0.92pp (5 seed 평균 측정)
- → multi-seed 검증 필요. 통계적 유의성 X.

### Why mention
- chipgrid V3 의 default 로 EMA 들어가있음 (cnn_eval_chipgrid.py 의 기본값)
- 만약 새 trainer 작성 시 EMA 빼면 단일 seed 성능 약간 떨어질 수 있음
- but multi-seed 평균 차이 통계적 유의성 미확정 → "EMA 가 wafer 도메인에서
  중요하다" 결정적 evidence X

## 전체 발견의 합 (사용자 결정 요약)

| ID | 발견 | 결정 |
|---|---|---|
| D1 | chip CNN obj_id mask 가 sufficient statistic | obj-only 단독 path 정식화 |
| D2 | R (palette idx) 는 obj label 에 redundant | V3 obj-only (no R) 가 best |
| D3 | 32×32 native > BICUBIC 384 | block_expand_2d 정책 + 32 native trainer |
| D4 | one-hot 5ch > 정수 1ch | V3 default encoding |
| D5 | 작은 model > 큰 model | 1.16M ChipGridCNN 채택 |
| D6 | Edge-Bottom/Top obj 6 chip = ceiling | 합성 한계 인정, GMM hybrid 시도 |
| D7 | EMA +0.55pp single seed | default 유지, multi-seed TODO |

## 6-Tier ensemble plan deprecated (V3 결과로)

V3 단독 0.9946 > Oracle ceiling 0.9919 → 모든 fusion plan 의 expected gain 0
또는 marginal. Plan 의 미실행 Tier 3-6 = deprecated.

이유:
- Oracle ceiling = R-only + obj-only 2-stream 의 absolute upper bound. V3 가 더 높음.
- V3 의 weak point (Edge-Bottom/Top 6 chip) 은 R 도 같은 6 chip 보고 같은 confusion
- mid-fusion / cross-attention / MoE 모두 정보 추가 X

다음 phase = GMM hybrid (사용자 진행 중) + 8 새 wafer-canvas class 합성.
상세 → `STATUS.md`.

## 관련 파일

- `cnn_eval_chipgrid.py` — V0~V4 변종 sweep + V3 best (1.16M ChipGridCNN)
- `cnn_train_objonly.py` — obj-only 4-layer CNN entry
- `cnn_train_compound.py` — compound 3ch (BICUBIC 384, ceiling 0.9784)
- `_chipgrid_resize.py` — block_expand_2d (D3 enabling factor)
- `_intra_dist_eval.py` — D6 distribution / object 분리 측정
- `_v3_fair_eval.py` — V3 ckpt 를 우리 0.8/0.2 val 1420 위 fair eval
- `_sample_gen.py:739` — DEFECT_BUDGET (D6 root cause)
- `logs_chipgrid/v3_full_260503_160436_*/best_model.pth` — V3 best ckpt
- `logs_wafer/overall/best_model.pth` — R-only best
- `logs_objid_ablation/overall/best_model.pth` — obj-only best

## 관련 memory / 정책

- `feedback_block_expand_only.md` — D3 의 정책 (categorical resize)
- `feedback_active_class_policy.md` — 20 active + 14 archive (V3 결과 기반)
- `feedback_no_tta_wafer.md` — TTA 금지 (V3 도 동일)
- `project_v3_chipgrid_best.md` — V3 best 종합 memory
- `project_chipgrid_v1.md` — V0~V6 chipgrid sweep (V3 발견 전)
