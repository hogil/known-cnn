# Wafer Classifier — V3 chipgrid + GMM hybrid

이 문서는 wafer 33-class (현재 active 20) 분류기의 **V3 best result + 7 핵심 발견 + active class 정책 + GMM hybrid plan** 인덱스. 기존 6-Tier ensemble plan 은 V3 단독이 oracle ceiling 추월하면서 **deprecated**.

## 위치

| 파일 | 역할 |
|---|---|
| `docs/wafer-ensemble/README.md` | 이 파일 (인덱스 + executive summary) |
| `docs/wafer-ensemble/DISCOVERY.md` | ★ 7 핵심 발견 + 실측 데이터 + 의의 |
| `docs/wafer-ensemble/RESULTS.md` | ★ 모든 model 결과 표 (sample, epoch, val_f1, test_f1, params, 학습 시간) |
| `docs/wafer-ensemble/INTRA_DIST.md` | intra-distribution per-class breakdown (Edge-Bottom/Top weak point) |
| `docs/wafer-ensemble/ACTIVE_CLASSES.md` | 20 active + 14 archive 정책 (데이터 보존 X) |
| `docs/wafer-ensemble/STATUS.md` | 현재 진행 상태 + Phase A/B/C 계획 + GMM hybrid in-progress |
| `docs/wafer-ensemble/ENSEMBLE_TIERS.md` | (legacy) 6-Tier 이론 + 측정. Tier 3-6 deprecated. |
| `docs/wafer-ensemble/PRODUCTION_RULE.md` | (legacy) 운영 룰 — V3 deploy 전 마이그레이션 필요 |
| `docs/wafer-ensemble/PAPERS.md` | 인용 논문 |
| `.claude/skills/wafer-classifier/SKILL.md` | ★ 학습/평가 명령 패턴 + 표 정책 + block_expand 정책 + TTA 금지 |

## Executive Summary

### V3 best (FAIR comparison)

**ChipGridCNN V3 obj-only** (1.16M params, 32×32×5 one-hot obj only):
- val_f1 **0.9946**, test_f1 0.9872 (epoch 6 best, ep 13 early stop)
- our 0.8/0.2 fair eval (val 1420): macro_f1 **0.9951**, errors 7
- chip CNN noise 10% 시 val_f1 0.9870 (R+5obj 0.9707 보다 +1.6pp robust)
- **R 채널 빼도 동률** — palette idx 는 chip CNN obj label 에 redundant

비교 base:
- R-only ConvNeXt 88M ep10: val_f1 0.9851 (errors 22)
- obj-only 4-layer 0.4M ep10: val_f1 0.9844 (errors 23, best ep 8)
- Tier 1 simple α=0.35 logit ensemble: val_f1 0.9886
- **Oracle ceiling (R+obj "either correct"): val_f1 0.9919** (12 both_wrong)

→ **V3 단독 0.9946 > Oracle ceiling 0.9919**.

### 7 핵심 발견 (상세 → DISCOVERY.md)

1. chip CNN forced-choice obj_id 32×32 mask = wafer 분류의 sufficient statistic
2. palette idx (R) 는 chip CNN obj label 에 redundant
3. chip-grid 32×32 native > BICUBIC 384 (categorical noise 0)
4. one-hot 5ch encoding > 정수 1ch (V1 95.05% vs V3 99.46%)
5. 작은 model (1M) > 큰 model (88M) — 단순 task 에 over-parameterize 무의미
6. Edge-Bottom + Edge-Top obj 식별 (chip 6개 안) = V3 의 진짜 weak point
7. EMA 효과 +0.55pp test (single seed, multi-seed 검증 필요)

### intra-distribution per-class (★ V3 weak point)

| Distribution | R-only | obj-only | Tier1 ens | V3 |
|---|---|---|---|---|
| Center | 1.0000 | 0.9954 | 1.0000 | 1.0000 |
| Donut | 1.0000 | 1.0000 | 1.0000 | 0.9954 |
| **Edge-Bottom** | 0.9630 | 0.9537 | 0.9583 | **0.9907** |
| Edge-Ring | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| **Edge-Top** | 0.9630 | 0.9722 | 0.9722 | **0.9954** |
| Full | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| Normal | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| Thick-Edge | 1.0000 | 1.0000 | 1.0000 | 1.0000 |

V3 가 모든 distribution 에서 best 이지만 Edge-Bottom/Top 의 obj 식별이 진짜 ceiling.

### Active class 정책

33-class → **20 active + 14 archive** (데이터 보존, 학습 list 만 제한).

| Group | Classes |
|---|---|
| Active 20 | Donut×5, Edge-Bottom×5, Edge-Ring×4 (-invalid_main), Edge-Top×5, Thick-Edge_invalid_main |
| Archive 14 | Center×5, Full×5, Edge-Ring_invalid_main, Normal_bank_boundary, Starburst, CommaCluster |

YAML: `experiments/active_classes_20.yaml`, `experiments/archive_classes_14.yaml`. archive 데이터: `D:/project/data/wm-811k/unknown_archive/<class>/` (copy, 원본 보존).
상세 → `ACTIVE_CLASSES.md` + `feedback_active_class_policy.md`.

### Block expand 정책 (V3 enabling factor)

categorical map (obj_id, one-hot, prob) 의 spatial resize **`_chipgrid_resize.block_expand_2d` 만 사용**. PIL/torch BICUBIC, NEAREST hardcode 금지. 자세한 건 `feedback_block_expand_only.md`.

### TTA 금지 (재확인)

wafer class identity 가 angle/위치에 묶여있어 (scratch_21deg, Edge-Top/Bottom) rotation/flip TTA 시 다른 class 답 ensemble. V3 도 같은 정책. 자세한 건 `feedback_no_tta_wafer.md`.

### 다음 phase (사용자 진행 중)

- `_chipgrid_kde_gmm.py` — Phase 1 + 2: per-class KDE (chip 위치) + GMM (count vector)
- `cnn_train_chipgrid_fusion.py` — Phase 3: V3 backbone + KDE/GMM late fusion
- `chipgrid_class30_target.yaml` — 8 새 wafer-canvas class 합성 후 활용 (DiagonalSmear, CrossScratch, CrescentArc, SpiralTrail, ParallelScratches, EdgeSmudge, BlobChain, BrokenRing)

상세 → `STATUS.md`.

### 6-Tier ensemble plan deprecated

V3 단독 0.9946 > Oracle 0.9919 → Tier 3 mid-fusion (14h), Tier 4 cross-attention, Tier 5 MoE, Tier 6 KD = expected gain 거의 0. Plan 의 미실행 Tier 3-6 = deprecated (`STATUS.md` 참조).

## 데이터 / 모델

| 종류 | 경로 | 라벨 |
|---|---|---|
| Wafer 33-class single-label | `D:/project/data/wm-811k/unknown/<class>/*.png` | 1 wafer = 1 class |
| Wafer archive 14 (copy 보존) | `D:/project/data/wm-811k/unknown_archive/<class>/*.png` | 학습 X, 보존 |
| chip-object 5-class | `D:/project/data/wm-811k/classification_chips/<obj>/*.png` | 1 chip = 1 obj |
| obj_id_maps | `D:/project/data/wm-811k/obj_id_maps/<basename>.npy` (32×32 uint8) | chip CNN inference 산출 (flat basename) |

| 모델 | 경로 | param | 학습 시간 |
|---|---|---|---|
| **V3 obj-only chipgrid** | `logs_chipgrid/v3_full_260503_160436_*/best_model.pth` | **1.16M** | <1 min |
| R-only ConvNeXtV2 base | `logs_wafer/overall/best_model.pth` | 88M | 12.5h (full) |
| obj-only 4-layer CNN | `logs_objid_ablation/overall/best_model.pth` | 0.4M | 2 min |
| chip-object CNN | `logs_chip/overall/best_model.pth` | 88M | ~3h |
| compound 3ch BICUBIC | `logs_compound/overall/best_model.pth` | 88M | 1.5h |

## Cross-link

- 7 핵심 발견 → `DISCOVERY.md`
- 모든 model 결과 → `RESULTS.md`
- intra-distribution 분석 → `INTRA_DIST.md`
- 20 active + 14 archive → `ACTIVE_CLASSES.md`
- 진행 상태 + GMM hybrid → `STATUS.md`
- 학습/평가 명령 → `.claude/skills/wafer-classifier/SKILL.md`
- chipgrid V0~V6 sweep 시간순 → `D:/project/known-cnn/docs/chipgrid/RESULTS.md`
- TTA 금지 → `~/.claude/projects/D--project-known-cnn/memory/feedback_no_tta_wafer.md`
- block_expand 정책 → `~/.claude/projects/D--project-known-cnn/memory/feedback_block_expand_only.md`
- active class 정책 → `~/.claude/projects/D--project-known-cnn/memory/feedback_active_class_policy.md`
- V3 best memory → `~/.claude/projects/D--project-known-cnn/memory/project_v3_chipgrid_best.md`

## 수정 금지

- V3 best result (val_f1 0.9946) 의 hparam, ckpt 경로, intra-distribution 표 변경 시
  `RESULTS.md` 와 `project_v3_chipgrid_best.md` 동시 갱신
- block_expand 정책 / active class 정책 / TTA 금지 어떤 변경도 금지
- 학습 결과 폴더 (`logs_*`) 무단 삭제 절대 금지 (글로벌 룰)
- 데이터 폴더 (`unknown/`, `unknown_archive/`) 무단 삭제 절대 금지
