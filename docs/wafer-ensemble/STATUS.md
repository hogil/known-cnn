# Status — Wafer Classifier Phase A/B/C

본 문서는 wafer 분류기의 **현재 진행 상태** + 다음 phase 계획. 새 세션 진입
시 이 문서로 즉시 다음 할 일 파악 가능.

## 현재 상태 (2026-05-03)

### V3 best result (FAIR comparison)
- **V3 obj-only chipgrid**: val_f1 **0.9946**, test_f1 0.9872
- ckpt: `logs_chipgrid/v3_full_260503_160436_0.99_0.99/best_model.pth`
- params: 1.16M, 32×32×5 one-hot obj only (R 채널 빼도 동률)
- Oracle ceiling 0.9919 도 추월 → 6-Tier ensemble plan deprecated

자세한 건 `RESULTS.md`, `DISCOVERY.md`.

### Active class 결정
- 33-class → 20 active + 14 archive (`ACTIVE_CLASSES.md`)
- archive 데이터: `D:/project/data/wm-811k/unknown_archive/<class>/` (copy 보존)
- YAML: `experiments/active_classes_20.yaml`, `experiments/archive_classes_14.yaml`

### 사용자 in-progress (touch X)
- `_chipgrid_kde_gmm.py` — Phase A: per-class KDE (chip 위치) + GMM (count vector) sanity check
- `cnn_train_chipgrid_fusion.py` — Phase B: V3 backbone + KDE/GMM late fusion 학습
- `chipgrid_class30_target.yaml` — Phase C: 8 새 wafer-canvas class 합성 후 활용 target list

## Phase A — GMM/KDE sanity check (사용자 진행 중)

### 목표
chip object 별 chip 위치 / count 분포가 wafer class 식별에 자체 충분한지 sanity.

### 산출 (예상)
- `_chipgrid_kde_gmm.py` 출력 → `logs_chipgrid_kde_gmm/<run>/`
- per-class KDE log-lik + GMM log-lik
- argmax (33 class) 정확도 → V3 와 비교
- KDE/GMM 자체로 75-90% 정도면 fusion concat 의 의미 있음

### 사용
```bash
python _chipgrid_kde_gmm.py --n-per-class 100 --seed 42
python _chipgrid_kde_gmm.py --n-per-class 220 --bandwidth 1.5 --n-components 3
```

## Phase B — V3 backbone + KDE/GMM late fusion (사용자 진행 중)

### 목표
V3 의 obj-only weak point (Edge-Bottom/Top obj 식별) 을 GMM count + KDE 위치
prior 로 보강.

### 아키텍처
```
입력 (32×32×6, R + 5 obj binary)
  ↓ ChipGridCNN body (penultimate 256-D)
                                + KDE log-lik (n_classes-D)
                                + GMM log-lik (n_classes-D)
  ↓ Concat → (256 + 2*n_classes)-D
  ↓ BN1d + Linear(., 128) + GELU + Dropout
  ↓ Linear(128, n_classes)
```

### 산출 (예상)
- `logs_chipgrid_fusion/<run>/best_model.pth`
- val_f1 측정. V3 단독 0.9946 추월 시 GMM hybrid 채택, 같으면 V3 단독.

### 사용
```bash
python cnn_train_chipgrid_fusion.py --n-per-class 100 --epochs 30 --seed 42 --model-tag fusion_seed42
```

## Phase C — 8 새 wafer-canvas class 합성

### 목표
Edge weak point 우회 — 새 wafer-canvas pattern (object-less) 8 종 추가:
DiagonalSmear, CrossScratch, CrescentArc, SpiralTrail, ParallelScratches,
EdgeSmudge, BlobChain, BrokenRing.

### 데이터 위치 (합성 후 예상)
- `D:/project/data/wm-811k/unknown/<NewClass>/*.png`
- target list: `configs/chipgrid_class30_target.yaml`

### 활용
- 기존 active 20 + 새 8 wafer-canvas + Starburst + CommaCluster (archive 에서 복귀)
  = 30 class
- target YAML 의 strict check 가 합성 전엔 fail 해야 정상

## 미실행 / Deprecated

### Tier 3-6 (deprecated)
- Tier 3 mid-fusion (14h) — V3 가 oracle 추월하면서 expected gain 0
- Tier 4 cross-attention (16h) — 같은 이유
- Tier 5 MoE (20h) — 같은 이유
- Tier 6 KD (8h) — V3 가 이미 1.16M (small) 이라 KD 의 의미 약함

자세한 건 `ENSEMBLE_TIERS.md` (legacy 참조).

## Critical files

| 파일 | 역할 | 상태 |
|---|---|---|
| `cnn_eval_chipgrid.py` | V3 + V0~V4 변종 sweep entry | ✅ |
| `_chipgrid_resize.py` | block_expand_2d (categorical resize) | ✅ |
| `cnn_train_objonly.py` | obj-only 4-layer entry + active YAML | ✅ |
| `cnn_train_compound.py` | compound 3ch + g-channel-mode + active YAML | ✅ |
| `_intra_dist_eval.py` | distribution / object 분리 정확도 | ✅ |
| `_v3_fair_eval.py` | V3 ckpt 를 우리 0.8/0.2 val 1420 위 fair eval | ✅ |
| `_chipgrid_kde_gmm.py` | Phase A KDE/GMM sanity (사용자 진행) | 🔄 |
| `cnn_train_chipgrid_fusion.py` | Phase B fusion 학습 (사용자 진행) | 🔄 |
| `experiments/active_classes_20.yaml` | active class list | ✅ |
| `experiments/archive_classes_14.yaml` | archive class list | ✅ |
| `configs/chipgrid_class20_hard.yaml` | chipgrid-eval 다른 selection | ✅ |
| `configs/chipgrid_class30_target.yaml` | Phase C 합성 후 target | ⏳ (합성 전 strict fail) |

## 산출 디렉토리 (gitignored)

| 위치 | 내용 |
|---|---|
| `logs_chipgrid/` | V0~V4 chipgrid sweep + V3 best |
| `logs_chipgrid_kde_gmm/` | Phase A 산출 |
| `logs_chipgrid_fusion/` | Phase B 산출 |
| `logs_objid_ablation/` | obj-only 4-layer 학습 |
| `logs_wafer/` | R-only ConvNeXt 학습 |
| `logs_compound/` | compound 3ch 학습 (ceiling 0.9784) |
| `results_v3_eval/` | V3 fair eval (val 1420) |
| `results_intra_dist/` | distribution / obj 분리 정확도 |
| `results_disagree/` | oracle ceiling + 4-group disagreement |
| `results_ensemble_ep10/` | ep10 fair compare base |

## TODO

### 즉시 (사용자 외 진행)
- [ ] V3 multi-seed 5 평균 검증 (ep30, n=220) — EMA 효과 통계적 유의성
- [ ] V3 ckpt 의 production deploy entry 작성 (`cnn_predict_chipgrid.py`)
- [ ] Normal pool max_prob threshold 측정 — open-set unknown rejection

### Phase A → B 사용자 진행
- [ ] Phase A 결과 본 후 fusion 효과 평가
- [ ] Phase B fusion val_f1 측정 vs V3 단독 0.9946 비교

### Phase C 합성 후
- [ ] 8 새 wafer-canvas class 합성 (`_sample_gen.py` 확장)
- [ ] active list → 30 class (target YAML 활용)
- [ ] V3 (또는 fusion) 30-class 학습 + intra-distribution 분석

## 다음 세션 진입 시

1. 본 STATUS.md 의 "현재 상태" + "사용자 in-progress" 확인
2. `RESULTS.md` 의 V3 best (val_f1 0.9946) 재확인
3. `ACTIVE_CLASSES.md` 의 20 active + 14 archive 정책 재확인
4. `feedback_block_expand_only.md`, `feedback_active_class_policy.md` 재로드
5. 우선 순서: Phase A 결과 확인 → Phase B fusion → Phase C 합성

## 관련 파일

- 본 문서 → `README.md`, `DISCOVERY.md`, `RESULTS.md`, `INTRA_DIST.md`, `ACTIVE_CLASSES.md`
- skill → `.claude/skills/wafer-classifier/SKILL.md`
- chipgrid (V0~V6 시간순) → `D:/project/known-cnn/docs/chipgrid/`
- legacy → `ENSEMBLE_TIERS.md` (Tier 3-6 deprecated), `PRODUCTION_RULE.md` (V3 deploy 전 마이그레이션 필요)
