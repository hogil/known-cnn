# Plan: iter 12 — Pure Baseline 부터 Soft-Proportional CutMix 까지 상세 step plan

> **★ 영구 보관**: 동일 내용이 `docs/chip-multilabel/iters/iter_12_plan.md` 에 복사되어 있음.
> **last sync: 260506 ~16:00 (재부팅 직전)**

## ⚠️ 재부팅 직전 status (260506)

### 코드 변경 적용 완료 ✅

| 파일 | 변경 |
|---|---|
| `chip_predict/cnn_predict.py` | pseudo-label 코드 전체 제거 (CLI args 4개 + setup + save logic + csv column) |
| `chip_predict/cnn_predict_chip.py` | `DEFAULT_PSEUDO_LABEL_OUT` 제거 |
| `chip_train/_chip_trim_inplace.py` | `STUB_DIRS = ['particle_blast', 'scratch_21deg']` 제거 |
| `dist_apply/_sample_gen.py` | v19: alpha_fork (sev 0.70-0.85, smear 5-8) / alpha_scratch (sev 0.85-0.95, smear 18-30) / alpha_scratch_rot (sev 0.78-0.90, smear 8-13) / **theta=-21° 우상향** / `INTENSITY_ALPHA_SCALE['weak']` 0.40→0.60 / `CHIP_OBJ_PER_CLASS_CAP` 100→200 / quality filter strict (0.10 / 0.02 strong) / `MIN_CHIP_STRONG_GRADE_RATIO` 신규 |
| `dist_apply/_sample_gen_gpu.py` | v19: alpha_fork_t / alpha_scratch_t (denser) / alpha_scratch_rot_t (★ theta=-21° 우상향, 사용자 directive "수직선 기준 오른쪽으로만") 별도 함수 + ALPHA_FNS_T mapping 갱신 |
| `chip_multilabel/_train_chip_variant.py` | T0/T9 variant enum 추가, scattered CutMix mode + soft proportional label CLI flags + 코드 + train_summary 새 fields |
| `chip_multilabel/losses.py` | T9 SigmoidFocalLoss 추가 (RetinaNet style) |
| `chip_multilabel/_bit_metrics.py` | NEW — per-bit metric (CF1=macro F1, OF1=micro F1, bit-FAR, chip-FAR, 3plus%) |
| `chip_multilabel/_threshold_sweep.py` | NEW — Pareto frontier + composite optima |
| `chip_multilabel/decision_tree.py` | ≥3 active top-2 truncate 폐기 → '3plus_active' decision_type |

### 데이터 정리 ✅

- `D:/.../classification_chips/scratch_21deg/`, `particle_blast/` 폴더 영구 삭제 (legacy)
- `classification_chips/` 의 양호-looking chip 들 quality filter (0.10/0.02) 로 cleanup
- 모든 stale `.pyc` 삭제

### chip 새로 만들기 — ⚠️ 미완성

**시도 history**:
1. ❌ Background `bnam0jos1` GPU 합성 (471/8450 진행) — fix 전 stale module → kill
2. ❌ Background `b267rzicm` GPU 합성 — v19 코드 적용 시작했지만 사용자 "예전에 만들던거로 돌아갔네" → kill
3. ❌ 모든 python.exe kill (재부팅 직전)

**현재 `classification_chips/` 상태 (혼합 — fix 전후 chip 섞임)**:
| obj | chip count | 신뢰도 |
|---|---:|---|
| bank_boundary | 0 | empty |
| fork | 200 (cap) | ⚠️ 일부 stale module 산출 |
| scratch | 200 (cap) | ⚠️ 일부 stale module 산출 |
| scratch_rot | 136 | ⚠️ angle 검증 안 됨 (slope -0.05 ~ -0.16, near vertical) |
| invalid_main | 200 (cap) | ⚠️ fix 전 chip |

**사용자 directive 사용 후**: "다 꺼라" → 모든 process kill 됨.

**chip 검증 결과 (재부팅 전 마지막 측정)**:
- fork chip defect_ratio = 12.86% (★ v19 강도 적용 — 기존 6.9% 의 2× ↑)
- scratch / scratch_rot defect_ratio ~11% (양호)
- scratch_rot angle slope -0.06~-0.16 (near vertical, -21° 검증 어려움 — chip area 작아 line 변동 미세)

### ★ 다음 세션 시작 시 first action (재부팅 후)

1. **classification_chips/{bank_boundary, fork, scratch, scratch_rot, invalid_main} 모두 비우기** (혼합 chip 정리)
2. **GPU 합성 fresh dispatch** — `cd dist_apply && python _sample_gen_gpu.py --n 200 --save-workers 8`
   - v19 코드 자동 적용 (이미 fix 됨, .pyc 도 정리)
   - ETA ~25-50분
3. **chip 검증**:
   - fork defect_ratio 측정 (목표: ≥10%, 기존 6.9%)
   - scratch_rot 시각 plot — line angle 우상향 (top tilts right) 확인
4. **chip_multilabel master 재생성** (gen_eval_set.py)
5. **iter 12 학습 재실행** (T0~T9)

---

## ★ 진행 status (last sync: 260506_141X)

### Phase 진행 현황

| # | Phase | Variant | 학습 시간 | Eval 시간 | Status |
|---|---|---|---:|---:|---|
| 0 | T0 pure CE (no CutMix, no LS) | ce_ls01 ls=0 | 6.6분 | 7.0분 | ✅ done |
| 1 | T1 = T0 + LS=0.1 | ce_ls01 ls=0.1 | 11.8분 | 12분 | ✅ done |
| 2 | T3 Focal (CutMix p=0.25, single mode) | focal | 4.5분 | 0.8분 | ✅ done |
| 2 | T4 ASL (★ first multi_hot) | asl | 4.5분 | ~50초 | ✅ done (의외로 약함) |
| 2 | T5 BCE | bce | 4.5분 | ~50초 | ✅ done (또 over-firing) |
| 2 | T6 BCE→ASL | bce_then_asl | 4.7분 | ~70초 | ✅ done (= T5, warmup phase best) |
| 2 | T7 BCE+LS=0.20 | bce_ls | 4.8분 | ~70초 | ✅ done (3plus 6.4% 등장 + over-firing) |
| 3 | T9 Sigmoid Focal | sigmoid_focal | 4.8분 | ~80초 | ✅ done (3plus 7.9%) |
| **2.5** ★ Threshold sweep | — | — | ~수초 | ✅ done (T9 + θ=0.6 → FAR 0.57%) |
| 4 | ★ Scattered CutMix sweep | — | (3시간) | (30분) | 🔄 시작 |
| 2 | T7 BCE+LS=0.20 | bce_ls | — | — | ⏳ |
| 2.5 | ★ Threshold sweep + Pareto | — | — | ~30분 | ⏳ |
| 3 | T9 Sigmoid Focal | sigmoid_focal | ~5분 | ~1분 | ⏳ |
| 4 | ★ Scattered CutMix sweep (T9+T7 × 4×4=32 trains) | — | ~3h | ~30분 | ⏳ (코드 patch ✓) |
| 5 | Ensemble + 5-seed | — | — | ~20분 | ⏳ |
| 6 | iter 12 paper doc | — | — | ~30분 | ⏳ |

### 핵심 성능 표 (per-bit metric, I3 inference)

| variant | loss | CutMix 작동 | macro F1 | bit-FAR | chip-FAR | F1_bb | F1_fork | F1_sc | F1_sr | 3plus% |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| T0 pure CE | ce_ls01 ls=0 | ❌ | 0.7247 | **0.70%** | 2.80% | 0.82 | 0.43 | 0.66 | 0.99 | 0% |
| T1 CE+LS=0.1 | ce_ls01 ls=0.1 | ❌ | 0.7549 | 24.5% ❌ | 96% ❌ | 0.82 | 0.68 | 0.55 | 0.97 | 1.6% |
| T3 Focal | focal | ❌ | **0.7974** ★ | 1.77% | 7.10% | 0.77 | 0.54 | 0.89 | 0.99 | 0% |
| T4 ASL γn=4 | asl | ✅ | 0.7158 ↓ | 6.48% | 25.8% | 0.77 | 0.39 ↓ | 0.70 | 1.00 | 0% |
| T5 BCE | bce | ✅ | 0.7173 ↓ | 24.0% ❌ | 96% ❌ | 0.91 | 0.40 ↓ | 0.56 | 0.99 | 0% |
| T6 BCE→ASL (warmup) | bce_then_asl | ✅ | 0.7173 = T5 | 24.0% ❌ | 96% ❌ | 0.91 | 0.40 | 0.56 | 0.99 | 0% |
| T7 BCE+LS=0.20 | bce_ls | ✅ | 0.7129 ↓ | 41.7% ❌❌ | 96% ❌ | 0.81 | 0.43 | 0.61 | 1.00 | **6.4%** ★ |
| T9 Sigmoid Focal | sigmoid_focal | ✅ | (eval) | | | | | | | |
| T6 BCE→ASL | bce_then_asl | ✅ | (대기) | | | | | | | |
| T7 BCE+LS=0.20 | bce_ls | ✅ | (대기) | | | | | | | |
| T9 Sigmoid Focal | sigmoid_focal | ✅ | 0.7637 | 29.7% | 96% | 0.94 | 0.44 | 0.70 | 0.98 | **7.92%** ★ |
| **T9 + global θ=0.50** ★ | sigmoid_focal | ✅ | **0.7513** | **4.23%** ★ | ? | | | | | |
| **T9 + global θ=0.60** ★ | sigmoid_focal | ✅ | 0.7154 | **0.57%** ★ | ? | | | | | |

### Findings (누적)

1. T0→T1 (LS): macro F1 +0.030 ↑, bit-FAR +35배 악화 ← LS naive 부작용
2. T1→T3 (Focal): macro F1 +0.042 ↑ AND bit-FAR -93% 개선 ← Focal calibration 효과 (LS 와 메커니즘 다름)
3. T0/T3/T4 의 3plus% = 0 (multi_hot 도 sigmoid 분리 잘 됨). T5-T9 BCE 부터 multi-active 자연스러울지 검증 대기
4. fork F1 = 0.43 / 0.68 / 0.54 / **0.39 (T4 ASL!)** — single-class 학습 한계. ASL γ_neg=4 의 negative suppression 이 weak signal 압살
5. **T4 ASL default γ_neg=4 너무 aggressive** — fork precision 1.0 / recall 0.24. asymmetric 의 함정.
6. **T5 BCE (no LS) 도 over-firing** — bit-FAR 24%, scratch FP 959 (1000 chip 모두 scratch 띄움). naive BCE 의 saturation 문제. fork F1 0.40 (T4 와 동일).
7. ★ 현재 winner 여전히 T3 Focal (macro 0.80, bit-FAR 1.8%) — single-class focal 이 multi_hot BCE 보다 우수.
8. T7 BCE+LS=0.20 이 결정적 (LS=0.20 이 saturation 풀어줄 가능성). T9 Sigmoid Focal 도.
9. ★ Phase 4 의 scattered CutMix + soft proportional label 가설: 단순 random CutMix p=0.25 + OR rule 은 부족함이 입증됨. soft proportional 이 진짜 fix.
10. **T6 BCE→ASL warmup 효과 없음** — best_epoch=1 이 BCE phase 안에서 나옴 → ASL transition 전 stop. T6 = T5 정확 동일. paper-grade negative finding (hybrid scheme 의 한계 — 빠른 saturation 으로 second phase 도달 못함).
11. fork F1 패턴: T4=0.39, T5=0.40, T6=0.40 — 4 chip 소수만 정답. 모두 "fork chip 의 75% 를 fork 라고 안 부름". CutMix random 이 fork chip 위에 다른 obj paste 했지만 OR rule [1,1,...] 학습이 fork bit 를 평균화. **soft proportional 이 본질적 fix**.
12. **★ T7 BCE+LS=0.20 의 3plus_active 6.4% 첫 등장** — multi-label 학습은 일어났지만 threshold 0.5 부적절 (LS 강함 → prob right-shift). Phase 2.5 threshold sweep 으로 macro F1 살리면서 bit-FAR 떨어뜨릴 가능성 큰 case. 사용자 가설 검증 정확한 case.
13. **현재 winner = T3 Focal 0.80 / 1.77% 변화 없음** — single-class focal 이 모든 multi_hot variants 압도. random CutMix 의 한계 명백 → Phase 4 scattered + soft proportional 가치 입증.
14. **★ Phase 2.5 threshold sweep 결과** (사용자 가설 검증):
    - **per-class F1-max threshold (I3)의 함정**: fork prob 분포 평탄한 class 에서 낮은 threshold → over-fire. global threshold 가 더 robust.
    - **T1/T7 (LS naive) 는 어떤 threshold 로도 못 살림** — LS 가 prob 분포 자체 평탄화 → 분리 불가 (paper-grade negative finding).
    - **★ T9 Sigmoid Focal + global θ=0.60 = macro 0.7154 / bit-FAR 0.57%** — T0 (0.70%) 보다 낮은 FAR + multi-label 학습 능력.
    - **T4/T5/T6 도 global θ 만 잘 잡으면 0.69 / FAR ≤3% 회복 가능** — naive default I3 가 over-firing 의 주범.
15. T9 가 진짜 winner candidate. Phase 4 의 scattered + soft proportional 이 macro F1 0.85+ + FAR < 1% 동시 달성 목표.

---

## 0. Context

**왜 이 plan**:
- 직전 visible run 의 "baseline" 은 `--cutmix-p 0.25 --cutmix-rect 0.5` 가 켜져있어 사용자 의도의 *순수 baseline* 이 아니었음 (CE 라 코드상 CutMix 분기 안 들어가서 사실상 OFF 였지만 명시적이지 않음).
- 사용자 directive (260506): "baseline 은 순수한거고 (CutMix) 이거 빼고 해야지". 명시적 pure CE baseline 을 다시 학습 후 modern variants 를 단계적으로 쌓아가며 paper-quality ablation 작성.
- 동시에 새 CutMix 설계 (scattered patches + soft proportional label `ratio × discount(0.7) × α`) 도입 요청. ratio·α sweep ablation 으로 best config 식별.
- 평가 framework 도 재정의됨: **10-defect macro F1 + FAR 단 2 metric만**. 5 OOD wafer-pattern class 의 어떤 성능 표도 작성 금지 (memory rule `feedback_no_ood_class_performance.md`).

**원하는 outcome**:
1. 각 실험이 무엇을 바꿨는지, 무엇을 측정하는지가 명확한 ladder (T0 → T1 → ... → T9 → T9-soft).
2. 사용자와의 용어 alignment (baseline 정의, CutMix 동작, label 의미).
3. 12-cell ratio×α ablation grid 결과 best (ratio*, α*) 식별.
4. Final paper iter doc `iter_12_*.md` (모든 표, 단 2 metric).

---

## 1. 데이터 설명 (★ 사용자와 alignment 우선)

### 1.1 학습용 (training)

**물리 위치**: `D:/project/data/wm-811k/chip_multilabel/` **의 4 single defect 폴더만** 사용.
- `bank_boundary/` 200 chip
- `fork/` 200 chip
- `scratch/` 200 chip
- `scratch_rot/` 200 chip
- (옵션) `Normal/` 200 chip — `--no-normal` 로 끄면 4-class only, 켜면 Normal y=-1 sentinel + multi-hot zero target.

**중요**: 별도 `4-single/` 폴더 없음 (현재 비어있음). 학습 trainer `--data-root D:/project/data/wm-811k/chip_multilabel` + `--include-classes bank_boundary,fork,scratch,scratch_rot` 로 master 의 4 폴더만 골라 씀.

**sample 특성**:
- 모두 single-positive (한 chip = 한 obj). 학습 데이터에 combo 없음.
- p70 strong-defect filtered (`source-strength-pct 70` 으로 master 합성됨) — 약한 defect 30% 까지 포함하지만 강한 70% 위주.
- 따라서 multi-label combo 학습은 **학습 데이터로는 불가** → CutMix mosaic 가 multi-label 학습 유일 source.

**split**: 80/20 stratified (seed=42), val_acc 로 best_model.pth 결정. (4-class val 은 ep1 부터 saturation 되는 경향 — final_epoch_model.pth 도 항상 저장.)

### 1.2 평가용 (evaluation)

**물리 위치**: 동일 master folder + `manifest.csv` (3,250 rows).

**class 구성 (17)**:
| group | class | n | 학습 | 성능 측정 |
|---|---|---|---|---|
| single | bank_boundary, fork, scratch, scratch_rot | 200 each | ✓ | ✓ (4-single F1) |
| combo | bb+fork, bb+scratch, bb+sr, fork+scratch, fork+sr, sc+sr | 200 each | ✗ | ✓ (6-combo F1) |
| Normal | Normal | 200 | ✗ (--no-normal) | **∗ FAR 만** (NON_DEFECT_GT) |
| Invalid | Invalid | 50 | ✗ | **∗ FAR 만** (NON_DEFECT_GT) |
| OOD wafer-pattern | CenterCircle, CenterDonut, CrescentArc, RingDots, Row | 200 each | ✗ | **★ class별 F1/distribution 절대 X** (단 FAR contribution) |

**∗ Normal/Invalid 의 F1 도 표시 X** (사용자 directive 260506 "Normal F1, Invalid F1 이거는 필요없다"). FAR 통한 간접 측정만.

**runtime sampling**: `discover_records_runtime(eval_root, n_per_class, strength_min/max, include_classes, seed)` — 평가 시 manifest 읽어 class 별 N 개 random pick. subset 폴더 절대 안 만듦 (memory rule).

### 1.2.1 ★ Multi-active declaration 정책 (사용자 directive 260506)

모델 head = 4 logit (sigmoid). threshold 후 활성 갯수 → declaration:

| 활성 갯수 | declaration | decision_type | 예 |
|---|---|---|---|
| 0 | Normal | normal | [0,0,0,0] |
| 1 | single defect (4 종류) | single | [0,1,0,0] → fork |
| 2 valid combo | combo defect (6 종류) | combo | [0,1,1,0] → fork+scratch |
| 2 invalid combo | top-1 collapse | combo_collapsed | [1,0,0,1] but bb+sr in COMBO_KEYS so OK; 다른 invalid 2-combo 시 top-1 |
| **≥3** | ★ **truncate 폐기** — 그대로 declare | **3plus_active** | [1,1,1,0] → "bank_boundary+fork+scratch" (COMBO_KEYS 에 없음 → 자동 wrong) |

**왜**: 사용자 directive "top2 keep 은 하지마라 이것도 얼마나 틀리나봐야지". ≥3 active 는 이전에 prob top-2 만 keep 했지만 — 폐기. raw active set 그대로 declare → 어떤 GT 와도 match 안 됨 → 자동 wrong → recall 페널티. **"모델이 ≥3 동시 declare 하는 빈도" 자체가 paper-grade 진단 지표**.

**구현**: `decision_tree.py::decide` 수정 완료 (260506). `decision_type='3plus_active'`, class_key='a+b+c' 형태.

---

### 1.3 Metric 정의 (★ per-bit framework, 사용자 directive 260506)

**핵심 원칙**: chip 단위 class_key matching 폐기. 각 chip = **4-bit GT × 4-bit pred → 4 binary classification problem**. 사용자 directive: "각각 맞췄는지 틀렸는지". 표준 multi-label classification methodology (Wang 2016, Chen 2019, Ridnik 2021).

**왜 F1 만 보나 (사용자 directive)**: class 수 ↑ → TN 폭증 → accuracy 왜곡 (모두 0 예측해도 high acc). F1 = 2·TP/(2·TP+FP+FN), TN 무관 → robust.

**4-bit GT 매핑**:
```
TRAIN_CLASSES = (bank_boundary, fork, scratch, scratch_rot)  # 4-bit indices

class_key                         GT bits
─────────────────────────────────────────────────
bank_boundary                  → [1,0,0,0]
fork                           → [0,1,0,0]
scratch                        → [0,0,1,0]
scratch_rot                    → [0,0,0,1]
bank_boundary+fork             → [1,1,0,0]
bank_boundary+scratch          → [1,0,1,0]
bank_boundary+scratch_rot      → [1,0,0,1]
fork+scratch                   → [0,1,1,0]
fork+scratch_rot               → [0,1,0,1]
scratch+scratch_rot            → [0,0,1,1]
Normal, Invalid, CenterCircle,
CenterDonut, CrescentArc,
RingDots, Row                  → [0,0,0,0]   ← NON_DEFECT_GT
```

**Metric set** (per cell = model × inference variant):

```
N = 모든 chip (eval split, e.g. 2,600)
For each c in {bb, fork, sc, sr}:                # 4 binary classifications
    TP_c = #{chip : GT[c]=1 AND pred[c]=1}
    FP_c = #{chip : GT[c]=0 AND pred[c]=1}
    FN_c = #{chip : GT[c]=1 AND pred[c]=0}
    F1_c = 2·TP_c / (2·TP_c + FP_c + FN_c)

★ CF1 (macro F1)  = mean(F1_bb, F1_fork, F1_sc, F1_sr)              # paper main
   OF1 (micro F1) = 2·ΣTP / (2·ΣTP + ΣFP + ΣFN)                    # 4N bit aggregate
   per-class F1   = F1_bb, F1_fork, F1_sc, F1_sr (개별 row)

★ bit-FAR  = ΣFP_c bits in NON_DEFECT_GT chips / (4 × |NON_DEFECT_GT chips|)    # real-env
   chip-FAR = #{chip in NON_DEFECT_GT : ≥1 FP bit} / |NON_DEFECT_GT chips|
   3plus_active 빈도 = #{chip : decision_type='3plus_active'} / N    # over-firing 진단
```

**chip 1개 = 항상 4 binary 결과** (한 비트만 판정 X). 예:
- GT [0,1,1,0] vs pred [0,1,0,0]: bb=TN, fork=TP, sc=**FN**, sr=TN → 1 TP, 0 FP, 1 FN, 2 TN
- GT [0,1,1,0] vs pred [0,1,1,1]: bb=TN, fork=TP, sc=TP, sr=**FP** → 2 TP, 1 FP, 0 FN, 1 TN

**5 OOD class별 F1/distribution 절대 측정 X** (memory rule). NON_DEFECT_GT chip 으로 묶여 FAR 에만 contribute.

**Reference**:
- Wang et al. 2016 CVPR (CNN-RNN) — CF1 / OF1 표준 명명
- Chen et al. 2019 CVPR (ML-GCN) — multi-label image classification 표 양식
- Ridnik et al. 2021 ICCV (ASL) — multi-label loss + per-class F1
- Tsoumakas & Katakis 2007 — multi-label evaluation overview

---

## 2. 학습 / 평가 인프라 (수정 X — 이미 다 있음)

| file | 역할 | 새로 patch? |
|---|---|---|
| `chip_multilabel/_train_chip_variant.py` | trainer entry | ★ Phase 4 에서 scattered CutMix patch |
| `chip_multilabel/losses.py` | loss factory + T9 SigmoidFocalLoss (이미 추가됨) | NO |
| `chip_multilabel/run_stage1.py` | inference (manifest sampling + I0-I13) | NO |
| `chip_multilabel/inference_variants.py` | I3/I7/I12/I13 등 | NO |
| `chip_multilabel/eval_dataset.py` | 17-class manifest reader | NO |
| `chip_multilabel/constants.py` | TRAIN_CLASSES + ALL_CLASS_KEYS | NO |
| ad-hoc metric script | 10-def macro + FAR 계산 + 표 생성 | ★ Phase 5 에서 신규 작성 |

**학습 환경 제약** (memory rule `feedback_chip_train_batch_safe.md`):
- 공유 GPU 환경에서 chip 학습은 `--batch 8 --accum 4` (effective 32) 만 안전. batch=16 OOM 사고.
- run_in_background 로 dispatch (Bash tool), `Start-Process -WindowStyle Hidden` 절대 금지.

**아우그멘테이션 제약** (memory rule `feedback_no_rotation_aug_chip.md`):
- Rotation/Flip 영구 금지 (scratch ↔ scratch_rot 회전 구분 깨짐).
- RandomAffine translate+scale 만 허용.
- TTA 영구 금지.

---

## 3. 실험 ladder — 단계별 상세 (★ 본 plan 의 핵심)

각 단계 = "직전 단계에서 무엇을 추가했는가 / 무엇을 측정하는가" 가 명확. 한 번에 한 axis 만 변경 (memory rule `feedback_atomic_method_iteration.md`).

### Phase 0 — T0 Pure Baseline (CutMix 없음, LS 없음)

**Hypothesis**: pure CE softmax 4-way classifier 가 master folder 4-single 200/class 학습 만으로 17-class 평가에서 어디까지 가는지 측정. modern variant 들의 비교 floor.

**데이터**:
- `data-root = D:/project/data/wm-811k/chip_multilabel`
- `include-classes = bank_boundary,fork,scratch,scratch_rot` (4 only, Normal 도 제외)
- `--no-normal` 명시 (Normal 학습 X — 가장 순수 형태)
- 200 chip × 4 class = 800 chip. 80/20 split → train 640 / val 160.

**기법**:
- loss = `ce_ls01` with `--ls 0.0` (= pure CE, smoothing 0)
- **CutMix OFF**: `--cutmix-p 0` (명시적으로 0)
- **Mixup OFF / RandomErasing OFF**
- aug: RandomAffine (translate ±3%, scale ±3%, **NO rotation, NO flip**)
- optimizer: AdamW, single LR 1e-4, cosine schedule, no warmup
- epochs: 8 (val saturates ep1 — 8 면 충분, final_epoch_model.pth 도 저장)
- batch=8 accum=4

**checkpoint**: best_model.pth (val_acc max), final_epoch_model.pth (마지막 ep)

**평가**: 17-class master eval, n_per_class = 200 (모든 master chip), I3/I7 inference variants 둘 다.

**예상 결과** (직전 visible run 참고):
- 10-def macro F1 ≈ 0.30~0.40 (4-single 잘 잡지만 6-combo 거의 0)
- FAR ≈ 15~20% (Normal/OOD chip 일부를 4 obj 로 declare)

**의의**: 이 숫자가 "modern variant 가 얼마나 개선했는가" 의 기준선.

```bash
python -m chip_multilabel._train_chip_variant \
  --tag T0_pure_baseline_master_seed42 \
  --data-root D:/project/data/wm-811k/chip_multilabel \
  --include-classes bank_boundary,fork,scratch,scratch_rot \
  --n-per-class 200 \
  --no-normal \
  --loss ce_ls01 --ls 0.0 \
  --cutmix-p 0 \
  --epochs 8 --batch 8 --accum 4 \
  --lr-head 1e-4 --warmup 0 \
  --seed 42 --target-kind class_index
```

---

### Phase 1 — T1 = T0 + Label Smoothing 0.1

**Hypothesis**: LS=0.1 이 over-confidence 를 누르면 generalization 향상 → 10-def macro F1 ↑.

**T0 와의 단 한가지 차이**: `--ls 0.1` (T0 = 0.0).

**기법**:
- loss = `ce_ls01` with `--ls 0.1` (target [1,0,0,0] → [0.925, 0.025, 0.025, 0.025])
- 나머지 모두 T0 와 동일 (CutMix 여전히 OFF)

**예상**: T0 대비 macro F1 +0.01~0.05 정도. paper-typical LS 효과.

```bash
# T1 — T0 + LS 0.1 (CutMix still OFF)
python -m chip_multilabel._train_chip_variant \
  --tag T1_ls01_master_seed42 \
  ... (same as T0) ...
  --loss ce_ls01 --ls 0.1 \
  --cutmix-p 0
```

---

### Phase 2 — T3/T4/T5/T6/T7 (loss design ladder, CutMix=0.25 random)

이 단계에서 처음 CutMix 도입 (단순 random rectangle, 기존 코드 그대로). 사용자가 이전에 학습한 T3-T7 모델들이 이미 있음 (`outputs/T3_T3_master_seed42_*` 등) — **재사용**.

#### T3 — Softmax Focal (Lin 2017)
- **차이**: loss = `focal` (γ=2.0, no LS). target_kind = class_index.
- **기법**: softmax + focal down-weighting. 어려운 chip (낮은 p) 에 학습 집중.
- **CutMix**: `--cutmix-p 0.25 --cutmix-rect 0.5` 켜져있지만 target_kind=class_index → 코드 분기상 mosaic 자체 안 일어남 (= effectively OFF).
- **목적**: T1 (CE+LS) 와 비교해 focal 의 single-class 학습 측면 효과 측정.

#### T4 — Asymmetric Loss (Ridnik 2021)
- **차이**: loss = `asl` (γ_pos=1, γ_neg=4, clip=0.05). target_kind = multi_hot (sigmoid).
- **기법**: positive 는 BCE-like, negative 는 focal-asymmetric — over-fired negative 강하게 누름.
- **CutMix**: `--cutmix-p 0.25` 작동 (multi_hot 분기). label = OR rule [0,1,1,0].
- **목적**: ASL + CutMix 가 fork over-firing 등 FP 줄이는지 측정.

#### T5 — BCE multi-hot (no smoothing)
- **차이**: loss = `bce` (LS=0). multi-hot.
- **기법**: 가장 단순 multi-label loss. CutMix mosaic = OR label [0,1,1,0].
- **목적**: ASL 과 비교해 단순 BCE 만으로 어디까지 가는지.

#### T6 — BCE → ASL (5 ep warmup hybrid)
- **차이**: loss = `bce_then_asl` (warmup_epochs=5). 처음 5 ep BCE, 그 후 ASL.
- **기법**: BCE 로 안정적 multi-label 표현 학습 후 ASL 로 negative 누름.
- **목적**: 두 phase hybrid 의 효과 측정.

#### T7 — BCE + LS=0.20
- **차이**: loss = `bce_ls` (LS=0.20). multi-hot smoothed.
- **기법**: BCE target [1] → [0.9], [0] → [0.1] (symmetric BCE smoothing).
- **목적**: iter 8 winner T9 family 의 핵심 — over-confidence 완화 + multi-hot. 이미 검증된 winner candidate.

**모두 동일**:
- batch=8 accum=4, epochs=8 (T6/T7 일부 12)
- `--cutmix-p 0.25 --cutmix-rect 0.5` (random rectangle, OR/soft-λ label)
- `--include-classes 4-single`, `--no-normal`
- aug 동일

**평가**: 17-class master, I3/I7 둘 다.

**현재 상태 + 사용자 결정**:
- T3/T4/T5/T6 outputs 폴더 존재하지만 ★ **사용자 directive: 모두 재학습** (final_epoch_model.pth 정책 일관 적용 + 신규 T0/T1 과 동일 condition).
- T7 ✗ killed mid-train → 재학습.
- T1/T0 ✗ → 이번 plan 에서 신규 학습.
- 따라서 Phase 2 = T3, T4, T5, T6, T7 모두 fresh 재학습 (5 trains × ~9분 = ~45분).

---

### Phase 2.5 — ★ Post-hoc threshold optimization + Pareto frontier (사용자 directive 260506)

**Hypothesis (사용자)**: T1 의 24.5% bit-FAR 은 threshold 잘못 정한 것일 수 있음. LS 가 prob 평탄화시키니 negative chip 의 prob 분포가 right-shift → threshold 0.5 부적합. 더 큰 θ 로 cutoff 하면 bit-FAR ↓ 시키면서 macro F1 유지 가능성. 표준 calibration analysis.

**현재 inference 동작**: I3 = val split 에서 per-class **F1-max threshold** 학습 — F1 만 보고 정함. **bit-FAR 고려 안 됨**. trade-off 가 F1-only objective 기반.

**작업** (학습 X — inference + 분석만):

1. **val prob 분포 시각화** (각 model 별):
   - per-class positive/negative class 의 sigmoid prob histogram
   - LS 모델은 prob 분포가 우측 평탄화되었는지 시각 확인

2. **Threshold sweep + Pareto curve**:
   ```
   For each trained model:
       For each θ ∈ {0.1, 0.15, 0.20, ..., 0.95}:
           decision = (sigmoid(logits) > θ) per-class same
           17-class master eval → (macro F1, bit-FAR) 측정
       → Pareto curve (macro F1 vs bit-FAR)
   ```

3. **Composite optima** (각 모델 별):
   - argmax macro F1 (F1-only — 현재 I3 와 비교)
   - **argmax macro F1 s.t. bit-FAR ≤ 5%** ★ 운영 제약 winner
   - argmax (macro F1 − λ·bit-FAR), λ ∈ {1, 2, 5} — soft trade-off
   - argmax (macro F1 − bit-FAR) — equal weighting

4. **(옵션) 새 inference variant I14**: `FAR-constrained per-class threshold` — val 에서 `argmax F1 s.t. bit-FAR ≤ 5%` per-class. 결과 좋으면 표준 variant 로 promote.

**산출**:
- per-model `prob_hist.png`, `pareto_curve.png`
- 표: 모델별 [θ_F1max, θ_FAR≤5%, θ_balanced] 별 (macro F1, bit-FAR)
- 진단: T1 의 LS 부작용이 진짜 over-firing 인지 vs threshold 만 잘못 정한 것인지 분리

**Code**: standalone `chip_multilabel/_threshold_sweep.py` (read-only on stage1 parquet + model logits — 새 inference 안 함, val split prob 만 reuse).

**시간**: 8 model × 18 θ values × 5분 inference 안 함 — val prob cache 만 forward 1회 + threshold loop CPU only. 모델당 ~3분, 총 ~30분.

---

### Phase 3 — T9 Sigmoid Focal Loss (RetinaNet style, 새로 추가)

**Hypothesis**: BCE multi-hot + focal down-weighting (RetinaNet) 이 T5 (pure BCE) / T7 (BCE+LS) 보다 우수할 것. multi-label 표준 loss.

**기법**:
- loss = `sigmoid_focal` (이미 `losses.py` 에 추가됨, line 143-175)
- 공식: `L = -α (1-p)^γ log(p)` for positives + `-(1-α) p^γ log(1-p)` for negatives
- 기본 hparam: α=0.25, γ=2.0 (RetinaNet default)
- target_kind = multi_hot
- CutMix `--cutmix-p 0.25 --cutmix-rect 0.5` (Phase 2 와 동일 단순 random)

**목적**: T7 (BCE+LS) 와 head-to-head 비교 — focal vs LS 어느 게 multi-label 에서 우수한가.

```bash
python -m chip_multilabel._train_chip_variant \
  --tag T9_sigmoid_focal_master_seed42 \
  --include-classes bank_boundary,fork,scratch,scratch_rot \
  --n-per-class 200 --no-normal \
  --loss sigmoid_focal --gamma 2.0 --alpha 0.25 \
  --cutmix-p 0.25 --cutmix-rect 0.5 \
  --epochs 8 --batch 8 --accum 4 --lr-head 1e-4 \
  --seed 42 --target-kind multi_hot
```

---

### Phase 4 — ★ 새 CutMix 설계 + Ablation (이번 plan 의 핵심 contribution)

**Hypothesis**: 단순 random rectangle CutMix 는 두 가지 문제 가짐:
1. paste 영역이 B 의 background 일 수 있음 (defect 영역 보장 X) → mislabeled sample.
2. BCE target = [1, 1, 0, 0] OR rule 은 면적 비례 무시 → 작은 paste 도 큰 paste 도 동일 label = over-confident.

**해결책** (사용자 제안):
- **Scattered patches**: 한 큰 사각형 대신 N 개 작은 patch (예: N=5, 각 ~30×30) 흩뿌림. 총 area = `ratio` (예: 0.3).
- **Soft proportional BCE target**: `label_B = ratio × discount(0.7) × α`. discount=0.7 fixed (noise 보정), α sweepable.
  - 예: ratio=0.3, α=1.0 → label_B = 0.21.

#### 4.1 코드 patch (`_train_chip_variant.py`)

새 CLI flag 추가:
```
--cutmix-mode {single, scattered}   # default 'single' (기존 동작), 'scattered' = 새 모드
--cutmix-n-patches 5                # scattered 일 때만 적용
--cutmix-total-ratio 0.3            # 총 paste area (sweepable)
--cutmix-discount 0.7               # noise discount (fixed default 0.7)
--cutmix-alpha 1.0                  # additional scale (sweepable)
```

CutMix 코드 분기 (line 366-426 영역):
```python
if cutmix_mode == "scattered":
    total_area_target = cutmix_total_ratio
    patch_area_each = total_area_target / n_patches
    patch_side = int(round(sqrt(patch_area_each * H * W)))
    actual_total = 0.0
    for k in range(n_patches):
        cy_k = randint(0, H - patch_side)
        cx_k = randint(0, W - patch_side)
        x[bi, :, cy_k:cy_k+patch_side, cx_k:cx_k+patch_side] = \
          x[perm[bi], :, cy_k:cy_k+patch_side, cx_k:cx_k+patch_side]
        actual_total += patch_side * patch_side  # 겹침은 over-count 되지만 acceptable
    actual_ratio = min(1.0, actual_total / (H * W))
    if target_kind == "multi_hot":
        soft_label = actual_ratio * cutmix_discount * cutmix_alpha
        tgt[bi, b_class] = soft_label  # ★ proportional, OR 아님
        # tgt[bi, a_class] = 1.0 그대로 유지
    elif target_kind == "soft_multihot":
        # CE-soft: 정규화 위해 lam 으로 환산
        lam_b = actual_ratio * cutmix_discount * cutmix_alpha
        tgt[bi, a_class] = 1.0 - lam_b
        tgt[bi, b_class] = lam_b
```

**중요**: BCE 의 `binary_cross_entropy_with_logits` 는 continuous target [0, 1] 자연스럽게 처리. soft proportional target = legal.

#### 4.2 Ablation grid (★ 4×4 = 16 cells, 사용자 확정)

T9 Sigmoid Focal variant 위에서 (best loss design 가정) ratio × α sweep:

| ratio \ α | 0.5 | 0.75 | 1.0 | 1.5 |
|---|---|---|---|---|
| **0.1** | T9s_r01_a05 | T9s_r01_a075 | T9s_r01_a10 | T9s_r01_a15 |
| **0.2** | T9s_r02_a05 | T9s_r02_a075 | T9s_r02_a10 | T9s_r02_a15 |
| **0.3** | T9s_r03_a05 | T9s_r03_a075 | T9s_r03_a10 | T9s_r03_a15 |
| **0.4** | T9s_r04_a05 | T9s_r04_a075 | T9s_r04_a10 | T9s_r04_a15 |

각 cell:
- T9 sigmoid_focal 학습 + scattered CutMix + soft proportional label
- label_B = ratio × discount(0.7) × α
  - 예: ratio=0.3, α=0.75 → label_B = 0.158
  - 예: ratio=0.4, α=1.5 → label_B = 0.420 (max)
  - 예: ratio=0.1, α=0.5 → label_B = 0.035 (min)
- p=0.25 fixed (CutMix 적용 batch 비율)
- discount=0.7 fixed (사용자 명시)
- 16 trains × ~6분 = ~96 분 (sequential, ~1.5h) 또는 GPU 여유 시 2-way parallel ~48분

각 cell 결과:
- 10-def macro F1 (I7 inference)
- FAR

→ 2D heatmap (ratio × α) 작성. best (ratio*, α*) 식별.

#### 4.3 ★ T9 + T7 둘 다 sweep (사용자 확정)

Phase 4 의 4×4 = 16 cells 를 **T9 (Sigmoid Focal) 와 T7 (BCE+LS) 둘 다** 위에서 실행 → **총 32 trains**.

**의의**: CutMix 설계 (scattered + soft proportional) 의 loss-agnostic 효과 확인. focal 과 LS-smoothed BCE 두 family 에서 같은 (ratio*, α*) 가 winner 인지 / 다른지 검증.

**비용**: 32 trains × ~6분 = ~3시간 sequential. GPU 여유 시 2-way parallel ~1.5시간.

**산출 표** (예시):

| variant | ratio | α | 10-def macro F1 (I7) | FAR |
|---|---|---|---:|---:|
| T9-soft | 0.1 | 0.5 | ? | ? |
| T9-soft | ... | ... | ? | ? |
| T9-soft | 0.4 | 1.5 | ? | ? |
| T7-soft | 0.1 | 0.5 | ? | ? |
| T7-soft | ... | ... | ? | ? |
| T7-soft | 0.4 | 1.5 | ? | ? |

→ 두 family 의 best (ratio*, α*) 비교. paper 의 핵심 Figure 후보.

---

### Phase 5 — Ensemble (logit avg)

**Hypothesis** (memory rule `feedback_logit_ensemble_complementary.md`): complementary 약점 (with-Normal vs without-Normal) 가진 두 모델 logit 평균이 best single 모델 + threshold 트릭보다 큼. iter 10 실측: single 0.91~0.97 → ensemble 0.995.

**ensemble pairs**:
1. T9-soft-best (winner from Phase 4) + iter 10 C_44 (with-Normal training)
2. T9-soft-best + T7 (loss diversity)
3. T9-soft-best + T0 (range diversity)

**evaluation**: I7 inference, 10-def macro + FAR.

**5-sample-seed verification** (winner ensemble): seed ∈ {42, 43, 44, 45, 46} runtime sampling 으로 mean ± std 측정.

---

### Phase 6 — Final paper iter doc

**산출**:
- `docs/chip-multilabel/iters/iter_12_pure_baseline_to_soft_cutmix.md`
- `docs/chip-multilabel/02_results.md` cross-iter row append
- `chip_multilabel/notes.md` iter 12 entry

**핵심 표** (per-bit metric, ML-GCN/ASL 양식 차용):

| Phase | Model | CutMix mode | LS | F1_bb | F1_fork | F1_sc | F1_sr | **CF1 (macro)** ★ | OF1 (micro) | **bit-FAR** ★ | chip-FAR | 3plus% |
|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | T0 pure CE | OFF | 0 | ? | ? | ? | ? | ? | ? | ? | ? | ? |
| 1 | T1 CE+LS | OFF | 0.1 | ? | ? | ? | ? | ? | ? | ? | ? | ? |
| 2 | T3 Focal | random p=0.25 | 0 | ? | ? | ? | ? | ? | ? | ? | ? | ? |
| 2 | T4 ASL | random p=0.25 | — | ? | ? | ? | ? | ? | ? | ? | ? | ? |
| 2 | T5 BCE | random p=0.25 | 0 | ? | ? | ? | ? | ? | ? | ? | ? | ? |
| 2 | T6 BCE→ASL | random p=0.25 | — | ? | ? | ? | ? | ? | ? | ? | ? | ? |
| 2 | T7 BCE+LS | random p=0.25 | 0.20 | ? | ? | ? | ? | ? | ? | ? | ? | ? |
| 3 | T9 Sigmoid Focal | random p=0.25 | 0 | ? | ? | ? | ? | ? | ? | ? | ? | ? |
| 4 | T9-soft (best ratio×α) | scattered | 0 | ? | ? | ? | ? | ? | ? | ? | ? | ? |
| 5 | T9-soft + C_44 ensemble | — | — | ? | ? | ? | ? | ? | ? | ? | ? | ? |

★ CF1 (macro F1) = paper main, bit-FAR = real-env. 5 OOD class별 row 절대 없음 (memory rule).

---

## 4. Verification (각 step end-to-end 검증)

### 4.1 Phase 0 (T0) 학습 완료 후
- [ ] `outputs/T0_pure_baseline_master_seed42_<TS>/best_model.pth` 존재
- [ ] `train_summary.json` 의 `cutmix_p` = 0 확인 (명시적 OFF)
- [ ] val_acc curve 가 ep1 saturation 확인 (이전 지적 패턴)
- [ ] 17-class master eval → 10-def macro F1, FAR 둘 다 산출
- [ ] eval 시 5 OOD class 의 어떤 metric 도 표시 안 됨 확인

### 4.2 Phase 4 (scattered CutMix) 코드 patch 후
- [ ] 새 CutMix mode 적용 시 batch sample 한 개 시각화 (mosaic 결과 PNG dump) — scattered patches 보이는지
- [ ] target tensor 출력 확인: `[1.0, 0.21, 0, 0]` 같은 soft proportional label 인지
- [ ] `--cutmix-mode single` 로 호출 시 기존 동작 유지 (regression test)

### 4.3 Phase 5 ensemble
- [ ] logit avg 결과 single 모델 best 보다 macro F1 ↑ 또는 FAR ↓
- [ ] 5-seed std < 0.01 (안정성)

### 4.4 Final
- [ ] iter 12 doc 표 row 11 개 (Phase 0~5 합), OOD row 0 개
- [ ] memory rules 모두 준수 (TTA X, rotation aug X, OOD perf X, subset 폴더 X)

---

## 5. Critical files (수정 대상)

| file | 수정 내용 | Phase |
|---|---|---|
| `chip_multilabel/_train_chip_variant.py` | scattered CutMix mode + soft proportional label + 새 CLI flags | 4 |
| `chip_multilabel/notes.md` | iter 12 진행 entry append | 5,6 |
| `docs/chip-multilabel/iters/iter_12_pure_baseline_to_soft_cutmix.md` | 신규 paper iter doc | 6 |
| `docs/chip-multilabel/02_results.md` | cross-iter row append | 6 |
| ad-hoc metric script | 10-def macro + FAR 계산 + 표 생성 | 5 |
| `chip_multilabel/losses.py` | (이미 T9 SigmoidFocalLoss 추가됨, 추가 변경 X) | — |

**모든 variant 재학습** (사용자 directive): T0, T1, T3, T4, T5, T6, T7, T9 모두 fresh 8 epoch + final_epoch_model.pth 일관 정책. 기존 outputs/T*_master_seed42_* 폴더는 유지하되 (rule: 결과 폴더 삭제 금지) 새 TS 로 별도 폴더 생성.

---

## 6. Hard rules (carry-over, 모든 Phase 적용)

- ★ TTA 영구 금지 (I5)
- ★ Rotation/Flip aug 영구 금지 (학습 시 RandomAffine translate+scale 만)
- ★ outputs/ 무단 삭제 금지
- ★ subset/archive 폴더 금지 — runtime sampling only
- ★ batch=8 accum=4 (공유 GPU 에서 안전 한계)
- ★ OOD class (5 wafer-pattern) 의 어떤 성능 표도 작성 X (F1, prediction distribution, 어떤 진단도)
- ★ Normal F1, Invalid F1 도 표시 X (FAR 만)
- ★ paper main metric = 10-def macro F1 + FAR (단 2 개)
- ★ ≥3 active 시 top-2 truncate 절대 금지 — '3plus_active' decision_type 으로 그대로 declare (auto-wrong, 빈도 분석)
- ★ 추가 분석 후보 (Phase 4 sweep 후): 동일 chip 을 ratio·α 다르게 학습한 모델들의 softmax output 분포 비교 — 사용자 nice-to-have 260506.

---

## 7. 실행 순서 요약 (사용자 승인 후)

1. **Phase 0**: T0 학습 (~9분) → 17-class eval (~2분)
2. **Phase 1**: T1 학습 (~9분) → eval (~2분)
3. **Phase 2 (★ 5 trains 재학습)**: T3, T4, T5, T6, T7 모두 fresh 재학습 (~45분 sequential 또는 GPU 여유 시 parallel ~20분) + eval (~10분)
3.5 **Phase 2.5 ★ Threshold sweep + Pareto** (사용자 directive 260506): 모든 학습된 model 의 threshold sweep, calibration analysis (~30분)
4. **Phase 3**: T9 학습 (~9분) → eval (~2분)
5. **Phase 4 코드 patch**: scattered CutMix 구현 (~30분 코딩 + 단위검증)
6. **Phase 4 sweep (★ 32 trains)**: T9-soft 16 cell + T7-soft 16 cell = 32 trains (~3시간 sequential) + eval (~64분)
7. **Phase 5 ensemble**: 3 pair 평가 (~10분), 5-seed verification (~10분)
8. **Phase 6 doc**: iter 12 doc + 02_results.md + notes.md (~30분)

**총 예상**: ~7~8시간 sequential. Phase 4 sweep 이 대부분. GPU 여유 시 parallel 로 절반.

---

## 8. 사용자 승인 / 결정 (resolved)

- (Q1) ✅ Phase 4 ablation grid = 4×4 = 16 cells (ratio ∈ {0.1, 0.2, 0.3, 0.4} × α ∈ {0.5, 0.75, 1.0, 1.5}). 사용자 확정.
- (Q2) Phase 4 의 cutmix-p = 0.25 fixed (sweep axis 아님). 코드 단순성 + 시간 절약 (이미 32 trains).
- (Q3) ✅ Phase 4 sweep variant = T9 (Sigmoid Focal) **+ T7 (BCE+LS) 둘 다**. 32 trains. CutMix 설계의 loss-agnostic 효과 검증.
- (Q4) ✅ Phase 0/1 baseline = `--no-normal` (Normal 학습 X — 가장 순수 baseline).
- (Q5) ✅ Phase 2 T3-T6 = **모두 재학습** (final_epoch_model.pth 일관 정책).
- (Q6) Phase 5 ensemble pair 의 iter 10 model path 는 dispatch 시점에 outputs/ glob 으로 확인 후 결정.
