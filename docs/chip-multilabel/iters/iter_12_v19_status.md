# iter 12 — Pure baseline ladder + v19 강도 ↑ (260506 재부팅 직전 status)

## 1. iter 12 학습 결과 (Phase 0~3 + Phase 2.5, ★ chip_multilabel master 사용 = v18 chip)

### 핵심 metric (per-bit, paper standard, I3 inference)

| variant | loss | macro F1 | bit-FAR | chip-FAR | F1_bb | F1_fork | F1_sc | F1_sr | 3plus% |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| **T0** pure CE | ce_ls01 ls=0 | 0.7247 | **0.70%** | 2.80% | 0.82 | 0.43 | 0.66 | 0.99 | 0% |
| **T1** CE+LS=0.1 | ce_ls01 ls=0.1 | 0.7549 | 24.5% ❌ | 96% ❌ | 0.82 | 0.68 | 0.55 | 0.97 | 1.6% |
| **T3** Focal ★ | focal | **0.7974** | **1.77%** | 7.10% | 0.77 | 0.54 | 0.89 | 0.99 | 0% |
| **T4** ASL γn=4 | asl | 0.7158 | 6.48% | 25.8% | 0.77 | 0.39 | 0.70 | 1.00 | 0% |
| **T5** BCE | bce | 0.7173 | 24.0% | 96% | 0.91 | 0.40 | 0.56 | 0.99 | 0% |
| **T6** BCE→ASL | bce_then_asl | 0.7173 | 24.0% | 96% | 0.91 | 0.40 | 0.56 | 0.99 | 0% |
| **T7** BCE+LS=0.20 | bce_ls 0.20 | 0.7129 | 41.7% ❌❌ | 96% | 0.81 | 0.43 | 0.61 | 1.00 | 6.4% |
| **T9** Sigmoid Focal | sigmoid_focal | 0.7637 | 29.7% | 96% | 0.94 | 0.44 | 0.70 | 0.98 | **7.92%** |

### Phase 2.5 — threshold sweep findings (사용자 가설 검증)

- **per-class F1-max threshold (I3) 의 함정**: fork prob 분포 평탄 → 낮은 threshold → over-fire
- **T1/T7 (LS naive) 어떤 θ 로도 못 살림** (LS 가 prob 분포 평탄화) — paper-grade negative finding
- **★ T9 + global θ=0.50** → macro 0.7513 / **bit-FAR 4.23%** ★
- **★ T9 + global θ=0.60** → macro 0.7154 / **bit-FAR 0.57%** ★ (T0 의 0.70% 보다 낮음 + multi-label 학습 능력)
- T4/T5/T6 도 global θ 조정 → bit-FAR 24% → 3% 회복 가능

## 2. paper-grade findings 누적

1. **T0→T1 (LS naive)**: macro F1 +0.030, **bit-FAR +35× 악화** (over-firing trade-off)
2. **T1→T3 (Focal)**: macro F1 +0.042 + bit-FAR -93% 동시 개선 (Focal calibration ≠ LS)
3. **T6 BCE→ASL hybrid 효과 없음** — best_epoch=1 BCE phase 안에서 saturate, ASL transition 안 함 = T5 동일
4. **T4 ASL default γ_neg=4 너무 aggressive** — fork precision 1.0 / recall 0.24
5. **T5 BCE no LS 도 over-firing** — naive BCE saturation. fork F1 = 0.40 (T4 와 동일)
6. **T7 BCE+LS=0.20 첫 multi-active (3plus 6.4%)** — but LS 강함이 over-firing 동반
7. **★ T3 Focal winner** (macro 0.80, bit-FAR 1.77%, single-class) — but fork F1 0.54 천장
8. **★ fork master chip 의 defect_pixel_ratio = 0.069 (다른 obj 의 절반)** — fork F1 0.43-0.68 천장의 본질적 원인. v19 강도 ↑ 필요

## 3. v19 코드 변경 (강도 ↑ + scratch_rot 우상향)

### CPU `_sample_gen.py` (이미 사용자가 수정)

| obj | weak severity | smear_factor |
|---|---|---|
| **fork** | 0.45-0.55 → **0.70-0.85** | 1.5-2.5 → **5.0-8.0** |
| **scratch** | 0.45-0.55 → **0.85-0.95** | 2.5-4.5 → **18-30** |
| **scratch_rot** | 0.45-0.55 → **0.78-0.90** | 1.5-2.5 → **8-13** |

추가:
- `INTENSITY_ALPHA_SCALE['weak']`: 0.40 → 0.60
- `CHIP_OBJ_PER_CLASS_CAP`: 100 → 200
- `MIN_CHIP_DEFECT_RATIO`: 0.03 → 0.10 (strict)
- `MIN_CHIP_STRONG_GRADE_RATIO`: 0.02 신규 (grade ≥3 picture 만)

### GPU `_sample_gen_gpu.py` (오늘 추가 fix)

이전: 3 obj 모두 `alpha_scratch_t` 공유 (round 26 placeholder)
v19: 별도 함수 분리 + 동일 hierarchy:
- `alpha_fork_t` — cross pattern (1 horizontal + 4-6 vertical legs), weakest tier
- `alpha_scratch_t` — vertical lines 8-17개 (denser 5-16), strongest tier  
- `alpha_scratch_rot_t` — **theta = -21°** rotated (top tilts RIGHT), middle tier

`ALPHA_FNS_T` mapping 갱신:
```python
'fork':        alpha_fork_t,
'scratch':     alpha_scratch_t,
'scratch_rot': alpha_scratch_rot_t,
```

### scratch_rot angle 수학적 검증

theta = -21°:
- cos_t = 0.9336, sin_t = -0.3584
- slope (dy/dx in image space) = cos_t/sin_t = **-2.605**
- slope < 0 in image space (Y-down) → X 증가시 Y 감소 → 위가 오른쪽
- ★ **TOP TILTS RIGHT** — 사용자 directive "수직선 기준으로 오른쪽으로만" 일치

## 4. chip 새로 만들기 — 미완 (재부팅 후 진행)

**시도**:
1. `bnam0jos1` (GPU): 471/8450 진행 후 kill (stale module)
2. `b267rzicm` (GPU v2): 366/8450 진행 후 kill
3. 모든 python.exe kill

**현재 `classification_chips/` 상태**:
- bank_boundary 0, fork 200, scratch 200, scratch_rot 136, invalid_main 200
- ⚠️ 혼합 chip (fix 전후 섞임) — 다음 세션 시작 시 모두 비우고 fresh 합성 필요

**chip 검증 결과 (재부팅 전)**:
- fork defect_ratio = 12.86% (기존 6.9% → 2× ↑) ★ **v19 강도 적용 확인**
- scratch_rot angle slope -0.06~-0.16 (near vertical, chip area 작아 -21° 시각 검증 어려움)

## 5. 다음 세션 시작 시 first action (재부팅 후)

```bash
# Step 0: classification_chips 5 obj 비우기
cd D:/project/known-cnn
for d in bank_boundary fork scratch scratch_rot invalid_main; do
  rm -f D:/project/data/wm-811k/classification_chips/$d/*.png
done
# stale .pyc 삭제 (이미 됐지만 안전)
rm -f dist_apply/__pycache__/_sample_gen*.pyc

# Step 1: v19 GPU wafer 합성
cd dist_apply
python _sample_gen_gpu.py --n 200 --save-workers 8 2>&1 | tee ../outputs/v19_wafer_gen.log
# ETA ~25-50분, 8450 task

# Step 2: chip 검증 (fork defect ≥10%, scratch_rot angle 시각)

# Step 3: chip_multilabel master 재생성
python -m chip_multilabel.gen_eval_set \
  --out-root D:/project/data/wm-811k/chip_multilabel \
  --classification-chips-root D:/project/data/wm-811k/classification_chips \
  --per-defect 200 --per-normal 200 --per-invalid 50 \
  --source-strength-pct 70

# Step 4: iter 12 학습 재실행 (T0~T9, ~50분 sequential)
```

## 6. Hard rules carry-over

- ★ pseudo-label code 영구 제거 (불량 폴더에 양호 sample 저장 차단)
- ★ STUB_DIRS / particle_blast / scratch_21deg 코드 reference 영구 제거
- ★ chip quality filter strict (0.10 + 0.02 strong)
- ★ scratch_rot angle = -21° (top tilts RIGHT) — 사용자 directive
- ★ TTA / Rotation aug 영구 금지
- ★ 5 OOD class 의 어떤 metric 도 측정 X
- ★ ≥3 active top-2 truncate 폐기 (3plus_active decision_type)
- ★ batch=8 accum=4 chip 학습 안전
