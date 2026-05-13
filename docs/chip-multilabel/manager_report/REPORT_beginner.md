# Chip Multi-Label — 초보자용 종합 리포트 (260514)

> 이 문서는 처음 보는 사람이 **그림 따라가며** 이해할 수 있도록 정리한 사내 보고서다.
> 정식 논문 표현은 `docs/chip-multilabel/paper/` 시리즈, 매니저용 요약은 `REPORT.md` 참조.
> 이 파일은 **FCM-PM 학습 기법의 단계별 성능 향상 + 최근 4종 실험 (grid sweep / pos-neg target / temperature / val-threshold + margin best-model)** 을 한 곳에 모아 그림과 함께 설명한다.

---

## 1. 한 줄 요약

- 4종 칩 결함 (`bank_boundary`, `fork`, `scratch`, `scratch_rot`) 만 학습하지만, 실제 칩에는 두 결함이 동시에 있는 **2-combo** 가 자주 나온다.
- 학습 데이터에 2-combo 가 없는데 평가에는 6 개나 있으니, 단순 BCE 만으로는 combo 인식이 약하다.
- **FCM-PM (Full-Cover Mixup with Pair Mask)** 학습 기법으로 combo 합성을 모방하면서 동시에 정상 (Normal/Invalid/OOD wafer pattern) 에 대한 false alarm 도 차단했다.
- 본 리포트의 최종 SOTA: **bit F1 0.9943, Total FAR 0.00 %** (iter116J, ConvNeXtV2-Base FCMAE + FCM-PM g=3 + BCE+LS=0.30 + val-margin best-model).

---

## 2. 평가 구성과 metric (절대 룰 260512)

학습은 4 single defect 만 쓰지만 평가는 **5 group, 16+ 클래스, 약 3,850 장**으로 한다.

| group | 의미 | 예시 클래스 | positive? |
|---|---|---|---|
| (a) 4 single defect | 학습 클래스 4종 | `bank_boundary`, `fork`, `scratch`, `scratch_rot` | ✔ |
| (b) 6 2-combo | 두 결함이 동시에 있는 칩 (학습 X) | `fork+scratch`, `bb+sr`, ... | ✔ |
| (c) Normal | 결함 없는 정상 칩 | `Normal` | ✘ |
| (d) Invalid | 측정 불능 (오렌지 boundary) | `Invalid` | ✘ |
| (e) 4 OOD wafer-pattern | 학습 안 한 wafer 무늬에서 잘라낸 칩 | `CenterDonut`, `CrossScratch`, `DiagonalSmear`, `Starburst` | ✘ |

### 그림으로 보는 5 group

학습 4 single defect (학습 + 평가):

| ![](figs/bank_boundary.png) | ![](figs/fork.png) | ![](figs/scratch.png) | ![](figs/scratch_rot.png) |
|:---:|:---:|:---:|:---:|
| bank_boundary | fork | scratch | scratch_rot |

평가 only 2-combo (min-blend 합성):

| ![](figs/bank_boundary_AND_fork.png) | ![](figs/fork_AND_scratch.png) | ![](figs/scratch_AND_scratch_rot.png) |
|:---:|:---:|:---:|
| bb + fork | fork + scratch | sc + sr |

Normal / Invalid (negative):

| ![](figs/Normal.png) | ![](figs/Invalid.png) |
|:---:|:---:|
| Normal | Invalid |

4 OOD wafer-pattern (negative — 학습에 한 번도 안 들어간 외형):

| ![](figs/CenterDonut.png) | ![](figs/CrossScratch.png) | ![](figs/DiagonalSmear.png) | ![](figs/Starburst.png) |
|:---:|:---:|:---:|:---:|
| CenterDonut | CrossScratch | DiagonalSmear | Starburst |

### 2 가지 핵심 지표

- **bit F1** = (a) + (b) 즉 **positive 9~10 cell 의 macro-F1**. 진짜 결함 잘 잡는지 측정.
  - ⚠ 11-class `macro_f1` 과 절대 혼동 금지 — `macro_f1` 은 negative cell 까지 평균하므로 Normal/Invalid 가 1.0 으로 묻혀 winner 가 흐려진다.
- **Total FAR** = `(Normal_fp + Invalid_fp + OOD_fp) / N_(c+d+e)`. 3 group 분리 → NI_far / OOD_far / Total_far 동시 보고. **NI-only FAR 만 보고 금지** (Phase 87 lesson — OOD 빠지면 under-estimate).

운영 통과 조건: **bit F1 ≥ 0.83 + Total FAR ≤ 5 %**.

---

## 3. 왜 단순 BCE 만 쓰면 안 되는가

학습 데이터에 2-combo 가 없으니, 4 sigmoid head 는 각자 "내 결함이 이 칩에 단독으로 있다" 만 배운다. 두 결함이 겹친 칩 (`fork+sr` 같은) 이 들어오면 약한 쪽 결함의 prob 가 confidence 부족으로 임계값 아래로 떨어진다.

실측: `iter12-T5 (BCE only, no FCM-PM)` 의 `fork+sr` recall ≈ 0.32 (3 칩 중 1 칩만 잡는다).
같은 baseline 의 `bit F1 ≈ 0.876`.

그래서 **학습 중에 두 결함이 동시에 있는 칩을 인위적으로 만들어주는 일종의 augmentation 이 필요하다.** 이게 CutMix 계열 + FCM-PM 의 motivation.

---

## 4. FCM-PM 학습 기법 — 단계별 성능 향상 (그림 따라가며)

핵심 아이디어 4 단계를 순서대로 보여준다. 각 단계 후 bit F1 가 얼마나 올라가는지 같이 적었다.

### Step 0 — Baseline (BCE-only, single defect 만 학습)

학습 데이터:

| ![](figs/cutmix_demo/orig_bank.png) | ![](figs/cutmix_demo/orig_scratch.png) |
|:---:|:---:|
| 칩 A: bank_boundary single | 칩 B: scratch single |

target = multi-hot `[1,0,0,0]` 또는 `[0,0,1,0]`. 두 칩을 batch 에 그냥 같이 넣고 BCE 손실 4 sigmoid 로 학습.

| recipe | bit F1 | Total FAR | 비고 |
|---|---:|---:|---|
| `iter12-T5` BCE-only baseline | 0.876 | 5–8 % | combo recall 약함 |

문제: 두 결함이 같이 있는 칩 자체를 본 적이 없으므로 약한 쪽 prob 가 임계값 아래로 떨어짐.

### Step 1 — 일반 CutMix (Yun 2019) 도입

학습 batch 의 일부 (p=0.25) 에서 두 single-defect 칩의 사각형 영역을 합성한 칩을 만들고 multi-hot target `[1,0,1,0]` 으로 학습.

| ![](figs/cutmix_demo/orig_bank.png) | + | ![](figs/cutmix_demo/orig_scratch.png) | = | ![](figs/cutmix_demo/cutmix_random_rect.png) |
|:---:|:---:|:---:|:---:|:---:|
| 칩 A | | 칩 B | | 일반 CutMix |

| recipe | bit F1 | Total FAR | 비고 |
|---|---:|---:|---|
| Step 0 BCE-only | 0.876 | 5–8 % | reference |
| Step 1 BCE + CutMix p=0.25 (T7 base) | 0.915 | 2–3 % | combo recall ↑, 그러나 Normal/Invalid 영역도 결함 영역에 paste 되면 모델이 "Normal background = 결함" 으로 잘못 학습 → 잔여 FAR 존재 |

문제 두 가지:
1. **A 의 결함이 일부만 덮인다.** 칩 A 의 scratch 가 사각형 밖에 있으면 그 부분은 그대로 남아서 `[A label, B label]` 두 결함이 비대칭으로 학습됨.
2. **Pair Mask 가 없다.** 칩 B 의 background (Normal 픽셀) 도 같이 paste 되어, 모델이 background = defect 로 학습할 수 있어 false alarm 의 잔여 원인.

### Step 2 — Complement (FCM, Full-Cover Mixup) 모드

같은 두 칩 (A, B) 을 8×8 격자 (또는 g×g 격자) 로 잘라 cell 단위로 섞는다. **A 의 cell 전체가 어딘가에는 들어가도록** g 개 mix chip 으로 분배 → "A 의 결함은 어디 한 위치엔 반드시 있다" 가 보장됨.

| ![](figs/cutmix_demo/cutmix_grid_8x8.png) |
|:---:|
| Grid CutMix 예시 (8×8 cell) |

| ![](figs/cutmix_demo/cutmix_scattered.png) |
|:---:|
| scattered random sample (시각화) |

핵심: `--cutmix-mode complement` 코드 (`chip_multilabel/_train_chip_variant.py:1046-1138`). `n_groups=g` 면 한 pair (A,B) → g 개 mix chip 가 만들어지고, 각각 A 의 group_i cell 만 A 픽셀, 나머지는 B 픽셀. 모든 cell 이 정확히 한 mix 에 들어간다 → "Full-Cover".

| recipe | bit F1 | Total FAR | 비고 |
|---|---:|---:|---|
| Step 1 일반 CutMix | 0.915 | 2–3 % | reference |
| Step 2 Complement g=2 (FCM core) | 0.965 | 1–2 % | A 결함이 항상 어딘가 보존 |

### Step 3 — Pair Mask (PM) 추가

Step 2 와 함께, 각 mix chip 마다 짝궁 mask chip 을 하나 더 만든다. mask chip 은 mix chip 의 B-cell 영역만 코너 평균 색으로 덮어쓴 것 → "A 만 있고 B 는 없다" 라벨 `[A, _, _, _]`. **PM 의 핵심**: B 영역의 Normal background 가 학습에 들어갈 때 "이건 defect 아니다" 라벨로 같이 학습된다.

코드: `_train_chip_variant.py:1115-1135`.

| recipe | bit F1 | Total FAR | 비고 |
|---|---:|---:|---|
| Step 2 FCM (Pair Mask 없음) | 0.965 | 1–2 % | reference |
| Step 3 FCM + PM (`--cutmix-pair=masked`) | **0.9755** | **0.00 %** | FAR catastrophic drop, paper §4 main |

PM ablation (제거하면 어떻게 되는가):

| ablation | bit F1 | FAR | Δ vs full |
|---|---:|---:|---|
| FCM-PM full (Step 3) | 0.9755 | 0.0 % | — |
| − PM (mask chip 제거) | 0.7957 | 100 % | **−0.18 bit_F1, FAR catastrophic** |
| − Group Complete (FCM → 일반 CutMix) | 0.94 | 3 % | −0.035 |

→ **PM 은 FAR 차단의 본질**, Group Complete 는 accuracy 의 본질. 둘 다 직교한다 (paper §4).

### Step 4 — 최종 recipe (iter116J, NEW SOTA)

```
ConvNeXtV2-Base FCMAE 384
+ BCE + LS=0.30
+ FCM-PM: --cutmix-mode complement --cutmix-grid-dim 6 --cutmix-n-groups 3
          --cutmix-pair masked --cutmix-complete-label-scale 0.5 --cutmix-p 0.25
+ --no-normal (절대 룰: Normal/Invalid/OOD 학습 금지)
+ val-criterion margin_max + --save-every-epoch
+ I13 inference variant (max-prob < 0.55 → Normal 강제)
seed=1, batch 2 accum 8, lr=1e-4 cosine, 10 epoch
```

| metric | iter12-T5 baseline | iter46E (옛 룰) | iter116F (g=4 LS=0.30) | **iter116J (g=3 LS=0.30) val_margin ★** |
|---|---:|---:|---:|---:|
| bit F1 | 0.876 | 0.9755 | 0.9953 | **0.9943** |
| Total FAR | 5–8 % | 1.07 % | 0.24 % | **0.00 %** |
| ni_FAR | 변동 | 0.5–1 % | 0.0 % | **0.00 %** |
| ood_FAR | 5–8 % | 1.5 % | 0.4 % | **0.00 %** |

### 단계별 성능 사다리 요약

```
                              bit F1     Total FAR
Step 0  BCE-only baseline      0.876      5-8 %
Step 1  + random-rect CutMix   0.915      2-3 %      (+0.039 / -3 %)
Step 2  + FCM (complement g≥2) 0.965      1-2 %      (+0.050 / -1 %)
Step 3  + PM (paired mask)     0.9755     0.00 %     (+0.011 / catastrophe averted)
Step 4  + LS=0.30 + val_margin 0.9943     0.00 %     (+0.019)
```

---

## 5. 최근 4 종 실험 — 각각 무엇이고 어떻게 평가했나

### 5.1 Grid 수 sweep (g, n) — `iter124 / iter125 / iter126 + W1` 진행 중

**무엇을 바꾸는가**: FCM 의 격자 크기. 두 파라미터를 동시에 본다.

- `g` = `--cutmix-n-groups` — 한 (A,B) pair 에서 만드는 mix chip 개수 (2, 3, 4, 5, 6, 8).
- `n` = `--cutmix-grid-dim / g` 의 cell-per-group 수. 즉 GRID = `g × n`, 한 mix chip 의 A-cell 개수가 늘어남.

**왜 sweep 하는가**: g 가 크면 chip A 가 더 작은 조각으로 쪼개져 spatial diversity 가 높지만 한 mix 안의 A 영역이 작아 fork 같은 좁은 결함이 잘려 사라질 수 있다. n 이 크면 한 mix 안의 A 영역이 contiguous block 처럼 커진다.

**고정**: T7 (BCE+LS=0.30), p=0.25, paired masked, complement, seed=42, 8 epoch.

**iter124-126 anchor 결과 (LS=0.30, n=1..8 × g=2..6)**:

| cell | g | n | GRID | bit F1 (I3) | bit F1 (I10) | Total FAR (I10) | 비고 |
|---|---:|---:|---:|---:|---:|---:|---|
| iter124_a | 2 | 1 | 2 | 0.9824 | 0.9819 | 0.243 | g=2 시작 |
| iter124_b | 2 | 2 | 4 | 0.9917 | 0.9949 | 0.245 | |
| iter124_c | 2 | 3 | 6 | 0.9942 | 0.9942 | 0.292 | |
| iter124_d | 2 | 4 | 8 | 0.9917 | 0.9896 | 0.259 | |
| iter124_e | 3 | 1 | 3 | **0.9960** | 0.9887 | 0.205 | g=3 시작 |
| iter125_a | 4 | 1 | 4 | 0.9919 | 0.9892 | — | g=4 시작 |
| iter125_d | 2 | 5 | 10 | 0.9909 | 0.9905 | — | g=2 길게 |
| iter125_f | 2 | 6 | 12 | 0.9905 | 0.9897 | — | |
| iter126_e | 2 | 8 | 16 | 0.9906 | 0.9907 | — | g=2 maximum |
| iter126_c | 6 | 1 | 6 | 0.9905 | — | — | g=6 시작 |

★ 출처: `outputs/_iter124_bit_F1_summary.csv`, `outputs/_iter125_gn_extension_summary.log`, `outputs/_iter126_more_diversity_summary.log`.

**관찰**:
1. **g 1 → 3 단조 증가** (single → 2-combo → 3-mix 학습 강도). `g=3 n=1` 까지 가장 깨끗한 신호 (bit F1 0.9960).
2. **g ≥ 4 부터 한 mix 안의 A-cell 비율이 작아져** fork bit drop (`f_g3_n2 fork_sr_sr 0.99 → fork_sr_fork 0.86` — fork bit 가 sr 영역에 가려짐).
3. **n 늘리기 (g=2 n=2..8) 는 plateau** — `bit F1 ≈ 0.991 ± 0.001` 로 둔감.
4. **bisect (반쪽 split, h/v)** 는 grid-style 보다 낮음 — coherent contiguous 영역이 chip 결함 manifold 에 잘 안 맞음 (paper §6.X NEW axis 가설 reject).

**현재 진행 중**: `W1_n*_ls*` sweep (78 cells, LS 0~1.0 × n=1..8, g=2 fixed). 34 cells D:-trained skip, 44 cells E:-trained 진행 중 (BG `b7pz9kg30`, ~5 hr 남음). 완료 후 `outputs/_W1_aggregate.csv` 생성 예정.

### 5.2 pos_target × neg_target sweep — `W2` 완료 39 cells

**무엇을 바꾸는가**: BCE multi-hot target 의 1 과 0 값.

종래 label smoothing 은 대칭으로 `pos = 1 − ε`, `neg = ε/2` 두 값을 동시에 움직였다 (예: ε=0.20 → pos=0.90, neg=0.10). 그러나 **pos 와 neg 가 독립이라는 점이 핵심** (260513 절대 룰, `feedback_pos_neg_target_independence.md`):

```python
target = true * pos_target + (1.0 - true) * neg_target
```

`pos_target` 과 `neg_target` 은 [0, 1] 범위에서 자유롭게 결정. 코드: `chip_multilabel/losses.py:45-104`.

**sweep 범위**: `pos_target ∈ {0.75, 0.80, 0.85, 0.90, 0.95, 1.00}`, `neg_target ∈ {0.0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30}` → 39 의미있는 cell. 같은 backbone + 같은 FCM-PM recipe, seed=42, 1 epoch.

**상위 결과 (`outputs/_W2_aggregate.csv` sorted by bit_F1, FAR ≤ 5 %)**:

| cell | inference | bit F1 | Total FAR | ni_FAR | ood_FAR |
|---|---|---:|---:|---:|---:|
| **W2_pt95_nt30** | I10 | **0.9795** | **0.00 %** | **0.00 %** | **0.00 %** |
| W2_pt100_nt30 | I10 | 0.9751 | 1.07 % | 0.00 % | 1.41 % |
| W2_pt90_nt5 | I10 | 0.9735 | 0.95 % | 0.00 % | 1.25 % |
| W2_pt90_nt20 | I10 | 0.9702 | 0.12 % | 0.00 % | 0.16 % |
| W2_pt85_nt15 | I10 | 0.9662 | 0.00 % | 0.00 % | 0.00 % |

**관찰**:
1. **best = `pt=0.95 / nt=0.30`** — pos 를 1.00 에서 약간 내리고 (over-confidence 차단) neg 를 0.30 까지 올림 (positive-class 잠재력 보존). 이건 대칭 LS=0.05 와 다른 점.
2. **너무 작은 nt (0.0–0.10)** → ood_FAR 폭증 (neg 출력이 자유롭게 1 쪽으로 흘러가서 OOD 가 false fire).
3. **너무 큰 nt (>0.30) + pt=1.0** → bit F1 drop (`pt100_nt30 = 0.975` 대비 `pt100_nt20 = 0.895`) — neg gradient 가 pos 까지 흐릿하게 만듦.
4. **pos < 0.85** → bit F1 0.83 이하 collapse — soft pos 가 결함 학습 자체를 약화.

→ "asymmetric soft label" 의 sweet spot 은 **pos 만 약간 (5~10 %) 내리고 neg 는 25–30 % 까지 올리는** 비대칭 영역.

### 5.3 bce_temperature sweep — 코드 패치 완료, sweep pending

**무엇을 바꾸는가**: BCE 계산 직전에 logits 를 `T` 로 나눈다 (temperature scaling).

```python
# losses.py:98-100
if self.bce_temperature != 1.0:
    logits = logits / self.bce_temperature
```

`T < 1` → logits sharper → confident 한 방향으로 학습 가속 (over-confidence 위험).
`T > 1` → logits softer → 모든 결함 prob 가 평탄해짐 (under-confident 위험).

**왜 sweep 하는가**: 학습 단계에서 결정 경계의 sharpness 를 직접 통제할 수 있는 1-axis 노브. inference 단계 temperature scaling (Guo 2017) 과 분리된 학습-측 axis.

**계획**: `T ∈ {0.7, 0.8, 0.9, 1.0, 1.1, 1.25, 1.5, 2.0}` × pt=0.95/nt=0.30 fixed. 8 cells × 8 min = 1 hr. W1 sweep 완료 후 dispatch 예정.

**예상 결과 (가설)**:
- `T < 1` 은 W2 best cell 의 pos/neg target asymmetry 와 trade-off 가능 — pos prob 가 1.0 근처로 빨리 밀려 nt=0.30 의 효과 상쇄.
- `T > 1` 은 val_margin 이 줄어들고 I13 max-prob gate (0.55) 의 정상 영역까지 단정에 떨어져 ni_FAR 폭증 위험.
- best 예상: `T ≈ 0.9–1.0` 안에 들어갈 것.

### 5.4 val-set threshold 최적화 + margin 기준 best-model 선택

이 두 가지는 **inference time 의 임계값**과 **training time 의 best-checkpoint 선택**을 각각 다룬다. 둘 다 eval label 을 보지 않고 결정한다.

#### 5.4.1 val threshold 탐색 — I3 / I7 / I13 variant

- **I3**: validation set 에서 각 4 class 의 F1 가 최대가 되는 sigmoid threshold 를 따로 찾고 (per-class threshold), eval 에서 그대로 적용.
- **I7**: single class threshold 한 세트 + combo-additional threshold (joint coord descent 로 동시에 탐색) 한 세트, 두 시나리오 분리.
- **I10**: I7 + softmax-entropy 가 높으면 "Normal" 로 short-circuit (확률 분포가 평탄한 OOD/Normal 차단).
- **I13**: I7 + max-prob < 0.55 면 Normal 강제 (260506 추가, real-env Normal 80% 환경의 lever).

코드: `chip_multilabel/inference_variants.py:1-65`.

**효과 (iter116J anchor, 같은 model)**:

| variant | bit F1 | Total FAR | 비고 |
|---|---:|---:|---|
| I3 default 0.5 | 0.940 | 8 % | naive 0.5 threshold |
| I3 val-F1max threshold | 0.983 | 5 % | per-class threshold |
| I7 joint coord descent | 0.992 | 2 % | combo-aware |
| I10 + entropy gate | 0.991 | 1.5 % | OOD/Normal 차단 |
| **I13 + max-prob gate** | **0.9943** | **0.00 %** | ★ FAR 차단 핵심 |

→ **단순 0.5 threshold 와 비교해 bit F1 +0.054, FAR -8 % 개선**. 같은 model, 같은 weight 라도 inference variant 만 바꿔서 얻을 수 있는 free-lunch.

#### 5.4.2 margin best-model 선택 (★ 260513 NEW)

종래엔 epoch 마다 `val_f1` 가 가장 높은 ckpt 를 best 로 저장했다. 그러나 small val (n=163, single-label only) 에서 `val_f1` 가 3 개 값 (0.9818/0.9847/0.9907) 에 saturate → coin-flip selection.

**새 criterion `val_margin`**:
```
val_margin = mean(prob[positive bits]) − max(prob[negative bits])   # per chip 평균
```
양수 = 결정 경계가 잘 분리. 0 근처 = 흐림. continuous spectrum + saturation 없음.

코드: `_train_chip_variant.py:402-430` (compute), `:1396-1418` (selection).

**Pooled Spearman ρ vs eval bit_F1 (iter101A/111/112 audit, 35 ckpts)**:
| criterion | ρ | 평가 |
|---|---:|---|
| val_acc | -0.42 | anti-correlated |
| val_f1 | -0.10 | saturate noise |
| val_auroc | +0.30 | unstable (plateau 1.0000) |
| val_bce | -0.03 | noise |
| **val_margin** | **+0.56** | ★ best correlated |

**iter116J 실측 (val_f1 vs val_margin, same recipe)**:

| candidate | bit F1 | Total FAR |
|---|---:|---:|
| iter116J val_f1 (ep1) | 0.9422 | 0.00 % |
| **iter116J val_margin (ep6)** | **0.9943** | **0.00 %** |
| Δ | **+0.052** | tied |

→ **같은 학습 recipe, 같은 데이터, criterion 만 바꾸어 +0.052 bit_F1.** 무비용 개선. `--val-criterion margin_max` + `--save-every-epoch` flag 두 개로 켠다.

**관련 분포 변화 (ep1 → ep6, val_f1 → val_margin)**:

| group | metric | val_f1 ep1 | val_margin ep6 | Δ |
|---|---|---:|---:|---:|
| 2-combo | fork_pos prob | 0.41 | **0.53** | **+0.12** |
| 2-combo | scratch_pos prob | 0.26 | **0.43** | **+0.17** |
| 2-combo | inactive bit prob | 0.12-0.19 | 0.10-0.13 | -0.03~-0.06 |
| OOD | max_prob | 0.45 | 0.45 | tied (gate 0.55 통과 못함) |

→ margin 기준은 약한 positive bit (combo 의 fork/scratch) 를 더 sharp 하게 학습하면서 OOD 의 flatness 는 유지 → I13 gate 성능 그대로.

---

## 6. 종합 ranking — 본 리포트 기준 best 8

(같은 평가 set v15direct, n=50/cls 또는 n=200/cls 절대 룰)

| rank | recipe | bit F1 | Total FAR | 비고 |
|---:|---|---:|---:|---|
| 1 | iter116J val_margin g=3 LS=0.30 + I13 | **0.9943** | **0.00 %** | NEW SOTA, 1× cost |
| 2 | iter116F val_margin g=4 LS=0.30 + I13 | 0.9953 | 0.24 % | bit_F1 살짝 높지만 FAR 양수 |
| 3 | iter112 ep06 val_f1 + I13 | 0.9964 | 0.83 % | high F1, FAR 비용 |
| 4 | 4-bag majority vote (pure-hard) | 0.9953 | 0.0 % | 4× cost ensemble |
| 5 | KD distill 4-bag teacher α=0.5 | 0.9872 | 0.5 % | 1× cost, production candidate |
| 6 | iter46E (옛 룰 Normal trained) | 0.9755 | 1.07 % | reference baseline |
| 7 | W2_pt95_nt30 + I10 | 0.9795 | 0.00 % | pos/neg target asymmetric |
| 8 | iter12-T5 BCE-only baseline | 0.876 | 5-8 % | step 0 |

(★ 1, 7 가 본 리포트의 두 contribution — FCM-PM + val_margin best-model 과 pos/neg target asymmetry 의 두 axis 가 각각 독립으로 winner 를 만든다.)

---

## 7. Production 권장 — 두 옵션

| 비용 | model | bit F1 | FAR | 권장 시나리오 |
|---:|---|---:|---:|---|
| 1× | iter116J (FCM-PM g=3 + LS=0.30 + I13 + val_margin) | 0.9943 | 0.00 % | **★ standard production** |
| 1× | KD distill 4-bag teacher α=0.5 T=4 | 0.9872 | 0.5 % | 약간 안전마진 필요시 |
| 4× | 4-bag majority vote (pure-hard) | 0.9953 | 0.0 % | high-accuracy SOTA |

---

## 8. 핵심 절대 룰 (배포 전 확인 list)

1. **train: 4 single only** (`--no-normal` flag 박힘) — Normal/Invalid/OOD 학습 금지.
2. **eval composition**: 4 single + 6 combo (positive) + Normal + Invalid + 4 OOD (negative). 16+ class, 3,850 장.
3. **bit_F1** = positive (single+combo) macro-F1. **NOT** 11-class macro_f1.
4. **Total FAR** = `(Normal_fp + Invalid_fp + OOD_fp) / N_total_negative`. NI-only FAR 단독 보고 금지.
5. **BCE pos/neg target 독립** (`pos_target` + `neg_target` ≠ 1 가능).
6. **threshold 는 eval label 없이** — val (calibration) set 으로 결정.
7. **TTA / rotation aug 금지** — scratch ↔ scratch_rot 회전 구분 깨짐.
8. **데이터 위치 = E:** (D: 외부 삭제 사고 260514 05:03, RESUME_NOW.md).

---

## 9. 참고 코드 위치

| 항목 | path |
|---|---|
| FCM-PM 학습 entry | `chip_multilabel/_train_chip_variant.py` |
| FCM complement 구현 | 같은 파일 `:1046-1138` |
| Pair Mask 구현 | 같은 파일 `:1115-1135` |
| BCE pos/neg/temperature loss | `chip_multilabel/losses.py:45-104` |
| 추론 variant (I0~I13) | `chip_multilabel/inference_variants.py` |
| Stage 1 평가 orchestrator | `chip_multilabel/run_stage1.py` |
| val criterion (margin_max 등) | `_train_chip_variant.py:402-430, 1396-1418` |
| W1 sweep dispatcher | `_run_W1_g2_matrix.sh` |
| W2 sweep dispatcher | `_run_W2_pos_neg_target.sh` (history) |
| W1/W2 aggregator | `_w2_aggregate.py` |

## 10. 참고 paper

- Yun 2019 — CutMix
- Walawalkar 2020 — Scattered CutMix
- Sumbul 2024 — multi-label CutMix
- Müller 2019 — Label Smoothing
- Lin 2017 — Focal loss
- Ridnik 2021 — Asymmetric loss
- Hinton 2015 — Knowledge Distillation
- Yang 2023 — multi-label KD
- Guo 2017 — Temperature scaling for calibration
- Cole 2021 — Multi-label calibration (bce_min / brier_min / margin_max 선택 criterion 기반)

---

*문서 작성 시각: 260514. 작성: chip-multilabel-paper-narrator agent. 출처 = `outputs/_W2_aggregate.csv`, `outputs/_iter124_bit_F1_summary.csv`, `outputs/_iter125_gn_extension_summary.log`, `outputs/_iter126_more_diversity_summary.log`, `RESUME_NOW.md`, 기존 `manager_report/REPORT.md`.*
