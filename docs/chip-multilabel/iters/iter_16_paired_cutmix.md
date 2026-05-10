# Iter 16 — Paired CutMix (Counterfactual Augmentation)

> 260508. 새 핵심 발견. 사용자 directive: "cut mix 하는 이미지와 원본에서 덮어질 부분을 mask로 해서 pair로 생성하는 기법". v5.2 chip + Normal v5.2.3 (paint-style 1000 wafer) 후 첫 학습 cycle.

## 핵심 아이디어

매 CutMix sample 마다 **pair** 만들기:

| sample | 영상 (chip A 위) | label |
|---|---|---|
| **A_mix** (기존 CutMix) | 직사각 mask 영역에 **chip B 의 patch paste** | A.label OR B.label |
| **A_masked** (새로 추가) | **동일 직사각 영역** 만 grade-0 (corner mean) 로 채움 | **A.label only** (변화 X) |

같은 chip A 가 **같은 mask 영역 위치** 에 대해 **두 outcome** 보여줌:
- A_mix: "이 영역에 chip B 결함 있음" → model 은 chip B 결함 식별 / fire
- A_masked: "이 영역에 background 만" → model 은 false fire 안 해야

## 왜 이게 효과적인가 (mechanism)

### 기존 CutMix 의 shortcut 문제

기존 CutMix 학습 시, model 은 **"mask 영역 위치 자체"** 를 chip B label 의 prior 로 학습. paste 영역의 grade 분포 변화 만 보고 "여기에 chip B 가 있다" 라고 fire.

→ **shortcut**: 실제 defect 모양 학습 안 함, location heuristic 만 학습.

→ inference 시 normal chip 에 random noise / occlusion / saliency edge 만 있어도 **false fire** (false positive 폭주, low precision).

### Paired training 의 disentanglement

같은 location 의 **다른 outcome** 두 sample:
- **A_mix label** = A.label OR B.label  ← location 에 B 의 content 있음
- **A_masked label** = A.label  ← location 에 grade-0 만

**model 은 location 만 보면 안 됨 — content 도 봐야 함**:
- A_mix 의 mask 영역 → "B 의 grade 패턴 있음 → fire B"
- A_masked 의 mask 영역 → "grade-0 만 → fire 안 함"

→ **content-vs-location decoupling** 강제. shortcut 차단.

비유: contrastive learning 의 positive/negative pair (vs random positive only), counterfactual augmentation (CDA Niu 2021), causal disentanglement.

## Implementation (atomic 1-change)

### 신규 CLI flags (`_train_chip_variant.py`)

```bash
--cutmix-pair masked            # default 'none'
--cutmix-pair-loss-w 1.0        # masked branch loss weight
--cutmix-pair-fill corner       # {corner, white, noise} — fill 종류
```

### Pair 생성 (single mode 만)

```python
# 1. 기존 CutMix paste 전 capture
x_pre = x.clone()
tgt_pre = tgt.clone()

# 2. 기존 CutMix paste (in-place modifies x and tgt)
for bi in valid_indices:
    x[bi, :, cy:cy+side, cx:cx+side] = x[perm[bi], :, cy:cy+side, cx:cx+side]
    tgt[bi, b_class] = 1.0   # OR rule

# 3. x_masked 생성: x_pre 의 동일 영역 fill
if pair_fill == 'corner':
    fill_per_chip = x_pre[:, :, :8, :8].mean(dim=(2,3), keepdim=True)
elif pair_fill == 'white':
    fill_per_chip = ImageNet-norm(palette grade 0 RGB white)
else:  # noise
    fill_per_chip = randn * 0.3
for bi in valid_indices:
    x_pre[bi, :, cy:cy+side, cx:cx+side] = fill_per_chip[bi]
x_masked = x_pre
tgt_masked = tgt_pre   # original label, no OR with B
```

### Dual forward + loss

```python
logits_mix = model(x)
logits_masked = model(x_masked)
loss = loss_fn(logits_mix, tgt) + w * loss_fn(logits_masked, tgt_masked)
```

학습 시간 +38% (dual forward overhead, expected).

## Single seed result (260508 v5.2 cycle)

| spec | iter | macro_f1 | macro_f1 (defect) | chip-FAR | per-class F1 |
|---|---|---:|---:|---:|---|
| baseline (T7+LS=0.20+CutMix) | **16-A** | 0.9168 | 0.9497 | 0.99 (198/200) | bb=0.977, fork=0.870, sc=0.822, sr=0.998 |
| **paired (16-A + cutmix-pair masked)** | **16-B** | **0.9466** | **0.9759** | **0.85** (170/200) | bb=0.994, fork=0.857, sc=**0.938**, sr=0.998 |
| **△** | | **+0.0298** | **+0.0262** | **-0.14** | **scratch P 0.72→1.00 dramatic** |

### per-class P/R breakdown (smoking gun)

| class | 16-A precision | 16-B precision | △ P | 16-A recall | 16-B recall | △ R |
|---|---:|---:|---:|---:|---:|---:|
| bank_boundary | 0.987 | 1.000 | +0.013 | 0.967 | 0.988 | +0.021 |
| fork | **0.996** | 0.782 | -0.214 | 0.772 | 0.948 | **+0.176** |
| **scratch** | 0.723 | **1.000** | **+0.277** | 0.952 | 0.883 | -0.069 |
| scratch_rot | 1.000 | 0.998 | -0.002 | 0.997 | 0.997 | 0 |

→ **scratch 의 precision 0.72 → 1.00 (FP 233 → 0)**: 가장 명확한 mechanism 입증. paired training 이 "mask 영역 = chip B prior" shortcut 막고, model 이 **content 만 보고 fire** 하게 강제 → scratch class 의 false fire 사라짐.

→ fork 는 P-R trade (P↓ R↑↑) — 다른 dynamic, 5-seed 로 확인 필요.

## 위치 prior 의 intervention 검증 (visualization 가능)

추후 attention map / saliency 시각으로:
- baseline (16-A): scratch label fire 시 attention = mask 영역 전체 cover (location prior heavy)
- paired (16-B): scratch label fire 시 attention = scratch line 만 cover (content focused)

(아직 실행 안 함, 후속 ablation 으로 가능)

## 통계 검증 (next step)

- single seed 1 만 측정 — lucky 가능성 배제 위해 **5-seed sweep** 필요
- 5-seed 후 mean ± std 측정, paired Mann-Whitney U test (or t-test, n=5 nonparametric 권장)
- △ macro_f1 > 1 std → robust improvement

## Sweep candidates (1 atomic each, paired effect 가 robust 확인 후)

| axis | values | hypothesis |
|---|---|---|
| `cutmix_pair_fill` | corner / white / noise | corner = chip background, white = grade-0, noise = informational distractor. 어느 게 강한 disentanglement? |
| `cutmix_pair_loss_w` | 0.5 / 1.0 / 2.0 | masked branch weight. 1.0 = equal, 2.0 = masked-dominant (regularization heavier) |
| `cutmix_mode` x pair | single (16-B) / scattered (16-C) / grid (16-D) / **grid_sparse (16-E, 신규 260508)** | spatial topology × pair 의 interaction |
| `cutmix_p` | 0.25 (현) / 0.5 / 0.75 | 더 자주 paired 적용 → 효과 강화? |

### grid_sparse (16-E) 의 motivation

grid (per-cell flip prob) 와 scattered (free-position N patches) 사이 **hybrid**:
- 정의: 8×8 grid 에서 K 개 cell **uniform random sample** → 그 K 개 만 flip (replace=False)
- vs grid: per-cell 독립 flip 이 아니라 cell 갯수 K 고정. flip 영역 sparsity 통제
- vs scattered: patch 위치가 grid 정렬 → boundary alignment 자연. content vs location decoupling 더 명확
- 사용자 directive 260508: "4번째 추가하자 grid에서 random 선택하는거 grid 와 scatter mix 겠지?"

`--cutmix-mode grid_sparse --cutmix-grid-k 8` (default 8 cells of 64 = ~12.5% area).

## Comparison vs prior iter (paper headline)

| iter | spec | seeds | macro_f1 | 비고 |
|---|---|---:|---:|---|
| iter 10 | T9 + ensemble (T7N + C_44 logit avg) | 5 | 0.9930 ± 0.005 | project headline |
| iter 12 | T7N + T5 v19zpp ensemble | 5 | ~0.97 | (logger backfill) |
| iter 14 | v20 fork sigma↑ T7N | 1 | 0.9226 | single model |
| iter 15 | P0/P1A 4-class only LS sweep | — | 0.9088 (LS=0.05) | Normal-OFF counter-example |
| **iter 16-A** | T7N+LS=0.20+CutMix (v5.2 baseline) | 1 | 0.9168 | 새 baseline |
| **iter 16-B** | iter 16-A + paired CutMix masked | 1 | **0.9466** | **single model new high** |

iter 16-B (single model) 가 **iter 14 v20 single 의 0.9226 + 0.024 ↑** — paired CutMix 가 single-model SOTA 갱신 후보. ensemble 없이도 0.95 근처 도달.

## Stop criterion 재확인 (plan 의)

- △ macro_f1 ≥ 0.005 OR △ chip_FAR ≤ -1pp → 5-seed sweep 진행
- **현재 △ +0.0298 (≥ 0.005) ✓** + **△ chip-FAR -14pp (≤ -1) ✓**
- → **5-seed sweep dispatch 권장**

## 4-variant comparison plan (queued 260508 09:42, b7bwnu5tj)

| iter | mode | k cells | flip mechanism | seed 1 status |
|---|---|---|---|---|
| 16-B | single | — | 1 rectangle (rect 0.5) | done — 0.9466 |
| 16-C | scattered | — | 5 patches free position | train done, eval pending |
| 16-D | grid | per-cell prob 0.5 | ~32 cells (binomial) | train running |
| 16-E | grid_sparse | K=8 fixed | random.choice K of 64 cells | queued (after 16-D) |

`_run_iter16e_queue.sh` background script: wait for 16-D → train 16-E → eval all 3 unevaluated → 4-row summary.

## Sources (절대 경로)

- Code patch: `D:/project/known-cnn/chip_multilabel/_train_chip_variant.py:564-606` (paired CutMix logic) + `:571-602` (grid_sparse 신규 mode)
- Queue: `D:/project/known-cnn/_run_iter16e_queue.sh`, log `outputs/_iter16E_queue.log`
- 16-A train: `D:/project/known-cnn/outputs/iter16A_T7N_baseline_v52/T7_T7_iter16A_baseline_seed1_260508_074646/`
- 16-B train: `D:/project/known-cnn/outputs/iter16B_T7N_pair_v52/T7_T7_iter16B_paired_seed1_260508_074809/`
- 16-A eval: `outputs/iter16A_*/eval_seed1/stage1_260508_080136/`
- 16-B eval: `outputs/iter16B_*/eval_seed1/stage1_260508_080219/`
- bit_metrics: `${RUN}/eval_seed1/bit_metrics_split.json`
- Plan: `~/.claude/plans/skills-memory-agent-starry-puzzle.md` (paired CutMix plan)

## 절대 영구 원칙 (carry-over)

1. palette PNG only.
2. rotation/flip aug 영구 금지.
3. TTA 영구 금지.
4. 1 atomic change/iter — pair flag 만 차이.
5. 5-seed sweep = same spec noise 측정 (atomic 위배 X).
