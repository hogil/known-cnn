# Iter 17 — Multi-class chip combo (2 + 3-class) — 합성 + 14-class eval

> 260508. 사용자 directive: "추후 multi class 이미지로 합성하는 것 한번 구상해봤지?" + "2+3-class combo 10 total". chip-level multi-class N-way pixel min-blend 으로 4-class chip combo 합성. eval set 12 → 14 class 확장 (3-combo 4 class 신규). 16-B paired CutMix 모델로 zero-shot 평가.

## 합성 spec — N-way pixel min-blend

```python
def _min_blend_n(arrs):
    return np.minimum.reduce(arrs).astype(np.uint8)
```

흰색 (255,255,255) 배경의 palette PNG chip 끼리 element-wise min — 결함 색이 흰색보다 어두우니 어느 한쪽에 결함이 있는 픽셀만 합성된 chip 에 살아남음. 3-class 도 같은 원리로 자연 확장.

### Sanity check

```python
ok = defect_pixel_ratio(blended) >= max(d_i for d_i in sources) - 0.01
```

3-class 는 sources 3 개 → 일반적으로 d_blend > d_max (defect 더 많이 보임). 통과율 매우 높음.

## 합성 결과 — 1000 chip 0% reject

| combo | n | def_mean | def_min | def_max |
|---|---|---:|---:|---:|
| **2-combo (6)** | | | | |
| bank_boundary+fork | 100 | 0.214 | 0.200 | 0.237 |
| bank_boundary+scratch | 100 | 0.232 | 0.208 | 0.264 |
| bank_boundary+scratch_rot | 100 | 0.232 | 0.210 | 0.261 |
| fork+scratch | 100 | 0.180 | 0.158 | 0.211 |
| fork+scratch_rot | 100 | 0.183 | 0.162 | 0.238 |
| scratch+scratch_rot | 100 | 0.198 | 0.172 | 0.247 |
| 2-combo mean | | **0.207** | | |
| **3-combo (4) — 신규** | | | | |
| bank_boundary+fork+scratch | 100 | 0.280 | 0.252 | 0.317 |
| bank_boundary+fork+scratch_rot | 100 | 0.281 | 0.257 | 0.321 |
| bank_boundary+scratch+scratch_rot | 100 | 0.298 | 0.268 | 0.331 |
| fork+scratch+scratch_rot | 100 | 0.251 | 0.224 | 0.305 |
| 3-combo mean | | **0.278** | | |
| **△** | | **+0.071 (34% ↑)** | | |

→ 3-combo 가 2-combo 보다 defect 픽셀 34% 더 많음 (1 source 추가 효과 자연 누적). sanity 0 reject (1000/1000 통과).

## 14-class eval (16-B paired CutMix model)

eval set: `D:/project/data/wm-811k/chip_multilabel_v14class/` (4 single + 10 combo + Normal + Invalid). per_class 50 random sample (strength_min 0.0). 80% eval split.

### Top-line metric

| 평가 set | n_classes | macro_f1 | top1_11c | model |
|---|---:|---:|---:|---|
| 12-class (iter 16-B 본 eval) | 12 | **0.9466** | 0.823 | T7+LS=0.20+CutMix paired single |
| 14-class (iter 17, 동 model) | 14 | **0.815** | — | 동일 |
| **△** | +2 | **-0.131** | | 3-combo 4 class zero-shot drop |

### Per-class accuracy (14-class breakdown)

| class | n | correct_11c | correct_mh |
|---|---:|---:|---:|
| **single (4)** | | | |
| bank_boundary | 40 | 1.000 | 1.000 |
| fork | 40 | 1.000 | 1.000 |
| scratch | 40 | 1.000 | 1.000 |
| scratch_rot | 40 | 1.000 | 1.000 |
| **2-combo (6)** | | | |
| bank_boundary+fork | 40 | 0.625 | 0.625 |
| bank_boundary+scratch | 40 | 1.000 | 1.000 |
| bank_boundary+scratch_rot | 40 | 0.975 | 0.975 |
| fork+scratch | 40 | 1.000 | 1.000 |
| fork+scratch_rot | 40 | 1.000 | 1.000 |
| scratch+scratch_rot | 40 | 0.775 | 0.775 |
| **3-combo (4) — 신규 zero-shot** | | | |
| bank_boundary+fork+scratch | 40 | **0.000** | 0.000 |
| bank_boundary+fork+scratch_rot | 40 | **0.000** | 0.000 |
| bank_boundary+scratch+scratch_rot | 40 | **0.000** | 0.000 |
| fork+scratch+scratch_rot | 40 | **0.000** | 0.000 |
| **special** | | | |
| Normal | 40 | 0.000 | 0.000 |
| Invalid | 40 | 0.000 | 0.000 |

### 핵심 발견

1. **3-combo 4 class 모두 0% accuracy** — single + 2-combo 만 학습한 model 이 3 bit 동시 발화 못함. paired CutMix 가 location prior shortcut 막는 mechanism 으로는 부족 — **학습 데이터에 3-combo 샘플 추가 필요**.
2. **2-combo F1 가 일부 구간 깨짐**: bb+fork 0.625 (낮음), sc+sr 0.775 (낮음) — 14-class 환경의 threshold 재조정이 일부 2-combo 의 정밀도 떨어뜨림.
3. **Normal/Invalid 0%**: 14-class 환경에서 I3 (per-class F1-max threshold) 가 3-combo 잡으려고 threshold 낮춤 → Normal 의 noise pixel 이 일부 bit 발화. **이전 12-class 환경 (iter 16-B Normal F1=1.0) 과 다른 trade-off**.

### Per-bit P/R/F1 (4 trained bit, 14-class GT 기반)

| bit | P | R | F1 | AP |
|---|---:|---:|---:|---:|
| bank_boundary | 0.985 | 0.946 | 0.965 | 0.999 |
| fork | 0.695 | 0.650 | 0.672 | 0.871 |
| scratch | 0.956 | 0.621 | 0.753 | 0.971 |
| scratch_rot | 0.991 | 0.775 | 0.870 | 0.998 |

→ fork bit 의 P/R 모두 떨어짐 (3-combo bb+fork+sc, bb+fork+sr 의 fork 발화 못함 → R 떨어지고, fork-only chip 에 다른 bit fire → P 떨어짐).

## ★ Paper headline 새 framing (260508)

사용자 directive: **"10 class 가 진짜 성능이고, Normal+Invalid 은 FP 만 따로 관리하는게 맞다"**

→ paper main metric 분리:

| metric | 영역 | 계산 |
|---|---|---|
| **CF1 / mF1_S+2** | 4 single + 6 2-combo (10 defect class) | per-bit F1 macro on 학습 분포 안 chip 만 |
| **chip_FAR** | Normal + Invalid | per-chip false fire rate (별도 관리) |
| **3-combo F1** (diagnostic) | 4 3-combo (zero-shot) | robustness 진단 (학습 안 함) |

이전 mF1_paper12 (S+2+N+I 합산) 폐기 — Normal/Invalid 의 chip-level FP 평가가 bit F1 에 묻혀 의미 흐려졌음. 새 framing 으로 paper Table 분리.

## 17B — 학습 통합 결과 (260508 신규)

`_train_chip_variant.py` 에 `--multi-combo-root` + `--multi-combo-n-per-class` flag 추가. dataset 가 multi-hot return (3-tuple `(x, y, mh)`) — single y∈[0..3], Normal y=-1, multi-combo y=-2. eval/loss 모두 mh 직접 사용.

```bash
python -m chip_multilabel._train_chip_variant \
    --variant T7 --ls 0.20 \
    --multi-combo-root D:/project/data/wm-811k/chip_multilabel_synth \
    --multi-combo-n-per-class 50 \
    --epochs 8 --batch 8 --accum 4 --seed 1 \
    --cutmix-p 0.25 --cutmix-pair masked
```

학습 데이터: 800 single + Normal + **500 multi-combo** = 1300 (1040 train + 260 val). best ep 2, val_acc 1.0.

### 17B 14-class eval (I3 threshold)

| metric | 16-B (no multi-combo) | 17B (multi-combo) | △ |
|---|---:|---:|---:|
| **macro_f1** | 0.815 | **0.8391** | **+0.024** |
| Normal accuracy | 0.000 | **1.000** | **+1.000** |
| Invalid accuracy | 0.000 | **1.000** | **+1.000** |
| bank_boundary+fork | 0.625 | **1.000** | **+0.375** |
| scratch+scratch_rot | 0.775 | **1.000** | **+0.225** |
| fork+scratch_rot | 1.000 | 0.875 | -0.125 |
| 4 single class | 1.000 | 1.000 | 0 |
| 4 3-combo (all 4) | 0.000 | **0.000** | **0** ★ |

### 핵심 발견 17B

1. **Normal/Invalid 회복 (0% → 100%)** — multi-combo 학습 데이터가 4 single-class 와 다른 distribution 의 chip 을 model 에 노출시켜 threshold 안정화. multi-hot zero-vector (Normal) 가 학습 신호로 명확해짐.
2. **2-combo 약점 회복** — bb+fork (0.625→1.0), sc+sr (0.775→1.0). 학습 데이터에 직접 본 combo 라 성능 ↑.
3. **3-combo 여전히 0%** — model 이 3 bit 동시 fire 못함:
   - bb+fork+sc → 70% bb (1 bit), 20% fork (1 bit), 5% bb+fork (2 bit). 0% 3 bit
   - threshold 0.78-0.85 에 맞춰 가장 강한 1 bit 만 통과
   - I7 (joint coord descent) 도 3-combo 0% — model 자체 한계
   - 50 chip/3-combo (총 200) 부족 가능성

### 17C 결과 (atomic: multi-combo n_per_class 50 → 100)

학습 데이터: 800 single + 1000 multi-combo (vs 17B 의 500). best ep5, val 1.0.

| metric | 17B (n=50) | 17C (n=100) | △ |
|---|---:|---:|---:|
| **macro_f1** | 0.8391 | 0.8225 | -0.017 |
| 2-combo mean | 0.975 | **0.996** | +0.021 ✅ |
| Normal | **1.000** | **0.050** | **-0.950** ❌ |
| Invalid | 1.000 | 1.000 | 0 |
| 3-combo mean | 0.000 | 0.000 | 0 |

**핵심 trade-off** — multi-combo 1000 이 Normal 200 압도 → Normal 학습 신호 묻혀 0.05 collapse. **17B (n=50) 이 sweet spot** (multi-combo 500 vs single 800 vs Normal 200 → balanced).

### 17D dispatch (atomic: epochs 8 → 16, n=50 유지)

Hypothesis: 3-combo 0% 가 model 학습 시간 부족 — 16 epoch 까지 내려가 3-bit fire 패턴 crystallize 가능.

```bash
python -m chip_multilabel._train_chip_variant \
    --variant T7 --ls 0.20 \
    --multi-combo-root D:/project/data/wm-811k/chip_multilabel_synth \
    --multi-combo-n-per-class 50 \
    --epochs 16 --batch 8 --accum 4 --seed 1 \
    --cutmix-p 0.25 --cutmix-pair masked \
    --tag T7_iter17D_pair_multicombo_ep16_seed1
```

### Stop criterion

| metric | 기준 |
|---|---|
| **3-combo accuracy mean** | **≥ 0.3** → atomic 효과 입증 (현 17B = 0%) |
| Normal F1 | 1.0 유지 |
| 14-class macro_f1 | 0.84 이상 유지 |
| 12-class macro_f1 (12-class set) | 0.94 ± 0.01 유지 |

## Sources (절대 경로)

- 합성 코드: `D:/project/known-cnn/chip_multilabel/gen_multi_combo_synth.py` (NEW)
- N-way blend 함수: `gen_eval_set._min_blend_n` (NEW), 기존 `_min_blend` 재사용
- spec doc: `D:/project/known-cnn/docs/synthesis/MULTI_COMBO_BLEND.md`
- master folder: `D:/project/data/wm-811k/chip_multilabel_synth/` (10 combo × 100 chip)
- eval set: `D:/project/data/wm-811k/chip_multilabel_v14class/` (14 class)
- eval result: `outputs/iter17_T7N_pair_14class_eval/stage1_260508_110154/`
- preds parquet: `outputs/iter17_T7N_pair_14class_eval/stage1_260508_110154/preds_chip.parquet`
- per_class metrics: `outputs/iter17_T7N_pair_14class_eval/stage1_260508_110154/per_class_metrics.parquet`
- plan: `~/.claude/plans/skills-memory-agent-starry-puzzle.md`

## 절대 영구 원칙 (carry-over)

1. palette PNG mode='P' source — RGB blend 결과 저장 (단, palette 색 set 내 보장).
2. rotation/flip aug 영구 금지.
3. TTA 영구 금지.
4. 1 atomic change/iter — 본 iter 는 eval set 12→14 추가만, 학습 데이터 추가는 별 iter.
5. 5-seed sweep — same spec noise 측정 (Step 6 후 확인).
6. master folder 만 (subset/archive 폴더 절대 안 만듦), runtime sampling.
7. **합성 후 [OUT] 절대 경로** 메시지 마지막 줄 표시 (CLAUDE.md 강제).

## 양 repo mirror (260508 완료)

| 파일 | known-cnn → unknown-contrastive |
|---|---|
| `chip_multilabel/constants.py` | mirrored |
| `chip_multilabel/gen_eval_set.py` | mirrored |
| `chip_multilabel/eval_dataset.py` | mirrored |
| `chip_multilabel/gen_multi_combo_synth.py` (NEW) | mirrored |

[OUT] D:/project/data/wm-811k/chip_multilabel_synth/_preview/  (10 combo preview PNG)
[OUT] D:/project/data/wm-811k/chip_multilabel_v14class/_preview/  (14 class preview PNG)
