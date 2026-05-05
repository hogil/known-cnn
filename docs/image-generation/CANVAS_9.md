# 9 obj-less wafer-canvas — alpha-based 합성 spec (round 12-25 진화)

기존 18 obj-active class + 9 wafer-canvas obj-less class 합성 후 active 27 class 완성.

이전 버전 `CANVAS_8.md` (8 class, 도형 기반 outline) 는 deprecated — 사용자 reject.
**새 spec: chip-internal alpha 매커니즘을 wafer 6400×6400 에 적용 (chip 단위 처리 X, 한 번에).**

## 핵심 설계 (사용자 round 12-25 누적 합의)

### alpha = P(defect pixel | position) 분포 함수

기존 obj-active chip-internal alpha 함수의 confidence-weighted distribution mixing 을
**wafer 전체 (6400×6400)** 한 번에 적용:

```
alpha(y, x) ∈ [0, 1]   # spatial 분포 모델 (line/arc/ring/donut 등)
cum_mixed = (1-alpha) * CUM_BASE + alpha * CUM_PEAK
grade(y, x) = searchsorted(cum_mixed, U(0,1))
```

- `alpha=0` → pure baseline (normal noise: P(0)=0.83, P(1)=0.15) → wafer normal 영역과 동일
- `alpha=0.3` → 70% baseline + 30% peak: P(0) 감소, P(1) 증가, peak grade 등장
- `alpha=1.0` → pure peak: class 별 grade 다양 분포 (CLASS_PEAK_DIST)

### Lorentzian 두 개 sum (sharp peak + heavy tail)

`_perp_sharp(d, σ)` (round 19-20 합의):

```python
sharp = 1.0 / (1.0 + (d/(0.5σ))**2)        # 좁고 매우 sharp peak
wide  = 0.60 / (1.0 + (d/(5σ))**2)          # 크고 넓은 heavy tail
return min(sharp + wide, 1.0)
```

| d | 0 | σ | 2σ | 4σ | 8σ | 15σ | 25σ |
|---|---|---|---|---|---|---|---|
| α | 1.00 | 0.78 | 0.58 | 0.40 | 0.18 | 0.06 | 0.024 |

Gaussian 단일 / 단일 Lorentzian 모두 round 18-20 시도 → 사용자 reject. **두 Lorentzian sum 만이
"가운데 매우 sharp + 양 끝 자연 0 fade" 동시 만족**.

### chip border decision (round 23 strict primary filter)

defect chip 결정은 chip 의 alpha 분포 둘 다 strict:

```python
chip_alpha_mean = chip_alpha[200x200].mean()
chip_alpha_max  = chip_alpha[200x200].max()
if chip_alpha_mean < 0.10: continue       # primary filter (line 직접 통과 chip 만)
if chip_alpha_max  < 0.30: continue       # secondary
p_def = min(chip_alpha_mean * 3.0, 1.0)   # mean 기준 (max 아닌)
```

기존 max-only filter (round 22 이전) 시 line 곁가지 / tail chip 까지 BIN 처리 → 외곽 산만.
**alpha mean primary filter 가 line 직접 통과 chip 만 BIN, 그 외 모두 normal.**

### invalid 비례 fix (`_sample_gen.py`, round 25)

obj-active 18 class 도 함께 재합성. 이전 `n=15 fixed` 였던 invalid_inside_mask 갯수 →
**defect 갯수의 ~15% 비례** (defect 적은 class 도 invalid 적게):

```python
# obj-active branch
invalid_inside_mask = select_random_invalid(rng, defect_mask, inside,
                                            n=max(2, int(defect_mask.sum() * 0.15)))
# invalid_main branch
invalid_random = select_random_invalid(rng, invalid_dist, inside,
                                       n=max(2, int(invalid_dist.sum() * 0.10)))
# Normal branch — defect 비례 0.15
```

## 9 canvas class 최종 spec (`_sample_canvas_gen.py`)

### Line/Arc 계열 (5)

| Class | alpha 함수 | sigma | peak | half_len | 변동 |
|---|---|---|---|---|---|
| **DiagonalSmear** | 1 line, angle=45°±5°, along-fade | CHIP·U(0.10, 0.20) | U(0.30, 0.50) | SIZE·U(0.20, 0.35) | center ±0.3 chip, σ_end=CHIP·U(2-4) |
| **CrossScratch** | 2 ⊥ line cross, max() | CHIP·U(0.10, 0.20) | U(0.30, 0.50) | SIZE·U(0.20, 0.32) | base_angle ±0.04 rad |
| **CrescentArc** | bottom-fixed 1/4-1/3 arc, radial × angular | r=R·U(0.78,0.88), σ_r=R·U(0.008, 0.016) | U(0.30, 0.50) | arc len π/3-π/2.5 | th_center 변동 ±0.15 |
| **ParallelScratches** | 3-5 parallel lines, max() | CHIP·U(0.10, 0.20) | U(0.30, 0.50) | SIZE·U(0.20, 0.32) | spacing CHIP·U(3.5-5.5) |
| **BrokenRing** | annular ring × 1-3 angular gap | r=R·U(0.85,0.93), σ_r=R·U(0.007,0.014) | U(0.30, 0.50) | gap π/6-π/3 each | gap 0.80 dip (완전 X) |

### Point/Ring 계열 (4)

| Class | alpha 함수 | 특징 | peak |
|---|---|---|---|
| **RingDots** | ring 위 14-23 angular dot | sigma_blob=CHIP·U(0.20-0.40), r=R·U(0.40-0.65) | U(0.40, 0.60) |
| **CenterDonut** | center 얇은 ring | r=R·U(0.08-0.16), σ_r=R·U(0.006-0.014) | U(0.30, 0.50) |
| **Row** | direct PIL Draw line (others alpha-based) | 20-40 mini lines, len=CHIP·U(1/6-1/4), 1-2px | binary 1.0 |
| **Starburst** | center 빈 ring + 8-14 radial ray | ray_len=R·U(0.55-0.75), σ_perp=CHIP·U(0.05-0.10) | center 0.60-0.85, ray 0.30-0.50 |

### CLASS_PEAK_DIST (peak alpha 일 때 grade dist)

| Class | grade 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 |
|---|---|---|---|---|---|---|---|---|
| DiagonalSmear | 0.00 | 0.60 | 0.28 | 0.10 | 0.02 | 0.00 | 0.00 | 0.00 |
| CrossScratch | 0.00 | 0.40 | 0.30 | 0.18 | 0.08 | 0.03 | 0.01 | 0.00 |
| CrescentArc | 0.00 | 0.50 | 0.30 | 0.15 | 0.05 | 0.00 | 0.00 | 0.00 |
| ParallelScratches | 0.00 | 0.30 | 0.28 | 0.20 | 0.12 | 0.07 | 0.03 | 0.00 |
| BrokenRing | 0.00 | 0.20 | 0.22 | 0.20 | 0.18 | 0.10 | 0.07 | 0.03 |
| RingDots | 0.00 | 0.45 | 0.30 | 0.15 | 0.07 | 0.03 | 0.00 | 0.00 |
| CenterDonut | 0.00 | 0.50 | 0.30 | 0.13 | 0.05 | 0.02 | 0.00 | 0.00 |
| Row | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 1.00 |
| Starburst | 0.00 | 0.40 | 0.28 | 0.18 | 0.10 | 0.03 | 0.01 | 0.00 |

사용자 spec: 대부분 grade 1, 2 만. 일부 class 만 더 강 (BrokenRing 1-6 wide, ParallelScratches
1-5 wide). Row 만 grade 7 fixed (선명).

### multi-scale field modulation (line 균일 X)

```python
field_medium = bilinear(rng, 32, 32, 0.50, 1.50)      # ~chip patch 강 변동
field_high   = bilinear(rng, 128, 128, 0.65, 1.40)    # ~chip 0.25
alpha *= field_medium * field_high
alpha += N(0, 0.04**2)   # fine noise
alpha = clip(alpha, 0, 1)
```

## 사용자 round 12-25 catch + fix history

| Round | 사용자 catch | fix | rationale |
|---|---|---|---|
| 12 | "도형 stroke (PIL Draw line) reject" | direct sample → alpha 분포 mix | wafer 합성 자연스러운 noise field 필요 |
| 13-15 | "line 균일하지 않게, 자연 변동" | multi-scale random field (8/16/24-32/96-128 coarse + bilinear) | line 따라 alpha 강/약 변동 |
| 16 | "low/high cut 인위적, 절벽" | alpha = baseline ↔ peak mix weight (cum_mixed) | smooth transition, cut 폐기 |
| 17 | "scratch_21deg angle 풀림 / line full diameter X" | angle/position lock per-class (rotation ±5°), line partial (along_taper) | class identity preserve |
| 18-19 | "Gaussian 너무 넓음, sharp peak + heavy tail 필요" | Lorentzian sharp + heavy tail | peak narrow, tail not zero |
| 20 | "두 분포 sum 으로 narrow + wide 동시" | sharp 0.5σ + wide 5σ Lorentzian sum (weight 0.6) | 가운데 sharp + 양 끝 fade |
| 21 | "어떤 class grade 1만, 일부만 3+" | CLASS_PEAK_DIST 8 entry per class | class 별 grade 다양성 |
| 22 | "RingDots, CenterDonut 새 class. line partial" | along_taper + 새 alpha 함수 추가 | wafer-canvas 다양성 확장 |
| 23 | "line 직접 지나가는 chip 만 BIN, 외곽 normal" | alpha mean (max 아닌) primary filter | chip border decision strict |
| 24 | "Starburst (center 빈 ring + radial rays), Row (짧은 ㅡ scatter)" | alpha_starburst, alpha_row (PIL Draw 예외) | 새 spatial class |

## Row 특별 spec (사용자 명시 round 24-25)

`alpha_row` 만 alpha-based stochastic X — **PIL ImageDraw.line 직접 그림** (vector approach).

| 항목 | 값 |
|---|---|
| line direction | **y 값 변하지 않고 x 값만 변함** (horizontal-locked, angle ±0.08 rad ≈ ±4.6°) |
| line length | `chip * uniform(1/6, 1/4)` — chip 1/6 ~ 1/4 매우 짧음 |
| line width | 1-2 px (매우 얇음, sharp) |
| n_lines | 40-69 random scatter inside wafer (R*0.85 안) |
| alpha 값 | binary 1.0 (line pixel) / 0 (그 외) |
| grade dist | `CLASS_PEAK_DIST["Row"] = [0,0,0,0,0,0,0,1.0]` — grade 7 only (선명) |
| chip border decision | `chip_alpha_max == 1.0` → max bonus path — alpha mean 작아도 line 위 chip BIN 처리 |
| 25 | "obj-active invalid 너무 많음, defect 비례로" | _sample_gen.py n=15 → defect.sum() * 0.15 | defect 적은 class invalid 도 적게 |

## 합성 산출

```
per class: 200 sample
total: 9 * 200 = 1800 wafer

PNG: D:/project/data/wm-811k/unknown/<class>/*.png  (6400×6400 palette)
JSON: D:/project/data/positions/unknown/<class>/*.json
```

obj 없으니 `chip_meta['obj']=None` → inline chip-object crop 안 만들어짐 (`_sample_gen.save_chip_crops` skip).

## 합성 실행

```bash
cd D:/project/known-cnn

# 작은 테스트 (1 sample per class = 9장)
python _sample_canvas_gen.py --n 1

# 본 합성 (200 per class = 1800장)
python _sample_canvas_gen.py --n 200

# 일부 class 만
python _sample_canvas_gen.py --n 200 --classes Starburst RingDots

# obj-active 18 + canvas 9 통합 재합성
python _sample_gen.py --n 200 --workers 8     # obj-active (invalid 비례 fix 포함)
python _sample_canvas_gen.py --n 200          # canvas 9
```

## active 27 class (`experiments/active_classes_27.yaml`)

| Group | Class |
|---|---|
| Donut × 5 obj | Donut_{bank_boundary, invalid_main, particle_blast, scratch, scratch_21deg} |
| Edge-Bottom × 5 obj | Edge-Bottom_{...} |
| Edge-Top × 5 obj | Edge-Top_{...} |
| 특수 obj-active (3) | Edge-Ring_invalid_main, Edge-Ring_particle_blast, Thick-Edge_invalid_main |
| Canvas 9 | DiagonalSmear, CrossScratch, CrescentArc, ParallelScratches, BrokenRing, RingDots, CenterDonut, Row, Starburst |

**제외 (round 25):**
- `Center_invalid_main`, `Full_invalid_main` — Center/Full 분포는 V3 chipgrid 에서 saturated +
  obj 분류 의 의미 약함. archive 14 의 일부와 함께 보존만.

## 절대 금기

- 도형 stroke (PIL Draw line/circle/polygon) 으로 spatial pattern 그리는 방식 금지 — round 12 reject
  - 예외: `Row` class 만 (사용자가 직접 PIL Draw 명시, 짧은 mini-line scatter)
- Gaussian 단일 (sharp 만) / Lorentzian 단일 → 양 끝 자연 fade 안 됨 (round 18-20)
- alpha low/high cut (절벽) — round 16 폐기, baseline ↔ peak mix 만
- chip border decision 에서 max only filter — round 23 폐기, mean primary
- `_sample_gen.py` 의 invalid `n=15 fixed` — round 25 폐기, defect 비례

## Cross-link

- 코드: `_sample_canvas_gen.py`, `_sample_gen.py` (invalid 비례 fix)
- spec history: `docs/image-generation/CANVAS_8.md` (deprecated 8 class outline)
- chip-internal alpha 원형: `docs/image-generation/SPEC.md` (obj-active alpha 함수)
- active class 정책: `~/.claude/projects/D--project-known-cnn/memory/feedback_active_class_policy.md`
- alpha 설계 lesson: `~/.claude/projects/D--project-known-cnn/memory/feedback_canvas_alpha_design.md`
- skill: `.claude/skills/image-generation/SKILL.md` (canvas section 추가)
- round 19/20 plan history: `~/.claude/plans/1-input-batch-hidden-patterson.md`
