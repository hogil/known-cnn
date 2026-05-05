# 8 obj-less wafer-canvas 합성 spec

기존 22 active class 외 obj-less wafer-canvas 8 class. 합성 후 active 30 class 완성.

## Origin: 다양한 fab 공정 이상 + WM-811K 분포 변형

각 패턴 = (semiconductor fab 의 다양한 mechanism 가설) × (WM-811K cca/* base 분포 변형).
chuck 만이 아니라 **chuck / lens / scanner / photoresist / etch / CMP / deposition /
particle / robot / ESD / thermal / vibration** 등 fab 의 다양한 origin 이 같은 spatial
pattern 을 만들 수 있음 — 한 패턴 의 origin 은 multi-source.

기존 22 class 와 distinguishable: chip-level object 없음 (`chip_meta['obj']='none'`), 순수 wafer-level spatial pattern.

## Fab origin 카탈로그 (참조)

각 패턴 의 가능한 origin pool — 실제 fab 에서 같은 spatial signature 만들 수 있는 mechanism:

| 카테고리 | Mechanism | 영향 spatial |
|---|---|---|
| Mechanical | chuck slip / pin / edge pressure | line / spot / arc / ring |
| Mechanical | robot finger / handler clamp | parallel / cross |
| Lithography | lens partial obstruction / contamination | sector / wedge |
| Lithography | scanner scan direction residue | parallel band |
| Lithography | dose non-uniformity (radial / linear) | gradient / ring |
| Coating | photoresist edge bead / spin direction | edge / spiral |
| Coating | dispense nozzle drip path | blob chain |
| Etch | plasma non-uniformity (radial / sector) | ring / arc |
| Etch | endpoint detection partial | partial pattern |
| CMP | pad wear / pad groove | parallel / spiral |
| CMP | slurry distribution / wafer rotation | radial / spiral |
| Deposition | target erosion / shadow effect | ring with shadow |
| Deposition | thickness gradient (rotation) | spiral / ring |
| Particle | contamination drop / chain | blob / blob chain |
| ESD | spot discharge | spot / cross spots |
| Thermal | temperature gradient (anneal) | radial / sector |
| Vibration | facility resonance | wave / parallel |
| Cleaning | water mark / dry pattern | radial / blob |

## Base distribution

WM-811K cca/* 8 class 의 chip grid 32×32 heatmap (`_dist_heatmaps/<cls>_p_defect_32.npy`):

| Class | n sample | max_p | mean_p | 특징 |
|---|---|---|---|---|
| Center | 290 | 0.886 | 0.070 | 중앙 집중 |
| Donut | 47 | 0.723 | 0.146 | annular ring |
| Edge-Loc | 382 | 0.273 | 0.071 | edge 주변 + 일부 cluster |
| Edge-Ring | 400 | 0.927 | 0.100 | 강한 edge ring |
| Loc | 279 | 0.154 | 0.051 | local cluster |
| Near-full | 9 | 1.000 | 0.693 | 전체 거의 dense |
| Random | 75 | 0.636 | 0.289 | scattered |
| Scratch | 88 | 0.143 | 0.021 | 선형 sparse |

8 새 패턴 base = 위 분포 의 변형/조합.

## 8 패턴 spec

### 1. DiagonalSmear — 대각 trail/smear
- **Possible origins** (multi-source):
  - Mechanical: chuck rotation slip / robot finger drag (한 방향 미끄러짐)
  - Lithography: scanner scan direction residue (대각 scan path)
  - Coating: photoresist spin direction streak
  - Vibration: facility resonance 의 directional wave
- **WM-811K base**: Loc + Random elongated (Scratch 의 angle 변형)
- **Spatial model** (32×32 chip grid 위):
  ```
  line: y = m·x + c  (m = tan(angle))
  distance: d(chip) = |y - m·x - c| / √(1+m²)
  P(defect | chip) = base · exp(-d² / 2σ²)
  ```
- **Variation**:
  - angle: ±30°, ±45°, ±60° (8 방향 중 sample)
  - σ: 1.5-3 chip cell width
  - line offset c: ±5 chip
  - density (defect chip count): 60-150 (Loc/Random 분포)

### 2. CrossScratch — + 자 scratch
- **Possible origins**:
  - Mechanical: 두 chuck pin/bar / robot dual finger (두 방향 stress)
  - ESD: 두 discharge spot (cross alignment)
  - CMP: pad groove cross pattern (가로/세로 wear)
  - Litho: alignment cross mark over-exposure
- **WM-811K base**: Scratch × 2 cross
- **Spatial model**:
  ```
  line1: angle θ1, dist d1
  line2: angle θ2 = θ1 + 90° (perpendicular), dist d2
  P(defect | chip) = max(exp(-d1²/2σ²), exp(-d2²/2σ²))
  ```
- **Variation**:
  - θ1: 0° (axis-aligned ⊥) or 45° (diagonal X)
  - cross center (cx, cy): wafer 중앙 ±3 chip
  - arm width σ: 1-2 chip
  - density: 80-180

### 3. CrescentArc — 초승달 1/4 arc
- **Possible origins**:
  - Mechanical: chuck 한쪽 edge 압력 불균형
  - Lithography: lens partial obstruction (1/4 quadrant 차단)
  - Etch: plasma sector non-uniformity (1/4 endpoint diff)
  - Deposition: shadow effect 한쪽
  - Thermal: 한쪽 heater 온도 편차
- **WM-811K base**: Edge-Ring 의 1/4 sector
- **Spatial model**:
  ```
  polar: r = √((x-cx)² + (y-cy)²) / R, θ = atan2(y-cy, x-cx)
  P(defect | chip) = base · I(r in [r_min, r_max]) · I(θ in [θ_start, θ_end])
  ```
- **Variation**:
  - sector 위치 θ_start: 4 방향 (top/right/bottom/left)
  - arc 길이: 1/4 ~ 1/3 (90° ~ 120°)
  - r range: [0.7, 0.95]
  - 두께 (r_max - r_min): 0.1-0.2
  - density: 40-100 (sparse arc)

### 4. SpiralTrail — spiral path
- **Possible origins**:
  - Mechanical: chuck rotation axis misalignment
  - CMP: wafer rotation under uneven slurry distribution (spiral 자국)
  - Coating: photoresist spin coater 의 spiral wave
  - Deposition: rotation 중 thickness gradient 가 spiral path 형성
  - Cleaning: rinse spin 의 spiral water trail
- **WM-811K base**: Loc 의 spiral 확장
- **Spatial model** (Archimedean spiral):
  ```
  r(θ) = r₀ + a·θ  (a = pitch)
  parametric: x = r·cos(θ), y = r·sin(θ)
  P(defect | chip) = base · exp(-d² / 2σ²)
    where d = chip 의 spiral path 까지 거리
  ```
- **Variation**:
  - 회전 방향 (CW / CCW)
  - pitch a: 0.05-0.1 (회전 당 r 증가)
  - 시작 r₀: 0.1-0.2
  - 길이 (회전 수): 1-3 회전
  - σ: 1-2 chip
  - density: 50-120

### 5. ParallelScratches — 평행 다중 scratch
- **Possible origins**:
  - Mechanical: chuck multi-bar / robot multi-finger contact
  - CMP: pad groove parallel pattern (radial groove worn down)
  - Lithography: scanner scan line 의 systematic residue
  - Etch: gas flow direction 의 line non-uniformity
  - Vibration: 일정 frequency vibration → parallel wave
- **WM-811K base**: Scratch × 3-5 parallel
- **Spatial model**:
  ```
  N lines (3-5) with same angle, parallel spacing s
  line_i: y = m·x + c_i  where c_i = c0 + i·s
  P(defect | chip) = max_i exp(-d_i² / 2σ²)
  ```
- **Variation**:
  - angle: horizontal (0°), vertical (90°), +45°, -45°
  - line 갯수 N: 3-5
  - spacing s: 4-7 chip
  - σ (line width): 0.7-1.5 chip
  - density: 70-150

### 6. EdgeSmudge — 외곽 wide smudge
- **Possible origins**:
  - Mechanical: chuck edge 강압 / edge handling clamp damage
  - Coating: photoresist edge bead (spin coating 의 edge accumulation)
  - Etch: edge effect (etch rate 가 edge 에서 다름)
  - Deposition: edge thickness gradient (sputter shadow)
  - Cleaning: edge 의 incomplete dry → water mark
- **WM-811K base**: Edge-Loc 의 wide annular
- **Spatial model**:
  ```
  r = √((x-cx)² + (y-cy)²) / R
  P(defect | chip) = base · sigmoid(α · (r - r_threshold))
    where α = sharpness, r_threshold = 0.55-0.65
  ```
- **Variation**:
  - r_threshold: 0.55-0.65 (시작 반경)
  - sharpness α: 5-10
  - sector: full ring (2π) 또는 1/2 ring (180°)
  - density: 100-250 (wide)

### 7. BlobChain — blob 들이 chain 연결
- **Possible origins**:
  - Mechanical: chuck multi-pin contact spot
  - Particle: contamination 입자 chain (drop trajectory)
  - Coating: dispense nozzle drip path (drop chain)
  - CMP: slurry residue chain spots
  - Lithography: scanner stage stop points (multi-station)
  - ESD: multi-spot discharge sequence
- **WM-811K base**: Loc × 3-5 connected
- **Spatial model**:
  ```
  N blobs (3-5) at positions (x_i, y_i) along a curve
  curve: line or arc
  P(defect | chip) = max_i exp(-||chip - (x_i, y_i)||² / 2σ²)
  ```
- **Variation**:
  - curve type: line, arc, S-curve
  - blob 갯수 N: 3-5
  - σ (blob radius): 1.5-2.5 chip
  - blob spacing: 4-7 chip
  - density: 60-150

### 8. BrokenRing — gap 있는 partial ring
- **Possible origins**:
  - Mechanical: chuck ring contact 부분 누락
  - Lithography: lens 의 ring obstruction with discontinuity
  - Etch: plasma ring uniformity 의 partial breakdown
  - Deposition: shadow ring with gap (특정 chamber geometry)
  - Thermal: heater ring 의 일부 dead zone
  - Coating: edge bead 가 일부 만 형성
- **WM-811K base**: Edge-Ring with random gaps
- **Spatial model**:
  ```
  base ring: r in [0.82, 0.95], θ in [0, 2π]
  gaps: K (1-3) sectors, each θ in [θ_g_start, θ_g_end]
  P(defect | chip) = base_ring · (1 - I(θ in any gap))
  ```
- **Variation**:
  - ring r range: [0.82, 0.95]
  - gap 갯수 K: 1-3
  - gap 위치 (θ): random within [0, 2π]
  - gap 폭 (각도): 30°-60° each
  - density: 80-180

## 합성 algorithm (per sample)

```python
def synth_canvas_class(class_name, sample_idx, seed):
    rng = np.random.default_rng(seed + sample_idx)
    grid = np.zeros((32, 32), dtype=np.float32)  # P(defect | chip)

    # 1. spatial mask 생성 (8 class 별 함수)
    mask = build_spatial_mask(class_name, rng)  # 32x32 P 값

    # 2. WM-811K base distribution 응용 (per class 매핑)
    base_class = BASE_MAP[class_name]  # e.g. DiagonalSmear → Loc
    base_heatmap = load_heatmap(base_class)  # _dist_heatmaps/<base>_p_defect_32.npy
    grid = mask * base_heatmap_factor + mask  # 변형

    # 3. defect chip count = density (class 별 range 에서 sample)
    target_count = rng.integers(*DENSITY_RANGE[class_name])

    # 4. 확률 따라 chip pick (top P 위주)
    chip_indices = sample_chips_by_prob(grid, target_count, rng)

    # 5. chip-level grade (b 값) 0-31 mapping
    chips = []
    for (cy, cx) in chip_indices:
        b = sample_grade(grid[cy, cx], rng)  # 1-31, peak 일수록 높은 grade
        chips.append({"x_abs": cx, "y_abs": cy, "b": b, "obj": "none", "f": [], "q": []})

    # 6. wafer 외 chip mask out (WM-811K wafer 모양 따라)
    chips = filter_inside_wafer(chips)

    # 7. PNG palette 그리기 + JSON 작성
    png = render_palette(chips, size=6400)
    json = {
        "partid": gen_partid(rng),
        "pgm": gen_pgm(rng),
        "ftn_keys": gen_ftn_keys(rng, "low"),  # obj-less 라 hot 없음 (low rate)
        "qtn_keys": gen_qtn_keys(rng, "low"),
        "chips": chips,
        "coord": {"tiles_w_rot": 32, "tiles_h_rot": 32}
    }
    return png, json
```

## Base mapping table

| 새 class | Multi-origin | WM-811K base | Variation factor |
|---|---|---|---|
| DiagonalSmear | chuck slip / scanner / spin / vibration | Loc + Scratch | line angle, σ, offset |
| CrossScratch | dual pin / ESD pair / CMP cross / litho cross | Scratch × 2 | center, arm width, axis vs diagonal |
| CrescentArc | chuck 1-side / lens 1/4 obstruct / etch sector / deposition shadow | Edge-Ring (1/4) | arc 위치, 길이, r range |
| SpiralTrail | rotation misalign / CMP rotation / spin coat / cleaning rinse | Loc spiral | pitch, 회전 수, 방향 |
| ParallelScratches | multi-bar / CMP groove / scan line / vibration wave | Scratch × N | angle, N, spacing |
| EdgeSmudge | chuck edge / edge bead / etch edge / sputter shadow / water mark | Edge-Loc wide | r_threshold, sharpness, sector |
| BlobChain | multi-pin / particle chain / nozzle drip / multi-station / ESD seq | Loc × N | curve type, N, spacing |
| BrokenRing | chuck partial / lens ring obstruct / etch ring / heater dead | Edge-Ring with gaps | gap 갯수, 위치, 폭 |

## chip-level grade (b) 분포

obj 없으니 chip 의 obj label="none". chip 의 fail bit count `b` (0-1024) 분포:
- spatial probability P(defect|chip) 가 높은 chip → b 높음 (200-1024 dense)
- P 낮은 chip → b 낮음 (1-50 sparse)
- non-defect chip 은 chips list 에 안 들어감 (b=0)

palette grade (0-31) mapping:
```
grade = round(b / 1024 * 31)  # 0=clean, 31=full
clamp [0, 31]
```

## 합성 산출

```
per class: 200 sample
total: 8 * 200 = 1600 wafer

PNG: D:/project/data/wm-811k/unknown/<class>/*.png  (6400×6400 palette)
JSON: D:/project/data/positions/unknown/<class>/*.json
```

obj 없으니 inline chip-object crop 안 만들어짐 (`_sample_gen.save_chip_crops` 가 chip['obj']='none' skip).

## 합성 코드 plan

option A: `_sample_gen.py` 의 main loop 에 새 class branch 추가
option B: 별도 script `_sample_canvas_gen.py` 작성 (8 class 전용, palette PNG + JSON 생성)

option B 권장 — 기존 _sample_gen.py 영향 최소 + obj-less 전용 logic 분리.

```python
# _sample_canvas_gen.py outline
import numpy as np
from PIL import Image
from _fq_metadata import gen_partid, gen_pgm, gen_ftn_qtn_keys

CANVAS_CLASSES = [
    "DiagonalSmear", "CrossScratch", "CrescentArc", "SpiralTrail",
    "ParallelScratches", "EdgeSmudge", "BlobChain", "BrokenRing",
]

PATTERN_FN = {
    "DiagonalSmear": build_diagonal_smear,
    "CrossScratch": build_cross_scratch,
    # ... 8 함수
}

def main():
    for cls in CANVAS_CLASSES:
        for i in range(200):
            png, jsn = synth_canvas_class(cls, i, seed=42)
            save_png_palette(png, f"D:/project/data/wm-811k/unknown/{cls}/{cls}_<id>.png")
            save_json(jsn, f"D:/project/data/positions/unknown/{cls}/{cls}_<id>.json")
```

## Verification

합성 후 _verify.py 실행:
```bash
python _verify.py --root D:/project/data/wm-811k/unknown --check-canvas
```

체크 항목:
- file 갯수 (200/class)
- PNG palette 384 mode, 6400×6400
- JSON schema (partid, pgm, ftn_keys, qtn_keys, chips, coord)
- chip['obj']='none' 모두
- chip['f'], chip['q'] empty 모두
- chip count distribution 합리적
- 시각 sanity (1 sample 씩 렌더 확인)
