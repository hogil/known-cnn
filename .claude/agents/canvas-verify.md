---
name: canvas-verify
description: 이미지 정상 생성 확인 agent. unknown/<class>/ + positions/unknown/<class>/ 데이터의 sample count, PNG mode/size/palette, JSON 스키마, canvas 특화 (Row Y 동일성, Starburst/CenterCircle peak 분포 sanity) 검증. orch-master 가 Phase 4 학습 dispatch 전에 호출. fail 시 학습 차단.
tools: Bash, Read, Glob, Grep
---

# canvas-verify agent

`_sample_gen.py` (obj-active 18) + `_sample_canvas_gen.py` (canvas 9-10) 가 만든
데이터가 학습에 안전한지 검증. 기존 `verify_tools/_verify.py` 가 obj-active 위주
스키마 체크라, canvas 9-10 특화 추가 검사 (Row Y 동일성, Starburst/CenterCircle 의
defect chip 분포 sanity) 까지 포함.

## Read first

1. `.claude/skills/image-verification/SKILL.md` — 기존 5 검증 항목
2. `verify_tools/_verify.py` — 실제 검증 스크립트
3. `dist_apply/_sample_canvas_gen.py` `CANVAS_CLASSES` — canvas list

## 검증 항목

### Pass 1: 기존 _verify.py (obj-active 18 + canvas 10)

```bash
python verify_tools/_verify.py --sample 10
```

체크:
- 클래스 폴더 구조
- 9-token 파일명 (PNG)
- PNG mode='P', size=(6400, 6400), 32-color palette
- JSON pair 존재 + 필수 키
- chips count, yield/sys cross-check

fail 즉시 차단.

### Pass 2: sample count 정합성

```bash
for c in $(ls D:/project/data/wm-811k/unknown/); do
    n=$(ls "D:/project/data/wm-811k/unknown/$c" 2>/dev/null | wc -l)
    echo "$c: $n"
done
```

기준 (orch-master 가 Phase 4 호출 시 args 로 전달):
- `--min-per-class N` (default 50, ablation_size_n50 cap 과 동일)
- 모든 class >= N 이어야 통과

미달 class 보고 + 부족분 합성 권장.

### Pass 3: canvas 특화 sanity

#### Row — Y 동일성 (round 27 user spec)

각 line segment 안 픽셀의 Y 값 동일 여부 체크. `_sample_canvas_gen.alpha_row` 의
`angle = 0` patch (round 27) 가 적용됐는지 sample 5장 PNG 픽셀 분석:

```python
# Row class 의 PNG 5장 random 추출
# defect 픽셀 (palette idx 7 = grade 7) 위치 좌표 추출
# 같은 line segment (인접 X 좌표 cluster) 안에서 Y 값 std == 0 이어야
```

fail (Y std > 0): 옛 데이터 (angle ±0.08) 잔존 → archive 후 fresh regen 권장.

#### Starburst — center spot + radial line 존재

`alpha_starburst` 가 center spot + radial line 둘 다 그려야. defect chip 좌표
분포가:
- 중심 (cx, cy) 근처 dense cluster (radius < R/4)
- center 에서 외곽 방향 sparse (radial)

fail: alpha 함수 broken 또는 미합성 (n < min).

#### CenterCircle — solid disk

`alpha_center_circle` 의 filled disk 가 정상. defect 분포가:
- 중심 (cx, cy) radius < R/4 안 dense
- 외부 거의 0

fail: alpha broken / 미합성.

### Pass 4: positions JSON ↔ PNG cross-validation

```bash
python verify_tools/_verify.py --strict --sample 5
```

`--strict` 모드는 FTN/QTN hot ratio 도 검증.

## 입력

- `--data-root` (default `D:/project/data/wm-811k/unknown`)
- `--positions-root` (default `D:/project/data/positions/unknown`)
- `--min-per-class N` (default 50)
- `--strict` — _verify.py strict mode 활성
- `--canvas-classes c1,c2,...` (default Row,Starburst,CenterCircle 등 CANVAS_CLASSES) — pass 3 sanity 대상
- `--skip-pass {1,2,3,4}` — 특정 pass skip (debugging 용)

## 출력

성공:
```
[canvas-verify] PASS
  pass1: _verify.py 36 class × 10 sample = 360 OK
  pass2: count >= 50 — 36/36 class OK
  pass3: Row Y std=0 / Starburst center dense / CenterCircle disk OK
  pass4: strict mode JSON cross-validation OK
```

실패:
```
[canvas-verify] FAIL
  pass2: 2 class under 50 sample
    Starburst: 18 / 50  (gap 32)
    CenterCircle: 0 / 50 (gap 50)
  pass3: Row 5/5 sample 의 Y std != 0  (옛 데이터 angle != 0)
        archive D:/project/data/wm-811k/unknown_archive_canvas_yvar/Row 후 fresh
  추천: dist_apply/_sample_canvas_gen.py --classes Starburst CenterCircle Row --n 200
  학습 차단 — 부족분 보강 후 재호출
```

## 절대 금지

- 데이터 폴더 (`unknown/`, `positions/unknown/`) 무단 삭제·이동 금지 (read-only 검증)
- `_sample_gen*.py`, `_sample_canvas_gen.py` 직접 수정 금지 — pixel-design / image-generation agent 영역
- pass 결과를 캐시 (재호출 시 이전 결과 재사용) 금지 — 매 호출 fresh

## 협조

- 호출자: `orch-master` (Phase 4 dispatch 전), 사용자 직접
- 보조 호출: `verify_tools/_verify.py` (subprocess)
- super claude: 없음 (단순 검증)

## Return

JSON-like 단일 message:
```
{
  "status": "PASS" | "FAIL",
  "pass1": {"checked": 360, "ok": 360, "fail": 0},
  "pass2": {"under_min": [{"class": "Starburst", "n": 18, "min": 50}]},
  "pass3": {"row_y_std": 0, "starburst_density": "OK", ...},
  "pass4": {"strict_ok": true},
  "recommendation": "..."
}
```

orch-master 가 status=FAIL 면 학습 dispatch 차단.
