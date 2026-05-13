# iter125-127 — FCM-PM (g, n) parameterization extension plan

## 컨텍스트

iter124 (9 cells) 가 GRID = g × n 의 첫 sweep 을 완료:
- g=2 axis: n ∈ {1, 2, 3, 4} — 4 cells (124a-d)
- g=3 axis: n ∈ {1, 2, 3} — 3 cells (124e-g)
- bisect_h, bisect_v — 2 cells (124h-i)

**남은 paper §5.47 빈 칸**:
1. g=4, 5 axis 미답 (4n × 4n, 5n × 5n grids)
2. n=5, 6 등 high-n region 미답
3. 같은 GRID 에서 (g, n) 변형 (label cardinality vs spatial 진짜 isolated comparison)
4. 최고 cell 의 multi-seed variance (paper rigor)
5. 최고 (g, n) × 다른 axis (cutmix-p, pair_fill, LS) 조합

---

## iter125 — Phase A+B+C: (g, n) map 완성 (6 cells, ~50 min)

### Phase A — g=4 axis 신설 (3 cells)

| cell | g | n | GRID | total cells | cells/group | 셀 px (384÷GRID) |
|---|---:|---:|---:|---:|---:|---:|
| **125a** | 4 | 1 | 4 | 16 | 4 | 96×96 |
| **125b** | 4 | 2 | 8 | 64 | 16 | 48×48 |
| **125c** | 4 | 3 | 12 | 144 | 36 | 32×32 |

→ 25:25:25:25 area split (A 1/4, B 3/4 — 또는 그 반대). label cardinality 4 의 효과.

### Phase B — high-n 보강 (2 cells)

| cell | g | n | GRID | 의미 |
|---|---:|---:|---:|---|
| **125d** | 2 | 5 | 10 | 100 cells, 38×38 — g=2 의 high-n 끝 |
| **125e** | 3 | 4 | 12 | 144 cells, 32×32 — g=3 의 high-n 끝 |

### Phase C — Triple matched-GRID 비교 (1 cell, +기존 cells 활용)

★ **paper §5.47 의 핵심 figure**: 같은 GRID 다른 g — label cardinality 의 isolated 효과

| GRID | cell | g | n | already done? |
|---:|---|---:|---:|---|
| **6** | 124c | 2 | 3 | ✓ iter124 |
| **6** | 124f | 3 | 2 | ✓ iter124 (★ key pair) |
| **12** | 125c | 4 | 3 | **이번 iter125** |
| **12** | 125e | 3 | 4 | **이번 iter125** |
| **12** | **125f** | **2** | **6** | **이번 iter125 (추가)** |

→ **GRID=12 triple-matched** (g=2/3/4) 가 새 데이터 포인트. paper figure 의 X-축 2개 (GRID=6 와 GRID=12) 에서 group 효과 측정.

**iter125 sweep 총 6 cells**:
- 125a: g=4 n=1
- 125b: g=4 n=2
- 125c: g=4 n=3 (triple-match GRID=12)
- 125d: g=2 n=5
- 125e: g=3 n=4 (triple-match GRID=12)
- 125f: g=2 n=6 (triple-match GRID=12)

iter125 후 paper §5.47 의 final figure:
- (g, n) heatmap: g={2,3,4} × n={1,2,3,4,5,6} → 18 cells 中 14 cells 측정 (iter124 7 + iter125 6 + 미답 4)
- triple-matched GRID 비교 (GRID=6 pair + GRID=12 triple)

---

## iter126 — Multi-seed variance (paper rigor, 6 cells, ~50 min)

iter124 + iter125 의 **top 2 cells** 를 3-seed 로 재학습 → mean ± std 표시.

| cell | seed=1 | seed=2 | seed=3 |
|---|---|---|---|
| top1 (winner) | 이미 학습 (iter124/125) | new | new |
| top2 (runner-up) | 이미 학습 | new | new |

→ 총 6 new trains. 단일 seed 우연 효과 배제, paper variance 보고.

---

## iter127 — Cross-axis combination on best (g, n) (6-8 cells)

iter124+125+126 winner 의 (g, n) 고정 + 다음 axis 조합:

### Phase A — cutmix-p sweep (winner × p ∈ {0.10, 0.15, 0.20, 0.25, 0.30}, 5 cells)
- iter116J recipe 의 default p=0.25 가 sweet spot 일 가능성 검증
- pending task #130 (iter117) 완성

### Phase B — pair_fill mode (winner × {corner, center, random}, 3 cells)
- corner (현 default) vs random fill 의 partner-bit suppression 차이

→ 총 8 cells, ~70 min.

---

## 의사결정 트리

```
iter124 b-i (analyst BG, 8 cells re-eval)
  ↓
어느 cell 이 winner?
  ↓
  ├── 124c (g=2 n=3) winner: iter125 Phase B/C 우선 (g=2 axis 정밀화)
  ├── 124f (g=3 n=2) winner: iter125 Phase A 우선 (g=4 도전)
  ├── 124b/d/g winner: iter125 그대로 6 cells
  ├── 모두 regression (≤ iter112 0.9964): 
  │     iter125 skip, 즉시 다음 axis (iter117 cutmix-p sweep 또는 iter122 data expansion) 으로 pivot
  └── bisect_h/v winner: 다른 ablation (rotation, scale) 로 분기
```

---

## 실행 trigger 조건

- iter124 b-i analyst eval 완료 (BG, ~20-30 min 추가)
- 종합 표 확인 → winner 식별
- iter125 또는 pivot 결정

## 출력 위치 (전부)

- 학습: `outputs/iter125_{a..f}_*/T7_*/best_model.pth`
- eval: `outputs/iter125_{a..f}_*/T7_*/eval_v15direct/stage1_*/`
- summary: `outputs/_iter125_summary.log`
- docs: `docs/chip-multilabel/03_ablations.md` (iter125 섹션 추가)
- paper: `docs/chip-multilabel/paper/05_experiments.md` §5.47 표 row 추가

## 절대 룰 (260512) 준수 모든 iter

- 학습 4 single defect only (`--no-normal`)
- eval = single + 2-combo + Normal + Invalid + OOD
- bit_F1 = positive macro, Total FAR = (Normal+Invalid+OOD) FP rate
- TTA / rotation aug 금지
