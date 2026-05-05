# 03 — Data Synthesis: Round-by-Round History

> **APPEND-ONLY.** 이 파일은 누적 기록. 한 번 적힌 row 절대 삭제·수정 금지 (수정도 새 row).

## 1. Iterative micro-tuning workflow (사용자 명시)

> wafer 합성 데이터셋 만들 때 **시각 sample 1-2장 보고 spec 수정** 반복하는 워크플로우
> 선호. 35-36 클래스 한꺼번에 200장 생성하기 전에 **1 클래스당 1장 빠른 테스트**(2-3분)
> 로 시각 검증 → 미세 조정 → 본 생성 순.
> (출처: `~/.claude/projects/D--project-known-cnn/memory/feedback_wafer_synth_iteration.md`)

**금기 (사용자 명시)**:
1. 사용자 합의 없이 본 200장 생성 시작 금지
2. 한 라운드에 여러 spec 동시 변경 금지 (어느 변경이 효과 냈는지 분리 못 함)
3. "거의 비슷하니 통과" 식 자체 판단 금지 — 사용자 시각 합의 필수

## 2. v1 ~ v13 obj-active spec 진화 (출처: `.claude/skills/pixel-design/SKILL.md`)

| 단계 | 핵심 변경 | 사용자 발화 / 이유 |
|---|---|---|
| v1 | NEAREST upscale 폐기, 직접 합성 시작 | "NEAREST 같은 것 안 됨" |
| v2 | 직사각형 chip grid + bg 색 | "동그라미 그리지 마" |
| v3 | invalid chip 산재 추가 | 사용자 추가 요구 |
| v4 | bin 숫자 텍스트 (invalid only) | fail-map docs 매칭 |
| v5 | 9-token 파일명 + JSON 페어 | fq_missing_test 참조 |
| v6 | DEFECT_BG_DIST 분리 (object 보단 grade 1만 elevated) | 양호 영역 normal 비슷하게 |
| v7 | 3-way mixing (BG / EDGE / CENTER) | 가운데일수록 main grade 비율↑ |
| v8 | 11단계 익스포넨셜 + 객체별 center_power | 더 많이 세분화 + 가운데 밀도↑↑ |
| v9 | bank_boundary Y축 산포 (10 segments) | "라인이 균일하지 않게" |
| v10 | EDGE_DIST P(1) 75→40, BG↔EDGE 0.10-0.20→0-0.40 | "양호 영역과 절벽 부드럽게" |
| v11 | bank_boundary sigma 변경 (0.7/3.0/12.0) | center 1/4 폭, line halo 부드럽게 |
| v12 | scratch 5-15 lines, scratch_21deg 12-18 균일 간격 | "scratch는 적게 불균일, 21deg는 많이 균일" |
| v13 | Thick-Edge_invalid_main 추가 | "외곽 매우 두껍게 한 클래스" |

## 3. round 12-25: wafer-canvas 9 obj-less 진화 (출처: `docs/image-generation/CANVAS_9.md` §2)

이전 도형-stroke 8 class (`CANVAS_8.md`) 사용자 reject 후 **chip-internal alpha 매커니즘
을 wafer 6400×6400 한 번에 적용** 으로 재설계.

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
| 25 | "obj-active invalid 너무 많음, defect 비례로" | _sample_gen.py n=15 → defect.sum() * 0.15 | defect 적은 class invalid 도 적게 |

## 4. round 26-28 변경 (commit-tracked)

| Round / Date | 변경 | commit | 사용자 / 근거 |
|---|---|---|---|
| round 26 / 2026-05-05T13:54 | chip-object name change: `particle_blast` → `fork`, `scratch_21deg` → `scratch_rot` | `f359fee` | "Round 25 legacy names removed" |
| round 27 / pre-2026-05-05T14:48 | Row Y=0 fix — y 값 변하지 않고 x 값만 변함 | (in-code spec) | 사용자: "y값이 같게 하자 지금 보면 ㅡ 이렇게 되는게 아니고 y값이 변경되기도하네" |
| round 28 / 2026-05-05T14:48 | CenterCircle 새 class — alpha 재설계 | (data synth) | 사용자: "컴퍼스로 그린것같고 영역도 딱끝긴다 그라데이션이 부족하다" → angular harmonics + Gaussian+Lorentzian gradient |

### 4.1 round 27 Row Y=0 spec 정정 (사용자 발화)

> "y값이 같게 하자 지금 보면 ㅡ 이렇게 되는게 아니고 y값이 변경되기도하네"

CANVAS_9 spec 의 `Row` class 가 horizontal-locked 이어야 하는데 일부 line 의 y 값이
변경되는 sample 발생 → fix. 결과: angle ±0 rad (완전 horizontal), x 값만 random.

### 4.2 round 28 CenterCircle redesign (사용자 발화)

이전 (round 27 이전) CenterCircle 시안:
> "컴퍼스로 그린것같고 영역도 딱끝긴다 그라데이션이 부족하다"

→ 재설계 핵심:
- **angular harmonics**: ring 의 두께가 angle 별 변동 (컴퍼스 같은 깔끔 ring 회피)
- **Gaussian + Lorentzian gradient**: ring 안쪽 / 바깥쪽 모두 자연 fade (영역 딱 끊기지 않음)
- 사용자 visual confirm OK (2026-05-05T14:48)

## 5. 현재 sample 분포 (2026-05-05T15:00 기준)

`D:/project/data/wm-811k/unknown/<class>/*.png` (출처: 직접 ls 측정).

| Group | 개수 | per-class | total |
|---|---|---|---|
| obj-active (Center / Donut / Edge-Bottom / Edge-Ring / Edge-Top / Full / Thick-Edge × {bank_boundary, fork, invalid_main, scratch, scratch_rot} ⊆ 일부) | 33 | 200 | 6600 |
| Normal_bank_boundary | 1 | 1000 | 1000 |
| wafer-canvas (BrokenRing, CenterCircle, CenterDonut, CrescentArc, CrossScratch, DiagonalSmear, ParallelScratches, RingDots, Row, Starburst) | 10 | 200 | 2000 |
| Edge-Ring_bank_boundary / Edge-Ring_fork / Edge-Ring_invalid_main / Edge-Ring_scratch / Edge-Ring_scratch_rot | (포함) | 200 | (포함) |
| **Active total** | **44** | mostly 200 | **9600** |

(43 class folder + 1 Normal × different size = 9600 samples; classification + classification_chips 폴더 제외)

총 wafer image: 9600 PNG (D:/project/data/wm-811k/unknown/), 모든 class 200/200 통일
완료 (Normal_bank_boundary 만 1000) — 사용자 round 28 명시 "모든 unknown/ class 200/200
통일 완료".

## 6. 데이터 합성 entry script

| Script | 역할 |
|---|---|
| `_sample_gen.py` | obj-active 18-class generator (multiprocessing). round 25 invalid 비례 fix 포함 |
| `_sample_gen_gpu.py` | GPU 가속 generator (single-proc + ThreadPool) |
| `_sample_canvas_gen.py` | 9 obj-less wafer-canvas generator (round 12-25) |
| `_fq_metadata.py` | synthetic partid / pgm + FTN/QTN 생성 |
| `dist_learn/_dist_learn.py` | WM-811K cca/* heatmap 학습 (1회) |
| `dist_learn/_dist_learn_per_class.py` | multi-label stage 1 per-class heatmap |

## 7. 외부 참조 (read-only, 수정 금지)

- `D:/project/fail-map/` — palette / 파일명 / JSON 원본 spec
- `D:/project/data/wm-811k/cca/<Class>/*.png` — WM-811K 8 클래스 학습 데이터 (heatmap input)
- `D:/project/data/positions/fq_missing_test/` — JSON 참조 sample

## Cross-link

- 합성 spec → `docs/image-generation/{SPEC,PIPELINE,CLASSES,OUTPUT,CANVAS_9}.md`
- pixel-design v1-v13 → `.claude/skills/pixel-design/SKILL.md`
- canvas alpha lesson → `~/.claude/projects/D--project-known-cnn/memory/feedback_canvas_alpha_design.md`
- micro-tuning workflow → `~/.claude/projects/D--project-known-cnn/memory/feedback_wafer_synth_iteration.md`
- chip-object dataset 정책 → `CLAUDE.md` line 64-68 (inline crop, post-process folder-suffix 라벨링 금지)
