# Multi-label Wafer Classification — Design Overview

이 문서는 single-label 학습된 wafer 33-class 분류기를 **multi-label 추론** 환경으로
확장하기 위한 8-stage paper-style ablation study 의 이론·설계 문서다. plan 자체는
`~/.claude/plans/1-input-batch-hidden-patterson.md` 에 있고, **이 docs/ 는 plan 의 이론
배경과 설계 근거** (왜·무엇·어떻게)를 담는다.

## 위치

| 파일 | 역할 |
|---|---|
| `docs/multi-label/README.md` | 이 파일 (개요·인덱스) |
| `docs/multi-label/THEORY.md` | multi-label / SPML / calibration / density 이론 + 수식 |
| `docs/multi-label/LOSS_DESIGN.md` | **★ deep-dive: loss 단일 + mix 조합 매트릭스** |
| `docs/multi-label/MATCHING_DESIGN.md` | **★ deep-dive: chip-wafer matching surface ensemble + CRF** |
| `docs/multi-label/DECISION_RULE.md` | **★ deep-dive: multi-label 판정 (threshold + calibration + top-K mix)** |
| `docs/multi-label/PAPERS.md` | 인용 논문 list + 우리 도메인 적용 |
| `docs/multi-label/EXAMPLES.md` | benchmark 사례 (MixedWM38 / COCO / ChestX-ray) + 실측 수치 |
| `docs/multi-label/STAGES.md` | 8 stage motivation + 가설 + 기대 효과 |
| `docs/multi-label/STATUS.md` | 진행 상태 + 산출 path |
| `~/.claude/plans/1-input-batch-hidden-patterson.md` | plan (실행 detail) |
| `.claude/skills/multi-label-ablation/SKILL.md` | 실행 패턴 + sweep range + 최적값 찾는 방법 |
| `.claude/agents/multi-label-ablation.md` | stage orchestrator agent spec |

★ 표시 3 문서가 본 ablation 의 **진짜 핵심 contribution** — 사용자 결정 우선순위:
loss 설계 + matching ensemble + multi-label 판정 mix.

## 본 ablation 의 핵심 질문

```
Single-label trained ConvNeXtV2 base FCMAE (val_macro_f1 ≈ 0.97) 모델을
multi-label production 환경에 적용할 때:

   Q1. 학습 추가 없이 (sigmoid heuristic) 어디까지 가능한가?
   Q2. 학습 추가 (AdaGC / BCE / ASL) 시 mAP 얼마나 회복?
   Q3. threshold + calibration mix 의 best 조합은?
   Q4. chip-wafer matching 의 best surface + ensemble 은?
   Q5. multi-label 판정 의 best decision rule (threshold vs top-K)은?
```

각 stage 가 한 question 이상 검증.

## 핵심 가설

- **H1**: single-label trained ConvNeXtV2 의 logit ranking 이 multi-label co-occurrence 를 부분적으로 반영. sigmoid + 적절한 threshold 면 multi-label F1 0.65+ 달성 (학계 oracle 대비 70-80%)
- **H2**: 학습 시 class imbalance 보강 (weighted BCE / focal / ASL) 이 multi-label F1 +3-7%
- **H3**: chip 위치 분포 기반 heatmap matching 으로 chip-wafer assignment accuracy 80%+ 달성 (별도 학습 X)
- **H4**: per-class F1 threshold sweep + Temperature scaling 조합으로 default 0.5 대비 macro F1 +5-10%
- **H5**: AdaGC (single-positive multi-label) 로 BCE 대비 mAP +3-5%, oracle full-multi-label 의 90%+ 회복
- **H6** (★ mix 가설): **ASL + label_smoothing + class_weight + focal** 의 mix 가 단일 ASL SOTA 대비 추가 +2-5% 가능
- **H7** (★ mix 가설): **heatmap_smooth + GMM ensemble + CRF post-process** 가 단일 hybrid 대비 chip matching accuracy +3-5%
- **H8** (★ mix 가설): **per-class threshold + Temperature + IDF + top-K floor** 의 mix 가 단일 strategy 대비 macro F1 +2-4%

## 데이터

| 종류 | path | 라벨 |
|---|---|---|
| single-label 합성 (학습) | `D:/project/data/wm-811k/unknown/<class>/*.png` (33 class) | 1 wafer = 1 class |
| **multi-label 합성 (ablation 평가)** | `D:/project/data/wm-811k/unknown_multi/*.png` (Stage 3 산출) | 1 wafer = 1~3 distribution + 1~3 object |
| chip-object | `D:/project/data/wm-811k/classification_chips/<obj>/*.png` (5 obj) | 1 chip = 1 obj |

## 모델

ConvNeXtV2 base FCMAE pretrained → 33-class supervised CrossEntropy on `unknown/`.
Production checkpoint: `logs_compound/overall/best_model.pth`
(또는 `logs_wafer/overall/best_model.pth` for R-only).

## 8 Stage 요약

```
┌────────────────────────────────────────────────────────────────────┐
│ Stage 1   분포 학습 ablation        — chip 좌표 surface 5 method   │
│ Stage 2   Hyperparameter ablation  — class_weight × ls × loss     │
│ Stage 3   unknown_multi/ 합성       — multi-label evaluation GT   │
│ Stage 4 ★ Multi-label 추론 path    — Phase A/B/C (loss 핵심)      │
│ Stage 5 ★ Threshold tuning         — 5 strategy + mix             │
│ Stage 6 ★ Chip-wafer matching      — surface ensemble + CRF       │
│ Stage 7   Prod predict 보강         — 2 parquet + ablation default│
│ Stage 8   Master comparison         — paper-style table + figure  │
└────────────────────────────────────────────────────────────────────┘

★ = 본 ablation 의 가장 중요한 stage (사용자 우선순위)
```

## Stage 와 deep-dive doc 의 관계

| Stage | 관련 deep-dive doc | 핵심 |
|---|---|---|
| Stage 2 / Stage 4 Phase B/C | LOSS_DESIGN.md | loss 단일 + mix 조합 |
| Stage 4 Phase A / Stage 5 | DECISION_RULE.md | sigmoid + threshold + calibration mix |
| Stage 1 / Stage 6 | MATCHING_DESIGN.md | surface 학습 + ensemble + CRF |
| Stage 3 | (THEORY.md 의 augmentation 섹션) | wafer-aware multi-pattern 합성 |
| Stage 7 | (skill 의 prod predict pipeline) | production deployment |
| Stage 8 | (skill 의 master report generator) | paper-style 종합 |

## Quick links

- 빠른 실행: `.claude/skills/multi-label-ablation/SKILL.md` "Quickstart" 섹션
- agent 호출: `Agent({subagent_type: "multi-label-ablation", prompt: "Stage 3 부터 실행"})`
- 전체 plan: `~/.claude/plans/1-input-batch-hidden-patterson.md`

## 외부 참조 (필독)

| 경로 | 역할 |
|---|---|
| `docs/image-generation/` | 합성 데이터 base spec — multi-label 합성 (Stage 3) 의 base |
| `cnn_train.py`, `cnn_train_compound.py` | 학습 engine — Stage 2 / 4B / 4C 의 base |
| `cnn_predict_compound.py`, `cnn_predict_compound_prod.py` | 추론 engine — Stage 7 의 base |
| `_dist_learn.py`, `_dist_heatmaps/` | 8-class distribution heatmap (Stage 1 의 base) |

## 수정 금지

- 본 docs 를 변경할 때는 plan 의 stage section 도 같이 갱신 (3-way sync)
- 학계 paper citation 정확성 유지 — 임의 추가 / 임의 삭제 금지
- ★ deep-dive 3 문서의 mix 조합 matrix 는 사용자 결정 사항 — 임의 변경 금지
