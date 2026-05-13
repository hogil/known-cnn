# CHANGELOG

> **APPEND-ONLY single source of truth.** 시간순 한 줄 entry. 한 줄 적힌 row 절대 삭제·수정 금지.
>
> Format: `YYYY-MM-DD[THH:MM] | <event_type> | <description>`
>
> Event types: `initial-commit` / `feature` / `code` / `restore` / `data-synth` /
> `training` / `training-done` / `analysis` / `paper` / `policy` / `round-end`

## 2026-04-30

2026-04-30 | initial-commit | known-cnn migrated from unknown-contrastive (commit 247b640, 2026-05-05 11:39 +0900 git timestamp)

## 2026-05-03

2026-05-03 | feature | V3 chipgrid 1.16M val_f1 0.9946 (oracle ceiling 0.9919 추월) — 32×32 native + one-hot 5ch obj-only (no R)
2026-05-03 | analysis | intra-distribution per-class breakdown — Edge-Bottom 0.9907 / Edge-Top 0.9954 가 V3 의 weak point
2026-05-03 | policy | 33 → 20 active + 14 archive class 결정 (V3 saturated subset 기반)

## 2026-05-05

2026-05-05T11:38 | restore | experiments/ folder restored (subset YAML + active-class YAML + fair-eval YAML, commit 6b825c7)
2026-05-05T11:45 | code | contrastive workflow refs 제거 (sister repo unknown-contrastive 가 owner, commit ed8ce07)
2026-05-05T13:53 | code | particle_blast → fork, scratch_21deg → scratch_rot rename (round 26 spec, commit f359fee)
2026-05-05T14:06 | feature | compound-loop orchestration team 도입 — 4 agents (orch-master / cnn-master / result-trace / compound-review) + slash /compound-loop (commit c3356f9)
2026-05-05T14:20 | training | chip CNN v3 dispatched (cnn-master + resource-monitor cnn-team) — cnn_train_chip.py n=100/cls subset, 5 class, batch 16, epoch 30, RTX 4060 Ti 16GB
2026-05-05T14:27 | training-done | chip CNN v3 best_model.pth val_f1 1.0000 / test_f1 0.9799 (6.7 min, ep 3 best of 10 ep early stop) — logs_chip/v3_chip_260505_142036_running/
2026-05-05T14:30 | data-synth | canvas v2 dispatch (Row + Starburst + CenterCircle 200 each)
2026-05-05T14:30 | training | obj_id_maps build dispatched (compound_train/_build_obj_id_maps.py --batch 64) — chip CNN v3 forward inline → wafer 마다 32×32 obj_id .npy
2026-05-05T14:48 | data-synth | CenterCircle round 28 alpha redesign + 사용자 OK (visual confirm — angular harmonics + Gaussian+Lorentzian gradient, 사용자 quote: "컴퍼스로 그린것같고 영역도 딱끝긴다 그라데이션이 부족하다")
2026-05-05T15:00 | data-synth | 모든 unknown/ class 200/200 통일 완료 — 43 class × 200 + Normal_bank_boundary × 1000 = 9600 wafer (D:/project/data/wm-811k/unknown/)
2026-05-05T15:15 | code | LR policy update (AD reference): lr_backbone 1e-5 → 2e-5, lr_head 1e-3 → 2e-4, warmup_epochs 2 → 5, LinearLR start_factor 0.1 → 0.05 — cnn_train_chip.py / cnn_train_wafer.py / cnn_train_compound.py 3 trainer 모두 (참조: D:/project/anomaly-detection/train.py:1497-1530)
2026-05-05T15:30 | paper | paper-scribe agent + docs/paper/ skeleton bootstrap — 9 markdown files (00_abstract / 01_motivation / 02_method / 03_data_synthesis / 04_experiments / 05_results / 06_analysis / 07_iteration_log + CHANGELOG.md), 13 feedback memory mining 완료

## 2026-05-14

2026-05-14 | paper | Hybrid CNN + obj_id_map prototype matching method proposal (round 29, 사용자 발화 2026-05-14) — 02_method.md §6 (Motivation / 확률분포 / 매칭방식 / alternative rules / 평가 protocol / 구현 plan / 본질과의 관계 7 sub-section) + 00_abstract.md v4 addendum 추가. 확률 분포 = per-class per-cell categorical histogram (P̂(obj_id=k | y, (i,j)), Laplace α=1, tensor (C, 32, 32, K+1) ≈ 1 MB, V3 best_run train split 만 사용). 매칭 방식 = confidence-gated max-likelihood Bayesian posterior (p_max < τ_gate → log_post = log p_cnn + λ · log L 의 argmax, 그 외 CNN 결과). Hyperparam (τ_gate, λ, α) val grid 5×5×3 = 75 cell, test 누수 금지. λ=0 = CNN only 회귀 → worst case = V3 그대로 (net negative 불가능). architecture-independent 후처리 — 본질 (compound > wafer-only) 와 독립.
2026-05-14 | paper | Hybrid method 초보자용 시각화 4 figure 추가 (hybrid_match/_paper_fig_hybrid.py 스크립트로 생성, docs/paper/figures/ 산출): hybrid_fig1_overview.png (실제 V3 오답 wafer + obj_id_map + 4 클래스 P̂ + CNN vs hybrid posterior bar), hybrid_fig2_distribution.png (6 클래스 P̂ argmax view + confidence view), hybrid_fig3_matching.png (한 wafer 4 단계 step-by-step), hybrid_fig4_flowchart.png (gating + Bayesian posterior 의사결정 flowchart). 02_method.md §6.1.1 / §6.2 끝 / §6.3 끝 세 위치에 figure 임베드 + 한글 초보자 caption (어떤 색 = 어떤 의미 / 어떤 cell = 어떤 신호 / Laplace α 효과 / step-by-step take-away). 실제 wafer PNG = outputs/logs_wafer/overall/wrong/val/Edge-Bottom_bank_boundary/Edge-Bottom_scratch/yvo668_*.png 사용 (V3 가 틀린 실제 case, hybrid 회복 target). obj_id_map 은 illustrative simulation (chip CNN 패턴 모사, 실제 .npy cache 가 이 머신에 없어 시각화용 only).
2026-05-14 | paper | 비AI 심사위원용 시각화 2 figure 추가: hybrid_fig5_zoomed_example.png (확대 32x32 obj_id_map + cell-level callout 2 개 + 우측 explainer 박스 "wafer 가 32x32=1024 chip 으로 나뉘고 chip CNN 이 6 카테고리 분류한 결과가 색깔 grid"), hybrid_fig6_confusion_pairs.png (사람 눈에도 비슷한 4 쌍: same spatial different obj / same obj different spatial / donut family / rotation variant — 각 쌍 두 obj_id_map + similarity 박스 + difference 박스). 02_method.md §6.1.1 재구성 — figure 순서표 (5→6→2→1→3→4) 로 비AI 독자가 입력 형태 → 왜 어렵나 → fingerprint → 회복 → 의사결정 흐름으로 점진적 이해. Figure 6 pair 4 가 TTA 금지 정책 (feedback_no_tta_wafer.md) 의 직접 이유 (rotation variant 식별) 임을 명시.
