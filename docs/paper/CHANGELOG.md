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
