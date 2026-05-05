# 00 — Abstract

> **APPEND-ONLY.** 이 파일은 누적 기록. 한 번 적힌 paragraph 절대 삭제·수정 금지 (수정도 새 paragraph 로 추가).

## v3 abstract (2026-05-05, bootstrap)

본 연구는 WM-811K wafer fail-bit 데이터의 합성 + supervised CNN 분류 pipeline 을 제안한다.
8 개 wafer-level distribution heatmap (Center / Donut / Edge-Ring / Edge-Loc / Loc /
Random / Near-full / Thick-Edge — `_dist_heatmaps/<Class>_p_defect_32.npy`, 출처:
`docs/image-generation/PIPELINE.md`) 을 WM-811K cca/* 에서 학습하고, 각 chip 안 200×200
픽셀에 대해 **chip-internal alpha 매커니즘** (Lorentzian sharp + heavy tail sum,
출처: `docs/image-generation/CANVAS_9.md`) 으로 5 종 chip-object (bank_boundary /
fork / scratch / scratch_rot / invalid_main, round 26 이후 명명, 출처: commit `f359fee`)
및 9 wafer-canvas pattern 을 mix 합성한다. 결과 dataset = 6400×6400 palette PNG +
positions JSON (chip 단위 BIN, FTN/QTN, 출처: `docs/image-generation/SPEC.md`).

학습 architecture 는 **3-stage** chain 이다:
(1) chip 5-class CNN (`cnn_train_chip.py`) 이 200×200 chip crop 을 학습,
(2) `chip_tools/_build_obj_id_maps.py` 가 wafer 마다 32×32 obj_id .npy map 생성,
(3) `cnn_train_compound.py` 가 R = failbit / G = obj_id (block_expand_2d categorical
resize 정책, 출처: `feedback_block_expand_only.md`) / B = zero 의 3-channel wafer
입력으로 ConvNeXtV2-base 88M (FCMAE in22k_in1k_384, 출처: `models/`) 을 fine-tune.

본 프로젝트의 **본질 (출처: `docs/paper/README.md` line 8)** = compound (R+G+B) test_f1
가 wafer-only (R 만) test_f1 을 초과해야 한다. chip-level obj 정보가 wafer-level
분류에 보탬 되는지 검증.

iterative orchestration: `orch-master` agent 가 `cnn-master` / `stage3-compound` /
`result-trace` / `compound-review` / `cnn-analyze` / `paper-scribe` 를 round 별
chain dispatch (출처: `.claude/agents/orch-master.md`), v_n → v_{n+1} 자동 반복하며
margin gate (default 1.5pp, 2-round 연속 유지) 도달까지 spec 변경 + 학습 + 비교 +
fix 반복.

현재 진행: round 28 chip CNN v3 학습 완료 (val_f1 1.0000 / test_f1 0.9799, 6.7 분,
출처: `logs_chip/v3_chip_260505_142036_running/best_history.txt`), obj_id_maps build
~92.6% (7965 / 8600 .npy, 출처: `data/wm-811k/obj_id_maps/`), wafer-only / compound
학습 미실행 (TBD).
