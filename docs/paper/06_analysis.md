# 06 — Analysis: Policies, Lessons, Weak Points

> **APPEND-ONLY.** 이 파일은 누적 기록. 한 번 적힌 row/section 절대 삭제·수정 금지.

## 1. Policy lessons — 누적 사용자 feedback (13 memory)

13 feedback memory 파일 (출처: `~/.claude/projects/D--project-known-cnn/memory/feedback_*.md`)
각 1-paragraph summary + quote.

### 1.1 Block expand only (categorical resize)

obj_id (32×32 categorical) / one-hot binary / probability 등 **categorical map 의 spatial
resize 는 `_chipgrid_resize.block_expand_2d` 만 사용**. PIL/torch 의 BICUBIC, NEAREST,
F.interpolate(...) 모두 코드 hardcode 금지. (출처: `feedback_block_expand_only.md`)

> "BICUBIC: chip 경계 fractional 보간 → 정수 카테고리 0/1/2/3/4/5 가 1.3, 2.7 같은 무의미한
> 실수로 변환. 모델 입력에서 categorical semantic 완전 깨짐."

이 정책이 V3 chipgrid (val_f1 0.9946) 의 enabling factor. compound R+G+B 384 BICUBIC
ceiling 0.9784 (val) 대비 V3 의 +0.16 pp = **error 75% 감소**. 새 trainer 작성 / 기존
trainer 수정 시 BICUBIC/NEAREST 발견하면 즉시 보고 + block_expand_2d 패치.

### 1.2 No TTA (wafer)

Wafer 33-class 분류기 inference 시 **어떤 형태의 TTA 도 금지**: rotation/flip ensemble,
multi-scale TTA, multi-crop TTA. wafer class identity 가 angle/위치에 직접 묶여있어
augmented 복사본의 답을 ensemble 하면 다른 class 답을 평균 내는 잘못된 결과. (출처:
`feedback_no_tta_wafer.md`)

> "scratch_rot 21° → rotation 90° 적용 시 scratch (수직 라인) 처럼 보임 → 다른 class 답
> ensemble. VFlip → Edge-Top → Edge-Bottom 으로 의미 변경 → 두 다른 class 답 ensemble."

학습 augmentation 도 동일: ±15° rotation only (stage 회전 오차 모사 범위 내). HFlip /
VFlip / 180° / ColorJitter / MixUp / CutMix / Cutout 모두 금지.

### 1.3 Fair-eval protocol (모든 backbone 비교)

> "평가는 항상 같은 wafer수 chip수 학습과 predict로 해야하는거 알지? epoch도 똑같이
> 돌리고 best model 비교."

(출처: `feedback_fair_eval_protocol.md`)

한 표 안 row 들은 동일 protocol. split / active class / epoch / sample 다르면 별 표
분리, header 에 protocol 명시. spec yaml: `experiments/fair_eval_protocol.yaml`. 통계적
claim 시 5-seed `[42, 1, 7, 100, 234]` mean ± std 필수.

### 1.4 Active class policy

33-class → 20 active + 14 archive. **데이터 자체는 자산 — 재합성 cost ≥ 수 시간. 학습
안 쓴다고 삭제 X.** archive 14 class 는 `unknown_archive/` 로 copy 보존, 원본 unknown/
도 그대로. `EXCLUDE_CLASSES` 같은 hardcoded list 에 새 class 추가 금지 (active YAML
사용). (출처: `feedback_active_class_policy.md`)

### 1.5 Canvas alpha design (round 12-25 + round 28)

> "도형 stroke (PIL Draw line) reject" (round 12)
> "Gaussian 너무 넓음, sharp peak + heavy tail 필요" (round 18-19)
> "두 분포 sum 으로 narrow + wide 동시" (round 20)
> "line 직접 지나가는 chip 만 BIN, 외곽 normal" (round 23)

(출처: `feedback_canvas_alpha_design.md`)

핵심 lesson:
1. wafer 합성은 **자연 noise field** 에 alpha 분포로 grade 변형 (도형 stroke X)
2. **Lorentzian sharp + heavy tail sum** 만이 양 끝 자연 0 fade 동시 만족
3. **alpha = baseline ↔ peak mix weight** (cum_mixed) — cut 절벽 X
4. **chip border decision = alpha mean primary filter** (max only X)
5. invalid 비례 fix (round 25) — `n=15 fixed` → `defect.sum() * 0.15`

round 28 추가: CenterCircle redesign — 사용자 "컴퍼스로 그린것같고 영역도 딱끝긴다
그라데이션이 부족하다" → angular harmonics + Gaussian+Lorentzian gradient, ring
두께가 angle 별 변동 + 안쪽/바깥쪽 자연 fade.

### 1.6 Input resolution = 1024 BICUBIC

CNN 입력 cache 는 **1024×1024 BICUBIC**, 학습 size 도 1024. 더 작은 size (384/512/768)
는 line/blob 모양 손실로 불충분. chip-aware (외곽 NEAREST + 내부 BOX) — 1024 에서는
일반 보간과 차이 미미하므로 폐기. **chip border 신호 자체가 분류에 불필요** (이미
라벨에 있는 정보). (출처: `feedback_input_resolution_decision.md`)

> "6400→1024 (6.25×): chip 32×32 px, line 1px → 3% stripe → 명확히 보임, ORIG 와 시각적
> 거의 동일."

### 1.7 Windows python dispatch — 좀비 누적 + torch hang

PowerShell `Start-Process -WindowStyle Hidden -FilePath python` 으로 학습 dispatch 시
detached python.exe 좀비 누적 (CPU 0.27s, WS 30~40MB inert). 누적 20+ 개 되면 새
`python -c "import torch"` 가 90s 도 안 끝남 — DLL 로드 hang. 좀비 다 kill 해도 일부
case 는 시스템 reset 까지 안 풀림. **Bash run_in_background 만 사용**. (출처:
`feedback_windows_python_dispatch.md`)

### 1.8 Multi-label priority (3 핵심)

> "loss 부분과 chip class 로 wafer class matching 하는 부분이 이론과 여러 기법 mix 등
> 굉장히 중요해 보인다 관건이다. 그리고 multi-label 판정 방식도."

(출처: `feedback_multi_label_priority.md`)

8 stage 중 다음 3 영역이 ablation 의 진짜 가치:
1. **Loss 설계** (CE / focal / BCE / ASL 단일 비교 X — mix 가능성: ASL + label_smoothing
   + class_weight + focal 결합)
2. **Chip-wafer matching** (heatmap / GMM / KDE / hybrid 단일 비교 X — surface ensemble
   + CRF post-process + Mahalanobis + consistency)
3. **Multi-label 판정** (sigmoid heuristic 단일 X — threshold + Temperature + IDF +
   KNN_local + top-K floor mix)

### 1.9 Wafer synth iterative micro-tuning

새 spec 변경 (1-3줄) → `_sample_gen.py` 즉시 수정 → 35 클래스 1장씩 빠른 생성 → 사용자
1-2개 대표 클래스 sample 이미지 합의 → 본 200장 background 생성. 한 라운드에 여러 spec
동시 변경 금지 (어느 변경이 효과 냈는지 분리 못 함). "거의 비슷하니 통과" 식 자체 판단
금지. (출처: `feedback_wafer_synth_iteration.md`)

### 1.10 GPU busy → CPU fallback

> "중간 중간 gpu 사용하고 있으면 cpu로 돌리고" (사용자 명시 2026-04-21)

학습/eval subagent dispatch 시 GPU state check (`nvidia-smi` 또는
`torch.cuda.memory_allocated()`) → busy 면 `--device cpu` fallback. polling 으로 GPU
free 까지 기다리지 X. CPU 는 ~5-10× 느리지만 acceptable. (출처:
`feedback_gpu_cpu_fallback.md`)

### 1.11 Honor task blockedBy

`TaskGet` / `TaskList` 의 `blockedBy` dependency 가 still open 인 상태로 task 시작 금지.
2026-04-20 Task #7 (persist `emb.npy` in `contrastive.py`) 를 Task #4 active 인 채로
ship 한 사례 — 동작은 깨지지 않았지만 process violation. **자기 판단으로 "dependency 가
적용 안 됨" 결론 X**, team-lead 에 unblock 요청. (출처: `feedback_honor_task_blockedby.md`)

### 1.12 None class — train + inference 유지, metric 만 drop

unknown-contrastive pipeline (sister repo) 에서 `none` class 는 train + inference 모두
포함하고 ARI/NMI/purity/silhouette **metric computation 만 GT=none drop**. `none` 이
~80% dataset 이라 그 제거는 encoder/clusterer 가 full distribution 못 보게 됨. 의미
있는 신호는 `eval_summary_*_defect_only.json` 에서 측정. (출처: `feedback_none_class_eval.md`)

(known-cnn 에서는 Normal class = 학습 제외, inference 시 max_prob threshold 로 unknown
처리 — 다른 정책. 출처: `CLAUDE.md` line 8-10)

### 1.13 Contrastive backbone = TAPT (cnn_train.py best_model.pth)

contrastive.py (sister repo unknown-contrastive) 의 backbone init 은 ImageNet FCMAE
직접이 아니라 cnn_train.py 33-class supervised 학습된 best_model.pth. 같은 wafer 데이터
라 supervised collapse 우려 < 도메인 정렬 이득. ConvNeXtV2 base capacity (88M) 면
collapse 우려 무시 가능. (출처: `feedback_contrastive_backbone_tapt.md`)

(known-cnn 에서는 직접 supervised, contrastive sister repo 의 init 정책)

## 2. Intra-distribution weak point — V3 obj-only 의 ceiling

(출처: `docs/wafer-ensemble/INTRA_DIST.md`)

### 2.1 측정 방법

1. wafer class name → (distribution, object) parse:
   - `Donut_scratch_rot` → (`Donut`, `scratch_rot`)
   - `Starburst` → (`Starburst`, None)
2. distribution accuracy = predicted distribution == GT distribution
3. **per-distribution obj_acc** = 그 distribution 안에서 obj 식별 정확도

### 2.2 핵심 표

per-distribution object accuracy (n=220, seed 42, 0.8/0.2 val 1420 protocol):

| Distribution | R-only ConvNeXt 88M | obj-only 4-layer 0.4M | Tier1 ens α=0.35 | V3 obj-only 1.16M |
|---|---|---|---|---|
| Center | 1.0000 | 0.9954 | 1.0000 | 1.0000 |
| Donut | 1.0000 | 1.0000 | 1.0000 | 0.9954 |
| **Edge-Bottom** ★ | 0.9630 | 0.9537 | 0.9583 | **0.9907** |
| Edge-Ring | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| **Edge-Top** ★ | 0.9630 | 0.9722 | 0.9722 | **0.9954** |
| Full | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| Normal | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| Thick-Edge | 1.0000 | 1.0000 | 1.0000 | 1.0000 |

★ = V3 의 best 이지만 weak point. Edge-Bottom + Edge-Top 의 chip 6 개 안에서 obj 식별
이 V3 의 진짜 ceiling.

### 2.3 Why Edge-Bottom / Edge-Top weak

`_sample_gen.py` 의 `DEFECT_BUDGET`: Edge-Top, Edge-Bottom 등 spatial 한정 class →
defect chip 6 개만 (전체 1024 중). 6 chip 안에서 5 obj 종류 구분 → **통계량 너무 작음**.
다른 distribution (Center / Full / Donut) 은 defect chip 더 많아 obj 분포 명확.

### 2.4 V3 의 약점이 안 풀리는 이유

1. Mid-fusion 으로도 못 잡음 — R 도 같은 6 chip RGB 보고 같은 confusion
2. Cross-attention 도 같음 — 두 stream 의 정보가 본질적으로 같음
3. MoE 도 같음 — Edge-Bottom expert 도 6 chip 만 본다
4. Knowledge Distillation 도 teacher 의 한계 그대로

→ Tier 3-6 ensemble 모두 expected gain 거의 0. V3 가 oracle ceiling 0.9919 도 추월 (V3
0.9946 vs oracle 0.9919) — V3 ensemble plan deprecated (출처:
`memory/project_v3_chipgrid_best.md`).

### 2.5 합성 spec 변경 후보 (priority 낮음)

1. `_sample_gen.py:739` `DEFECT_BUDGET` ↑ (Edge-Bottom 6 → 12 chip) — but spec 변경 큼,
   WM-811K 분포 학습 결과와 충돌
2. obj-only 학습 시 inter-class margin loss 추가 (사용자 진행 중,
   `cnn_train_chipgrid_fusion.py` + `_chipgrid_kde_gmm.py` Phase 3)
3. 데이터 증가 (220 → 500/class) — diminishing returns 예상

## 3. GPU resource lesson (round 28)

### 3.1 obj_id_maps build GPU 사용 측정

round 28 에서 `chip_tools/_build_obj_id_maps.py --batch 64` dispatch 결과:
- 측정 GPU 사용량: **7-8 GB** (RTX 4060 Ti 16GB 의 ~50%)
- chip CNN forward (88M ConvNeXtV2-base, batch 64 inline) + numpy I/O dominant

### 3.2 다른 작업과 동시 실행 시 throttle

obj_id_maps build 와 chip_multilabel 학습이 동시에 GPU 점유 → contention. 사용자
정책 (round 28 내부 결정):

> "다음 build 는 `--batch 32` 로 GPU 점유 절반으로 throttle 해서 다른 학습과 공유 가능
> 하게."

→ batch 32 시 예상 GPU 사용 ~3-4 GB, build 시간 ~2× 증가하지만 GPU sharing 유연.

### 3.3 학습 dispatch policy (`cnn-master`)

자원 가드 team (출처: `CLAUDE.md` line 148-156):
- RAM 80% 한계 자동 polling
- GPU 90% 한계 자동 polling
- 한계 초과 시 process kill + `log/<run>` `_PAUSED` rename (삭제 X)
- 자원 회복 polling + 재시작

## 4. Edge-Bottom_bank_boundary / scratch — V3 baseline weak class

(출처: `docs/chipgrid/RESULTS.md` §V3 per-class)

V3 baseline (n=220, seed 42) val split 의 per-class F1 < 1.0:

| class | F1 | Sup | 메모 |
|---|---|---|---|
| Edge-Bottom_bank_boundary | 0.800 | 11 | weak (특히) |
| Edge-Bottom_scratch | 0.778 | 9 | weak (특히) |
| Edge-Top_scratch | 0.800 | 6 | weak (작은 sup) |
| Edge-Bottom_invalid_main | 0.919 | 17 | |
| Edge-Top_invalid_main | 0.966 | 15 | |

→ active_classes_20.yaml 에 Edge-Bottom × 5, Edge-Top × 5 모두 보존된 이유. round 28
compound vs wafer-only 비교 시 이 weak class 의 마진을 우선 모니터링.

## Cross-link

- 13 feedback memory raw → `~/.claude/projects/D--project-known-cnn/memory/feedback_*.md`
- intra-distribution data → `docs/wafer-ensemble/INTRA_DIST.md`
- V3 7 핵심 발견 → `docs/wafer-ensemble/DISCOVERY.md`
- active class 결정 → `docs/wafer-ensemble/ACTIVE_CLASSES.md`
- multi-label 3 핵심 영역 → `docs/multi-label/{LOSS,MATCHING,DECISION}_DESIGN.md`
- block_expand 정책 → `feedback_block_expand_only.md`, `_chipgrid_resize.py`
