# 01 — Motivation

> **APPEND-ONLY.** 이 파일은 누적 기록. 한 번 적힌 row/section 절대 삭제·수정 금지.

## 1. 본 프로젝트의 본질 (사용자 명시)

> **compound (R+G+B = failbit + obj_id + zero) 가 wafer-only (R 만) 보다 wafer 분류
> test_f1 가 높도록** — chip-level obj 정보 (chip CNN inference 결과) 가 wafer-level
> 분류에 보탬을 주는지 검증.
>
> 다양한 확률 분포 / loss 설계 / matching / threshold 등은 모두 이 목적을 위한 수단.
> (출처: `docs/paper/README.md` line 8-10)

이 한 문장이 모든 design decision 의 우선순위 기준. 작은 마진이라도 compound 가
wafer-only 를 넘어서야 "chip-level obj 신호가 wafer 학습에 도움 된다" 가 empirical
하게 입증된다. 못 넘으면 obj signal pipeline (chip CNN → obj_id_maps → 3ch concat) 에
근본적 문제가 있다는 뜻.

## 2. Why chip-level obj_id 신호가 wafer-level 분류에 도움 되어야 하는가

WM-811K 33-class wafer set 의 class 정의 자체가 (distribution × chip-object) 의
cross-product:

```
Donut_scratch         = wafer-level Donut shape × chip-level scratch object
Edge-Bottom_fork      = wafer-level Edge-Bottom × chip-level fork object
Edge-Ring_invalid_main = wafer-level Edge-Ring × all-invalid chips
```

R 채널 (palette grade) 만 보는 wafer-only 모델은 두 차원 모두 **하나의 RGB 신호** 에서
풀어내야 한다. 반면 compound 는 chip-level CNN 이 미리 chip object 를 분류한 결과를
별도 채널 (G) 로 받으니 **wafer 분류기는 distribution 에만 집중** 가능 — 이론적으로
compound 가 우월해야 한다.

V3 chipgrid 결과 (출처: `docs/wafer-ensemble/RESULTS.md`):
- V0 (R-only chipgrid) val_f1 0.4359
- V1 (argmax obj_id 정수 1ch) val_f1 0.9505 → **+51 %p** (obj_id 채널 신호가 dominant)
- V3 (one-hot 5ch obj) val_f1 0.9689 (n=100/cls), 0.9946 (n=220/cls)

이 차이가 R 단일 채널이 obj 차원을 풀기 어렵다는 강한 증거다. 즉 **chip CNN distill
된 obj label 채널을 추가하면 wafer 분류 천장이 올라갈 것** 이라는 가설.

## 3. Why fair-eval protocol 필수

다른 model 비교 시 split / sample / epoch / hparam 모두 다르면 대조 의미가 없다 (출처:
`feedback_fair_eval_protocol.md`):

> "평가는 항상 같은 wafer수 chip수 학습과 predict로 해야하는거 알지? epoch도 똑같이
> 돌리고 best model 비교"

본 프로젝트의 "compound vs wafer-only" 비교는 다음 단일 protocol 위에서만 의미를 가진다
(출처: `experiments/fair_eval_protocol.yaml`, `docs/wafer-ensemble/FAIR_EVAL_PROTOCOL.md`):

| 항목 | 값 |
|---|---|
| active class | `experiments/active_classes_30.yaml` (= `configs/chipgrid_class30_target.yaml`, 합성 후) |
| per-class sample | 200 (모든 class 통일) |
| split | 0.8 / 0.1 / 0.1 stratified, seed 42 |
| epoch | 30 (early stop 끔), best val_f1 epoch model |
| optimizer / scheduler | AdamW wd 0.05, LinearLR warmup → CosineAnnealing (★ AD 참조 적용) |
| augmentation | rotate ±15°, translate/scale ±3°, gaussian σ=0.01 (no flip / colorjitter / mixup / cutmix) |
| TTA | 절대 금지 (출처: `feedback_no_tta_wafer.md`) |

**유일하게 다른 점 = G 채널 (obj_id) 의 유무**. 그래야 만약 compound test_f1 >
wafer-only test_f1 라면 그 마진이 obj 신호의 contribution 이라고 결론낼 수 있다.

## 4. wafer class identity 가 angle/위치에 묶인 결과 — TTA 금지의 직접 결과

wafer class 정의가 spatial / angular identity 에 직접 묶여있다:
- `scratch_rot` (이전 `scratch_21deg`) — 21° 회전된 라인 자체가 class
- `Edge-Top_*` vs `Edge-Bottom_*` — 위/아래 위치 자체가 class
- `Edge-Ring` — 가장자리 ring 위치

→ rotation 90° / VFlip / 180° TTA 적용 시 class 정체성 변형 (Edge-Top → Edge-Bottom).
**다른 class 의 답을 평균** 내는 잘못된 ensemble 이 되므로 TTA 절대 금지 (출처:
`feedback_no_tta_wafer.md`). 학습 augmentation 도 동일 — ±15° rotation, translate/scale
±3%, gaussian σ=0.01 만 허용.

## 5. 작은 마진이어도 의미가 있는 이유

만약 compound 가 wafer-only 보다 +0.5 ~ +2 %p 만 넘는 작은 마진이어도:
1. obj_id pipeline 의 정보가 R 신호에 redundant 하지 않다 = 학계 multi-stream fusion
   benchmark 에 가까운 결과
2. chip CNN noise robustness (출처: `docs/wafer-ensemble/RESULTS.md` 0~10% noise 까지
   robust) 와 결합하면 production deploy 가능
3. 향후 multi-label 추론 path (출처: `docs/multi-label/STAGES.md`) 의 chip-wafer matching
   stage 6 의 base 가 됨

만약 compound 가 wafer-only 를 못 넘으면:
1. block_expand 정책 (출처: `feedback_block_expand_only.md`) 이 BICUBIC 손상은 잘 막았지만
   chip CNN inference distillation 자체에 손실
2. V3 chipgrid (val_f1 0.9946, 1.16M params, 32×32 native + one-hot 5ch) 가 더 효율 —
   BICUBIC 정수 보간이 아닌 native 32 grid 위 학습으로 가야 함
3. orch-master loop 가 round 별 spec 변경으로 마진 회복 시도

## Cross-link

- 본질 statement → `docs/paper/README.md`
- fair-eval protocol → `docs/wafer-ensemble/FAIR_EVAL_PROTOCOL.md`
- TTA 금지 → `~/.claude/projects/D--project-known-cnn/memory/feedback_no_tta_wafer.md`
- V3 chipgrid 결과 → `docs/wafer-ensemble/RESULTS.md`
- compound > wafer-only loop → `.claude/agents/orch-master.md`
