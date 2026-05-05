# chipgrid 평가 — 인덱스

`cnn_eval_chipgrid.py` 단일 스크립트로 obj_id encoding 변종 + chip CNN noise robustness 비교. 32×32 chip-grid native resolution + 작은 CNN.

## 동기

기존 `cnn_train_compound.py` (3채널 R+G+B BICUBIC at 384) val_f1 97.84% 에서 막힘. weak class = Edge-Bottom/Top × chip object 4종. 원인 가설 = obj_id 정수 (0~5) 가 BICUBIC 보간되며 categorical 신호 깨짐 (1.7, 2.3 등).

→ **input 해상도를 chip 격자 32×32 로 낮추면 obj_id 보간 자체 불필요**. 같은 정보를 손상 없이 모델에 전달 + 데이터·hparam fair 비교.

## 핵심 결과 (2026-05-03)

| 모델 | data | model | val_f1 | test_f1 |
|---|---|---|---|---|
| compound 3ch BICUBIC 384 | 5,680 train | ConvNeXtV2-base 88M | 97.84% | 97.36% |
| **V3 chipgrid 32 one-hot 5ch** | **2,656 train** | **tiny CNN 1.16M** | **96.89%** | **98.79%** ★ |

→ **데이터 절반 + 모델 1/76 크기로 compound 동등 이상**. 핵심 = chip-grid native + one-hot binary 인코딩.

## 문서

| 문서 | 내용 |
|---|---|
| [RESULTS.md](RESULTS.md) | 모든 학습 run 표 (variant / hparam / val_f1 / test_f1) + 변종 비교 + per-class 분석 |
| [PRODUCTION.md](PRODUCTION.md) | production 배포 설계 — chip CNN noise robustness, unknown rejection, ensemble 전략 |
| [CLASS30_GMM_HYBRID_PLAN.md](CLASS30_GMM_HYBRID_PLAN.md) | class 30 재설계, object-less wafer-canvas 10 class, GMM/KDE hybrid feature plan |
| [OBJECT_IMPROVEMENT_PLAN.md](OBJECT_IMPROVEMENT_PLAN.md) | V4 soft obj maps, factorized heads, hard contrastive queue |
| `.claude/skills/chipgrid-eval/SKILL.md` | 스크립트 사용법 + 변종 catalog |
| `.claude/agents/chipgrid-eval.md` | agent 운영 spec (자동 dispatch + RESULTS.md 누적) |

## 실행

```bash
# 변종 sweep
python cnn_eval_chipgrid.py --variant V0 --n-per-class 100 --epochs 30 --model-tag v0_n100
python cnn_eval_chipgrid.py --variant V1 --n-per-class 100 --epochs 30 --model-tag v1_n100
python cnn_eval_chipgrid.py --variant V2 --target-id 3 --n-per-class 100 --epochs 30 --model-tag v2_particle_n100
python cnn_eval_chipgrid.py --variant V3 --n-per-class 100 --epochs 30 --model-tag v3_onehot_n100

# hard/special 20-class subset (데이터 삭제 없이 active list 만 제한)
python cnn_eval_chipgrid.py --variant V3 --no-r-channel --active-classes-yaml configs/chipgrid_class20_hard.yaml

# robustness
python cnn_eval_chipgrid.py --variant V3 --chip-noise 0.10 --chip-noise-eval --model-tag v3_noise10
```

각 학습 ~5-10 분 (PNG cache load + 30 epoch). 출력: `logs_chipgrid/<tag>_<TS>_<test_f1>_<val_f1>/`.

## 변종 사용법 요약

- **V0** (R only baseline) — 천장 ~44% (chip object 정보 없음)
- **V1** (argmax/divisor 1ch) — 정수 양자화. `--obj-norm` 으로 divisor 조절
- **V2** (binary single class 1ch) — `--target-id` 로 한 chip object 만 표시
- **V3 ★** (one-hot 5ch binary) — 정보 손실 0, 권장 default
- **V4** (chip CNN softmax 5ch) — `_build_obj_id_maps.py --save-prob-maps` 로 생성한 probability map 사용

## Class pruning / hybrid

- `configs/chipgrid_class20_hard.yaml`: 현재 데이터로 바로 실행 가능한 hard/special 20-class active list.
- `configs/chipgrid_class30_target.yaml`: 8개 object-less wafer-canvas class 생성 후 사용할 target list. 생성 전에는 strict check 로 실패해야 정상.
- `_chipgrid_kde_gmm.py` 와 `cnn_train_chipgrid_fusion.py` 도 `--active-classes-yaml` 을 지원한다.
- `_chipgrid_gmm_options.py` 는 alpha/beta/gamma/delta GMM feature option 을 비교하고, `--save-features` 로 fusion용 score matrix 를 저장한다.
