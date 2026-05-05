---
name: chipgrid-eval
description: chip-grid 32×32 native resolution wafer 분류 평가용 단일 스크립트 (cnn_eval_chipgrid.py) wrapper. obj_id encoding 변종 sweep + chip CNN noise robustness 테스트. 작은 데이터 (n=30~220/class) 빠른 ablation.
---

# chipgrid-eval skill

`cnn_eval_chipgrid.py` 단일 스크립트로 obj_id encoding 변종 (V0~V6) + chip CNN noise robustness 를 빠르게 비교한다. 기존 `cnn_train_*.py` 와 분리. 평가용 ablation, production 설계 검증용.

## 핵심 발상

- wafer-level 분류 의 자연 해상도 = chip 격자 32×32
  - R 채널: wafer PNG 6400 → AvgPool 32 (chip 별 fail-bit grade 평균)
  - obj_id: chip CNN 결과 그대로 32×32 (보간 0)
- 입력이 native 해상도라 BICUBIC 보간 손상 자체가 없음
- 작은 CNN (~1.2M params) 만으로 충분. ConvNeXtV2-base 88M 불필요.

## 입력 → 산출

| 입력 | 산출 |
|---|---|
| `--variant V0..V6`, `--n-per-class 30~220`, `--epochs 20~30`, `--seed 42` | `logs_chipgrid/<tag>_<TS>_<test_f1>_<val_f1>/` 폴더 (best_model.pth, best_history.txt, history.json, hparams.json, run.log) |

## 변종 (obj_id encoding) catalog

| Variant | Encoding | in_ch (R 포함) | 특성 |
|---|---|---|---|
| V0 | obj_id 없음 | 1 | R-only baseline. 25 sub-class 가 같아 보임 → ~44% F1 천장 |
| V1 | argmax 정수 / divisor | 2 | 1ch 정수, normalize. `--obj-norm` 으로 divisor 조절 (default 5). |
| V2 | binary single class | 2 | `--target-id 1..5` 로 1 class 만 표시. 그 class 만 풀고 나머진 V0 수준. |
| V3 ★ | one-hot 5채널 binary | 6 | 정보 손실 0. 가장 강력. n=100 에서 val 97% / test 99%. |
| V4 | chip CNN softmax 5채널 | 6 | 구현됨. `_build_obj_id_maps.py --save-prob-maps` 로 `obj_prob_maps` 생성 후 `--obj-prob-dir` 지정. |
| V5 | softmax max 1채널 | 2 | NotImplemented. |
| V6 | softmax entropy 1채널 | 2 | NotImplemented. |

## chip CNN noise robustness

`--chip-noise <p>` (default 0) — 각 chip 의 obj_id 를 확률 p 로 랜덤 1~5 swap 시뮬레이션. 실전 chip CNN 오류 대비.
`--chip-noise-eval` — val/test 에도 동일 noise 적용 (production-time 시나리오).

검증 결과 (V3, n=100):
- noise 0%: val 96.89% / test 98.79%
- noise 5%: val 96.67% / test 99.10%
- noise 10%: val 97.07% / test 99.19% (베이스라인 안 차이는 단일 seed 분산 안 = 운)
- noise 20%: val 95.95% / test 96.36% (명확 degrade)

→ V3 = chip CNN 10% 오류까지 robust. 20%+ 망가짐.

## 실행 패턴

### Pattern #1 — 변종 baseline sweep (V0~V3 순차)

```bash
python cnn_eval_chipgrid.py --variant V0 --n-per-class 100 --epochs 30 --model-tag v0_n100
python cnn_eval_chipgrid.py --variant V1 --n-per-class 100 --epochs 30 --model-tag v1_n100
python cnn_eval_chipgrid.py --variant V2 --n-per-class 100 --target-id 3 --epochs 30 --model-tag v2_particle_n100
python cnn_eval_chipgrid.py --variant V3 --n-per-class 100 --epochs 30 --model-tag v3_onehot_n100
```

각 학습 ~1분 미만 (cache load 포함 ~5분).

### Pattern #2 — robustness curve (V3 + chip-noise)

```bash
for p in 0 0.05 0.10 0.20; do
  python cnn_eval_chipgrid.py --variant V3 --chip-noise $p --chip-noise-eval --model-tag v3_noise_${p}
done
```

### Pattern #3 — full data scaling (V3 천장)

```bash
python cnn_eval_chipgrid.py --variant V3 --n-per-class 220 --epochs 30 --model-tag v3_full
```

### Pattern #4 — quantization variants (V1)

```bash
for d in 1 5 10 31; do
  python cnn_eval_chipgrid.py --variant V1 --obj-norm $d --model-tag v1_norm$d
done
```

### Pattern #5 — V4 soft object maps + factorized/hard loss

```powershell
python _build_obj_id_maps.py `
  --chip-model logs_chip/overall/best_model.pth `
  --save-prob-maps `
  --device cuda

python cnn_eval_chipgrid.py --variant V4 --no-r-channel --n-per-class 220 --model-tag v4_soft_objonly

python cnn_eval_chipgrid.py --variant V4 `
  --aux-heads factorized `
  --dist-loss-weight 0.20 `
  --obj-loss-weight 0.30 `
  --hard-contrastive-weight 0.05 `
  --hard-contrastive-scope edge `
  --n-per-class 220 `
  --model-tag v4_factorized_edge_supcon
```

Guarded queue:

```powershell
.\experiments\run_chipgrid_object_plan.ps1 -BuildProbMaps
```

The queue logs GPU/CPU memory before, during, and after each Python process and kills only the process tree it started when configured limits are exceeded.

### Pattern #6 — hard/special active classes and class-30 target

```powershell
# 현재 data root 에서 바로 가능한 20-class pruning
python cnn_eval_chipgrid.py `
  --variant V3 `
  --no-r-channel `
  --active-classes-yaml configs/chipgrid_class20_hard.yaml

python _chipgrid_kde_gmm.py `
  --active-classes-yaml configs/chipgrid_class20_hard.yaml

python cnn_train_chipgrid_fusion.py `
  --image-branch r-only `
  --active-classes-yaml configs/chipgrid_class20_hard.yaml

python _chipgrid_gmm_options.py `
  --active-classes-yaml configs/chipgrid_class20_hard.yaml `
  --n-components 4 `
  --save-features
```

Fusion image branch choices: `r-only`, `v3`, `r-plus-v3`.

`configs/chipgrid_class30_target.yaml` 은 새 object-less wafer-canvas class 8개를 생성한 뒤 사용한다. active class 는 기본 strict mode 이므로 누락 class 가 있으면 실패해야 정상이다.

Object-less class generation is implemented in `_sample_gen_gpu.py`:

```powershell
# GPU idle 확인 후 smoke
python _sample_gen_gpu.py --only-class DiagonalSmear --n 5 --seed-offset 9000000
```

The 10 wafer-canvas classes are `Starburst`, `CommaCluster`, `DiagonalSmear`, `CrossScratch`, `CrescentArc`, `SpiralTrail`, `ParallelScratches`, `EdgeSmudge`, `BlobChain`, `BrokenRing`.

## 출력 위치

- **`logs_chipgrid/<tag>_<TS>_<test_f1>_<val_f1>/`** — 정상 종료 시
- **`logs_chipgrid/<tag>_<TS>_running/`** — 학습 중
- **`logs_chipgrid/<tag>_<TS>_ABORTED/`** — silent crash 등

각 폴더 내용:
- `best_model.pth` — model_state + classes + in_ch + 메트릭 + epoch + args
- `best_history.txt` — BEST OVERALL + per-class F1 (val/test) + BEST UPDATES SUMMARY
- `history.json` — 매 epoch tr_loss / tr_acc / val_acc / val_f1
- `hparams.json` — 모든 args + 환경 메타
- `run.log` — append-only 시간순 로그

## 주의사항

- **wafer PNG 캐시 시간** — 첫 학습 시 ~5분 (PNG 6400×6400 파일 1010장 → 32×32 AvgPool). 이후 변종마다 다시 캐시 (스크립트 종료시 메모리 비움).
- **GPU 점유** — 작은 모델이라 ~1-2 GB. wafer/compound 학습과 동시 GPU 사용 가능 (총 GPU mem 한도 내).
- **결과 변동성** — n=100 / val_sup=332 라 ±2%p 분산 정상. 단일 seed 결과 차이 0.5%p 이하 → noise 로 간주, 통계 검증 위해선 5 seed 평균.
- **train acc 매우 빠름 / val 진동** — 1.2M params + 800 train sample 이면 epoch 1-2 만에 train 90% 도달, val 은 epoch 4-10 사이 saturate. 이건 정상 (early stop patience 7 이 잡아줌).

## docs 위치

- 결과 표 + 비교: `docs/chipgrid/RESULTS.md`
- production 설계: `docs/chipgrid/PRODUCTION.md`
- class-30/GMM hybrid 설계: `docs/chipgrid/CLASS30_GMM_HYBRID_PLAN.md`
- V4/factorized/hard-loss queue: `docs/chipgrid/OBJECT_IMPROVEMENT_PLAN.md`
- 전체 인덱스: `docs/chipgrid/README.md`

## 절대 금기

- `cnn_train.py / cnn_train_compound.py / cnn_train_wafer.py` 등 기존 trainer **수정 금지** (이 스크립트는 standalone).
- `logs_chipgrid/` 외 다른 logs 디렉토리 건드리지 말 것.
- 학습 결과 폴더 / 파일 어떤 형태로도 삭제·rename·overwrite 금지 (CLAUDE.md absolute rule).
- 변종 추가 시 기존 V0~V6 인덱스 유지. 새 변종은 V7 부터.

## 양자화 변종 (사용자 명시)

V1 의 `--obj-norm` divisor 는 정수 양자화 표현 ablation:
- `5` (default): 0~5 정수 → [0, 0.2, 0.4, 0.6, 0.8, 1.0]
- `1`: raw 정수 (BatchNorm 가 정규화)
- `10`: [0, 0.1, ..., 0.5] 좁은 범위
어느 값이 잘 학습되는지 비교용.
