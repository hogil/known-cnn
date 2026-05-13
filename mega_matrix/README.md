# Mega Matrix Sweep — train × eval × selection

Self-contained default pipeline: data 생성 → 학습 → 평가 → report.md.
Pseudo-label retrain은 기본 OFF이며 `--with-pseudo`를 줄 때만 실행한다.

## 목적

| 축 | 값 (★ 모두 per-class 단위) |
|---|---|
| train_n / class | {50, 100, 200} |
| eval_n / class | {200, 2000, 20000} |
| best-model selection | {`val_f1`, `val_margin`} |
| pseudo-label retrain | 기본 OFF. `--with-pseudo` 때만 val 의 high-confidence pred 모아 train 에 추가 후 재학습 + 평가 |

기본은 총 18 primary cells (6 train × 3 eval) + report.

## 폴더

```
mega_matrix/
├── README.md            # ★ this — 사용법
├── run.sh               # 1-backbone all-in-one (single GPU)
├── run_ddp.sh           # 1-backbone parallel across N GPUs
├── run_all.sh           # ★ NEW — loops every backbone in weights/
├── download.py          # ★ creates/checks <backbone>.pth; download only with --allow-download
├── weights/             # pre-downloaded backbone <name>.pth (gitignored)
├── gen_data.py          # data 생성 (train + eval, per-class scale)
├── pseudo_label.py      # pseudo-label stage (per-backbone)
└── make_report.py       # summary.md + plots
```

## 산출

```
outputs/_mega_matrix/
├── logs/
│   ├── YYYYMMDD_HHMMSS_pidPID_<backbone>_run.log
│   ├── YYYYMMDD_HHMMSS_pidPID_<backbone>_run_ddp.log
│   ├── YYYYMMDD_HHMMSS_pidPID_<backbone>_run_ddp.log.train
│   ├── YYYYMMDD_HHMMSS_pidPID_<backbone>_run_ddp.log.gpuN
│   └── YYYYMMDD_HHMMSS_run_all_summary.log
├── train_n50/, train_n100/, train_n200/                # train subsets
├── eval_n200/, eval_n2000/, eval_n20000/               # eval subsets (per-class)
├── YYYYMMDD_HHMMSS_model_train{N}_{sel}/YYYYMMDD_HHMMSS_T7_*/  # 6 trained models
│   ├── best_model.pth
│   └── eval_{N}/stage1_*/                              # 18 eval results
├── pseudo_train/                                       # only with --with-pseudo
├── YYYYMMDD_HHMMSS_model_pseudo_{sel}/YYYYMMDD_HHMMSS_T7_*/     # only with --with-pseudo
│   └── eval_{N}/stage1_*/

docs/chip-multilabel/manager_report/
├── summary_mega_sweep.md                               # ★ final report
└── figs_mega/{bit_F1_heatmap, total_far_heatmap, scaling_curves,
              pseudo_label_gain}.png
```

## 사용법

### Single GPU (그대로 5 hr)
```bash
cd /path/to/known-cnn
bash mega_matrix/run.sh
```

### Multi-GPU server (DDP-style, 4 GPU → ~80 min)
```bash
bash mega_matrix/run_ddp.sh --gpus 4
```

### 특정 backbone (1개) — `--backbone <name>`
```bash
bash mega_matrix/run.sh     --backbone swinv2_base_window12to24_192to384.ms_in22k_ft_in1k
bash mega_matrix/run_ddp.sh --gpus 4 --backbone vit_base_patch16_384.augreg_in21k_ft_in1k
# img-size 는 name 의 "384"/"256"/"224" 패턴으로 자동 결정
# weights/<backbone>.pth 있으면 offline mode 자동
# 없으면 HF/timm 다운로드 없이 즉시 실패
# 결과는 outputs/_mega_matrix/<backbone>/ 아래
```

### ★ 모든 backbone 순차 평가 — `run_all.sh`
```bash
# weights/*.pth 모두 발견 → 순차로 run.sh / run_ddp.sh 호출
bash mega_matrix/run_all.sh                       # auto-detect GPU
bash mega_matrix/run_all.sh --gpus 4              # DDP
bash mega_matrix/run_all.sh --only convnextv2     # name substring filter
bash mega_matrix/run_all.sh --with-pseudo         # optional pseudo-label stage

# 데이터 생성은 첫 backbone 에서만 → 이후 backbone 은 --skip-data 자동
# 각 backbone 산출은 outputs/_mega_matrix/<backbone>/ 아래로 격리
# 종합 로그: outputs/_mega_matrix/logs/YYYYMMDD_HHMMSS_run_all_summary.log
```

### 부분 실행
```bash
bash mega_matrix/run.sh --skip-data        # data 이미 있으면
bash mega_matrix/run.sh --skip-train       # eval+report only
bash mega_matrix/run.sh --with-pseudo      # pseudo-label 단계까지 실행
bash mega_matrix/run.sh --data-base data/wm-811k
bash mega_matrix/run.sh --report-only      # report 만 재생성
```

기본 데이터 루트는 `<project>/data/wm-811k` 이다. 다른 위치를 쓰려면
`--data-base <path>` 또는 `WM811K_ROOT=<path>`를 사용한다. 프로젝트 루트
자체는 입력하지 않는다. 실행 스크립트 위치에서 자동 계산한다.

## Data 정책 (★ user 명확화)

### Train (per-class)
- `classification_chips/<defect>/` 에서 sort + first N 픽 (deterministic)
- {50, 100, 200} / class → {200, 400, 800} 총 chips
- master pool 부족 시 `_synth_chips_only.py --per-class 200` 자동 호출

### Eval (per-class)
- `gen_eval_set.py --per-defect N --per-normal N --include-triples` 자동 호출
- N ∈ {200, 2000, 20000}:
  - 4 single defect × N
  - 5-6 × 2-combo × N
  - 4 × 3-combo × N (참고용, paper metric 에서 제외)
  - Normal × N, Invalid × N/4
  - 4 OOD wafer pattern × N (CenterDonut, CrossScratch, DiagonalSmear, Starburst)
- 총 chip 수: ~(4+5+4+1+0.25+4) × N = ~18N
  - eval_n200: ~3.6K
  - eval_n2000: ~36K
  - eval_n20000: ~360K (★ generation ~1-2 hr)

## Selection criteria

### val_f1 (legacy)
- per-bit BCE F1 (threshold=0.5), macro 4 bits
- 작은 val (n=163) 에서 saturate → 3 reachable values
- pick: first epoch hitting max

### val_margin (★ NEW, paper §3 contribution)
- per chip: `mean(prob[positive bits]) - max(prob[negative bits])`
- aggregate avg over val chips
- decision boundary sharpness metric
- continuous spectrum, no saturation
- pick: epoch with max margin

### 왜 val_margin?
- multi-label friendly (boundary 직접 측정)
- 35-ckpt audit Spearman ρ = +0.56 (best) vs val_f1 ρ = −0.10
- iter116J empirical: val_margin pick (ep6) bit_F1 0.9943 vs val_f1 pick (ep1) 0.9422 = **+0.052** ★
- side effect: OOD-friendly (낮은 neg_prob)

## Optional Stage 4 — Pseudo-label retrain

### 동기
6 trained models 중 best (e.g., train200_margin_max) 로 **unlabeled chip pool 에 prediction 적용** → high-confidence prediction 만 모아 train data 에 추가 → semi-supervised 재학습.

### 절차 (pseudo_label.py)
1. **Best model 선택**: 18 cells 중 highest bit_F1 모델 (e.g., train200_margin_max)
2. **Pseudo-label source**: `chip_multilabel_v15direct_n1000/` 의 single-class chip (1000/class) — labels 알지만 pseudo-label 시뮬레이션 위해 prediction 사용
3. **Filter**: `max_prob > 0.85` AND `prediction = folder label` (정확도 보장) → 신뢰 chip 만
4. **Add to training**:
   - 기존 train (200/class) + pseudo-labeled (filtered, ~600-900/class) → ~800-1100/class
5. **Retrain**: 동일 recipe, val_margin selection
6. **Evaluate** on eval_n2000 → 비교 (pseudo-label gain Δ bit_F1, Δ FAR)

### 절대 룰 준수
- pseudo-label 한 chips 도 **single defect class only** (4 trained classes)
- Normal/Invalid/OOD/2-combo/3-combo 는 pseudo-label X
- → 절대 룰 위반 X

### 기대 효과
- pos_prob ↑ (training pool ↑ → bit confidence ↑)
- bit_F1 +0.005~0.020
- FAR 변동 (chip diversity ↑ → 더 robust 또는 confused — empirical)

## Recipe (모든 cells)

```yaml
backbone: convnextv2_base.fcmae_ft_in22k_in1k_384
img_size: 384
variant: T7 (BCE+LS)
ls: 0.30
epochs: 10
batch: 2
accum: 8           # effective 16
seed: 1
lr: 1.0e-4
optimizer: AdamW (wd=0.05)
scheduler: cosine
no_normal: true
val_criterion: f1 | margin_max
save_every_epoch: true
recipe: iter126e
cutmix_mode: complement
cutmix_pair: masked
cutmix_pair_fill: corner
cutmix_p: 0.25
cutmix_grid_dim: 16
cutmix_n_groups: 2
cutmix_complete_label_scale: 0.5
```

## DDP 자세한 사용 (server)

### Option A — Cell-level parallelism (★ run_ddp.sh 가 적용)
- 6 train cells 를 N GPUs 에 분산 (CUDA_VISIBLE_DEVICES 사용)
- 각 cell = single-GPU vanilla training
- 4-GPU server → train 30 min, eval 50 min, total 1.5 hr (vs single-GPU 5 hr)

```bash
bash mega_matrix/run_ddp.sh --gpus 4
```

### Option B — Per-cell DDP (advanced, 코드 수정 필요)
- 각 cell 을 N-GPU DDP 로 학습 (torchrun)
- trainer 가 `DistributedSampler` + `init_process_group` 추가 필요 (out of scope)
- batch 4 → 4×N effective 가능

## 보고서 (auto-generated summary.md)

`summary_mega_sweep.md` 에 포함:
1. **§1** selection criteria 설명 (val_f1 vs val_margin)
2. **§2** 18-cell main matrix
3. **§3** val_f1 vs val_margin direct Δ
4. **§4** per-OOD class FAR (Normal/Invalid/4 OOD)
5. **§5** per-group prob distribution
6. **§6** analysis
7. **§7** plots (heatmap × 2, scaling curves, pseudo-label gain)
8. **§8** ★ pseudo-label stage results
9. **§9** recipe spec

## 트러블슈팅

| 증상 | 해결 |
|---|---|
| `eval_n20000` 생성 1-2 hr | 인내 (~360K chips). `--skip-data` 로 부분 |
| GPU OOM (batch 2) | accum 16 으로 |
| disk full (각 model 350 MB × 18 evals) | `rm -f epoch_*.pth` 자동 적용 |
| `chip_multilabel.gen_eval_set` 실패 | classification_chips/ 가 가능 max 200/class 만 보유 — 더 큰 combo pool 필요 시 `_synth_multi_chips.py` 별도 호출 |

## 폐쇄망 (closed-network) 서버 사용

서버는 폐쇄망 기준이다. `run.sh`, `run_ddp.sh`, trainer, pseudo retrain은
HF/timm 자동 다운로드를 시도하지 않는다. `mega_matrix/weights/` 안에 로컬 weight
파일이 없으면 즉시 실패한다.

해결: 인터넷 머신에서 weight를 받아 `mega_matrix/weights/` 폴더 통째 서버에 복사.

### Step 1. 인터넷 머신에서 weight 다운로드

```bash
cd /path/to/known-cnn
python mega_matrix/download.py --allow-download                  # 모든 backbone (~2-3 GB)
python mega_matrix/download.py --allow-download --only convnext  # substring filter
python mega_matrix/download.py --list                            # 목록 + timm registry 검증
# 기본 `python mega_matrix/download.py` 는 verify/skip only; 네트워크 사용 안 함
# → mega_matrix/weights/<backbone>.pth
```

기본 MODELS (`download.py` 안에 정의, `timm.list_models(pretrained=True)`로 검증):
- convnextv2_base.fcmae_ft_in22k_in1k_384 (~360 MB, paper baseline winner)
- convnextv2_large.fcmae_ft_in22k_in1k_384 (~800 MB)
- swinv2_base_window12to24_192to384.ms_in22k_ft_in1k (~340 MB)
- vit_base_patch16_384.augreg_in21k_ft_in1k (~340 MB)
- deit3_base_patch16_384.fb_in22k_ft_in1k (~340 MB)
- convnextv2_tiny.fcmae_ft_in22k_in1k (~115 MB, 224 input)
- convnextv2_base.fcmae_ft_in22k_in1k (~355 MB, 224 input)
- tf_efficientnetv2_s.in21k_ft_in1k (~88 MB, 224 input)
- swin_tiny_patch4_window7_224.ms_in22k_ft_in1k (~113 MB, 224 input)
- maxvit_tiny_tf_224.in1k (~124 MB, 224 input)
- vit_base_patch16_clip_224.laion2b_ft_in12k_in1k (~344 MB, 224 input)

### Step 2. 서버로 폴더 복사

```bash
scp -r mega_matrix/weights/ user@server:/path/to/known-cnn/mega_matrix/
# 또는 git LFS / rsync / USB
```

### Step 3. 서버에서 평소대로 실행

```bash
bash mega_matrix/run_ddp.sh --gpus 4
# run.sh / run_ddp.sh 가 mega_matrix/weights/*.pth 존재 시
# --backbone-timm-weights <path> 자동 passthrough → HF download 안 함
# pseudo_label.py 도 동일 적용
```

### 다른 backbone 추가하려면

`mega_matrix/download.py` 의 `MODELS` 리스트에 timm model id 추가:

```python
MODELS = [
    "convnextv2_base.fcmae_ft_in22k_in1k_384",
    "swinv2_base_window12to24_192to384.ms_in22k_ft_in1k",
]
```

### 수동 override (다른 backbone 학습할 때)

```bash
python -m chip_multilabel._train_chip_variant \
    --backbone-timm convnextv2_base.fcmae_ft_in22k_in1k_384 \
    --backbone-timm-weights mega_matrix/weights/convnextv2_base.fcmae_ft_in22k_in1k_384.pth \
    (... 나머지 args)
```

## 절대 룰 (260512) 준수

- 학습: 4 single defect only (`--no-normal`)
- pseudo-label: single defect only (Normal/Invalid/OOD/combo pseudo 금지)
- bit F1: positive (single + 2-combo) macro
- Total FAR: (Normal + Invalid + OOD) FP rate
