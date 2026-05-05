# Multi-label Ablation Configs

이 디렉토리는 본 ablation 의 mix 조합 spec 을 YAML 로 명시한다. 각 script 가 직접
sweep range 를 hardcode 안 하고 이 config 들을 참조하면 (확장성) hyperparameter 변경
시 코드 수정 없이 가능.

## Files

| 파일 | 역할 | 사용 script |
|---|---|---|
| `chipgrid_class20_hard.yaml` | 현재 data root 에서 바로 실행 가능한 hard/special 20-class active list | `cnn_eval_chipgrid.py`, `_chipgrid_kde_gmm.py`, `cnn_train_chipgrid_fusion.py` |
| `chipgrid_class30_target.yaml` | 8개 object-less wafer-canvas class 생성 후 사용할 30-class target list | same |
| `loss_M1.yaml` ~ `loss_M7.yaml` | Stage 4 loss mix 조합 (LOSS_DESIGN.md) | `cnn_train_multilabel.py` |
| `decision_D1.yaml` ~ `decision_D8.yaml` | Stage 5 threshold strategy (DECISION_RULE.md) | `_threshold_sweep.py` |
| `matching_C1.yaml` ~ `matching_C7.yaml` | Stage 6 chip-wafer matching mix (MATCHING_DESIGN.md) | `_eval_chip_matching.py` |

## Chipgrid active-class configs

```powershell
# 현재 데이터로 hard 20-class V3 평가/학습
python cnn_eval_chipgrid.py --variant V3 --no-r-channel --active-classes-yaml configs/chipgrid_class20_hard.yaml

# Stage A KDE/GMM 또는 Stage B fusion 도 같은 class list 를 사용
python _chipgrid_kde_gmm.py --active-classes-yaml configs/chipgrid_class20_hard.yaml
python cnn_train_chipgrid_fusion.py --active-classes-yaml configs/chipgrid_class20_hard.yaml
```

`chipgrid_class30_target.yaml` 은 `DiagonalSmear`, `CrossScratch`, `CrescentArc`, `SpiralTrail`, `ParallelScratches`, `EdgeSmudge`, `BlobChain`, `BrokenRing` 생성 전에는 기본 strict check 로 실패해야 정상이다.

## Quick reference

각 config 의 핵심 mix 조합:

### Loss (M1-M7)

| ID | Loss | CW | LS | Other |
|---|---|---|---|---|
| M1 | bce | none | 0 | baseline |
| M2 | asl | effective | 0.05 | γ_pos=1, γ_neg=4, clip=0.05 (Ridnik default) |
| M3 | asl | effective(0.9999) | 0.05 | γ_pos=1, γ_neg=4 (★ stronger CW) |
| M4 | adagc_asl | effective | 0.05 | λ_gc=0.5, ASL aux 0.5 (★ hybrid) |
| M5 | bce_warmup_asl | effective | 0.05 | warmup=5 epoch BCE → ASL |
| M6 | focal_asl | effective | 0.05 | γ_pos=2 (positive boost) |
| M7 | adagc_ls | effective(0.9999) | 0.1 | λ_gc=0.5 (★ calibration focus) |

### Decision (D1-D8)

| ID | Base | Calibration | Top-K Floor |
|---|---|---|---|
| D1 | default 0.5 | none | off |
| D2 | per-class F1 | none | off |
| D3 | per-class F1 | Temperature | off |
| D4 | per-class F1 | Platt | off |
| D5 | per-class F1 | Temp + Platt mix | off |
| D6 | IDF | none | off |
| D7 | IDF | Temperature | K=3, floor=0.3 |
| D8 | KNN_local | Temp + Platt mix | K=3, floor=0.3 (★ best) |

### Matching (C1-C7)

| ID | Surface | CRF | Consistency | Outlier |
|---|---|---|---|---|
| C1 | E1 single heatmap_smooth | off | off | 0.001 |
| C2 | E2 hybrid | off | strict | 0.001 |
| C3 | E3 3-method weighted (0.4 hm_smooth + 0.4 GMM + 0.2 KDE) | off | strict | 0.001 |
| C4 | E3 | constant | strict | 0.001 |
| C5 | E3 | constant | strict | percentile 5% |
| C6 | E4 geometric (3-method) | off | strict | percentile 5% |
| C7 | E4 geo (4-method incl. hybrid) | constant | strict | percentile 5% (★ best) |
