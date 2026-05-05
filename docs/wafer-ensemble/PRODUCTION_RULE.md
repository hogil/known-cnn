# Production Rule — Logit Ensemble + Per-Class Threshold + Review Queue

이 문서는 wafer 33-class 분류기의 **production 운영 룰** 을 정의한다.
Tier 1 (logit ensemble V1C) + Tier 2 (per-class threshold + Conformal) 조합이
**deploy-ready** 이고, 사용자 결정으로 default 운영 룰로 채택.

## 결정 요약

```
1. R-only logits z_R + obj-only logits z_obj 동시 산출
2. Temperature scaling: z_R / T_R, z_obj / T_obj  (T fit on val 1회)
3. Logit average: z_final = α × z_R + (1-α) × z_obj  (α=0.10)
4. softmax → max_prob, argmax
5. max_prob ≥ τ[c] (per-class threshold) → auto, else → review queue
6. 매주 review queue ≤ 8% + auto 의 정확도 99.7%+ verification
```

## Recommended config (concrete)

```yaml
# wafer_ensemble_production.yaml
version: 2026-05-03

models:
  r_only:
    path: logs_wafer/overall/best_model.pth
    backbone: convnextv2_base_fcmae
    img_size: 1024
    in_chans: 3
    n_classes: 33
  obj_only:
    path: logs_objid_ablation/overall/best_model.pth
    backbone: ObjOnlyCNN_4layer
    img_size: 32
    in_chans: 1  # uint8 obj_id (Embedding inside model)
    n_classes: 33

ensemble:
  method: logit_avg_with_temperature  # V1C
  alpha: 0.10                          # weight on R-only (obj-only = 0.90)
  temperature_r: 1.42                  # fit on val (Guo et al. ICML 2017)
  temperature_obj: 1.18

calibration:
  method: temperature                  # only T (no Platt/Isotonic)
  recalibrate_when: monthly OR new_chip_CNN OR new_data_distribution
  recalibration_set: val (n=1416, seed=42 split)

decision:
  rule: per_class_threshold + global_max_prob
  per_class_threshold_path: configs/per_class_threshold.json
  fallback_max_prob: 0.77              # if class missing from per-class config
  review_queue:
    expected_rate: 0.05                # 5% (Conformal 95% coverage)
    upper_bound: 0.08                  # alarm if >8% (model degradation 신호)

logging:
  log_logits: true                     # per wafer, both streams (audit trail)
  log_max_prob: true
  log_per_class_score: true
  parquet_output: result_wafer/<product>/<line>/<date>/preds.parquet
  parquet_columns:
    - wafer_id
    - pred_class
    - max_prob
    - z_R_top1, z_R_top2
    - z_obj_top1, z_obj_top2
    - flag (auto / review)
    - τ_per_class (해당 class 의 threshold 값)

residual_review:
  forced_routing: edge_bottom_subgroup
  classes:
    - Edge-Bottom_particle_blast
    - Edge-Bottom_scratch
    - Edge-Bottom_scratch_21deg
    - Edge-Top_invalid_main
    - Edge-Top_particle_blast
  reason: "D5 — DEFECT_BUDGET 6 chip → both stream 합성 모호"
  policy: "max_prob ≥ τ 라도 manual review 강제 (D5 known-limit)"
```

## Per-class threshold table (val 측정 기반)

```json
{
  "Center": 0.85,
  "Donut": 0.80,
  "Edge-Bottom": 0.82,
  "Edge-Bottom_particle_blast": 1.00,
  "Edge-Bottom_scratch": 1.00,
  "Edge-Bottom_scratch_21deg": 1.00,
  "Edge-Loc": 0.78,
  "Edge-Ring": 0.75,
  "Edge-Top": 0.82,
  "Edge-Top_invalid_main": 1.00,
  "Edge-Top_particle_blast": 1.00,
  "Loc": 0.80,
  "Random": 0.70,
  "Scratch": 0.85,
  "Near-full": 0.75,
  "scratch_21deg": 0.85,
  "particle_blast": 0.80,
  "invalid_main": 0.80,
  "bank_boundary": 0.80,
  "Starburst": 0.85,
  "CommaCluster": 0.85,
  "default": 0.77
}
```

`τ = 1.00` 항목은 **threshold 도달 불가능** = 항상 review queue (D5 forced routing).

생성 명령:
```bash
python _selective_prediction.py \
    --val-logits-r logs_wafer/overall/val_logits.npy \
    --val-logits-obj logs_objid_ablation/overall/val_logits.npy \
    --val-y logs_wafer/overall/val_y.npy \
    --alpha 0.10 \
    --temperature-r 1.42 \
    --temperature-obj 1.18 \
    --output configs/per_class_threshold.json
```

## Inference reference implementation

```python
import torch
import torch.nn.functional as F
import json
import numpy as np

# Load
cfg = yaml.safe_load(open("configs/wafer_ensemble_production.yaml"))
r_model = load_model(cfg["models"]["r_only"]["path"])
obj_model = load_model(cfg["models"]["obj_only"]["path"])
T_R = cfg["ensemble"]["temperature_r"]
T_obj = cfg["ensemble"]["temperature_obj"]
alpha = cfg["ensemble"]["alpha"]
per_class_th = json.load(open(cfg["decision"]["per_class_threshold_path"]))
fallback_th = cfg["decision"]["fallback_max_prob"]
forced_review = set(cfg["residual_review"]["classes"])

@torch.no_grad()
def predict_wafer(x_rgb, x_obj_id, classes):
    z_R = r_model(x_rgb)            # [1, 33] logits
    z_obj = obj_model(x_obj_id)     # [1, 33] logits

    # Temperature normalize then logit-average (V1C)
    z_final = alpha * (z_R / T_R) + (1 - alpha) * (z_obj / T_obj)
    p = F.softmax(z_final, -1)
    max_prob, pred_idx = p.max(-1)
    pred_class = classes[pred_idx.item()]

    # Decision
    th = per_class_th.get(pred_class, fallback_th)
    auto = (max_prob.item() >= th) and (pred_class not in forced_review)

    return {
        "pred_class": pred_class,
        "max_prob": float(max_prob),
        "flag": "auto" if auto else "review",
        "z_R_top1": float(z_R.max()),
        "z_obj_top1": float(z_obj.max()),
        "threshold_used": th,
    }
```

## Verification — weekly batch check

```bash
# 1. Collect last 7 days result_wafer/*/*/2026MMDD/preds.parquet
python _verify_production.py \
    --result-root result_wafer/ \
    --days 7 \
    --output reports/weekly_<YYYYMMDD>.md

# 2. 자동 alarm 조건
#    - review_rate > 8%       → model degradation
#    - auto_accuracy < 99.7%   → recalibration 필요
#    - 특정 class skew > 2× baseline → distribution shift
```

`_verify_production.py` 산출:
- review_rate (전체 + per-class)
- auto_accuracy (sample audit n=100)
- threshold violation count
- recalibration recommendation

## Recalibration trigger

월 1회 또는:
- chip CNN 재학습 (`logs_chip/overall/best_model.pth` 갱신) 시 obj_id 분포 변경
  → T_obj 재fit 필수
- R-only 재학습 시 T_R 재fit
- 새 wafer class 추가 시 per-class threshold 재산출
- review queue 8% 초과 1주 연속 시 alarm + 사용자 확인

## 사용자 결정 명세

| 항목 | 결정 | 이유 |
|---|---|---|
| ensemble 단위 | logit-level (V1C) | softmax-level 보다 +0.15pp + multi-label 호환 |
| α (R-only weight) | 0.10 (obj-only 위주) | per-class α (V1A) 는 overfit 위험, 단일 α 유지 |
| Temperature | per-stream T_R, T_obj | logit scale 차이 흡수 |
| Calibration method | Temperature only | Platt/Isotonic 은 in-sample overfit (oracle 0.9923 추월) |
| Selective rule | per-class threshold + global fallback | val F1-maximizing per-class |
| Review queue cap | 8% | Conformal 95% coverage + 3% margin |
| D5 forced routing | Edge-Bottom subgroup → review | 합성 한계, 어떤 fusion 으로도 구분 불가 |
| Knowledge Distillation | NOT YET | 단일 모델 deploy 우선 ★ Tier 6 미래 |
| TTA | 절대 금지 | `feedback_no_tta_wafer.md` |

## 관련 파일

- `cnn_predict_wafer.py` — R-only inference engine (V1C 통합 시 `cnn_predict_ensemble.py` 신규)
- `_selective_prediction.py` — per-class threshold 학습 산출
- `_calibration_ensemble.py` — Temperature fitting
- `_tier1_variants.py` — V1A/V1B/V1C 비교
- 본 문서 → `ENSEMBLE_TIERS.md` (Tier 1 + 2 detail), `STATUS.md`, `DISCOVERY.md`
- agent → `.claude/agents/wafer-ensemble.md`
- skill → `.claude/skills/wafer-ensemble/SKILL.md`
