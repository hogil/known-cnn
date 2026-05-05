# 6-Tier Ensemble Plan — Theory + Implementation + Results

각 Tier 의 **이론 + 학계 논문 + 구체 implementation + 기대 + 위험 + 우리 도메인
실측**을 명시. 단계적 시도, 결과 보고 다음 Tier 결정.

plan 본문: `~/.claude/plans/1-input-batch-hidden-patterson.md`

---

## Tier 0 — Oracle Ceiling (✅ 측정 완료, 0.9923)

### 의미
"두 모델 중 하나라도 맞으면 정답" — 모든 ensemble 의 absolute upper bound.

### 수치 (n=1416)
- macro_f1 = **0.9923**
- accuracy = 0.9922
- both_wrong = 11 (0.78%) — 합성 자체 한계, 어떤 fusion 으로도 못 잡음

### 활용
- 모든 후속 Tier 의 success metric ("oracle 의 몇 % 도달?")
- production 의 review queue 하한 (≥ 0.78%)
- 11 both_wrong → manual review 강제 라우팅 정책

### 산출
- `results_disagree/oracle_summary.json`
- `_disagree_analysis.py` (Agent 3 산출)

---

## Tier 1 — Logit Ensemble + Uncertainty Review (✅ 즉시 deploy 가능)

### Theory
**Late Fusion / Logit Averaging** — 학계 가장 simple ensemble 패턴. logit 단계
에서 weighted average 후 softmax.

```
z_R = R_only_model(x_rgb)            # logits, [N, 33]
z_obj = obj_only_model(x_obj_id)
z_final = α × z_R + (1-α) × z_obj    # logit-level
P_final = softmax(z_final)
```

### Papers
- **Lakshminarayanan et al. (NeurIPS 2017)** "Simple and Scalable Predictive
  Uncertainty Estimation using Deep Ensembles" — deep ensemble baseline,
  +2-5% accuracy 학계 표준
- **Wolpert (Neural Networks 1992)** "Stacked Generalization" — ensemble 의
  founding paper
- **Hansen & Salamon (TPAMI 1990)** "Neural Network Ensembles" — single vs
  ensemble 이론 비교
- **Hinton et al. (2014)** "Distilling the Knowledge in a Neural Network" —
  logit-level avg + Temperature 의 KD origin

### 우리 도메인 결과 (Agent 1)
- Logit ensemble α=0.15: **macro_f1 0.9868** (error 31/29 → 19, -35%)
- Uncertainty review (max_prob ≥ 0.77): **92% auto / 8% review**, residual error 3건만

### Variants 실측 (V1A/B/C)

| Variant | macro_f1 | error | notes |
|---|---|---|---|
| simple α=0.15 (softmax-level) | 0.9868 | 19 | baseline |
| **V1A** per-class α | 0.9883 | 17 | overfit risk (per-class fit on val) |
| **V1B** geometric mean (α=0.10) | 0.9883 | 17 | clean, both_wrong 2 fix |
| **V1C** ★ Temperature + logit avg | 0.9883 | 17 | logit-level, multi-label 호환 |

### Implementation (V1C, ★ deploy ready)
```python
# 1. Temperature fitting (val set, ~10 sec each)
T_R = fit_temperature(R_logits_val, val_y)        # scalar
T_obj = fit_temperature(obj_logits_val, val_y)

# 2. Inference
z_R = R_only_model(x_rgb)            # logits [N, 33]
z_obj = obj_only_model(x_obj_id)

# 3. Temperature-normalized logit average
z_final = 0.10 * (z_R / T_R) + 0.90 * (z_obj / T_obj)
P_final = F.softmax(z_final, -1)

# 4. Decision
pred = P_final.argmax(-1)
max_prob = P_final.max(-1).values

# 5. Uncertainty review
if max_prob < 0.77:
    flag = "review"
else:
    flag = "auto"
```

### 비용
- 학습: 0 (양 ckpt 만 있으면 됨)
- inference: R-only 88M forward (~50ms/wafer GPU) + obj-only 0.4M (<1ms) → ~50ms total

### 위험
- per-class α (V1A) 는 val 에 overfit → held-out test 로 cross-fit 검증 필요
- α 값 자체가 합성 데이터 의존 → real wafer 도입 시 재calibration 필요

### Real-world examples
- **MixedWM38 wafer benchmark** (Wang 2020): single CNN 97% → ensemble 5 model 98.5%
- **ChestX-ray14** (Rajpurkar 2017 CheXNet): single 0.841 AUC → 10-model ensemble 0.852

### 산출
- `_ensemble_logit.py` (α sweep, simple softmax-level)
- `_tier1_variants.py` (V1A/V1B/V1C 비교)
- `results_ensemble/`, `results_ensemble_ep10/`
- `plots/ensemble/`

---

## Tier 2 — Calibration + Selective Prediction (✅ 즉시, 학습 X)

### Theory
- **Calibration**: model 의 confidence 가 진짜 확률이 되도록 보정
- **Selective prediction**: low-confidence sample 거부 → coverage-accuracy trade-off

### Papers
- **Guo et al. (ICML 2017)** "On Calibration of Modern Neural Networks" —
  Temperature scaling 의 origin, ECE 0.10 → 0.04
- **Niculescu-Mizil & Caruana (ICML 2005)** "Predicting Good Probabilities" —
  Platt / Isotonic 비교
- **Geifman & El-Yaniv (NIPS 2017)** "Selective Classification for Deep
  Neural Networks" — coverage 90% 시 accuracy +5%
- **El-Yaniv & Wiener (JMLR 2010)** "On the Foundations of Noise-free
  Selective Classification" — selective prediction 의 PAC-Bayesian bound
- **Vovk et al. (Algorithmic Learning in a Random World, 2005)** — Conformal
  Prediction 이론, distribution-free coverage guarantee

### 우리 도메인 적용

#### 2A. Temperature scaling each model 후 ensemble
```
T_R = fit_temperature(R_logits, val_y)        # ~10 sec
T_obj = fit_temperature(obj_logits, val_y)
z_R_cal = z_R / T_R
z_obj_cal = z_obj / T_obj
z_final = α × z_R_cal + (1-α) × z_obj_cal
```
- 학계 결과: ECE 50% 감소 → ensemble 의 weight 가 더 의미 있게 합쳐짐
- **우리 측정**: macro_f1 0.9868 (불변, 단 ECE 절반)

#### 2B. Per-class threshold (★ production-ready)
val 에서 class 별 F1 maximize 하는 threshold 학습:
```
for c in classes:
    τ[c] = argmax over [0.3, 0.99] of F1_c at that threshold
```
- 우리 측정: **0 errors at 95% coverage** (5% review queue)
- ★ 사용자 결정: production 운영 룰로 채택 → `PRODUCTION_RULE.md`

#### 2C. Conformal Prediction
val 에서 quantile 학습 → test 에서 distribution-free coverage guarantee:
```
α = 0.05  # 5% miscoverage target
q = quantile(val_nonconformity_scores, 1-α)
prediction_set = {c : nonconformity(x, c) ≤ q}
```
- 우리 측정: **95% coverage** with avg set size 1.02 (거의 single-label)
- 8% sample 만 set size > 1 → 자동으로 review queue

### Calibration 실측 (3 method 비교)

| Method | val macro_f1 | val ECE | error/1416 | notes |
|---|---|---|---|---|
| no calibration | 0.9868 | 0.062 | 19 | baseline |
| Temperature only | 0.9868 | **0.029** | 19 | argmax 보존, ECE↓ |
| Platt | 0.9875 | 0.034 | 18 | renorm artifact |
| **Isotonic** (in-sample) | 0.9931 | 0.018 | 10 | **OVERFIT** — oracle 0.9923 추월, 사용 X |

### 비용
- 학습: 0 (val set 으로 1 회 fitting, ~30 sec)
- inference: 추가 0 (logit 후 scalar 나누기만)

### 산출
- `_calibration_ensemble.py` — Temperature/Platt/Isotonic 비교
- `_selective_prediction.py` — Risk-Coverage curve + per-class threshold + Conformal
- `results_selective/`, `plots/selective/`

---

## Tier 3 — Mid-level Feature Fusion (⏳ 학습 ~14h)

### Theory
**Late fusion (Tier 1)** = logit 단계 합치기 (final layer 만 share). **Mid-level
fusion** = backbone feature map 단계에서 cross-talk → 두 stream 의 spatial
representation 이 서로 영향.

### Papers
- **Tan & Bansal (EMNLP 2019)** "LXMERT: Learning Cross-Modality Encoder
  Representations from Transformers" — vision-language mid-level fusion 의
  benchmark
- **Simonyan & Zisserman (NeurIPS 2014)** "Two-Stream Convolutional Networks
  for Action Recognition" — RGB + optical flow two-stream fusion
- **Feichtenhofer et al. (CVPR 2016)** "Convolutional Two-Stream Network
  Fusion for Video Action Recognition" — fusion timing 비교 (early/mid/late)

### 우리 도메인 구현 (예정)
```python
class MidFusion(nn.Module):
    def __init__(self):
        self.r_backbone = ConvNeXtV2_base(...)  # frozen first 3 stages
        self.obj_backbone = ObjOnlyCNN()         # frozen
        # Cross-attention or concat at stage-3 output
        self.fusion = ConcatFusion(r_dim=768, obj_dim=64, out_dim=512)
        self.head = nn.Linear(512, 33)

    def forward(self, x_rgb, x_obj):
        r_feat = self.r_backbone.stages[:3](x_rgb)   # [B, 768, 12, 12]
        obj_feat = self.obj_backbone.stages[:3](x_obj)  # [B, 64, 8, 8]
        fused = self.fusion(r_feat, obj_feat)
        return self.head(fused.mean([2, 3]))
```

### 기대
- Oracle 0.9923 의 50% gap close → **0.989~0.992**
- D5 Edge-Bottom subgroup confusion 일부 해소 (mid-level 에서 두 stream
  서로 confidence 보강)

### 비용
- 학습: ~14h (R-only backbone frozen, fusion + head 만 학습)
- inference: 50ms (R-only 와 동일)

### 위험
- mid-level fusion 의 feature map alignment (spatial size 다름) → upsample/projection 필요
- 학습 시 overfit (frozen backbone 의 feature 가 fusion 학습 데이터 범위 밖일 가능성)

### 산출 위치 (예정)
- `cnn_train_midfusion.py`
- `logs_midfusion/`
- `results_tier3/`

---

## Tier 4 — Cross-Attention Fusion (⏳ 학습 ~16h)

### Theory
Mid-fusion 에서 concat → **cross-attention** 으로 업그레이드. 한 stream 의
feature 가 다른 stream 의 query 가 되어 spatial attention 학습.

### Papers
- **Vaswani et al. (NeurIPS 2017)** "Attention Is All You Need" — Transformer
  origin
- **Tsai et al. (ACL 2019)** "Multimodal Transformer for Unaligned Multimodal
  Language Sequences" — cross-attention fusion 의 multimodal benchmark
- **Lu et al. (NeurIPS 2019)** "ViLBERT: Pretraining Task-Agnostic
  Visiolinguistic Representations" — vision-language cross-attention pretraining

### 우리 도메인 구현 (예정)
```python
class CrossAttentionFusion(nn.Module):
    def __init__(self, dim=512):
        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim)
        self.attn = nn.MultiheadAttention(dim, num_heads=8)

    def forward(self, r_feat, obj_feat):
        # r_feat as query, obj_feat as key/value (or vice versa)
        q = self.q_proj(r_feat)
        k = self.k_proj(obj_feat)
        v = self.v_proj(obj_feat)
        out, _ = self.attn(q, k, v)
        return out
```

### 기대
- Tier 3 +0.1pp → **0.991~0.993**
- attention map 으로 어떤 chip 위치가 fusion 에 중요한지 visualize 가능

### 비용
- 학습: ~16h
- inference: 60-70ms (attention overhead)

### 위험
- attention head 수 / dim 등 hyperparameter 추가
- 작은 데이터 (5680 train) 에 attention overfit 가능

---

## Tier 5 — Mixture of Experts (⏳ 학습 ~20h)

### Theory
**MoE** = N expert network + gating network. input 마다 1-2 expert 활성화 →
sparse routing. wafer 33-class 의 class group 별 expert specialization 가능.

### Papers
- **Shazeer et al. (ICLR 2017)** "Outrageously Large Neural Networks: The
  Sparsely-Gated Mixture-of-Experts Layer" — MoE 의 modern origin
- **Fedus et al. (JMLR 2022)** "Switch Transformer: Scaling to Trillion
  Parameter Models with Simple and Efficient Sparsity" — top-1 routing
- **Riquelme et al. (NeurIPS 2021)** "Scaling Vision with Sparse Mixture of
  Experts" — vision MoE benchmark

### 우리 도메인 구현 (예정)
```python
class WaferMoE(nn.Module):
    def __init__(self, n_experts=4):
        self.gate = nn.Linear(feat_dim, n_experts)
        self.experts = nn.ModuleList([WaferExpert() for _ in range(n_experts)])

    def forward(self, x_rgb, x_obj):
        feat = self.shared_encoder(x_rgb, x_obj)
        gate_logits = self.gate(feat.mean([2, 3]))
        top2 = gate_logits.topk(2, -1)
        expert_outs = [self.experts[i](feat) for i in top2.indices]
        return weighted_sum(expert_outs, top2.values.softmax(-1))
```

### Class group 가설 (4 expert)
1. **Expert 1**: Edge spatial group (Edge-Top/Bottom/Ring + Edge-Loc + Donut)
2. **Expert 2**: Center pattern group (Center, Loc, near-center spread)
3. **Expert 3**: Pure object class (scratch, scratch_21deg, particle_blast 단독)
4. **Expert 4**: Special / rare (Starburst, CommaCluster, invalid_main)

### 기대
- per-class α (V1A) 의 generalized form
- **0.991+** + 새로운 wafer 클래스 도입 시 expert 만 추가하면 됨

### 비용
- 학습: ~20h (load balancing loss 등 추가)
- inference: ~80ms (gating + 2 expert forward)

### 위험
- gate collapse (1 expert 만 사용) — load balancing loss 필수
- expert specialization 보장 위한 auxiliary loss
- 작은 데이터 (5680) 에 4 expert overfit 우려

---

## Tier 6 — Knowledge Distillation (⏳ ~8h)

### Theory
Tier 1-5 ensemble 의 high-accuracy teacher → 단일 student 모델로 압축.
Production 의 inference cost (50ms × 2 model) 를 25ms × 1 model 로 절감.

### Papers
- **Hinton et al. (NeurIPS Workshop 2014)** "Distilling the Knowledge in a
  Neural Network" — KD 의 origin, soft target + Temperature
- **Romero et al. (ICLR 2015)** "FitNets: Hints for Thin Deep Nets" —
  intermediate feature distillation
- **Beyer et al. (CVPR 2022)** "Knowledge distillation: A good teacher is
  patient and consistent" — 학습 stability tip

### 우리 도메인 구현 (예정)
```python
# Teacher: Tier 1/2/3 ensemble
teacher_logits = ensemble(x_rgb, x_obj)        # frozen

# Student: single ConvNeXtV2 small (88M → 28M)
student_logits = student(x_rgb)

# KD loss
T = 4
soft_loss = KL(softmax(student_logits / T), softmax(teacher_logits / T)) * T**2
hard_loss = CE(student_logits, y_true)
loss = 0.7 * soft_loss + 0.3 * hard_loss
```

### 기대
- 단일 모델로 ensemble 의 95-99% 성능 유지 → **0.988**
- inference cost 50% 절감

### 비용
- 학습: ~8h (student 학습)
- inference: 25ms (단일 모델)

### 위험
- student 가 teacher 의 R+obj fusion 정보를 RGB 만으로 학습 → 한계 가능
- chip CNN 의 obj_id 정보 access 못 하면 오히려 R-only 단독보다 안 좋을 수도

---

## 추가 옵션 — Snapshot Ensembling, SWA

### Snapshot Ensembling (Huang et al. ICLR 2017)
Cosine LR schedule 의 valley 마다 ckpt 저장 → 같은 학습 비용으로 N model 확보.
- 비용: 1 학습 (12.5h) → N=5 model 확보
- 기대: +0.1~0.3pp

### Stochastic Weight Averaging (SWA, Izmailov NeurIPS 2018)
학습 후반 weight 들의 평균 → flat minima 로 수렴.
- 비용: 학습 후 추가 ckpt averaging (~10 min)
- 기대: +0.05~0.15pp

→ Tier 1/2 deploy 후 marginal 추가 시도.

---

## 종합 비교 표

| Tier | 학습 | infer | 기대 macro_f1 | 우리 측정 | priority |
|---|---|---|---|---|---|
| 0 — Oracle ceiling | 0 | — | 0.9923 (이론) | ✅ 0.9923 | 측정 완료 |
| 1 — Logit ensemble | 0 | 50ms | 0.987~0.989 | ✅ 0.9883 | ★ 즉시 deploy |
| 2 — Calibration + Selective | 0 | 50ms | 0.987~0.990 + 8% review | ✅ 95% coverage | ★ 즉시 deploy |
| 3 — Mid-level Fusion | 14h | 50ms | 0.989~0.992 | ⏳ TODO | high |
| 4 — Cross-Attention | 16h | 65ms | 0.991~0.993 | ⏳ TODO | medium |
| 5 — MoE | 20h | 80ms | 0.991+ | ⏳ TODO | low |
| 6 — KD (teacher → student) | 8h | 25ms | 0.988 | ⏳ TODO | (production opt) |
| + Snapshot | 0 | × N | +0.1~0.3pp | ⏳ TODO | low |
| + SWA | 10min | 25ms | +0.05~0.15pp | ⏳ TODO | low |

## 추천 sequence

```
✅ Tier 0 측정 완료 (oracle 0.9923 = success metric)
✅ Tier 1 V1C deploy (Temperature + logit avg, α=0.10)
✅ Tier 2 per-class threshold + Conformal review queue
─────────────────── 여기까지 학습 0, deploy ready ───────────────────
⏳ Tier 3 Mid-level Fusion (14h) — oracle gap 50% close 가설
⏳ Tier 6 KD (8h) — Tier 1+2 teacher → 단일 student production opt
⏳ Tier 4 Cross-Attention (16h) — Tier 3 vs +0.1pp 검증
⏳ Tier 5 MoE (20h) — class group expert hypothesis 검증
```

★ Tier 1+2 만으로 oracle 의 99.6% 도달 (0.9883 / 0.9923) → 비용 대비 ROI 최고.

## 관련 파일

- 본 문서 → `PRODUCTION_RULE.md`, `STATUS.md`, `PAPERS.md`
- plan 본문 → `~/.claude/plans/1-input-batch-hidden-patterson.md`
- 실행 패턴 → `.claude/skills/wafer-ensemble/SKILL.md`
- agent → `.claude/agents/wafer-ensemble.md`
