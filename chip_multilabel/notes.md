# Chip Multi-Label — 작업 노트 (실시간 갱신)

## iter37 — explicit (A_label, B_label) asymmetric sweep (260510, queued)

★ paper §6.16 NEW axis — symmetric (default) / area-prop (iter35) 외 3rd label rule.

### Trainer patch (~15 lines)
- new CLI flag `--cutmix-ab-labels "A,B"` (e.g. `1.0,0.5`) — overrides `--cutmix-complete-label-scale` and `--cutmix-label-area-prop`.
- mix chip: `mix_t[a_cls]=a_lbl`, `mix_t[b_cls]=b_lbl` 독립 지정.
- mask chip: A only → `mask_t[a_cls]=a_lbl` (b_lbl 무관).

### Sweep (12 cells, T7+LS0.20, batch=2 accum=8, seed=1, complement masked corner, p=0.25)

| cell | g | A_label | B_label | 의도 |
|---|---:|---:|---:|---|
| A | 2 | 1.0 | 0.5 | A hard, B half |
| B | 2 | 1.0 | 0.75 | A hard, B 3/4 |
| C | 2 | 0.5 | 1.0 | A half, B hard (대칭 swap) |
| D | 2 | 0.75 | 1.0 | A 3/4, B hard |
| E | 3 | 1.0 | 0.5 | g=3 분포로 동일 패턴 |
| F | 3 | 1.0 | 0.75 | |
| G | 3 | 0.5 | 1.0 | |
| H | 3 | 0.75 | 1.0 | |
| I | 4 | 1.0 | 0.5 | g=4 |
| J | 4 | 1.0 | 0.75 | |
| K | 4 | 0.5 | 1.0 | |
| L | 4 | 0.25 | 1.0 | area-prop matched (g=4 → a_frac=0.25, b_frac=0.75) |

### Dispatch
- script: `D:/project/known-cnn/_run_iter37_AB_labels.sh` (100755).
- chained polling on `outputs/_iter36_g2_LS_sweep.log` "[iter36] DONE".
- log: `outputs/_iter37_AB_labels.log`.
- bg pid: dispatched via `nohup bash _run_iter37_AB_labels.sh &`.
- 12 trains × ~12 min ≈ 2.5 hr (start when iter36 finishes).

### Eval
- v15direct only (`chip_multilabel_v15direct`), I3/I6/I7/I10, n=50/cls, strength≥0.0, seed=42.

### Hard rules 준수
- TTA 영구 금지, batch=2 accum=8 안전, classification_chips/ 만 학습, 결과 폴더 삭제 X, atomic 변경 1회 (label rule axis).

---

## v20 T7N retrain (260507)

source: master `classification_chips/` v20 (fork sigma 1.0-1.5 → 1.8-2.5, 두께↑) + 7 fork-containing eval-set classes (4 single + 6 2-combo + 4 OOD-overlay 중 3 fork-containing) v20 reblended.
spec: T7 + LS 0.20 + CutMix p=0.25 rect=0.5, 8 epochs, batch=8 accum=4, lr-head=1e-4, seed=42, Normal training 자동 활성화 (`--no-normal` X).
run: `outputs/T7_T7N_v20_seed42_260507_063032/` (학습 ~6분 23초, val_acc 1.0 saturate at ep1, eval ~1분).

### 결과 비교

| metric | T7N v19zpp Cycle B baseline | T7N v20 | Δ |
|---|---:|---:|---:|
| CF1 (macro F1) | 0.9406 | **0.9226** | -0.0180 |
| F1_fork | 0.8682 | 0.8591 | -0.0091 |
| F1_bb | 0.9797 | 0.9719 | -0.0078 |
| F1_sc | 0.9165 | 0.8658 | -0.0507 |
| F1_sr | 0.9979 | 0.9937 | -0.0042 |
| ni_chip_FAR | 0.00% | 0.00% | 0 |
| ood_chip_FAR | 1.41% | 0.94% | -0.47pp |
| ood_overlay 2bit_recall (overall) | (Cycle B 부분) | **0.7500** | — |

### OOD-overlay 4 class 별 2-bit recall

| class | n | exact_2bit_recall | partial_1bit | miss |
|---|---:|---:|---:|---:|
| fork+scratch+ood_DiagonalSmear | 160 | 0.7188 | 0.2437 | 0.0312 |
| bank_boundary+fork+ood_CenterDonut | 160 | 0.8000 | 0.2000 | 0.0000 |
| fork+scratch_rot+ood_CrossScratch | 160 | **0.5687** | 0.3812 | 0.0000 |
| scratch+scratch_rot+ood_Starburst | 160 | 0.9125 | 0.0875 | 0.0000 |

★ Cycle B weak point (`fork+scratch_rot+ood_CrossScratch` 77/160) 가 v20 에서 91/160 → **0.5687** 로 +9% 향상. 다른 OOD 도 미미한 변동 내 안정.

### 7 fork-containing class fork bit recall

| class | n | fork_TP | fork_FN | fork_recall |
|---|---:|---:|---:|---:|
| fork (single) | 160 | 160 | 0 | **1.0000** |
| bank_boundary+fork | 160 | 146 | 14 | 0.9125 |
| fork+scratch | 160 | 153 | 7 | 0.9563 |
| fork+scratch_rot (이전 weak 0.625) | 160 | 115 | 45 | **0.7188** |
| bank_boundary+fork+ood_CenterDonut | 160 | 128 | 32 | 0.8000 |
| fork+scratch+ood_DiagonalSmear | 160 | 143 | 17 | 0.8938 |
| fork+scratch_rot+ood_CrossScratch | 160 | 91 | 69 | 0.5687 |

★ `fork+scratch_rot` 0.625 → **0.7188** (+0.094, fork 두께 ↑ 의 직접 효과 — fork bit 가 sr 에 가려지던 약점 부분 회복).
★ fork single recall **1.0000** — 두께 ↑ 후 fork single 패턴 인식은 saturated.

### fork 두께 ↑ effect summary

- ✓ fork single recall 100% saturate, `fork+scratch_rot` recall +9.4% (이전 weak point fix).
- ✗ overall CF1 -0.018 (0.9406 → 0.9226) — 주로 F1_sc -0.051 drop. v20 retrain 이 fork 외 class 의 fine-tuned threshold 를 미세하게 흔든 듯.
- ✓ ni_chip_FAR 0.00% lock 유지 (Normal training 효과 보존).
- ✗ OOD-overlay overall 2bit_recall 0.75 — `fork+scratch_rot+ood_CrossScratch` 0.5687 여전히 weak (sr+CrossScratch overlap 의 본질적 어려움).

### Conclusion

v20 두께 ↑ 가 fork 자체 (single 1.0, fork_sr +9%) 는 향상시켰으나 single-seed retrain noise 로 sc/CF1 가 약간 저하. 이는 atomic 변경 1회의 단일 측정 — Cycle B baseline 도 단일 seed.
다음 step 후보:
1. seed sweep (42, 1, 7) 로 v20 평균 측정 → noise 제거.
2. v20 + T7N+T5 70:30 ensemble 재현 (iter 12 Cycle A 의 logit-avg lever) → CF1 0.9083+ 회복 가능성.
3. `fork+scratch_rot+ood_CrossScratch` 0.57 의 본질적 한계 — sr+CrossScratch (둘 다 회전 패턴) 분리는 single training 으로 부족, augment / loss 변경 필요.

### 산출 파일

- `outputs/T7_T7N_v20_seed42_260507_063032/` — best_model.pth + final_epoch_model.pth + history.json + train_summary.json
- `outputs/T7_T7N_v20_seed42_260507_063032/eval_I3/stage1_260507_064111/` — preds_chip.parquet, report.md, errors/
- `outputs/T7_T7N_v20_seed42_260507_063032/eval_I3/bit_metrics_split.json` — split FAR + per-class + OOD-overlay 4-class breakdown

---

## iter 12 v19z++ Cycle A (260507) — split FAR + Normal training + ensembles

source: 8 v19zpp models (260506 23~24시) + 1 new T7-with-Normal model (260507 00:22).
goal: chip_FAR 96% lock 깨기 (no-Normal training 한계) + fork F1 ceiling lift.

### 핵심 변경
1. `_bit_metrics.py` patch — chip_FAR 을 3 group 으로 split:
   - **normal_invalid**: ('Normal', 'Invalid') 200 chip ★ paper main metric
   - **normal_only**: ('Normal',) 160 chip
   - **ood**: 5 wafer-pattern OOD 800 chip (diagnostic only)
   - legacy bundled `chip_FAR` (1000 chip 합산) backward-compat 유지
2. T7-with-Normal 학습 (`classification_chips/Normal/` 200 chip 추가, y=-1 sentinel)
3. `_logit_avg_ensemble.py` 신규 — post-hoc prob-avg + 새 thresholds + decision_tree

### Step 1: 8 v19zpp split-FAR 결과

| variant | CF1    | F1_micro | ni_chip_FAR | normal_chip_FAR | ood_chip_FAR | 3plus% |
|---------|--------|----------|-------------|-----------------|--------------|--------|
| T0      | 0.7645 | 0.7346   | 80.00%      | 100.00%         | 100.00%      | 0.09%  |
| T1      | 0.7403 | 0.7141   | 80.00%      | 100.00%         | 100.00%      | 0.03%  |
| T3      | 0.7604 | 0.7098   | 80.00%      | 100.00%         | 100.00%      | 3.33%  |
| T4      | 0.7642 | 0.7134   | 80.00%      | 100.00%         | 100.00%      | 0.06%  |
| T5      | 0.8349 | 0.7878   | 80.00%      | 100.00%         | 100.00%      | 1.79%  |
| T6      | 0.6531 | 0.6300   | 80.00%      | 100.00%         | 100.00%      | 0.09%  |
| T7      | 0.8490 | 0.8038   | 80.00%      | 100.00%         | 100.00%      | 1.98%  |
| T9      | 0.8258 | 0.7772   | 80.00%      | 100.00%         | 100.00%      | 0.99%  |

**핵심 발견**: 96% bundled chip_FAR 의 정체 = 100% Normal_only_chip_FAR + 100% ood_chip_FAR.
no-Normal 학습 → Normal 인식 0% (BCE 가 Normal 신호 학습 못 함). Invalid 는 heuristic 으로 깔끔히 잡혀서 (40 chip 0% FP) ni 의 80% = 160/200 (Normal 100% × 160chip).

### Step 2: no-Normal singles 9 ensemble pairs

| pair    | weights | CF1    | F1_micro | F1_fork | ni_chip_FAR | ood_chip_FAR | 3plus% |
|---------|---------|--------|----------|---------|-------------|--------------|--------|
| T5+T9   | 50:50   | 0.8456 | 0.7918   | 0.5255  | 80.00%      | 100.00%      | 3.95%  |
| T5+T9   | 60:40   | 0.8457 | 0.7918   | 0.5268  | 80.00%      | 100.00%      | 3.80%  |
| T5+T9   | 40:60   | 0.8454 | 0.7926   | 0.5261  | 80.00%      | 100.00%      | 4.04%  |
| T5+T9   | 70:30   | 0.8455 | 0.7916   | 0.5270  | 80.00%      | 100.00%      | 3.89%  |
| T5+T9   | 30:70   | 0.8449 | 0.7902   | 0.5263  | 80.00%      | 100.00%      | 4.81%  |
| T5+T7   | 50:50   | 0.8533 | 0.7997   | 0.5282  | 80.00%      | 100.00%      | 2.47%  |
| T5+T7   | 60:40   | 0.8533 | 0.7999   | 0.5283  | 80.00%      | 100.00%      | 2.90%  |
| T7+T9   | 50:50   | 0.8576 | 0.8100   | 0.5297  | 80.00%      | 100.00%      | 1.39%  |
| T7+T9   | 60:40   | 0.8583 | 0.8136   | 0.5317  | 80.00%      | 100.00%      | 1.79%  |

**핵심 발견**: no-Normal × no-Normal ensemble 은 ni_chip_FAR 100% lock 그대로 (80% = 160/200).
fork F1 도 0.53 ceiling. ensemble 만으론 약점 못 깬다 — Normal 학습 lever 필요.

### Step 3: T7-with-Normal single training

학습: `classification_chips/Normal/` 에 master 의 200 Normal chip 복사 후 (`--no-normal` 빼고)
T7 setting (epochs=8, batch=8 accum=4, lr-head=1e-4, ls=0.20, cutmix-p=0.25 rect=0.5).
data: train=800 val=200 + 200 Normal (y=-1 sentinel multi-hot [0,0,0,0]).
out: `outputs/T7_T7_with_normal_v19zpp_seed42_v2_260507_002217/`

| Model | CF1    | F1_micro | F1_fork | F1_sc  | F1_sr  | F1_bb  | ni_chip_FAR | normal_chip_FAR | ood_chip_FAR | 3plus% |
|-------|--------|----------|---------|--------|--------|--------|-------------|-----------------|--------------|--------|
| T7-no-Normal (baseline) | 0.8490 | 0.8038 | 0.4933 | 0.9489 | 0.9982 | 0.9555 | 80.00% | 100.00% | 100.00% | 1.98% |
| T7-with-Normal (★ new)  | 0.9042 | 0.9041 | 0.7796 | 0.8676 | 0.9973 | 0.9722 |  0.00% |   0.00% |  16.38% | 1.42% |
| Δ                       | +0.055 | +0.100  | +0.286  | -0.081 | -0.001 | +0.017 | -80%        | -100%           | -83.62%      | -0.56% |

**핵심 발견**: Normal training 이 chip_FAR 96%→13.1% lock 을 단독으로 깬다 (memory rule 입증).
ni_chip_FAR 80%→0% (Normal 학습 직접 효과). ood_chip_FAR 100%→16% (Normal 학습 한 번에 OOD
generalization 도 따라온다 — high-confidence threshold suppresses cross-domain false alarms).
fork F1 0.49→0.78 (+0.29) — Normal training 으로 fork 의 sigmoid 분포가 더 sharp.
trade-off: scratch F1 0.95→0.87 (-0.08, fork 와 cross-class 영향).

### Step 4: T7-with-Normal × no-Normal ensembles (★ paper main)

| pair    | weights | CF1    | F1_micro | F1_fork | F1_sc  | F1_sr  | ni_chip_FAR | ood_chip_FAR | 3plus% |
|---------|---------|--------|----------|---------|--------|--------|-------------|--------------|--------|
| T7N+T5  | 50:50   | 0.8844 | 0.8899   | 0.6697  | 0.8912 | 0.9955 | 12.50%      | 22.50%       | 0.28%  |
| T7N+T5  | 60:40   | 0.9018 | 0.9035   | 0.7389  | 0.8878 | 0.9964 |  2.00%      | 22.38%       | 0.77%  |
| T7N+T5  | 40:60   | 0.8648 | 0.8530   | 0.5901  | 0.8952 | 0.9947 | 80.00%      | 66.75%       | 0.62%  |
| T7N+T5  | 70:30   | **0.9083** | 0.9080 | **0.7656** | 0.8853 | 0.9969 |  **0.50%** | 21.88%   | 1.45%  |
| T7N+T9  | 50:50   | 0.8847 | 0.8840   | 0.6634  | 0.9088 | 0.9951 | 77.50%      | 26.87%       | 0.34%  |
| T7N+T9  | 60:40   | 0.9001 | 0.9030   | 0.7281  | 0.9039 | 0.9960 | 13.00%      | 19.25%       | 0.34%  |
| T7N+T7  | 50:50   | 0.8805 | 0.8887   | 0.6025  | 0.9366 | 0.9982 |  0.00%      | 32.12%       | 0.09%  |
| T7N+T7  | 60:40   | 0.9043 | 0.9089   | 0.6988  | 0.9379 | 0.9978 |  0.00%      | 23.13%       | 0.19%  |

**핵심 발견** (iter 10 winner mechanism 재현):
- T7N+T5 70:30 = ★ overall winner (CF1 0.9083, ni_FAR 0.5%, fork 0.77).
- T7N anchor (≥60% weight) 가 Normal 인식 lock-in. T5 minority weight 가 sc 에서 lift (T7N 0.87 → ensemble 0.89).
- T7N+T7 60:40 = sc/sr ceiling 표 (sc 0.9379, sr 0.9978) 만 보면 best 지만 fork 약함 (0.70).
- complementary diversity > correlated quantity (T7N+T7 보다 T7N+T5 가 평균적으로 ↑).

### ★ FAR ≤ 5% 제약 winner ranking

constraint: CF1 ≥ 0.83 + F1_fork ≥ 0.55 + **normal_invalid_chip_FAR ≤ 5%** (paper threshold).

| rank | name                | CF1    | fork_f1 | ni_FAR | ood_FAR |
|------|---------------------|--------|---------|--------|---------|
| 1    | **T7N+T5_w70_30** ★ | 0.9083 | 0.7656  |  0.50% | 21.88%  |
| 2    | T7N+T7_w60_40       | 0.9043 | 0.6988  |  0.00% | 23.13%  |
| 3    | T7N_single          | 0.9042 | 0.7796  |  0.00% | 16.38%  |
| 4    | T7N+T5_w60_40       | 0.9018 | 0.7389  |  2.00% | 22.38%  |
| 5    | T7N+T7_w50_50       | 0.8805 | 0.6025  |  0.00% | 32.12%  |

vs. **v19y T5 baseline** (paper old headline): CF1 0.8162, chip_FAR 3.30% (bundled).
vs. **v19z++ T7-no-Normal** (paper "fix"): CF1 0.8490, ni_chip_FAR 80%.

### Conclusions (Cycle A)

1. ★ paper headline change: `chip_FAR` 단일 metric 폐기 → `normal_invalid_chip_FAR` (real-env)
   + `ood_chip_FAR` (diagnostic) 분리. v19zpp 96% chip_FAR 의 80% 가 OOD 잡음, 16% 가 Normal-no-train 한계.
2. ★ Normal training 이 single lever — chip_FAR 96%→0% 단독 해결. iter 10 finding 재확인.
3. ★ Logit-avg ensemble (T7N+T5 70:30) 이 single 모델 추가 lift: CF1 0.9042→0.9083 (+0.004), fork 0.78 유지.
4. ood_chip_FAR 16~22% 잔여 — 다음 cycle 의 lever (cross-domain regularization, 5-class OOD aware loss).
5. ★ winner: **T7N+T5 70:30 ensemble** — CF1 **0.9083** ni_chip_FAR **0.50%** fork F1 **0.77** (vs v19y T5 baseline 0.8162 / 3.30%, vs v19z++ T7 0.8490 / 80%).

### 산출 파일

- `chip_multilabel/_bit_metrics.py` (patched, 3-group split FAR)
- `chip_multilabel/_logit_avg_ensemble.py` (new, post-hoc prob-avg ensemble)
- `outputs/T*_v19zpp*/eval_I3/bit_metrics_split.json` (8 + 1 model split-FAR)
- `outputs/T7_T7_with_normal_v19zpp_seed42_v2_260507_002217/` (T7-with-Normal model)
- `outputs/_iter12_v19zpp_logs/ensemble/*.json` (17 ensemble configs)
- `outputs/_iter12_v19zpp_logs/T7_with_normal_train_v2.log`, `T7_with_normal_eval_v2.log`

---

## iter 12 v19y (260506 17시 시작)

source: `D:/project/data/wm-811k/classification_chips/` (v19y, 200/class 4 train + 200 invalid_main, no Normal)
eval: `D:/project/data/wm-811k/chip_multilabel/` master (17 class incl 5 OOD), `--n-per-class 200` runtime.
fixed args: `--epochs 8 --batch 8 --accum 4 --lr-head 1e-4 --seed 42 --no-normal`, inference `I3`.

| variant | run_dir | CF1 | F1_bit | F1_bb | F1_fork | F1_sc | F1_sr | bit_FAR | chip_FAR | 3plus% |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| T0 | T0_master_v19y_seed42_260506_171913 | 0.7659 | 0.6991 | 0.8654 | 0.4097 | 0.9223 | 0.8660 | 24.45% | 96.00% | 0.88% |
| T1 | T1_master_v19y_seed42_260506_173500 | 0.7329 | 0.7648 | 0.8458 | 0.4025 | 0.7242 | 0.9593 | 0.70% | 2.80% | 0.00% |
| T3 | T3_master_v19y_seed42_260506_174127 | 0.7434 | 0.7766 | 0.8707 | 0.4119 | 0.7376 | 0.9535 | 0.20% | 0.80% | 0.00% |
| T4 | T4_master_v19y_seed42_260506_174755 | 0.7379 | 0.7735 | 0.7957 | 0.4060 | 0.7514 | 0.9984 | 4.45% | 16.50% | 0.00% |
| T5 | T5_master_v19y_seed42_260506_175422 | 0.8162 | 0.8590 | 0.8910 | 0.3985 | 0.9769 | 0.9984 | 0.83% | 3.30% | 0.04% |
| T6 | T6_master_v19y_seed42_260506_180100 | 0.6639 | 0.6685 | 0.8029 | 0.4559 | 0.5460 | 0.8507 | 8.30% | 27.70% | 0.04% |
| T7 | T7_master_v19y_seed42_260506_180717 | 0.7761 | 0.7983 | 0.8282 | 0.4163 | 0.8702 | 0.9897 | 6.63% | 15.80% | 0.04% |
| T9 | T9_master_v19y_seed42_260506_181321 | 0.8109 | 0.7039 | 0.8899 | 0.4151 | 0.9673 | 0.9714 | 24.60% | 96.00% | 7.15% |

### Final ranking (CF1 desc, FAR ≤ 5% 제약)

| rank | variant | run_dir | CF1 | F1_bit | F1_bb | F1_fork | F1_sc | F1_sr | bit_FAR | chip_FAR | 3plus% | FAR_pass |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 1 | T9 | T9_master_v19y_seed42_260506_181321 | 0.8109 | 0.7039 | 0.8899 | 0.4151 | 0.9673 | 0.9714 | 24.60% | 96.00% | 7.15% | ✗ |
| 2 | **T5** ★ | T5_master_v19y_seed42_260506_175422 | **0.8162** | 0.8590 | 0.8910 | 0.3985 | 0.9769 | 0.9984 | 0.83% | 3.30% | 0.04% | ✓ |
| 3 | T7 | T7_master_v19y_seed42_260506_180717 | 0.7761 | 0.7983 | 0.8282 | 0.4163 | 0.8702 | 0.9897 | 6.63% | 15.80% | 0.04% | ✗ |
| 4 | T0 | T0_master_v19y_seed42_260506_171913 | 0.7659 | 0.6991 | 0.8654 | 0.4097 | 0.9223 | 0.8660 | 24.45% | 96.00% | 0.88% | ✗ |
| 5 | T3 | T3_master_v19y_seed42_260506_174127 | 0.7434 | 0.7766 | 0.8707 | 0.4119 | 0.7376 | 0.9535 | 0.20% | 0.80% | 0.00% | ✓ |
| 6 | T4 | T4_master_v19y_seed42_260506_174755 | 0.7379 | 0.7735 | 0.7957 | 0.4060 | 0.7514 | 0.9984 | 4.45% | 16.50% | 0.00% | ✓ (bit) |
| 7 | T1 | T1_master_v19y_seed42_260506_173500 | 0.7329 | 0.7648 | 0.8458 | 0.4025 | 0.7242 | 0.9593 | 0.70% | 2.80% | 0.00% | ✓ |
| 8 | T6 | T6_master_v19y_seed42_260506_180100 | 0.6639 | 0.6685 | 0.8029 | 0.4559 | 0.5460 | 0.8507 | 8.30% | 27.70% | 0.04% | ✗ |

**★ best variant (FAR ≤ 5% pass): T5 (BCE + CutMix p=0.25 rect=0.5)** — CF1 0.8162, F1_bit 0.8590, bit_FAR 0.83%, chip_FAR 3.30%.

### Observations
- **T9 (sigmoid_focal)** has highest CF1 raw 0.8109 but **bit_FAR 24.6% / chip_FAR 96%** (massive over-firing — 1800 fork FP) → fails FAR constraint.
- **T0 (pure CE no LS no CutMix)** also catastrophic FAR 24.45% / 96% — BCE/CutMix combo essential.
- **T6 (BCE→ASL warmup 5)** worst CF1 0.6639 — ASL switch at ep6 destabilizes.
- **fork** is universally weakest (F1 0.40-0.46 across all variants) — chip-level fork pattern hardest.
- **scratch_rot** is universally strong (F1 0.85-1.00) — v19y angular fix held up.
- T5 wins by ASL/BCE-only loss + CutMix p=0.25 + no LS (cf T7 ls=0.20 hurt fork distinction).



---

## iter 12 v19y Phase 4 (260506 18시) — scattered CutMix soft proportional label sweep

base = T5 BCE multi_hot, no LS. fixed: `--cutmix-p 0.25 --cutmix-rect 0.5 --cutmix-mode scattered --cutmix-n-patches 5 --cutmix-discount 0.7`. axis = `--cutmix-total-ratio` × `--cutmix-alpha`. soft `label_B = ratio × discount × alpha`.

| cell | ratio | α | label_B | run_dir | CF1 | F1_bit | F1_bb | F1_fork | F1_sc | F1_sr | bit_FAR | chip_FAR | 3plus% |
|---|---:|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| baseline T5 | — | — | — | T5_master_v19y_seed42_260506_175422 | **0.8162** | 0.8590 | 0.8910 | 0.3985 | 0.9769 | 0.9984 | 0.83% | **3.30%** | 0.04% |
| T5g | 0.3 | 1.5 | 0.315 | T5_T5g_v19y_r03_a15_seed42_260506_190329 | **0.8325** | n/a | 0.7720 | **0.5833** | 0.9794 | 0.9953 | 5.60% | 22.20% | 0.00% |
| T5a | 0.1 | 0.5 | 0.035 | T5_T5a_v19y_r01_a05_seed42_260506_183149 | 0.8198 | n/a | 0.9260 | 0.4339 | 0.9914 | 0.9280 | 24.07% | 96.00% | 2.81% |
| T5b | 0.1 | 1.0 | 0.070 | T5_T5b_v19y_r01_a10_seed42_260506_183811 | 0.8156 | n/a | 0.9117 | 0.4313 | 0.9833 | 0.9360 | 24.20% | 96.00% | 3.81% |
| T5f | 0.3 | 1.0 | 0.210 | T5_T5f_v19y_r03_a10_seed42_260506_185722 | 0.8139 | n/a | 0.7600 | 0.5525 | 0.9640 | 0.9793 | 5.60% | 22.40% | 0.00% |
| T5d | 0.2 | 1.0 | 0.140 | T5_T5d_v19y_r02_a10_seed42_260506_182515 | 0.8085 | n/a | 0.9028 | 0.4204 | 0.9961 | 0.9145 | 24.37% | 96.00% | 1.42% |
| T5c | 0.2 | 0.75 | 0.105 | T5_T5c_v19y_r02_a075_seed42_260506_184438 | 0.7990 | n/a | 0.9062 | 0.4065 | 0.9907 | 0.8927 | 24.30% | 96.00% | 10.08% |
| T5e | 0.3 | 0.5 | 0.105 | T5_T5e_v19y_r03_a05_seed42_260506_185107 | 0.7839 | n/a | 0.7901 | 0.4474 | 0.9676 | 0.9307 | 3.95% | 15.80% | 0.00% |
| T5h | 0.4 | 1.0 | 0.280 | T5_T5h_v19y_r04_a10_seed42_260506_190933 | 0.7511 | n/a | 0.7285 | 0.4127 | 0.8673 | 0.9961 | 0.53% | **2.10%** | 0.00% |

### Phase 4 Findings

**Success threshold 검증:**
- ✗ min (CF1 ≥ 0.8162 + chip_FAR ≤ 3.3%): **0 cells pass** — 모든 sweep cell 이 baseline chip_FAR 못 따라잡음.
- ✗ paper-worthy (CF1 ≥ 0.83 + F1_fork ≥ 0.50 + chip_FAR ≤ 5%): T5g 가 CF1 0.8325 + F1_fork 0.5833 도달했으나 chip_FAR 22.20% — FAR 5배 초과로 fail.
- ✗ strong (CF1 ≥ 0.85 + F1_fork ≥ 0.55): no cell.

**ratio×α grid 의 monotonic trend 분석:**
- α=1.5 cell (T5g) 가 sweep 최고 CF1 0.8325 + 최고 F1_fork 0.5833 — α↑ → F1_fork ↑ 가설 약하게 검증 (α=0.5, 1.0, 1.5 in r=0.3 sweep: F1_fork = 0.4474, 0.5525, 0.5833 단조증가).
- 그러나 α↑ 가 F1_fork 향상시키는 동시에 chip_FAR 증가 (α=0.5 → 15.80%, α=1.0 → 22.40%, α=1.5 → 22.20%) — trade-off 명확.
- ratio 의 효과 (α=1.0 fixed): r=0.1 → r=0.4 진행하면 chip_FAR 96% (over-firing) → 22.40% → 2.10% 단조 감소. r=0.4 (T5h) 가 chip_FAR 가장 낮으나 F1_fork 0.4127 회복 안 됨.
- soft label_B 가 너무 작을 때 (≤0.14, T5d/T5c/T5b/T5a/T5e=0.035~0.14): 모델이 patch 영역의 negative class 신호를 confuse → fork over-firing 96% chip_FAR.
- soft label_B 가 클수록 (≥0.21): chip_FAR 감소 + scratch_rot/scratch F1 향상, 그러나 bank_boundary recall 저하 (0.65 수준).

**가장 유망 cell 가설 (T5d, r=0.2 α=1.0) 의 actual fork F1:**
- 예측 (가설): paper-worthy 도달 가능
- 실측: F1_fork 0.4204, chip_FAR 96.00% — over-firing 으로 paper-worthy fail.
- 결론: scattered CutMix label_B 0.14 영역은 baseline T5 (no scattered, single random rect) 의 fork F1 0.3985 와 거의 동일 (0.4204) — scattered patch 가 fork pattern 학습에 직접 도움 안 됨.

### ★ Phase 4 winner (sweep 내부): **T5g** (r=0.3 α=1.5)
- CF1 0.8325 (+0.0163 vs baseline T5)
- F1_fork 0.5833 (+0.1848 — fork F1 큰 향상, sweep 최고)
- chip_FAR 22.20% (baseline 3.30% 대비 6.7배 악화)

### ★ Phase 4 winner (FAR ≤ 5% 제약): **baseline T5** 그대로 (sweep 못 이김)
- 모든 scattered CutMix sweep cell 이 baseline T5 의 FAR 제약 못 만족 또는 CF1 못 따라잡음.
- T5e (r=0.3 α=0.5, label_B=0.105, chip_FAR 15.80%) 가 가장 가까움 — fork F1 0.4474 향상 + bit_FAR 3.95% 만족 but chip_FAR 5% 초과 (15.80%) + CF1 0.7839 < baseline.

---

## iter 12 v19y Phase 4.5 (260506 19시) — CutMix + LS sweep

base = T7 (BCE+LS) / T8 (CE-soft+LS) with random CutMix (single rect, NOT scattered) p=0.25 rect=0.5. axis = LS value.

| cell | variant | LS | run_dir | CF1 | F1_bit | F1_bb | F1_fork | F1_sc | F1_sr | bit_FAR | chip_FAR | 3plus% |
|---|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| baseline T5 | T5 BCE | 0 | T5_master_v19y_seed42_260506_175422 | **0.8162** | 0.8590 | 0.8910 | 0.3985 | 0.9769 | 0.9984 | 0.83% | **3.30%** | 0.04% |
| baseline T7 | T7 BCE+LS | 0.20 | T7_master_v19y_seed42_260506_180717 | 0.7761 | 0.7983 | 0.8282 | 0.4163 | 0.8702 | 0.9897 | 6.63% | 15.80% | 0.04% |
| T7 ls 0.05 | T7 BCE+LS | 0.05 | T7_T7_ls005_v19y_seed42_260506_191635 | **0.8196** | n/a | 0.9180 | 0.3965 | 0.9803 | 0.9836 | 1.73% | 6.90% | 0.00% |
| T7 ls 0.10 | T7 BCE+LS | 0.10 | T7_T7_ls010_v19y_seed42_260506_192301 | 0.8059 | n/a | 0.8387 | 0.4295 | 0.9640 | 0.9913 | 26.82% | 96.00% | 8.04% |
| T7 ls 0.15 | T7 BCE+LS | 0.15 | T7_T7_ls015_v19y_seed42_260506_192920 | 0.8024 | n/a | 0.8878 | 0.4301 | 0.9006 | 0.9913 | 25.70% | 96.00% | 6.38% |
| T8 default | T8 CE-soft+LS | default | T8_T8_v19y_seed42_260506_193540 | 0.7401 | n/a | 0.6289 | 0.4219 | 0.9256 | 0.9842 | 1.85% | 7.40% | 0.00% |

### Phase 4.5 Findings

**Success threshold 검증 (winner = CF1 ≥ T5 baseline 0.8162 + chip_FAR ≤ 5%):**
- ✗ T7 ls 0.05: CF1 0.8196 (+0.0034 over baseline) but chip_FAR 6.90% (5% 초과) → fails.
- ✗ 나머지 모두 baseline T5 못 이김.

**LS axis trend:**
- T7 ls 0.05 → 0.10 → 0.15 → 0.20 진행하면 chip_FAR = 6.90% → 96% → 96% → 15.80% (non-monotonic, 0.10/0.15 catastrophic over-firing).
- LS 0.05 가 sweet spot — over-confidence 약간만 풀어주는 것이 fork over-firing 막음.
- LS 0.10 이상에서는 fork prob distribution 이 너무 평탄해져서 threshold 못 누름 → fork over-fire (FP 1500+).
- F1_fork 는 LS 변화에 거의 무관 (0.39~0.43) — fork pattern 자체가 chip-level 에서 본질적으로 어려움.

**T8 (CE-soft+LS) 결과:**
- CF1 0.7401 (baseline 대비 -0.0761) — multi-label 데이터에 CE-soft 가 BCE 보다 부적합.
- bank_boundary F1 0.6289 (baseline 0.8910 대비 -0.26) — softmax 합=1 제약이 multi-label 패턴 분리 손상.

### ★ Phase 4.5 winner (sweep 내부, CF1만): **T7 ls 0.05**
- CF1 0.8196 (baseline +0.0034 marginal)
- F1_fork 0.3965 (baseline 동등)
- chip_FAR 6.90% (baseline 3.30% 대비 2배 악화)
- FAR ≤ 5% 제약 fail.

### ★ Phase 4.5 winner (FAR ≤ 5% 제약): **baseline T5** 그대로
- 모든 LS sweep cell 이 baseline T5 chip_FAR 3.30% 제약 못 만족.
- T7 ls 0.05 가 chip_FAR 6.90% 로 가장 가깝지만 5% 제약 fail.

---

## ⚠️ 재부팅 직전 status (260506 ~16:18)

### iter 12 진행 status
- ✅ Phase 0~3 완료 (T0~T9 학습 + 17-class master 평가) — 결과는 `docs/chip-multilabel/iters/iter_12_v19_status.md`
- ✅ Phase 2.5 (threshold sweep) 완료 — T9 + global θ=0.6 → macro 0.7154 / bit-FAR 0.57%
- ⚠️ v19 강도 ↑ + scratch_rot 우상향 코드 fix 완료, but **chip 새로 만들기 미완**
- 모든 wafer-gen / canvas-gen / multiprocessing python.exe 종료

### 재부팅 후 first action
1. `classification_chips/{bank_boundary, fork, scratch, scratch_rot, invalid_main}` 모두 비우기
2. `cd dist_apply && python _sample_gen_gpu.py --n 200 --save-workers 8` (v19 GPU 합성, ~25-50분)
3. fork chip defect_ratio 검증 (목표 ≥10%, 기존 6.9%)
4. scratch_rot chip angle 시각 검증 (top tilts right, slope < 0)
5. `python -m chip_multilabel.gen_eval_set ...` (master 재생성)
6. iter 12 학습 재실행 (T0~T9)

상세: `docs/chip-multilabel/iters/iter_12_plan.md`, `iter_12_v19_status.md`.

---

## Hard Rules (사용자 확정)

- **TTA 절대 금지** — I5 (4-view averaging) 가 macro_f1 -0.018 손해. chip 패턴이 회전 의존적 (scratch vs scratch_rot) 이라 averaging 이 신호를 흐림. **앞으로 어떤 inference variant 도 TTA 안 씀**.

---

## Iter 1 (Stage 1 baseline) — 260505_162842

**eval set**: 2200 chip / 11 class @ `D:/project/data/wm-811k/chip_multilabel_eval_full/`
**모델**: `chip5_round4_v14_260505_061558_running/best_model.pth` (학습 X)
**소요**: 6분 (forward 1.2분 + TTA 4.4분 — 마지막 TTA 시간 낭비)

### 6 cell 결과
| cell | macro_f1 | top1_11 | T |
|---|---|---|---|
| **I3** sigmoid+F1max | **0.8466** | 0.6017 | 1.0 |
| I4 TS+I3 | 0.8466 | 0.6017 | 0.376 |
| I1 softmax+F1max | 0.8444 | **0.6324** | 1.0 |
| I5 TTA+TS+I3 | 0.8287 | 0.6011 | 0.362 |
| I2 sigmoid 0.5 | 0.7673 | 0.5739 | 1.0 |
| I0 argmax | 0.7302 | 0.4472 | 1.0 |

### 진단

1. **fork 가 가장 약함** (F1=0.63, threshold=0.12, precision=0.48, recall=0.91)
   - 중요: precision 0.48 — 절반은 잘못된 fork 선언. recall 0.91 — fork 들어간 진짜 케이스는 잘 잡음.
   - 즉 **over-firing**: threshold 너무 낮음 → noise/normal 까지 fork 라고 함
2. Top errors:
   - 160× Normal → fork
   - 155× bank_boundary → bank_boundary+fork
   - 141× bank_boundary+scratch_rot → bank_boundary+fork
3. **TS 무용** (I3 == I4) — threshold tuning 이 calibration 효과 흡수
4. **TTA 손해** — 영구 폐기
5. **softmax+threshold (I1) top1_11 가 최고** (0.6324) — softmax 의 합=1 제약이 11-class 결정에 도움. 다만 multi-hot macro_f1 은 I3 보다 약간 낮음.

### 가설 / 다음 시도

**Inference-side (학습 X, 빠름)**:
- **I6 — prior-aware logit shift**: 학습 분포 (chip_train: 25% per class) vs eval 분포 (combo+normal+invalid 섞임) 의 prior gap 보정. fork 가 over-fire 한다는 건 training prior 가 eval 보다 높음 → log(p_eval/p_train) 만큼 logit 빼기.
- **I7 — combo-aware threshold**: single 결정 threshold 와 combo 추가 declare threshold 분리. combo 추가 declare 는 더 보수적 (예: 0.6 이상).
- **I8 — top-k cap with margin**: top-2 만 항상 고려, 2nd prob > top1 prob × 0.5 일 때만 combo 선언.
- **I9 — per-class temperature** (단일 T 대신 per-class T): fork 만 logit 다운-스케일.

**Training-side (Stage 2)**:
- T1 LS 0.1: overconfidence 완화
- T4 ASL γ_neg=4: fork FP 직접 페널티
- T5 BCE: sigmoid 친화 학습

전략: 먼저 inference-side I6/I7/I8/I9 (몇 분) → 그래도 부족하면 Stage 2 진입.

---

## Iter 2 (Stage 1 + new variants) — 260505_165400

**TTA forward path 영구 제거**, 9 variant (I0-I4 + I6-I9) 동시 실행 / forward 1회 (~72s).

### 9 cell 결과
| cell | macro_f1 | top1_11 | T |
|---|---|---|---|
| **I7** joint coord descent | **0.8485** | **0.6210** | 1.0 |
| I3 sigmoid+F1max | 0.8466 | 0.6017 | 1.0 |
| I4 TS+I3 | 0.8466 | 0.6017 | 0.376 |
| I8 top-2 margin (m=0.6) | 0.8456 | 0.6017 | 1.0 |
| I1 softmax+F1max | 0.8444 | **0.6324** | 1.0 |
| I6 F1max + floor 0.3 | 0.8177 | 0.5881 | 1.0 |
| I9 per-class T | 0.7741 | 0.5341 | 0.730 |
| I2 sigmoid 0.5 | 0.7673 | 0.5739 | 1.0 |
| I0 argmax | 0.7302 | 0.4472 | 1.0 |

### 진단

1. **I7 새 best 이지만 +0.002 미미** — 인퍼런스 트릭 ceiling 임. 0.85 가 학습 변경 없는 한계.
2. **I6 (floor 0.3) 후퇴 −0.029** — fork threshold 0.12 가 그렇게 낮은 것이 실제 최적이었음. 즉 fork prob distribution 자체가 noise 영역에서 평균이 높음 (모델이 noise 를 fork 처럼 봄). floor 로 막는 게 아니라 모델이 noise→fork 못 보게 학습 손봐야.
3. **I9 (per-class T) 후퇴 −0.072** — multi-hot binary CE 로 per-class T LBFGS 가 unstable. 특히 fork class 의 multi-positive (single fork + 4 combo) val 분포가 binary 가정 깨짐.
4. **I8 (top-2 margin) ≈ I3** — margin gating 효과 없음. 이미 threshold 가 비슷한 보호 제공.
5. **top1_11class 1위는 I1** (softmax+thresh, 0.6324) — softmax 의 sum=1 제약이 11-class final decision 에 유리. multi-hot 평가에서는 sigmoid 가 약간 우세.
6. **sigmoid `np.exp` overflow warning** 한 번 — 일부 logit 이 큰데 무시해도 됨 (exp 음수 큰 값 → 0 underflow 만, 결과 정확).

### 결론

**인퍼런스 트릭 plateau ~0.85**. Stage 2 학습 변경 없이는 더 못 깸.

### 다음: Stage 2 (학습 변경)

fork over-firing 근본 원인 = 학습 데이터의 noise 영역에서 fork 패턴 (vertical stripes) 이 다른 패턴 noise 와 구분 약함. 처방:

| variant | 가설 | 예상 효과 |
|---|---|---|
| T1 LS 0.1 | overconfidence 완화 → fork prob 0.9+ noise 에서 안 뜸 | 소폭 개선 (0.86) |
| T4 ASL γ_neg=4 | negative class wrong prediction 직접 페널티 | 강력 (0.87+) |
| T5 BCE multi-hot 1-positive | sigmoid 학습 → inference logit dist 일관 | 강력 (0.87+) |
| T6 BCE→ASL warmup 5ep | M5 패턴, BCE 안정 시작 후 ASL 강하게 | 가장 강력 (0.88+) |

skip: T0 (=I3 baseline 재현, 시간 낭비), T2 (mixup α=0.1 작은 chip 위험 / 효과 미미), T3 (focal — ASL 이 보통 더 좋음).

학습 4 variant × 5 inference variant (I0,I1,I3,I7,I8 — top performer + diversity) = **20 cell 매트릭스**. 각 학습 ~3분 + inference 캐시. 총 ~25분.

---

## Iter 3: I10 entropy-Normal short-circuit — 260505_170827

**Variant**: I10 = I7 (joint coord descent thresholds) + softmax entropy gate. 입력 로짓의 softmax entropy ≥ 0.85·log(4) (= 4-class uniform 의 85%) 이면 모델이 어느 한 chip 패턴에도 confident 하지 않다는 신호 → 곧장 **Normal** 로 선언, threshold 비교 단계 스킵.

### 결과

| cell | macro_f1 | top1_11 | Δ vs I7 |
|---|---|---|---|
| **I10 entropy→Normal** | **0.8542** | **0.6517** | **+0.0057 / +0.031** |
| I7 joint coord descent | 0.8485 | 0.6210 | — |
| I3 sigmoid+F1max | 0.8466 | 0.6017 | -0.0076 / -0.050 |

**새 best macro_f1**. top1_11 점프 (+0.031) 가 macro_f1 점프보다 큼 → Normal 결정 정확도 자체가 좋아짐 (11-class single-pick 평가에서 Normal 이 1 클래스).

### Error type delta (T0__I7 vs T0__I10)

| error_type | I7 | I10 | Δ |
|---|---:|---:|---:|
| wrong_combo | 292 | 273 | **−19** |
| false_positive_fork | 215 | 215 | 0 |
| missed_normal | 160 | 106 | **−54** |
| wrong_normal_entropy | 0 | 19 | +19 |
| **total** | **667** | **613** | **−54** |

### Insight

I10 의 entropy gate 가 직접 노린 것은 **missed_normal** (Normal GT 인데 모델이 fork/scratch 등 declare 한 케이스). 결과:

- **missed_normal −54 (-34%)** — 가장 큰 감소. fork over-firing 의 주된 출처가 Normal 이미지에서 fork 가 살짝 뜨는 것이었는데, 그 케이스들은 실제로 4 chip class 모두에 대해 logit 이 평탄 (high entropy) 하다. entropy ≥ 0.85·log4 컷으로 골라낼 수 있었음.
- **wrong_combo −19** — bonus. 일부 noise 패턴에서 두세 class 가 비슷하게 떠 combo 잘못 선언하던 케이스도 entropy 로 잡힘 → Normal 로 흡수.
- **false_positive_fork 0 변화** — fork 가 *단독으로* 강하게 뜨는 케이스 (low entropy, single peak on fork) 는 entropy gate 가 못 잡음. 여전히 215 건 그대로. Stage 2 학습 변경 (T4 ASL γ_neg, T6 BCE→ASL warmup) 이 필요.
- **wrong_normal_entropy +19** — entropy gate 의 새로운 false positive: 진짜 chip 패턴인데 logit 이 평탄해서 잘못 Normal 처리됨. 19 건은 missed_normal -54 에 비해 작아 net +35 정정.

요약: I10 의 entropy 컷은 **fork-vs-Normal 판별** 의 큰 부분을 해결. 남은 215 false_positive_fork (Normal/다른 패턴 → fork 단독) 가 entropy 로 안 풀리는 핵심 잔존 에러.

### Cross-iteration best macro_f1

| iter | best cell | macro_f1 | Δ |
|---|---|---:|---:|
| 1 | I3 sigmoid+F1max | 0.8466 | — |
| 2 | I7 joint coord descent | 0.8485 | +0.002 |
| 3 | **I10 entropy→Normal** | **0.8542** | **+0.006** |

iter 2→3 의 +0.006 은 iter 1→2 의 +0.002 보다 훨씬 큼 — entropy gate 가 단순 threshold 튜닝보다 더 직교한 신호를 활용. 그래도 inference-side ceiling 은 가까이 왔으니 (남은 215 false_positive_fork 단독 에러), 다음은 Stage 2 학습 변경으로 fork over-fire 의 근원을 다뤄야 함.

---

## Iter 4: Stage 2 학습 + I10 매트릭스 — 260505_173649~174123

Stage 2 main run (`outputs/stage2_260505_170121/`) 은 inference variants I0-I9 만 평가 (I10 추가 전 dispatch 됨). 학습 4 variant 끝난 후 I10 추가 평가 실행.

### 풀 매트릭스 (train × {I3, I7, I10})

| train | inference | macro_f1 | top1_11 | 비고 |
|---|---|---|---|---|
| **T1 CE+LS** | **I10** | **0.8634** | **0.7006** | **OVERALL BEST** |
| T0 (기존) | I10 | 0.8542 | 0.6517 | iter 3 best |
| T0 (기존) | I7 | 0.8485 | 0.6210 | iter 2 best |
| T0 (기존) | I3 | 0.8466 | 0.6017 | iter 1 best |
| T6 BCE→ASL | I3 | 0.8396 | 0.5108 | Stage 2 main best (I10 없을 때) |
| T1 CE+LS | I3 | 0.8378 | 0.6420 | |
| T1 CE+LS | I7 | 0.8289 | 0.6210 | |
| T6 BCE→ASL | I10 | 0.8193 | 0.6256 | |
| T6 BCE→ASL | I7 | 0.8190 | 0.6244 | |
| T5 BCE | I3 | 0.8018 | 0.4426 | |
| T4 ASL | I3 | 0.7806 | 0.5881 | |
| T4 ASL | I7 | 0.7766 | 0.5830 | |
| T4 ASL | I10 | 0.7759 | 0.5830 | |
| T5 BCE | I7 | 0.7589 | 0.5432 | |
| T5 BCE | I10 | 0.7589 | 0.5432 | |

### 진단

**T1 (CE+LS 0.1) 만 학습이 도움** — TAPT backbone 의 강한 prior 를 살짝 부드럽게 정규화. Δ vs T0 best:
- macro_f1: 0.8542 → 0.8634 (+0.0092)
- top1_11: 0.6517 → 0.7006 (+0.0489) **큰 도약**

**T4 (ASL), T5 (BCE), T6 (BCE→ASL) 전부 손해** — 327 chip 단일-positive 학습에 ASL γ_neg=4 / BCE 가 너무 강한 perturbation. 결과:
- T4 macro_f1 −0.078 vs T0
- T5 macro_f1 −0.052 vs T0
- T6 macro_f1 −0.035 vs T0 (warmup 덕분에 손해 작음)

**I10 효과 가 train 별 다름**:
- T0, T1: I10 가 I3 / I7 보다 강함 (entropy gate 발동, 미스 normal 50+개 회수)
- T4, T5, T6: I10 == I7 ~ I3 (entropy gate 거의 발동 안 함). 새 손실로 학습된 모델 logit 분포가 더 sharp 한 single-peak 형태 → softmax entropy 항상 낮음 → entropy gate 트리거 안 됨.
- 따라서 ASL/BCE 모델은 fork over-fire 가 어떤 식으로든 발생하지만 그 "noise → fork" 패턴이 entropy 로 안 잡힘.

### 결론 (iter 4 시점)

**최선 조합: T1 (CE + label smoothing 0.1) + I10 (joint coord descent thresholds + softmax entropy → Normal)** = **0.8634 macro_f1, 0.7006 top1_11**.

학습 측: ASL/BCE 처럼 강한 손실 변경은 작은 데이터 + 강한 TAPT init 환경에서 부정적. mild regularization (LS 0.1) 만 도움.

### Cross-iteration progression

| iter | best cell | macro_f1 | top1_11 | Δ macro_f1 |
|---|---|---|---|---|
| 1 | T0__I3 | 0.8466 | 0.6017 | — |
| 2 | T0__I7 | 0.8485 | 0.6210 | +0.002 |
| 3 | T0__I10 | 0.8542 | 0.6517 | +0.006 |
| **4** | **T1__I10** | **0.8634** | **0.7006** | **+0.009** |

**iter 4 의 +0.009 는 학습+추론 공동 최적화 효과**. 단일 inference 트릭 (+0.006) 또는 단일 학습 변경 (이번 case 기준 −0.005~−0.078) 보다 큼.

### 남은 약점

- **fork single FP 215 건 그대로** (Normal/bank 등 → fork 단독). T1 학습 + I10 entropy 모두 fork 단독 강한 logit 케이스에는 무력. Stage 2 변종 더 (T2 mixup, T7 fork-targeted hard negative) 가 마지막 carries.
- **scratch_rot 헤드 noise prior** — error-analyst 진단 (mean prob 0.74 on Normal). TAPT backbone level 의 문제, retrain 으로 안 풀림.

---

## Iter 5 — Phase A1 LS sweep — 260505_175105

T1 (CE+LS) 의 LS 만 sweep, LR=1e-4, ep=8 fixed, 4 trains 각 ~350s sequential. inference I3/I7/I10 동시.

### A1 + extension 풀 결과

| LS | I3 | I7 | I10 |
|---|---|---|---|
| 0.05 | 0.7899 | 0.7964 | 0.7941 |
| 0.10 | 0.8363 | 0.8220 | 0.8317 |
| 0.15 | 0.8961 | 0.8959 | 0.8900 |
| **0.20** | 0.9239 | **0.9268** ★ | 0.8841 |
| 0.25 | 0.8663 | 0.8647 | 0.8398 |
| 0.30 | 0.8185 | 0.8048 | 0.7680 |
| 0.35 | 0.7279 | 0.7204 | 0.6719 |

곡선 모양: 0.05→0.20 monotonic 상승 → 0.20 sharp peak → 0.20~0.35 monotonic 하락. **LS=0.20 = 명확한 peak**.

best A1: **LS=0.20 + I7 = macro_f1 0.9268, top1_11 0.8449**.

vs iter 4 baseline (LS=0.10 + I10 = 0.8634 / 0.7006): **+0.0634 macro_f1, +0.144 top1_11** — 엄청난 도약.

### 진단

1. **LS monotonic 상승** — 0.05→0.20 까지 macro_f1 가 0.79→0.93 로 단조 증가. 0.20 이 grid 최대값이라 **extension 필수**.
2. **LS 0.10 (default 한 거) 가 진짜 sub-optimal 였음** — fork over-firing 의 근본 원인이 overconfidence 였고, LS 가 0.20 정도 강하게 가야 model 이 noise 영역에서 fork prob 안 띄움.
3. **I10 (entropy → Normal) 이 LS=0.20 에서 후퇴** — LS 강하게 주면 logit 엔트로피 자연스럽게 높아져 entropy gate 가 너무 자주 발동 → Normal 오인 증가. 따라서 LS=0.20 + I7 (joint coord descent threshold) 이 최적.
4. **+0.144 top1_11 가 macro_f1 (+0.063) 보다 큼** — 11-class single-equivalent 결정이 훨씬 정확해짐. Normal/Invalid 잡는 것도 좋아짐.

### 다음

- **A1 extension**: LS ∈ {0.25, 0.30, 0.35} 추가 sweep — LS 더 높여도 계속 좋아질 가능성. 3 trains × 6분 = ~18분
- **A2**: LR sweep at LS* (0.20 또는 extension 결과)
- **A3**: epochs sweep

### A3 결과 (ep ∈ {3, 5, 12}, LS=0.20, LR=1e-4)

| epochs | I3 | I7 | I10 | best variant |
|---|---|---|---|---|
| 3 | 0.8467 | 0.8500 | **0.8763** | I10 |
| 5 | 0.8254 | 0.8236 | **0.8567** | I10 |
| **8** (A1) | 0.9239 | **0.9268** ★ | 0.8841 | **I7 overall** |
| 12 | **0.8926** | 0.8872 | 0.8351 | I3 |

**Phase A 풀체인 최종**: **LS=0.20, LR=1e-4, epochs=8, I7 → 0.9268 / 0.8449**.

Regime change 패턴 (epochs 축):
- ep≤5 학습 부족 → logit 평탄 → I10 entropy gate 효과
- ep=8 sweet spot → logit sharp → I7 joint threshold 정확
- ep=12 slight overfit → I3 단독 per-class 안전

→ paper-narrator §6.2 "entropy gate regime change" 가설이 LS 축뿐 아니라 epochs 축에서도 재확인됨. **두 변수 모두 logit sharpness 를 결정** — 이 sharpness 가 inference 변종 선택을 결정.

---

## 참조 도입 — anomaly-detection BKM (사용자 directive 260505 18:55)

`D:/project/anomaly-detection/train.py:1494-1530` + `:214-253` + `docs/summary.md` 의 검증된 기법:

### LR scheduler: warmup + cosine

현재 우리: 단일 LR, `CosineAnnealingLR(T_max=epochs)` 만, **warmup 없음**.
검증된 형태:
```python
warmup = LinearLR(optimizer, start_factor=0.05, total_iters=warmup_epochs)
cosine = CosineAnnealingLR(optimizer, T_max=epochs - warmup_epochs, eta_min=1e-6)
scheduler = SequentialLR(optimizer, [warmup, cosine], milestones=[warmup_epochs])
```
`start_factor=0.05` 가 핵심 — **gradient spike 방지**. 0.1 도 일부 seed 에서 ep4-8 spike 발생 (anomaly-detection 보고).

우리 LR=3e-4 가 epoch 1 부터 2.88e-4 peak 으로 들어가 collapse — warmup 1-2 ep 추가하면 LR 영역 더 넓힐 수 있음 (LR=3e-4 도 활용 가능 검토).

### Two-LR group (backbone vs head)

검증된 (anomaly-detection):
- `lr_backbone = 2e-5`
- `lr_head = 2e-4` (10x)

우리는 단일 1e-4. backbone 은 TAPT 강하니 더 작게 (예 5e-5) + head 는 더 크게 (예 5e-4) 가 합리적.

### EMA with dynamic decay warmup

`ModelEMA(decay=0.95)` + `decay_t = min(target, (1 + step) / (10 + step))` — 초기엔 빠르게 model 따라가고 점차 0.95 수렴.
small dataset (700 samples × 132 iter/ep × 20ep ≈ 2640 step) 에서 target=0.95 권장.

### gradient clip

검증된 (BKM): `grad_clip = 0.5`. 우리는 1.0 사용 중.

### stochastic depth

검증된 (BKM): `0.05`. 우리는 0.

### Best Known Method (BKM) 테이블 — 도메인 다름 주의

`docs/summary.md:159-172` 에서 binary anomaly chart 도메인 BKM:
| axis | baseline | BKM | F1 |
|---|---|---|---|
| label_smoothing | 0 | **0.02** | 0.9981 |
| focal_gamma | 0 | **2.0** | 0.9964 |
| stochastic_depth | 0 | **0.05** | 0.9985 |
| ema | off | **0.95** | 0.9964 |
| grad_clip | 1.0 | **0.5** | 0.9964 |

**우리 chip multi-label 도메인은 다름** — 우리는 LS=0.20 이 winner (anomaly LS=0.02 보다 10x 큼). 이는 task 차이 (multi-class chip 패턴 vs binary chart) 때문. 그러나 **warmup / EMA / stochastic_depth / two-LR group / grad_clip** 같은 구조적 기법은 도메인 무관하게 도움될 가능성.

### Phase F (best-known-method) 후보

Phase A3 끝나고 (사용자 승인 후) Phase F 에 다음 조합 시도:
1. **Warmup 추가** (start_factor=0.05, warmup_epochs=2) + cosine eta_min=1e-6
2. **Two-LR group**: backbone 5e-5, head 5e-4
3. **EMA(0.95) + dynamic decay warmup**
4. **Stochastic depth 0.05** (timm 의 ConvNeXtV2 drop_path_rate=0.05)
5. **Grad clip 0.5** (현재 1.0)
6. **(우리 winner 유지)** LS=0.20 + I7 inference

이 5개 중 어떤 게 효과적일지는 새 sweep 필요. 모두 동시 도입하면 confound — coordinate descent 권장.

---

## Iter 6 — Phase F warmup/EMA 음성 + T7 CutMix peak — 260505_19~20

### F1 / F2 음성 (anomaly-detection BKM transfer 실패)

| variant | best | Δ vs T1 (0.9268) | 진단 |
|---|---|---|---|
| F1 (warmup 2ep, start_factor 0.05, eta_min 1e-6) | 0.8181 | **−0.109** | small data + TAPT init 에 warmup 불필요. ep1 LR=5e-6 너무 낮아 undertrained. |
| F2 (EMA 0.95, dynamic decay) | 0.8377 | **−0.089** | EMA averaging 이 small data 에 과한 smoothing. ep=8 × ~12 effective steps 동안 EMA ramp-up 부족. |

→ anomaly-detection BKM 이 도메인 다르면 부정 transfer. paper-worthy negative result.

### I11 pair-aware threshold (no retrain) 

T1 best 위에 적용: bb+sr recall **0.325 → 0.481** (+25 chips), but bb+fork chips 31 FP (over-trigger). net macro_f1 **−0.007**. Band-aid 효과, 근본 처방 X.

### T7 (BCE+LS=0.20 + CutMix) cutmix-p sweep

atomic 분해: T1 (CE+LS=0.20) → T7a (BCE+LS=0.20, 같은 ls, NO cutmix) → T7c (+ cutmix=0.5).

| step | cutmix-p | best macro_f1 | bb+sr recall | top1_11 |
|---|---|---|---|---|
| T7a | 0.0 | 0.8577 | — | 0.5534 |
| T7b | 0.3 | 0.8626 | 0.7312 | 0.5511 |
| **T7c ★** | **0.5** | **0.9271** | **0.9562** | **0.8307** |
| T7d | 0.7 | 0.9038 | — | 0.7432 |

**Atomic decomposition**:
- T1 → T7a (CE→BCE only): macro_f1 **−0.0691** (BCE 자체 손해)
- T7a → T7c (+ CutMix p=0.5): macro_f1 **+0.0694** (CutMix 회복)
- T1 → T7c (net): **+0.0003** (tied)

**핵심 발견**: T7c 와 T1 macro_f1 동률, but T7c **bb+sr recall 0.32 → 0.96 (+0.63)**. Operational 우위 T7c.

`scratch_rot` per-class F1 perfect (1.0000) in T7c — CutMix multi-hot 학습이 bb+sr combo 의 visual co-occurrence 를 직접 가르침.

### 잔존 약점 (T7c)
- bb+scratch combo: 78건 → scratch single 로 collapse (bank lost)
- fork+scratch combo: 52건 → scratch single (fork lost)
- bb single: 58건 → bb+fork combo (fork over-fire 잔존)

### 다음 candidates (analyst opus 호출 예정)
- CE+LS+CutMix soft-target hybrid (CE 의 강점 + CutMix 의 multi-label 학습)
- BCE LS sweep (0.10, 0.15) — T7c 의 LS=0.20 BCE 에 정확한 hparam 점인지 검증
- ASL+CutMix
- cutmix-rect sweep (0.25, 0.75)
- Class-balanced CutMix (bb+sr pair force)

---

## Iter 8 — T9 LS sweep on BCE+CutMix base (260505_21~)

T7c 위에 LS sweep (atomic 1 hparam 변경).

| LS | seed | best macro_f1 | top1_11 | bb+sr |
|---|---|---|---|---|
| 0.00 (T9c) | 42 | 0.8609 | 0.6443 | — |
| 0.05 (T9b) | 42 | 0.9449 | 0.8670 | — |
| 0.06 (T9f) | 42 | 0.9401 | 0.8648 | — |
| **0.07 (T9d)** | 42 | **0.9705** | **0.9267** | **0.96** |
| 0.07 (T9g) | 43 | 0.9408 | 0.8307 | — |
| 0.08 (T9e) | 42 | 0.8085 | 0.4523 | — |
| 0.10 (T9a) | 42 | 0.9364 | 0.8489 | 0.88 |
| 0.20 (T7c) | 42 | 0.9271 | 0.8307 | 0.96 |

### Variance 발견

T9d (seed=42) 0.9705 vs T9g (seed=43) 0.9408 = **±0.030 single-seed variance**.

LS curve at cutmix=0.5 robust 결론:
- LS in [0.05, 0.10]: macro_f1 ~ **0.94-0.95** (single-seed 측정 ±0.03)
- LS=0.20: 0.9271
- LS=0.00: 0.8609 (overconfidence collapse)
- LS=0.08 (T9e cliff): 0.8085 ← single-seed outlier 또는 정말로 unstable point

### Honest claim

**T9 family (BCE+LS in [0.05,0.10] + CutMix p=0.5) ≈ macro_f1 0.94 mean**, +0.02 over T1/T7c. bb+sr recall 0.85-0.96 robust.

다음: LS sweep diminishing returns. 다음 axis 로 — analyst (opus) 호출.

---

## Iter 9 — Negative axis sweep (drop_path / cutmix-rect / two-LR 모두 HURT) — 260505_21~

T9d 의 lucky-seed peak 0.9705 + T9g (seed=43) 0.9408 = mean 0.9557 baseline 위에서 4 axis 시도, 모두 음성.

| run | atomic change | seed | macro_f1 | Δ vs T9 baseline (=T9d/T9g 같은 seed) |
|---|---|---|---|---|
| T10a | + drop_path 0.05 | 42 | 0.9160 | **−0.054** |
| T10b | + drop_path 0.05 | 43 | 0.8918 | **−0.049** |
| T11a | cutmix-rect 0.5 → 0.25 | 42 | 0.8646 | **−0.106** |
| T12a | single LR → two-LR (bb 5e-5, head 2e-4) | 42 | 0.8862 | **−0.084** |

drop_path / smaller paste / two-LR — 셋 다 HURT. anomaly-detection BKM 도메인 transfer 또 실패.

T7c → T9 stable winner (baseline+CutMix+small LS). 추가 axis 모두 plateau under negative side.

### Final stable claim (iter 9)

**Winner config**: T7 variant + LS in [0.05,0.10] + CutMix p=0.5 + rect=0.5 + single LR=1e-4 + ep=8 + RandomAffine aug only.

| metric | mean (2 seeds) | min~max |
|---|---|---|
| macro_f1 | **~0.94** | 0.94~0.97 (single-seed lucky) |
| top1_11 | **~0.87** | 0.83~0.93 |
| bb+sr recall | **~0.92** | 0.88~0.96 |
| scratch_rot F1 | **~1.00** | perfect both seeds |

vs Phase A baseline T1 (CE+LS=0.20):
- macro_f1: +0.01~+0.04
- top1_11: +0.02~+0.08
- bb+sr recall: **+0.55~+0.63 (massive)**

### 다음 (analyst 추가 호출 또는 마무리)

추가 시도해볼 axis (analyst-iter8 후보 잔존):
- ASL+cutmix (γ_neg=2 light) — analyst 후보 #9
- per-class LS (fork 0.10, scratch_rot 0.05) — analyst 후보 #6
- inference ensemble T1+T9d sigmoid average — analyst 후보 #7

또는 multi-seed (seed 44, 45) 로 T9 winner mean ±std 정밀화.

Phase G (extended metrics ROC-AUC / PR-AUC / Brier / OOD AUROC) 도 GPU 거의 안 쓰니 가능.

---

## Iter 10 — Variance reality check (T1 multi-seed) — 260505_22

T9h (seed=44) = 0.8803 / bb+sr 0.350 — T9 의 bb+sr "fix" 도 부분적으로 seed-dependent.

T1 baseline 도 multi-seed 측정:
- T1 seed=42: 0.9268 / bb+sr 0.325
- T1 seed=43: 0.8788 / bb+sr 0.819

**seed=43 에서 T1 baseline 도 bb+sr 0.82 자력 달성** — 즉 T7c 의 "0.96 fix" 은 부분적으로 lucky seed convergence 였음.

### Multi-seed 정리

| metric | T1 (2 seeds) | T9 (3 seeds) | per-seed paired (seed=42 / 43) |
|---|---|---|---|
| macro_f1 mean | 0.9028 ± 0.034 | 0.9305 ± 0.046 | T9 (+0.044, +0.062) |
| bb+sr recall mean | 0.572 ± 0.245 | 0.729 ± 0.314 | T9 (+0.631, +0.062) |

**Key finding**: per-seed paired comparison 에서 **T9 항상 T1 보다 우세** (양 metric). 그러나 absolute level 은 ±0.04 single-seed variance 가 매우 큼.

**Honest paper claim**:
1. T9 family (BCE+LS small + CutMix p=0.5) 이 T1 (CE+LS=0.20) 보다 **paired-seed 비교 macro_f1 +0.05, bb+sr +0.10~+0.6** 우세
2. Single-seed mean 만 보면 +0.028 macro_f1, +0.16 bb+sr (둘 다 std 보다 작음)
3. **CutMix mechanism 자체가 우세** — 그러나 small data + 8ep 학습 budget 의 본질적 noise 가 ±0.05
4. paper 에서는 multi-seed std 명시 필수

### 자율 loop 수렴

여태 시도한 axes:
- ✓ Phase A: LS / LR / epochs sweep (real signal, LS=0.20 winner for CE)
- ✓ Phase F: warmup / EMA / drop_path / two-LR (all transfer-fail, BKM 도메인 부적합)
- ✓ T7 CutMix: 진짜 mechanism (single-positive → multi-positive supervised signal)
- ✓ T9 LS sweep: lucky-seed trap 발견 (multi-seed 필요)
- ✓ T8 CE-soft + cutmix: HURT (softmax sum=1 multi-label 부적합)
- ✓ T11 cutmix-rect 0.25: HURT (compositional signal 약화)
- ✓ T12 two-LR: HURT
- ✓ T10 drop_path: HURT both seeds

**남은 candidates** (analyst-iter8 후보 잔존):
- ASL+cutmix γ_neg=2 light (시도 안 됨)
- per-class LS (시도 안 됨)
- inference ensemble T1+T9 sigmoid average (no GPU)

또는 **multi-seed (seed 45, 46) 로 T1/T9 winner mean 정밀화** — variance bound 줄이기.

또는 **Phase G**: extended metrics (ROC-AUC / PR-AUC / Brier / OOD AUROC) on existing best cells — paper appendix.

---

## RESTORE POINT — 260506 00:38 재부팅 직전

### 현재 상태

- **자율 loop 수렴**: T9 family (BCE+LS in [0.05,0.10]+CutMix p=0.5) winner. 3-seed mean macro_f1 0.9305 ± 0.046, bb+sr recall 0.7542 ± 0.350.
- **8 atomic axis 음성 결과** 누적 (warmup, EMA, drop_path, cutmix-rect, two-LR, CE-soft, ASL light, I11).
- **strong-50% eval set** (top-50% defect strength source 으로 합성, 11 class) 에서 T9d macro_f1 0.9815, 9-defect-class macro 0.9497.

### 마지막 작업 (재부팅 시 이어서)

사용자 directive: **scratch+scratch_rot combo 재추가** (이전 plan Q1 에서 제외했던 것 reverted).

**완료된 작업 (저장됨)**:
1. ✅ `constants.py`: COMBO_KEYS 에 `scratch+scratch_rot` 추가 (6번째 combo)
2. ✅ `decision_tree.py`: combo collapse rule 주석 (defensive only)
3. ✅ `_train_chip_variant.py`: CutMix sc+sr disallow filter 제거
4. ✅ `D:/project/data/wm-811k/chip_multilabel_eval_strong50/scratch+scratch_rot/` 200 chips 합성 완료
5. ✅ `chip_multilabel_eval_strong50/manifest.csv` 200 행 append

**다음 step (재부팅 후 즉시)**:
1. T9d 모델로 strong-50 12-class eval inference:
   ```
   python -m chip_multilabel.run_stage1 \
     --eval-set D:/project/data/wm-811k/chip_multilabel_eval_strong50 \
     --out-root outputs --batch-size 16 \
     --model outputs/logs_chip_multilabel/T7_T9d_BCE_LS07_cutmix50_260505_211038/best_model.pth \
     --variants I3,I7,I10,I11
   ```
2. sc+sr per-class F1 측정 (예상 약함 — 학습 시 disallow 였음). 만약 F1 < 0.8 이면 사용자 directive "성능 향상시키고" 따라 재학습 필요.
3. 재학습 시: T7 base + cutmix-p=0.5 + LS=0.07 + sc+sr CutMix **허용** (위 #3 수정 후).

### 파일 변경 (git modified)

| file | 변경 |
|---|---|
| `chip_multilabel/constants.py` | COMBO_KEYS +1 (sc+sr) |
| `chip_multilabel/decision_tree.py` | collapse 주석 |
| `chip_multilabel/_train_chip_variant.py` | sc+sr disallow 제거, CLI flags (ASL, ema, warmup, drop_path 등) |
| `chip_multilabel/gen_eval_set.py` | `--source-strength-pct` flag |
| `chip_multilabel/inference_variants.py` | I6/I7/I8/I9/I10/I11 추가, TTA path dead |
| `chip_multilabel/losses.py` | CESoftLabel, FocalLoss, BCE+LS, build_loss kwargs |
| `chip_multilabel/notes.md` | iter 1~10 + restore point |
| `docs/chip-multilabel/*` | paper sections + iter docs |
| `docs/chip-multilabel/tables/all_runs_macro_f1.csv` | 100+ row |

### 새 파일 (git untracked)

- `.claude/agents/chip-multilabel-analyst.md`, `chip-multilabel-paper-narrator.md`, `chip-multilabel-runner.md`, `chip-multilabel-logger.md`
- `.claude/skills/chip-multilabel-pipeline/SKILL.md`
- `chip_multilabel/run_phase_a.py`, `compare_runs.py`, `_train_chip_variant.py`
- `docs/chip-multilabel/iters/iter_0{6,7,8,9}_*.md`
- `docs/chip-multilabel/analysis/` (analyst-iter6/iter8 보고서)
- `docs/chip-multilabel/paper/` (9 section + diary)
- `outputs/stage1_*` (~30+ runs), `outputs/logs_chip_multilabel/T*` (~25 trains)

### Memory (영구 룰, .claude/memory/)

- TTA 영구 금지
- Rotation/Flip aug 영구 금지
- Atomic 1-method/iter
- analyst/planning agent = opus
- chip class expansion postponed (4 class 유지)
- HDBSCAN cfg sweep OK (encoder 학습 ablation 별개)
- **★ Master 저장 갯수 ≠ runtime 사용 갯수** (260506 추가) — 데이터 폴더 buffer 크게 (defect 200), runtime `--n-per-class` 로 사용 갯수 제어. 사용자 "불량 class 별로 200개 만들어놓고 사용만 50개" directive.

### 오늘 GPU 누적 ~5시간 학습 + ~30 inference + 4 agent 호출.

---

## ★ POLICY (260506 01:30 사용자 directive — 영구 적용)

### Master 폴더 저장 갯수 ≠ runtime 사용 갯수 (분리 정책)

**Rule**: 데이터 폴더 (`D:/project/data/wm-811k/chip_multilabel/`) 의 per-class chip 저장 갯수는 항상 buffer 크게 (200+) 둔다. 학습/평가에서 사용할 갯수는 runtime CLI flag (`--n-per-class`) 로 manifest 에서 sample.

**260506 적용 spec**:
- 저장 (master): defect 10 class × **200** + Normal 200 + Invalid 50 = **2450 chip**
  - ★ defect source 는 **강한 것만** (`--source-strength-pct 50`) — 사용자 directive "불량 200개는 불량강도가 강한것들로만"
- 사용 (eval at runtime): `--n-per-class 50` → defect 10×50 + Normal 50 + Invalid 50 = 600 chip 추론
  - 또는 다른 N 으로 sample (sanity, ablation, multi-seed 모두)

**Why**:
- 사용자 명시 directive: "불량 class 별로 200개 만들어놓고 사용만 50개하라고"
- 같은 데이터 폴더에서 다양한 N (50/100/200) sample 가능해야 함
- 한 번 50/class storage 만들면 100/class 으로 나중에 sample 못 함 — 재합성 시간 낭비
- memory rule `feedback_master_storage_vs_runtime_sampling.md` (260506)

**Subset 폴더 절대 안 만듦** (memory rule `feedback_no_subset_archive_folders.md`).

---

## RESTORE POINT — 260506 01:00 세션 재시작 직전 (이관 후)

### 이관 완료 (unknown-contrastive → known-cnn)

| 항목 | 위치 (known-cnn) | 갯수/크기 |
|---|---|---|
| Python module | `chip_multilabel/` | 19 files |
| Docs | `docs/chip-multilabel/` | 34 files |
| Agents | `.claude/agents/chip-multilabel-{runner,analyst,logger,paper-narrator}.md` | 4 |
| Skill | `.claude/skills/chip-multilabel-pipeline/` | 1 |
| Outputs | `outputs/logs_chip_multilabel/` + `stage1_*` 51 + `stage2_*` 1 + `phase_a_*` 4 | 14GB |
| Memory | `~/.claude/projects/D--project-known-cnn/memory/` (7 chip files + index) | 7 |
| CLAUDE.md chip-multilabel 섹션 | known-cnn tail | ✅ |

unknown-contrastive 의 chip 관련 흔적은 모두 제거됨.

### 재시작 후 첫 inference 실측 결과 (260506 00:57)

T9d (BCE+LS=0.07+CutMix=0.5 winner) on `chip_multilabel_eval_strong50` (12-class, sc+sr 추가됨).

- **Best cell: T0__I3** macro_f1=0.9687 (4-multi), 12-class macro=0.9193, 10-defect macro=0.9095
- **scratch+scratch_rot F1 = 0.755** (P=1.000, R=0.606, sup=160) ★ 약점
  - 64% 만 잡고 36% 는 single sc 또는 single sr 로 흘림
  - 원인: T9d 학습 시 sc+sr CutMix pair **disallowed** (모델이 두 angle 동시 학습 불가)
- 다른 class 모두 F1 ≥ 0.82, combo 5종 ≥ 0.85 robust

run dir: `outputs/stage1_260506_005758/`

### 다음 step (재시작 후 즉시)

**Goal**: sc+sr F1 0.755 → 0.90+ (다른 class 회귀 없이)

1. T9 retrain seed=42 (코드 patch 이미 완료, sc+sr CutMix 허용):
```bash
cd D:/project/known-cnn
python -X utf8 -m chip_multilabel._train_chip_variant \
  --variant T7 --ls 0.07 --lr 1e-4 --epochs 8 --batch 8 --accum 4 \
  --cutmix-p 0.5 --cutmix-rect 0.5 --seed 42 \
  --tag T9d_scsr_seed42 \
  --out-root outputs
```

2. 학습 끝나면 strong-50 12-class 재평가:
```bash
python -X utf8 -m chip_multilabel.run_stage1 \
  --eval-set D:/project/data/wm-811k/chip_multilabel_eval_strong50 \
  --out-root outputs --batch-size 16 \
  --model outputs/logs_chip_multilabel/T7_T9d_scsr_seed42_<TS>/best_model.pth \
  --variants I3,I7,I10,I11
```

3. sc+sr F1 비교 + 다른 class 회귀 검증.
4. seed=43, 44 추가 (3-seed mean — 이전 variance ±0.04 컸음).

### Agent 활용 (재시작 후)

세션 재시작하면 agent 4개 자동 등록:
- `chip-multilabel-runner` — 학습/inference dispatch + GPU 가드
- `chip-multilabel-analyst` (opus) — 결과 분석 + 다음 실험 제안
- `chip-multilabel-logger` — notes.md / docs 업데이트
- `chip-multilabel-paper-narrator` (opus) — paper section 작성

자율 loop 호출 패턴: `Agent(subagent_type='chip-multilabel-runner', prompt='Stage 1 ...')`.

### 미진행 작업

- T9 sc+sr retrain dispatch (1분 진행 후 세션 정리로 kill, 학습 결과 없음)
- 좀비 python 정리 필요할 수 있음 (memory rule: Windows python dispatch 좀비 누적 → torch hang)

### 주의

- `outputs/_train_t9d_scsr_seed42.log` (0 bytes), `outputs/logs_chip_multilabel/` 의 새 dir 없음 — kill 시점에 trainer 가 setup 단계
- 재시작 후 `tasklist /FI "IMAGENAME eq python.exe"` 또는 `Get-Process python` 으로 좀비 잔재 확인. 누적 시 `Stop-Process -Name python -Force` (단 jupyter/IDE 등 사용 중 X 확인 후)

---

## ★ Iter 10 FINAL — H Ensemble Winner (260506 04:00~07:30)

### Phase journey
1. **시점 0**: T9d on master (12-class incl sc+sr) → 10-def macro 0.9095, sc+sr F1 0.755 약점
2. **A1 retry-1 (cutmix=0.5)**: sc+sr 1.000 ✅ but 다른 class 무너짐, Normal 0.000 — net negative
3. **D (cutmix=0.25)**: 회복 0.9116, Normal 0 잔존 (4-class only 한계)
4. **C (Normal training, 5-class y=-1)**: Normal 1.000 lock ✅, 4-multi 0.9610 ± 0.012, **fork+scratch 0.673 새 약점**
5. **G (threshold sweep)**: marginal, weak signal 한계
6. **H (Ensemble baseline + C_44 logit avg)**: ★★★ **10-def macro 0.9950**, all class F1 ≥ 0.987, FAR 0%
7. **F (CutMix fork+scratch pair bias)**: net negative — fork+sc 0.95 살리지만 bb 0.78, fork+sr 0.81 깨짐. ensemble 도 baseline+C_44 보다 낮음 (0.91 vs 0.99)

### Key finding: H ensemble = winner
- baseline T9d (no Normal train, fork-combo prob 0.46 alive) + C_44 (Normal train, sc+sr/Normal locked, fork-combo collapsed) → **complementary**
- logit avg = (0.46 + 0.16) / 2 ≈ 0.31 → joint thr 0.10 catch
- 5-sample-seed mean **10-def macro 0.9930 ± 0.005**, FAR 0.00% ± 0.00% — paper-quality robustness

### 4-single + 6-combo 분리 (eval split 480 chips)
| 그룹 | macro F1 (mean ± std, 5 sample seeds) | 비고 |
|---|---:|---|
| 4-single | 0.9963 ± 0.0045 | 거의 perfect |
| 6-combo | 0.9908 ± 0.0063 | 거의 perfect |
| 10-defect | 0.9930 ± 0.0049 | sc+sr/Normal/Invalid lock 0.000 std |
| 12-class | 0.9942 ± 0.0041 | |
| Normal F1 | 1.000 ± 0.000 | perfect lock |
| Invalid F1 | 1.000 ± 0.000 | heuristic |
| FAR (Normal→defect) | 0.00% ± 0.00% | 1000-chip wafer 0 false alarms |

### 새 Memory rules (260506)
- `feedback_logit_ensemble_complementary.md` — H 결과 영구 기록
- `feedback_normal_training_open_set.md` — C 결과 영구 기록  
- `feedback_cross_class_suppression.md` — fork combo prob collapse 메커니즘
- `feedback_master_storage_vs_runtime_sampling.md` — 200 store / 50 use
- `feedback_chip_train_batch_safe.md` — 공유 GPU batch=8 강제

### 이관 후 폴더 정리 (260506)
- ✅ 옛 eval 폴더 3개 (`_full`, `_strong50`, `_smoke`) 삭제됨
- ✅ master `chip_multilabel/` 만 single SoT (2450 chip, source p50 strong)
- ✅ paper iter_10 doc: `docs/chip-multilabel/iters/iter_10_master_consol_sc_sr.md`

---

## Iter 11 — Paper-style 4-row Ablation Matrix + p30 + Normal diversity (260506 ~)

### 사용자 directive 260506 22:30 + 22:45
1. 전통 single-chip CNN (T1 CE+LS=0.1) + multi-sigmoid pred (Row 1)
2. T1 train + 다양한 inference variants (Row 2)
3. ASL/Focal 등 loss 변경 + I3 (Row 3)
4. Loss × inference full 6×6 matrix (Row 4)
5. 그후 1: p50 → **p30 (top 70%, 약한 defect 까지 포함, 더 어려운 eval)**
6. 그후 2: Normal 너무 동일한 형태 → **noise/분산 추가 다양화**

### 결정사항 (260506 confirmed)
- p30 = top 70% (`--source-strength-pct 70`)
- Normal class 학습 = NO (4-class only, 전통 baseline)

### Common config (모든 6 trains)
- ep=8, batch=8 accum=4, lr=1e-4, cutmix-p=0.25, cutmix-rect=0.5, seed=42
- `--no-normal` flag 추가 (260506) — classification_chips/Normal/ skip
- 6 variants: T1 (CE+LS=0.1), T3 (Focal), T4 (ASL), T5 (BCE), T6 (BCE→ASL), T7 (BCE+LS=0.1)
- 6 inference: I3, I7, I10, I11, I12, I13

### 진행 상태 (실시간) → 모두 완료 ✅

| step | 상태 | 결과 |
|---|---|---|
| Trainer patch (`--no-normal`) | ✅ | classification_chips/Normal/ skip option |
| Phase 1a — 6 trains (T1/T3/T4/T5/T6/T7) | ✅ | 모두 완료, ~3-4분/train |
| Phase 1b — 6 stage1 inference (each × 6 variants = 36 cells) | ✅ | best 0.905 (T6+I3, but FAR 100%) |
| `_make_normal_chip` 다양화 patch | ✅ | 5 variation sources, whiteness ≥ 0.70 sanity 100% pass |
| Phase 2 — p30 regen (top 70%) + 6 re-inference | ✅ | 모든 train Δ macro < 0.02 (robust) |
| Phase 3 — Normal diverse + 6 re-inference | ✅ | T4 ASL 만 N +0.07 / FAR -12.5% (asymmetric mechanism) |
| Phase 4 — paper iter_11 doc | ✅ | `docs/chip-multilabel/iters/iter_11_paper_ablation_matrix.md` |

### 최종 결과 (iter 11)

| Train | Best (over 3 phases) | macro | Normal | FAR |
|---|---|---:|---:|---:|
| **T6 (BCE→ASL)** | P1+I3 / P3+I3 | **0.905** | 0.000 | 100% ❌ |
| T5 (BCE) | P3+I11 | 0.894 | 0.000 | 100% ❌ |
| T7 (BCE+LS) | P2+I7 | 0.860 | 0.000 | 100% ❌ |
| T4 (ASL) | P1+I10 | 0.803 | 0.857 | 18% ⚠ |
| T1 (CE+LS) | P3+I11 | 0.620 | 0.000 | 100% ❌ |
| T3 (Focal) | P1+I11 | 0.513 | 0.974 | 5% ⚠ |

**결론**: 4-class only 학습 catastrophic on Normal. iter 10 ensemble (0.995, FAR 0%) 이 모든 single 압도.

### 7 핵심 finding (paper-worthy)

1. Normal training 누락 = 4-class only 모든 BCE-family 모델 FAR 100% (operationally 불가)
2. Asymmetric (T4 ASL) / Focal (T3) 만 4-class only 환경에서 Normal 자연스럽게 generalize
3. Distribution-shift (p50 → p30) robust — 모든 model Δ macro < 0.02
4. Normal diversity 효과 marginal — T4 ASL 만 receptive (+0.07 N, -12.5% FAR)
5. iter 10 ensemble (baseline + C_44 logit avg) 이 모든 single iter 11 model 압도 by +0.09
6. Best single iter 11 = T6+I3 = 0.905 (단 FAR 100% — 운영 불가)
7. Cross-class suppression 의 fix 는 ensemble — single 모델 + threshold tuning 으로 못 해결

---

## iter 12 v19z++ on stable master (260506-07 새벽)

source: `D:/project/data/wm-811k/classification_chips/` (v19z++, 200/class 4 train, no Normal)
eval: `D:/project/data/wm-811k/chip_multilabel/` master (★ **21 class** — 4 single + 6 2-combo + 4 3-combo + Normal + Invalid + 5 OOD), `--n-per-class 200` runtime.
fixed args: `--epochs 8 --batch 8 --accum 4 --lr-head 1e-4 --seed 42 --no-normal`, inference `I3`.
fresh dispatch: 이전 `_v19z_` 결과는 master schema-shift 도중 → stale. 새 `_v19zpp_` tag 로 재학습.

### 8-variant matrix (CF1 desc)

| variant | run_dir | CF1 | F1_bit | F1_bb | F1_fork | F1_sc | F1_sr | bit_FAR | chip_FAR | 3plus% |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| T7 ★ | T7_T7_v19zpp_seed42_260506_234656 | **0.8490** | 0.8994 | 0.9684 | 0.5248 | 0.9066 | 0.9964 | 24.90% | 96.00% | 1.98% |
| T5 | T5_T5_v19zpp_seed42_260506_233403 | 0.8349 | 0.8897 | 0.9651 | 0.5236 | 0.8686 | 0.9823 | 26.62% | 96.00% | 1.79% |
| T9 | T9_T9_v19zpp_seed42_260506_235314 | 0.8258 | 0.8753 | 0.9191 | 0.5209 | 0.8951 | 0.9683 | 25.35% | 96.00% | 0.99% |
| T0 | T0_T0_v19zpp_seed42_260506_230631 | 0.7645 | 0.8396 | 0.8693 | 0.5453 | 0.8195 | 0.8240 | 28.65% | 96.00% | 0.09% |
| T4 | T4_T4_v19zpp_seed42_260506_232740 | 0.7642 | 0.8205 | 0.9590 | 0.5185 | 0.7618 | 0.8175 | 28.68% | 96.00% | 0.06% |
| T3 | T3_T3_v19zpp_seed42_260506_232032 | 0.7604 | 0.8696 | 0.6269 | 0.5240 | 0.9165 | 0.9741 | 48.60% | 96.00% | 3.33% |
| T1 | T1_T1_v19zpp_seed42_260506_231343 | 0.7403 | 0.8213 | 0.9151 | 0.5601 | 0.6643 | 0.8217 | 31.47% | 96.00% | 0.03% |
| T6 | T6_T6_v19zpp_seed42_260506_234033 | 0.6531 | 0.7378 | 0.8130 | 0.5403 | 0.5363 | 0.7228 | 36.28% | 96.00% | 0.09% |

### FAR ≤ 5% 제약 winner

★ **NONE** — 모든 variant 가 chip_FAR = 96% 로 동일. 이유: master 에 5 OOD class (DiagonalSmear / CenterDonut / CrossScratch / Row / Starburst, 각 200) 가 multihot=zero 로 GT-Normal 처럼 처리되지만 모델은 학습 안 한 OOD 패턴을 4 train class 중 하나로 fire → FAR 분자 1000/1000 = 96% (Normal 200 정확 + Invalid 50 정확 = 분모 250, 분자 0 의 가벼운 contribution 무시되고 OOD 가 dominant). `--no-normal` baseline 한계.
chip_FAR 의미를 보다 정확히 분리하려면: (a) Normal-only chip_FAR (200 chip 분모) 과 (b) OOD chip_FAR (1000 chip 분모) 두 metric 분리 필요. 현재 bit_metrics.json 은 둘을 합쳐 보고 → 추후 split.

### 3-combo 정답률 (pred class_key == GT, all 3 bits exact match)

| variant | bb+fk+sc | bb+fk+sr | bb+sc+sr | fk+sc+sr |
|---|---:|---:|---:|---:|
| T7 ★ | **0.831** | 0.444 | 0.144 | 0.562 |
| T5 | 0.525 | 0.362 | **0.588** | 0.569 |
| T9 | 0.556 | 0.237 | 0.375 | **0.819** |
| T3 | 0.419 | **0.812** | 0.006 | 0.719 |
| T4 | 0.287 | 0.456 | 0.000 | 0.575 |
| T0 | 0.344 | 0.169 | 0.013 | 0.225 |
| T1 | 0.212 | 0.138 | 0.000 | 0.237 |
| T6 | 0.619 | 0.000 | 0.019 | 0.062 |

3-combo 관찰:
- bb+fk+sc 는 T7 (LS=0.20+CutMix) 이 0.831 dominant — bb / sc / sr 각각 F1 0.97+ 인 model.
- **bb+sc+sr** 가 가장 어려운 3-combo (T7 의 sr F1=0.9964 인데 3-bit 매칭 0.144) — fork 신호가 sr 와 confused 되어 over-fire 추정 (사실 fork F1=0.52 가 모든 model 에서 worst).
- T5 가 bb+sc+sr 에서 0.588 로 best — BCE 만 단독 + CutMix 가 fork over-firing 을 가장 잘 억제.
- T9 가 fk+sc+sr 에서 0.819 — sigmoid_focal + CutMix 가 fork true-pos 환경에서 best.
- 어떤 single variant 도 4 3-combo 모두 dominant 안함 → **logit-avg ensemble (T7+T5+T9) 후보**.

### 5 OOD chip_FAR contribution

| variant | DiagSmear | CenterDonut | CrossScratch | Row | Starburst |
|---|---:|---:|---:|---:|---:|
| 모든 variant (T0-T9) | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |

**모든 5 OOD class 100% mis-fire** (Normal 으로 예측 안 함). `--no-normal` baseline 의 expected behavior — 모델이 4 train class 만 안다 → OOD chip 도 4 중 하나로 강제 분류.
이는 OOD 신호가 4 train defect 중 하나와 시각적으로 닮았기 때문. iter 10 의 paper finding 그대로: **Normal training 누락 = FAR 100%, ensemble (with vs without Normal) 만이 fix.**

### Iter 12 final return

★ **winner: T7 (LS=0.20 + CutMix-rect 0.5)**
- CF1 = **0.8490**, chip_FAR = 0.9600, bit_FAR = 0.2490
- 3plus% = 1.98% (decision_type 새 'combo' counter — 4 3-combo class 매칭 시 single decision)
- v19y T5 (old 11-class master) baseline 0.8162 대비 **Δ CF1 = +0.0328**
- 단 chip_FAR 비교는 master schema 다름 (v19y: 11 class, no OOD; v19z++: 21 class, +5 OOD) — 직접 비교 불가
- bb+fk+sc 3-combo 0.831 (best 단일), 그러나 bb+sc+sr 는 T5 가 0.588 로 더 나음

### 다음 step 제안 (pending 사용자 승인)

1. **Logit-avg ensemble** T7 + T5 + T9 — 각자 최강 3-combo cell 다름 → diversity 확보. iter 10 의 0.91 → 0.995 패턴 재현 시도.
2. **Fork F1 0.52 lock** — 모든 8 variant 의 weakest. Fork-specific augment 또는 over-firing 억제 inference variant (I7 joint coord descent, I9 per-class T) 시도.
3. **Normal training 추가 학습 (ensemble half)** — 현재 baseline 은 모두 `--no-normal`. 별도 `--with-normal` half 만들고 logit-avg → chip_FAR 96% → expected ~3-5% with Normal training.
4. **chip_FAR split metric** — bit_metrics.py 에 `chip_FAR_normal_only` (Normal 200 분모) 와 `chip_FAR_ood_only` (OOD 1000 분모) 두 column 추가. 현재 합쳐서 보면 winner 식별 어려움.

산출 파일:
- `outputs/T*_v19zpp_seed42_*` × 8 (각 best_model.pth + eval_I3/{stage1_*, bit_metrics.json})
- `outputs/_iter12_v19zpp_logs/` 24 log files (8×{train,eval,bitmetric})
- `outputs/_iter12_v19zpp_logs/_summary.py` (재실행 가능 aggregator)
- `chip_multilabel/_iter12_v19zpp_summary.md`

---

## TODO — Stage 2 끝난 후 (사용자 directive 260505)

1. **합성 난이도 조절** — 현재 combo 합성 (min-blend) 결과 라벨링이 너무 어려움 (eval 결과 macro_f1 ~0.85 근처 plateau). 보강:
   - source chip 중 **불량 정도 강한 것들끼리 우선** combo: 각 single class chip 의 defect_pixel_ratio 계산 → 상위 50% 만 source 로 사용. 약한 chip 끼리 blend 하면 noise 와 구분 안 됨.
   - 새 변종 generator option `--source-strength-pct 50` 추가.

2. **Pixel grade variation** — 현재 합성/소스 chip palette grade 가 거의 0 (white) + 1 (grey) 만. 더 강한 defect grade (2 green, 3 blue 등) 분포 시도:
   - `_make_normal_chip` / `_min_blend` 에서 일부 픽셀을 grade 2/3 으로 강제 elevated 변종.
   - 또는 source chip 자체에 grade 2 우세인 것만 골라 사용 (chip 마다 grade histogram 으로 필터).
   - 새 변종 generator option `--grade-mode {default, elevated_2, elevated_3}` 추가.

3. **재합성 후 동일 inference variants 다시 평가** — 새 eval set 으로 stage1 + stage2 둘 다 재실행. iter 4 ~ 로 notes 추가.

이건 Stage 2 (현재 T5 BCE 학습 중) 완료 후에 실행 — GPU 1잡 룰 + 사용자 명시 "학습 다 끝나고".


---

## Cycle B+1 (260507): fork+sr fix — pos_weight axis only

### 배경
Cycle B winner T7N alone CF1 0.9406, fork F1 0.87, **fork+sr 2-combo recall 0.625** (★ weak).
Hypothesis: fork legs visually weak vs scratch_rot dense lines after min-blend → model ignores fork.

### Experiments dispatched
- **B1**: T7 + BCE pos_weight={fork: 2.0} (4 axis only fork upweighted)
  - code: `losses.py::BCEMultiHot` + `--pos-weight` CLI flag (T5/T7 only)
  - tag `T7N_pwfk20_seed42`, 8ep, batch=8 accum=4, lr=1e-4, ls=0.20, cutmix-p=0.25
- **B4 SKIPPED** — paired training requires multi-hot folder support in `collect_samples` +
  dataset `__getitem__` + training loop tgt-build branch (cross-cutting). User directive
  ("SKIP B4 if code complex") honored. `chip_multilabel/gen_pair_chips.py` written for future use.

### Results — fork+sr recall comparison (I3, n=200/class, 3080 chips)

| variant                         | CF1    | F1_fork | **fk+sr recall** | fk+sr+CrossScratch (OOD) | ni_FAR |
|---|---:|---:|---:|---:|---:|
| Cycle B baseline (T7N reported) | 0.9406 | 0.87    | **0.625**        | n/a                      | 0.00%  |
| **B1 T7N + pos_weight fk=2.0**  | 0.8797 | 0.7734  | **0.6375**       | 0.4125                   | 0.00%  |

### per-class F1 (B1)
- bank_boundary: F1=0.9414 (P=0.9391 R=0.9437)
- fork:          F1=0.7734 (P=0.9665 R=0.6446)  ← high precision low recall
- scratch:       F1=0.8371 (P=0.9659 R=0.7385)
- scratch_rot:   F1=0.9667 (P=0.9978 R=0.9375)

### per-combo 2-bit recall (B1)
- bank_boundary+fork:        0.6250
- bank_boundary+scratch:     0.0563  ← weak
- bank_boundary+scratch_rot: 0.4250
- fork+scratch:              0.7375
- **fork+scratch_rot:        0.6375**  ← +0.0125 marginal vs Cycle B
- scratch+scratch_rot:       0.9812

### 분석
- pos_weight fk=2.0 가 fork **precision 0.9665** 만 끌어올리고 **recall 0.6446 그대로** —
  loss 가 fork 양성을 더 강조했지만, threshold 검색 후 conservative 한 fork prob 분포 때문에
  recall 안 따라옴. fork+sr 2-combo 도 +0.0125 negligible.
- 가설 변경: fork+sr 약점은 **threshold/loss tuning 으로 못 풀고 데이터 (paired 합성) 필요**.
  → B4 (paired supervised) 가 fundamental fix 일 가능성. 다음 cycle 에서 코드 patch
  full implementation 하고 dispatch.
- 흥미로운 sub-finding: bb+sc 2-combo 0.0563 만 극도로 약함 (다른 combo 0.42+) — bb+sc
  visual pair 가 가장 어려운 듯. 별도 조사 trigger.

### 산출
- `outputs/logs_chip_multilabel/T7_T7N_pwfk20_seed42_260507_073921/best_model.pth` (fk@ep1)
- `outputs/.../eval_I3/stage1_260507_074611/{eval_summary.json, report.md, bit_metrics_split.json}`
- `chip_multilabel/gen_pair_chips.py` (B4 datagen ready, training-side code TBD)
- code patches: `losses.py::BCEMultiHot.pos_weight`, `_train_chip_variant.py::--pos-weight`

### 결론 / 후속 axis
- pos_weight axis 결과 **net negative for fork+sr** (CF1 0.9406 → 0.8797, fork F1 0.87 → 0.77).
  paper ablation 에 negative result 로 기록 가치 있음.
- ★ **다음 cycle: B4 paired supervised** — `collect_samples_with_multihot` 새 helper 만들어
  `(p, multihot_tensor or -1)` return → dataset 바꾸고 training tgt build branch 패치.
  Effort ~1h, expected fork+sr recall lift > 0.85.

## CutMix area-ratio ablation (260507) — 6 variants × T7N base

base: T7 + BCE_LS=0.20 + Normal training, 8ep batch=8 accum=4 lr-head=1e-4 seed=42, Invalid eval-set 50 chip 새 텍스트 재합성됨. data: classification_chips master (200 defect+200 Normal), eval: chip_multilabel master (12 class, n_per_class=200).

코드 patch (`_train_chip_variant.py`):
- `--cutmix-mode` choices 에 `grid` 추가 (8×8 binary mask, OR rule for BCE).
- `--cutmix-grid-prob` (default 0.5) — grid 의 per-cell flip prob (sweep axis).

| variant | mode | area | CF1 | F1_bb | F1_fork | F1_sc | F1_sr | fork+sr R | bit_FAR | ni_FAR | ood_FAR |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| T7N **no_cutmix** ★ true baseline | none | 0% | **0.9162** | 0.936 | 0.832 | 0.936 | 0.961 | 0.675 | 5.48% | 20.00% | 14.69% |
| T7N random_rect | single rect | ~50% | **0.9188** | 0.972 | 0.844 | 0.866 | 0.994 | 0.719 | 1.40% | 20.00% | 0.94% |
| T7N scattered (10×~70px, r=0.62) | scattered | 62% | 0.8423 | 0.911 | 0.691 | 0.775 | 0.992 | **0.869** | 6.88% | 20.00% | 23.44% |
| T7N grid 8×8 p=0.5 | grid binary | 50% | 0.8967 | 0.906 | 0.763 | 0.930 | 0.987 | 0.794 | 1.99% | 20.00% | 0.31% |
| T7N grid 8×8 p=0.25 | grid binary | 25% | 0.8849 | 0.935 | 0.745 | 0.872 | 0.987 | 0.812 | 2.59% | 20.00% | 3.12% |
| T7N grid 8×8 p=0.125 | grid binary | 12.5% | 0.8596 | 0.935 | 0.778 | 0.809 | 0.917 | 0.794 | 4.76% | 20.00% | 12.03% |
| (ref) T7N v19zpp Cycle B (random rect r=0.5, prev master) | single rect | ~50% | 0.9406 | 0.980 | 0.870 | 0.92 | 1.00 | 0.625 | — | 0.00% | 1.41% |

★ winner (CF1 + ni_FAR ≤ 5% 만족):
- 단순 CF1 max: **T7N random_rect = 0.9188** (아주 근소; no_cutmix 0.9162 와 동률 수준).
- 단순 baseline (no_cutmix) 가 CF1 0.9162 로 grid/scattered 보다 높음 → **CutMix area 단순 증가 = CF1 개선 X**. random_rect 0.5 만 미세 +0.0026 lift. 이는 v20 master + ni 새 텍스트 환경에서 CutMix 의 marginal value 가 작아졌음을 의미.
- **fork+sr 2-combo recall 만 보면 winner = scattered 0.869** (baseline 0.675 → +0.194), 다음 grid25 0.812. 그러나 scattered 는 OOD over-fire (ood_FAR 23.4%) 와 fork F1 -0.14 trade-off.
- ★ **ni_FAR 20% 모든 variant 동일** — 이는 새로 합성된 Invalid 가운데 큰 텍스트가 모든 모델에서 동일 패턴으로 fail (40/200 = 20%). CutMix mode 와 무관, **Invalid 텍스트 → 모델이 defect 신호로 해석**. 이건 next-step 에서 Invalid heuristic 보강 필요 (학습 데이터에 Invalid 추가 또는 추론 규칙).

### 핵심 finding (260507)
1. **area sweep 곡선 (no_cutmix → 50% → 62%) 단조 X** — random_rect (CF1 0.9188) > grid50 (0.8967) > grid25 (0.8849) > grid12 (0.8596) > scattered (0.8423). single rect mode 가 grid binary 보다 일관 우위 (50% area 비교: 0.9188 vs 0.8967).
2. **scattered는 fork+sr 2-combo recall booster** (0.675 → 0.869) 지만 fork F1 collapse (0.83 → 0.69) 와 OOD over-fire (14.7% → 23.4%) 동반 — paper 에 negative trade-off 사례.
3. **grid binary OR-label 모드 (이번 신규 추가)** 는 grid_prob 0.5 이 가장 좋고 (0.8967), 0.25/0.125 로 줄이면 scattered 와 비슷한 패턴 성능 저하. binary OR-label 의 본질적 한계 — soft area-prop label 안 씀.
4. **ni_FAR 20% 고정** — 모델 변경으로 안 풀림. Invalid eval set 새 텍스트 vs 학습 시 Invalid 부재 mismatch (학습은 4 defect + Normal only).
5. **ood_FAR random_rect 0.94% 가 best** — 단순 single rect 로도 OOD 분리력 유지. scattered 는 OOD 도 over-fire.

### 다음 axis 제안
- ★ Invalid 학습 추가 (`--include-invalid` 같은 flag 신설; y=-2 sentinel + zero-vector target). ni_FAR 20% 직접 타격.
- B4 paired supervised (이전 propose) 는 fork+sr recall 만 노린 변경 — scattered 가 이미 0.869 달성하므로 priority 낮춤.
- random_rect rect=0.3 / 0.7 sweep — 50% rect 가 best, 위/아래 곡률 확인 가치.

### 산출
- 6 train run dir: `outputs/T7_T7N_{no_cutmix,random_rect,scattered,grid50,grid25,grid12}_seed42_260507_*/`
- 6 eval (I3) + bit_metrics_split.json each.
- 코드 patch: `_train_chip_variant.py` (`grid` mode + `--cutmix-grid-prob`).

## iter 19 complement CutMix sweep (260508)

**Spec**: T7 + LS 0.20 + epochs=8 seed=1 cutmix-p=0.25 cutmix-mode=complement, 12 trains 매트릭스 (4 group × label_scale + 2 baselines pair=none). eval = run_stage1 I3,I6,I7,I10 n-per-class=50 strength-min=0.0.

### iter19A 회수 결과 (학습 crash 후 best_model 만 남았던 cell)
- run dir: `outputs/iter19A_complement_g2_l0.5_pmasked/T7_T7_iter19A_complement_g2_l0.5_masked_seed1_260508_140548/`
- 학습은 epoch 1 직후 best 저장만 하고 crash (history.json/train_summary.json 모두 부재) — eval 만 별도로 회수.
- **best cell = T0__I3, macro_f1 = 0.8078** (top1_11=0.5813, T=1.000, ECE=0.0312)
- per-class F1: bank_boundary 0.9438 / fork 0.7993 / scratch 0.7064 / scratch_rot 0.7817
- I10 0.8030 / I7 0.8003 / I6 0.7806 (I3 < I10 subset_acc 0.6562 > I3 0.5813)
- 해석: epoch 1 best 라 sub-baseline. 실제 sweep evidence 로 부적합 — B-L 결과 비교 시 A 는 저평가 plot 이므로 footnote 처리.
- eval report: `outputs/iter19A_.../eval_seed1/stage1_260508_141521/report.md`

### sweep 진행 (B-L 11 trains, background)
- launcher: `_run_iter19_complement_resume.sh` (set -e sequential, A line 제외)
- log: `outputs/_iter19_complement_resume.log`
- 백그라운드 task id: b3uinaqq5 (start 260508 14:16, 예상 ETA ~77분)
- GPU 14:16:17 측정: 13.8 GB used / 99% util — train iter19-B 가동 중 confirm.

### 12 train spec 매트릭스

| TAG | group | label_scale | pair | batch | 비고 |
|---|---|---|---|---|---|
| A | 2 | 0.5  | masked | 4 | crashed → eval 만 회수 (macro 0.8078) |
| B | 2 | 0.75 | masked | 4 | running (14:16) |
| C | 2 | 1.0  | masked | 4 | queued |
| D | 3 | 0.33 | masked | 4 | queued |
| E | 3 | 0.67 | masked | 4 | queued |
| F | 3 | 1.0  | masked | 4 | queued |
| G | 4 | 0.25 | masked | 2 | queued (2N=8 chips) |
| H | 4 | 0.5  | masked | 2 | queued |
| I | 4 | 0.75 | masked | 2 | queued |
| J | 4 | 1.0  | masked | 2 | queued |
| K | 2 | 1.0  | none   | 4 | baseline (no pair) |
| L | 4 | 1.0  | none   | 4 | baseline (no pair) |

## iter 22 19C tune (260509)

19C (FCM-PM, paper winner — T7 LS=0.20 + cutmix-p=0.25 complement g=2 LS=1.0 masked corner, batch 4 accum 4, 8 ep, seed 1) 의 (a) 3-seed verification + (b) same-mechanism hparam fine-tune sweep. 10 trains, 모두 dual-eval (v14class + v15direct, --variants I3,I6,I7,I10 --n-per-class 50 --seed 42).

- launcher: `_run_iter22_19C_tune.sh` (set -e sequential, BASE19C 공통 + per-tag override)
- log: `outputs/_iter22_19C_tune.log`
- 백그라운드 task id: b7zv9mb8l (start 260509 12:53, 학습 ~7 min × 10 = ~70 min ETA)
- GPU 12:53:27 측정: 2.3 GB used / 11% util — A_seed7 로딩 중 confirm.
- 19C base reference: v14 bit_F1 0.9913 / v15 0.9691, ni_FAR 0%/3.75%, F1bb 1.0.

### 10 train spec 매트릭스 (BASE19C: T7 --ls 0.20 --cutmix-p 0.25 --cutmix-mode complement --cutmix-n-groups 2 --cutmix-complete-label-scale 1.0 --cutmix-pair masked --cutmix-pair-fill corner)

| TAG | seed | override | 의도 |
|---|---|---|---|
| A_seed7        | 7  | (none)                   | 3-seed verify (vs seed 1) |
| B_seed42       | 42 | (none)                   | 3-seed verify (vs seed 1) |
| C_LS010        | 1  | --ls 0.10                | LS sweep down |
| D_LS030        | 1  | --ls 0.30                | LS sweep up |
| E_cutmix015    | 1  | --cutmix-p 0.15          | CutMix freq down |
| F_cutmix040    | 1  | --cutmix-p 0.40          | CutMix freq up |
| G_droppath005  | 1  | --drop-path-rate 0.05    | stochastic depth |
| H_ema095       | 1  | --ema-decay 0.95         | EMA target |
| I_warmup2      | 1  | --warmup-epochs 2        | LR warmup |
| J_lrhead5e5    | 1  | --lr-head 5e-5           | head LR ↓ |

### 진행 status

- 12:53 dispatched (background id b7zv9mb8l)
- 13:42 KILLED externally — completed: A, B, C, D, E (train+dual-eval). F: train OK + eval_v14class started but interrupted mid-eval. G,H,I,J: not started.
- 결과 path: `outputs/iter22<TAG>/T7_iter22<TAG>_seed*/best_model.pth + eval_v14class/ + eval_v15direct/`

### 결과 (best cell macro_f1 across I3/I6/I7/I10)

| TAG | seed | override | v14 best cell | v14 macro_f1 | v15 best cell | v15 macro_f1 | status |
|---|---|---|---|---:|---|---:|---|
| A_seed7        | 7  | —                       | I10 | **0.8285** | I10 | **0.8364** | done |
| B_seed42       | 42 | —                       | I10 | **0.8506** | I6  | **0.8250** | done |
| C_LS010        | 1  | --ls 0.10               | I6  | **0.8536** | I6  | **0.7540** | done |
| D_LS030        | 1  | --ls 0.30               | I10 | **0.8457** | I10 | **0.8160** | done |
| E_cutmix015    | 1  | --cutmix-p 0.15         | I10 | **0.7729** | I10 | **0.6933** | done |
| F_cutmix040    | 1  | --cutmix-p 0.40         | —   | —          | —   | —          | train OK, eval interrupted |
| G_droppath005  | 1  | --drop-path-rate 0.05   | —   | —          | —   | —          | not started |
| H_ema095       | 1  | --ema-decay 0.95        | —   | —          | —   | —          | not started |
| I_warmup2      | 1  | --warmup-epochs 2       | —   | —          | —   | —          | not started |
| J_lrhead5e5    | 1  | --lr-head 5e-5          | —   | —          | —   | —          | not started |

reference (19C seed 1, prior known): v14 0.9913 / v15 0.9691.

### Observations (5 completed cells)

- **모든 완료 cell (A~E) 가 19C seed 1 reference 0.9913 보다 한참 낮다 (0.77~0.85)** — biggest unexpected gap. 3-seed verify A (0.83) / B (0.85) 가 seed 1 (0.99) 와 0.14~0.16 격차 — 19C seed-1 결과가 lucky outlier 거나 iter19 이후 코드/데이터 drift 의심. F~J 재실행 + 19C 재현 verify 필요.
- LS↓ (C 0.10) v14 0.8536 sweep 내 best; LS↑ (D 0.30) v14 0.8457; LS 는 이 범위에서 dominant lever 아님.
- CutMix↓ (E 0.15) v14 0.7729 vs A 0.8285 = -0.056 — CutMix freq down 유의미하게 손해.
- v14 > v15 일관 (+0.02~+0.10) — 도메인 gap 정상 패턴.

### 다음 액션 후보 (사용자 결정 대기)

1. F~J 재개 (G,H,I,J 신규 + F 는 모델 보존되어 있으니 eval 만 재실행 가능)
2. 19C seed 1 reference 재현 verify (코드/데이터 drift 점검)
3. 두 가지 병행

---

## Iter 30 — FCM-PM n_groups + LS sweep + 26B 3-seed (260509)

T7 + cutmix-p 0.25 mode=complement pair=masked fill=corner, --ls 0.20 --epochs 8 batch 4 accum 4 (D/E/F batch 4 with g≤3, B/C batch 2 with g=5/6). Dual-eval v14class + v15direct, runtime sample n=50 strength 0.0 seed 42.

**Reference baseline iter26B (g=3 LS=0.50 seed=1) on same dual-eval**: v14 **0.8524 (I6)** / v15 **0.8278 (I10)**.

| cell             | n_groups | LS   | seed | v14 cell | v14 macro_f1 | v15 cell | v15 macro_f1 | status |
|------------------|----------|------|------|----------|--------------|----------|--------------|--------|
| A_g5_LS020       | 5        | 0.20 | 1    | I10      | 0.7919       | I10      | 0.6981       | done   |
| B_g5_LS050       | 5        | 0.50 | 1    | I10      | 0.7752       | I6       | 0.7050       | done   |
| C_g6_LS030       | 6        | 0.30 | 1    | I7       | 0.7625       | I10      | 0.7446       | done   |
| D_g2_LS050       | 2        | 0.50 | 1    | I6       | **0.8617**   | I10      | **0.7805**   | done   |
| E_g3_LS050_seed7 | 3        | 0.50 | 7    | —        | —            | —        | —            | running |
| F_g3_LS050_seed42| 3        | 0.50 | 42   | —        | —            | —        | —            | queued |

### Observations (4 done)

- **No iter30 cell beats iter26B baseline (0.8524 / 0.8278) on either eval.** D (g=2 LS=0.50) closest: −0.011 v14 (I6 vs I6) but +0.034 still under v15 (0.7805 vs 0.8278 = −0.047).
- FCM-PM `n_groups` higher (5/6) actively hurts vs g=2/3 — A/B/C all worse than D on v14 (≤0.79 vs 0.86) and v15.
- **g=2 (D) > g=3 (26B) on v14** (0.8617 vs 0.8524, +0.009) but loses on v15 (0.7805 vs 0.8278). Mixed signal — E + F (g=3 multi-seed) needed to confirm.
- Cell winner shifts per protocol: I10 dominant on v14, I6/I10 split on v15. Decision_tree variant choice is eval-set-sensitive.
- LS sweep (0.20 / 0.30 / 0.50) inside iter30 not separated from g sweep — confound. Future: hold g=2 fixed and sweep LS only.

### Next dispatched

- **iter31** (`_run_iter31_26B_tune.sh`, b8epseiwz polling) — 7-train regularization tune around 26B base (EMA / drop_path / warmup / longer epochs / lr-head / combos). Will fire once iter30-resume queue clears.
- **26B 3-seed mini-ensemble** (post-E/F) — prob-avg of iter26B seed=1 + iter30E seed=7 + iter30F seed=42 on T0__I10. If +Δ vs single seed, fold into 17-bag.
- **iter32 KD** — placeholder script `_run_iter32_KD_skel.sh`. Awaits analyst spec at `docs/chip-multilabel/iters/iter_32_KD_spec.md`. Teacher = iter26B best_model.pth.


## iter 32 KD distillation (260509)

**Goal**: distill 14-bag ensemble (~0.99 v15) into single student via Hinton-2015 KD — close 0.97 → 0.99 single-model gap.

**Teacher**: 14-member ensemble = iter21 (E,F,H) + iter22 (A,B,D,G) + iter24 (LS030 seed7/42) + iter26 (B,D,F,G,H). Avg sigmoid probs over `classification_chips/<class>/*.png` (5 dirs: 4 single + invalid_main; Normal absent).

**Student recipe**: 26B base = T7 LS=0.20 epochs=8 batch=4 accum=4 seed=1 cutmix-p=0.25 mode=complement n-groups=3 label-scale=0.50 pair=masked fill=corner. **KD knobs (iter32A)**: α=0.5, T=4.0, skip-on-cutmix.

**Files**:
- `chip_multilabel/_kd_make_teacher_probs.py` (NEW, ~155 lines) — pre-compute parquet `outputs/_teacher_probs_14bag.parquet` (chip_path → 4 sigmoid prob columns).
- `chip_multilabel/_train_chip_variant.py` — 4 CLI flags + 3 patches:
  - 3a (after argparse): load parquet → `teacher_prob_map: dict[str, np.ndarray(4,)]`.
  - 3b (`ChipFolderDataset`): accept `teacher_prob_map`, return 4-tuple `(x, y, mh, teacher_prob)` (zero-vec when missing). `evaluate()` drops the 4th element.
  - 3c (train loop): KD loss = T²·BCE(sigmoid(z_S/T), sigmoid(z_T/T)) where z_T = log(p/(1−p)) with eps=1e-6 clamp. `loss = α·loss_pre_kd + (1−α)·l_kd`. Skip when: (i) batch shape mismatch (e.g., complement rebuilt x), (ii) cutmix-applied + `--kd-skip-on-cutmix`, (iii) flag empty.
- `_run_iter32_KD.sh` (NEW, +x) — chains after iter31 (`grep [iter31] DONE`), runs Step 1 teacher parquet (skip-if-exists) → Step 2 student train → Step 3 dual eval v14class + v15direct (I3/I6/I7/I10).

**Hard-rule compliance**: TTA disabled, no folder deletion, classification_chips/ only (no multi_combo_root), training-only KD (val drops teacher_prob 4th tuple slot).

**Status (260509)**: code patched, dispatch script staged +x via `git update-index`, bash bg launched (chains on iter31 finish).


## iter 35 area-proportional FCM-PM (260509)

**Goal**: paper §6.13 missing axis. Current complement mode uses **symmetric** label rule (`mix_t[A] = mix_t[B] = label_scale`) which over-credits A relative to its actual cell area. iter35 tests **area-proportional** label: A occupies group_i (1/n cells), B occupies the other (n−1)/n cells → labels should reflect that ratio.

**Patch** (`chip_multilabel/_train_chip_variant.py`):
- New CLI flag `--cutmix-label-area-prop` (action=store_true), default off (= symmetric, backward compat).
- Label rule (when flag set, complement mode only): `a_frac = 1/n_groups`, `b_frac = (n_groups−1)/n_groups`; `mix_t[a_cls] = a_frac * label_scale`, `mix_t[b_cls] = b_frac * label_scale`.
- mask-chip label unchanged (mask is A only — area-prop irrelevant).

**8-cell sweep table** (T7 LS=0.20 epochs=8 batch=2 accum=8 seed=1 cutmix-p=0.25 complement masked corner; `--cutmix-label-area-prop` always on):

| Cell | n_groups | scale | A label | B label | hypothesis |
|---|---|---|---|---|---|
| A | 3 | 1.0 | 0.333 | 0.667 | pure area-prop, sum=1.0 |
| B | 4 | 1.0 | 0.250 | 0.750 | pure, finer split |
| C | 3 | 1.5 | 0.500 | 1.000 | B saturated (cap) |
| D | 4 | 1.33 | 0.333 | 1.000 | B saturated, finer |
| E | 3 | 0.5 | 0.167 | 0.333 | conservative |
| F | 4 | 0.5 | 0.125 | 0.375 | conservative + finer |
| G | 2 | 1.0 | 0.500 | 0.500 | sanity (= symmetric LS=0.5) |
| H | 3 | 0.3 | 0.100 | 0.200 | extra-conservative |

**Hard-rule compliance**: TTA disabled, no folder deletion, classification_chips/ only, batch=2 accum=8 (effective 16), v15-only eval (I3/I6/I7/I10 @ n=50).

**Status (260509)**: trainer patched (≈12 lines including CLI), import OK, argparse usage shows flag. `_run_iter35_areaprop.sh` written +x mode 100755. **Chained on `grep [iter33] DONE`** (iter33 currently waiting on iter32 → iter32 not yet done). bash bg launched.


## iter 122 — T6 (BCE warmup 3 → ASL γ_neg=4 clip=0.05) on iter116J val_margin recipe (260513)

**Hypothesis (analyst opus)**: ASL γ_neg=4 + clip=0.05 가 easy-negative loss floor 를 깎아 partner bit (target=1, prob 0.43-0.53 weak) 의 gradient 상대적 증폭 → bb+sr / fork+sr partner recall ↑. T7 → T6 atomic 1-line loss swap, iter116J recipe (cutmix complement masked corner p=0.25 g=3 cls=0.5, val_margin, save_every_epoch) 동결.

**Bugfix**: `_train_chip_variant.py` 에 `import os` 누락 → 이전 iter122 dispatch 4 cell 모두 NameError 로 즉시 종료. fix 후 재실행.

**Run**: `outputs/iter122_T6_asl_gn4/T6_iter122_T6_asl_gn4_260513_085714/`. epochs=10, BCE→ASL switchover ep6, val_criterion=margin_max, save_every_epoch 켜짐. 학습 518.8s. best (val_margin pick) ep3 v_margin=0.9707 (still BCE phase).

**bit_F1 / Total FAR / partner-recall (T0__I10 cell, n=200/class, seed=42, eval=chip_multilabel_v15direct)**:

| run | ep | bit_F1 | Total FAR | NI FAR | OOD FAR | bb+sr_sr | fork+sr_sr | fork+sr_fork |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| iter116J T7 BCE+LS0.3 (best, val_f1 sel) | 1 | 0.7911 | 0.0% | 0.0% | 0.0% | 0.831 | **1.000** | 0.919 |
| iter122 T6 ep3 (val_margin pick, BCE) | 3 | 0.8122 | **84.2%** | 76.5% | 86.6% | 0.869 | 0.988 | 0.912 |
| iter122 T6 ep6 (first ASL) | 6 | 0.8298 | 74.2% | 79.0% | 72.7% | 0.900 | 0.787 | 0.775 |
| iter122 T6 ep10 (final ASL) | 10 | 0.8297 | **9.4%** | 29.0% | 3.3% | **0.981** | 0.750 | 0.819 |

**판정 — REGRESSION (partial)**:
- bit_F1: 0.7911 → 0.8297 (ep10) (+0.039) — surface 개선 보이나
- **Total FAR**: 0% → 9.4% (ep10) — Normal/Invalid/OOD 가 train-class 로 새는 비율 큰 증가
- bb+sr partner-recall (sr bit): 0.831 → 0.981 (+0.150) — 가설 부분 입증
- fork+sr partner-recall (sr bit): 1.000 → 0.750 (-0.250) — partner recall 의 **trade-off**: sr-on-bb 살리면 sr-on-fork 죽음
- val_margin selection 이 ep3 (BCE warmup phase) 를 골랐는데 그 시점은 BCE-only 학습이라 가설 검증 불가능. ASL 단계 ep6/10 가 진짜 T6 효과인데 selection 으로 못 뽑힘.

**Winner**: iter116J T7 (clean 0% FAR + 가장 안정한 fork+sr partner) 유지. iter122 T6 은 paper-grade ablation evidence 로만 가치.

**왜 실패**:
1. ASL clip=0.05 이 over-aggressive — fork/scratch threshold 0.02 까지 떨어져 (iter116J 0.180/0.140 대비) negative 도 train-class 로 흡수 → FAR 폭증
2. val_margin criterion 이 BCE warmup phase 의 saturated margin (>0.97) 을 ASL phase margin (~0.96) 보다 선호. ASL phase 가 partner-bit gradient 증폭 시작하는 구간인데 selection 이 못 뽑음
3. partner-recall trade-off: ASL 이 bb+sr 살리면서 fork+sr 죽임. γ_neg=4 가 fork-class 의 negative 도 강하게 누름 → fork+sr 의 sr 신호 약해질 때 fork 까지 같이 죽음

**후속 1 atomic 제안 (실행하지 않음, 사용자 승인 대기)**:
- (a) **clip=0.1** 또는 clip 제거 (γ_neg=4 만 유지) — FAR over-shoot fix 1 후보
- (b) **val_criterion=f1** + ASL phase 만 selection — phase-aware best 선택
- (c) **γ_neg=2** (mild) — partner recall trade-off mitigate
- (d) **bce-asl switch ep=8** (현재 ep6) — BCE 더 길게 + ASL 마지막 2 ep 만

raw artifacts:
- `outputs/iter122_T6_asl_gn4/T6_iter122_T6_asl_gn4_260513_085714/best_model.pth` (ep3 val_margin pick)
- `.../epoch_{01..10}_model.pth` (save_every_epoch)
- `.../eval_v15direct_n200/stage1_260513_090615/` (best ep3 eval)
- `.../eval_ep06/stage1_260513_090739/` (first ASL eval)
- `.../eval_ep10/stage1_260513_090849/` (final ASL eval)
- `outputs/_iter122_T6_train.log`, `_iter122_T6_eval_best.log`

---

## iter 123 — T6 (BCE warmup 3 → ASL γ_neg=4 **clip=0.10**) — iter122 clip 0.05→0.10 atomic swap (260513)

**Hypothesis (이전 iter122 root-cause 후속)**: iter122 ep10 의 9.4% Total FAR + scratch threshold 0.02 붕괴는 ASL clip=0.05 가 over-aggressive (probability-shift 너무 큼) 한 탓. clip=0.10 (덜 aggressive) → defect threshold 보존 + FAR 회복 가설.

**Run**: `outputs/iter123_T6_asl_clip01/T6_iter123_T6_asl_clip01_260513_091520/`. epochs=10, BCE→ASL switchover ep4, val_criterion=margin_max, save_every_epoch. 학습 750.9s (iter122 518.8s 대비 +45% — clip 0.10 이 backward 더 무거움). best (val_margin) ep3 v_margin=0.9707 (BCE phase, iter122 와 완전 동일 — clip 무관). final_epoch ep10 val_acc 0.9816.

**bit_F1 / Total FAR / partner-recall (T0__I10 cell, n=200/class, seed=42, eval=chip_multilabel_v15direct, same eval set 재합성 X)**:

| run | ep | bit_F1 | Total FAR | NI FAR | OOD FAR | bb+sr→sr | fork+sr→sr | fork+sr→fork |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| iter116J T7 BCE+LS0.3 (val_f1 sel) | 1 | 0.7911 | 0.0% | 0.0% | 0.0% | 0.831 | **1.000** | 0.919 |
| iter122 T6 ep3 (val_margin BCE) | 3 | 0.8122 | 84.2% | 76.5% | 86.6% | 0.869 | 0.988 | 0.912 |
| iter122 T6 ep10 (final ASL clip=0.05) | 10 | 0.8297 | **9.4%** | 29.0% | 3.3% | 0.981 | 0.750 | 0.819 |
| **iter123 T6 ep3 (val_margin BCE)** | 3 | 0.7132 | (BCE — 동일) | — | — | 0.869 | 0.988 | 0.912 |
| **iter123 T6 ep10 (final ASL clip=0.10)** | 10 | 0.8297 | **5.0%** | 16.0% | 1.6% | **0.988** | **0.838** | **0.838** |

★ **iter123 ep10 vs iter122 ep10 (clip 0.05 → 0.10, single-atomic)**:
- bit_F1: 0.8297 → 0.8297 (Δ=0 — macro F1 자체는 clip 변경 영향 없음)
- Total FAR: 9.4% → **5.0%** (-4.4pp)
- NI FAR: 29.0% → **16.0%** (-13pp)
- OOD FAR: 3.3% → **1.6%** (-1.7pp)
- bb+sr partner recall: 0.981 → 0.988 (+0.007)
- fork+sr partner recall: 0.750 → **0.838** (+0.088 회복 — iter122 의 trade-off mitigate)
- fork+sr fork recall: 0.819 → 0.838 (+0.019)

★ **best (ep3 val_margin pick) → iter122 와 완전 동일**: val_margin criterion 이 BCE warmup ep3 를 selection 한 결과는 clip 무관 (ASL 진입 ep4). 이전 iter122 의 셋업 약점 (selection 이 ASL phase 못 잡음) 재현.

**판정 — DRAW (partial improvement, winner 미달)**:
- Criterion: bit_F1 ≥ 0.99 AND Total FAR ≤ 0.5% → **미달** (bit_F1 0.83 < 0.99, FAR 5.0% > 0.5%)
- vs iter122: partial improvement (FAR -4.4pp, partner recall trade-off 회복)
- vs iter116J baseline: 여전히 regression (bit_F1 0.79→0.83 surface 개선이나 FAR 0%→5% trade-off)
- clip=0.10 이 over-aggressive 가설 입증 — clip=0.05 의 fork+sr → sr 깎임 (0.750) 이 0.838 로 부분 회복
- 그러나 FAR 5% 는 production 으로 가기엔 여전히 너무 큼 (NI 16% = Normal/Invalid 중 1/6 가 defect 잘못 분류)

**왜 winner 못 됨**:
1. ASL γ_neg=4 의 본질적 trade-off: easy-negative loss 깎으면 fork/scratch threshold 자동 조정 (auto-tuned p* 0.78/0.06) → 0.06 인 scratch threshold 가 OOD chip 의 scratch-like noise 잡음
2. clip 0.10 → 0.05 로 더 풀어도 ASL 의 probability-shift 본질이 NI/OOD chip 으로 새는 걸 막지 못함
3. val_margin selection (ep3 BCE pick) 이 ASL 효과 비교를 또 방해 — 이건 iter122 와 같은 약점

**후속 1-atomic 제안 (실행 X, 사용자 결정 대기)**:
- (a) **γ_neg=2** (clip=0.10 고정) — γ 자체를 milder. iter116J 0% FAR 회복 + bit_F1 개선 동시 가능성. ASL 의 강도 다이얼링.
- (b) **bce-asl switch ep=8** (clip=0.10 고정) — BCE 더 길게 + ASL 마지막 2 ep 만 → BCE 의 conservative threshold (iter116J 처럼 0.18 fork / 0.14 scratch) 보존하면서 ASL 의 partner-recall 효과만 살짝
- (c) **val_criterion=f1 + warm_min_ep=4** — ASL phase (ep4+) selection 가능. 그러면 ep10 이 best 로 뽑힐 가능성 (현재 ep3 BCE 가 selection 으로 가려져 ASL 효과 직접 측정 불가)
- (d) **ASL 폐기 + BCE+LS 0.30 (iter116J 변종) 만 다시** — clip=0.10 으로도 ASL 약점 (FAR cost) 못 푼 결론 그대로 받아 다른 axis 로 pivot

raw artifacts:
- `outputs/iter123_T6_asl_clip01/T6_iter123_T6_asl_clip01_260513_091520/best_model.pth` (ep3 val_margin pick)
- `.../epoch_{01..10}_model.pth` (save_every_epoch)
- `.../eval_v15direct_n200/stage1_260513_092827/` (best ep3 eval)
- `.../eval_ep10/stage1_260513_092951/` (final ASL clip=0.10 eval)
- `outputs/_iter123_T6_train.log`, `_iter123_T6_eval_best.log`, `_iter123_T6_eval_ep10.log`
- analysis scripts: `_iter123_metrics.py`, `_iter123_partner.py`

