# chipgrid 학습 결과 누적

`cnn_eval_chipgrid.py` 모든 학습 run 의 결과 표. 새 run 끝나면 `chipgrid-eval` agent 가 한 행 append.

> **상세 분석은 [RESULTS_DETAIL.md](RESULTS_DETAIL.md)** — hparams, 데이터 분포 (wafer class 별 train/val/test, obj_id chip-object 분포), BEST OVERALL (val/test acc/f1), best/total epoch, per-class FP/FN, BEST UPDATES progression 자동 추출.
> 갱신 명령: `python _chipgrid_summary.py -o docs/chipgrid/RESULTS_DETAIL.md`

## 전체 표 (시간순)

| variant | n/cls | epochs | seed | obj_norm | target_id | chip_noise | val_f1 | test_f1 | best_ep | run_dir | finished_at |
|---|---|---|---|---|---|---|---|---|---|---|---|
| V0 | 30 | 20 | 42 | — | — | 0 | 35.03% | 37.00% | 20 | v0_260503_130538_0.37_0.35 | 2026-05-03 13:07 |
| V0 | 100 | 30 | 42 | — | — | 0 | 43.59% | 43.85% | 13 | v0_n100_260503_130816_0.44_0.44 | 2026-05-03 13:14 |
| V1 | 100 | 30 | 42 | 5 | — | 0 | 95.05% | 97.26% | 10 | v1_n100_260503_131535_0.97_0.95 | 2026-05-03 13:21 |
| V2 | 100 | 30 | 42 | — | 3 (particle_blast) | 0 | 65.43% | 64.79% | 13 | v2_particle_n100_260503_132203_0.65_0.65 | 2026-05-03 13:27 |
| **V3** ★ | 100 | 30 | 42 | — | — | 0 | **96.89%** | **98.79%** | 6 | v3_onehot_n100_260503_132834_0.99_0.97 | 2026-05-03 13:34 |
| V3 | 100 | 30 | 42 | — | — | 0.05 | 96.67% | 99.10% | 5 | v3_noise05_260503_151438_0.99_0.97 | 2026-05-03 15:20 |
| V3 | 100 | 30 | 42 | — | — | 0.10 | 97.07% | 99.19% | 8 | v3_noise10_260503_152034_0.99_0.97 | 2026-05-03 15:26 |
| V3 | 100 | 30 | 42 | — | — | 0.20 | 95.95% | 96.36% | 18 | v3_noise20_260503_152725_0.96_0.96 | 2026-05-03 15:33 |
| **V3** ★ | **220** | 30 | 42 | — | — | 0 | **99.45%** | 98.66% | 5 | v3_full_260503_160436_0.99_0.99 | 2026-05-03 16:18 |
| V0 | 220 | 30 | 42 | — | — | 0 | 46.98% | 40.39% | 5 | v0_full_260503_163546_0.40_0.47 | 2026-05-03 16:49 |
| V1 | 220 | 30 | 42 | 5 | — | 0 | 98.05% | 97.35% | 11 | v1_full_260503_163549_0.97_0.98 | 2026-05-03 16:49 |
| V1 | 100 | 30 | 42 | 1 | — | 0 | 96.06% | 99.36% | — | v1_norm1_260503_165049_0.99_0.96 | 2026-05-03 16:55 |
| V1 | 100 | 30 | 42 | 10 | — | 0 | 95.70% | 97.37% | — | v1_norm10_260503_165054_0.97_0.96 | 2026-05-03 16:55 |
| V2 | 100 | 30 | 42 | — | 1 (bank_boundary) | 0 | 64.53% | 64.66% | — | v2_bank_boundary_260503_165821_0.65_0.65 | 2026-05-03 17:08 |
| V2 | 100 | 30 | 42 | — | 4 (scratch) | 0 | 55.27% | 53.91% | — | v2_scratch_260503_165826_0.54_0.55 | 2026-05-03 17:08 |
| V2 | 100 | 30 | 42 | — | 2 (invalid_main) | 0 | 51.14% | 54.05% | — | v2_invalid_main_260503_171834_0.54_0.51 | 2026-05-03 17:24 |
| V2 | 100 | 30 | 42 | — | 5 (scratch_21deg) | 0 | 65.87% | 65.53% | — | v2_scratch_21deg_260503_171840_0.66_0.66 | 2026-05-03 17:24 |
| V3 | 100 | 30 | **1** | — | — | 0 | 99.01% | 98.38% | — | v3_seed1_260503_172707_0.98_0.99 | 2026-05-03 17:34 |
| V3 | 100 | 30 | **7** | — | — | 0 | 98.21% | 98.68% | — | v3_seed7_260503_172711_0.99_0.98 | 2026-05-03 17:34 |
| V3 | 100 | 30 | **100** | — | — | 0 | 98.42% | 99.35% | — | v3_seed100_260503_173539_0.99_0.98 | 2026-05-03 17:42 |
| V3 | 100 | 30 | **234** | — | — | 0 | 99.36% | 98.26% | — | v3_seed234_260503_173544_0.98_0.99 | 2026-05-03 17:42 |
| V3 | 220 | 30 | 42 | — | — | 0.10 | 99.17% | 97.85% | — | v3_full_noise10_260503_174409_0.98_0.99 | 2026-05-03 17:59 |

## 변종 비교 (n=100/class, seed=42)

| 변종 | encoding | in_ch | val_f1 | test_f1 | 의미 |
|---|---|---|---|---|---|
| V0 | R only | 1 | 43.59% | 43.85% | chip object 정보 없으면 25 sub-class 가 같아 보임 → 1/5 = 20% within group + 9 special class = ~41% 천장 |
| V1 | argmax /5 | 2 | 95.05% | 97.26% | 정수 1채널 만으로 거의 saturate. compound BICUBIC 384 (97.84%) 와 동급 |
| V2 | binary particle_blast | 2 | 65.43% | 64.79% | 한 chip object 만 표시 → 그 class 만 풀고 나머진 V0 수준 (1/4 within group) |
| **V3** | one-hot 5ch | 6 | **96.89%** | **98.79%** | 정보 손실 0. compound 같은 데이터로 학습 시 동등 이상 예상 |

핵심:
- V0 → V1: +51%p (val) — obj_id 채널 신호 막대
- V1 → V3: +1.84%p (val) — one-hot 분리 가 정수 압축 보다 약간 우위
- V3 ≈ compound (97.84%) — **데이터 4× 적게 + 모델 76× 작게** 동급 도달

## V3 + chip CNN noise robustness curve

n=100/class, seed=42, --chip-noise --chip-noise-eval (학습/평가 양쪽 적용).

| noise | val_f1 | test_f1 | val 차이 (vs V3 0%) |
|---|---|---|---|
| 0% | 96.89% | 98.79% | — |
| 5% | 96.67% | 99.10% | -0.22%p |
| 10% | 97.07% | 99.19% | +0.18%p |
| 20% | 95.95% | 96.36% | -0.94%p |

## V3 5-seed 평균 (통계 유의성 검증)

같은 hparam (n=100, epochs=30) 으로 seed 5개:

| seed | val_f1 | test_f1 |
|---|---|---|
| 42 | 96.89% | 98.79% |
| 1 | 99.01% | 98.38% |
| 7 | 98.21% | 98.68% |
| 100 | 98.42% | 99.35% |
| 234 | 99.36% | 98.26% |
| **mean ± std** | **98.38% ± 0.92** | **98.69% ± 0.41** |

→ **val 단일 seed 분산 = ±0.92%p**. chip-noise 결과의 0.18~0.40%p 차이는 std 안 → **noise injection 효과 통계 유의성 0**. 0~10% 까지 robust 만 확실.

해석:
- **5%, 10% 차이는 단일 seed 분산 안** (332 val sample → 1 sample = 0.30%p 변동). 통계적 유의 없음.
- 진짜 신호 = **noise 10% 까지 model 망가지지 않는다 (robust)**. 20% 부터 명확 degrade.
- 즉 **production chip CNN 이 90%+ 정확하면 V3 그대로 적용 가능**.
- regularization 효과 (chip-noise 가 도움 된다) 주장은 통계적 근거 없음. 5 seed 평균 검증 필요.

## V3 per-class (TEST split, n=332)

V3 baseline (noise 0) 의 weak class (F1 < 1.0):

| class | F1 | P | R | Sup |
|---|---|---|---|---|
| CommaCluster | 0.857 | 1.000 | 0.750 | 8 |
| Donut_particle_blast | 0.923 | 0.857 | 1.000 | 12 |
| Edge-Bottom_invalid_main | 0.933 | 0.875 | 1.000 | 7 |
| Edge-Top_invalid_main | 0.963 | 1.000 | 0.929 | 14 |
| Full_invalid_main | 0.957 | 0.917 | 1.000 | 11 |
| Thick-Edge_invalid_main | 0.957 | 1.000 | 0.917 | 12 |

다른 28 class = F1 1.000.

## V3 per-class (VAL split, n=332)

VAL 가 더 어려움 (test 보다 낮음 — split 운):

| class | F1 | Sup | 메모 |
|---|---|---|---|
| Edge-Bottom_bank_boundary | 0.800 | 11 | weak |
| Edge-Bottom_invalid_main | 0.919 | 17 | |
| Edge-Bottom_scratch | 0.778 | 9 | weak |
| Edge-Bottom_scratch_21deg | 0.909 | 5 | |
| Edge-Top_bank_boundary | 0.933 | 7 | |
| Edge-Top_invalid_main | 0.966 | 15 | |
| Edge-Top_particle_blast | 0.909 | 12 | |
| Edge-Top_scratch | 0.800 | 6 | weak |
| Edge-Top_scratch_21deg | 0.929 | 13 | |

**Edge-Bottom / Edge-Top 4종 chip object 분별** 이 여전히 어려움. compound·wafer-only 와 같은 family. chip CNN 의 일부 chip 오분류가 여기 누적되는 것으로 추정. → V4 (chip CNN softmax) 에서 풀릴 가능성.

## 비교 — compound vs V3 chipgrid

| 측면 | compound 3ch BICUBIC 384 | V3 chipgrid 32 one-hot |
|---|---|---|
| 입력 해상도 | 384 | **32** |
| 입력 채널 | 3 (R+G+B) | **6 (R + 5 obj one-hot)** |
| obj_id 인코딩 | 정수 BICUBIC → categorical 깨짐 | **per-class binary, 보간 0** |
| 백본 | ConvNeXtV2-base 88M params | tiny CNN 1.16M params |
| 학습 데이터 | 5,680 train (full) | 2,656 train (n=100/class) |
| 학습 시간 | 8분/epoch (compound 12 epoch ≈ 1.5h) | <1초/epoch (V3 30 epoch ≈ <1분) |
| val_f1 | 97.84% | **96.89%** |
| test_f1 | 97.36% | **98.79%** |
| 종료 형태 | epoch 13 MemoryError crash | 정상 종료 |

핵심 trade-off:
- compound: 큰 모델 + 큰 입력 + 보간 손상 → 천장 ~97.8%
- V3 chipgrid: 작은 모델 + 작은 입력 + 손상 0 → 천장 ~98.8% (test 운 영향)
- **V3 가 더 효율적** + **production deploy 비용 적게**.

## 미실시 / 추후 작업

- [ ] V3 + full data (n=220/class, 7,260 sample, compound 와 같은 양) — 천장 확인
- [ ] V4 (chip CNN softmax 5채널) — `_build_obj_id_maps.py` 확장 후 시도
- [ ] V5/V6 (max prob, entropy) — V4 결과 본 후
- [ ] 5 seed 평균 — robustness 결과 통계 유의성 검증
- [ ] V3 + larger CNN (256 → 384 channel) — capacity 천장 확인
- [ ] hierarchical 2-stage (wafer pattern → chip object 분리) — Edge-Bottom/Top weak class 해결
- [ ] Normal pool max_prob threshold 측정 — open-set unknown rejection
- [ ] ensemble V1+V3 (logit 평균) — robustness 추가 향상

## 파일 위치

- 스크립트: `cnn_eval_chipgrid.py` (repo root)
- 학습 결과: `logs_chipgrid/<tag>_<TS>_<test>_<val>/`
- 데이터: `D:/project/data/wm-811k/unknown/<class>/*.png` + `D:/project/data/wm-811k/obj_id_maps/<subdir>/*.npy`
- 스킬: `.claude/skills/chipgrid-eval/SKILL.md`
- 에이전트: `.claude/agents/chipgrid-eval.md`
