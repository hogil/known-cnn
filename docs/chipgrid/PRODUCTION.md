# chipgrid V3 production 배포 설계

학습 환경 (synthetic + GT) 에서 production (real wafer + 라벨 없음) 으로 옮길 때 고려사항.

## 1. 학습 vs production 차이

| 측면 | 학습 환경 | production |
|---|---|---|
| 라벨 | GT 알고있음 | 없음 |
| chip CNN 정확도 | val_f1 1.0 (학습 데이터) | 도메인 시프트 시 떨어질 수 있음 |
| wafer class 분포 | 33 class 알고있음 | Normal / unknown 들어올 수 있음 |
| 데이터 | synthetic (균일) | real wafer (drift, anomaly 포함) |

## 2. 추론 파이프라인 (V3 기준)

```
[real wafer 들어옴]
   ↓ failbit map PNG (palette 6400×6400)
   ↓ positions JSON (chip rect)

[Step 1] chip 단위 inference
   각 chip 추출 → chip CNN (logs_chip/overall/best_model.pth)
   → softmax 5 prob 또는 argmax

[Step 2] 32×32 obj_id map 생성
   - V3 인코딩: argmax 결과를 5채널 one-hot binary 로 분리
   - V4 인코딩 (대안): softmax 5채널 그대로

[Step 3] wafer 분류
   wafer PNG 6400 → AvgPool 32×32 (R)
   + obj_id 32×32×5 (G..K)
   → V3 chipgrid 모델 (logs_chipgrid/v3_*/best_model.pth)
   → 34 class softmax + max_prob

[Step 4] 분기
   max_prob < threshold → Normal / unknown 처리
   max_prob ≥ threshold → 해당 class 출력

[Step 5] (option) 모니터링
   chip CNN 평균 max_prob 추세 (drift 감지)
   wafer CNN entropy 추세 (새 패턴 등장 감지)
```

## 3. chip CNN 오류 robustness — 검증 완료

V3 학습 시 chip-noise 인젝션으로 chip CNN 오류 시뮬레이션 결과 (n=100, seed=42):

| chip CNN 오류율 | val_f1 | test_f1 |
|---|---|---|
| 0% (이상) | 96.89% | 98.79% |
| 5% | 96.67% | 99.10% |
| 10% | 97.07% | 99.19% |
| 20% | 95.95% | 96.36% |

→ **chip CNN 90%+ 정확하면 V3 그대로 production 가능**. 80% 부근 떨어지면 명확 degrade.

production 모니터링: chip CNN 의 평균 max_prob 추적 → 임계 (예: 0.85) 아래 떨어지면 alert.

## 4. unknown / Normal 처리 — threshold 설계

학습 시 Normal class 는 제외 (CLAUDE.md 정책). production 에서 들어오는 Normal wafer 는 기존 33 class 중 max_prob 가 낮게 나옴 → threshold 로 reject.

### threshold 측정 절차

1. `D:/project/data/wm-811k/Normal/` 5,000 장 inference (chipgrid V3)
2. max_prob 분포 plot
3. 5%-tile 또는 10%-tile 을 threshold 로 채택 (false positive trade-off)
4. defect class 의 max_prob 분포와 비교 → 분리 가능한지 확인

### 권장 threshold 설계 패턴

```
Normal pool max_prob 분포    Defect pool max_prob 분포
       (low side)                  (high side)
        ▁▂▃▄                          ▁▃▆▇█
              ↑ threshold (예: 0.7)
        unknown 영역                정상 분류
```

threshold 너무 높으면 → defect 미스 (false negative)
threshold 너무 낮으면 → Normal 이 defect 로 분류 (false positive)

작업 절차:
- chipgrid V3 dispatch 후 → cnn_predict.py 또는 별도 스크립트로 Normal pool inference
- max_prob 분포 csv → 5%, 10%, 25% percentile 보고
- defect 33 class max_prob 분포와 ROC 분석 (unknown 검출 능력)

## 5. ensemble 전략 (선택)

단일 모델 한계 보완:
- **V1 (argmax 정수)** + **V3 (one-hot)** logit 평균 → 두 인코딩 보완적 신호
- **V3 (chipgrid 32)** + **compound (BICUBIC 384)** logit 평균 → 다른 해상도 보완
- voting confidence 가 단일 모델 max_prob 보다 stable

ensemble 비용: inference latency × N 모델. 운영 환경 서버 GPU 가용 따라 결정.

## 6. 제품별 분리 학습 (CLAUDE.md prod 패턴)

각 제품 (line) 마다 chip 격자 크기 / 배치 다름:
- 제품 A: 33×33 chip grid → obj_id 32×32 native fitting
- 제품 B: 25×25 → obj_id 25×25 (resize 필요)
- 제품 C: 50×50 → obj_id 50×50

→ **제품별 별도 chip CNN + chipgrid 모델 학습** 권장. CLAUDE.md prod predict pipeline 의 `--model-glob "logs_<kind>/{line}/overall/best_model.pth"` 패턴 그대로.

```
logs_chipgrid/
├── overall/                       # 글로벌 best (개발용)
├── K1AB/overall/best_model.pth    # 제품 K1AB 전용
├── K1AC/overall/best_model.pth    # 제품 K1AC 전용
└── ...
```

prod inference 호출 시 line dir 명에서 자동 substitute.

## 7. 추론 entry script (TODO)

현재 chipgrid 는 학습/평가만 가능. production 추론 entry 필요:

```python
# cnn_predict_chipgrid.py (구상)
- 단일 wafer PNG + obj_id .npy → 34 class prob + max_prob
- 폴더 walk → batch inference → JSON / parquet output
- threshold-based unknown 분기
- 기존 cnn_predict_*.py 패턴 따라 logs_predict_chipgrid/<TS>_<input>/
```

prod 변종:
```python
# cnn_predict_chipgrid_prod.py (구상)
- <image_root>/<product>/<line>/<YYYYMMDD>/*.png 트리 walk
- 자동 chip CNN inline inference + obj_id 32×32 binary mask 생성
- chipgrid 모델 inference → result_chipgrid/<product>/<line>/<date>/preds.parquet
- 기존 prod 패턴 (CLAUDE.md) 따름
```

## 8. drift 모니터링 권장

production 운영 중 다음 metric 누적:
- 일일 평균 chip CNN max_prob (chip CNN 정상 작동 확인)
- 일일 평균 wafer CNN max_prob (새 패턴 / drift 감지)
- 일일 unknown ratio (Normal 분류 비율)
- 일일 class 분포 (특정 class 급증 → 공정 이상)

drift 임계 초과 시 alert + 재학습 트리거.

## 9. open question / 추후 검증

- [ ] V4 softmax 인코딩이 V3 binary 보다 production 에서 우월한가? (chip CNN 불확실성 정보 활용)
- [ ] 5 seed 평균 — chip-noise robustness 의 통계적 유의성
- [ ] real wafer 데이터셋 (있다면) 으로 actual deploy 검증
- [ ] hierarchical 2-stage (wafer pattern → chip object) 가 Edge-Bottom/Top weak family 풀어주는지
- [ ] active learning loop — production 에서 unknown 으로 분류된 sample 누적 → 새 class 후보 발견 → 라벨링 → 재학습

## 10. 즉시 가능한 다음 액션

1. **Normal pool max_prob 측정** (V3 best_model 로) — threshold 확정
2. **`cnn_predict_chipgrid.py` 작성** — production 추론 entry
3. **V4 softmax 인코딩 build** — `_build_obj_id_maps.py` 확장
4. **5 seed 평균** — robustness 통계 검증
