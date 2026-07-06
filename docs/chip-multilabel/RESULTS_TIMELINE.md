# Results Timeline — 논문용 baseline → 기법별 성능 향상 history

**생성**: 2026-05-16 (chip-multilabel-paper-recorder 영구 기록)
**갱신 정책**: 매 iter 완료 시 1 row 추가 (append-only, 기존 row 수정 X)
**Eval set**: `E:/data/images/chip_multilabel_v15direct[_n2000]` (n200 단속 / n2000 final)
**Metric 정의** (CLAUDE.md ★ 절대규칙 260512):
- **bit_F1** = positive cells (4 single + 5 2-combo, sc+sr 제외 = 9) macro-F1
- **NI-FAR** = (Normal + Invalid) FP rate (%)
- **OOD-FAR** = OOD 4-class strict (CenterDonut, CrossScratch, DiagonalSmear, Starburst) FP rate (%)
- **Total FAR** = (NI + OOD) FP rate (%) ★ 주요 metric
- `macro_f1` legacy column 사용 안 함 (bit_F1 와 혼동 금지)

## 표 양식 (CLAUDE.md ★ 절대규칙 260515)

code block + single consolidated + padded columns + no emoji + `|` 세로 정렬

---

## A. Single model 진화 (학습 기법 별 step)

```
| Step | Recipe                                                        | Variant | bit_F1 | NI-FAR | OOD-FAR | Total FAR | Δ vs prev | 기법                                |
|------|---------------------------------------------------------------|---------|--------|--------|---------|-----------|-----------|-------------------------------------|
| 0    | T0 = backbone 직접 (no CE smoothing, no augmentation)        | -       | -      | -      | -       | -         | baseline  | ConvNeXtV2 base FCMAE pretrained    |
| 1    | T1 CE+LS=0.20 (iter 5 phase A1 peak)                         | I7      | 0.9268 | -      | -       | -         | +0.06     | label smoothing                     |
| 2    | T4 ASL γp=0 γn=2                                              | I7      | 0.85+  | -      | -       | -         | -         | asymmetric loss (collapse direction)|
| 3    | T5 BCE-only                                                   | I7      | 0.87+  | -      | -       | -         | -         | BCE multi-label baseline            |
| 4    | T7 BCE+LS=0.10 (warmup)                                       | I7      | 0.91+  | -      | -       | -         | +0.04     | BCE + LS                            |
| 5    | T9 BCE+LS=0.05-0.10 + CutMix p=0.5 + rect=0.5 (iter 8 winner) | I7      | 0.9305 | -      | -       | -         | +0.02     | + CutMix augment                    |
| 6    | T7 LS=0.30 g=3 cutmix-pair=masked (iter116J, single SOTA)    | I10     | 0.9927 |   0.00 |    0.00 |      0.00 | +0.06     | FCM-PM g=3 + pair masking + entropy gate ★ |
| 7    | T7 LS=0.30 g=3 (iter50_seed99_v3, fresh data path E:/images)  | I10     | 0.9577 |   0.00 |    0.16 |      0.04 | -0.035    | seed=99 variance (-0.035 vs s=1)    |
```

### A1. iter chain v5 결과 (n2000, POS9 strict, OOD=4 strict)

```
| iter   | TAG                            | LS   | g | seed | Variant | bit_F1 | NI-FAR | OOD-FAR | Total FAR | Status        |
|--------|--------------------------------|------|---|------|---------|--------|--------|---------|-----------|---------------|
| iter1  | iter50_clone_seed99_v3         | 0.30 | 3 | 99   | I3      | 0.9628 |   1.30 |    5.16 |      2.23 | seed var      |
| iter1  | iter50_clone_seed99_v3         | 0.30 | 3 | 99   | I7      | 0.9590 |   1.55 |    7.50 |      2.99 | seed var      |
| iter1  | iter50_clone_seed99_v3         | 0.30 | 3 | 99   | I10     | 0.9577 |   0.00 |    0.16 |      0.04 | seed var FAR  |
| iter1  | iter50_clone_seed99_v3         | 0.30 | 3 | 99   | I13     | 0.9388 |   0.00 |    0.16 |      0.04 | seed var      |
| iter2  | iter50_clone_seed42_v4         | 0.30 | 3 | 42   | I3      | 0.9681 |   8.10 |   13.59 |      9.43 | F1+ FAR-      |
| iter2  | iter50_clone_seed42_v4         | 0.30 | 3 | 42   | I7      | 0.9591 |   4.80 |    9.22 |      5.87 | regress       |
| iter2  | iter50_clone_seed42_v4         | 0.30 | 3 | 42   | I10     | 0.9577 |   0.00 |    1.25 |      0.30 | regress       |
| iter2  | iter50_clone_seed42_v4         | 0.30 | 3 | 42   | I13     | 0.9133 |   0.00 |    1.09 |      0.27 | regress       |
| iter3  | iter50_clone_seed07_v4         | 0.30 | 3 | 7    | I3      | 0.9200 | 100.00 |  100.00 |    100.00 | broken        |
| iter3  | iter50_clone_seed07_v4         | 0.30 | 3 | 7    | I7      | 0.9126 | 100.00 |  100.00 |    100.00 | broken        |
| iter3  | iter50_clone_seed07_v4         | 0.30 | 3 | 7    | I10     | 0.8732 |  19.40 |   14.53 |     18.22 | broken        |
| iter3  | iter50_clone_seed07_v4         | 0.30 | 3 | 7    | I13     | 0.7658 |  23.65 |   19.06 |     22.54 | broken        |
| iter4  | iter50_clone_LS025_s1_v4       | 0.25 | 3 | 1    | -       | -      |      - |       - |         - | eval pend     |
| iter5  | iter50_clone_LS035_s1_v4       | 0.35 | 3 | 1    | -       | -      |      - |       - |         - | queued        |
| iter6  | iter50_clone_g2_LS030_s1_v4    | 0.30 | 2 | 1    | -       | -      |      - |       - |         - | queued        |
```

### A2. Seed variance 결론 (s=1 vs s=7/42/99 같은 recipe T7 LS=0.30 g=3 corner)

```
| seed | best variant | bit_F1 | Total FAR | comment                            |
|------|--------------|--------|-----------|------------------------------------|
| 1    | I10          | 0.9927 |      0.00 | iter116J past best — favorable init|
| 7    | I10          | 0.8732 |     18.22 | broken (seed 차이 큼)               |
| 42   | I10          | 0.9577 |      0.30 | regress -0.035                     |
| 99   | I10          | 0.9577 |      0.04 | regress -0.035                     |
```

**핵심**: iter116J seed=1 만 favorable. seed 변경 시 ±0.035 ~ -0.12 variance. paper 에 multi-seed mean ± std 추가 필요.

---

## B. Ensemble 진화 (inference-side 기법)

```
| Step | Recipe                                                | bit_F1 | NI-FAR | OOD-FAR | Total FAR | Δ vs single | 기법                                          |
|------|-------------------------------------------------------|--------|--------|---------|-----------|-------------|-----------------------------------------------|
| E0   | iter116J single I10 (baseline)                        | 0.9927 |   0.00 |    0.00 |      0.00 | 0           | reference                                     |
| E1   | 3 fcmpm logit-avg + I10 entropy gate (iter116J thr)   | 0.9935 |   0.00 |    0.00 |      0.00 | +0.0008     | logit averaging + softmax-entropy short-circuit|
| E2   | 3 fcmpm vote_union_bits                               | 0.9958 |      - |       - |      0.08 | +0.0031     | per-bit majority OR (highest recall)         |
| E3   | 3 fcmpm vote_majority                                 | 0.9891 |      - |       - |      0.00 | -0.0036     | majority class vote                          |
| E4   | iter39 4-bag k=2 (old data) I10                       | 0.9555 |   4.10 |    3.91 |      4.05 | -0.0372     | bagging k-of-N                               |
| E5   | iter39 4-bag k=2 (FRESH data) I10                     | 0.9680 |   0.05 |    0.31 |      0.11 | -0.0247     | bagging fresh data                           |
| E6   | iter39 4-bag k=3 (FRESH data) I10                     | 0.8391 |   0.00 |    0.00 |      0.00 | -0.1536     | k=3 over-conservative                        |
| E7   | chain v7 3-stud vote_majority_bits I10 (s1+s77+KDv7)  | 0.9941 |   0.00 |    0.00 |      0.00 | +0.0014     | NEW CHAMPION per-bit majority (2/3)          |
| E8   | chain v7 3-stud vote_union_bits I10  (s1+s77+KDv7)    | 0.9965 |   0.40 |    1.88 |      0.76 | +0.0038     | peak F1 Pareto extreme (+FAR cost)           |
| E9   | chain v7 3-stud vote_majority I10    (s1+s77+KDv7)    | 0.9936 |   0.00 |    0.00 |      0.00 | +0.0009     | label-level majority (smaller margin)        |
| E10  | chain v8 reconfirm vote_majority_bits I10             | 0.9941 |   0.00 |    0.00 |      0.00 | +0.0014     | bit-identical to E7 (stability check)        |
| E11  | chain v8 reconfirm vote_majority_bits I13             | 0.9600 |   0.00 |    0.00 |      0.00 | -0.0327     | I13 gate collapses fork-scratch combos       |
| E12  | chain v8 reconfirm vote_union_bits   I13              | 0.9923 |   0.05 |    1.88 |      0.49 | +0.0          | moderate Pareto at I13                       |
| E13  | chain v10 Model Soup uniform mean 3-way I10           | 0.9748 |   0.00 |    0.00 |      0.00 | -0.0179       | weight-space avg below vote (Wortsman 2022)  |
| E14  | chain v10 Model Soup uniform mean 3-way I13           | 0.9564 |   0.00 |    0.00 |      0.00 | -0.0363       | weight-space avg below vote at I13           |
| E15  | no_kd 3-way s1+s77+s33_v15 vote_majority_bits I10     | 0.9929 |   0.35 |    0.00 |      0.27 | +0.0002       | base-only ensemble champion (no KD)          |
| E16  | no_kd 3-way s1+s77+s33_v15 vote_majority      I10     | 0.9928 |   0.35 |    0.00 |      0.27 | +0.0001       | label-level majority (matches E15 within px) |
| E17  | no_kd 3-way s1+s77+s33_v15 vote_unanimous     I10     | 0.9375 |   0.00 |    0.00 |      0.00 | -0.0552       | all-3-agree strict (FAR floor, F1 collapse)  |
| E18  | no_kd 3-way s1+s77+s33_v15 vote_intersection  I10     | 0.9701 |   0.00 |    0.00 |      0.00 | -0.0226       | per-bit AND (zero FAR, mid F1)               |
| E19  | no_kd 3-way s1+s77+s33_v15 vote_union_bits    I10     | 0.9880 |  20.05 |    2.97 |     15.91 | -0.0047       | per-bit OR (FAR explodes vs E2 fcmpm union)  |
| E20  | iter39 4-bag paperMain {24_LS030_s42+26B+26D+26H} I10 | 0.9555 |   4.10 |    3.91 |      4.05 | -0.0372       | n=2000 reverify of past n=200 headline 0.9955 — past headline = sample-size over-fit artifact |
| E21  | 4-way {s1+s77+LS20_s77+KDv7} vote-bits k=2 I10        | 0.9953 |   0.00 |    0.00 |      0.00 | +0.0026       | NEW CHAMPION (260518) — adds LS=0.20 axis to chain v7 pool, +0.0012 vs E7 at same 0% FAR |
| E22  | 6-way {5+LS20_s1} vote-bits k=3 I10                   | 0.9947 |   0.00 |    0.00 |      0.00 | +0.0020       | 2nd-place 0-FAR (-0.0006 vs E21) — adds LS20_s1 6th member, no further lift over 4-way |
| E23  | 5-way {4+LS30_s11} vote-bits k=3 I10                  | 0.9942 |   0.00 |    0.00 |      0.00 | +0.0015       | 3rd 0-FAR — LS30_s11 5th member ties E7 (-0.0011 vs E21)                                       |
| E24  | 3-strong {LS30_s1+s77+LS20_s77} vote-bits k=2 I10     | 0.9940 |   0.00 |    0.00 |      0.00 | +0.0013       | base-only 3-way w/ LS=0.20 axis (no KD) — tied E7 within px (-0.0013 vs E21)                   |
| E25  | 5-way {4+LS30_s11} vote-bits k=2 I10 (looser FAR)     | 0.9964 |   0.05 |    1.56 |      0.42 | +0.0037       | Pareto extreme (peak F1 +0.0011 vs E21 at +0.42 pp FAR)                                        |
| E26  | 4-way {s1+s77+LS20_s77+KDv7} vote-bits k=1 I13        | 0.9955 |     -  |      - |      0.49 | +0.0028       | Pareto loose-FAR I13 alt (+0.49 pp FAR cost)                                                   |
```

**Insight (E21 NEW CHAMPION, 260518 paper-recorder cron).** Adding the
**LS=0.20 seed=77** member (`iter116J_g3_ls20_s77_v17`, freshly trained 260518
12:05) to the chain v7 champion pool {iter116J_s1, iter116J_clone_s77, KD_v7}
produces a **4-way per-bit majority k=2** ensemble at **bit_F1 0.9953 /
Total FAR 0.00 %** — a strict improvement over E7 (+0.0012 bit_F1, same 0% FAR)
and the past iter39 n=2000 paper main (+0.0398 bit_F1, -4.05 pp FAR). The
LS=0.20 student adds a **second independent label-smoothing axis** (LS=0.30 was
the only LS value in chain v7), confirming that LS diversity is a
not-yet-exhausted lift direction. 6-way (E22) and 5-way (E23) extensions add
more members but **no further bit_F1 lift over the 4-way at zero FAR** — the
k-of-N voting margin saturates at 4-of-strong-students; additional KD/seed
members past the 4-way only add cost. For paper headline reporting, **E21 =
NEW CHAMPION** at 0.9953 / 0.00 % (4 models, 4× inference cost vs single SOTA
0.9927). For F1-max Pareto reporting, **E25 = 5-way k=2 at 0.9964 / 0.42 %**
gives +0.0011 bit_F1 at the cost of +0.42 pp Total FAR — both publishable as
the conservative and aggressive operating points. _Source:
`outputs/_ens_4way_3strong_plus_KDv7_I10.json`,
`outputs/_ens_5way_3strong_KDv7_LS30s11_I10.json`,
`outputs/_ens_6way_5plus_LS20s1_I10.json`,
`outputs/_ens_3strong_v18analyst_I10.json`,
`outputs/_ens_4way_3strong_KDv7_I13.json`._

**Insight (E15-E19, paper-recorder cron #49 06:10).** Replacing KD_v7
with iter116J_s33_v15 in the 3-way per-bit majority ensemble yields
**E15 = vote_majority_bits 0.9929 / 0.27 % Total FAR** — the new
"base-only ensemble" champion (E7 with-KD remains the absolute champion
at 0.9941 / 0.00 %). The −0.0012 bit_F1 drop and +0.27 pp FAR gap vs
E7 quantifies the KD calibration contribution to the 3-way ensemble: the
KD_v7 student's softmax-entropy gate at I10 was the deciding vote on
the 0.35 % NI-FAR cells that E15 (s33_v15 replacing KD_v7) now mis-flags.
Per user directive 06:00 ("학습 → KD → ensemble → 최종 KD"), the paper
will report E15 as the *base ensemble headline* (no KD) and E7 as the
*KD-mixed final ensemble* — separating the two stages so reviewers can
attribute the +0.0012 / −0.27 pp lift specifically to the KD student
inclusion. _Source: `outputs/_ensemble_no_kd_s1_s77_s33_I10.json`._

---

## C. KD 시도 (negative results 기록 — paper 의 "what didn't work")

```
| Tag                                | Teacher                              | α   | T   | LS   | bit_F1 best variant | Total FAR best | Status                          |
|------------------------------------|--------------------------------------|-----|-----|------|---------------------|----------------|---------------------------------|
| KD_iter116J_3fcmpm_teach           | 3 fcmpm                              | -   | -   | 0.30 | I3 = 0.9491         | 100.00         | I7-I13 collapsed (val_f1=0)     |
| KD_v3_iter50setting_T8_a03         | 3 fcmpm                              | 0.3 | 8   | 0.30 | -                   | -              | no eval (best_val_acc 0 ep1)    |
| KD_v4_iter50B_exact_T4_a05_LS20    | 3 fcmpm                              | 0.5 | 4   | 0.20 | I3 = 0.9408 (n200)  | 100.00         | I7-I13 < 0.2 (broken)           |
| KD_v5_alpha02_iter116J_recipe      | 3 fcmpm                              | 0.2 | 4   | 0.30 | I3 = 0.8658         | 100.00         | full collapse (best ep1)        |
| KD_v6_4sota_iter50B_clone          | 4 SOTA (3 fcmpm + iter116C g=2)      | 0.5 | 4   | 0.20 | -                   | -              | no eval (train log 0 byte)      |
| KD_v7_iter116J_a03_T2_skipcutmix   | iter116J g3_ls30 single              | 0.3 | 2   | 0.30 | I10 = 0.9265        | 0.00           | FIRST non-collapse (--kd-skip-on-cutmix) |
| KD_v8_a05_T2_skipcm                | iter116J g3_ls30 single              | 0.5 | 2   | 0.30 | I10 = 0.8924        |          57.15 | re-eval landed cron 7 (15:04 dispatch / 15:04 done); ep01 collapse (val_macro_f1=0); FAR explodes 0→57% vs KD_v7 |
| KD_v9_a02_T2_skipcm                | iter116J g3_ls30 single              | 0.2 | 2   | 0.30 | -                   | -              | 2h+ stalled (cron 10 19:34; best mtime 17:24 = 2h 10m stale; pid 17680 alive); α=0.2 collapse suspected; user kill-decision boryu; eval pending |
| KD_v10_a03_T1_skipcm               | iter116J g3_ls30 single              | 0.3 | 1   | 0.30 | -                   | -              | chain v9b phase 1 second slot, queued after KD_v9 |
| KD_v11_a025_T2_skipcm_v15          | iter116J g3_ls30 single              | 0.25| 2   | 0.30 | I10 = 0.9192        |           0.00 | chain v15 #1; non-collapse, F1 -0.073 vs E7 champ; NI=0 OOD=0 (n2000)   |
| KD_v12_a030_T3_skipcm_v15          | iter116J g3_ls30 single              | 0.30| 3   | 0.30 | I10 = 0.9470        |           0.00 | chain v15 #2 NEW best KD; T=3 sweet spot vs T=2; +0.0205 vs KD_v7 (I10) |
| KD_v13_a030_T4_skipcm_v15          | iter116J g3_ls30 single              | 0.30| 4   | 0.30 | I10 = 0.9347        |           0.00 | chain v15 #3 LANDED cron 46 (eval 05:14); below KD_v12 (T=3) by -0.0123 — T=3 still sweet spot, T=4 over-smooths |
| KD_v14_a035_T25_skipcm_v15         | iter116J g3_ls30 single              | 0.35| 2.5 | 0.30 | -                   | -              | chain v15 #4 train running 05:14 → ETA ~05:22                              |
| iter116J_s33_v15                   | iter116J g3_ls30 single (seed=33)    | -   | -   | 0.30 | I10 = 0.9576        |           0.00 | s33 v15 LANDED cron 46 (eval 05:26); Phase 2 ensemble candidate; -0.0044 vs champ 0.9620 floor; per-class F1: bb 0.9369 / fork 0.9430 / scratch 0.9503 / scratch_rot 1.0000 |
| iter116J_s55_v15                   | iter116J g3_ls30 single (seed=55)    | -   | -   | 0.30 | -                   | -              | s55 v15 train running 05:26 → ETA ~05:34; chain v15 final slot                          |
| KD_E1_a030_T2_skipcm_v16           | 3-way ensemble (s1+s77+iter116J avg) | 0.30| 2   | 0.30 | I10 = 0.7040 (POS9) |           8.71 | chain v16 NEGATIVE — POS9 strict bit_F1 0.7040 (I10) / 0.6672 (I13). eval_log macro_f1=0.8761(I10). vs KD_v7 single-teacher POS9 0.9265 (I10): -0.2225. Ensemble-as-teacher washes per-seed calibration; flatter soft target → student fits mixture mean. I13 FAR=0%, I10 NI-FAR=11.50%. Champion unchanged (E7 single-teacher 0.9941) |
```

**결론** (paper § negative results, updated 2026-05-17 cron 7): KD_v1-v6 6 recipe 모두 student 가
degenerate solution 으로 수렴. val_acc=1.0 + v_f1=0 = constant prediction (None for all).
원인 = teacher prob computed on clean chip vs student sees mixed chip → KL mismatch on the
25 % CutMix-active batches.  **KD_v7 (chain v6 phase 4) 의 `--kd-skip-on-cutmix` flag 로
해결** — KD loss 가 cutmix-active batch 에서만 disabled, 나머지 batch 는 normal KD.
첫 non-collapse KD 결과 bit_F1 0.9265 / Total FAR 0.00 % (I10).  Single-teacher 한계로
bit_F1 < teacher (0.9748), 그러나 chain v7 ensemble 에서 KD_v7 의 calibrated I10 vector 가
deciding-vote 로 작동 → ensemble 0.9941 NEW CHAMPION 의 essential third member.

**KD α grid at T=2 — viable corner is α=0.3 only** (chain v9 cron 7 closure).  Re-eval
of KD_v8 (α=0.5, T=2, skip-cm) landed at cron 7 (15:04 dispatch after GPU drop to 55 %)
and showed an ep01 collapse: `val_macro_f1=0`, I10 bit_F1 0.8924 with Total FAR **57.15 %**
(NI 32.15 / OOD 79.46).  Combined with the prior KD_v2 (α=0.7, over-smoothed) and KD_v5
(α=0.2, full collapse) results, the 4-point α grid (0.2, 0.3, 0.5, 0.7) at T=2 has
**only one viable cell — α=0.3**.  The KD recipe family at T=2 is therefore closed:
no second corner exists to fall back on, and the chain v9 KD search budget should
not re-spend on T=2 cells.  (T=1, T=4, T=8 corners are still partially unmapped, but
KD_v3 at T=8 / α=0.3 already collapsed at 0.6435 / 100 %, so the T-axis appears not
to widen the viable window.)

---

## D. 핵심 발견 (paper § contribution, updated 2026-05-17 cron 6)

1. **FCM-PM (Free Chip Mix - Partial Match)** = cutmix-mode=complement + n_groups=3 + cutmix-pair=masked + LS=0.30 + corner fill → **0.9927 / 0%** single SOTA (iter116J)
2. **I10 inference variant** = I7 (per-class threshold) + softmax-entropy gate → Normal/Invalid/OOD chip 모두 active 예측 차단 (FAR 0% 달성 핵심)
3. **logit averaging + entropy gate** = single model 보다 +0.0008 (marginal). vote_union_bits 가 더 큼 (+0.0031).
4. **seed variance** = s=1 vs s=99 → bit_F1 ±0.035. paper 에 multi-seed 표 필수.
5. **KD collapse fix** = `--kd-skip-on-cutmix` (KD_v7) breaks the 6-attempt collapse streak.  Single-teacher KD reaches bit_F1 0.9265 / FAR 0.00 % (I10), below the teacher ceiling but ensemble-essential.
6. **★ Ensemble champion** = `vote_majority_bits` of {iter116J s=1, iter116J_clone_s77, KD_v7} → **bit_F1 0.9941 / Total FAR 0.00 %** (I10). First cell across chain v5+v6+v7+v8 to beat the iter116J SOTA on bit_F1 with **zero FAR penalty** (+0.0014).  Re-confirmed by chain v8 at identical metric (E10).
7. **Pareto extremum** = `vote_union_bits` recovers the hard `bank_boundary+scratch` combo (0.9791 → 0.9913) at +0.76 pp Total FAR — publishable as the F1-max corner of the Pareto envelope, not as the headline.
8. **Diversity > tuning** = the three students individually span bit_F1 [0.4738, 0.9786]; one is a near-disaster at I10, yet per-bit majority over the three yields 0.9941.  Bit-level majority tolerates single-student degradation when the other two agree.
9. **★ Past paper main (iter39 4-bag) n=200 → n=2000 degradation** (260518 cron, E20) — iter39 4-bag `{24_LS030_seed42 + 26B + 26D + 26H}` headline 0.9955/0% at n=200 collapses to **0.9555/4.05%** at n=2000 (POS9 strict): **-0.0400 bit_F1, FAR 0 → 4.05 %** (NI 4.10 / OOD 3.91). Root cause: `24_LS030_seed42` base member's OOD weakness, masked by 200-chip sampling, surfaces in the larger n=2000 chip distribution; 4-bag majority cancel mechanism reaches its limit. The current chain v7 champion `{iter116J_s1 + s77 + KD_v7}` (E7) at 0.9941/0% n=2000 **out-performs the past paper main by +0.0386 bit_F1 and -4.05 pp FAR** while remaining iter116J-based (g3 LS=0.30, single-recipe seed-diverse). The n=200 headline was a sample-size over-fit artifact; **n=2000 POS9 strict is the production-grade reliable metric** that should anchor the paper. _Source: `outputs/_ensemble_iter39_4bag_paperMain_n2000_I10.json`._
10. **★ NEW CHAMPION 4-way LS-axis-extended ensemble** (260518 cron, E21) — Adding the freshly trained `iter116J_g3_ls20_s77_v17` (LS=0.20 seed=77) as the 4th member to the chain v7 champion pool yields **bit_F1 0.9953 / Total FAR 0.00 %** under per-bit majority k=2 at I10 — strictly dominating E7 (+0.0012 bit_F1 at same 0% FAR) and the past iter39 n=2000 paper main (+0.0398 bit_F1, -4.05 pp FAR). Confirms **label-smoothing diversity (LS=0.20 ⊕ LS=0.30)** is a not-yet-exhausted lift axis: chain v7 had only LS=0.30 members; the LS=0.20 student introduces a complementary calibration profile. 5-way (E23: +LS30_s11) and 6-way (E22: +LS20_s1) extensions add cost but no further bit_F1 lift at zero FAR — the k-of-N margin saturates at 4-of-strong-students. Pareto extreme E25 (5-way k=2 at 0.9964 / 0.42 % FAR) gives +0.0011 bit_F1 for +0.42 pp FAR — publishable as the aggressive operating point alongside E21 as the conservative point. _Source: `outputs/_ens_4way_3strong_plus_KDv7_I10.json`._

---

## E. 다음 진행 계획 (chain v6+v7+v8 완료 후 — chain v9 in progress 2026-05-17 15:03)

1. ✅ **DONE chain v6** — 3-seed sweep (s=11, s=23, s=77) + KD_v7 skipcutmix.  s=77 micro-win (+0.0038 bit_F1 at +0.76 pp FAR); KD_v7 first non-collapse.
2. ✅ **DONE chain v7** — 5-mode vote aggregation over the 3 v6 students → vote_majority_bits NEW CHAMPION 0.9941 / 0.00 %.
3. ✅ **DONE chain v8 phase 1** — ensemble re-confirmation at I10 + I13.  I10 bit-identical to v7; I13 below I10 (gate collapse on combos).
4. **IN PROGRESS chain v9** — GPU-gated supervisor (wait_gpu_free < 60 % eval / < 50 % train, 30-120 min fallback).
   - Phase 0: re-eval KD_v8_a05_T2_skipcm (started 15:03:58 after GPU dropped 55 → free).
   - Phase 1: retry KD_v9 (α=0.2 T=2) + KD_v10 (α=0.3 T=1) — both OOM'd in chain v7.
   - Phase 2: cutmix-p sweep (5 cells: 0.15 / 0.20 / 0.30 / 0.35 / 0.40 at seed=42).
   - Phase 3: complement-label-scale sweep (3 cells: 0.3 / 0.7 / 1.0).
5. **POST-CHAIN-v9 plan**: per-class confidence-weighted bit aggregation on `bank_boundary+scratch` and `fork+scratch` (only remaining gap to push bit_F1 > 0.9965 without paying FAR), 5-student odd ensemble once KD_v9/v10 land.
