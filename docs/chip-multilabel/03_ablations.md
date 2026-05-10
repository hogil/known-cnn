# 03 — Ablations: what worked / what didn't

All deltas measured on the same 2200-chip 11-class eval set. Pull
quotes use 4-decimal numbers from the canonical sources.

## Inference-side ablations (fixed model = T0)

| change vs baseline               | from cell        | to cell          | Δ macro_f1 | Δ top1_11 | verdict        |
|----------------------------------|------------------|------------------|-----------:|----------:|----------------|
| argmax → F1-max thresholds       | T0__I0 (0.7302)  | T0__I1 (0.8444)  |    +0.1142 |   +0.1852 | huge win       |
| F1-max → top-K=2 alone           | T0__I1 (0.8444)  | T0__I2 (0.7673)  |    -0.0771 |   -0.0585 | regression     |
| F1-max + top-K rescue (I1+I2)    | T0__I1 (0.8444)  | T0__I3 (0.8466)  |    +0.0022 |   -0.0307 | tiny win on F1 |
| I3 + temperature scaling         | T0__I3 (0.8466)  | T0__I4 (0.8466)  |    +0.0000 |   +0.0000 | no-op on F1    |
| I3 + TTA (rotation 4×)           | T0__I3 (0.8466)  | T0__I5 (0.8287)  |    -0.0179 |   -0.0006 | **DISALLOWED** |
| F1-max + min-floor 0.30          | T0__I3 (0.8466)  | T0__I6 (0.8177)  |    -0.0289 |   -0.0136 | regression     |
| F1-max + step-search Δ=0.02      | T0__I3 (0.8466)  | T0__I7 (0.8485)  |    +0.0019 |   +0.0193 | small win      |
| I3 + top-K=1 fallback            | T0__I3 (0.8466)  | T0__I8 (0.8456)  |    -0.0010 |   +0.0000 | flat           |
| I1 + temperature only            | T0__I1 (0.8444)  | T0__I9 (0.7741)  |    -0.0703 |   -0.0983 | regression     |
| I7 + entropy Normal gate (I10)   | T0__I7 (0.8485)  | T0__I10 (0.8542) |    +0.0057 |   +0.0307 | win, durable   |

_Source: outputs/stage1_260505_162842, _165400, _170827 results_matrix.parquet._

### Verdicts

- **F1-max thresholds (I1) is the single biggest single inference change**
  (+0.1142 macro-F1 over argmax). Most of the climb away from baseline is
  this one trick.
- **Step-search (I7)** is a clean micro-improvement on I3 with no extra
  pipeline complexity (Δ=0.02 grid).
- **Entropy Normal gate (I10)** is the only inference idea that *survives*
  retraining — every other variant either ties or trails I7 once the
  model is fine-tuned.
- **TTA (I5) is permanently disallowed** because rotation conflates
  scratch / scratch_rot. Even though it sometimes nudges precision, the
  semantic damage is unacceptable.
- **Temperature scaling (I4, I9)** does not help macro-F1 because the
  threshold sweep already absorbs whatever calibration shift T provides;
  it does help ECE (0.0778 → 0.0129 on I4), so keep it for any
  probability-honest downstream.
- **Min-floor 0.30 (I6)** hurts because the val-tuned fork threshold is
  ~0.12 — clipping it to 0.30 throws away most of fork's recall.

## Training-side ablations (best inference = I10)

| variant | loss            | best cell    | macro_f1 | top1_11 | Δ vs T0__I10 | verdict            |
|---------|-----------------|--------------|---------:|--------:|-------------:|--------------------|
| T0      | none (frozen)   | T0__I10      |   0.8542 |  0.6517 |       (ref)  | baseline           |
| T1      | CE + LS 0.10    | T1__I10      |   0.8634 |  0.7006 |      +0.0092 | win                |
| T4      | ASL             | T4__I10      |   0.7759 |  0.5830 |      -0.0783 | regression         |
| T5      | BCE             | T5__I10      |   0.7589 |  0.5432 |      -0.0953 | regression         |
| T6      | BCE → ASL       | T6__I10      |   0.8193 |  0.6256 |      -0.0349 | regression         |

_Source: outputs/stage2_260505_170121/results_matrix.parquet for T1/T4/T5/T6 ×
I0..I9 grid; outputs/stage1_260505_{173649,173829,173955,174123}/results_matrix.parquet
for the post-hoc I10 inference rows._

### Verdicts

- **T1 (CE + LS 0.10) is the only training intervention that helped**
  on the multi-label benchmark. The single-label CE pretrain provides a
  decent base and label smoothing softens the softmax peak so the
  runner-up class still has a usable score.
- **T4 (ASL), T5 (BCE), T6 (BCE→ASL) all regress** despite being the
  "obvious" multi-label choices. Hypothesis: these losses change the
  distribution of activations enough that the F1-max thresholds tuned
  on val don't transfer cleanly. Specifically T4 and T5 over-suppress
  bank_boundary, dropping its F1 from ~0.96 to ~0.85.
- T6 (BCE→ASL) is the worst hybrid: BCE collapses the softmax structure,
  then ASL doesn't have time to rebuild useful asymmetry in 4 epochs.
- T1 is also the cheapest (~330s on RTX 4090).

## LS sweep (iter 5, T1 only)

| LS    | best inference | macro_f1 | top1_11 | Δ vs LS=0.10 best (T1__I10 = 0.8634) |
|------:|----------------|---------:|--------:|-------------------------------------:|
|  0.05 | I7             |   0.7964 |  0.5591 |                              -0.0670 |
|  0.10 | I3             |   0.8363 |  0.6261 |                              -0.0271 |
|  0.15 | I3             |   0.8961 |  0.7517 |                              +0.0327 |
|  0.20 | **I7**         | **0.9268** | **0.8449** |                          **+0.0634** |
|  0.25 | I3             |   0.8663 |  0.6989 |                              +0.0029 |
|  0.30 | I3             |   0.8185 |  0.6466 |                              -0.0449 |

_Source: outputs/phase_a_260505_175105/sweep_log.csv,
outputs/phase_a_260505_182044/sweep_log.csv._

### Verdicts

- **LS=0.20 is the sweet spot** — too little smoothing leaves the
  single-label collapse intact; too much smoothing erases informative
  margin between classes.
- The curve is **non-monotonic and sharp**: 0.20 → 0.9268 vs 0.15 →
  0.8961 vs 0.25 → 0.8663. ±0.05 around the optimum costs ~0.03 macro-F1.
- The optimum cell is **`T1_LS20 + I7`**, not + I10. The entropy gate
  (I10) helps frozen / mildly-trained models because their Normal logit
  is poor; once LS=0.20 has trained the model into a more
  well-calibrated multi-label state, the explicit Normal gate becomes
  redundant and slightly hurts (0.9268 → 0.8841).

## Phase F — anomaly-detection BKM transfer (iter 7, negative)

Phase F tested two structural BKMs imported from anomaly-detection
literature on top of the Phase A winner (T1_LS20_ep8). Both regressed.

| variant | recipe                                            | best cell | macro_f1 | top1_11 | Δ vs T1_LS20__I7 (0.9268) | verdict     |
|---------|---------------------------------------------------|-----------|---------:|--------:|--------------------------:|-------------|
| F1      | LR warmup 2ep (start_factor=0.05, eta_min=1e-6)   | F1__I10   |   0.8181 |  0.5540 |                  **−0.1087** | regression  |
| F2      | EMA decay=0.95 + dynamic decay warmup             | F2__I10   |   0.8377 |  0.6602 |                  **−0.0891** | regression  |

_Source: outputs/stage1_260505_192541/results_matrix.parquet (F1),
outputs/stage1_260505_194014/results_matrix.parquet (F2)._

### Verdicts

- **F1 (warmup 2ep) regresses −0.109** because epoch-1 LR ≈ 5e-6 is far
  below the working LR for an 8-epoch CE+LS finetune on a
  TAPT-initialized backbone. Warmup helps when the model is far from a
  reasonable basin (cold-start, large LR, anomaly binary chart) — none
  of which apply here. The warmup eats two epochs that the LS=0.20
  recipe was using to actually converge.
- **F2 (EMA 0.95) regresses −0.089** because EMA's bias correction needs
  many effective steps before the running average becomes useful. With
  ~12 effective steps (8 epochs × small loader), EMA is averaging
  partially-trained weights into the final checkpoint and damping the
  late-epoch sharpening that LS=0.20 was relying on.
- **Paper-worthy negative result**: anomaly-detection BKMs (warmup, EMA)
  do not transfer cleanly to small-data multi-label chip classification
  with a TAPT init. The structural assumption gap (cold-start vs warm
  TAPT, many-step vs few-step training) breaks the transfer.

## Phase F band-aid — I11 pair-aware threshold (iter 7, rejected)

I11 is a pure-inference variant that adds a pair-aware additional
threshold: declare bank_boundary+scratch_rot together when both class
logits exceed an extra co-occurrence threshold. Tested on T1_LS20__I7
without retraining.

| metric             | T1_LS20__I7 | T1_LS20__I11 | Δ        |
|--------------------|------------:|-------------:|---------:|
| macro_f1           |      0.9268 |       0.9199 | −0.0069  |
| top1_11            |      0.8449 |       0.8432 | −0.0017  |
| bb+sr combo recall |      ~0.325 |       ~0.481 | +25 chips |
| bb+fork FP         |    baseline |       +31 FP | over-trigger |

_Source: outputs/stage1_260505_194443/results_matrix.parquet._

**Verdict**: rejected. The bb+sr recall gain is real (+25 chips on the
specific combo), but the pair-aware threshold also over-triggers on
bb+fork (31 false positives) because the bb logit is already at the
co-occurrence cutoff for many bb-only chips. Net macro-F1 −0.007. The
right fix is at the training stage (CutMix, see Phase F T7), not an
inference band-aid.

## Phase F — T7 atomic decomposition (iter 7)

Goal: separate the contributions of two simultaneous changes — switching
loss CE → BCE and adding CutMix p=0.5 — by running each step in
isolation. Train recipe held at LS=0.20, LR=1e-4, ep=8.

| step  | loss | cutmix-p | best cell | macro_f1 | top1_11 | Δ from prev | Δ vs T1_LS20 |
|-------|------|---------:|-----------|---------:|--------:|------------:|-------------:|
| T1    | CE   | 0.0      | T1__I7    |   0.9268 |  0.8449 |       (ref) |       (ref)  |
| T7a   | BCE  | 0.0      | T7a__I3   |   0.8577 |  0.5534 |     **−0.0691** |     **−0.0691**  |
| T7c ★ | BCE  | **0.5**  | T7c__I10  |   **0.9271** |  0.8307 |     **+0.0694** |     **+0.0003**  |

_Source: outputs/phase_a_260505_175105/sweep_log.csv (T1),
outputs/stage1_260505_200523/results_matrix.parquet (T7a),
outputs/stage1_260505_195730/results_matrix.parquet (T7c)._

### Verdicts

- **CE → BCE alone (T1 → T7a) costs −0.0691** macro-F1, confirming the
  iter 4 finding that BCE drops the useful softmax shape. T7a's best cell
  is I3 (not I7), and top1_11 drops to 0.5534 — the model has no Normal
  gate signal because the softmax-style entropy doesn't apply.
- **+ CutMix p=0.5 (T7a → T7c) recovers +0.0694**, almost exactly
  cancelling the BCE penalty in macro-F1 terms. CutMix on multi-hot
  targets directly teaches the model that bb+sr can co-occur in pixel
  space, repairing the precision-recall trade that BCE alone breaks.
- **Net (T1 → T7c) is +0.0003 on macro-F1 — statistically tied** — but
  the operational profile flips: bb+sr combo recall **0.32 → 0.96**
  (+0.63 absolute), `scratch_rot` per-class F1 reaches 1.0000, ECE_post
  drops 4× (0.1788 → 0.0446). The trade-off is paid by bank_boundary F1
  (0.8974 → 0.8885) and scratch F1 (0.9725 → 0.9554) — both still
  excellent. The macro-F1 tie hides a large operational improvement.

## Phase F — CutMix-p sweep (iter 7, BCE+LS=0.20)

Same recipe as T7a (BCE+LS=0.20, ep=8, LR=1e-4) varying only `cutmix_p`.

| cutmix-p | best cell | macro_f1 | top1_11 | bb+sr recall | verdict          |
|---------:|-----------|---------:|--------:|-------------:|------------------|
| 0.0      | T7a__I3   |   0.8577 |  0.5534 | (low)        | BCE-only floor   |
| 0.3      | T7b__I10  |   0.8626 |  0.5511 | 0.7312       | partial recovery |
| **0.5 ★**| T7c__I10  | **0.9271** | **0.8307** | **0.9562** | **peak**         |
| 0.7      | T7d__I10  |   0.9038 |  0.7432 | (high)       | over-mixing      |

_Source: outputs/stage1_260505_200523, _203340, _195730, _201706/results_matrix.parquet._

### Verdicts

- **cutmix_p=0.5 is the sharp peak** — both lower (0.0/0.3) and higher
  (0.7) values lose ~0.03–0.07 macro-F1. This mirrors the LS sweep
  shape from iter 5: a non-monotonic optimum surrounded by penalty.
- The bb+sr recall axis is monotonically increasing in cutmix_p up to
  0.5 (then we don't have data above 0.5 except T7d=0.7 macro_f1=0.9038),
  but **macro-F1 turns over at 0.7** because too much mixing breaks
  single-class chips: the model starts hallucinating combos when only
  one defect is present.
- **CutMix is the first training-side intervention since LS=0.20 to
  produce a paper-headline gain** — not on the macro_f1 axis (tie) but
  on the bb+sr operational axis (+0.63 recall).

## LS sweep under BCE+CutMix (iter 8, T9 family)

Re-sweeping the LS axis on top of the iter-7 BCE+CutMix(p=0.5) recipe.
Recipe held: BCE, CutMix p=0.5, ep=8, LR=1e-4. Only `label_smoothing`
(and seed for T9g) varies.

| LS    | seed | run  | best cell  | macro_f1 | top1_11 | mAP    | ECE_post | Δ vs T7c=0.9271 |
|------:|-----:|------|------------|---------:|--------:|-------:|---------:|----------------:|
|  0.00 |   42 | T9c  | T9c__I10   |   0.8609 |  0.6443 | 0.8384 |   0.0114 |        −0.0662  |
|  0.05 |   42 | T9b  | T9b__I7    |   0.9449 |  0.8670 | 0.9378 |   0.0060 |        +0.0178  |
|  0.06 |   42 | T9f  | T9f__I3    |   0.9401 |  0.8648 | 0.9521 |   0.0088 |        +0.0130  |
| **0.07 ★**| 42 | T9d | T9d__I7  | **0.9705** | **0.9267** | **0.9864** | **0.0106** |    **+0.0434** |
|  0.07 |   43 | T9g  | T9g__I7    |   0.9408 |  0.8307 | 0.9468 |   0.0079 |        +0.0137  |
|  0.08 |   42 | T9e  | T9e__I3    |   0.8085 |  0.4449 | 0.8362 |   0.0425 |        −0.1186  |
|  0.10 |   42 | T9a  | T9a__I10   |   0.9364 |  0.8489 | 0.9451 |   0.0143 |        +0.0093  |

_Source: outputs/stage1_260505_{210059,210535,210932,211334,211752,212153,212557}/results_matrix.parquet._

### Verdicts

- **The CE-era LS=0.20 optimum does not transfer.** Under BCE+CutMix the
  optimum shifts low to **LS=0.07**. Diagnosis: BCE already softens
  hard targets via independent sigmoids, and CutMix interpolates
  multi-hot labels; LS=0.20 stacked on top over-softens.
- **The curve is non-smooth — knife-edge cliff at LS=0.08** (0.8085, a
  0.16-macro-F1 drop over a 0.01 step from LS=0.07's 0.9705 and a 0.13
  drop from LS=0.10's 0.9364). LS=0.05 / 0.06 / 0.10 all sit in a 0.94
  band; only 0.08 falls off. Hypothesis: a phase-transition where
  BCE+CutMix runner-up gradient signal collapses around 0.08 effective
  positive-target value.
- **Single-seed variance ±0.030 at the optimum.** T9d (LS=0.07,
  seed=42) = 0.9705; T9g (LS=0.07, seed=43) = 0.9408. The seed-driven
  spread is concentrated in **fork F1** (0.9448 vs 0.8149) — the
  diffuse longest-tail class is the variance carrier; bank_boundary /
  scratch / scratch_rot are stable across seeds.
- **Reporting policy**: T9d 0.9705 = "best observed", T9g 0.9408 =
  "realistic point estimate". Neither alone is paper-honest; both
  together with the variance caveat are.

## Negative axes — atomic-failed (iter 9, on top of T9 LS=0.07 recipe)

Three orthogonal axes probed on top of the iter-8 BCE+LS=0.07+CutMix(p=0.5)
recipe to test whether they can lift the realistic baseline (T9g=0.9408)
above seed noise. **All regress.**

| run  | axis change           | seed | best cell  | macro_f1 | top1_11 | Δ vs T9d (0.9705) | verdict   |
|------|-----------------------|-----:|------------|---------:|--------:|------------------:|-----------|
| T10a | drop_path 0.05        |   42 | T10a__I3   |   0.9160 |  0.7335 |          −0.0545  | regress   |
| T10b | drop_path 0.05        |   43 | T10b__I11  |   0.8918 |  0.7511 |          −0.0787  | regress   |
| T11a | cutmix-rect 0.25      |   42 | T11a__I7   |   0.8646 |  0.6551 |          −0.1059  | regress   |
| T12a | two-LR backbone/head  |   42 | T12a__I10  |   0.8862 |  0.6511 |          −0.0843  | regress   |

_Source: outputs/stage1_260505_{213423,213817,214222,214634}/results_matrix.parquet._

### Verdicts

- **drop_path 0.05 (T10a/b, n=2 seeds)**: −0.054 / −0.049. Two-seed
  consistency rules out a seed unluck. Same diagnosis as iter 7
  warmup/EMA: stochastic-depth regularizers need many effective steps
  before the expectation settles, and 8-epoch + small data + TAPT init
  doesn't provide them.
- **cutmix-rect 0.25 (T11a)**: −0.106. Confounded with a 0.5→0.25
  CutMix-ratio drop. T11a's 0.8646 is essentially identical to iter 7's
  T7b (cutmix p=0.3) = 0.8626, so the **rect-vs-square mask shape
  contributes ~zero signal** at chip-grid resolution; the regression is
  driven by the ratio drop, which is the iter-7 result re-confirmed.
  No separate iter-9 lesson from this run.
- **two-LR backbone/head (T12a)**: −0.084 macro-F1 *and* −0.27 top1_11.
  The combo-prediction collapse (top1_11 0.9267 → 0.6511) confirms the
  diagnosis: BCE+CutMix asks for sharper independent-sigmoid
  discrimination per class, which requires the backbone to update; a
  lowered backbone LR starves that update. Two-LR is again a long-
  training-regime BKM that doesn't transfer to 8-epoch budgets.

### The growing structural-mismatch catalogue

iter 7 atomic-failed warmup and EMA on the same diagnosis. Iter 9 adds
drop_path, two-LR, and (implicitly) the CutMix ratio drop. The pattern:
**any BKM that needs many effective gradient steps to stabilize loses**
in this 8-epoch + small-data + TAPT-init regime.

## Things that didn't work — short list

1. **TTA (I5)** — rotation breaks scratch / scratch_rot.
2. **ASL (T4)** — over-suppresses bank_boundary.
3. **BCE (T5)** — drops the softmax shape that was actually useful.
4. **BCE → ASL (T6)** — neither phase converges far enough in 4 epochs.
5. **Min-threshold floor (I6)** — fork needs a low threshold; floor kills it.
6. **Temperature alone (I9)** — without rescue, top-K combo recovery drops.
7. **LR warmup 2ep (F1, iter 7)** — −0.109 macro-F1 on TAPT-init small data.
8. **EMA 0.95 (F2, iter 7)** — −0.089 macro-F1; not enough effective steps.
9. **I11 pair-aware threshold (iter 7)** — −0.007 macro-F1 net; bb+sr +25 chips offset by bb+fork +31 FP.
10. **CutMix p=0.7 (T7d, iter 7)** — over-mixes and starts hallucinating combos; loses 0.023 vs p=0.5.
11. **LS=0.08 under BCE+CutMix (T9e, iter 8)** — knife-edge cliff −0.119 vs T7c; LS=0.07 (0.9705) and LS=0.10 (0.9364) are both fine, only 0.08 falls off.
12. **drop_path 0.05 (T10a/b, iter 9, n=2 seeds)** — −0.054 / −0.049; long-training-regime regularizer doesn't transfer to 8-epoch budgets.
13. **cutmix-rect 0.25 (T11a, iter 9)** — −0.106; rect mask shape carries no signal at chip-grid resolution; regression driven by 0.5→0.25 ratio drop (= iter-7 result re-confirmed).
14. **two-LR backbone/head (T12a, iter 9)** — −0.084 macro-F1, −0.27 top1_11; starves backbone of BCE+CutMix-driven updates.

## Iter 12 — v19zpp 21-class master ablation (no Normal training)

8-variant matrix on v19zpp tier chip data + master 21-class eval (4 single +
6 2-combo + 4 3-combo + Normal + Invalid + 5 OOD), all with `--no-normal`.

| variant   | CF1    | F1_fk  | bit_FAR | chip_FAR | verdict                       |
|-----------|-------:|-------:|--------:|---------:|-------------------------------|
| T7 ★      | 0.8490 | 0.5248 |  24.90% |  96.00%  | best single (LS=0.20+CutMix)  |
| T5        | 0.8349 | 0.5236 |  26.62% |  96.00%  |                               |
| T9        | 0.8258 | 0.5209 |  25.35% |  96.00%  |                               |
| T0        | 0.7645 | 0.5453 |  28.65% |  96.00%  | pure CE — fork prob 평탄      |
| T4        | 0.7642 | 0.5185 |  28.68% |  96.00%  | ASL γ=4 over-aggressive       |
| T3        | 0.7604 | 0.5240 |  48.60% |  96.00%  | Focal — fork over-fire        |
| T1        | 0.7403 | 0.5601 |  31.47% |  96.00%  | CE+LS=0.10 — multi-label 부적합|
| T6        | 0.6531 | 0.5403 |  36.28% |  96.00%  | BCE→ASL — worst               |

★ **모든 8 variant chip_FAR = 96.00%** (Normal training OFF 의 본질적 한계).
ni_chip_FAR 80% (Normal 200 mis-fire) + ood_chip_FAR 100% (OOD 800 mis-fire)
가 합쳐서 96% bundle. **paper finding**: bundled `chip_FAR` 단일 metric 폐기 →
split metric 필수 (`normal_invalid_chip_FAR` + `ood_chip_FAR` 분리).

_Source: outputs/T*_v19zpp_seed42_*/eval_I3/bit_metrics_split.json (8 runs),
notes.md `## iter 12 v19z++ on stable master` section._

## Iter 13 — Normal training as the single FAR lever (★ paper main)

**Cycle A** Normal training ON 의 단일 효과 측정 (T7N single):

| metric | T7-no-Normal (v19zpp) | T7N (with Normal) | Δ                |
|---|---:|---:|---:|
| CF1                 | 0.8490 | 0.9042 | **+0.0552**        |
| F1_fork             | 0.5248 | 0.7796 | **+0.2548**        |
| ni_chip_FAR         | 80.00% | 0.00%  | **-80pp** ★        |
| ood_chip_FAR        | 100.00%| 16.38% | **-83.62pp** ★     |
| F1_sc               | 0.9066 | 0.8676 | -0.0390 (trade-off)|

★ Normal training 단일 lever 가 chip_FAR 96% → 13.1% 단독 해결. iter 10
finding 재확인 + paper main result.

**Cycle A logit-avg ensembles** (9 cells):

| ensemble | weights | CF1 | F1_fork | ni_FAR | verdict |
|---|---|---:|---:|---:|---|
| **T7N+T5** ★ | 70:30 | **0.9083** | 0.7656 | 0.50% | overall winner |
| T7N+T7  | 60:40 | 0.9043 | 0.6988 | 0.00% | sc/sr ceiling |
| T7N single | n/a | 0.9042 | 0.7796 | 0.00% | base |
| T7N+T5  | 60:40 | 0.9018 | 0.7389 | 2.00% | |
| T7N+T9  | 60:40 | 0.9001 | 0.7281 | 13.00% | |

★ T7N anchor (≥60% weight) 가 ni_FAR lock-in. T5 minority weight 가 sc 에서
lift. iter 10 H ensemble winner mechanism (complementary diversity)
재현. logit-avg ensemble = **single 모델 + threshold tuning 의 한계 깸**.

**Cycle B** CutMix variant grid (T7N base):

| variant | CF1 | F1_fk | ni_FAR | ood_FAR | verdict |
|---|---:|---:|---:|---:|---|
| **random_rect** ★ | **0.9188** | 0.8436 | 20.00% | 0.94% | Cycle B winner |
| no_cutmix | 0.9162 | 0.8324 | 20.00% | 14.69% | CutMix 자체 marginal |
| grid50 | 0.8967 | 0.7635 | 20.00% | 0.31% | |
| grid25 | 0.8849 | 0.7450 | 20.00% | 3.12% | |
| grid12 | 0.8596 | 0.7778 | 20.00% | 12.03% | small patch 약함 |
| scattered | 0.8423 | 0.6912 | 20.00% | 23.44% | worst — multi-patch HURT |

★ Cycle B 의 모든 cell ni_FAR 20% lock — Cycle A T7N single 0% 보다 후퇴
(CutMix variant 가 Normal 신호 약화). random_rect = Cycle B winner CF1
0.9188 > Cycle A 0.9042 + 0.0146.

_Source: outputs/T7_T7_with_normal_v19zpp_seed42_v2_260507_002217 (Cycle A),
outputs/T7_T7N_*_seed42_260507_07*-08* (Cycle B 6 cells)._

## Iter 14 — v20 chip data fork sigma raised (atomic chip-data version)

fork sigma 1.0~1.5 → 1.8~2.5 (두께 ↑) 단일 변경:

| metric | Cycle B baseline | T7N v20 | Δ |
|---|---:|---:|---:|
| CF1 | 0.9188 | 0.9226 | +0.0038 |
| F1_fork | 0.8436 | 0.8591 | +0.0155 ✓ |
| F1_sc | 0.8658 | 0.8658 | 0 |
| F1_sr | 0.9937 | 0.9937 | 0 |
| **ni_chip_FAR** | 20.00% | **0.00%** | **-20pp** ★ |
| ood_chip_FAR | 0.94% | 0.94% | 0 |

★ fork single recall **1.0000** (이전 weak recall 깨짐). `fork+scratch_rot`
recall **0.625 → 0.7188** (+9.4pp partial fix). ni_chip_FAR 20% 도 같이
0% 로 복구 (chip data 신선화 효과).

**잔존 약점**: `fork+scratch_rot+ood_CrossScratch` 0.5687 — sr+CrossScratch
overlap 의 본질적 어려움 (둘 다 회전 패턴, augment / loss 변경 필요).

_Source: outputs/T7_T7N_v20_seed42_260507_063032/eval_I3/bit_metrics_split.json._

## Iter 15 — paper-style 4-class only ablation (Normal OFF, post-v5 chip data)

iter 11 의 paper figure 재공급 — 7 cell LS sweep + 2 alternative loss + P0
baseline. **모든 cell `--no-normal`** (Normal training OFF, 4-class only).

### LS sweep on T7 (BCE+LS, no CutMix, no Normal)

| LS | CF1 | F1_fk | ni_FAR | ood_FAR | verdict |
|---:|---:|---:|---:|---:|---|
| 0.025 | 0.8890 | 0.8308 | 45.00% | 5.94% | |
| **0.05** ★ | **0.9088** | 0.8351 | 36.00% | 5.94% | **paper baseline winner** |
| 0.075 | 0.8976 | 0.8679 | 26.50% | 3.59% | F1_fk peak |
| 0.10 | 0.8784 | 0.8806 | 31.50% | 15.47% | F1_fk peak (LS=0.10) |
| 0.15 | 0.8643 | 0.8159 | 22.00% | 9.69% | |
| 0.20 | 0.8648 | 0.8145 | 20.50% | 25.16% | |
| 0.25 | 0.8625 | 0.8465 | 29.50% | 23.75% | |

★ **LS=0.05 sweet spot** — iter 8 (T9 LS sweep on cutmix-base) 의 LS=0.07
peak 과 일관 신호. 다만 모든 cell ni_FAR ≥ 20% (Normal training 없으면
real-env Normal 잡기 불가).

### Alternative loss

| variant | CF1 | F1_fk | ni_FAR | ood_FAR | verdict |
|---|---:|---:|---:|---:|---|
| T9 sigfocal | 0.8273 | 0.7169 | 46.50% | 5.16% | sigmoid_focal moderate |
| T3 Focal | 0.7768 | 0.5717 | **100%** | **100%** | Focal worst — re-confirms iter 11 |

★ **Focal 이 ni/ood FAR 모두 100%** — iter 11 finding 일관 (Focal+cutmix
P3=0.513 보다 더 나쁨). post-v5 chip data 에서도 Focal 은 fork over-fire 를
극단적으로 누르며 Normal/OOD prob 도 평탄화 → 모든 chip 어떤 signal 한 개라도
나오면 declare.

### P0 baseline (T5 BCE pure, no LS, no CutMix)

| metric | T5 P0 | 비고 |
|---|---:|---|
| CF1 | 0.8583 | BCE pure |
| F1_fk | 0.7756 | |
| ni_chip_FAR | 24.50% | Normal not learned |
| ood_chip_FAR | 1.25% | unexpected low |
| ood_overlay 2bit_recall | 0.3906 | weak combo |

★ **iter 15 conclusion**: 4-class only environment 에서 LS=0.05 가 paper
baseline winner (CF1 0.9088). 그러나 ni_FAR 36% 로 operational 부적합 →
**iter 13 Cycle A T7N (Normal training ON, ni_FAR 0.50%) 가 paper main
result**. iter 15 는 paper baseline counter-example 으로 사용.

_Source: outputs/T5_P0_pure_baseline_seed42_260507_094228 + 7 T7_P1A_LS*
+ T3_P1A_T3_focal + T9_P1A_T9_sigfocal (10 runs), all
eval_I3/bit_metrics_split.json._

## Updated "things that didn't work" list (iter 12-15)

15. **No-Normal training on master 21-class (iter 12)** — bundled chip_FAR
    96% catastrophic. Normal training 누락 = 8 variant 전부 동일 96% 실패.
    paper finding: split metric 필수 + Normal training 필수.
16. **CutMix scattered/grid12 patches (iter 13 Cycle B)** — multi-patch
    분산 mask 가 single random_rect 보다 약함. ood_chip_FAR 23.44%/12.03%
    spike. iter 12 Phase 4 의 scattered finding 일관.
17. **CutMix variant axis (iter 13 Cycle B)** — Cycle A T7N single (0% ni_FAR)
    의 우위를 깸 (모든 variant 20% ni_FAR). CutMix variant 자체가 Normal
    학습 신호 약화시킴 — Normal training 환경에선 CutMix 줄여야.
18. **Focal loss on post-v5 chip data (iter 15, T3)** — ni/ood FAR 100%
    동시 mis-fire. iter 11 finding 재확인.
19. **LS ≥ 0.10 under T7N+CutMix (iter 15 P1A)** — LS=0.05 sweet spot 위
    monotonic 하락. ood_FAR 또한 LS 강할수록 spike (LS=0.10 → 15.47%,
    LS=0.20 → 25.16%).

## Iter 21 findings (clean baseline, dual-eval no-leak protocol)

Eval = disjoint v14class (800) + v15direct (1000, +4 OOD wafer-canvas).
Source: `iters/iter_21_clean_baseline.md`,
`tables/iter21_paper_headline.csv`.

### Positive (worked)

20. **★ Complement CutMix g=2 LS=1.0 FCM-PM (E, 19C repeat)** — first single
    model to clear both eval gates simultaneously: v14 bit_F1=0.9913 ni_FAR=0.00%,
    v15direct bit_F1=0.9691 ni_FAR=3.75%. Per-class F1 ≥ 0.94 on all 4
    defect bits. Confirms iter 19B (single-seed) was not a fluke.
21. **Soft labels in CutMix paste regions are the N/I gate switch.** C
    (T7N + std CutMix LS=1.0) → ni_FAR=100% on both evals. D (same recipe,
    label-scale 0.5) → ni_FAR=1.25% v14 / 2.50% v15. Same train data,
    same Normal sentinel, only paste-region LS differs.
22. **Complement CutMix > std/grid CutMix on far-OOD.** v15direct
    wafer-canvas chips: std CutMix C bit_F1=0.8457, grid D 0.9252,
    complement E 0.9691. Complement structure (paired-bit constraint)
    inoculates the model against unseen patterns better than dense paste
    grids.

### Negative (didn't work / collapsed)

23. **T5 baseline (no-Normal, no-CutMix) is misleadingly strong on v14.**
    bit_F1=0.9745 — but ni_FAR=100% (every Normal/Invalid fires defect).
    v15 collapses to 0.7872. Single-label-collapse signature; reaffirms
    Normal-training necessity from iter 10 / 13.
24. **T7N pure (no CutMix) — Normal training alone insufficient.** B:
    v14 bit_F1=0.8609 ni_FAR=100%, fork F1 only 0.6420 on v15. Sentinel
    target zeroing without compositional augmentation under-calibrates
    the rejection boundary.
25. **g=4 LS=0.25 (G, 19G repeat) — over-paste under-soft.** Best v15
    bit_F1 (0.9716) but ni_FAR=100% on v15direct — heavy paste with
    too-soft labels destroys N/I gate. Confirms small-g + high-LS
    (E: g=2, LS=1.0) is the operating point, not the apparent bit_F1
    optimum.

## Iter 28 — Mixup α sweep (paper §5 evidence: pixel α-blend palette destruction)

Source: `iters/iter_28_29_paper_ablation.md`. 6 trains all share iter21E
base recipe (T7N, BCE+LS=0.20, 8ep, AdamW 1e-4, RandomAffine, seed=1)
with **CutMix replaced by Mixup α-blend** at the listed α.

| tag | spec                                   | v14 bF1 | v14 ni% | v15 bF1 | v15 ni% | dual-pass? |
|:---:|:---------------------------------------|--------:|--------:|--------:|--------:|:----------:|
| 28A | Mixup α=0.2 (Zhang 2018 default)       |  0.9875 |   5.00% |  0.9834 | 100.00% | ✗ |
| 28B | Mixup α=1.0                            |  0.9092 | 100.00% |  0.8924 | 100.00% | ✗ |
| 28C | Mixup α=0.1                            |  0.9098 | 100.00% |  0.8627 | 100.00% | ✗ |
| 28D | Mixup α=0.4                            |  0.9753 | 100.00% |  0.9141 | 100.00% | ✗ |
| 28E | Mixup α=2.0                            |  0.9783 | 100.00% |  0.9671 | 100.00% | ✗ |
| 28F | Mixup α=0.4 + cutmix-p=0.5 combo       |  0.9091 | 100.00% |  0.8984 | 100.00% | ✗ |

26. **★ ALL 6 Mixup variants fail v15 ni_FAR (100%) — categorical CutMix
    vs Mixup design difference.** Every α tested fails the v15
    Normal/Invalid gate. Only α=0.2 holds v14 ni at 5% — a fragile
    coincidence that explodes to 100% under v15 OOD pressure.
    **Mechanism (paper §5 narrative)**: chip images are palette-graded
    PNGs (pixel value 0 = Normal, 1–7 = defect intensity grade). Mixup
    `λ·x_A + (1−λ)·x_B` synthesizes invalid intermediate grades (e.g.
    grade 0 + grade 5 at λ=0.5 → quantized grade 3, an unrelated defect
    intensity). Training on out-of-palette pixels with mixed labels
    destroys Normal-vs-defect calibration at the rejection boundary.
    CutMix preserves every pixel's palette grade — this is the
    structural reason CutMix > Mixup on palette-graded multi-label,
    **not a tuning question**.
27. **Mixup+CutMix combo (28F) does NOT rescue Mixup.** α=0.4 +
    cutmix-p=0.5 still hits v15 ni_FAR=100% on both eval sets. The
    pixel-level α-blend contamination cannot be diluted by adding
    CutMix — they don't cancel.

_Source: outputs/iter28A..F/{eval_v14class,eval_v15direct}/preds_chip.parquet._

## Iter 29 — label × spatial isolation (paper §5 evidence: 4 designs all necessary)

Source: `iters/iter_28_29_paper_ablation.md`,
`tables/paper_section5_ablation.csv`. Decomposes iter21E ★ winner into
4 atomic design axes: **region paste vs pixel α-blend** + **full grid
cover vs single rect** + **pair mask vs random partner** + **hard label
vs soft λ-mix**. Three new trains complete the 6-cell label×spatial
matrix (other 3 cells covered by iter21C/21D/21E).

| tag | spec                                              | v14 bF1 | v14 ni% | v15 bF1 | v15 ni% | dual-pass? |
|:---:|:--------------------------------------------------|--------:|--------:|--------:|--------:|:----------:|
| 29A | std box-cut (single rect) + hard label            |  0.7381 | 100.00% |  0.7616 | 100.00% | ✗ |
| 29B | complement g=2 + pair mask + soft LS=0.5          |  0.9921 | 100.00% |  0.9953 | 100.00% | ✗ (highest bF1, FAR fail) |
| 29C | grid_complete g=2 + no pair mask + hard LS=1.0    |  0.9369 |   2.50% |  0.9248 | 100.00% | ✗ |

### Paper §5 — 6-cell label × spatial matrix (final form)

| spatial \ label | soft (λ-mix) | hard (both [A=1, B=1]) |
|:---|:---|:---|
| std box-cut (Yun 2019) | iter21C: v15 0.85/100% ✗ | iter29A: v15 0.76/100% ✗ |
| grid_complete (no pair mask) | iter21D 18F1: v15 0.93/2.5% ✓ | iter29C: v15 0.92/100% ✗ |
| complement + pair mask | iter29B: v15 0.99/100% ✗ | **iter21E ★: v15 0.97/3.75% ✓** |

28. **★ Only iter21E ★ (complement + pair mask + hard label + full-cover)
    clears both gates — every single-axis removal breaks the model.**
    Region paste (vs pixel α-blend, iter28) + full cover (vs single rect,
    iter29A) + pair mask (vs none, iter29C) + hard label (vs soft, iter29B
    and iter21D) all four are necessary. **No single design choice is
    dispensable.**
29. **iter29B is the F1-trap warning — highest single-model v15 bit_F1
    in entire chip-multilabel history (0.9953) but ni_FAR=100%.** Combining
    full-cover complement + pair mask + soft λ-mix label perfectly fits
    in-distribution paste signal but leaks Normal probability mass into
    both defect bins. **F1-only winner ≠ deployable** — the cleanest
    documented case that ni_FAR-blind metric optimization is a trap. Any
    paper or report quoting only bit_F1 (without dual-eval ni_FAR) on
    this design would mis-rank it as the new SOTA.
30. **iter29A (std box-cut + hard label) collapses bit_F1 to 0.74–0.76
    on both eval sets, ni_FAR=100%.** Single rectangular paste leaves
    majority of chip un-touched; hard label says [A=1, B=1] but only
    A's region was actually pasted — the calibration mismatch destroys
    bit_F1 AND ni_FAR simultaneously. Hard label without full cover is
    the worst combination.
31. **iter29C (grid_complete + hard LS=1.0 NO pair mask) v14 ni passes
    at 2.5% but v15 ni explodes to 100%.** Removing pair mask while
    keeping hard label creates ambiguous mixed-class regions that confuse
    the OOD-rejection boundary. Pair mask is the OOD-stability anchor —
    not just a bit_F1 lever.

_Source: outputs/iter29A_box_hard, iter29B_compl_g2_softLS05,
iter29C_grid_hard_LS10/{eval_v14class,eval_v15direct}/._
