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
