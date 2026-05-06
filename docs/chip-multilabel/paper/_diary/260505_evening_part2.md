# Diary — 2026-05-05 (evening, part 2)

Continues from `260505_evening.md`. Covers Phase F (warmup / EMA) +
I11 + the T7 CutMix multi-source sweep that became the iter 6
headline.

## ~19:25 — Phase F dispatch

Phase A (LS sweep + LR + epochs) is closed at `T1_LS20__I7 = 0.9268`.
Next thought: are there easy wins from importing best-known-methods
from the sister anomaly-detection chart? Two candidates that
consistently lift anomaly-detection from-scratch trains:
- F1: 2-epoch warmup (start_LR=0.05·peak) + cosine to η_min=1e-6.
- F2: EMA(0.95) on weights, eval the EMA copy.

Dispatched both at the T1_LS20 LR=1e-4 ep=8 recipe.
- F1 → `outputs/stage1_260505_192541/`
- F2 → `outputs/stage1_260505_194014/`
- T1 reference replay → `outputs/stage1_260505_192733/` (sanity
  check that today's launcher still gives 0.9268; it does).

## ~19:35 — Phase F results

Both negative.

```
T1 (ref)                | macro_f1 = 0.9268    top1_11 = 0.8449
F1 warmup + cosine      | macro_f1 = 0.8181    top1_11 = 0.5540    Δ = −0.1087
F2 EMA(0.95)            | macro_f1 = 0.8377    top1_11 = 0.6602    Δ = −0.0891
```

Both regressions are larger than any single previous iter's gain
(except iter 1's I0→I3 jump). This is not noise. Reading:
- **F1 (warmup):** with 8 total epochs and start_LR=5e-6, the first
  2 epochs barely move the head off TAPT init. Cosine tail spends
  the last few epochs at η_min=1e-6 which is also too low. Effective
  training reduces to ≈3 epochs of useful gradient — LS=0.20's
  optimum is structurally not reached.
- **F2 (EMA):** train_n=327, batch=32, ~12 optimiser steps/epoch, 8
  epochs ⇒ ~96 useful steps. EMA(0.95) has averaging window
  1/(1−0.95) = 20 steps ≈ 20% of training. Oversmooths the
  late-epoch sharpening that LS=0.20 needs.

The lesson is the regime-mismatch one: BKMs from another domain
(training-from-scratch, abundant data, vanilla CE) do not transfer
to small-data + strong-TAPT + tuned-LS. Captured in §7.4.3.

Next analyst notes that the *real* residual problem under T1+I7 is
the bb+scratch_rot combo. Pull the confusion to confirm: 52/160
correct = 0.3250 recall. The wrong predictions land on
`scratch_rot` (43), `fork+scratch_rot` (36), `bank_boundary` (15).

## ~19:44 — I11 attempt (no-retrain band-aid)

Quick test: can we rescue bb+sr recall without retraining? Add a
pair-aware threshold-relaxation rule (I11): when `s_{c1} ≥ θ_{c1}`
and `s_{c2} ∈ [0.7·θ_{c2}, θ_{c2})`, assert both.

`outputs/stage1_260505_194443/`:
```
T0__I7   | 0.9268    bb+sr recall 0.3250
T0__I11  | 0.9199    bb+sr recall 0.4812    Δ recall +0.156
```

Recall is genuinely up. macro-F1 down 0.0070. Why? Same rule
applies to *every* defect pair — 31 spurious bank_boundary+fork
false positives appear because fork's low threshold + a coincident
borderline bank_boundary sigmoid trigger the relaxation on chips
that should be single-bank_boundary.

I11 is a band-aid. The real diagnosis is that bb+sr's *combined
logit* never reaches threshold under I7 because the model has zero
training-time supervision on combo outputs. Reframe as a training
data-distribution problem.

## ~19:51 — T7 launch (analyst recommends multi-source CutMix)

T7 idea: BCE + LS=0.20 + per-batch CutMix between distinct
TRAIN_CLASSES with `p` per-batch. Patch-area `λ ~ U[0,1]`. Multi-hot
target = `λ·y_A + (1−λ)·y_B`. BCE loss because CE can't handle
mixed targets. Citations queued: Yun 2019 (CutMix), Wightman 2021
(ResNet strikes back), Wang 2024 SpliceMix, Wang 2024 cutmix
multi-label label propagation.

First T7 = T7c at `p=0.5`. Wall-clock identical to T1 (CutMix is
~free at training time). Dispatched 19:51, finishes 19:57.

## ~19:57 — T7c result (FIRST READ)

`outputs/stage1_260505_195730/`:
```
T7c__I3   | 0.9050    top1_11 = 0.7545
T7c__I7   | 0.9035    top1_11 = 0.7432
T7c__I10  | 0.9271    top1_11 = 0.8307    bb+sr recall 0.9562
T7c__I11  | 0.8945    top1_11 = 0.7307
```

bb+sr recall is **153/160 = 0.9562**. T1 was 52/160 = 0.3250.
**+0.6312 absolute recall gain.**

macro-F1 is 0.9271, tied with T1's 0.9268. The headline number
barely moves but the underlying combo decoding is dramatically
healthier.

Also: inference winner shifts from I7 (under T1) to I10 (under T7c).
This is direct re-confirmation of §6.2.1's logit-sharpness
hypothesis on a third axis (loss / augmentation). Captured in
§6.6.6 / §7.4.1.

## ~20:00 — Atomic decomposition planning

T7c has two atomic deltas vs T1:
1. CE → BCE (loss switch)
2. + CutMix p=0.5 (augmentation add)

Need to isolate. Run T7a = BCE + LS=0.20 + p=0.0 (BCE only, no
mixing).

## ~20:08 — T7a result

`outputs/stage1_260505_200523/`:
```
T7a__I3   | 0.8577    top1_11 = 0.5534    bb+sr recall 0.5125
T7a__I7   | 0.8516    top1_11 = 0.5386
T7a__I10  | 0.8364    top1_11 = 0.5199
T7a__I11  | 0.8516    top1_11 = 0.5386
```

T7a regresses by **−0.0691** macro-F1 vs T1.

The decomposition is now clean:
- CE → BCE alone: −0.0691 macro-F1 (T7a vs T1)
- BCE → BCE + CutMix p=0.5: +0.0694 macro-F1 (T7c vs T7a)
- Net: +0.0003 macro-F1 (T7c ≈ T1 on macro-F1)

But on bb+sr recall:
- CE → BCE: +0.1875 (0.3250 → 0.5125)
- BCE → BCE + CutMix: +0.4437 (0.5125 → 0.9562)
- Net: +0.6312

The **macro-F1 gain comes solely from CutMix**, not from the loss
switch — the BCE switch is a structural prerequisite (CutMix
targets are mixed) but is itself a macro-F1 cost. CutMix is the
load-bearing augmentation.

This is the cleanest atomic decomposition in the paper so far.

## ~20:13 — T7d (cutmix p=0.7) — does more mixing help?

`outputs/stage1_260505_201706/`:
```
T7d__I10  | 0.9038    top1_11 = 0.7432    bb+sr recall 0.9562
```

bb+sr recall held at 0.9562 (saturated). macro-F1 drops 0.0233 vs
T7c. top1_11 drops 0.088 — the model loses single-class identity
because too few clean-single-defect batches survive.

The error pattern at fixed bb+sr recall is *different*: T7d's 7
errors land on `fork+scratch_rot` (over-mixing makes the model
guess combo even when the actual second class is wrong), whereas
T7c's 7 errors land on the singleton `scratch_rot`.

## ~20:33 — T7b (cutmix p=0.3) — does less mixing help?

`outputs/stage1_260505_203340/`:
```
T7b__I10  | 0.8626    top1_11 = 0.5511    bb+sr recall 0.7312
```

Far worse than T7c. p=0.3 fires CutMix on only 30% of batches —
not enough combo gradient to overcome the BCE-only per-class
threshold collapse.

## CutMix-p sweep summary

```
p   | macro_f1 | bb+sr recall | top1_11 | best inference
0.0 |  0.8577  |    0.5125    |  0.5534 | I3   (T7a)
0.3 |  0.8626  |    0.7312    |  0.5511 | I10  (T7b)
0.5 |  0.9271  |    0.9562    |  0.8307 | I10  (T7c)  ← peak
0.7 |  0.9038  |    0.9562    |  0.7432 | I10  (T7d)
```

Sharp peak at p=0.5. ±0.2 in p costs 0.02–0.06 macro-F1.

## Reading

p=0.5 balances clean and mixed batches roughly equally (≈half each
in expectation), giving the model both single-class and combo-class
gradient signal. p<0.5 starves combo regime; p>0.5 starves
single-class regime.

## Updated paper sections

- **abstract.md** — refresh headline: T7c bb+sr 0.32→0.96 +0.6312
  highlight, atomic decomposition framing.
- **03_data.md** — new §3.7 on synthesis-side intervention (CutMix
  at training time), distinct from §3.2 eval-set min-blend.
- **05_experiments.md** — new §5.6 covering iter 6 in three
  sub-iters (Phase F negative, I11 rejected, T7 sweep). New §5.7
  cross-iter timeline including iter 6.
- **06_analysis.md** — new §6.6 "BCE penalty vs CutMix gain" with
  the atomic decomposition table, p sweep reading, and
  inference-winner-shift narrative re-confirming §6.2.1. §6.7
  computational cost updated.
- **07_discussion.md** — §7.4.1 three-axis evidence summary;
  §7.4.2 T7a outlier and threshold-collapse caveat to the
  hypothesis; §7.4.3 BKM-transfer-is-regime-dependent
  (Phase F negative result). §7.6 limits updated for T7c (5 items
  including top1_11 trade-off). §7.7 narrative updated for 6
  iterations.

## Why this matters

iter 6 produces three distinct contributions:

1. **The headline result** — T7c lifts bb+sr recall by +0.6312
   absolute, the largest single-combo-class lift in the project,
   while keeping macro-F1 within +0.0003 of T1.

2. **A clean atomic decomposition** — CE→BCE costs −0.069, +CutMix
   recovers +0.069. The two deltas of equal magnitude let us
   identify CutMix as the load-bearing element of T7c.

3. **A third axis on the §6.2.1 / §7.4 logit-sharpness
   hypothesis** — T7c's I10-winner (vs T1's I7-winner) extends the
   hypothesis from LS / epochs axes to the loss/augmentation axis,
   strengthening the unified claim.

4. **A negative-result lesson** — Phase F (warmup, EMA) hardens the
   §7.1 default-trap and §7.4.3 regime-mismatch narrative. Both
   F1 and F2 are well-known anomaly-detection BKMs that fail
   reliably in our regime.

Next iteration questions:
- Phase G: does CutMix-p sweet spot drift under different LS?
- Does T7c + Phase B's ASL γ sweep stack? (T7+ASL is queued.)
- Does the §7.4 hypothesis predict the inference winner from a
  scalar entropy proxy? Need to log mean softmax entropy on val
  for every checkpoint going forward.

End of part 2. Iter 6 closed at T7c__I10 = 0.9271, bb+sr 0.9562.
