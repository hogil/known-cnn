# Diary — 2026-05-05 (initial day)

A first-day journal of the chip multi-label experiments. Five iterations
ran in a single working session. Numbers are reported as they were
captured, in chronological order, with the small reasoning beats that
connected each iter to the next.

## 16:28 — Iter 1: stage 1 baseline

Took the existing `chip5_round4_v14_260505_061558_running` checkpoint
(single-label CE, 5 classes) and applied 6 inference variants on the
2200-chip 11-class eval set without retraining.

The argmax @0.5 baseline (T0__I0) landed at macro-F1 **0.7302**. The
per-class F1-max threshold variant (I1) jumped to **0.8444** — already
+0.114 from a single inference change. The winner was T0__I3
(F1-max + top-K rescue) at **0.8466**.

Three things stood out:
1. Per-class F1-max thresholds were absurdly far from 0.5: fork at
   **0.1195**, bank_boundary at 0.4994, scratch at 0.7682, scratch_rot
   at 0.8355. The single-label CE model has wildly miscalibrated
   per-class scores when re-purposed for multi-label.
2. Temperature scaling (I4) gave identical macro-F1 to I3 — the F1-max
   sweep absorbed the rescaling. ECE dropped 0.0778 → 0.0129, useful
   only for honest probabilities.
3. **TTA (I5) regressed by 0.018.** Rotation conflated `scratch` ↔
   `scratch_rot`. Permanently disallowed.

Top error mode: 277 false-positive-fork. fork's threshold of 0.1195
was so low that fork fired on roughly half of all non-fork chips.

## 16:54 — Iter 2: extended inference variants

Added I6 (min-floor 0.30), I7 (Δ=0.02 step-search), I8 (top-2 margin),
I9 (per-class T). Re-ran I0-I4 for sanity.

Winner: **T0__I7 = 0.8485** (+0.002 over I3). Tiny gain mostly from
fork's threshold lifting 0.1195 → 0.1400.

Significant losses:
- I6 (min-floor) -0.029. fork *needs* a low threshold; the floor
  killed 12% of fork recall.
- I9 (per-class T) -0.072. L-BFGS on small val data gave unstable
  per-class temperature optima.

Diagnosis: inference-side ceiling near 0.85. 0.85 is what the frozen
backbone can produce *without* relearning the logit distribution.

## 17:08 — Iter 3: I10 entropy-based Normal gate

The single-label model has no positive supervision for `Normal`.
Hypothesis: declare Normal when softmax entropy exceeds 0.85·log(C),
i.e. the model is *confidently uncertain*. Hard-coded the constant.

Result: **T0__I10 = 0.8542** (+0.006 over I7).

The error breakdown was the more interesting story than the F1
number:
- `missed_normal` 160 → 106 (-34%) — the gate caught fork-on-Normal.
- `wrong_combo` 292 → 273 (-19) — bonus from confused multi-class
  noise being absorbed as Normal.
- `false_positive_fork` 215 → 215 (no change) — fork-confident-on-other
  chips don't trigger high entropy.
- `wrong_normal_entropy` 0 → 19 — the gate's new false-positive,
  small relative to the Normal recovery.

The macro-F1 jump (+0.006) was small but the top1_11 jump (+0.031)
was much bigger — Normal had been completely missing from the
diagonal of the 11-class confusion matrix; now it had mass.

Ceiling reached: I10 cannot help with the 215 single-confident fork
FPs. To break those we have to retrain.

## 17:01-17:41 — Iter 4: stage 2 retraining

Trained four loss recipes on the same 327-chip dataset with the same
backbone topology: T1 (CE+LS=0.10), T4 (ASL γ_+=1 γ_-=4 m=0.05), T5
(BCE), T6 (BCE→ASL curriculum 4ep+4ep). Each ~340-720 sec on the
4090. All four hit val-acc 1.0 in 1-2 epochs.

The Stage 2 main run dispatched before I10 was added to the variant
list, so we re-ran each of T1-T6 with I10 separately (procedural bug
documented).

**Winner: T1__I10 = 0.8634** (+0.009 over T0__I10).

Surprises:
- **Only T1 helped.** T4 (-0.078), T5 (-0.095), T6 (-0.035). The
  multi-label-native losses all *regressed*.
- fork F1 jumped 0.6607 → 0.7426 at T1__I10. Precision nearly
  doubled (0.5360 → 0.7014) at the cost of some recall (0.8609 →
  0.7891). LS softened the dominant logit and lifted fork's
  threshold from 0.14 to 0.22.
- I10 won at every train cell — the entropy gate generalised.
- Total errors dropped 701 → 527 (-25%) vs iter 1.

Hypothesis for T4/T5/T6 underperformance: small-data + strong TAPT
init + 8 epochs is not enough to rebuild useful asymmetry under
ASL/BCE; the existing softmax structure was load-bearing for the
threshold decoder.

## 17:51-18:30 — Iter 5: Phase A1 LS sweep

T1 won at α=0.10 — was that the right α? Swept α ∈ {0.05, 0.10, 0.15,
0.20, 0.25, 0.30, 0.35} at LR=1e-4, ep=8 fixed, three inferences each.

Curve:
```
α    | best macro_f1
0.05 | 0.7964
0.10 | 0.8363
0.15 | 0.8961
0.20 | 0.9268   ← peak
0.25 | 0.8663
0.30 | 0.8185
0.35 | 0.7279
```

**Winner: T1_LS20__I7 = 0.9268** (+0.0634 over iter 4 winner). top1_11
= 0.8449 (+0.144).

The size of the +0.0905 gap between α=0.10 (literature default) and
α=0.20 (our optimum) was the most striking number of the day. The
default was simply wrong for our regime.

A second surprise: the inference winner *flipped* at α=0.20. I10 had
won every prior iter; at α=0.20 it lost to I7 by 0.04. Mechanism:
LS raises softmax entropy on every prediction, and I10's hard cutoff
(0.85·log C) starts triggering on real-defect chips. The fix is to
re-tune the gate threshold per α; we did not run it because the
LS sweep budget was committed. Phase G to follow.

Single-label val accuracy turned out to be a bad selector for
multi-label macro-F1: T1_LS25 hit val 1.0 but multi-label only 0.8663;
T1_LS20 hit val 0.9756 and multi-label 0.9268.

A2 LR sweep (now done) confirmed LR=1e-4 as best. LR=3e-4 + LS=0.20
was catastrophic (0.4155) — gradient explosion destroyed the TAPT
init. A3 epochs sweep is in flight at end-of-day.

## End of day

Cumulative: argmax 0.7302 → final 0.9268, +0.1966 macro-F1, in ~115
GPU-minutes. The single biggest individual jump was per-class F1-max
thresholds (+0.1142). The single biggest "free" win was tuning the
LS hyperparameter (+0.0905 vs the literature default).

Three Stage-2 phases (B = ASL, C = Focal, D = BCE pos_weight) and
two synthesis variations (strong-defect filter, grade-elevated chips)
queued for tomorrow. Notes pinned in `chip_multilabel/notes.md`
section 250-263.

Open question for next session: does the same "default-was-wrong"
pattern hold for ASL γ_-? If yes, Phase B should produce a similar
+0.06–0.09 jump from a hyperparameter sweep alone.
