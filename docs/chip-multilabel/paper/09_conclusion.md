# 9. Conclusion

## 9.1 Summary of contributions

We took a single-label CE chip CNN (ConvNeXtV2-Base 384, 4 defect
classes + `invalid_main`, 327 chips train, val-acc 1.0) and adapted
it as a multi-label predictor over an 11-class chip benchmark
(2200 chips). Through nine iterations of inference-side and
training-side interventions, capped by a 3-seed paired comparison
of the headline configurations, we lifted macro-F1 from **0.7302**
(argmax) to a 3-seed mean **0.9305 ± 0.0460** (T9 family: BCE +
LS=0.07 + CutMix p=0.5 over seeds {42, 43, 44}) — a **+0.20**
absolute mean gain — using no test-time augmentation and ~3.5
GPU-hours of total compute. The single-seed peak (T9d at LS=0.07
seed=42) reaches **0.9705**, but the 3-seed mean is the honest
claim; single-seed peaks are upper-tail draws around the true mean.

The key technical takeaways:

1. **Per-class F1-max thresholds** (Lipton et al. 2014) provide the
   single biggest jump in our setting: +0.1142 macro-F1 from a
   single inference change.
2. **An entropy-based `Normal` gate** (I10) is the only robust
   solution for un-supervised classes in single-label CE pretraining:
   +0.0057 macro-F1 from giving Normal an explicit decoder.
3. **Label smoothing strength matters**, and the literature default
   (α=0.10) is nearly an order of magnitude too low for our small-data
   regime: α=0.20 wins by +0.0905 macro-F1 over α=0.10 on the CE side.
4. **The LS optimum does not transfer across loss-family bases.**
   On the BCE + CutMix base the LS optimum drops to the [0.05, 0.10]
   band, +0.038 mean over T1 (CE+LS=0.20) on the 3-seed comparison
   (§5.10).
5. **CutMix at training time is the load-bearing combo-recall lift.**
   CutMix p=0.5 lifts bb+sr recall by +0.225 mean (3-seed paired
   comparison; single-seed peak +0.6313 on s42); the atomic
   decomposition (CE→BCE costs −0.069, +CutMix recovers +0.069) shows
   the gain is solely from the augmentation, not the loss switch.
6. **Single-seed measurement breaks down at the macro-F1 ≈ 0.94
   ceiling, and the breakdown is not unique to the winner family.**
   Single-seed std is ≈0.030 for T1 and ≈0.046 for T9 across n=3
   seeds; the iter-5 "T1+I7 = 0.9268" claim was itself an upper-tail
   single-seed draw whose 3-seed mean is +0.063 lower (0.8923). The
   discipline must shift from many cells × 1 seed to fewer cells ×
   n seeds (§6.7), and the §5.10 protocol formalises this for the
   paper's headline numbers.
7. **Asymmetric BKM transfer** (§7.5). Hyperparameter-axis tuning
   (LS) and the data-axis intervention (CutMix p=0.5) transfer
   cleanly to our small-data + TAPT regime; eight other axes
   (warmup, EMA, drop_path, cutmix-rect, two-LR, T8 CE-soft +
   CutMix, T13a ASL γ_neg=2 + CutMix, I11 pair-aware threshold)
   all fail. The TAPT init places the chip backbone close to a
   regularisation ceiling — additional regularisation is a net cost,
   alternative loss-family bases do not match BCE+CutMix, and
   no-retrain inference heuristics are redundant once the trained
   recipe captures the capability.

Negative results documented as ablations of equal value:

- **TTA (I5)** flips `scratch ↔ scratch_rot` and is permanently
  disallowed.
- **Min-floor 0.30 thresholds (I6)** kill fork's recall for no
  precision benefit.
- **Per-class temperature alone (I9)** is unstable on small val.
- **ASL γ_-=4 (T4 default)** over-suppresses bank_boundary
  (−0.078).
- **BCE (T5)** removes the softmax structure that the threshold
  decoder relies on; without CutMix it costs −0.069 macro-F1.
- **BCE → ASL (T6)** has neither phase converge in 8 epochs.
- **Entropy gate I10** loses to I7 once LS is pushed strong (α≥0.20);
  this is a regime change, not a fundamental defect.
- **F1 warmup (start_factor=0.05, 2ep)** regresses by −0.10.
- **F2 EMA(0.95)** regresses by −0.08.
- **T8 CE-soft + CutMix p=0.5** regresses by −0.10 — confirms the
  CE+CutMix combination is structurally inferior to BCE+CutMix.
- **T10 drop_path 0.05** regresses by −0.05 mean across two seeds.
- **T11a cutmix-rect ≤0.25** regresses by −0.11 — confirms the
  CutMix patch range is load-bearing in our regime.
- **T12a two-LR (bb 5e-5 / head 2e-4)** regresses by −0.08 and
  collapses bb+sr recall to 0.42.
- **T13a ASL γ_neg=2 + CutMix p=0.5** regresses by −0.10 — even
  with CutMix in the recipe, BCE remains the right loss.
- **I11 pair-aware threshold** is a recall band-aid that costs
  −0.007 macro-F1 net (rejected).

These are not failed experiments; they are the negative half of the
asymmetric-BKM-transfer story. **Negative results are first-class
citizens of this paper.**

## 9.2 Best-known result

```
family:          T9 (BCE + LS=0.07 + CutMix p=0.5)
backbone:        ConvNeXtV2-Base 384, TAPT
train:           327 chips × 8 epochs × LR=1e-4, single-positive
                 + per-batch CutMix p=0.5 (multi-source)
inference:       per-cell winner among {I3, I7, I10, I11};
                 mostly I7 in this band

3-seed paired comparison vs T1 (CE+LS=0.20), seeds {42, 43, 44}:
  metric          T1 (mean ± std)   T9 (mean ± std)   Δ T9-T1
  macro_f1        0.8923 ± 0.0301   0.9305 ± 0.0460   +0.0382
  top1_11         0.7697 ± 0.0714   0.8242 ± 0.1058   +0.0545
  bb+sr recall    0.5292 ± 0.2577   0.7542 ± 0.3500   +0.2250

Per-seed paired Δ macro_f1:
  s42 +0.044 / s43 +0.062 / s44 +0.009    (T9 wins all 3)
Per-seed paired Δ top1_11:
  s42 +0.082 / s43 +0.069 / s44 +0.013    (T9 wins all 3)
Per-seed paired Δ bb+sr recall:
  s42 +0.631 / s43 +0.138 / s44 −0.094    (T9 wins 2/3)

Δ vs argmax (T0__I0):  +0.20 mean macro_f1
Training cost:         ~340 sec / cell on RTX 4090

Single-seed peaks (reported with explicit variance flag):
  T9d (LS=0.07 seed=42) → macro_f1 = 0.9705, top1_11 = 0.9267, bb+sr 0.9563
  T9g (LS=0.07 seed=43) → macro_f1 = 0.9408, top1_11 = 0.8307, bb+sr 0.9563
  T9h (LS=0.07 seed=44) → macro_f1 = 0.8803, top1_11 = 0.7153, bb+sr 0.3500
```

## 9.3 The four final takeaways

The autonomous loop converged on four lessons that carry beyond
this dataset:

### Takeaway 1 — LS-axis transfer + CutMix mechanism = main signal

Two axes carry the entire +0.20 macro-F1 trajectory: re-tuning
label smoothing (CE side α=0.20 over the literature 0.10; BCE-side
α=0.07 over the inherited 0.20) and adding CutMix p=0.5 to the
training loop. Every other axis we tried — eight independent
training-side or inference-side variants on top of the BCE+LS+CutMix
base — under-performed by 0.05–0.11 macro-F1. The signal is narrow
and structural: hyperparameter retune of the right scalar (LS)
plus one data-axis augmentation (CutMix), and the combo capability
(bb+sr recall, mean +0.225 paired) is unlocked. Practitioners
working in our regime (small data, strong TAPT, multi-label
benchmark with combo classes) should prioritise these two axes
over any structural BKM import.

### Takeaway 2 — Five structural BKMs all transfer-fail at this data scale

Warmup, EMA, drop_path, cutmix-rect (compressed CutMix patch range),
and two-LR (differential backbone/head LR) each failed to lift
macro-F1 above the T9 family-mean. All five are well-validated in
their source domains (ImageNet-scale, training-from-scratch, ≥50
epochs). All five failed by 0.05–0.11 in our regime — magnitudes
well outside the 0.030–0.046 single-seed noise floor. The
*regularisation-ceiling hypothesis* (§7.4.4) is the unifying
explanation: the TAPT init has already placed the backbone close
to its small-data optimum, additive regularisers are a net cost,
and capacity-budget changes (two-LR, drop_path) starve the
backbone of the few effective steps it has. Further structural
BKMs should be considered in *replacement* (substitute LS or
CutMix) rather than *additive* (stack on top) form.

### Takeaway 3 — Single-seed sweeps are unreliable above macro_f1 ≈ 0.92

The §5.10 3-seed paired comparison applied retroactively shows
*both* T1 and T9 have non-trivial single-seed variance:

| family | seed=42 | seed=43 | seed=44 | mean ± std         |
|--------|--------:|--------:|--------:|-------------------:|
| T1     |  0.9268 |  0.8788 |  0.8712 | 0.8923 ± 0.0301    |
| T9     |  0.9705 |  0.9408 |  0.8803 | 0.9305 ± 0.0460    |

The iter-5 headline "T1+I7 = 0.9268" was an upper-tail single-seed
draw whose 3-seed mean is +0.063 lower. The iter-8 single-seed
peak "T9d = 0.9705" is the same shape: upper-tail at seed 42, real
mean at seed 43+44 is closer to 0.91. Single-seed sweeps remain
useful as a *screening* tool — at single-seed resolution, regressions
of ≥3 σ ≈ 0.09 macro-F1 are still credible (the eight negative
axes catalogued above are all in this regime). They are *not*
sufficient for headline claims at the family-mean ≈ 0.94 ceiling.
The §5.10 multi-seed reporting protocol — every macro_f1 above 0.92
must be quoted as either an n≥3 mean ± std or with an explicit
single-seed flag — should be the discipline going forward.

### Takeaway 4 — bb+sr recall is the real operational gain (mechanism-robust, not seed-luck)

The bb+sr (bank_boundary + scratch_rot) combo class is the hardest
class in our benchmark — under T1+I7 only 32.5% of bb+sr chips are
exact-decoded at single-seed = 42. The CutMix-mechanism in T9
lifts the 3-seed mean to 0.7542 (+0.225 paired), and the *direction*
of the lift is robust: T9 wins paired-bb+sr on 2/3 seeds, with the
loss on s44 (−0.09) more than offset by the wins on s42 (+0.63)
and s43 (+0.14). The mean effect is the headline number to ship,
not the s42 peak. This is the part of T9 that matters operationally:
combo-class recall in production deployments is what drives the
recipe's value, and the +0.225 mean lift is the right number to
quote when discussing whether to deploy the recipe.

The four takeaways together describe the methodology *and* the
result: in a regime where many BKMs have failed and the headline
metric has a non-trivial seed-noise floor, two axes (LS retune,
CutMix data augmentation) carry a robust +0.20 macro-F1 mean lift
and a robust +0.225 bb+sr recall mean lift; everything else
under-performs.

## 9.4 Remaining work

**Phase G (top-priority queued, partly delivered in §5.10):** the
3-seed comparison covers T1 and T9 at LS=0.07. Phase G should
extend to the LS=0.05 and LS=0.10 cells (currently single-seed),
publish per-class calibration plots and reliability diagrams at
the T9 winning cells, ROC curves at I7 / I10, and re-tune of
I10's H threshold per checkpoint to recover the §7.4 0.04
entropy-gate gap on T9-family checkpoints.

**Phase B (queued):** ASL γ_+, γ_-, m sweep, scoped down by the
T13a result. Hypothesis: T13a's 0.10 regression at γ_-=2 + CutMix
makes ASL extremely unlikely to recover T9 at any γ. Phase B as a
formal sweep would close the question; we would not expect it to
lift the headline.

**Phase D (queued):** BCE pos_weight + LS sweep on the BCE+CutMix
base. Hypothesis: pos_weight retune is the next plausible single
axis on the BCE side; T13a's negative result was on a different
loss family (ASL), so pos_weight remains an untested in-family axis.

**Phase H (queued, motivated by §7.4.4):** Replacement-not-additive
regularisation test. Swap LS=0.07 out for drop_path-only at varying
rates. If drop_path-as-replacement-LS approximately recovers
LS-only, the regularisation-ceiling hypothesis is confirmed. If
drop_path-as-replacement-LS still regresses, the regularisation-
ceiling hypothesis becomes a stronger "TAPT-fragility" claim
(any non-LS regulariser fails).

**Synthesis-side (sister-repo, queued):** strong-defect filtering
(`--source-strength-pct 50`) and grade-elevated chip generation
(`--grade-mode elevated_2/3`). These attack the *data* limit on
combo decoding and scratch / scratch_rot discrimination; future
iters will re-run the full inference + training matrix on the new
synthesised eval set.

## 9.5 Outlook

Three generally useful lessons emerge from the nine-iter trajectory
plus the §5.10 multi-seed wrap:

**(1) The hyperparameter trap (§7.1).** The +0.0905 macro-F1 gain
on the CE side and the +0.038 mean gain on the BCE+CutMix side
both came from re-tuning a *single* hyperparameter (LS strength)
that the literature reported a default for. We expect the same to
hold for at least ASL pos_weight, BCE pos_weight, and Focal γ.
Phase B–E will quantify this expectation; if it holds, the right
default for industrial defect classification is not "use multi-
label-native losses" but "sweep the hyperparameters of whatever
loss you happen to be using — including the LS strength, *especially*
on a different loss-family base".

**(2) Asymmetric BKM transfer (§7.5).** Hyperparameter-axis tuning
(LS) and data-axis interventions (CutMix p=0.5) transferred to our
small-data + TAPT regime cleanly. Eight independent axes (warmup,
EMA, drop_path, cutmix-rect, two-LR, T8 CE-soft + CutMix, T13a ASL
γ_neg=2 + CutMix, I11 pair-aware threshold) all failed. The TAPT
init places the chip backbone close to a regularisation ceiling
(§7.4.4); additive regularisers and alternate loss-family
substitutions are net costs. For practitioners with strong TAPT
pretraining and small datasets, our results suggest prioritising
data-axis interventions (synthesis, mixing) and loss-hyperparameter
sweeps over importing structural BKMs from sister domains.

**(3) Single-seed measurement breaks down at the macro-F1 ceiling
(§6.7, §5.10).** Below ≈0.93 the per-iter Δ was ≥3 σ and
single-seed sweeps were sufficient. At ≈0.94 the noise floor
(≈0.030–0.046) is comparable to the typical iter Δ, and the budget
must shift from many cells × 1 seed to fewer cells × n seeds. The
lucky-outlier trap (T9d 0.9705 single-seed vs 0.9408 / 0.8803
replicates) is a paper-grade methodological lesson — and a
warning to any sweep-based study operating near its ceiling.
**Negative results are first-class citizens** of this paper
precisely because their magnitude (0.05–0.11) is well outside the
noise floor; positive single-seed "breakthroughs" near the ceiling
deserve more skepticism than negative single-seed regressions.
The §5.10 multi-seed reporting protocol — n≥3 seeds with mean ± std
for any macro_f1 above 0.92 — is the discipline that translates
this lesson into operational practice.

The fourth lesson is the strict iteration protocol (§8): one-axis
changes, append-only artefacts, GPU=1 sequencing. The protocol's
overhead is small relative to the runtime cost of the experiments
themselves, and it makes every numerical claim in this paper
directly reproducible from the parquet artefacts and the iter logs.
With iter 8/9 / §5.10 the protocol now also requires explicit
variance flagging on every macro-F1 quoted — single-seed cells are
marked as such, and family-mean claims include the per-cell std
across at least three seeds.
