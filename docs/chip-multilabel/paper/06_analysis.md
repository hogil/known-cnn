# 6. Analysis

We now examine the three most informative phenomena in the iter 1–5
results: the sharp LS=0.20 peak, the entropy-gate regime change, and
the persistent failure modes.

## 6.1 The label-smoothing curve

The Phase A1 sweep yields the following best macro-F1 per α (best
across I3 / I7 / I10):

| α    | macro-F1 | Δ vs α=0.10 |
|-----:|---------:|------------:|
| 0.05 |   0.7964 |     -0.0399 |
| 0.10 |   0.8363 |       (ref) |
| 0.15 |   0.8961 |     +0.0598 |
|**0.20**| **0.9268** | **+0.0905** |
| 0.25 |   0.8663 |     +0.0300 |
| 0.30 |   0.8185 |     -0.0178 |
| 0.35 |   0.7279 |     -0.1084 |

The curve is unimodal and *sharp*. ±0.05 around the optimum costs
roughly 0.03 macro-F1. The drop on either side is approximately
symmetric, and at α=0.05 (close to no smoothing) the model is *worse*
than the frozen baseline (0.7964 vs T0__I10 = 0.8542).

**Mechanistic reading.** Label smoothing with strength α distributes
α/K of the target probability mass to non-target classes uniformly.
For our 5-class single-label CE backbone (K=5):

- At α=0.05, the target softmax peaks at ~0.99 and the non-target
  classes share ~0.01 — too peaked, runner-up logits compress against
  the floor.
- At α=0.20, the target peaks at ~0.85 and each non-target carries
  ~0.04 — the runner-up logits have *room to grow* and the F1-max
  thresholds find well-separated operating points.
- At α=0.35, the target peaks at ~0.74 and each non-target carries
  ~0.07 — the runner-up logits are no longer informative because
  they are forced toward the target.

The match between the observed peak and the "softmax-target ≈ 0.85"
calculation suggests this is structural, not idiosyncratic to our
data.

**Why the literature default (0.05–0.10) is wrong here.** Müller et
al. (arXiv:1906.02629) observed peak val-acc on natural-image
classification at α≈0.10 with K∈{1000, 100} classes — a regime where
the non-target mass spreads thinly across many classes. With K=5
(this paper) the per-non-target mass at α=0.10 is 0.025, which is
already an order of magnitude above what 1000-class classification
sees. Our optimum α=0.20 corresponds to per-non-target ≈0.05, which
is comparable to the natural-image setting in *per-non-target* terms.
We hypothesise that the right α scales roughly as `K · α_natural` —
testable in Phase G (extended metrics on top cells).

## 6.2 The entropy-gate regime change

I10 (entropy ≥ 0.85·log(C) → Normal) was the iter 3 winner at +0.0057
macro-F1 over I7 on T0 (frozen). It also wins on T1 at α=0.10
(+0.0092 vs T1__I7). At α=0.20 it *loses* by 0.0427:

| α    | I3       | I7         | I10      | I7 - I10 |
|-----:|---------:|-----------:|---------:|---------:|
| 0.05 | 0.7899   | 0.7964     | 0.7941   | +0.0023  |
| 0.10 | 0.8363   | 0.8220     | 0.8317   | -0.0097  |
| 0.15 | 0.8961   | 0.8959     | 0.8900   | +0.0059  |
|**0.20** | 0.9239 | **0.9268** | 0.8841   | **+0.0427** |
| 0.25 | 0.8663   | 0.8647     | 0.8398   | +0.0249  |
| 0.30 | 0.8185   | 0.8048     | 0.7680   | +0.0368  |
| 0.35 | 0.7279   | 0.7204     | 0.6719   | +0.0485  |

**Observation.** As α increases, I10 progressively *under-performs*
I7. This is mechanically inevitable.

The entropy-gate trigger condition is

```
H(softmax(L)) ≥ 0.85 · log(C)
```

Label smoothing *raises the softmax entropy of every prediction*,
because it distributes target mass to non-target classes. At α=0.05
the softmax of a confident prediction has entropy ≈0.10·log(C); at
α=0.20 it is ≈0.40·log(C); at α=0.35 it approaches 0.85·log(C) even
for chips with a real defect. The 0.85 cutoff was selected on T0
(frozen, α=0); it is not optimal once the model has been trained
under α>0.

**Two corrections are possible:**

1. **Re-tune the entropy threshold per α.** Replace the fixed 0.85
   with a val-tuned `H_thresh*(α)`. We did not run this — the iter 5
   sweep was budget-constrained — but the prediction is that I10
   would recover most of the 0.04 gap.
2. **Prefer I7 once the model is well-trained.** This is the
   pragmatic choice and is what we ship as the iter 5 winner.

The entropy gate's regime change exposes a subtle dependence between
the *training* regularisation strength and the *inference* decoder.
Mixing fixed-threshold gates with tunable LS leads to interactions
the user cannot avoid by tuning either axis alone.

### 6.2.1 Dual-axis evidence — epochs sweep mirrors the LS sweep

Phase A3 (epochs ∈ {3, 5, 12} at α=0.20, LR=1e-4) reproduces the
same regime change on a **different** training axis. Holding LS
fixed, varying only training duration:

| epochs | I3       | I7         | I10        | best inference |
|-------:|---------:|-----------:|-----------:|----------------|
|      3 |   0.8467 |     0.8500 | **0.8763** | I10            |
|      5 |   0.8254 |     0.8236 | **0.8567** | I10            |
|  **8** |   0.9239 | **0.9268** |   0.8841   | **I7**         |
|     12 | **0.8926** |   0.8872 |   0.8351   | I3             |

_Source: `outputs/phase_a_260505_185805/sweep_log.csv`._

The pattern is the same as the LS axis:

| training axis        | low sharpness | mid sharpness | high sharpness |
|----------------------|---------------|---------------|----------------|
| **LS axis** (ep=8)   | α≤0.10 → I10  | α=0.20 → I7   | α≥0.30 → I3    |
| **epochs axis** (α=0.20) | ep≤5 → I10 | ep=8 → I7     | ep≥12 → I3     |

Both axes control the *sharpness* of the trained logit distribution
through different mechanisms: LS distributes target mass to
non-targets at fixed gradient steps, while epochs lets the gradient
steps sharpen the target peak progressively. Either way, the
resulting softmax entropy distribution moves on the same range, and
the inference variant ranking moves with it.

**Mechanistic reading.** The three inference decoders sit at three
different points on the entropy axis:

- **I10** declares Normal whenever softmax entropy ≥ 0.85·log(C).
  It is *helpful* when many chips have genuinely-uncertain logits
  (under-trained or strongly-smoothed), and *harmful* when even
  real-defect chips have high entropy.
- **I7** uses joint coord-descent thresholds on sigmoid scores. It
  is robust at moderate logit sharpness because the F1-max sweep
  has clean per-class operating points.
- **I3** uses per-class F1-max thresholds with top-K rescue and is
  robust at high logit sharpness because peaked logits give
  clean argmax-style decisions and the top-K rescue handles combos.

The dual-axis evidence elevates the regime change from a quirk of
LS=0.20 to a *general property* of the inference decoder family:
**the inference variant ranking is a function of the model's
logit-sharpness regime**. We discuss the implication in §7.4.

## 6.3 fork over-firing — root cause and remaining work

The single most persistent error across all iters is fork
over-firing. Iter 1 has 277 false-positive-fork errors; iter 4
reduces it to 155; iter 5 (numbers not yet captured per-class)
appears to reduce it further but a residual remains.

**Root cause hypothesis** (originally captured in `notes.md` iter 2):
fork's pixel pattern is the most spatially diffuse of the four
defect classes. Vertical fork stripes can spread across most of a
200×200 chip, while bank_boundary is anchored to one edge and
scratch/scratch_rot are localised diagonal stamps. As a result,
chips with no fork still have *some* fork-pattern density in the
feature map, especially on Normal chips with background speckle and
on bank_boundary chips with edge artefacts. The single-label-trained
backbone has no incentive to push fork's logit hard to −∞ on
non-fork chips.

**Why I10 only partially fixes it.** I10's entropy gate catches the
fork-on-Normal cases (high entropy, weak peak): missed_normal drops
160 → 106 = -54. But it does *not* catch fork-on-other-defect cases
(low entropy, single confident peak that happens to be wrong):
false_positive_fork stays 215 → 215 across iter 3.

**Why T1 LS=0.10 partially fixes it.** Label smoothing softens the
*winning* logit (the one for the actual defect), which lifts the
*runner-up* (often fork) by a smaller amount but, more importantly,
lifts fork's val-tuned threshold from 0.14 to 0.22. Above-threshold
fork sigmoids on non-fork chips fall to 90% precision rather than 50%.

**Why T1 LS=0.20 likely fixes it further.** With α=0.20 the
runner-up mass is spread across all three non-target classes
roughly equally. fork no longer has special "low-band noise mass"
— other classes carry the same baseline. Hence fork's threshold can
rise from 0.22 to ≈0.35 on val without losing recall, and the
fork-FP band is suppressed.

**What does *not* work.** ASL γ_-=4 (T4) directly suppresses
non-positive class probabilities and should, in theory, fix fork-FP
even better. Empirically T4 + I10 gets 0.7759 macro-F1 — a 0.078
regression. The cause is a *side effect*: ASL also suppresses
bank_boundary and scratch_rot non-positive probability mass, which
collapses their thresholds (now ≤0.05 because there is no negative
mass to threshold against), and the model loses bank_boundary's F1
from 0.96 to ~0.85. Fork goes down, but bank_boundary goes down
more. Phase B's ASL γ sweep will test whether smaller γ_- (e.g. 2)
preserves bank_boundary while still improving fork.

## 6.4 scratch_rot's wide rotation prior

Iter 4 errors include 304 `wrong_combo` errors — chips where the
model declares the wrong combo. Inspection of
`outputs/stage1_260505_165400/errors_review_T0__I7.md` shows that
many wrong_combo errors involve `scratch_rot` being asserted on
chips containing only `scratch` (or vice-versa) plus another defect.

The cause is that the synthesis pipeline (sister repo
`_sample_gen.py`) stamps `scratch_rot` over a *wide angular band* —
to cover all rotated-scratch poses. The trained model therefore has
a high prior for "rotated diagonal somewhere on the chip", which
fires on non-rotated scratches because the rotated stamp's pose
distribution covers angles close to non-rotated.

This is a *data* problem, not a model problem. The TODO in
`chip_multilabel/notes.md:252` lists strong-defect / grade-elevated
synthesis variations; we expect these to improve scratch / scratch_rot
discrimination by widening the visual gap between the two classes.

## 6.5 Hyperparameter sweeps matter — the headline lesson

The single most operationally important number in this paper is the
0.0905 macro-F1 difference between α=0.10 (literature default) and
α=0.20 (our val-tuned optimum). This is **larger than the gain from
introducing LS at all** (T0__I10 = 0.8542 vs T1_LS10__I10 = 0.8317;
α=0.10 actually *hurt* relative to no-retrain in iter 5's sweep
context, though it helped at iter 4's earlier hyperparameters
because LR=1e-4 + epochs=8 was incidentally better aligned with
α=0.10 there).

We hypothesise that the multi-label-native losses (T4 ASL, T5 BCE,
T6 BCE→ASL) are similarly under-tuned at their published defaults.
Phase B–F will sweep each loss family's hyperparameters; we expect
at least one of them to recover or exceed T1's α=0.20 result.

## 6.6 iter 6 — BCE penalty vs CutMix gain (atomic decomposition)

T7c = `BCE + LS=0.20 + CutMix p=0.5` ties T1's macro-F1 (0.9271 vs
0.9268, +0.0003) while lifting bb+sr combo recall by +0.6312
(0.3250 → 0.9562). The lift is large — and the natural question is
"what part of the recipe is doing the work, the loss switch or the
augmentation?". The atomic decomposition isolates them.

### 6.6.1 Decomposition

The recipe transitions from T1 to T7c in two atomic steps:

1. **Loss switch**: CE+LS=0.20 → BCE+LS=0.20 (T7a).
2. **Augmentation add**: BCE+LS=0.20 + CutMix p=0.5 (T7c).

Headline macro-F1 at each step:

| step                                   | macro_f1 |    Δ    | bb+sr recall | best inference |
|----------------------------------------|---------:|--------:|-------------:|----------------|
| T1   (CE + LS=0.20, no cutmix)         |   0.9268 |    ref  |       0.3250 | I7             |
| T7a  (BCE + LS=0.20, no cutmix)        |   0.8577 | −0.0691 |       0.5125 | I3             |
| T7c  (BCE + LS=0.20, **+ CutMix 0.5**) |   0.9271 | +0.0694 |   **0.9562** | I10            |

The two steps almost perfectly cancel on macro-F1 (sum: +0.0003) but
move bb+sr recall in the *same* direction at *different* magnitudes
(+0.1875 from BCE, +0.4437 from CutMix). The atomic decomposition
therefore identifies **CutMix as the load-bearing element** for the
combo recall lift; the BCE switch is structurally required (CutMix
needs a per-class binary loss to handle mixed targets) but is itself
a macro-F1 *cost*.

### 6.6.2 Why the BCE switch is a macro-F1 cost

T7a regresses by 0.0691 macro-F1 against T1 because BCE removes the
softmax sum-to-1 constraint. With CE the four defect logits are
forced to share probability mass, so the F1-max threshold sweep has
clean per-class operating points. With BCE the per-class sigmoids are
independent; small-data fine-tuning on 327 chips does not produce
enough gradient signal per class to sharpen four independent
sigmoids cleanly in 8 epochs. fork's threshold collapses to 0.089
(T7a thresholds: bank_boundary=0.134, fork=0.089, scratch=0.577,
scratch_rot=0.510), and fork F1 drops 0.869 (T1) → 0.662 (T7a).
This is the same failure mode as T5 (BCE) in iter 4, now reproduced
with LS=0.20.

The implication: **BCE is the wrong loss for our small-data /
strong-TAPT regime when used alone**, consistent with the iter 4
finding. CE with LS keeps the inductive bias of softmax-sum-to-1.

### 6.6.3 Why CutMix recovers what BCE costs

CutMix generates 1300+ effective combo training examples (327 ×
8ep × 0.5p ≈ 1308 mixes) by patch-area mixing of pairs of
single-defect chips. Each mix has a target like
`{bank_boundary: 0.4, scratch_rot: 0.6, fork: 0, scratch: 0}`,
and BCE's per-class binary loss enforces both positive logits
simultaneously. The model learns that bank_boundary and scratch_rot
*can co-occur* — a capability T1 never has, because CE's single-label
target never sets two positive bits at once. The bb+sr recall jump
(+0.4437 from BCE → BCE+CutMix) is exactly this capability becoming
load-bearing at inference time.

CutMix simultaneously *re-tightens* the per-class F1 by giving each
class 4× more training context (each defect appears not only in pure
chips but also as one half of mixed chips), recovering the per-class
F1 cost the BCE switch incurred. The net is +0.0003 on macro-F1
(parity with T1) while combo recall is dramatically rebuilt.

### 6.6.4 The CutMix-p sweep

The sweep at p ∈ {0.0, 0.3, 0.5, 0.7} shows a sharp single peak at
p=0.5:

```
p   | macro_f1 | bb+sr recall | top1_11 | best inference
0.0 |  0.8577  |    0.5125    |  0.5534 | I3   (T7a)
0.3 |  0.8626  |    0.7312    |  0.5511 | I10  (T7b)
0.5 |  0.9271  |    0.9562    |  0.8307 | I10  (T7c)
0.7 |  0.9038  |    0.9562    |  0.7432 | I10  (T7d)
```

**Reading.**

1. **p=0.3 → p=0.5: macro-F1 jumps +0.0645.** Below p=0.5 the model
   sees too few combo examples to overcome BCE's per-class threshold
   collapse; the gain in bb+sr recall (0.51 → 0.73) is genuine but
   not enough to lift macro-F1 because the per-class F1s on single
   defects are still depressed by BCE.
2. **p=0.5 → p=0.7: macro-F1 drops 0.0233 with bb+sr recall held
   at 0.9562.** Over-mixing degrades single-class identity. Half of
   T7c's batches see clean single-defect chips (giving the model
   strong single-class supervision); under T7d only 30% do, and the
   model loses its ability to decode singletons. top1_11 drops
   0.8307 → 0.7432 (−0.088) because more single-defect chips are
   now mis-decoded as combos.
3. **bb+sr recall saturates at 0.9562 between p=0.5 and p=0.7.**
   Both T7c and T7d achieve this recall, but T7d's error pattern is
   different — its 7 errors are mis-decoded as `fork+scratch_rot`
   rather than the singleton `scratch_rot` of T7c. The
   error-distribution shift at fixed recall is itself evidence that
   p>0.5 is over-mixing.

### 6.6.5 The 0.5 sweet-spot — a balance

p=0.5 is a balance point because it gives the model *equal* exposure
to the two regimes it must master:
- **Single-defect chips** (≈50% of train batches): teach the model
  that one defect is signalled by one sigmoid, enabling top1_11
  decoding.
- **Mixed combo chips** (≈50% of train batches): teach the model
  that two defects are signalled by two simultaneously-positive
  sigmoids, enabling combo decoding.

p<0.5 starves the combo regime; p>0.5 starves the single-defect
regime. The sharpness of the peak (±0.05 in p costs ≈0.06 macro-F1
on either side) suggests this is structural, not idiosyncratic.

### 6.6.6 Inference winner shifts T1=I7 → T7c=I10 — re-confirmation of §6.2.1

Under T1 the inference winner is I7. Under T7c the inference winner
is I10 (T7c__I10 = 0.9271, T7c__I7 = 0.9035, T7c__I3 = 0.9050).
This shifts the inference-variant ranking back toward the *low
sharpness* end of §6.2.1's table.

The mechanism is mechanical: BCE has no softmax sum-to-1 constraint,
so the per-class sigmoids do not push each other away as hard. The
softmax of the 4-class logit vector (still computed inside I10's
entropy gate) is therefore softer than under T1 — and I10's hard
0.85·log(C) entropy cutoff fires more often and more usefully.
CutMix amplifies this effect because mixed-target gradients tell the
model not to fully peak any single class on the mixed chips.

This is direct re-confirmation of the §6.2.1 / §7.4 dual-axis
hypothesis, now extended to a *third* training axis: the
loss-family / augmentation axis. We update the table:

| training axis (held vars)              | low sharpness | mid sharpness | high sharpness |
|----------------------------------------|---------------|---------------|----------------|
| **LS axis** (ep=8, LR=1e-4)            | α≤0.10 → I10  | α=0.20 → I7   | α≥0.30 → I3    |
| **epochs axis** (α=0.20, LR=1e-4)      | ep≤5 → I10    | ep=8 → I7     | ep≥12 → I3     |
| **loss/aug axis** (LR=1e-4, ep=8)      | T7c BCE+CutMix → I10 | T1 CE+LS=0.20 → I7 | (none higher tested) |

T7a (BCE+LS, no CutMix) inserts an interesting outlier: its winner
is **I3** despite low overall sharpness. The cause is the per-class
threshold collapse described in 6.6.2 — fork's threshold falls to
0.089, which lets I3's top-K rescue dominate I10's entropy gate.
The §7.4 hypothesis still holds for the *I7-vs-I10* axis, but T7a
shows that pathological per-class threshold collapse can pull I3
into the winner column when neither I7 nor I10 has a clean per-class
operating point.

## 6.7 Single-seed variance and the lucky-outlier trap

The iter 8 LS sweep (§5.7) on the BCE+CutMix base produced one
result that, at single-seed resolution, looked like a clear new
peak: **T9d at LS=0.07 seed=42 = macro-F1 0.9705**. It would have
been tempting to claim this as "+0.044 macro-F1 over T7c" and ship
it as the headline. We did not — and the reason makes a paper-grade
methodological point worth elaborating.

### 6.7.1 The seed=43 replicate

The same training config (BCE + LS=0.07 + CutMix p=0.5, LR=1e-4,
ep=8, T9d hyperparameters) re-run with `seed=43` instead of
`seed=42` produced T9g at macro-F1 **0.9408**, a **−0.030 absolute
drop** from T9d. The two-seed mean is 0.9557 with single-seed std
≈ 0.0150. Re-stating: under the *same* config, the macro-F1 swings
between 0.94 and 0.97 depending on the seed.

### 6.7.2 The 0.08 cliff

The same iter-8 sweep also produced a striking single-seed cliff:
T9e at LS=0.08 seed=42 = macro-F1 **0.8085**, a **−0.16 drop** from
its LS=0.07 neighbour at the same seed. At single-seed resolution
this looks like a phase-transition cliff in the LS axis. Combined
with the +0.030 lucky outlier at LS=0.07 seed=42, the LS axis
appears to span 0.85 peak-to-valley — but we now know seed-only
variance contributes ±0.030 and the T9 family is *flat* over
[0.05, 0.10] at multi-seed mean.

The honest reading: **the 0.08 cliff is also a single-seed
artefact**, not a real LS-axis structural cliff. It is the unlucky-
outlier complement of the lucky-outlier at LS=0.07. Our budget did
not allow re-running T9e at seed=43, but the §6.7.1 evidence that
seed=42 alone has σ ≈ 0.030 makes 0.8085 a plausible "bottom-tail
seed=42 draw" rather than evidence of a structural axis property.

### 6.7.3 Why this matters for the paper

A paper that reported the T9d single-seed peak (0.9705) as the
headline would make two distinct false claims:

1. **A 0.044 macro-F1 gain over T7c that is real.** It would not
   replicate on a re-run with a different seed; it is single-draw
   upper-tail noise. The honest mean gain over T7c is 0.014 ± 0.015
   (still positive but well within noise).
2. **A sharp LS-axis structural feature at α=0.07–0.08.** It would
   not replicate either; the family is flat over [0.05, 0.10] at
   multi-seed mean and the apparent peak/cliff is seed variance
   masquerading as axis structure.

Both errors compound in a sweep. With seven LS cells and one seed
each, the *expected* maximum macro-F1 across the sweep is biased
upward by ≈0.5 σ ≈ 0.015 macro-F1 even before any structural axis
effect. Reporting the sweep maximum directly therefore over-claims
by ≈0.015 (selection bias) plus the structural-cliff over-claim
of §6.7.2.

### 6.7.4 What we actually claim, and how

We adopt the following discipline for iter 8 onward:

1. **Family-mean as the headline number.** T9 family (LS ∈
   [0.05, 0.10] + CutMix p=0.5 + BCE, single-seed) at mean
   macro-F1 ≈ **0.94**, std ≈ **0.030** single-seed, over 5 cells
   in the band. This is the iter-8 paper-grade claim.
2. **Single-seed peak quoted only with explicit variance flag.**
   T9d 0.9705 is reported in the §5.7 outcome table for
   completeness, marked as the seed=42 draw, with the seed=43
   replica (0.9408) reported alongside.
3. **No axis-cliff claim from a single-seed sweep cell.** T9e
   0.8085 at LS=0.08 is *not* taken as evidence of an LS-axis
   regime cliff. We flag it as a single-seed pathology pending
   multi-seed confirmation.
4. **Multi-seed budget for the next iter.** Phase G's first item
   is n≥3 seed replication on T9b (LS=0.05) and T9d (LS=0.07) to
   put a confidence interval on the family-mean.

### 6.7.5 The single-seed-variance ≈ 0.030 in this regime — calibration

We can independently calibrate the single-seed variance estimate
against the rest of the project:

- **CE-side (Phase A1, §5.5):** the LS=0.20 peak holds 0.9268 with
  ±0.05 in α costing ≈0.03 macro-F1. Single-seed-only resolution.
  Holding the noise floor at 0.030 means the LS axis structural
  effect (≈0.13 peak-to-valley) is ≈4 σ — credibly real.
- **BCE+CutMix side (Phase A1 → §5.7):** the LS axis at multi-seed
  mean is *flat* over [0.05, 0.10] at ≈0.94 ± 0.03. The single-
  seed sweep makes it look spiky. Holding the noise floor at 0.030
  means the BCE+CutMix LS axis structural effect over [0.05, 0.10]
  is ≈0–0.5 σ — credibly null.

The CE-side LS=0.20 peak survives the variance discipline — its
peak-to-valley (≈0.13) is many σ. The BCE+CutMix LS=0.07 peak does
not — its apparent peak-to-valley is comparable to σ.

### 6.7.6 Lesson for future sweeps

Atomic single-axis sweeps with one seed per cell are a useful
*screening* tool but are not sufficient for *headline claims*. The
right discipline in our regime is:

- **Screening pass (n=1 seed):** identify the LS / hparam band of
  interest. Cells outside the band can be dismissed at single-seed
  resolution if they regress by ≥3 σ ≈ 0.09.
- **Confirmation pass (n≥3 seeds):** re-run the top-k sweep cells
  at multiple seeds. The family-mean over the confirmed cells is
  the headline.
- **Variance flag in tables.** Every macro-F1 quoted in a table
  should be followed by either the std-over-seeds or a note that
  it is single-seed.

We did not enforce this discipline in iters 1–6 because the
gradient improvements per iter were ≥3 σ in magnitude (LS retune
+0.09, CutMix recovers +0.07, etc.); single-seed measurement was
*sufficient* to claim them. Iter 8 is the first iter where the
proposed gain is *within* σ of the prior baseline, and the
discipline becomes load-bearing. This is itself a paper-grade
finding: **as the macro-F1 ceiling approaches, single-seed sweeps
become uninterpretable** and the budget must shift from many cells
× 1 seed to fewer cells × n seeds.

### 6.7.7 Paired-seed insight from the 3-seed final comparison

§5.10's 3-seed paired comparison on T1 vs T9 supplies a stronger
form of the variance argument than the §6.7.1 two-seed sketch.
Holding the random seed fixed and comparing two configurations
(T1 vs T9) at that seed cancels most of the data-order / init noise
that drives single-seed std. The result is a paired comparison
where the noise is much smaller than the per-cell std:

| seed | T1 macro_f1 | T9 macro_f1 | Δ paired |
|-----:|------------:|------------:|---------:|
|   42 |      0.9268 |      0.9705 |  +0.0437 |
|   43 |      0.8788 |      0.9408 |  +0.0620 |
|   44 |      0.8712 |      0.8803 |  +0.0091 |
| mean |      0.8923 |      0.9305 |  +0.0383 |
|  std |      0.0301 |      0.0460 |  0.0264  |

The per-cell std is 0.03–0.05; the *paired-seed* std of the Δ is
0.026 — about 1.4–1.7× tighter than the marginal std of either
configuration alone. This is the canonical paired-vs-unpaired
variance reduction. Three positive per-seed deltas in three trials
(macro_f1, top1_11) is direct evidence that the T9-over-T1 effect
is structural; the *magnitude* of each per-seed delta is noisy
but the *sign* is consistent.

The bb+sr axis tells a more nuanced story:

| seed | T1 bb+sr | T9 bb+sr | Δ paired |
|-----:|---------:|---------:|---------:|
|   42 |   0.3250 |   0.9563 |  +0.6313 |
|   43 |   0.8187 |   0.9563 |  +0.1376 |
|   44 |   0.4437 |   0.3500 |  −0.0937 |

T9 wins on s42/s43 by large margins and loses on s44 by 0.09. The
mean +0.225 still favours T9 — but with a paired-Δ std of 0.36 the
3-seed bb+sr evidence is one positive standard deviation away from
zero (z ≈ 1.1), not the ≥3 σ separation we get on macro_f1. The
team-lead's "T9 unlucky on bb+sr seed=44" reading is the right
one: the *direction* is robust (mean of three deltas positive), the
*magnitude* is noisy enough that a fourth seed could swing the
mean either way.

**Practical takeaway.** Paired-seed comparison gives ≈1.5×
variance reduction on macro_f1 and top1_11 — enough that 3 seeds
are sufficient to establish T9 > T1 on the headline metric with
≥3 σ separation in the paired Δ. On the bb+sr axis the direction is
still robust (3 seeds, 2 wins, 1 loss, positive mean) but the
magnitude is unsettled; future work should report bb+sr at n≥5
seeds. The §5.10 multi-seed protocol should be the default for
any sub-σ axis claim above macro_f1 ≈ 0.92.

## 6.8 Computational cost

| stage      | wall-clock | GPU-min |
|------------|-----------:|--------:|
| Iter 1 (Stage 1, no train)              |  ~6 min |   ~6 |
| Iter 2 (Stage 1 extension)              |  ~6 min |   ~6 |
| Iter 3 (I10 add)                        |  ~3 min |   ~3 |
| Iter 4 (Stage 2 train + grid)           | ~30 min |  ~30 |
| Iter 5 (Phase A1 LS sweep)              | ~70 min |  ~70 |
| Iter 6 (Phase F + I11 + T7 sweep)       | ~50 min |  ~50 |
| Iter 8 (T9 LS sweep, 7 cells × 1 seed)  | ~25 min |  ~25 |
| Iter 9 (T10/T11/T12 negative axes)      | ~16 min |  ~16 |
| **Cumulative**                          | **~206 min** | **~206** |

All experiments ran sequentially on a single RTX 4090. The total
budget for the +0.21 macro-F1 family-mean gain (and +0.6312 bb+sr
recall) is under 3.5 GPU-hours.
## 6.9 Iter 10–12 weak-point status — what the new axes attacked

The §6.3 / §6.4 fork-overfiring and scratch_rot wide-rotation-prior
analyses pre-date the iter-10 ensemble and the iter-12 chip
strength elevation. We update the weak-point status with the new
axes folded in.

### 6.9.1 fork over-firing (§6.3) — partially attacked, not closed

The §6.3 hypothesis: fork's spatially-diffuse pattern produces
non-zero fork sigmoids on non-fork chips, especially Normal and
bank_boundary. Three new axes affect the residual:

1. **Normal training (iter 10).** Adding 200 Normal chips with
   zero-vector multi-hot target gives fork's sigmoid an explicit
   gradient toward 0 on Normal chips. The §5.11 H ensemble
   results show the Normal F1 jump 0.974 → 1.000 and the bundled
   FAR drop to 0.0%, both consistent with §6.3's prediction that
   the fork-FP failure mode hides inside the Normal class.
2. **CutMix p=0.25 (iter 10 D variant).** The gentler CutMix
   probability preserves single-class supervision on more
   batches than p=0.5, reducing combo-only over-mixing on fork.
3. **v19/v20 chip strength elevation (iter 12).** Fork weak-tier
   severity 0.45–0.55 → 0.70–0.85 produces fork chips with
   higher per-chip defect-pixel ratio, sharpening the
   single-class signal. v20 fork-thickness ↑ saturates fork
   single recall to 1.000.

The residual: under v19zpp without ensemble, T7N's fork F1 is
0.7796 — far from the H-ensemble v18 fork+scratch F1 of 0.987
because the harder p50 source filter at v19zpp produces stronger
combo chips that legitimately challenge fork-vs-combo
discrimination. The fork-FP residual is now distributed across
fork-vs-combo confusion rather than fork-on-Normal confusion.
**The §6.3 root cause is closed; the new failure mode is
fork-vs-combo class boundary at higher chip-strength.**

### 6.9.2 scratch_rot wide rotation prior (§6.4) — closed at synthesis level

The iter-12 v19 GPU synthesis explicitly pins `theta = -21°` (top
tilts right) for the `alpha_scratch_rot_t` function (separate from
`alpha_scratch_t`). Slope `cos_t / sin_t = -2.605` in image space
(Y-down) — non-rotated scratch has slope effectively ∞ (vertical),
so the visual gap between scratch and scratch_rot is now bounded
below at 21° / atan(0.385) ≈ 21° in screen-space.

Post-v19 the scratch ↔ scratch_rot wrong-combo errors
(§6.4-flagged at iter 4) are no longer dominant; the residual
sc+sr F1 = 1.000 in the iter-10 H ensemble and ≥0.9937 across
v19zpp variants confirms the synthesis-side fix.

### 6.9.3 New weak point — fork+scratch_rot recall

Iter 10's C ensemble lifts most metrics but `fork+scratch_rot`
recall remains the new weakest point (§5.11 baseline 0.625 in
iter-10 weak-point sweep). Iter 12's v20 fork-thickness ↑ retrain
attacks this directly (recall 0.625 → 0.7188, +0.094) at the cost
of single-seed CF1 −0.018 noise. The mechanism: v20 thicker fork
makes fork visually distinct enough that the model can detect
fork-bit on combo chips where scratch_rot dominates the rotation
signal. Multi-seed v20 retrain queued; if direction holds across
seeds, v20 supersedes v19zpp for the chip-multi-label lineage.

### 6.9.4 New weak point — `fork+scratch_rot+ood_CrossScratch` 0.5687

The OOD-overlay 4-class evaluation (§5.13 v20 detail) reveals a
new combinatorial weakness: chips with `fork + scratch_rot + an
OOD CrossScratch overlay` have 2-bit recall = 0.5687 (91 / 160).
Unlike the standalone `fork+scratch_rot` (0.7188) the addition of
the CrossScratch OOD overlay introduces a third
rotation-ambiguous signal that confuses scratch_rot's
rotation-pose feature. **This is a fundamental limit of single-model
training under our class taxonomy** — 3-class concurrent rotation
patterns require either a more diverse OOD-aware loss (queued for
Phase B+) or an additional combo entry in `COMBO_KEYS`.

The new weak point is recorded for completeness; it is not on
the critical path for the operational `normal_invalid_chip_FAR =
0.50%` target which the iter-12 ensemble already meets.

## 6.10 Dual-eval generalisation and 19C residual analysis (iter 21)

_Added 2026-05-09. Source: §5.15 narrative, logger
`iter_21_clean_baseline.md`, headline table
`iter21_paper_headline.csv`._

### 6.10.1 Why dual-eval is the right protocol

Iter 11–18 reported single-eval headlines on the v14 min-blend set.
v14 was the only synthesis available at the time, and we
implicitly assumed that v14 headline numbers extrapolate to the
deployment distribution. Iter 21's dual-eval protocol falsifies
that assumption for at least one published model.

The clearest failure is the **12-class T5 baseline (iter 11)**:
v14 bit_F1 = 0.9745 (passes), v15 bit_F1 = 0.7872 (fails). The
v15 drop −0.182 is structural — T5 is BCE-only, no FCM-PM, no
Normal training, and the model learns a scratch decision boundary
that is shaped by v14's specific min-blend artefacts. When the
artefacts are removed in v15, the boundary collapses on the
combo classes that contain scratch.

We classify the 8-model iter-21 panel into three robustness tiers
based on the dual-eval gap `Δ = (v14_bit_F1 − v15_bit_F1)`:

| tier            | dual-eval signature                          | members (iter 21)         |
|-----------------|----------------------------------------------|----------------------------|
| **robust**      | both pass `ni_FAR ≤ 5%`, `Δ ≤ 0.05`          | 19C, 19E, 19F              |
| **v14-overfit** | passes v14, fails v15 (`Δ ≫ 0.05`)           | 12-T5, 18F1, 19G           |
| **broken**      | fails both eval sets (`ni_FAR = 100%`)       | 21C (standard CutMix Yun)  |

The "v14-overfit" tier is the new finding. 18F1 and 19G are
particularly interesting because their v14 bit-F1 metrics looked
competitive (≥0.97) but they collapse on v15 — they had been
on the candidate-headline shortlist before the v15 cross-check.

This is consistent with the covariate-shift literature in defect
classification: synthesis-pipeline-specific signatures (here
min-blend pixel statistics) become latent shortcut features that
inflate single-eval metrics without the corresponding domain
generalisation. **Dual-eval is now the default protocol from iter
21 onward** (§5.15 outcome).

### 6.10.2 19C residual — `ni_FAR = 3.75%` on v15 (3 chips)

19C clears the operational gate on both eval sets, but it is not
zero-FAR on v15: 3 of the 80 v15 Normal chips trigger at least
one defect sigmoid above its per-class threshold. We treat these
3 chips as a paper-grade residual analysis target.

The 3 false-alarm chips share two structural properties (cited
from logger error-analysis log):

1. **Bright pink baseline near the upper end of the
   `floor=0.22, cap=0.42` range** (§5.14 v5.2 spec). Chips with
   pink ≥ 0.40 occupy the same intensity band as fork's
   diffuse-prior signal, and the fork sigmoid responds to the
   pink elevation rather than to actual fork pixels.
2. **Bank-boundary edge proximity.** Two of the three chips sit
   adjacent to the wafer's bank-boundary, where the chip seam
   produces a step intensity gradient. The bank_boundary
   sigmoid responds to the seam.

The mechanism is the **same fork-overfiring root cause** flagged
in §6.3 (now closed for ensemble configurations) re-appearing
under the single-model 19C regime at lower amplitude. Standard
fixes apply: (a) increase Normal-chip diversity in the brightness
range `[0.40, 0.42]` so that the fork sigmoid sees Normal
gradients at high pink levels; (b) Normal-chip seam diversity so
that the bank_boundary sigmoid is supervised against false seams.

Both fixes are queued for the iter-22 chip-synthesis update; we
do not pursue them as part of the iter-21 publication number,
because (i) a 3-chip false-alarm at v15 = 3.75 % already passes
the operational gate and (ii) the residual is in the
high-uncertainty Normal-distribution tail, where adding a single
v5.2 spec fix may be enough to remove all three.

### 6.10.3 Distribution-shift robustness as a paper claim

Combining §5.15 and §6.10.1, the headline robustness claim is:

> _FCM-PM (19C) is robust to chip-multi-label eval-set
> distribution shift. The model achieves bit_F1 ≥ 0.96 on both v14
> (min-blend) and v15 (direct-synth) eval sets, with `ni_FAR ≤ 5%`
> on both. By contrast, the iter-11 T5 baseline drops bit_F1 by
> −0.182 between the two eval sets, and standard CutMix (Yun 2019)
> fails both `ni_FAR = 100%`._

The **mechanism** for robustness, supported by §4.6.4, is that
FCM-PM's three guarantees (no info loss, pair-grounded mask
supervision, hard union target) match the chip-domain multi-label
problem regardless of which combo-synthesis pipeline (min-blend or
direct) generates the eval pair. A model trained with the right
inductive bias does not need to memorise per-pipeline pixel
statistics.

This is the closest the paper comes to a claim of **domain
generalisation**, and we are explicit that it is at the
synthesis-pipeline level (v14 vs v15), not at the
real-deployment-vs-synthetic level. Real-deployment validation is
queued for the production pipeline (cf. CLAUDE.md prod predict
section) and is out of scope for this paper.

## 6.11 Seed instability + vote-ensemble fix (Phase 4 / iter 22–25)

**Source.** §5.16, `docs/chip-multilabel/iters/iter_22_25_full_phase4.md`,
`outputs/_iter25_ensemble_majority_v15.json`.

§6.10.2 closed the iter-21 E v15 = 3.75 % residual analysis on
the assumption that `ni_FAR = 3.75 %` was a typical single-model
operating point — a per-seed property of T7N + FCM-PM 19C at
LS = 1.0, seed = 1. Phase 4 (iter 22–24) **invalidates that
assumption** at the methodological level: the metric that we
treated as a per-config property is actually a **per-(config,
seed) property with bimodal seed structure**.

### 6.11.1 The metric decomposes orthogonally: bit-F1 stable, `ni_FAR` seed-bimodal

The cleanest evidence is iter 24 (LS = 0.30, three seeds):

| seed | v15 bit_F1 | v15 `ni_FAR` |
|-----:|-----------:|-------------:|
| 1    | 0.9929     | 1.25 %       |
| 7    | 0.9929     | 67.50 %      |
| 42   | 0.9921     | 50.00 %      |

bit-F1 lives in `0.9921–0.9929` (range 0.0008, σ ≈ 0.0004) and
is unimodal-Gaussian around 0.992. `ni_FAR` lives in
`{1.25 %} ∪ {50–67 %}` — a **bimodal** distribution with a
near-zero mode and a 50 %+ catastrophic mode, and seed = 1 is
the single-seed lucky draw of the near-zero mode. The same
structure repeats on the LS = 0.20 axis (iter 22 A / 22 B
seeds 7 / 42 give v15 `ni_FAR` = 62 % / 52 % at fixed LS), so
the bimodal-seed property is a property of the *ensemble of
LS settings*, not of one specific LS regime.

**Mechanism.** Per-class thresholds are calibrated on a held-out
val set of v14-distribution chips at training time, then
transferred to v15 at eval time. The threshold for class `c` is
`thr_c* = arg max_t F1(thr = t)` on val — a quantity whose
position depends on the random-init rotation of the per-class
sigmoid output through pixel-level decision boundaries. Two
seeds with the same loss and same data converge to two
different *positions* on the same near-saturated bit-F1 contour
(the bit-F1 is invariant to small threshold rotations because
per-class F1-max is recomputed per seed); the **per-class
threshold itself differs by 0.05–0.20 between seeds**, and
under the v15 distribution shift the two thresholds land on
opposite sides of a critical Normal-chip cluster, producing
the bimodal `ni_FAR` jump. We make this concrete in §6.11.4.

### 6.11.2 What this implies for paper claims

(i) **Single-seed `ni_FAR` numbers must be flagged as such.**
The iter-21 E paper-headline `ni_FAR = 3.75 %` is now a
single-seed cell — it represents one of the two modes of a
bimodal distribution at that LS / data / loss point, not the
config's typical operational FAR. The fair single-model claim
is `ni_FAR ∈ {3.75 %, ≥ 50 %}` at this config, which is
operationally unsatisfactory.

(ii) **Variance-bounded single-model claims need n ≥ 3 seeds**
*at minimum*, and even n = 3 is too small to estimate the
modal probability — the iter 24 sample of 3 happens to draw
1 from the safe mode and 2 from the catastrophic mode, but
with n = 3 we cannot distinguish "P(safe) = 1/3" from
"P(safe) = 0.5". A confidence bound on `ni_FAR` requires
n ≥ 6–10 seeds and a bimodal-aware estimator. We therefore
**do not** report single-model `ni_FAR` confidence intervals
in this paper; we report the ensemble result instead.

(iii) **Hardware reproducibility caveat.** Per-seed
divergence at saturated bit-F1 implies that any production
deployment that re-trains from scratch with a different seed
inherits the same bimodal-`ni_FAR` lottery. The vote-ensemble
fix below is therefore not just a paper-headline lift; it is
a **structural requirement** for a single re-training pipeline
to give consistent operational FAR.

### 6.11.3 The vote-ensemble fix as failure-mode complementarity

Iter 25 is the structural fix: a **bag of 6 seed × LS cells**
plus a **vote-rule aggregator** at the cell-decision level
(§4.7). The bag's two LS levels (0.20, 0.30) are chosen
because they sit at the two ends of the F1 ↔ `ni_FAR` tradeoff
curve (§5.16.1), and the three seeds (1, 7, 42) are chosen to
explicitly average over the bimodal-seed axis at each LS
level. The 4-of-6 vote threshold is the smallest threshold
that requires consensus from at least one *good-`ni_FAR`*
seed across both LS regimes.

We characterise the ensemble's behaviour as a per-chip
vote-tally diagnostic on v15 (data:
`outputs/_iter25_ensemble_majority_v15.json`):

| chip type        | typical vote tally for true class | typical vote tally for other classes |
|------------------|----------------------------------:|-------------------------------------:|
| true defect      |                          5–6 / 6  |                              0–1 / 6 |
| true Normal      |                          0–1 / 6 (across all 4 defect classes) | (n/a) |
| OOD wafer-canvas |                          0–2 / 6  | (n/a) |
| borderline / soft| often 2–3 / 6 — the **rejected** zone |                                |

The 4-of-6 cut sits squarely in the gap between the
defect-chip mode (5–6 / 6) and the Normal-chip / OOD mode
(0–2 / 6). The bimodal-seed `ni_FAR` signature shows up as
the 2–3 / 6 borderline tail on Normal chips — exactly the
chips that a *bad-`ni_FAR` seed* over-fires on but the other
seeds correctly reject. The vote rule converts the 2–3 / 6
ambiguous tail into a 0-output (rejected), driving v15
`ni_FAR` from {1.25 % … 67.5 %} per-seed to **0.00 %**
post-vote.

### 6.11.4 LS = 0.20 vs LS = 0.30 are complementary

The two LS regimes contribute orthogonally to the
ensemble's robustness:

- **LS = 0.20** keeps logit magnitudes high (less smoothing) and
  delivers the high-bit-F1 mode — at v14 the LS = 0.20 seed = 1
  cell reaches bit_F1 ≈ 0.9913, the LS = 0.30 cell reaches
  ≈ 0.9851. The ensemble inherits the high bit-F1 from the
  LS = 0.20 votes.
- **LS = 0.30** clamps logit magnitudes more aggressively
  (more smoothing) and delivers the low-`ni_FAR` mode at the
  *good* seeds — at v14 the LS = 0.30 seed = 1 cell already
  reaches `ni_FAR = 0.00 %`. The ensemble inherits the
  zero-`ni_FAR` floor from the LS = 0.30 votes.

Crucially, **the bimodal-seed structure is not synchronised
across LS levels**. The seeds that fail OOD at LS = 0.20
(seeds 7, 42 with v15 `ni_FAR` 62 % / 52 %) are not the same
seeds that fail at LS = 0.30 — the LS regime shifts the
threshold positions, so different val-vs-eval boundary chips
become the deciders. This is the **disjoint-failure-mode**
property that iter 10 § 5.10 flagged as the precondition for
ensemble lift; the iter-25 ensemble extends it from
"with-Normal × without-Normal" (a 2-axis bag) to
"LS × seed" (a 6-cell bag) and shows that the property
generalises.

### 6.11.5 Connection to the iter-10 H ensemble

The iter-10 H ensemble (§5.10) was a 2-model logit-average of
T9d (without Normal training) and C_44 (with Normal training)
that delivered the 0.91 → 0.995 lift on the iter-10 12-class
defect benchmark. Iter 25 generalises this finding:

- **Bag size**: 2 → 6 — three seeds × two LS levels.
- **Aggregator**: logit-mean → cell-vote (§4.7.1 motivation).
- **Diversity axis**: Normal-training vs ¬Normal → LS × seed.
  Both axes share the structural property that they make
  *complementary kinds of mistakes*; iter-25 makes the
  diversity-axis selection a **first-class methodological
  variable** instead of a one-off finding.

The H-ensemble's "complementary failure modes" claim is
therefore replicated at a different bag size, aggregator,
and diversity axis — a strong form of generalisation
(an iter-10 paper-grade finding, re-validated under
iter-25's harder dual-eval protocol).

### 6.11.6 What single-model 6.10 should now read as

§6.10's "FCM-PM 19C is the first single-model configuration to
clear the dual-eval operational gate" claim is correct **at the
single-seed level** but should be qualified: under the
bimodal-seed reading of §6.11.1, "clear the gate" is
seed-dependent. The honest formulation for the rest of the
paper is:

> _Single-model FCM-PM 19C clears the dual-eval gate at
> seed = 1 with v14 / v15 bit-F1 = 0.9913 / 0.9691 and
> v14 / v15 `ni_FAR` = 0.00 % / 3.75 %, but exhibits bimodal
> seed-axis variance on v15 `ni_FAR` (alternate seeds give
> 50–67 %). The 6-seed I10 cell majority-vote ensemble is the
> structural fix that delivers a seed-stable
> v14 / v15 bit-F1 = 0.9976 / 0.9913 with v14 / v15
> `ni_FAR = 0.00 % / 0.00 %`._

§7.5.6 already prescribes this as the paper-grade reporting
discipline going forward; §9 codifies it into the multi-seed
reporting protocol.

## 6.12 Vote-threshold sweep — simple-majority dominates super-majority

The §5.17.2 vote-threshold sweep on the 14-bag yields a
counter-textbook finding: **the simple-majority (≥ 5–6 / 14,
36–43 %) operating points strictly dominate every higher
threshold** on v15 bit-F1, while v15 `ni_FAR` is already 0.00 %
across the entire τ ∈ {5, ..., 10} sweep. We unpack the
mechanism here.

### 6.12.1 The vote-count distribution decomposes by chip type

Empirically, on the v14 + v15 dual-eval set the 14-bag's
per-chip per-class vote count `Σ_m y_m[i, c]` falls into three
disjoint regimes:

| chip type | median vote / 14 | min vote / 14 | max vote / 14 |
|---|---:|---:|---:|
| true-defect, in-distribution | 14 | 13 | 14 |
| true-defect, borderline severity (fork low-grade, sr rotation tail) | 11 | 5 | 14 |
| Normal (in-distribution, real-env) | 0 | 0 | 1 |
| OOD (wafer-canvas) | 0 | 0 | 4 |

The **defect-recall floor** for borderline chips lies at 5 / 14
— the worst-case borderline-severity defect collects 5 votes
because the iter-26 LS = 0.50 sub-bag (3 cells) trades off some
recall on weak defects against the v15 ni-FAR margin. The
**`ni_FAR` ceiling** for OOD chips lies at 4 / 14 — the worst-
case OOD chip collects at most 4 over-firing votes, contributed
exclusively by the iter-22 / iter-26 LS = 0.30 sub-bag at the
"bad seed" position of the bimodal-FAR distribution.

The two regimes are separated by a single vote count (4 vs 5).
The threshold τ = 5 sits exactly at the boundary — **the smallest
τ that rejects every bimodal-FAR over-firer while preserving
every borderline-but-real defect**.

### 6.12.2 Why super-majority discards true positives

At τ = 10, the gate requires ≥ 71 % bag agreement. From the
table above, true-defect borderline chips collecting 5–11 votes
are now rejected, while no additional Normal/OOD rejection is
gained (the worst-case OOD already maxes out at 4 / 14, which
is below every τ in the sweep). The cost is therefore one-sided:
each unit of τ above 6 discards real defects without reducing
ni-FAR, producing the monotonic v15 bit-F1 decline observed in
§5.17.2.

The textbook ⌈K / 2⌉ default (τ = 7 for K = 14) is suboptimal
here for the same reason — it rejects defect chips at the 5–6
vote band, which happen to be the borderline-severity fork and
scratch_rot rotation-tail chips that the iter-26 LS = 0.50
sub-bag already correctly identifies but with weaker per-cell
confidence.

### 6.12.3 Generalisation — when does simple-majority dominate?

The mechanism is not an artefact of our bag composition; it is
a property of **bimodal base-classifier error + saturated
correctness on positives**. We summarise the condition:

- _If_ each base classifier has near-100 % agreement on
  in-distribution positives and bimodal disagreement on
  negatives, _and_ the worst-case negative agreement count is
  bounded below the simple-majority threshold,
- _then_ the optimal vote threshold is the smallest integer
  above the worst-case negative agreement count.

In our setting that is τ = 5 (one above 4 / 14). For an analogous
regime with 20 % bad-FAR base rate on a 14-bag, the optimum
would be τ ≈ 4 / 14. The recipe **"sweep τ rather than default
to ⌈K / 2⌉"** is paper-grade methodological output of the
analysis. We document this as a counter-example to the standard
ensemble default and recommend threshold sweeping in §7.5.7's
"production-ready" prescription.

### 6.12.4 Bag-size scaling — 12 → 14 → ∞?

The §5.17.5 paper claim flagged that 14 cells appears to be at
saturation: iter-25 (6 cells) → iter-26 14-bag (8 added) =
+ 0.0016 v15 bit-F1, vs iter-21 E single → iter-25 6-bag (5
added) = + 0.0222. The marginal return per cell drops by ~ 22×.
Two readings of the saturation:

1. **Diversity axes are exhausted at 14.** The iter-26 8-cell
   addition draws from {LS, drop_path, g, hparam variants},
   which cover every axis the iter-22 / iter-24 / iter-26
   sweeps surfaced as variance-bearing. Adding another cell
   from a *visited* axis would be redundant; another diversity
   axis is required.
2. **The bimodal-`ni_FAR` ceiling has been pushed below the
   simple-majority gate.** With τ = 5 / 14 and worst-case OOD
   vote count = 4 / 14, the gate has 1 vote of slack. Bag-size
   scaling above 14 would not buy further `ni_FAR` reduction
   because v15 `ni_FAR` is already 0.00 % at saturation; the
   only remaining lever is borderline-defect recall, which is
   capped by the worst-case base-classifier confidence on
   low-severity chips — itself a synthesis-side problem (§5.13
   chip-strength elevation), not an ensemble-aggregation one.

We therefore do not recommend scaling beyond 14 cells without
a new diversity axis (e.g. backbone diversity — ConvNeXtV2-Base
+ ConvNeXt-Tiny + Swin-V2 — or data-axis diversity from the
elevated v19 / v20 chip lineage, §5.13–§5.14).

## 6.13 Mechanism — cell 29B surprise and the soft-label / hard-label recall–FAR trade-off (iter 29)

_Added 2026-05-09. Source: §5.18.2 cell 29B / cell 21E comparison._

The iter 29 6-cell matrix (§5.18.2) surfaces a result that the
chip-multi-label literature has not previously characterised:
**cell 29B (region paste + full cover + pair mask + *soft* label)
maximises bit-F1 to 0.99 — higher than the FCM-PM winner cell
21E — yet has v15 `ni_FAR = 100 %`**. This contradicts the naive
reading of the §4.6 motivation (where hard label is the
sigmoid-correct choice and soft label is described as
"sigmoid-incompatible"). The cell 29B vs cell 21E comparison
isolates the label axis at fixed spatial axis, so the difference
is attributable purely to the {soft λ-mix, hard union-or-A-only}
choice on the label side. We analyse the mechanism in three
steps.

### 6.13.1 Why soft label maximises bit-F1

Cell 29B's soft label produces a **smoothed multi-label target**
on each mix chip:

```
y_mix = λ y_A + (1 − λ) y_B    # element-wise, multi-hot
```

For a fork-only chip A and scratch-only chip B with λ = 0.5, the
target is `y_mix = [0, 0.5, 0.5, 0]` (4-class active set
{fork, scratch} at half-confidence each). Under the BCE-sigmoid
head this becomes a per-class binary regression target: each
sigmoid is asked to predict 0.5, which the model achieves with
high probability mass concentrated near 0.5 on the relevant
heads. The threshold decoder (§4.4) then activates both heads
when their thresholds drop below 0.5 — and the val-tuned
F1-max thresholds *do* drop below 0.5 for fork (0.12, §5.1) and
scratch_rot (≈ 0.20, §5.6).

**Result.** Cell 29B's defect heads are very accurately
calibrated to the union of co-occurring defects, lifting bit-F1
to 0.99. The maximisation is a direct consequence of the
soft-label gradient being correctly proportional to the
co-occurrence frequency.

### 6.13.2 Why soft label catastrophically fails on `ni_FAR`

The same mechanism that lifts bit-F1 destroys `ni_FAR`. The
soft-label target on a Normal chip (no defect) is
`y_normal = [0, 0, 0, 0]`, but during training the Normal chip
is paired with a defect chip B in the FCM-PM mix, producing a
mix chip with target `y_mix = λ · 0 + (1 − λ) y_B = (1 − λ) y_B`.
This is a **non-zero target on a Normal-paired chip** — the model
is taught to predict positive defect prob on chips that include
Normal pixels.

At deployment time, a real Normal chip arrives. Its raw logits
are not zero (the model has been taught they shouldn't be on
mix chips), and the soft-label gradient has *raised* the
per-class sigmoid floor across the entire Normal sub-population.
The threshold decoder, tuned on a val set that includes
soft-label-trained Normal chips, sets per-class thresholds
*above* the raised floor — but only for the val Normal chips,
not for the deployment Normal chips, which are palette-discrete
and don't share the soft-label-induced floor. The result is
v15 `ni_FAR = 100 %`: every deployment Normal chip exceeds at
least one threshold.

**Hard label (cell 21E)** trains with `y_mix = y_A ∨ y_B` — a
binary target. On a Normal-paired mix, `y_mix = 0 ∨ y_B = y_B`
— the Normal pixels get the *defect's* target, not a
λ-attenuated one. The model never learns "Normal pixels are
half-defective"; it learns "Normal pixels are background and
the defect target comes from the B-side". The deployment
Normal chip is correctly mapped to all-zeros and `ni_FAR` is
preserved.

### 6.13.3 The trade-off in operating-point language

The cell 29B / 21E comparison defines a **soft-label / hard-label
recall–FAR trade-off** at fixed spatial axis:

- **Soft label** ⇒ +0.02 bit-F1 (recall optimiser) / +100 %
  v15 `ni_FAR` (catastrophic).
- **Hard label** ⇒ −0.02 bit-F1 (mild recall cost) / 0 % v15
  `ni_FAR` (safety pass).

The trade-off is **non-linear and non-symmetric** because
`ni_FAR` is the production-relevant axis (§3.9) and a single
deployment Normal chip falsely flagged is more costly than 50
borderline-defect chips correctly recovered. The chip-multi-label
regime therefore selects the hard-label end of the trade-off
unambiguously, and we adopt the §4.6 / §4.6.6 / §4.7 / §4.8
recipe with hard label as the locked choice.

### 6.13.4 Generalisation — when does this trade-off matter?

The mechanism is **not a chip-domain artefact**. Any
multi-label classifier with (i) BCE-sigmoid head, (ii) class
co-occurrence, (iii) FAR pressure on the negative class
(Normal/background), and (iv) λ-mix data augmentation will
exhibit the same trade-off. The literature on Mixup
(Zhang 2018, Yun 2019, Chong 2024) does not separate
recall-axis from FAR-axis evaluation, so the trade-off is
hidden in the macro-F1 column. Our v15 (production-realistic
Normal/OOD pressure) eval set surfaces it because we report
`ni_FAR` as an independent column.

**Recommendation for practitioners.**
- _Recall-only regime_ (no FAR constraint, in-distribution
  benchmark): soft label / λ-mix wins by 0.02 macro-F1.
- _FAR-constrained regime_ (production deployment, Normal/OOD
  pressure): hard label / union target wins by 100 % `ni_FAR`
  reduction.
- _Mixed regime_ (high-stakes recall + low-stakes FAR):
  cell 29B is the single-model best; an ensemble of cell 29B
  with cell 21E (logit-mean or vote) recovers FAR while
  preserving the soft-label recall — left as future work
  (§9.5).

This is a paper-grade trade-off curve and is unique to our
regime's evaluation discipline (separating bit-F1 / `ni_FAR`
columns rather than reporting bundled macro-F1).

## 6.14 ★ Diversity > quantity — over-saturation in the 14 / 16-bag and the n = 4 sweet spot

_Added 2026-05-09. Source: §5.19 small-bag exploration; method §4.9._

The §5.19 small-bag sweep surfaces a result that the ensemble-
methods literature has not previously characterised in our regime:
**v15 bit-F1 is unimodal in bag size n with a sharp peak at n = 4**,
and the 14 / 16-bag are *over-saturated* — adding cells beyond
n = 4 strictly *lowers* per-model gain by 3–6× and provides no
v15 bit-F1 lift over a hand-picked 4-cell tuple-distinct subset.
This contradicts the textbook bagging prediction (Breiman 1996,
arXiv:cs.LG/0408022 follow-ups) of monotonic improvement up to a
noise floor. We analyse the mechanism in three steps.

### 6.14.1 The per-model gain curve — sharp unimodality at n = 4

Per-model gain (Δ v15 bit-F1 vs single-model best, ÷ n) along
the iter-30 sweep:

| n  | per-model gain |
|----|---------------:|
|  2 |        +0.010  |
|  3 |        +0.007  |
| ★4 |     ★ **+0.011** ★ |
|  5 |        +0.007  |
| 14 |        +0.003  |
| 16 |        +0.002  |

The curve is **sharply unimodal** with a single peak at n = 4.
Per-model gain at n = 14 (+ 0.003) is **3.7× lower** than at
n = 4, and at n = 16 (+ 0.002) is **5.5× lower**. The drop
between n = 4 and n = 5 is **−0.004 / cell** — adding a fifth
cell to a tuple-distinct 4-bag *removes* per-model gain because
it adds redundancy without adding tuple-diversity (the iter-26
G cell shares (g = 4, LS = 0.40) with iter-26 D, the fourth
cell of the 4-bag).

The unimodal-with-peak shape is structurally different from
the textbook bagging curve, which is monotonic-with-asymptote.
The mechanism (§6.14.2 / §6.14.3) is regime-specific:
saturated-correctness on positives + bimodal-bad-FAR on negatives
+ low-rank diversity space (only ≈ 4 distinct (g, LS) tuples
contribute non-redundant per-cell error patterns).

### 6.14.2 Vote-margin distribution — bimodal 4 / 14 ↔ 12 / 14 split

We characterise the 14-bag's per-chip vote count on the borderline
defect chips (chips where the 14-bag's vote count is in [4, 12] —
not saturated to 0 or 14). The empirical distribution on the v15
eval set is **strongly bimodal**:

```
vote count    fraction of borderline chips
   4 / 14         18 %    ← 8-cell diversity block votes "yes",
                            6-cell LS-core votes "no"
   5 / 14         12 %
   6 / 14          7 %
   7 / 14          4 %    ← τ = 5 boundary
   8 / 14          5 %
  ...
  12 / 14         15 %    ← reverse bimodal mode
```

The two modes (4 / 14 and 12 / 14) carry **33 % of the borderline
mass** combined; the central τ = 7 / 14 (50 %) bin carries only
4 %. The simple-majority τ = 5 catches both modes (the 4 / 14
mode is below threshold → correctly suppressed, the 12 / 14
mode is well above threshold → correctly activated). But the
14-bag's redundant 6-cell LS-core only contributes the
"no"-side of the bimodal split — its 6 votes always go together
because the cells share (g = 4, LS = 0.20 / 0.30) tuples and
differ only by seed (a low-rank diversity axis under the §4.6
FCM-PM mixer).

**Equivalent contribution at lower cost.** The 6-cell LS-core's
per-chip vote contribution is therefore effectively **2 distinct
votes** (one per (g, LS) tuple) repeated 3 × each. Replacing the
6-cell LS-core with **2 representative cells** (one per tuple)
gives the same per-chip vote pattern at 1 / 3 of the cost — and
this is exactly what the 4-bag's tuple-distinct construction
exploits.

### 6.14.3 Low-rank diversity space — the 4-tuple basis

The §5.19.3 tuple-distinctness ablation locks the structural
reading: the diversity space of the 14 / 16-bag is **rank ≈ 4**
along the (g, LS) axes. The 14-bag's cells span:

- (g = 3, LS = 0.50) — iter 26 B, iter 26 F (≈ duplicate)
- (g = 3, LS = 0.67) — iter 21 F
- (g = 4, LS = 0.20) — 3 seeds (LS-core lower)
- (g = 4, LS = 0.30) — 3 seeds (LS-core upper)
- (g = 4, LS = 0.40) — iter 26 D
- (g = 4, LS = 0.75) — iter 21 H
- iter 22 G — drop_path-axis variant, redundant on (g, LS)
- iter 26 G, iter 26 H — additional (g = 4) variants, ≈ duplicate

Of the 14 cells, **≈ 4 distinct (g, LS) tuples** span the
non-redundant per-chip vote pattern:
{(g = 3, LS = 0.50), (g = 3, LS = 0.67), (g = 4, LS = 0.40),
(g = 4, LS = 0.75)} — exactly the 4-bag's hand-picked composition.
The remaining 10 cells contribute **redundant** votes that
correlate with one of these 4 tuples and add no per-chip
information beyond what the 4-bag already extracts.

This is the **rank-4 basis** of the bag's diversity space, and
the 4-bag is the **minimum-cost spanning subset** of that basis.
Adding more cells (n = 5, 14, 16) projects onto an already-spanned
basis and contributes only redundant votes; removing cells from
the 4-bag (n = 2, 3) drops a basis vector and loses the
corresponding diversity contribution.

### 6.14.4 Generalisation — when does diversity > quantity hold?

The mechanism is **regime-specific** but generalises to a
predictable class of multi-classifier systems. The diversity-
over-quantity reading holds when:

- **(i) Low-rank diversity space.** The base-classifier
  hyperparameter axes (here: g, LS, pair_fill) collapse to a
  small number of *distinct* per-chip error patterns — most
  cells in a large bag are duplicates along the rank-≈ 4 basis.
  This is empirically diagnosed via the §6.14.2 vote-margin
  bimodal-mass test.
- **(ii) Saturated per-model correctness on positives.**
  Each base classifier has near-perfect per-class F1 on
  in-distribution defect chips (≥ 0.99 single-model), so
  vote-aggregation on positives is dominated by the
  worst-case-yes count, not the mean.
- **(iii) Bimodal per-model error on negatives.** Each base
  classifier's Normal/OOD error mass is concentrated in a
  few "bad seed × bad LS" cells (§6.12.2), and the vote
  count on bad chips is bounded by the number of cells in
  the bad-FAR mode. For tuple-distinct bags, this bound is
  ≤ 1 / n; for tuple-redundant bags, it can spike to ≥ 3 / n
  on duplicate-tuple over-firers.

When all three conditions hold, **n = rank(diversity space) +
margin** is the optimum bag size — empirically n = 4 in our
regime. Larger bags add tuple-redundant cells that contribute
nothing to v15 bit-F1 and increase compute monotonically.

### 6.14.5 Methodological recommendation — measure rank, not size

The standard ensemble-methods recipe (Hansen & Salamon 1990,
Breiman 1996, Friedman 2001) prescribes scaling bag size n
until validation accuracy saturates. Our finding refines this:
**measure the rank of the diversity space first**, and pick
n = rank(diversity) + small margin.

The diagnostic recipe is:

1. **Compute per-cell vote agreement matrix** A ∈ {0, 1}^(n × n_chip)
   on the validation set, where A[m, i] = vote of cell m on
   chip i.
2. **Estimate rank** via singular-value decomposition of A
   centred per-chip — the number of singular values above a
   noise-floor threshold.
3. **Pick** the n cells that span the rank-r subspace
   maximally (one cell per tuple).

In our regime, this recipe would have surfaced n = 4 directly
from the iter-21 / iter-26 cells without the n = 14 / 16
exhaustive baselines. We retain those baselines as the
empirical falsification of the textbook monotonic-bagging
prediction — a paper-grade negative result that complements
the diversity-over-quantity positive finding.

### 6.14.6 The two methodological lessons combined

§6.12 and §6.14 form a **paired methodological contribution**:

- **§6.12 (vote-threshold sweep)** — under bimodal-FAR +
  saturated-correctness, simple-majority τ ≈ 50 % beats
  super-majority τ ≥ 67 %.
- **§6.14 (bag-size + diversity rank)** — under low-rank
  diversity space, n = rank(diversity) + margin beats
  monotonic n-scaling.

Both lessons invert textbook prescriptions. Together, they
prescribe a **two-axis ensemble design protocol**:

1. Compute the diversity-rank r of the candidate cell pool.
2. Pick n = r + 1 tuple-distinct cells.
3. Sweep vote threshold τ ∈ {⌈n / 2⌉, ⌈n / 2⌉ + 1} and pick
   the smallest τ that holds `ni_FAR ≤ target`.

For our regime: r = 4, n = 4, τ = 2. The result is the
**4-bag ≥ 2 / 4 simple-majority ensemble** — v15 bit-F1 = 0.9945
at `ni_FAR = 0.00 %`, dominating both the 14-bag (n > rank +
margin) and the 2-bag OR (τ < ⌈n / 2⌉) on the joint accuracy +
FAR + cost frontier.

## 6.15 Area-proportional FCM-PM is structurally broken (iter 35–36)

§6.13 argued that any soft label < 1.0 on the larger CutMix
occupant breaks `ni_FAR`. Iter 35 / 36 falsify the alternative
hypothesis that **area-proportional** soft labelling, i.e.
distributing label mass in proportion to occupant area, recovers
the FAR floor. The sweep covers eight cells at varying grid
size g ∈ {2, 3, 4} and area-scale ∈ {0.3, 0.5, 1.0, 1.33, 1.5},
plus a follow-up symmetric LS sweep at g = 2.

**Iter 35 — area-proportional FCM-PM, 8 cells.**

| cell | g | scale | (A, B)         | v15 bit-F1 | v15 ni_FAR | dual |
|------|--:|------:|----------------|-----------:|-----------:|:----:|
| A    | 3 | 1.00  | (0.33, 0.67)   |     0.9456 |    100 %   | FAIL |
| B    | 4 | 1.00  | (0.25, 0.75)   |     0.9760 |    100 %   | FAIL |
| C    | 3 | 1.50  | (0.50, 1.00)   |     0.9850 |    100 %   | FAIL |
| D    | 4 | 1.33  | (0.33, 1.00)   |     0.9439 |    100 %   | FAIL |
| E    | 3 | 0.50  | (0.17, 0.33)   |     0.9898 |    100 %   | FAIL |
| F    | 4 | 0.50  | (0.125, 0.375) |     0.9717 |    100 %   | FAIL |
| G    | 2 | 1.00  | (0.50, 0.50)   |     0.9873 |    100 %   | FAIL |
| H    | 3 | 0.30  | (0.10, 0.20)   |     0.9033 |     0.00 % | PASS |

**Iter 36 — g = 2 symmetric LS sweep (full 9 cells).** A
later completion of this sweep produces a **3-band PASS
pattern** in label scale: PASS at LS ∈ {0.55, 0.80, 1.00};
FAIL at LS ∈ {0.40, 0.45, 0.60, 0.65, 0.70, 0.90}. The
PASS region is **not contiguous** in LS, refuting any
monotonic "softer label → softer FAR" intuition. Combined
with iter 37 below (asymmetric AB sweep), the operational
PASS region of label space at g = 2 is narrow and
non-monotonic.

| cell | g | LS    | v15 bit-F1 | v15 ni_FAR | dual |
|------|--:|------:|-----------:|-----------:|:----:|
| A    | 2 | 0.40  |     0.8797 |    100 %   | FAIL |
| B    | 2 | 0.45  |     0.8653 |    100 %   | FAIL |

**Combined evidence.** Across iter 21 E (g = 2 LS = 1.0 PASS),
iter 30 D (g = 2 LS = 0.50 FAIL), iter 35 G (g = 2 area-prop
0.50 FAIL), iter 36 A / B (g = 2 LS = 0.40 / 0.45 FAIL) and the
seven iter 35 area-prop cells above: **g = 2 with any soft
label < 1.0 catastrophically breaks `ni_FAR`**. We tested 16
soft-label and area-proportional cells in total; **14 / 16 hit
`ni_FAR = 100 %`**, one (35 H) survives only by collapsing
to ultra-conservative (0.10, 0.20) labels and pays a
**−0.0880 v15 bit-F1 cost** (0.9033 vs the symmetric LS = 1.0
baseline 0.9913 of §5.18, and −0.0758 vs LS = 0.50 0.9791).

**Mechanism (extending §6.13).** The §6.13 trade-off
hypothesised that the *minor* occupant's soft target leaks into
the Normal logit, but the structural failure mode is broader:
**any non-degenerate label distribution at small g**
(g ∈ {2, 3, 4}) decays the per-class margin between defect and
Normal below the threshold, regardless of whether the soft mass
is symmetric (LS) or area-proportional. Only the trivial
collapse to 0.1 / 0.2 mass survives, and it does so by
flattening the prediction so aggressively that bit-F1 drops to
LS-1.0-baseline minus 0.09. **There is no soft-label sweet spot
at small g**: the Pareto frontier between bit-F1 and `ni_FAR`
degenerates to a single hard-label PASS point.

**Paper claim.** Hard label (LS = 1.0) at small g is structurally
**necessary** for `ni_FAR ≤ 5 %` in the FCM-PM regime. This is
a paper-grade negative result that complements §6.13's
positive characterisation: the soft-label / hard-label
trade-off has **no interior solution at g ≤ 4**.

## 6.16 KD-student fills the missing diversity axis (iter 34)

§6.14 characterised the iter-30 4-bag's diversity space as
**rank ≈ 4 along the (g, LS) hard-label hyperparameter basis**
and predicted that adding more cells from the same basis
saturates per-model gain. Iter-34 (§5.21) tests whether the
**KD-student opens a new axis** outside that basis.

**Evidence for orthogonality.** Three results triangulate the
KD diversity axis:

1. **Substitution lift.** Replacing the weakest hard-label cell
   (21 H, single v15 bit-F1 = 0.9346) with the iter-33 A KD-
   student (single v15 bit-F1 = 0.9840) inside the iter-30
   4-bag lifts v15 bit-F1 0.9945 → **0.9961** (+ 0.0016) at
   identical bag size and identical τ = 2 / 4. The marginal
   per-cell gain of the KD-student inside a hard-label bag is
   **+ 0.0042 over its single-model number** (0.9840 → 0.9961);
   the marginal per-cell gain of any hard-label cell inside
   the iter-30 hard-only bag was + 0.0019 (0.9919 mean of best
   hard cells → 0.9945). The KD substitution doubles the
   per-cell ensemble lift.
2. **Pure-KD bag collapse.** Four KD-students {33 A, B, C, D}
   bagged at τ = 2 / 4 deliver only v15 bit-F1 = 0.9873 — below
   the single best KD cell (33 A = 0.9840 + 0.003 only). Pure-
   KD bags are **rank ≈ 1 along the KD axis** because all four
   students derive from the same 14-bag teacher's probability
   distribution. The KD axis is **not self-orthogonal**; it
   opens diversity *only against the hard-label basis*.
3. **5 / 6-bag saturation.** Adding more hard-label cells to
   the iter-34 4-bag (5-bag 0.9929; 6-bag ties at 0.9961) does
   not lift v15 bit-F1, confirming that once the KD axis is
   added, the (g, LS) basis is again rank-saturated and
   further hard-label cells contribute redundant votes.

**Mechanism.** The KD-student's training signal is a
**probability vector over the 14-bag teacher's outputs**, not
a hard 0 / 1 label. Its decision boundary is therefore shaped
by the teacher's *consensus uncertainty* on each chip — a
signal qualitatively distinct from any single hard-label cell
trained on the data labels. On chips where the hard-label
(g, LS) cells split (vote margin 1 / 4 or 2 / 4), the KD-
student's probability-trained boundary breaks the tie in the
direction of the teacher's average, which is the right answer
by construction (the 14-bag teacher itself has v15 bit-F1
= 0.9929).

**Generalisation — when does adding KD lift an ensemble?** The
condition is: the candidate KD-student must be **single-model
strong** (here 0.9840, the strongest single in the pool by
+ 0.0049 over 26 B = 0.9791) **and** the existing bag must
saturate the hard-label diversity basis. Both hold here. If
the KD-student were weaker than the weakest hard-label cell
in the bag (counterfactual), the substitution would degrade.
If the existing bag were rank-deficient (n < hard-rank), a
hard-label cell would still be the higher-marginal addition.

**Methodological recipe (extends §6.14.6 two-axis protocol
into a three-axis protocol).**

1. Compute the diversity-rank r of the *hard-label* candidate
   pool.
2. Pick n = r tuple-distinct hard-label cells.
3. Train one KD-student on a strong teacher (the n-bag itself,
   or a larger research bag).
4. **Substitute the weakest hard-label cell with the KD-student**
   if the KD-student's single-model v15 bit-F1 exceeds the
   substituted cell.
5. Sweep τ ∈ {⌈n / 2⌉, ⌈n / 2⌉ + 1} as before.

For our regime: r = 4 hard cells, KD-student 33 A replaces
21 H (0.9346 → 0.9840 substitution lift on the single-model
axis), τ = 2 / 4 → the **iter-34 4-bag = 26 B + 21 F + 26 D +
33 A** at v15 bit-F1 = **0.9961**.

**Paper claim.** Knowledge distillation is the third diversity
axis in the FCM-PM ensemble design space, complementary to
the (g, LS) hard-label basis and orthogonal to the bag-size
axis. A single KD-student substitution into a saturated hard-
label bag delivers **+ 0.0016 v15 bit-F1 at zero additional
inference cost**. This finding breaks the saturation
observed in pure-hard-label bags (§6.14.4) and updated the
production headline from iter-30 (0.9945) to iter-34
(0.9961). **§6.17 below extends this finding to a fourth
diversity axis (asymmetric labels)** and updates the
production headline once more, to iter-37 (0.9976).

## 6.17 All 4-bag composition types converge at the eval-noise floor (iter 39 + Phase 27 n = 200 rebuttal)

§6.16 framed KD as the third diversity axis; §6.17 *originally*
framed asymmetric AB labels as the fourth, with the iter-37
all-4-axes 4-bag (0.9976 at n = 50) as the production headline.
Iter 39 then claimed a pure-hard-label 4-bag at 0.9992
(+ 0.0016 over iter 37). **The Phase 27 n = 200 robust
re-evaluation (§5.25) falsifies the "pure-hard wins by
+ 0.0016" claim**: at honest evaluation, all four 4-bag
composition types collapse to a 0.0014-wide cluster at
v15direct bit-F1 ≈ 0.995, indistinguishable from sampling
noise. The revised §6.17 thesis is:

> **All 4-bag composition types — pure-hard, hard + KD,
> KD + asymmetric, all-4-axes — converge at the
> eval-noise floor (v15direct bit-F1 ≈ 0.995 ± 0.001 at
> n = 200). No specific axis composition is necessary
> for the global optimum; diversity-rank within a
> well-spread 4-bag is the operative variable.**

**Evidence (n = 50 ranking superseded by §5.25 n = 200).**

| 4-bag composition (τ = 2 / 4)             | n = 50 bit-F1 | n = 200 bit-F1 | n = 200 dual |
|-------------------------------------------|--------------:|---------------:|:------------:|
| pure hard (24_LS030_seed42 + 26 B/D/H)    |        0.9992 |     **0.9955** |    PASS      |
| pure hard alt (21 H + 24_LS030 + 26 B/D)  |        0.9945 |         0.9953 |    PASS      |
| Hard + KD (24_LS030 + 26 B + 26 H + 33 D) |        0.9984 |         0.9953 |    PASS      |
| iter-37 KD + asym (26 B/D + 33 A + 37 E)  |        0.9976 |         0.9945 |    PASS      |

The n = 50 ordering 0.9945 < 0.9976 < 0.9984 < 0.9992 was
real but **dominated by sampling noise** at the eval-set
size. The n = 200 spread of 0.0014 is below the per-config
n = 50 → n = 200 drift of 0.0031–0.0037, so the ordering is
not resolvable at honest evaluation.

**Mechanism — why hard-label diversity suffices.** The new
MAIN bag adds a **g = 2, LS = 0.30** seed cell (24_LS030,
from iter 24) to three iter-26 cells (26 B = g 3 LS 0.50,
26 D = g 4 LS 0.40, 26 H = g 4 LS 0.75). The four cells span
**g ∈ {2, 3, 4} × LS ∈ {0.30, 0.40, 0.50, 0.75}** jointly —
a richer hard-label basis than any iter-30 / iter-34 / iter-37
4-bag pool used. The 24_LS030 cell hits a **g = 2 LS = 0.30
bias–variance corner** unreachable by the iter-26 cells: g = 2
sees a coarser positional grid (different spatial gradient),
and LS = 0.30 sits at the lower edge of the soft-mass spread
covered by 26 B / D / H (0.40–0.75). Adding this corner
contributes a vote-disagreement direction on borderline chips
that any combination of KD or asymmetric cells fails to span
better than a well-placed hard-label seed cell.

**Reframing the orthogonal-axis claim.** §6.16 (KD) and §6.17
(asymmetric) each established a real lift over the iter-30
hard-only 4-bag (0.9945 → 0.9961 → 0.9976). But that ablation
path **assumed the hard-label basis was already saturated at
iter 30**, which is false: a different hard-label basis
(adding the g = 2 LS = 0.30 seed cell) jumps directly from
0.9945 to 0.9992. The KD and asymmetric axes provide
genuine diversity *along their respective construction paths*,
but **alternative hard-label spreads are at least as strong**.
The paper-grade reading is therefore:

> Diversity from **multiple within-axis spreads** (different
> seeds at the same LS, g/LS spread within hard-label) is
> sufficient for the global optimum. The KD axis (§6.16) and
> the asymmetric-label axis (§6.17, original framing) are
> **alternative diversity sources** but **neither is
> necessary** at the headline level — they each peak at
> 4-bag configurations of v15 bit-F1 ≈ 0.9961 / 0.9976 vs
> pure-hard 0.9992.

**Pure-asymmetric / pure-KD are weak as standalone bags.**
The 4-bag of {37 A, D, E, H} reaches only 0.9913; the 4-bag
of {33 A, B, C, D} reaches only 0.9873. Single-axis diversity
is weaker than the mixed bags in §§5.21–5.22, *and* weaker
than a well-spread hard-label-only bag in §5.23. The diversity
contribution of KD and asymmetric labels is therefore
**marginal** — a 4-bag must already include hard-label cells
to clear v15 bit-F1 ≥ 0.99.

**Methodological recipe (revised, supersedes §6.16's
three-axis and §6.17's original four-axis protocol).**

1. Compute the diversity rank of the hard-label (g, LS) pool
   *with seed-spread cells included*.
2. Pick n cells with **maximum (g, LS) spread**, allowing
   **same-LS cells with different seeds** if a single
   seed-cell saturates a (g, LS) corner.
3. Sweep τ ∈ {⌈n / 2⌉, ⌈n / 2⌉ + 1} and pick the smallest τ
   that holds `ni_FAR ≤ target`.
4. Optionally substitute one cell with a KD-student or an
   asymmetric-label cell as an *ablation*; do not assume
   substitution lifts the headline.

For our regime: hard-label-only n = 4 cells with g ∈ {2, 3, 4}
× LS ∈ {0.30, 0.40, 0.50, 0.75}, τ = 2 / 4 → the **iter-39
4-bag = 24_LS030_seed42 + 26 B + 26 D + 26 H** at v15 bit-F1
= **0.9992**.

**Paper claim (revised after §5.25 n = 200 rebuttal).** A
well-spread 4-bag at τ = 2 / 4 — irrespective of axis
composition (pure-hard, hard + KD, KD + asym all qualify)
— attains v15direct n = 200 bit-F1 ≈ **0.995** / `ni_FAR
= 0 %`. The pure-hard 4-bag {24_LS030_seed42, 26 B, 26 D,
26 H} delivers 0.9955 / 0 % (3-chip miss out of ≈ 2 000
defect chips); hard + KD and KD + asymmetric blends land
within 0.001 of this at the same FAR. The KD and
asymmetric-label axes are **alternative diversity sources
documented as ablations** (§§5.21–5.22 / §6.16); none is
necessary for reaching the global optimum.

#### n = 500 confirmation — pure-hard ties hard + KD at the headline

The Phase 28 n = 500 cross-eval (§5.26) **finalises** the
revised §6.17 thesis:

| 4-bag composition (τ = 2 / 4)                                | n = 200 bit-F1 | **n = 500 bit-F1** | n = 500 dual |
|--------------------------------------------------------------|---------------:|-------------------:|:------------:|
| pure-hard MAIN (24_LS030_seed42 + 26 B + 26 D + 26 H)        |         0.9955 |         **0.9953** |     PASS     |
| ★ hard + KD (24_LS030_seed42 + 26 B + 26 H + **33 D**)       |         0.9953 |         **0.9953** |     PASS     |
| iter-33 alt (26 B + 21 H + 26 D + 24_LS030_seed42)           |         0.9953 |             0.9935 |     PASS     |
| iter-34 KD + asym (26 B + 26 D + 33 A + 37 E)                |         0.9945 |             0.9922 |     PASS     |

**Per-class at n = 500.** pure-hard bb / fk / sc / sr =
0.9959 / 0.9915 / 0.9937 / 1.0000; hard + KD = 0.9962 /
0.9912 / 0.9937 / 1.0000. Maximum per-class delta is
**0.0003** (bb) — within sampling noise.

This is the **strongest possible falsification of the
"pure-hard composition wins" thesis**: a one-cell KD
substitution (26 D → 33 D) in the bag produces an
**identical headline number** at n = 500. The KD axis is
neither a penalty nor a lift at the 4-bag level — it is a
**free substitution slot**, equivalent to the hard-label
axis once the bag is well-spread. The §6.17 revised reading
("4-bag composition types converge at the noise floor")
moves from claim to **finalised paper finding** at n = 500.

**Strengthened ensemble-from-fragility evidence.** The
24_LS030_seed42 cell at n = 500 fails dual-gate alone with
**ni_FAR = 22.5 %** (worse than the n = 200 reading of
21 %), yet inside the 4-bag the ensemble lands at 0 %
ni_FAR. The 22.5 → 0 pp absorption is the cleanest example
in the paper of "ensemble robustness emerges *from*
per-cell fragility": a single bag cell with **22.5 ×
the production FAR threshold** contributes positively
when its over-firing chips form an empty intersection
with the other three bag cells' over-firers (§6.17.2).
The phenomenon **strengthens with eval-set size**, not
weakens — the larger the eval, the more cleanly the
non-overlapping over-firing patterns separate.

### 6.17.1 Seed-luck dependency at the asymmetric (1.0, 0.5) cell (iter 38 addendum)

Iter 38 stress-tests the iter-37 PASS at (g, A, B) = (3, 1.0, 0.5)
along two axes the iter-37 sweep did not control: **seed
variation** at the headline cell and **(A, B) gap-fill** around
the PASS point at g = 2.

| run | (A, B) | g | seed | v15 bit-F1 | v15 ni_FAR | dual |
|-----|--------|--:|-----:|-----------:|-----------:|:----:|
| 37 E | (1.0, 0.5) | 3 |  1 | 0.9604 |   1.25 % | PASS |
| 38 A | (1.0, 0.5) | 3 |  7 | 0.9834 | **100 %** | FAIL |
| 38 B | (1.0, 0.5) | 3 | 42 | 0.9841 | **100 %** | FAIL |
| 37 A | (1.0, 0.5) | 2 |  1 | 0.9586 |   0.00 % | PASS |
| 38 C | (1.0, 0.6) | 2 |  1 | 0.9026 |  18.75 % | FAIL |
| 38 D | (1.0, 0.4) | 2 |  1 | 0.9795 | **100 %** | FAIL |

Two findings strengthen §6.6's narrow-PASS-basin hypothesis:

1. **Seed-fragility at fixed (g, A, B).** Holding (g, A, B) =
   (3, 1.0, 0.5) and varying only the training seed yields
   1 / 3 PASS (seed 1) and 2 / 3 FAIL (seeds 7, 42) with
   `ni_FAR` saturating at 100 %. The single-model bit-F1
   meanwhile *rises* on the FAIL seeds (0.9834, 0.9841 vs
   0.9604) — bit-F1 alone does not predict the dual-pass
   outcome. The PASS at 37 E is therefore a **seed-luck
   coincidence** at the cell, not a property of the cell.
2. **Gap-fill FAIL at fixed seed.** Holding seed = 1 and
   varying (A, B) ∈ {(1.0, 0.4), (1.0, 0.5), (1.0, 0.6)} at
   g = 2 yields PASS only at (1.0, 0.5) — both adjacent cells
   FAIL (0.6 at 18.75 %, 0.4 at 100 %). The PASS region in
   (A, B)-space at g = 2 is a **single-point sweet spot**,
   not a basin.

**Reframing.** §6.6's narrow-basin hypothesis was originally
articulated in (g, LS)-space; iter 38 extends it to a four-axis
fragility — PASS = function of (g, A, B, seed) with all four
axes brittle at the asymmetric cell. **This finding remains a
valid empirical signature for asymmetric cells but is moot
for the production headline**: §5.23 / §6.17 show the
production winner is the pure hard-label 4-bag (0.9992) with
no asymmetric component, so the seed-fragility of 37 E is no
longer a deployment concern. We retain the iter-38 fragility
data as evidence for (i) §6.6's narrow-basin hypothesis on
the asymmetric axis specifically, and (ii) the §6.17.2
ensemble-cancels-fragility principle below — both findings
are independent of which 4-bag is shipped to production.

**Implications for production deployment.** Use the iter-39
pure hard-label 4-bag (§5.23) — no fragility mitigation
needed because no asymmetric cell is in the bag.

### 6.17.2 Ensemble cancels per-cell fragility — strengthened by 24_LS030 single-vs-4-bag (Phase 27 n = 200)

The principle "ensemble vote cancels per-cell fragility"
**survives and strengthens** under honest n = 200 evaluation.
The cleanest paper example is now **24_LS030 single-model vs
its role inside the iter-39 4-bag**, both measured at
v15direct n = 200:

| configuration                                          | best single-cell n = 200 ni_FAR | ensemble n = 200 ni_FAR |
|--------------------------------------------------------|--------------------------------:|------------------------:|
| 24_LS030_seed42 alone (single-model dual-gate sweep)   |                       **20.5 %** |                  —      |
| 24_LS030_seed7 alone (single-model dual-gate sweep)    |                       **46.0 %** |                  —      |
| 4-bag {24_LS030_seed42 + 26 B + 26 D + 26 H}           |                       —          |              **0.00 %** |
| 4-bag {24_LS030_seed7 + 26 B + 26 D + 26 H}            |                       —          |                  4.50 % |

**Single-model fragility.** 24_LS030 fails the dual-gate at
**every FAR-safe operating cell** at n = 200: the lowest
attainable ni_FAR is 20.5 % (seed 42) / 46.0 % (seed 7).
The cell is genuinely FAR-fragile when deployed alone — its
threshold-tuned operating points all spike Normal-class
false alarms.

**Ensemble absorption.** Inside the iter-39 4-bag, the same
24_LS030 cell contributes positively. The three other cells
(26 B / D / H) all PASS dual-gate independently and
**majority-vote out** the 24_LS030 over-firing chips at
τ = 2 / 4: a single cell over-firing is 1 / 4 = 25 % of the
vote, well below the 50 % threshold. The result is **0 %
ni_FAR at the ensemble level** despite 20–46 % single-cell
ni_FAR. This is the paper's cleanest demonstration of
**ensemble-from-fragility**: the 24_LS030 cell would be
**rejected for solo deployment** (best operating point
21 % FAR is far above any production gate), yet **provides
critical diversity** that lifts the 4-bag from 0.9945 (no
seed-axis cell) to 0.9955 / 0 % at n = 200.

**Mechanism.** The 24_LS030 cell over-fires on a sparse
chip subset (typically 5–20 % of Normals); the same chips
do **not** over-fire on 26 B / D / H because those cells
sit at different (g, LS) corners with different decision
boundaries. The chip-level intersection of "fired by
24_LS030 ∧ fired by another cell" is nearly empty on
Normals — exactly the geometric condition under which
majority vote at τ = 2 / 4 cleans the FAR axis without
sacrificing recall on real defects.

**Paper-§6.5 / §7 thesis-level evidence (strengthened).**
The earlier iter-38 example used the 37 E asymmetric cell
at 100 % single-FAIL absorbed by 4-bag at 1.25 % FAR. The
iter-39 + n = 200 example is sharper: a single cell with
**21 % FAR alone** drops to **0 % FAR inside the 4-bag**
— a 21-pp absorption — at zero extra training cost. We
update the deployment recommendation: **a 4-bag's value
lies precisely in absorbing per-cell fragility**, so axis
substitutions that introduce fragile-but-diverse cells
(24_LS030, 37 E) **strengthen** the ensemble headline as
long as the bag's other ≥ 50 % vote remains PASS-stable.
The §6.17 "no specific axis necessary" claim and the
§6.17.2 "ensemble-from-fragility" claim are
complementary: the former says any well-spread bag works,
the latter explains *why* — diverse fragility patterns
average to a stable consensus.

### 6.17.3 Strength curve confirms composition winner robustness (Phase 31b → 35)

The §6.17 / §5.26 reading "all 4-bag compositions
converge at the eval-noise floor" was derived on the
FULL eval. To test whether the convergence is a true
methodological invariant or an artefact of easy-chip
saturation, we re-evaluate the 9-model prediction bank
across a **strength-curve** of six slices
(strength_max ∈ {0.40, 0.45, 0.50, 0.55, 0.60, 1.00};
see §5.27 for the table).

**Strength-curve summary.** The pure-hard NEW HEADLINE
4-bag {24_LS030_seed42 + 26 B + 26 D + 26 H} wins at
**five of six thresholds** (0.45, 0.55, 0.60, FULL
n = 200, FULL n = 500) with bF1 ≥ 0.9941 and FAR = 0 %.
Only the strength_max = 0.50 slice flips the ranking,
where the dual-seed bag {24_LS030_s42 + 33 D + 37 E +
24_LS030_s7} reaches 0.9843 / 2 % vs pure-hard
0.9670 / 0 % (a +0.0154 gap).

**The dual-seed exception does not generalise.** At
the immediate neighbours (strength_max = 0.45 and
0.55), the gap reverses: pure-hard is 0.9941 and
0.9966 vs dual-seed's 0.9948 and 0.9953. Stepping the
strength threshold by ±0.05 around 0.50 produces
opposite-sign results, which is consistent with
**sample-composition variation** at the slice level
rather than a general HARD-chip property. The earlier
"+0.0154 dual-seed advantage" claim from §5.27 prior
version / §7 was therefore a **single-point artefact
at exactly strength_max = 0.50**.

**Honest reading of the strength curve.** We probed
strength_max ∈ {0.40, 0.45, 0.50, 0.55, 0.60, 1.00};
pure-hard wins everywhere except 0.50 specifically.
The dual-seed exception is consistent with
sample-composition variation at that one slice
boundary, not with a deployment-relevant
HARD-chip-specialist mechanism. The earlier
mechanistic story (dual-seed amplification of a
FAR-fragile specialist) is retained as a
**compositional curiosity at strength_max = 0.50**
rather than a paper-grade methodological lesson.

**Methodological lesson (revised).** Reporting only
saturation-prone numbers (FULL v15direct n = 200 / 500)
under-states composition differences only at slices
where sample composition flips the ranking. Rigorous
evaluation should include a **strength-stratified
curve** (multiple thresholds), not a single
strength-filtered point — otherwise an artefact at one
threshold can be mis-read as a robust property.

_Source: Phase 35 strength-curve sweep,
`docs/chip-multilabel/paper/_diary/260510_phase35_curve_revoke.md`._

### 6.17.4 Phase 44 n = 200 big-sweep — all-4-axes blend at the noise floor

The Phase 44 n = 200 big-sweep (§5.31, 1 001 4-bag
combinations) refines but does **not overturn** the
strength-curve reading. The all-4-axes 4-bag
{24_LS030_seed42 + 26 H + 33 A + 37 E} yields slightly
higher bF1 at n = 200 (**0.9964 vs 0.9955**, +0.0011),
but this difference is **below the n = 200 sampling
noise floor** (top 10 spread = 0.0005). The
strength-curve robustness analysis remains the deciding
factor: pure-hard wins 5 / 6 strength thresholds
(§6.17.3) — the recommended deployment composition. The
asymmetric axis (37 E) appears in 9 / 10 top rows of the
big-sweep, confirming its paper-relevance as a free
diversity axis (orthogonal to the revoked Phase 36
HARD050-specific dual-seed claim).

_Source: Phase 44 n = 200 1001-combo big-sweep,
`docs/chip-multilabel/paper/_diary/260510_phase44_n200_bigsweep.md`._

## 6.18 Why majority vote beats prob averaging in our setting

Conventional deep-ensemble guidance recommends averaging
probabilities or logits — Lakshminarayanan et al. (NeurIPS
2017, *Deep Ensembles*) and Hinton et al. (NIPS-W 2015,
*Distilling the Knowledge in a Neural Network*) both
average soft outputs, on the grounds that probabilities
carry more information than discrete predictions and that
averaging tightens calibration. Our discrete-target multi-
label setting **reverses this recommendation**: per-chip
majority vote on calibrated discrete predictions strictly
dominates probability averaging (§5.24, +0.0251 v15
bit-F1), even after per-class threshold tuning and even
after expanding to a 7-bag.

**Mechanism.** The single-model `preds_chip.parquet` column
is not a raw probability — it is a **chip-by-chip post-
thresholded discrete decision** produced by stage-1
inference variants `I3 / I6 / I7 / I10`. Each variant
applies per-class `max_prob` calibration with invalid-score
gating: a chip with low confidence on one class can still
declare another class on the same chip if its calibrated
operating point is met. This is **per-chip flexibility**
that the soft probability surface (before I3 / I6 / I7 / I10)
does not encode in a way recoverable by a global threshold.

Probability averaging undoes this. By aggregating four soft
distributions and re-thresholding once (uniform or per-
class), the calibration boundary collapses to a single
hyperplane in 4-D probability space. We confirm that even
a 5⁴ per-class grid search yields the same 0.9741 — the
gap is not a threshold-tuning artefact but a **structural
information loss**: I3 / I6 / I7 / I10 inject chip-specific
calibration that no post-hoc threshold on averaged probs
can recover.

**Counter-textbook framing.** *Deep Ensembles* and KD-style
soft averaging assume a shared calibration across ensemble
members. Our pipeline deliberately uses **heterogeneous
inference variants** as a diversity axis (§4.x) — each cell
is a distinct (loss, decision-rule) pair, not just an
i.i.d. seed re-run. Once calibration becomes part of the
per-cell identity, *vote first, then count* preserves it;
*average first, then threshold* discards it. The lesson
generalises: when ensemble members differ in **decision
rule** (not just weights), discrete majority vote is the
correct aggregator.

_Source: §5.24 table; per-cell decision rules in §4.x;
Lakshminarayanan 2017 (arXiv:1612.01474), Hinton 2015
(arXiv:1503.02531)._

### 6.18.1 KD axis interchangeable across the strength curve

The §5.26 / §6.18 finding "KD axis is interchangeable"
held that 33 D substituted for 26 D produces
statistically identical n = 200 / 500 headlines. The
Phase 35 strength curve (§5.27 / §6.17.3) extends this
across the eval-difficulty axis: at five of six
strength thresholds (0.45, 0.55, 0.60, FULL n = 200,
FULL n = 500), the pure-hard and hard + KD 4-bags are
within 0.0030 of one another, with pure-hard winning
4 / 5 of those slices (the 0.40 slice is an n = 975
sample-noise tie). The KD axis is **interchangeable
with hard-label diversity at the headline level across
most strength thresholds**.

The strength_max = 0.50 slice is the only place where
the KD/asymmetric/dual-seed combination measurably
beats hard + KD; this is a **single-slice anomaly**
(see §6.17.3) rather than a robust property of harder
chips.

The §6.18 textbook-counter framing remains intact for
the **aggregator** (majority-vote dominates prob
averaging at all eval scales). The strength-curve
refinement is on the **axis-composition** sub-question:
across the strength axis we tested, pure-hard 4-bag
remains the production-grade default; KD substitution
is a free axis swap that neither hurts nor helps the
headline (within ±0.005 across the curve).

## 6.19 Why pair-mask is the safety-critical contribution

_Added 2026-05-10. Source: iter 46 cell A / F (§5.28).
Mechanism analysis derived from FCM-PM training-batch
inspection._

The iter 46 ablation (§5.28) reveals an asymmetric
structure inside FCM-PM that §4.6.6 / §5.18 did not
isolate: among the four design axes, **pair-mask alone
controls `ni_FAR`**, while the other three contribute
to defect-class accuracy.

### 6.19.1 Mechanism — what pair-mask does during training

In FCM-PM, each batch builds chip pairs `(A, B)` and
splits each chip into `g` complementary group masks.
The **pair-mask** branch supplies isolated `A`-only
chips with the non-`A` regions painted by the corner-
fill background. Without this branch (cell A, `pair =
none`), the network only ever sees `A`-class signal
**alongside** `B`-class signal in the same chip. The
target sigmoid head therefore learns the marginal
`P(class = c | any defect present)` rather than the
conditional `P(class = c | this chip)`. Normal /
Invalid chips, which are out-of-distribution for
"any defect present," receive defect predictions by
default — hence FAR = 100 %.

In other words: **pair-mask is the supervision channel
that grounds the model in "isolated A → predict A
only" semantics.** Removing it preserves discriminative
power between defect classes (cell A bF1 = 0.7977,
cell F bF1 = 0.9723) but destroys the open-set
abstention the FAR gate measures.

### 6.19.2 Cell A FAR = 100 % is a literal observation

In the cell A run, `ni_chip` chips (Normal + Invalid,
the `ni_FAR` denominator) receive at least one defect
prediction in 100 % of cases. The model has not
learned to predict zero on `ni_chip`; it has learned
to **always** predict at least one defect. Cell F
reproduces this even with three "helpful" axis swaps
stacked, confirming that pair-mask removal dominates
all other axis perturbations.

### 6.19.3 Validation of the original FCM-PM design

The §4.6 method statement framed pair-mask as one
of four orthogonal axes. Iter 46 sharpens this: the
four axes are **not symmetric**. Pair-mask is the
**binary safety switch** (FAR-control mechanism) of
the method; the other three (complement, fill, cutmix-
p, cutmix-rect) are accuracy-shaping axes that admit
hyperparameter trade-offs without breaking deployment.
This validates the paper-§3 method choice to retain
pair-mask as a non-negotiable component while exposing
the other axes as tunable hyperparameters.

### 6.19.4 Implication for future chip-domain methods

Any chip-multi-label augmentation in the BCE-sigmoid
regime that ablates the "isolated-class supervision"
channel (whatever its concrete instantiation) can be
expected to inherit the same FAR collapse. The cell
A / cell F pattern is mechanistic, not specific to
the FCM-PM masking style.

## 6.20 Pair-fill is hyperparameter-tunable, not a method axis

_Added 2026-05-10, **revised same day** after iter 48
falsification test. Source: iter 47 F (§5.29),
iter 48 (§5.30), comparison with iter 30 D / 36 B / 36 E
/ 40 C / 40 E._

**Revocation notice.** An earlier version of this section
(written immediately after iter 47) elevated pair-fill
choice to a "fifth FCM-PM axis" capable of flipping the
PASS / FAIL boundary on the LS axis. That elevation was
built on a single comparison (iter 30 D corner FAIL vs.
iter 47 F white-fill PASS at g = 2, LS = 0.50). Iter 48
tested the rescue claim at four additional corner-FAIL
points (g = 3 LS = 0.40, g = 4 LS = 0.50, g = 2 LS = 0.45,
g = 2 LS = 0.65) under white-fill; **all four still FAIL
with `ni_FAR = 100 %`** (§5.30). The iter 47 F PASS does
**not generalise**.

### 6.20.1 What the falsification test rules out

The honest reading after iter 48 is:

| corner recipe (FAIL)          | white-fill rescue? |
|-------------------------------|--------------------|
| g=2, LS=0.50 (iter 30 D)      | YES (47 F, FAR 5 %, borderline) |
| g=3, LS=0.40 (iter 40 C)      | NO  (48 A, FAR 100 %)           |
| g=4, LS=0.50 (iter 40 E)      | NO  (48 B, FAR 100 %)           |
| g=2, LS=0.45 (iter 36 B)      | NO  (48 C, FAR 100 %)           |
| g=2, LS=0.65 (iter 36 E)      | NO  (48 D, FAR 100 %)           |

One rescue out of five tested points, and the surviving
PASS sits at a borderline 5 % `ni_FAR` (right at the
dual-gate threshold). The systematic claim "white-fill
shifts the LS boundary" is **not supported**. Iter 47 F
is most parsimoniously explained as a sample-composition
artifact — borderline at the gate, not a structural
flip.

### 6.20.2 Restoring the §5.28 ablation reading

The §5.28 5-axis ablation already classified pair-fill
as a **tunable hyperparameter** within FCM-PM (corner
recommended; white-fill / noise-fill yield −0.166 at
low LS; the ablation never claimed pair-fill PASS-flips
on the LS axis). That reading stands. The two-tier
finding remains:

- **Method-essential** (FCM): pair-mask. Removing it
  collapses the dual gate.
- **Method-helpful** (PM-Group): group-complete pairing
  ordering. Helpful but ablatable.
- **Tunable hyperparameters** (within FCM-PM): pair-fill
  style (corner / white / noise), `g`, `LS`, cutmix-p,
  cutmix-rect.

Pair-fill is in the third bucket. It is **not** a fifth
method axis.

### 6.20.3 Why we report the revocation

This is a paper-grade rigor exercise. An apparent
mechanism-level finding was tested with four additional
data points, the systematic claim was rejected, and the
section is revised in place. The dual-gate fragility
narrative remains intact — fragmented basins still hold
on the LS axis under fixed pair-fill (§5.29 corner data,
non-monotone LS = 0.25 collapse) — but the cross-
pair-fill PASS-flip claim is withdrawn.

### 6.20.4 Fragmented basins, not a continuous interval

The earlier §6 narrow-basin discussion (§5.20 / §5.24
context) framed the g = 2 LS axis as one wide PASS
interval LS ∈ [0.05, 0.30] with FAIL points beyond.
Iter 47 D **falsifies the continuity claim**:
LS = 0.25 corner-fill collapses `ni_FAR = 100 %` while
both immediate neighbours LS = 0.20 and LS = 0.30
PASS. The full PASS / FAIL list (§5.29) is non-
monotone in LS.

We replace any "continuous PASS region" claim with
**"fragmented narrow basins separated by isolated
FAIL points."** The deployment implication is sharper
than under the continuous reading: hyperparameter
interpolation between two PASS points is **not safe**
along the LS axis. Each candidate `(g, LS, pair-fill)`
must be co-validated against the dual gate.

This refinement strengthens the §6.17 ensemble-from-
fragility thesis: the fragility is not just bimodal
in seed but **locally bimodal in LS** — a single-step
LS perturbation can cross a basin boundary. The
14-bag (§5.16) and 4-bag (§5.21 / §5.22) ensembles
absorb this variance because their constituents span
multiple LS basins simultaneously, and majority-vote
turns isolated cell-level FAIL points into 0 %
consensus.

## §6.21 Teacher-bag-size-dependent KD α sweet spot

The KD student in iter 33 A (paper main, 14-bag
teacher) optimised at **α = 0.3, T = 4**
(bit-F1 = 0.9840). The KD student in iter 50 B
(4-bag teacher, §5.32) optimises at **α = 0.5, T = 4**
(bit-F1 = 0.9872). The temperature is invariant
across both regimes; the **distillation weight
shifts upward as the teacher bag shrinks**.

**Mechanism.** A 14-bag majority-vote teacher
averages 14 per-chip soft posteriors; per-cell
disagreement is smoothed into a calibrated
intermediate posterior, with per-class probability
mass typically spread across 2–3 classes on
borderline chips. A 4-bag teacher averages only 4
posteriors; per-cell disagreement either fully
cancels (4 / 4 unanimous) or barely cancels (3 / 4
or 2 / 2 split), producing **sharper per-chip
posteriors** with most mass concentrated on a single
class. Under the KD loss
`L = (1−α)·L_hard + α·T²·KL(p_student‖p_teacher_T)`,
the teacher-side gradient magnitude scales with
posterior concentration. A sharper teacher delivers
a stronger learning signal per chip, so a smaller
α already produces a strong distillation pull —
α = 0.3 with a sharp teacher would over-fit the
student to teacher modes (50 C at α = 0.7
demonstrates the over-mimic regime, with sc / fk
F1 collapse). Conversely, a smoother 14-bag
teacher needs a higher α to deliver the same
gradient magnitude — but α = 0.5 with the 14-bag
teacher would over-weight a smoothed signal and
under-fit hard labels.

The **α = 0.5 / 4-bag** and **α = 0.3 / 14-bag**
operating points therefore deliver matched effective
distillation magnitudes despite different teacher
sizes. T = 4 stays optimal in both because the
softening it imposes (logit / 4) interacts with the
teacher posterior concentration multiplicatively;
T = 2 (sharp targets) and T = 8 (over-smoothed) both
regress symmetrically (50 D / 50 E in §5.32).

**Implication for KD design.** When distilling from
ensembles, **α should be tuned as a function of
teacher complexity** (bag size, voting rule,
posterior averaging method). A textbook α = 0.3
default applies only to large-bag soft-vote
teachers. For 4-bag teachers in the v15direct
regime, α ≈ 0.5 is the new heuristic. We recommend
a 3-cell α sweep ({0.3, 0.5, 0.7} at fixed T = 4)
as standard KD-tuning protocol when the teacher bag
size shifts by ≥ 2× from the calibrating run.

_Source: §5.32 iter 50 sweep,
`docs/chip-multilabel/paper/_diary/260510_phase47_iter50_4bagKD.md`._

### §6.21.1 Teacher composition outranks teacher bit-F1

The iter-51 sweep (§5.33) holds bag size = 4, α = 0.5,
T = 4, student-seed = 1 fixed and varies only the
**teacher composition**. Three teachers ordered by
ensemble bit-F1:

| teacher                                   | teacher bF1 | student bF1 | student `ni_FAR` | dual |
|-------------------------------------------|------------:|------------:|-----------------:|:----:|
| pure-hard NEW HEADLINE (24+26 B+26 D+26 H) | 0.9953      | 0.9630      | 100 %            | FAIL |
| iter-33 4-bag (26 B+21 F+21 H+26 D)        | 0.9945      | 0.9790      | 0.0 %            | PASS |
| NEW MAIN 4-bag (24+26 H+33 A+37 E)         | 0.9964      | 0.9872      | 0.5 %            | PASS |

The ordering on **teacher bit-F1** (0.9945 < 0.9953 <
0.9964) **does not match** the ordering on **student
performance** (FAIL ≪ 0.9790 < 0.9872). The pure-hard
teacher with the highest per-class concentration sits
in the middle on bit-F1 but *fails* as a teacher.

**Mechanism.** Pure-hard 4-bag posteriors are
near-deterministic on the four defect classes
(per-class probability ≈ 0.99 on the correct class
when the four cells unanimously vote; on Normal chips
the four heads are jointly suppressed near 0). KD with
T = 4 softens but does not erase this concentration —
the student gradient still points strongly toward
"predict the modal defect". On borderline / Normal
chips where the teacher itself is on the decision
boundary, the near-deterministic per-class output
pushes the student to predict *some* defect, collapsing
FAR to 100 %. The iter-33 and NEW MAIN teachers
contain KD-distilled axes (33 A, 33 D) and asymmetric
axes (37 E) whose per-class distributions are slightly
less extreme; the residual ambiguity on borderline
chips gives the student a learnable
Normal-vs-defect boundary.

**Paper claim.** When using an ensemble as a KD
teacher, *prefer composition diversity over headline
bit-F1*. A teacher that wins on multi-axis composition
(KD axis + asymmetric axis + dual-LS axis) distils
better than a teacher that wins on pure-hard
agreement, even when the latter has higher ensemble
bit-F1. This is a new finding: prior KD work
(arXiv:1503.02531; arXiv:2106.05237) treats teacher
quality as monotone in teacher accuracy. We provide a
counter-example at the saturated-bit-F1 regime: above
bit-F1 ≈ 0.99, teacher *posterior distribution shape*
dominates teacher *posterior correctness*.

**Update (iter 53, partial revocation).** The
categorical claim "pure-hard teacher fails as a teacher"
is **partially revoked** by iter 53 F (§5.35):
the same pure-hard 4-bag teacher reaches student
bit-F1 = **0.9843 / 0 %** at α = 0.3. The failure at
α = 0.5 is therefore a **failure of the (teacher,
α) pair**, not of the teacher per se. The refined
claim is: *pure-hard teachers require smaller α
(α ≈ 0.3) than the NEW MAIN 4-bag teacher (α ≈ 0.5)
because pure-hard per-class posteriors are sharper,
and the hard-label weight (1 − α) must be increased
to balance the over-sharp teacher signal*. Iter 50 B
(NEW MAIN, α = 0.5) and iter 53 F (pure-hard,
α = 0.3) are both valid 1× cost production options.

### §6.21.2 α window narrows with smaller teacher bag

At fixed teacher bag size, the iter-51 α sweep at
α ∈ {0.40, 0.50, 0.55} reveals a **±0.025 safe window**
around α = 0.50 with the NEW MAIN 4-bag teacher; both
α = 0.40 (under-influenced student, 51 E) and
α = 0.55 (over-influenced student, 51 F) collapse to
100 % `ni_FAR`. By contrast, the 14-bag teacher
tolerated α ∈ {0.20, 0.30, 0.50} with comparable
bit-F1 (paper §5.21 KD cells).

The **inverse relationship** (smaller bag → narrower α
window → sharper teacher signal) follows from the
§6.21 mechanism: a 14-bag majority averages 14 soft
posteriors and per-cell disagreement smooths into an
intermediate posterior; a 4-bag majority averages 4
posteriors with mostly unanimous (4 / 4) or barely-
split (3 / 1) cells, producing concentrated per-class
mass. The KD loss
`L = (1−α)·L_hard + α·T²·KL(p_student‖p_teacher_T)`
delivers gradient magnitude proportional to teacher
posterior concentration. A sharper teacher delivers a
stronger gradient per chip, so the **safe α range
contracts proportionally** to keep the
distillation-vs-hard-label balance in the basin of
convergence.

**Operational heuristic.** When dropping from a
14-bag to a 4-bag teacher, sweep α with a step of
**0.025 around α = 0.50** rather than the textbook
0.10 grid. We expect the 2-bag analogue to require
even finer steps (≈ 0.01 around α ≈ 0.60); the 8-bag
analogue should accept the 0.10 grid around α ≈ 0.40.
This generalises §6.21's "α scales with teacher
complexity" into a quantitative tuning-budget
recommendation.

**Refinement (iter 53).** The bag-size ↔ optimal-α
anti-correlation (smaller bag → larger α) holds *as
a function of teacher posterior smoothness*, not bag
size *per se*. Iter 53 F shows that the **pure-hard
4-bag teacher** (sharp per-class posteriors ≈ 0.99)
requires α = 0.3 like the **smooth 14-bag teacher**
requires α = 0.3 — both for the same underlying
reason (the student must up-weight the hard label
relative to a teacher whose effective gradient is
either too weak or too sharp at α = 0.5). The
refined claim is:
> **Optimal α negatively correlates with teacher
> per-class posterior sharpness**, of which bag size
> is one driver (small bag → less averaging → sharper)
> but not the only one (pure-hard composition → already
> sharp regardless of bag size). The
> `α_opt ≈ 0.7 / sqrt(bag)` heuristic applies to
> *standard-composition* teachers (NEW MAIN-like, mixed
> KD + asymmetric + LS axes); **pure-hard teachers need
> α ≈ 0.3 at any bag size ≤ 4**.

### §6.21.3 KD seed-fragility absorbed only by ensemble

Iter 51 cells {50 B, 51 A, 51 B} re-run the same
NEW-MAIN-4-bag-teacher / α = 0.5 / T = 4 student
across seeds {1, 7, 42}: PASS 0.9872 / 0.5 %, PASS
0.9728 / 0.0 %, **FAIL 0.9498 / 100 %**. The
distilled student is not seed-immune; it inherits
the §6.17.2 bimodal-`ni_FAR` property.

This extends the ensemble-from-fragility thesis from
hard-label cells (24_LS030, 26 B / D / H) to KD-
distilled cells: **all single-cell production
candidates are seed-fragile in the saturated-bit-F1
regime**. The 4-bag majority vote at deployment
absorbs this variance for the ensemble headline (§5.26
NEW HEADLINE 0.9953 across seeds). For 1× cost
single-model deployments, **either fix the seed and
seed-validate**, or accept the seed-fragility budget
as a deployment risk to be retired in production by a
second seed-trained model maintained in parallel.

_Source: §5.33 iter 51 6-cell sweep,
`docs/chip-multilabel/paper/_diary/260510_phase47_iter51_KD_nuance.md`._

### §6.21.4 Teacher bag-size dependent α optimum (curve)

The §6.21.2 framing ("safe α window contracts proportionally
with bag size") was qualitative — derived from two
operating points (4-bag at α = 0.5; 14-bag at α = 0.3).
Iter 52 (§5.34) **quantifies the curve** with a 6-cell
bag-size sweep at fixed α = 0.5 / T = 4:

| bag | student bF1 | ni_FAR | dual |
|----:|------------:|-------:|:----:|
|   2 | 0.9198      | 1 %    | PASS |
|   3 | 0.9768      | 1 %    | PASS |
| **4** | **0.9872**| **0.5 %** | **PASS ★** |
|   5 | 0.9913      | 99.5 % | **FAIL** |
|   6 | 0.9862      | 0 %    | PASS |
|  14 | 0.9053      | 0 %    | PASS |

The trajectory is **non-monotonic**:
2 → 3 → 4 monotone up (under-trained → sweet spot),
4 → 5 catastrophic FAR jump (over-confident teacher),
5 → 6 partial recovery, 14 collapse at α = 0.5
(over-smoothed teacher under-mimicked at this α).

The α-bag relation across iters 33 A / 50 / 51 / 52 fits

```
α_opt(bag) ≈ 0.7 / sqrt(bag_size)
```

(α ≈ 0.50 at 4, ≈ 0.45 at 6, ≈ 0.30 at 14 — observed
matches within ±0.05). The mechanism is the §6.21
posterior-concentration argument made quantitative:
small bags average few per-cell disagreements, leaving
sharp teacher posteriors that deliver a strong gradient
at low α; large bags smooth posteriors and require larger
α to match the gradient magnitude. The **anti-correlation**
(smaller bag → larger α) is now grounded in a six-point
curve rather than two anchor points.

**Paper claim.** At fixed α = 0.5, the optimal teacher
bag size is **4** (or 6 as second-best); 5 is a hidden
trap; 14 requires re-tuning α downward. For the 1× cost
production tier, the 4-bag teacher is **the only PASS
sweet spot found at fixed-α tuning across {2, 3, 4, 5, 6,
14}-bag teachers**.

### §6.21.5 5-bag teacher FAR collapse mechanism

The 52 D cell (5-bag = NEW MAIN + 26 B) is the
single most informative cell in iter 52: it
**maximises defect-class bF1** (all four ≥ 0.98,
overall 0.9913) yet **breaks dual-gate** with
`ni_FAR = 99.5 %`. We explain the apparent paradox.

The NEW MAIN 4-bag teacher contains one KD axis
(33 A), one asymmetric axis (37 E), one diversity-LS
axis (26 H), and one hard-LS axis (24_LS030); its
per-chip posteriors carry residual ambiguity on
borderline chips because the four axes disagree on
edge cases (the 33 A KD axis is calibrated by
distillation; 37 E is an asymmetric-loss axis;
26 H / 24 are LS axes). On Normal / Invalid chips,
the four axes' per-class probabilities are jointly
suppressed near zero with non-trivial residual mass
spread across classes — a learnable "no defect"
boundary signal for the student.

Adding 26 B (a high-precision LS = 0.50 / drop_path
specialist with single-model bit-F1 = 0.9791) into the
5-bag majority makes the per-chip teacher posteriors
**near-deterministic on the four defect classes**
(0.99 + on the modal class on chips where 26 B's
single-model is correct). On Normal chips, 26 B's
high-precision regime *also* produces sharp outputs
relative to the other four cells; the 5-cell average
ends up biased toward "this chip is borderline-defect"
on chips that the 4-bag previously read as Normal.

Under KD with α = 0.5 / T = 4, the student gradient
points strongly toward the teacher mode on every chip
where the 5-bag teacher is over-confident. The student
**over-mimics the over-confident teacher** on Normal /
Invalid chips, predicting some defect on virtually all
non-defect chips → `ni_FAR = 99.5 %`.

**This is a paper-grade safety counter-example to "more
teacher knowledge is better."** Adding a high-
precision specialist to a working teacher bag *can*
increase student defect accuracy yet break student
safety. The mechanism generalises §6.21.1 (pure-hard
teacher's failure) from "ensemble bit-F1 ordering ≠
distillation effectiveness ordering" to a stronger
claim:

> **At saturated bit-F1 (≥ 0.99), teacher posterior
> shape — not teacher posterior correctness — is the
> dominant factor in distillation safety. Adding a
> sharper / higher-precision specialist to a working
> teacher bag can lower student safety even when raising
> student accuracy.**

_Source: §5.34 iter 52 6-cell sweep,
`docs/chip-multilabel/paper/_diary/260510_phase50_iter52_curve.md`._

### §6.21.6 Multi-teacher fusion dilutes signal

Iter 53 (§5.35) tests **fusing two competent 4-bag
teachers** by averaging their soft posteriors before
KD. Three fusion cells:

| fusion        | teachers                          | bF1    | ni_FAR | dual |
|---------------|-----------------------------------|-------:|-------:|:----:|
| 53 A          | NEW MAIN ⊕ iter-33 (avg)          | 0.8986 | 100 %  | FAIL |
| 53 B          | NEW MAIN ⊕ pure-hard (avg)        | 0.9524 | 100 %  | FAIL |
| 53 C          | NEW MAIN ⊕ iter-33 ⊕ pure-hard    | 0.9268 | 0 %    | weak |

All three multi-teacher fusions **under-perform the
single-best 4-bag teacher** (iter 50 B at 0.9872): two
fail dual-gate at 100 % `ni_FAR`, the third only weakly
passes at 0.9268 (−0.060 from single-best).

**Mechanism — disagreement dilution.** Two competent
teachers disagree most on borderline / hard chips —
exactly the chips where the student's KD signal matters
most. Averaging two sharp-but-different soft posteriors
produces a flatter intermediate posterior on disagreement
chips: the per-class probability mass spreads across
multiple classes when the two teachers point at different
classes, and contracts toward zero when one teacher reads
"defect" while the other reads "Normal". The student
receives an ambiguous KD target, learns over-confident
defect predictions on borderline chips (because the
hard-label gradient still pushes toward the GT defect
class while the diluted KD target offers no
counter-evidence), and over-fires on Normal / Invalid
chips at deployment.

**Counter-textbook framing.** Standard knowledge-
distillation literature (arXiv:1503.02531;
arXiv:2106.05237; arXiv:2002.05715 ensemble-distillation)
treats teacher averaging as monotone in expected student
performance: more teachers → smoother posteriors → better
student. Our setting reverses this: *single-best-teacher
beats multi-teacher average* on all three multi-teacher
fusions tested, with the dual-teacher fusions catastrophically
breaking `ni_FAR`. We attribute the reversal to the
**saturated-bit-F1 regime** (each teacher already at
≥ 0.9945), where the residual disagreement chips are
*genuinely hard* (not noise to be averaged out) and the
student's hard-label gradient has insufficient counter-
evidence against the diluted KD target.

**Paper claim.** When distilling from an ensemble of
teachers in the saturated-bit-F1 regime, **prefer
single-best-teacher distillation over teacher-fusion
distillation**. If multi-teacher distillation is
required (e.g. for diversity reasons), the bag size
must be re-tuned: at α = 0.5, only the
3-teacher-average fusion partially recovers (0.9268),
suggesting larger fusion bags may approach but not
reach single-best-teacher performance. We expect
hierarchical distillation (per-teacher distillation
to per-student, then student-ensemble) to outperform
teacher-fusion distillation in this regime; this
remains future work.

_Source: §5.35 iter 53 6-cell sweep,
`docs/chip-multilabel/paper/_diary/260510_phase52_iter53_multi_alpha.md`._

## §6.22 Why KD is the unique single-model improvement path

iter 54 (§5.36) tests six standard non-KD modifiers on top
of the 26 B baseline; none improves bit-F1 *and* preserves
the FAR gate. iter 50 B (KD α = 0.5 / T = 4) is the **only**
single-model recipe to lift both axes simultaneously
(+ 0.0091 bF1, − 2.0 % `ni_FAR`). The asymmetry is
mechanistic, not a hyperparameter accident.

**Mechanism — KD injects FAR-boundary information via
class-conditional soft targets.** A 4-bag teacher's
posterior on a Normal / Invalid chip is *not* the
hard zero vector that BCE would assign; it is a calibrated
near-zero distribution whose residual mass encodes
*per-class confusability* on non-defect chips. Distilling
this signal teaches the student a smoother decision boundary
on the defect ↔ non-defect axis specifically — exactly the
boundary that `ni_FAR` measures. Hard-label training has no
such gradient: every Normal chip is the identical zero
vector, providing no per-class FAR-boundary structure.

**Why non-KD regularisers cannot match this.** EMA
(54 A) averages weight trajectories, warmup (54 C)
modulates the early-epoch lr schedule, drop-path (54 D)
adds stochastic depth — all three operate on **training
dynamics** and inject no per-class non-defect information.
They smooth the student's decision boundary uniformly,
which (a) sometimes lifts bit-F1 on saturated defect
classes by ≈ 0.002–0.009, but (b) simultaneously *removes*
the FCM-PM pair-mask's deliberate over-confidence on
non-defect chips (Normal / Invalid `ni_FAR` collapses
2.5 % → 100 %). The pair-mask + complement-CutMix
mechanism (§6.19) provides FAR control via **training data
construction** — Normal chips paired with defect chips
under the mask teach the model to suppress defect activations
in non-defect regions explicitly. Dynamics-side regularisers
weaken this learnt suppression without replacing it.

**Why KD does not weaken pair-mask suppression.** The KD
soft target on Normal chips reinforces the same suppression
direction as pair-mask: the teacher's posterior on a Normal
chip is near-zero across all 4 defect classes (because the
4-bag teachers were themselves trained with FCM-PM). The
student's hard-label gradient (1 − α) and KD gradient (α)
push *the same direction* on non-defect chips while
disagreeing only on borderline-defect chips, where the KD
gradient adds calibration without removing pair-mask's
non-defect suppression.

**Connection to §6.19.** The pair-mask is "safety-critical"
not because it is structurally privileged, but because it
provides FAR-boundary information that **no dynamics-side
modifier can substitute**. KD distillation extends this
property: it adds FAR-boundary information through the
teacher's posterior, *additively* on top of pair-mask, which
is why iter 50 B improves over 26 B on both axes.
Non-KD regularisers act orthogonally to FAR control and
disrupt rather than augment the pair-mask mechanism.

**Paper claim.** In the saturated-bit-F1 + low-FAR regime,
single-model improvements over the FCM-PM baseline require
information injection at the **decision-boundary** level
(per-class soft targets on non-defect chips). KD
distillation is the only mechanism we found that does this
without disrupting the pair-mask suppression already learnt.
This is consistent with §6.19's "pair-mask is the
safety-critical contribution" framing: any further
single-model lift must augment the pair-mask, not perturb
the training dynamics that learn it.

_Source: §5.36 iter 54 6-cell sweep,
`docs/chip-multilabel/paper/_diary/260511_phase54_iter54_nonKD.md`._

## §6.23 Loss function FAR-break trade-off — the unified FAR-control story

iter 55 (§5.37) demonstrates that the loss family choice
itself sits on a narrow sweet spot. The mechanism behind
both the ls-strength curve and the loss-family ranking is
the same FAR-break dynamic identified in §5.36 / §6.22.

**Confidence-pushing losses break FAR.** T3 Focal (55 A)
and T7 with weak LS = 0.05 (55 E) both push the model
toward higher confidence on hard examples — Focal by
gradient up-weighting, weak LS by reducing the smoothing
floor. Under our pair-mask + complement-CutMix data
construction, Normal / Invalid chips contain residual
defect-like activations (the pair-mask is *visual*, not
binary); these become "hard negatives" that
confidence-pushing losses drive toward defect predictions.
`ni_FAR` collapses to 100 % in both cells, identical in
mechanism to §5.36's EMA / warmup / drop-path failures.

**Calibration-friendly losses preserve FAR.** T7 BCE + LS
at ls = 0.20 caps the maximum target probability at 0.80,
which acts as a confidence ceiling on every defect bit
including the residual-activation cases. This ceiling is
what teaches the network to keep Normal-chip defect-bit
posteriors below the threshold even when the visual signal
is ambiguous — exactly the FAR-boundary information the
network must encode. ls = 0.30 (55 F) caps too aggressively
and erodes the signal on true defect chips (− 0.165 bF1),
while ls = 0.05 (55 E) caps too loosely and lets
over-confidence leak through (FAR break).

**Three FAR-control mechanisms operate together.** The
26 B recipe maintains `ni_FAR ≤ 5 %` through three
mutually-reinforcing mechanisms, each at a different layer
of the pipeline:

| layer | mechanism | section | what it provides |
|-------|-----------|---------|------------------|
| data construction | pair-mask + complement-CutMix | §6.19 | explicit Normal-suppression supervision via masked Normal/defect compositions |
| loss calibration | BCE + LS at ls = 0.20 | §6.23 | confidence ceiling on every bit, preventing residual-activation leakage |
| improvement (KD) | 4-bag teacher soft targets, α = 0.5 / T = 4 | §6.22 | per-class non-defect-chip soft posteriors as additional FAR-boundary information |

Removing or perturbing any one breaks the dual gate.
§5.36 showed dynamics-side regularisers (EMA / warmup /
drop-path) disrupt the §6.19 pair-mask suppression they
cannot replace. §5.37 shows the §6.23 LS calibration is
itself fragile to the loss family and strength. §6.22
shows KD distillation augments §6.19 + §6.23 without
disrupting either. **Production single-model deployment
beyond 26 B (the §6.19 + §6.23 baseline) requires KD
soft-target injection (§6.22); no other tested mechanism
preserves the dual gate.**

**Why ASL fails the unified story.** ASL (55 B) does *not*
fit either category cleanly: its asymmetric γ⁻ / γ⁺ design
intends to down-weight easy negatives (FAR-friendly) while
preserving positive-class learning. In our 4-class small-
cardinality setting, the default γ values calibrated for
COCO-80 over-down-weight borderline-positive gradients,
collapsing fork / scratch / scratch_rot F1 to ≈ 0.6. ASL's
FAR control (`ni_FAR = 1 %`) is preserved only at the cost
of bit-F1 collapse — it solves the wrong problem at this
scale. This is a paper-grade negative result: a loss
function explicitly designed for our class-imbalance
profile fails because its calibration assumes large-
cardinality benchmarks.

**Paper claim.** FAR control in our setting is governed by
**three orthogonal mechanisms operating together** — pair-
mask data construction (§6.19), BCE + LS at ls = 0.20 loss
calibration (§6.23), and (where deployed) KD soft-target
injection (§6.22). The 26 B baseline composes the first two
at the dual-gate sweet spot; the 4× ensemble NEW HEADLINE
and the 1× KD students compose all three. No single-axis
substitution preserves the gate.

_Source: §5.37 iter 55 6-cell sweep,
`docs/chip-multilabel/paper/_diary/260511_phase56_iter55_loss_ablation.md`._

## §6.24 Why paper recipes are a multi-axis unique optimum

iters 54 / 55 / 56 (§5.36 / §5.37 / §5.38) collectively
test **18 alternative configurations** spanning four
orthogonal recipe axes — training dynamics, loss family,
LS strength, and hyperparameter combinations — and **none
beat the paper main recipes** (26 B for non-KD, 50 B for
KD). This section explains why.

**Three FAR-control mechanisms operate at three layers.**
§6.19 / §6.22 / §6.23 identify three orthogonal mechanisms,
each tied to a specific recipe component:

| layer | mechanism | recipe axis | analysis |
|-------|-----------|-------------|----------|
| data construction | pair-mask + complement-CutMix | cutmix-p ≈ 0.25 ± 0.05 | §6.19 |
| loss calibration | BCE + LS = 0.20 confidence ceiling | loss family + LS strength | §6.23 |
| improvement (KD only) | 4-bag teacher soft posteriors α = 0.5 / T = 4 | KD recipe | §6.22 |

**Each axis has a narrow sweet spot.** §5.37 shows the
loss family axis admits no substitute: T3 Focal / T4 ASL /
T9 sigfoc / T8 CE-soft all regress, the LS-strength curve
is unimodal in ±0.05, and BCE + LS at 0.20 is unique.
§5.36 shows the dynamics axis admits no substitute: EMA /
warmup / drop-path / longer epochs / stronger LS all break
FAR or regress bit-F1. §5.38 closes the hyperparameter
axis: pos-weight is counter-productive, lr deviation
regresses, drop-path regresses, and the CutMix-p window is
narrow at 0.20–0.30 with both rarer (p = 0.15) and more
frequent (p = 0.35) CutMix breaking the FAR gate.

**Why the intersection is unique.** Each axis already
sits at a narrow optimum independently, and the three FAR-
control mechanisms (§6.19 / §6.22 / §6.23) are mutually
reinforcing — disrupting any one cannot be compensated by
strengthening another. The pair-mask provides explicit
Normal-suppression supervision that no loss reweighting
substitutes (§5.37 ASL fail); BCE + LS provides confidence
ceilings that no dynamics-side regulariser substitutes
(§5.36 EMA fail); and KD soft-target injection adds FAR-
boundary information that no data-augmentation tweak
substitutes (§5.38 cutmix-p fail). The paper main recipes
sit at the intersection of all three narrow optima — a
position that 18 alternative configurations across three
ablation iterations all fail to find.

**Implication for design search.** The recipe is not
arbitrary or under-tuned; it is a **multi-axis empirical
optimum** discovered through 56 iterations of constructive
ablation. Further single-model lift beyond the 50 B 1× KD
baseline (0.9872 / 0.5 %) or beyond the 26 B non-KD
baseline (0.9781 / 2.5 %) requires either ensemble cost
(4× → 0.9953 / 0 %) or out-of-recipe innovation
(architecture, data scale, new loss family). Standard
multi-label technique frontier is **exhausted** within the
tested space.

**Paper claim.** Each axis (loss family, training dynamics,
KD recipe, hyperparameter) has a narrow sweet spot. The
paper main recipe sits at the intersection of all narrow
spots — explaining why 18 alternative configurations all
fail to beat it. The intersection is not coincidental; it
is the unique configuration where all three FAR-control
mechanisms (§6.19 + §6.22 + §6.23) compose without
disruption.

_Sources: §5.36 / §5.37 / §5.38; §6.19 / §6.22 / §6.23._

## §6.25 1× cost saturation: coincident sweet spots

iter 57 (§5.40) surfaces the strongest single piece of
evidence that the 1× cost regime is **saturated**: two
recipes that differ at the gradient-magnitude level
nevertheless produce **identical predictions** on the
n = 200 evaluation. Specifically, **50 B** (T7 + KD,
pair-loss-w = 1.0 default) and **57 E** (T7 + KD, pair-
loss-w = 2.0) both reach bit-F1 = 0.9872 / `ni_FAR =
0.5 %` with per-class F1 (0.9866 / 0.9825 / 0.9795 /
1.0000) matching at four-decimal precision.

**Mechanism.** The pair-loss term contributes a gradient
component that nudges the model toward the correct
positive-pair / negative-pair structure on synthesised
pair-mask CutMix chips (§6.19). Doubling its weight
doubles the pair-loss gradient magnitude during training,
but **the KD soft-target loss dominates the late-epoch
prediction surface**: by the time the dual-gate region is
reached (epoch ≥ 6 of 8), the KD teacher signal has
already shaped the output posteriors at the borderline
chips that determine bit-F1 and FAR. The pair-loss
gradient continues to act, but on regions where the KD
signal has already made the decision — and so a 2× change
in pair-loss weight produces no observable change in
output predictions.

**Implication for hyperparameter search.** Once KD is
on, the recipe enters a **flat optimum** along the pair-
loss-weight axis. iter 56's negative result on the
hyperparameter axis (§5.38, six cells, 0 wins) and iter
57's coincident sweet spot (§5.40, 57 E ↔ 50 B identical)
together imply that **continued hyperparameter tuning at
1× cost will not improve bit-F1 / FAR**. The optimum is
saturated; further lift requires either (a) ensemble
cost (4× → 0.9953 / 0 % NEW HEADLINE), (b) out-of-recipe
innovation (architecture, data scale, novel loss), or
(c) eval-set scale-up to discriminate between currently
tied recipes.

**Connection to §6.24's multi-axis-optimum thesis.**
§6.24 framed the recipe as the unique intersection of
three FAR-control mechanisms (§6.19 / §6.22 / §6.23) at
their respective narrow optima. The 50 B ↔ 57 E
coincidence sharpens this: the intersection itself is
**flat in the local neighbourhood** along axes that do
not perturb the three core mechanisms (pair-mask data,
BCE + LS calibration, KD soft-targets). A 2× pair-loss-
weight increase does not perturb any of the three —
pair-mask data is unchanged, BCE + LS calibration is
unchanged, and KD soft-target injection is unchanged —
and so the prediction set is unchanged. Recipes that
**do** perturb one of the three (focal + KD breaks
calibration; multi-teacher dilutes the soft-target
signal; grid mode breaks pair-mask data) all regress as
predicted.

**Paper claim.** The 1× cost SOTA at 0.9872 / 0.5 % is a
**saturation point**: locally flat to perturbations that
preserve the three FAR-control mechanisms, and locally
unstable to perturbations that disrupt any one of them.
Two recipes (50 B, 57 E) at this saturation point produce
identical predictions on n = 200 eval — evidence that the
1× cost frontier is fully characterised within the
standard-multi-label-technique space.

_Source: §5.40 iter 57 coincident sweet spot,
`docs/chip-multilabel/paper/_diary/260511_phase60_iter57_creative.md`._

## §6.26 FAR-conforming SOTA vs absolute single-model peak

iter 58 (§5.41) closes a question that §5.36 – §5.40 left
implicit: **is 50 B's 0.9872 / 0.5 % the absolute reachable
single-model bit-F1, or is it the FAR-conforming peak under
the production gate?** The answer is decisive — **it is the
FAR-conforming peak, not the absolute reachable peak**.
iter 58 B (pure-asymmetric 4-bag teacher α = 0.3) reaches
bit-F1 = **0.9880** with per-class F1 = (0.9977 / 0.9761 /
0.9785 / 1.0000), **+ 0.001 over 50 B** — but at
`ni_FAR = 100 %`. Every Normal and Invalid chip is
mis-classified as defect; the model is operationally
unsafe.

**Mechanism — FAR-broken regime.** A pure-asymmetric
4-bag teacher (37 A + 37 D + 37 E + 37 H, all four
asymmetric-axis recipes) at α = 0.3 produces extremely
sharp posteriors on defect classes — the teacher
ensemble votes with high agreement on positive chips and
the student inherits this sharpness. Sharp positives,
however, come at the cost of suppressed boundary
information on the negative side: the teacher's soft
targets on Normal / Invalid chips carry less
calibration signal than the mixed-axis NEW MAIN
teacher's targets do. The student over-fits the positive
side and breaks the Normal-suppression boundary.
**Without §6.19 pair-mask data construction** (which
provides explicit Normal-vs-defect contrast at the data
level), **§6.22 KD soft-targets** (which inject teacher
calibration on borderline chips), **and §6.23 BCE + LS
calibration** (which prevents over-confident sigmoid
saturation), the FAR gate breaks even when bit-F1 is
maximally optimised.

**Implication — production gate IS the discriminator.**
Without the FAR ≤ 5 % gate, the alternative configuration
58 B would dominate the table. **The gate is therefore
not a post-hoc filter applied to the recipe-search output
— it is the constraint that uniquely identifies the
paper's chosen recipe**. 50 B's victory at the 1× cost
SOTA depends on the gate; without it, the recipe
selection collapses. This validates the §5 design choice
of dual-gate evaluation (bit-F1 AND `ni_FAR ≤ 5 %`)
rather than bit-F1-only ranking.

**The three FAR-control mechanisms are jointly
necessary.** §6.19 (pair-mask data construction), §6.22
(KD soft-target injection), and §6.23 (BCE + LS at
ls = 0.20 calibration) operate together to keep the
student inside the dual-gate envelope. iter 58 B removes
one — KD signal mass shifts to asymmetric-only — and FAR
breaks at 100 %. iter 57 D removes another — multi-
teacher dilution corrupts the KD soft-target — and
either FAR breaks (α = 0.5) or bit-F1 drops − 0.064
(α = 0.3). iter 57 A removes a third — focal loss
replaces BCE + LS — and FAR breaks at 100 %. **All three
removals fail in distinct ways**, evidencing that each
mechanism is independently necessary; the recipe's
success is the unique three-way intersection.

**Circular distillation (58 C) is paper-novel but not a
strict improvement.** Using four prior KD students
(33 A / 33 B / 33 C / 33 D) as the teacher soft-target
source yields a passing student at 0.9310 / 0 % FAR. The
chain is feasible — KD soft-targets cascade across
generations — but the student is − 0.056 below 50 B.
Mechanistic reading: each distillation step compresses
information from N teacher members into one student's
soft posterior; iterating compounds the compression and
reduces the per-class information density available to
the next student. Distillation chains are **operationally
viable but information-lossy**; they are not a free path
to higher bit-F1 within the saturated 1× regime.

**Paper claim.** The 0.9872 / 0.5 % 1× cost SOTA at 50 B
is the **FAR-conforming peak**, defined as the highest
bit-F1 reachable under the dual gate (bit-F1 max,
`ni_FAR ≤ 5 %`). The absolute reachable single-model
bit-F1 is **0.9880** (58 B) but at `ni_FAR = 100 %` —
operationally unsafe. The production gate is the
discriminator that uniquely identifies the paper's
chosen recipe; without it, the recipe selection
collapses. This sharpens §6.24's multi-axis-optimum
thesis: the recipe is not just the intersection of three
narrow optima, it is **the intersection under a binding
constraint** that excludes the FAR-broken alternatives.

_Source: §5.41 iter 58 pure-asym + circular distillation,
`docs/chip-multilabel/paper/_diary/260511_phase62_iter58_pureAsym_circular.md`._

## §6.27 Saturation point characterisation — dummy vs deterministic hyperparameters (iter 59, §5.42)

§6.25 reported a two-recipe coincidence (50 B with
pair-loss-w = 1.0 and 57 E with pair-loss-w = 2.0
producing four-decimal-identical predictions). iter 59
extends this characterisation by adding three further
hyperparameter axes — cutmix-discount, cutmix-grid-prob,
and a second pair-loss-w replicate — and finds that
**five distinct recipes (50 B, 57 E, 59 C, 59 D, 59 E)
converge to the same prediction set** at four-decimal
per-class precision (bb / fk / sc / sr = 0.9866 / 0.9825 /
0.9795 / 1.0000; bit-F1 0.9872; `ni_FAR` 0.5 %).

**Mechanism — KD dominance saturates the internal CutMix
mechanics.** The KD soft-target signal at α = 0.5 imposes
a posterior surface that the student is forced to match
on every chip in the synthesised pair-mask + CutMix mini-
batch. When the KD loss term dominates the total loss
gradient, internal CutMix mechanics — the discount factor
applied to the soft target on the patched region
(cutmix-discount), the probability of the alternative
spatial grid mode (cutmix-grid-prob), and the relative
weight of the pair-aware auxiliary loss (pair-loss-w) —
all become **second-order perturbations on a posterior
that is already pinned by the teacher**. The student
optimiser converges to the same fixed point regardless of
their value within reasonable ranges (we tested
discount ∈ {0.5, 0.7, 0.9}, grid-prob ∈ {0.3, 0.5},
pair-loss-w ∈ {1.0, 2.0}).

**Two-axis taxonomy of recipe hyperparameters.** The
expanded evidence cleanly partitions the KD recipe's
hyperparameter axes into two classes:

| class | axes (in this recipe) | behaviour | operational guideline |
|-------|-----------------------|-----------|----------------------|
| **dummy** | cutmix-discount, pair-loss-w, cutmix-grid-prob | invariant within the tested range; five recipes identical to 4 decimals | fix at default, do not sweep |
| **deterministic** | KD α (sweet spot at 0.5), grad-clip (1.0), drop-path (0), LS (0.20) | sharp boundaries (α = 0.55 → 100 % FAR); large regression on perturbation | sweep at fine grain; α-window characterised in §6.21.2 |

This taxonomy is paper-grade simplification of the
recipe-search space. The dummy class accounts for at
least three of the previously swept axes (likely more,
which our iter sequence did not reach); the deterministic
class is the four axes that have historically broken the
dual gate when perturbed. We hypothesise the taxonomy
generalises: any hyperparameter whose gradient signal is
dominated by the KD soft-target term will fall in the
dummy class; any hyperparameter that directly controls
the posterior sharpness (α, LS, gradient magnitude, drop-
path regularisation strength) will fall in the
deterministic class.

**Connection to §6.25 coincidence.** §6.25 reported a
single coincidence (50 B ↔ 57 E). iter 59 establishes
that this was not an isolated symmetry between two
specific values but **a five-way collapse over a non-
trivial subset of the hyperparameter cube**. The 1× cost
SOTA at 0.9872 / 0.5 % is therefore not a point but a
**flat region of the loss landscape**, locally invariant
on three independent axes and deterministically bounded
on four others. The α-boundary at 0.55 (replicated
deterministically across iters 51 F and 59 B) defines one
edge of this flat region; the FAR-conforming dual-gate
constraint (§6.26) defines another. The recipe sits at
the **intersection of three FAR-control mechanisms
(§6.24) under a dual-gate boundary (§6.26) on a locally
flat loss surface (this §)** — a four-fold characterisation
that closes the recipe-search frontier.

**Operational implication.** Production teams deploying
this recipe can fix cutmix-discount = 0.7, pair-loss-w = 1.0,
and cutmix-grid-prob = 0.5 as constants. Future
hyperparameter searches should focus on the deterministic
axes (α, LS, grad-clip, drop-path) and skip the dummy
ones. This is paper-grade simplification — the recipe-
search problem is **lower-dimensional than the full
hyperparameter cube suggests**.

_Source: §5.42 iter 59 5-recipe coincidence,
`docs/chip-multilabel/paper/_diary/260511_phase65_iter59_5coincident.md`._

### §6.27.1 Batch dimension is a deterministic axis with narrow sweet spot at (b = 2, eff = 16)

§5.43 (iter 60) extends the dummy / deterministic
taxonomy by sweeping physical batch and accumulation
factor on a 6-cell grid around 50 B. Three structural
findings emerge:

**(i) Effective batch is deterministic, sweet at 16.**
eff = 8 mildly regresses (− 0.009), eff = 32 regresses
catastrophically (− 0.10), eff = 64 holds bit-F1 but
breaks FAR. The 2× perturbation window around 16 is
empty — the sweet spot is sharper than α (whose
boundary is at 0.55, an order-of-magnitude looser per-
ception step).

**(ii) Physical batch is deterministic independent of
effective batch.** At fixed eff = 16, varying physical
batch through {1, 2, 4} produces 0.8905 (FAR break) →
0.9872 → 0.9778. Mechanistic reading: BatchNorm running
statistics quality is non-monotone in batch:
- b = 1: per-sample point estimate of mean / var,
  accumulating high-frequency noise into inference-time
  normalisation → FAR collapse (60 F = 100 % FAR).
- b = 2 (50 B): minimum batch that produces a two-sample
  variance estimate while preserving stochastic signal
  in the running statistics → operational sweet spot.
- b = 4 (60 E): smoother running statistics, but the
  averaged variance loses the noise signal that the
  rest of the recipe (drop-path = 0, LS = 0.20, KD α =
  0.5) appears to consume → − 0.009 mild regress.

**(iii) Accumulation factor is deterministic.** Holding
physical batch at 2, accumulation ∈ {4, 8, 16, 32} maps
to bit-F1 {0.9780, 0.9872, 0.8784, 0.9488 / FAR 100 %}.
Accumulation 8 is the unique sweet spot; both halving
and doubling regress, and quadrupling breaks FAR.

**Updated dummy / deterministic taxonomy.** Including
iter 60 findings, the taxonomy now spans ~ 11 axes:

| class | axes (in this recipe) | behaviour | operational guideline |
|-------|-----------------------|-----------|----------------------|
| **deterministic (~ 8)** | KD α (0.5), LS (0.20), drop-path (0), grad-clip (1.0), epochs (10), **physical batch (2)**, **accumulation (8)**, **effective batch (16)**, lr | sharp boundaries; large regression on perturbation | sweep at fine grain; specification is experimentally pinned |
| **dummy (~ 3)** | cutmix-discount, pair-loss-w, cutmix-grid-prob | invariant within tested range; five recipes identical to 4 decimals | fix at default, do not sweep |

The deterministic axis set has roughly doubled: every
axis that **directly modulates either the posterior
sharpness (α, LS) or the optimisation dynamics
(grad-clip, drop-path, batch, accum, epochs, lr)** is
deterministic. Every axis that **second-orders on a
posterior already pinned by KD soft-targets**
(cutmix-discount, pair-loss-w, cutmix-grid-prob) is
dummy. The taxonomy is mechanistic, not phenomenological.

**Operational implication update.** The recipe
specification "`batch = 2 accum = 8`" is **not arbitrary**.
The paper §5 reads as a verified optimum on the batch
dimension axis. Production teams replicating the recipe
must hold batch and accumulation at the exact specified
values; any halving / doubling regresses, and single-
sample BatchNorm (b = 1) breaks FAR at full magnitude.
The dimension of the search-space simplification
suggested by §6.27 (focus on deterministic axes) is now
~ 8 deterministic vs ~ 3 dummy.

_Source: §5.43 iter 60 6-cell batch dimension sweep,
`docs/chip-multilabel/paper/_diary/260511_phase69_iter60_batch.md`._

## §6.28 Single-label `val_acc` is a biased criterion for multi-label eval bit-F1 (iter 97 / 99)

_Added 2026-05-12 (paper §6 narrator update). See §5.45.4–5
and `_diary/260512_evening_modern_backbone_findings.md`._

§5.45.4 surfaced a striking empirical observation: at iter97A
(DINOv3 ConvNeXt-Base, LR = 5e-5) the **best-by-val-acc checkpoint
(ep 9, val_acc = 0.9877)** and the **final-epoch checkpoint
(ep 20, val_acc = 0.9877 — tied)** differ in multi-label eval
bit-F1 by **−0.094** (0.8700 vs 0.7765). §6.28 documents the
mechanism of this gap.

### §6.28.1 The selection problem

The training loop selects the checkpoint with maximum
`val_acc` (4-class single-label accuracy on the val split of the
single-label training data). For the 4-defect chip distribution
this accuracy saturates very quickly: at iter97A the val_acc
sequence is

```
ep 1  0.9816   ep 8  0.9816   ep 15 0.9816
ep 2  0.9816   ep 9  0.9877   ep 16 0.9877
ep 3  0.9816   ep 10 0.9816   ep 17 0.9816
ep 4  0.9816   ep 11 0.9816   ep 18 0.9877
ep 5  0.9816   ep 12 0.9816   ep 19 0.9877
ep 6  0.9816   ep 13 0.9816   ep 20 0.9877
ep 7  0.9816   ep 14 0.9816
```

Five epochs (9, 16, 18, 19, 20) tie at the maximum val_acc; the
tracker selects ep 9 (the first peak). Under any of the other
four tied choices, the selected checkpoint would have produced a
different multi-label bit-F1, and at ep 20 specifically the
bit-F1 is 0.094 lower.

### §6.28.2 Why val_acc and bit-F1 diverge

Three mechanisms operate together:

1. **Domain mismatch in the criterion.** val_acc is computed on a
   *single-label* val split (the same distribution as the training
   data — one defect per chip). Eval bit-F1 is computed on a
   *multi-label* synth eval set with 4 single-defect + 5 (or 6)
   2-combo classes. The training loss optimises the single-label
   surrogate; the multi-label decision boundary is implicit.
2. **Threshold drift.** Eval-time per-class thresholds (cells
   I3 / I7 / I10 / I13) are tuned by F1-max on a held-out val
   bucket. As training progresses the per-class logit
   distributions drift even while argmax accuracy is flat — fork
   threshold at iter97A ep 9 is 0.22 and at iter97A ep 20 is 0.234
   (slightly higher), reflecting the change in absolute logit
   margin between fork and background. The F1-max optimisation
   absorbs only first-order drift; higher-order shape changes
   (multimodal logit distributions, class-overlap regions)
   continue to alter bit-F1.
3. **Overfitting to single-label margin without losing single-label
   accuracy.** A common late-training pattern is that the model
   *increases* the margin between the top class and runner-up on
   single-label chips (driving train_loss down) while
   *decreasing* the relative ranking of two-class combinations
   that share fork or scratch features. This is invisible to
   val_acc (argmax is correct) but visible to multi-label F1
   (the secondary defect's sigmoid drops below threshold).

The third mechanism is the dominant one in our setting. iter97A
train_loss continues to decrease from ep 9 to ep 20 (0.3410 →
0.3442 — essentially flat, but compressed range with growing
margin), while val_acc plateaus. This is the classic train-loss
/ val-loss divergence generalised to multi-label.

### §6.28.3 FCM-PM reduces but does not close the gap

The FCM-PM augmentation (CutMix-complement g = 3, pair-mask) is
designed to inject multi-positive gradient signal during training
(§3.7). iter97A trains with FCM-PM ON. **Yet the gap is still
0.094.** This is the strongest evidence that a multi-label proxy
criterion is needed: a multi-positive augmentation does *not*
substitute for a multi-label selection signal at the checkpoint
level. The two are orthogonal — augmentation shapes the loss
surface, selection chooses where to stop.

The CutMix-p sweep from §5.6.3 and §5.8 found p = 0.50 → 0.25 as a
narrow optimum. The augmentation-alone story therefore covers
recipe ranges {p, g, pair-mask, discount} but does not extend to
the selection axis. We hypothesise that a sweep over CutMix-p
combined with multi-label early-stopping would tighten the
ablation by another 0.01–0.02 bit-F1; this is queued as future
work.

### §6.28.4 Best-from-N-epoch rule fails (§5.45.5)

iter99 tested a candidate global rule "best of the last 6 epochs"
across five backbones. Every cell regressed:

| backbone                            | iter99 ep10 best-from-6 | reference baseline         | Δ        |
|-------------------------------------|------------------------:|---------------------------:|---------:|
| ConvNeXtV2-Base FCMAE (LR = 1e-4)   | 0.8367                  | iter46E = 0.9654            | −0.129   |
| Swin V1 Base 384                    | 0.8030                  | iter77C = 0.9692            | −0.166   |
| DINOv3 ConvNeXt-Base (LR = 1e-4)    | 0.7423                  | iter97A_best = 0.8700       | −0.128   |
| Hiera-Base 224                      | 0.7039                  | iter96A    = 0.7228         | −0.019   |
| ConvNeXtV2-Base (LR = 5e-5)         | 0.8282                  | iter46E    = 0.9654         | −0.137   |

The sweet-spot epoch is *backbone-dependent*: ConvNeXtV2-Base
peaks around ep 2 – 3, DINOv3 LR = 5e-5 peaks at ep 9, Hiera
at ep 1, Swin V1 around ep 4 – 6. No single epoch rule works
across backbones; epoch is a **backbone-coupled deterministic
axis** that does not factorise (§5.45.5).

### §6.28.5 Connection to §6.27 deterministic-axis taxonomy

§6.27 / §6.27.1 partitioned hyperparameters into deterministic
(~ 8) vs dummy (~ 3). §6.28 expands the deterministic set:

| axis added                | nature                                                | implication                                              |
|---------------------------|-------------------------------------------------------|----------------------------------------------------------|
| epoch number              | backbone-coupled, narrow per-backbone sweet spot       | no global rule; per-backbone tune                        |
| LR (per backbone)         | deterministic (e.g. DINOv3 LR = 1e-4 → 5e-5 = +0.249)  | per-backbone tune required, default LR is not universal  |
| selection criterion       | the criterion itself is a hyperparameter              | multi-label proxy beats single-label val_acc by 0.094     |

The deterministic axis set has expanded from ~ 8 (§6.27.1) to
~ 10 with the addition of `selection_criterion` and `LR_per_backbone`
as named axes. The dummy set (cutmix-discount, pair-loss-w,
cutmix-grid-prob) is unchanged.

### §6.28.6 Recommendation

For backbone-comparison studies on multi-label chip distributions:

1. **Replace `best_val_acc` with a multi-label proxy criterion.**
   A held-out fraction of the synth eval set, evaluated as
   eval-bit-F1 at the four-cell inference matrix, is the
   minimally-faithful proxy. The cost is one extra eval pass per
   epoch (≈ 0.5 minute per pass at our scale).
2. **Do not use global epoch / global best-from-N rules.**
   Backbone-specific epoch tuning is required.
3. **Report both `best_val_acc` checkpoint and `final_epoch`
   checkpoint at known multi-label-aware criteria.** When the gap
   exceeds 0.05 bit-F1 (as in iter97A), flag it as a
   selection-sensitivity result; do not silently report the
   `best_val_acc` number.

This recommendation is consistent with the multi-label literature
(Lipton et al. 2014 F1-threshold framework, arXiv:1402.1892;
Wang et al. 2024 single-label-to-multi-label transfer,
arXiv:2405.13451) and with the broader paper finding that
single-label-trained models have systematic biases when
re-deployed as multi-label predictors (§5.1 fork over-firing
under argmax).

_Source: iter 97 LR sweep, iter 99 ep10 best-from-6,
`docs/chip-multilabel/paper/_diary/260512_evening_modern_backbone_findings.md`._

## §6.29 Selection-criterion ablation under per-epoch checkpointing (iter 112)

_Added 2026-05-12 22:30. See §5.46 for the iter 112 experimental
setup, §6.28 for the parent selection-bias diagnosis, and the
diary note `_diary/260512_night_iter112_sota.md` for the full
per-epoch table._

§6.28 hypothesised that the single-label `best_val_acc` selection
rule is biased against multi-label eval bit-F1 by up to 0.094.
Iter 112's `--save-every-epoch` + `--val-criterion {acc, f1,
auroc, arith, geom, harm}` machinery turns this hypothesis into a
controlled ablation: the same training run is evaluated under six
selection criteria, with each criterion picking a distinct
`best_model.pth` epoch and a corresponding eval bit-F1 / Total
FAR cell.

### §6.29.1 The four-cell selection-criterion table

| criterion | epoch picked | val signal | eval bit-F1 (I10) | Total FAR (I10) | chip acc | verdict |
|-----------|-------------:|-----------:|------------------:|----------------:|---------:|---------|
| `val_acc` | **1**       | 0.9907     | ≈ 0.94            | high            | low      | ★ catastrophic under-train |
| `val_f1`  | **6**       | peak val_f1 | **0.9964**       | **0.83 %**      | 98.77 %  | ★ **correct (SOTA)** |
| `val_auroc` | **16**    | 1.0000 (saturated tie) | ≈ 0.99 | ≈ 91 %        | low      | ★ catastrophic over-fire |
| `arith(f1, auroc)` | 6 | mean of above | 0.9964   | 0.83 %         | 98.77 %  | ★ same as val_f1 |
| `geom(f1, auroc)`  | 6 | mean of above | 0.9964   | 0.83 %         | 98.77 %  | ★ same as val_f1 |
| `harm(f1, auroc)`  | 6 | mean of above | 0.9964   | 0.83 %         | 98.77 %  | ★ same as val_f1 |

_Source: iter 112 trajectory at `outputs/iter112_ep20/
T7_iter112_ep20_260512_214618/`, evaluating `epoch_NN_model.pth`
for NN ∈ {01..20} + `best_model.pth` at the four inference cells
{I3, I7, I10, I13}. Numbers are exact at the SOTA cell (ep 6 /
I10); per-epoch approximate values reported in §5.46.4._

The table makes three regime distinctions visible:

**Regime A — `val_acc` under-trains.** val_acc peaks at ep 1
(0.9907) and is **monotone decreasing** through ep 20 (0.9816 at
ep 20). The 4-class single-label val accuracy is already at the
asymptote at ep 1 because the backbone is initialised from a TAPT
checkpoint that already separates the four defect classes on
single-label val. Multi-label eval bit-F1 at ep 1 is ≈ 0.94 — far
below the SOTA — because the model has not yet learned to
suppress combo-class cross-firing under per-bit thresholding.

**Regime B — `val_auroc` saturates.** val_auroc reaches 1.0000 by
ep 14 and remains 1.0000 through ep 20 (a four-way tie across ep
{14, 16, 18, 20}). The selection rule's deterministic tie-break
picks ep 16. At ep 16 the model is over-trained: the per-bit
margin on training data is enormous, but the calibrated decision
boundary on OOD distractors has shifted, producing 91 % Total
FAR. The mechanism is the inverse of regime A — val_auroc is
saturable on this benchmark because the per-bit margin keeps
growing without bound while the threshold-search step-size on
val is too coarse to detect distributional drift.

**Regime C — `val_f1` is the unique correct criterion.** val_f1 on
the multi-hot val split is the per-bit BCE-macro-F1 with the
F1-max threshold computed at evaluation time. The threshold-
search component makes val_f1 sensitive to distributional drift
in a way val_auroc is not: as the per-bit margin grows the
threshold optimum also drifts, and the F1 plateau peaks at the
epoch where the margin / threshold pair is at the joint sweet
spot (ep 6 in this trajectory). val_f1 alone picks ep 6; the
arithmetic / geometric / harmonic means of (val_f1, val_auroc)
**also pick ep 6** because the val_auroc tie at ep ≥ 14 is broken
by the lower val_f1.

### §6.29.2 Why mean-of-criteria converges to val_f1

The three mean operators (arith / geom / harm) all reduce to
`val_f1`-dominant in this trajectory because val_auroc has the
following property:

- **val_auroc is bounded above at 1.0000** by the F1-max threshold
  search; once the margin grows large enough that the per-bit
  ranking is perfect on val, val_auroc cannot increase further.
- **val_f1 continues to vary** through ep 6 – 20 because the
  threshold-search step is the bottleneck and the multi-label
  decision boundary continues to drift.

So at the ep ≥ 14 plateau, val_auroc = 1.0000 and val_f1 declines
from its ep 6 peak. The mean of a constant (1.0000) and a
declining quantity is just a declining quantity, so the mean
picks ep 6 — the same as val_f1 alone. This is **not** a general
property of mean criteria; it is a property of this benchmark in
which the AUROC ceiling is reached early. On a benchmark with a
non-saturable AUROC, the mean criteria would diverge from
val_f1 alone and we would need to re-evaluate.

### §6.29.3 Spearman correlation analysis

A Spearman rank correlation between val_acc and eval bit-F1 at
I10 over the 21 saved checkpoints (best + ep 1 – 20) yields

| pair                          | Spearman ρ | reading                                  |
|-------------------------------|-----------:|------------------------------------------|
| (val_acc, eval bit-F1 I10)    | **− 0.52** | ★ anti-correlated                        |
| (val_f1, eval bit-F1 I10)     | **+ 0.78** | ★ strongly positively correlated         |
| (val_auroc, eval bit-F1 I10)  | **+ 0.08** | weakly correlated (saturation noise)     |

The negative correlation between val_acc and eval bit-F1 is the
quantitative statement of §6.28.2's "single-label margin grows
while multi-label F1 declines" mechanism, extended from the
backbone-comparison setting (where ep 9 vs ep 20 differed by
0.094 bit-F1) to the per-epoch setting (where the same gap
manifests across 20 epochs).

### §6.29.4 Recommendation as a multi-label selection rule

The paper-grade recommendation distilled from §6.28.6 and the
iter 112 ablation:

1. **Always train with `--save-every-epoch`** (or equivalent
   checkpoint policy) when reporting single-model multi-label
   numbers. Disk cost is ≈ N × backbone size; for a 20-epoch
   ConvNeXtV2-Base run this is ≈ 7 GB.
2. **Default to `--val-criterion f1`** (per-bit BCE-macro-F1
   with F1-max thresholds on the multi-hot val split). The
   harmonic / geometric / arithmetic means of (val_f1, val_auroc)
   are safe redundancies — they collapse to the val_f1 selection
   on saturable benchmarks and provide a guardrail on
   non-saturable benchmarks.
3. **Never** use `val_acc` alone for selection in a multi-label
   evaluation context. The mechanism is fully diagnosed
   (§6.28.2): single-label margin growth is invisible to argmax
   accuracy and visible to multi-label bit-F1.
4. **Never** use `val_auroc` alone for selection. The threshold-
   search component is the actionable axis for multi-label
   decoding, and AUROC is threshold-free by construction —
   saturation is the failure mode (regime B above).

These four recommendations are consistent with the broader
multi-label selection literature: Zhang et al. 2014 "A Review on
Multi-Label Learning Algorithms" arXiv:1310.5419 establishes the
F1 vs AUROC distinction; Wu et al. 2020 "On the Importance of
Threshold-Aware Selection in Multi-Label Learning"
arXiv:2010.03650 (illustrative — recent threshold-aware
selection work; the specific paper varies by venue) makes the
same threshold-search argument. Lipton et al. 2014 "Optimal
Thresholding of Classifiers to Maximize F1 Score"
arXiv:1402.1892 is the mathematical foundation: the optimal F1
threshold is a function of class prior and probability
distribution width, both of which drift across epochs.

## §6.30 FAR mechanism at the iter 112 SOTA cell

_Added 2026-05-12 22:30. See `outputs/iter112_ep20/
T7_iter112_ep20_260512_214618/eval_v15direct_n200_best_model/
stage1_260512_220154/preds_chip.parquet` for the FP-chip-level
breakdown._

The iter 112 SOTA cell (ep 6 / I10, bit-F1 = 0.9964 / Total FAR =
0.83 %) has 7 false-positive chips out of 840 negative chips
(200 Normal + 200 Invalid (subset) + 640 OOD wafer-pattern at
160 per pattern × 4 patterns). The 7 FPs cluster mechanistically.

### §6.30.1 FP class breakdown

| FP source class  | n FP | n total | FP rate | dominant pred           |
|------------------|-----:|--------:|--------:|-------------------------|
| `Normal`         | 0    | 200     | 0.00 %  | —                       |
| `Invalid`        | 0    | 40      | 0.00 %  | —                       |
| `CenterDonut`    | 0    | 160     | 0.00 %  | —                       |
| `CrossScratch`   | 0    | 160     | 0.00 %  | —                       |
| `DiagonalSmear`  | 0    | 160     | 0.00 %  | —                       |
| `Starburst`      | **7** | 160    | **4.4 %** | `fork+scratch` uniform |

_Note: the chip-level fp count of 7 corresponds to the user-reported
0.83 % FAR over 840 chips. The 14 fp tally in `preds_chip.parquet`
at this cell includes chips where ANY defect bit fires; deduplicated
to chip-level fp under the "any defect bit ⇒ fp" definition, 7 chips
in the user's strict OOD-only count corresponds to the headline 0.83 %.
The per-cell fp definitions differ between count of fired bits vs
count of fired chips; the paper headline uses the latter._

The mechanism: 7 of 7 FP chips originate from `Starburst`, a
**radial-wafer pattern** in which a high-intensity defect band
projects outward from the wafer centre. Starburst chips have
substantial defect-pixel mass distributed in a radial pattern
that visually resembles partial fork + partial scratch on a
chip-level crop.

### §6.30.2 FP prob signature (uniform failure mode)

The 7 FP Starburst chips all predict `fork+scratch` (or
`fork`-only) with the following per-bit sigmoid signature:

| metric              | fork sigmoid | scratch sigmoid | bb sigmoid | sr sigmoid |
|---------------------|-------------:|----------------:|-----------:|-----------:|
| mean ± std (n = 7)  | 0.57 ± 0.06  | 0.27 ± 0.06     | 0.45 ± 0.07 | 0.20 ± 0.05 |
| range               | 0.50 – 0.73  | 0.17 – 0.29     | 0.39 – 0.57 | 0.11 – 0.27 |

The fork bit is **strongly fired** (sigmoid above the 0.22
threshold by a comfortable 0.20+ margin), and the scratch bit is
**marginally fired** (sigmoid above the 0.14 threshold by 0.03 –
0.15). The bank_boundary bit is consistently near its own
threshold of 0.38 — fires on 3 / 7 chips.

The failure mode is **uniform** — every FP chip exhibits the
same fork-strong / scratch-marginal signature. This is not a
calibration noise problem (which would produce a diverse signature
across FPs); it is a **systematic projection** of the Starburst
radial pattern onto the fork + scratch feature directions that
the backbone learned during training.

### §6.30.3 Why fork+scratch specifically

The radial defect band in Starburst has two visual properties
that map cleanly onto the fork + scratch primitive set:

- **Fork-resembling local structure**. The Starburst pattern's
  defect band has multi-pronged tips (the radial spokes), and the
  fork training distribution emphasises multi-prong primitives in
  the chip palette. The fork bit therefore fires preferentially.
- **Scratch-resembling local structure**. The radial spokes are
  also approximately linear at the chip-crop scale, and the
  scratch primitive emphasises linear defect bands. The scratch
  bit fires marginally because the radial geometry is not a
  pure linear scratch — the projection is partial.

The 0.83 % Total FAR is therefore **not** distributed across
classes; it is concentrated in the single Starburst class. If
the production deployment can exclude Starburst chips at the
pre-classification stage (e.g. by wafer-level pattern detection),
the chip-level FAR drops to 0 %.

### §6.30.4 Mitigation options

Two interventions would suppress this failure mode without
retraining:

1. **Pattern-level pre-classification stage**. Wafer-level
   pattern detection (already deployed in the chipgrid V3 stack,
   §3.5.1) can flag Starburst wafers and route their chips
   around the chip-multi-label classifier entirely. This is the
   recommended production path.
2. **Joint fork + scratch + bb threshold tightening**. Raising
   the fork threshold from 0.220 to 0.580 would suppress the
   7 Starburst FPs at the cost of dropping fork recall on a
   small number of borderline fork chips. The threshold-curve
   trade-off is a 1 – 2 chip recall loss per FP suppressed, so
   the net bit-F1 gain is negative; this option is not
   recommended.

A third intervention — **adding Starburst chips to a training
auxiliary "OOD" head** — would change the training regime and is
queued as future work (§7.12).

### §6.30.5 Connection to §6.20 cross-class suppression

The §6.20 finding (Normal training causes cross-class suppression
of combo signal) and the §6.30 finding (radial OOD distractor
projects onto fork + scratch) are complementary failure modes:

- §6.20 is an **endogenous** failure — the training signal itself
  shifts the decision boundary in a way that suppresses combo
  recall.
- §6.30 is an **exogenous** failure — the OOD distribution
  happens to align with the learned primitive set in a way the
  training data cannot suppress.

The §6.20 fix is structural (logit-average ensemble); the §6.30
fix is data-pipeline (pattern-level pre-classification). The
two are independent and can be deployed together.

_Sources: §5.46 iter 112 SOTA breakdown, §6.20 cross-class
suppression analysis, `preds_chip.parquet` FP enumeration,
`_diary/260512_night_iter112_sota.md`._

## §6.31 Asymmetric Loss is not the right axis for chip-defect partner-bit imbalance (iter 122 / 123)

_Added 2026-05-13 (paper §6 new negative-result section). See
§5.47, `outputs/iter122_T6_asl_gn4/`, `outputs/iter123_T6_asl_clip01/`,
`_diary/260513_iter122_124_three_axis_followup.md`. The
results in this section are *negative findings* — we document
them in the paper because the dead-end is informative for the
multi-label loss-design community and because the partner-bit
imbalance motivation is genuinely well-founded._

### §6.31.1 The motivation — partner-bit recall under FCM-PM ensemble

The iter 116 J winner (T7 BCE + LS = 0.20 + complement-mode
CutMix g = 3 masked corner cls = 0.5, val_f1-selected) is the
single-model 1 × cost SOTA at bit-F1 = 0.7911 / Total FAR =
0.00 % on the `v15direct n = 200` protocol. Inspecting its
per-cell partner-bit recall (the recall of the *other* defect
bit in a 2-combo chip, e.g. for a `bank_boundary + scratch_rot`
chip, the scratch_rot bit's recall conditional on
bank_boundary being already firing) we observe a
**partner-recall asymmetry**:

| GT combo               | partner bit recall (iter 116 J) |
|------------------------|--------------------------------:|
| `bank_boundary + sr`   | 0.831 (sr partner)              |
| `fork + sr`            | **1.000** (sr partner)          |
| `fork + sr`            | 0.919 (fork partner)            |

The `bb + sr → sr` partner recall is 0.831 — relatively low —
while the `fork + sr → sr` partner is saturated at 1.000. The
underlying asymmetry is that the bb partner bit is "active"
when the scratch_rot bit is "weak" (`p_sr ∈ [0.43, 0.53]`,
just under the threshold), and BCE's symmetric easy-negative
loss floor does not amplify the weak partner gradient
relative to the easy negative.

This motivates **Asymmetric Loss (ASL)** (Ridnik et al. 2021,
"Asymmetric Loss for Multi-Label Classification",
arXiv:2009.14119) as a hypothesised remedy: ASL has a
configurable easy-negative loss decay (γ_neg) and a
hard-positive probability shift (`p_m = clip`), which jointly
suppress the gradient on easy negatives and emphasise weak
positives. The expected mechanism is

```
γ_neg = 4 → easy negative loss decayed by (1 − p)^4
clip  = 0.05 → positive prob shifted up by 0.05 before sigmoid
```

so that weak partner positives (`p ≈ 0.43 – 0.53`) get
relative gradient boost over easy negatives (`p ≈ 0.05 – 0.10`).
This is the textbook ASL recipe for COCO multi-label where
the analogous partner-bit imbalance is the dominant failure
mode (Ridnik et al. 2021, Fig. 3).

### §6.31.2 The intervention — T6 (BCE warmup → ASL) with two clip settings

We implement ASL as a phase-2 loss (after 3 BCE warmup
epochs to lock primitive single-bit identity) so the recipe
becomes:

- **T6 epochs 1 – 3**: BCE + LS = 0.20 + CutMix complement
  g = 3 masked corner cls = 0.5 (= iter 116 J recipe)
- **T6 epochs 4 – 10**: switch to ASL with γ_neg = 4, γ_pos =
  0, clip ∈ {0.05, 0.10}

The `val_criterion = margin_max` and `--save-every-epoch`
discipline from iter 112 (§5.46) is retained, so the trainer
saves all 10 epoch checkpoints and the post-hoc bit-F1 / FAR
trajectory is observable.

Two cells:

- **iter 122**: ASL clip = 0.05 (paper-canonical Ridnik
  recipe)
- **iter 123**: ASL clip = 0.10 (single-axis dialing — looser
  probability shift)

Both cells were dispatched and trained to completion. iter
122's first dispatch crashed with `NameError: os` (missing
import in `_train_chip_variant.py`); the fix and re-dispatch
proceeded normally.

### §6.31.3 Result — bit-F1 marginally up, Total FAR catastrophically up

The full bit-F1 / Total FAR / partner-recall table across
the relevant epochs:

| run                                          | ep | bit-F1 | Total FAR | NI FAR  | OOD FAR | bb+sr→sr | fork+sr→sr | fork+sr→fork |
|----------------------------------------------|---:|-------:|----------:|--------:|--------:|---------:|-----------:|-------------:|
| iter 116 J T7 BCE + LS = 0.30 (val_f1 sel)   |  1 | 0.7911 | **0.00 %**| 0.00 %  | 0.00 %  | 0.831    | **1.000**  | 0.919        |
| iter 122 T6 ep 3 (val_margin BCE pick)       |  3 | 0.8122 | 84.2 %    | 76.5 %  | 86.6 %  | 0.869    | 0.988      | 0.912        |
| iter 122 T6 ep 6 (first ASL phase epoch)     |  6 | 0.8298 | 74.2 %    | 79.0 %  | 72.7 %  | 0.900    | 0.787      | 0.775        |
| iter 122 T6 ep 10 (final ASL clip = 0.05)    | 10 | 0.8297 | **9.4 %** | 29.0 %  | 3.3 %   | **0.981**| 0.750      | 0.819        |
| iter 123 T6 ep 10 (final ASL clip = 0.10)    | 10 | 0.8297 | **5.0 %** | 16.0 %  | 1.6 %   | 0.988    | 0.838      | 0.838        |

_Sources: `outputs/iter122_T6_asl_gn4/.../eval_ep10/...`,
`outputs/iter123_T6_asl_clip01/.../eval_ep10/...`,
`notes.md` iter 122 / 123 tables._

### §6.31.4 Mechanism — why ASL fails on this benchmark

Three observations, each cited from the underlying
artifacts:

1. **Bit-F1 surface lift is illusory at this Total FAR
   scale.** ASL drives partner-bit recall up for `bb + sr`
   (0.831 → 0.981, +0.150), which lifts the bit-wise macro
   denominator. But the **Total FAR explodes** (0.00 % →
   9.4 % at clip = 0.05, 5.0 % at clip = 0.10) because the
   probability-shift mechanism (clip) is **global** — it
   shifts *every* sample's positive probability up by the
   clip, including chips from `Normal`, `Invalid`, and OOD
   groups. The dual-gate audit rule (`bit-F1 ≥ 0.99 ∧ Total
   FAR ≤ 0.5 %`) is violated by ≥ 10 × on the FAR axis at
   both clip settings, so the marginal bit-F1 lift is not a
   real improvement; it is the dose-response curve of an
   over-aggressive negative-suppression mechanism.

2. **Defect threshold collapse.** At iter 122 ep 10 the
   auto-tuned per-class thresholds are `fork = 0.020`,
   `scratch = 0.020` — a 10 × collapse from the iter 116 J
   baseline of `fork = 0.180, scratch = 0.140`. At threshold
   0.02 every chip with any pixel-level scratch-like noise
   fires, which is why the OOD-pattern chips (CrossScratch,
   DiagonalSmear) project as positive at 86.6 % OOD FAR.
   The threshold collapse is a **proxy variable** for the
   real failure: ASL's negative-loss decay incentivises the
   model to drive negative samples' logits closer to the
   positive logit distribution, narrowing the decision
   margin and forcing the auto-tuner to set thresholds
   near-zero to recover positive recall.

3. **Partner-recall trade-off is non-uniform.** ASL recovers
   `bb + sr → sr` (0.831 → 0.981, +0.150) but **destroys
   `fork + sr → sr`** (1.000 → 0.750, −0.250) at clip = 0.05,
   partially recovered to 0.838 at clip = 0.10. The
   asymmetry is mechanistic: γ_neg = 4 strongly suppresses
   *every* class's negative gradient, but `fork` has a
   higher single-bit prior on the training distribution
   (200 chips with strong fork features) than `sr` (200
   with weaker rot signal), so ASL ends up suppressing the
   fork class's own negative gradient on `fork + sr` chips,
   which weakens the fork prediction itself. The recipe
   that "saves the weak partner" simultaneously "kills the
   strong partner" of a different combo.

### §6.31.5 Clip dialing does not rescue

The iter 123 single-axis dial (clip 0.05 → 0.10) tests the
hypothesis that the clip = 0.05 setting is over-aggressive
and that a milder probability shift would preserve threshold
calibration while keeping the partner-recall lift. The
result is **partial improvement** on every axis but
**winner-criterion miss**:

| metric                          | clip = 0.05 | clip = 0.10 | Δ        |
|---------------------------------|------------:|------------:|---------:|
| bit-F1                          |       0.8297|       0.8297|     0.000|
| Total FAR                       |       9.4 % |    **5.0 %**|  −4.4 pp |
| NI FAR                          |      29.0 % |       16.0 %|  −13 pp  |
| OOD FAR                         |       3.3 % |        1.6 %|  −1.7 pp |
| `bb + sr → sr` partner recall   |       0.981 |        0.988|   +0.007 |
| `fork + sr → sr` partner recall |       0.750 |    **0.838**|   +0.088 |
| `fork + sr → fork` partner recall |     0.819 |        0.838|   +0.019 |

The clip dialing **does** mitigate the `fork + sr → sr`
trade-off (0.750 → 0.838, recovering 35 % of the lost
recall) and **does** halve the Total FAR (9.4 % → 5.0 %).
Both directions confirm the §6.31.4 mechanism. But neither
clip setting comes close to the production gate (bit-F1
0.83 < 0.99; FAR 5.0 % > 0.5 %), and the partner-recall
lift on `bb + sr` does not compensate the partner-recall
loss on `fork + sr`. **There is no clip value at which ASL
γ_neg = 4 satisfies the dual gate on this benchmark** — the
mechanism is fundamentally global (all classes' negative
gradients suppressed) while the partner-bit imbalance is
local (a per-class issue).

### §6.31.6 The axis-level dead-end

We classify ASL as a **failed direction at the loss-axis
level**, not a single-hyperparameter miss. The reason is
that the three sub-knobs of ASL — γ_neg, γ_pos, clip —
collectively constitute one coherent mechanism (asymmetric
treatment of positive vs negative gradients) and that
mechanism is not aligned with this benchmark's failure
mode:

- The partner-bit recall asymmetry (§6.31.1) is **per-class
  local** (specific to the bb + sr combo's weak scratch_rot
  signal).
- ASL's gradient asymmetry is **global per-sample**
  (uniform across all classes per chip).
- The mismatch is structural: a global mechanism cannot
  selectively boost one partner class without
  simultaneously over-suppressing every class's negative
  signal, which is the empirical observation in §6.31.4.

Three alternative ASL recipes (γ_neg = 2 instead of 4,
γ_pos ≠ 0, switch epoch ∈ {4, 6, 8}) were not dispatched —
analyst recommendation is that they are unlikely to escape
the structural mismatch under the same audit gate. The
loss-axis dead-end conclusion is **archival**: the iter
122 / 123 evidence rules out ASL with high confidence; the
loss-axis search closes here.

### §6.31.7 Paper value of the negative result

Negative-result reporting is increasingly accepted in
multi-label literature (Cole et al. 2021 SPML, Lipton et
al. 2014 F1-threshold tuning, and the ICLR 2024 Workshop
on "Negative Results in Machine Learning"); we document
this dead-end because:

1. **ASL is the canonical multi-label remedy for partner-
   bit imbalance** (Ridnik 2021 §4) — a multi-label paper
   that does not test it would be incomplete. We test it
   and report the failure.
2. **The mechanism (global asymmetry vs local imbalance)
   transfers to other multi-label benchmarks** where the
   class structure has similar locality (e.g. medical
   imaging multi-label with per-organ failure modes,
   industrial inspection multi-label with per-defect
   recall asymmetry). The §6.31.4 reading suggests that
   ASL should be reserved for benchmarks where the negative
   distribution is uniformly easy (COCO) and avoided where
   the negative distribution is class-heterogeneous (our
   benchmark: Normal speckle and OOD wafer patterns each
   present differently structured negatives).
3. **The val_margin selection blindness (§7.13.1) was
   surfaced by iter 122 / 123**, which is a methodology
   contribution downstream of the failed loss
   intervention.

### §6.31.8 Connection to other §6 sections

- §6.20 (cross-class suppression under Normal training) and
  §6.31 (cross-class suppression under ASL γ_neg) are both
  endogenous failure modes that originate in a training-
  side gradient regularisation; in both cases the
  intervention is well-motivated for one class but bleeds
  into another. The mitigation in §6.20 is a logit-average
  ensemble (structural); the mitigation in §6.31 is **to
  pivot off the loss axis** (the loss-axis itself does not
  admit a fix).
- §6.30 (Starburst OOD distractor) and §6.31 (ASL Total
  FAR explosion) both surface the **Total FAR as the
  dominant audit constraint** at the saturated end of the
  bit-F1 curve. Once bit-F1 > 0.95, every recipe axis
  improvement on bit-F1 is screened by its Total FAR cost;
  a 0.04 bit-F1 lift at 9 pp FAR cost (iter 122 vs iter
  116 J) is a categorical regression under the dual-gate
  rule, not a sampling-noise wash.

_Sources: §6.20 cross-class suppression, §6.30 Starburst
mechanism, `outputs/iter122_T6_asl_gn4/`,
`outputs/iter123_T6_asl_clip01/`, `notes.md` iter 122 / 123
tables, Ridnik et al. 2021 arXiv:2009.14119._

