# 7. Discussion

We summarise five observations whose generality goes beyond the
specific dataset.

## 7.1 The default hyperparameter trap

The largest win in this paper — α=0.20 over α=0.10 — was not
methodological. It was the result of treating the published
"default" label-smoothing strength as a hyperparameter rather than
a constant. We hypothesise that the same will hold for at least
ASL and BCE→ASL: their published γ values target large-scale
multi-label datasets (MS-COCO, OpenImages) and may be sub-optimal
for our small-data, strong-pretrain setting.

This is consistent with a broader pattern in defect-classification
deployment: practitioners adopt a literature recipe with default
hyperparameters and miss the actual operating point by 5-10
percentage points. The remedy is mechanical — sweep one axis at a
time, keep the rest fixed, decide on a single eval metric — and
the reward in this case is +0.0905 macro-F1 from a 4-hour sweep.

## 7.2 Single-label CE is a strong starting point — for multi-label

A counter-intuitive finding is that a *single-label* CE pretrain on
the same data, followed by *single-label* CE fine-tuning with label
smoothing, beats every *multi-label-native* loss we tested (BCE, ASL,
BCE→ASL). Our explanation is two-part:

1. **Softmax structure is informative for sigmoid decoding.** The
   F1-max threshold sweep relies on the per-class scores being
   well-separated; the softmax constraint that the scores sum to 1
   provides a stronger inductive bias than per-class binary BCE in
   our small-data regime. BCE training removes this constraint and
   the threshold sweep loses leverage.

2. **TAPT init is fragile.** The TAPT backbone has spent its capacity
   building a chip-specific feature representation that single-label
   CE knew how to read. Switching the read-out objective (to ASL or
   BCE) re-shapes the last layer and partially undoes that
   representation in 8 epochs.

For practitioners with strong single-label pretraining and few
multi-label labels, our results suggest:
**don't switch the loss family — tune label smoothing.**

## 7.3 Inference-only methods are a real free lunch — to a point

The first three iters lifted the frozen baseline from 0.7302 to 0.8542
without re-training. That is a +0.124 absolute macro-F1 from
inference-side calibration alone. We attribute this to:

- **F1-max thresholds (I1)** providing the single biggest jump
  (+0.114). Most of the climb is here.
- **Step-search Δ=0.02 (I7)** adding small but consistent gains.
- **Entropy-gate Normal (I10)** adding +0.006 by giving an
  unsupervised class an explicit decoder.

But inference-only methods plateau near 0.85 in our setting. Crossing
0.85 requires re-training, and the right training intervention is the
mildest one (LS).

## 7.4 The entropy gate does not generalise across training regimes

I10's hard cutoff (`H ≥ 0.85·log(C)`) was tuned on T0 and works
through T1 LS=0.10. At T1 LS=0.20 it under-performs I7 by 0.04, and
at T1 LS=0.30 the gap widens to 0.04 also.

This is a structural limitation of fixed-threshold gates: they
assume the model's softmax-entropy distribution is approximately
invariant to training. Label smoothing breaks that assumption.

A robust deployment of I10 would re-tune the H threshold per-model
(or per-checkpoint), exactly the way per-class θ_c is re-tuned. We
do not do this in iter 5 because the budget was committed to the LS
sweep; re-tuning H per-α is on Phase G's list and we expect it to
recover the 0.04 gap.

**Unified hypothesis: logit-sharpness dictates the inference
variant.** Phase A3 (§6.2.1) showed that varying the *epochs* axis
at fixed α=0.20 reproduces exactly the same I10 → I7 → I3 ranking
shift: ep≤5 favours I10, ep=8 favours I7, ep≥12 favours I3. Both
axes (LS strength, training duration) act on the same underlying
quantity — the sharpness of the trained logit distribution — and
the inference decoder's optimum tracks that sharpness:

- **Low sharpness** (under-smoothed entropy distribution): I10's
  hard entropy cutoff fires often and usefully on uncertain chips.
- **Mid sharpness**: F1-max + step-search (I7) finds clean
  per-class operating points without help from a Normal gate.
- **High sharpness** (peaked logits with little entropy
  variability): I3's per-class F1-max + top-K rescue handles
  near-argmax decisions and combos with one rescue.

We hypothesise this generalises beyond LS and epochs: any training
intervention that changes logit sharpness (LR schedule, optimiser
change, data augmentation strength, gradient clipping, ASL γ_+/γ_-,
focal γ, etc.) will move the inference-variant optimum along the
same axis. The practical consequence is that **inference variant
selection cannot be decided once at deployment**; it must be
re-decided whenever the training regime changes by re-evaluating
the small {I3, I7, I10} grid on val. The cost (~3 min per
checkpoint) is negligible compared to a re-train, and the swing
in macro-F1 (0.04–0.05 in our setting) is not.

This is testable in Phase B–F. We commit to re-running the
{I3, I7, I10} grid at every Phase-B/C/D/E checkpoint and to
classify each checkpoint by its winning inference variant. If the
hypothesis holds, the variant-vs-sharpness mapping should be a
near-monotonic function of an entropy proxy (e.g. mean softmax
entropy on val) regardless of which loss family produced the
checkpoint.

### 7.4.1 iter 6 — extension to the loss/augmentation axis

Iter 6's T7c (BCE + LS=0.20 + CutMix `p=0.5`) provides a third
training axis on which to test the unified hypothesis. T7c moves the
training regime in three coupled directions: (i) loss family CE → BCE,
(ii) augmentation set + CutMix, (iii) gradient targets single-positive
→ multi-positive. The combined effect is a *softer* per-class logit
distribution than T1's, because BCE removes the softmax sum-to-1
push-away and CutMix's mixed targets ask the model not to fully peak
any one class on combo chips.

The inference winner moves accordingly: under T1 (CE+LS=0.20) the
winner is I7; under T7c the winner is I10 (T7c__I10 = 0.9271, T7c__I7
= 0.9035, T7c__I3 = 0.9050). The §6.2.1 prediction that "softer
logits → I10 wins" is confirmed on a *third* training axis in
addition to the LS axis (5.5) and epochs axis (5.5/A3).

**Three-axis evidence summary.**

| axis (held vars)                       | low sharpness   | mid sharpness  | high sharpness |
|----------------------------------------|-----------------|----------------|----------------|
| LS strength (ep=8, LR=1e-4)            | α≤0.10 → I10    | α=0.20 → I7    | α≥0.30 → I3    |
| epochs (α=0.20, LR=1e-4)               | ep≤5 → I10      | ep=8 → I7      | ep≥12 → I3     |
| loss/aug (LR=1e-4, ep=8)               | T7c BCE+CutMix → I10 | T1 CE+LS=0.20 → I7 | (none tested)   |

The hypothesis now has positive support on three independent
training axes. We strengthen the claim: the inference-variant choice
is a deterministic function of a single underlying scalar (logit
sharpness, e.g. mean softmax entropy on val), and any training
intervention that moves that scalar will move the inference optimum
along the same axis.

### 7.4.2 The T7a outlier — pathological threshold collapse

T7a (BCE+LS=0.20, no CutMix) is the iter 6 outlier. Its inference
winner is I3 (0.8577) rather than the I10 the §7.4 hypothesis would
predict for a soft-logit checkpoint. The cause is mechanical: T7a's
per-class F1-max thresholds collapse — fork's θ falls to 0.089,
bank_boundary's to 0.134 — because BCE without CutMix gives almost
no negative-class supervision (all 327 train chips are
single-positive, so each non-target class only sees negative
gradient and never benefits from the mixed-positive update CutMix
provides). With θ_fork = 0.089 the F1-max sweep is degenerate, and
I7 / I10 both inherit thresholds that fire on too many chips. I3's
top-K rescue is the only variant that survives because it ignores
the worst-collapsed thresholds via top-K.

The lesson: the §7.4 hypothesis assumes the per-class thresholds are
*non-pathological*. If the training regime causes threshold
collapse (BCE on single-positive small-data without combo
synthesis), the I3 / I7 / I10 ranking is determined by which variant
*tolerates* the collapse, not by sharpness. T7c rescues the
pathology through CutMix's mixed-positive supervision, restoring the
sharpness-driven I10 winner.

This sharpens the testable prediction: at any future checkpoint we
should report **both** mean softmax entropy on val and per-class
threshold spread `max θ_c − min θ_c`. The hypothesis predicts I10 in
the soft-logit regime *only* when threshold spread is bounded
(say, ≤0.6). When threshold spread blows out, I3 becomes the
winner regardless of sharpness.

### 7.4.3 Phase F — BKM transfer is regime-dependent

Phase F (warmup + cosine, EMA 0.95) imported two best-known-methods
from a sister anomaly-detection chart and both regressed
(−0.109, −0.089 macro-F1). The cause is not bug, hyperparameter
mistune, or implementation error — it is *regime mismatch*. Both
techniques assume the training regime that birthed them
(training-from-scratch, abundant data, vanilla CE), and our regime
(small-data + strong-TAPT init + tuned LS) violates the
assumptions:

- **Warmup expects** that early gradient steps are noisy and the
  optimum is far from init. *Our regime*: TAPT places init close to
  the optimum, and 8 epochs at LR=1e-4 is the right scale to
  converge. Warmup spends 2 of 8 epochs near zero gradient and
  cosine spends the last few near zero gradient — effective training
  is reduced to ~3 epochs and LS=0.20's optimum is never reached.

- **EMA 0.95 expects** that >100 useful gradient steps occur. *Our
  regime*: ≈96 effective optimiser steps over 8 epochs at our train
  size. EMA's averaging window covers ~20% of total training,
  oversmoothing the late-epoch sharpening that LS=0.20 needs.

The takeaway is methodological: **BKM transfer is regime-
dependent**. The published-default trap (§7.1) generalises beyond
loss hyperparameters to *any* technique whose recipe assumes a
specific training regime. For practitioners with strong TAPT init
and small datasets, our results suggest skipping warmup, skipping
EMA, and concentrating on LS / data-augmentation hyperparameter
sweeps that *match* the regime.

### 7.4.4 Iter 9 — three more BKM-transfer failures on a new axis pair

Iter 9 tested three orthogonal structural BKMs (drop_path
arXiv:1603.09382, cutmix-rect arXiv:1905.04899, two-LR
arXiv:2110.00476) as 1-axis atomic changes on the T9d (LS=0.07
seed=42) base. All three regress macro-F1 family-mean by 0.05
–0.11 — well outside the ±0.030 single-seed noise floor (§6.7.5).

The iter-9 axes group cleanly with the iter-6 Phase F axes:

| iter | axis              | source domain         | Δ macro_f1 |
|-----:|-------------------|------------------------|-----------:|
|   6  | warmup + cosine   | anomaly-detection chart |    −0.109  |
|   6  | EMA(0.95)         | anomaly-detection chart |    −0.089  |
|   9  | drop_path 0.05    | ImageNet ConvNeXt       |    −0.052  |
|   9  | cutmix-rect ≤0.25 | ImageNet ResNet         |    −0.106  |
|   9  | two-LR (5e-5/2e-4)| ImageNet ResNet (RsB)   |    −0.084  |

Five separate structural BKMs imported from large-data /
training-from-scratch regimes; five separate negative results in our
small-data + TAPT regime. The pattern is now firm enough to elevate
to a hypothesis:

**Regularisation ceiling hypothesis.** The TAPT init has already
placed the chip backbone close to a small-data optimum that does
not benefit from additional regularisation. Each of the five BKMs
above adds a regulariser of some form — temporal (warmup, EMA),
structural (drop_path), spatial (cutmix-rect compresses CutMix's
patch range to the regularising-noise regime), or capacity-budget
(two-LR forces the backbone below its productive learning rate).
In our regime, every additional regulariser is a *cost* on
macro-F1, not a free win.

This hypothesis predicts which future iter axes will fail:
gradient-clip thresholds, weight-decay sweeps, mixup, and any
"Noise injection" technique should also regress unless they
*replace* an existing regulariser (LS or CutMix) rather than
adding to it. Phase G can test it directly by swapping LS=0.07
out for drop_path-only — if drop_path-as-replacement-LS ≈ LS-only,
the ceiling holds.

## 7.5 The asymmetric BKM transfer story

The five-iter trajectory (1 → 9) decomposes BKM-transfer into two
asymmetric regimes:

### 7.5.1 Phase A (LS retune, §5.5) — transfer success

Label smoothing is the only training-side technique that *worked*
on transfer. The Phase-A1 sweep moved α from the literature default
0.10 to our val-tuned optimum 0.20 (CE side, §5.5) and again to the
[0.05, 0.10] band (BCE+CutMix side, §5.7). LS is a single
hyperparameter with a well-understood entropy-redistribution
effect; its action does not interact with TAPT init beyond a
soft-target shift, and tuning it sweeps the same scalar (logit
sharpness) that the inference-variant ranking depends on (§7.4).

The transfer of *the technique* (LS) succeeded. The transfer of
*the published default* (α=0.10) did not. Re-tuning the one
hyperparameter of the technique was the load-bearing step.

### 7.5.2 Phase F + iter 9 — structural BKMs all fail

Five structural BKMs (warmup, EMA, drop_path, cutmix-rect, two-LR)
all fail in transfer to our regime:

- **Warmup, EMA**: change the *time profile* of training — both
  assume more total optimiser steps than we have.
- **drop_path**: changes the *capacity* available per step —
  assumes the backbone benefits from additional stochastic
  regularisation, which it does not under TAPT.
- **cutmix-rect**: compresses CutMix's *patch range* away from
  the combo-dominant tail — discards exactly the signal that
  drove iter-6's bb+sr recall lift.
- **two-LR**: forces the backbone *below* its productive learning
  rate — the loss switch never propagates and the head receives
  high-LR signal into a misaligned backbone.

Each failure is rooted in a different mechanistic mismatch with
our regime, but the same structural property unites them: each
modifies the *training dynamics or capacity regime* assumed by the
TAPT init.

### 7.5.3 The asymmetry

The asymmetric pattern is paper-grade and worth stating directly:

> **In a small-data + strong-TAPT + tuned-LS regime,
> hyperparameter-axis tuning (LS) transfers; structural-axis BKMs
> (warmup, EMA, drop_path, cutmix-rect, two-LR) do not.**

This is *not* a claim that any one of these BKMs is fundamentally
broken — each is well-validated in its source domain. It is a
claim that the source-domain assumptions (large data, training-
from-scratch, vanilla CE, abundant optimiser steps) are
*structurally inverted* in our setting (small data, strong TAPT,
tuned LS, ≈96 optimiser steps), and the BKMs cannot be safely
imported one-at-a-time.

The practical advice for industrial defect-classification
deployments matches the asymmetry:

1. **Sweep your loss-family hyperparameter** (LS strength, ASL γ,
   BCE pos_weight). Defaults are often miscalibrated by 5–10%
   absolute macro-F1.
2. **Skip the structural BKM imports** unless you can replace an
   existing regulariser rather than add a new one. The
   regularisation ceiling (§7.4.4) makes additive regularisers a
   net cost.
3. **Add data-axis interventions** (CutMix, multi-source synthesis)
   when the failure mode is class-decoding (combo recall). The
   iter-6 CutMix gain was the only non-LS technique that worked
   *and* it acted on the data axis, not the training-dynamics axis.

The asymmetric-BKM-transfer story is paper-grade because it has
positive and negative evidence on both sides: 1 successful axis
(LS, with explicit retune), 5 failed axes (warmup, EMA, drop_path,
cutmix-rect, two-LR), 1 successful data-axis intervention (CutMix
in iter 6). Future work should test whether the asymmetry holds for
the queued Phase B/C/D/E loss-family hyperparameter sweeps (these
should *succeed*) and for any further structural-BKM imports
(these should *fail* unless they replace existing regularisers).

### 7.5.4 Eight negative axis attempts confirm the regularisation-ceiling hypothesis

After the T9 family was identified as the iter-8 winner, we ran a
total of **eight orthogonal atomic axes** on top of the
BCE+LS=0.07+CutMix p=0.5 base (or, in two cases, on adjacent
loss-family bases) to test for further headroom. **All eight
under-perform T9.**

| #  | axis                                 | scope        | best Δ macro_f1 vs T9d | source iter |
|---:|--------------------------------------|--------------|-----------------------:|-------------|
| 1  | F1 warmup (start_factor=0.05, 2ep)   | training     |                  −0.10 | iter 6      |
| 2  | F2 EMA(0.95)                         | training     |                  −0.08 | iter 6      |
| 3  | T8 CE-soft + CutMix p=0.5            | training     |                  −0.10 | iter 7/8    |
| 4  | T10 drop_path 0.05                   | training     |                  −0.05 (mean of 2 seeds) | iter 9 |
| 5  | T11a cutmix-rect ≤0.25               | training     |                  −0.11 | iter 9      |
| 6  | T12a two-LR (bb 5e-5, head 2e-4)     | training     |                  −0.08 | iter 9      |
| 7  | T13a ASL γ_neg=2 + CutMix p=0.5      | training     |                  −0.10 | iter 9      |
| 8  | I11 pair-aware threshold (no retrain)| inference    |               −0.007 net | iter 6      |

**Seven of eight are training-side; one is inference-side.** The
seven training-side failures are not the same kind of failure:

- **Temporal regularisers** (F1 warmup, F2 EMA) — assume more
  optimiser steps than we have; both fail at the magnitude
  (≈0.08–0.10) consistent with the §7.4.4 small-data + few-step
  diagnosis.
- **Capacity regularisers** (T10 drop_path) — assume the backbone
  benefits from additional stochastic noise, which the TAPT init
  has already replaced with feature-space alignment; fails by
  ≈0.05 mean across two seeds.
- **Spatial regularisers** (T11a cutmix-rect) — compress CutMix's
  patch range away from the combo-dominant tail; fails by ≈0.11,
  the single biggest training-side regression in the catalogue,
  and a confound with the iter-7 CutMix-p sweep's known p=0.3
  trough (0.8626 vs p=0.5 0.9271).
- **LR regimes** (T12a two-LR) — starve the backbone below its
  productive learning rate; fails by ≈0.08 with a dramatic bb+sr
  recall collapse (0.9563 → 0.4188), the strongest evidence yet
  that the backbone *needs* to update under BCE+CutMix targets.
- **Loss-family alternates** (T8 CE-soft + CutMix, T13a ASL γ_neg=2
  + CutMix) — replace BCE with a different loss on the same
  CutMix base. Both regress by ≈0.10 macro_f1. T8 retains the
  CE softmax-sum-to-1 structure that fights CutMix's mixed
  targets; T13a's ASL γ_neg=2 over-suppresses non-positive class
  probabilities (the same failure mode as T4 in iter 4, see §6.3),
  collapsing per-class thresholds despite preserving bb+sr recall
  (0.95). **Even with CutMix in the recipe, BCE remains the right
  loss** — the loss switch is not a lever to pull.

The single inference-side failure (I11 pair-aware threshold) is a
no-retrain heuristic that adds a +0.05 lower threshold to the
bb+sr pair when the runner-up is consistent with bb+sr. It costs
−0.007 macro-F1 net (small precision hit on the singletons offsets
the recall gain). I11 is rejected for the same reason every
inference-side heuristic is rejected once a retrained model
already captures the capability: the retrained recipe (T9) makes
the pair-aware heuristic redundant.

**Why these eight tighten the regularisation-ceiling claim.** The
five iter-6 / iter-9 axes (warmup, EMA, drop_path, cutmix-rect,
two-LR) had already established that *adding regularisers* to the
T9 base is a net cost. The three new axes (T8, T13a, I11)
generalise the claim:

1. **T8 (CE-soft + CutMix)** shows that *swapping* BCE for a
   different loss on the same CutMix base also under-performs.
   The BCE choice in T9 is not arbitrary — it is the right loss
   for CutMix's per-pixel mixed-multi-hot targets, and replacing
   it with a softmax-style CE-soft loss costs ≈0.10. The §6.6
   BCE+CutMix story is not just one viable recipe among several;
   it is *the* recipe at this scale.
2. **T13a (ASL γ_neg=2 + CutMix)** rules out a hyperparameter-axis
   alternative on a *different* loss family. ASL γ_neg=2 was the
   most promising untested ASL hparam (small enough to avoid T4's
   over-suppression failure in iter 4); even with CutMix in the
   recipe it costs ≈0.10. This is the strongest evidence yet that
   the T9 BCE choice is structural, not a default we happened to
   test first.
3. **I11 (pair-aware threshold)** rules out a no-retrain inference
   heuristic on top of T9. Even when the heuristic is targeted
   at the very combo class (bb+sr) where T9 wins, it costs more
   than it gains because T9 already decodes bb+sr cleanly.

The eight-axis catalogue is the empirical foundation of the
regularisation-ceiling hypothesis. Predictions that follow:

- **Phase B (ASL γ_pos / γ_neg full sweep) should not produce a
  cell that beats T9 by more than 0.05 macro-F1.** T13a already
  ruled out the most promising ASL configuration; the rest of the
  ASL grid should be similarly capped.
- **Any further additive structural BKM (mixup, stochastic
  weight averaging, gradient clipping at non-trivial magnitude)
  should regress** unless it *replaces* an existing regulariser
  rather than adding to it.
- **Replacement-not-additive substitutions** (LS=0.07 → drop_path
  at varying rates, queued as Phase H) are the only structural
  axes still expected to potentially match T9; they would
  *not* prove the ceiling wrong, only that the ceiling is at the
  same height under a different parameterisation.

We treat the eight-axis pattern as a strong negative result: not
"these axes are bad" but "in this regime (small data, strong TAPT,
tuned LS, BCE+CutMix), *no axis we tested* clears the T9 family
mean within the noise floor". The *cumulative* probability of all
eight independent axes regressing simply by chance — given a true
non-zero positive expected effect on any one — is vanishingly
small. The ceiling is real.

## 7.6 TTA and rotation-aware classes do not mix

Test-time augmentation by rotation is one of the strongest single
inference improvements in general image classification. In our task,
it directly damages a class boundary: `scratch` and `scratch_rot`
differ *by* rotation. A 4-view rotation TTA collapses the two
classes; iter 1 measured −0.018 macro-F1 and we permanently
disallow it.

This is a class-taxonomy-aware design rule, not a hyperparameter.
We flag it because it is exactly the kind of "obvious good idea"
that ML pipelines copy from natural-image SOTA without checking
class-level invariants.

## 7.7 Limits of the current best

Our iter-8/9 best (T9 family-mean ≈ 0.94 macro-F1, single-seed std
≈ 0.030, bb+sr recall robust at 0.85–0.96) leaves the following
known weaknesses:

1. **fork single-confident FPs** are no longer the dominant error
   mode but persist. Phase B (ASL γ_- sweep) should target this
   directly — ASL with γ_-=2 may suppress fork-FP without breaking
   bank_boundary recall.

2. **Synthesis-side combo difficulty (eval set).** `min`-blend
   reduces the defect signal of source chips. Strong-defect filtering
   (`--source-strength-pct 50`, queued for sister-repo work) should
   raise the eval-set defect contrast and make combo decoding
   easier. Note that T7c's CutMix-on-train-side already addresses
   the *training-side* combo deficiency — but the eval-set itself
   is still synthesised with `min`-blend, so a second improvement
   path is open.

3. **scratch_rot wide rotation prior.** Stamping `scratch_rot` over
   a wide angular band makes it look like non-rotated scratch on a
   non-trivial fraction of chips. Grade-elevated chips
   (`--grade-mode elevated_2`, queued) should sharpen this
   distinction.

4. **top1_11 trade-off vs T1.** T7c's 11-class single-pick
   accuracy (0.8307) is below T1's (0.8449) by 0.014. The chips
   that move are recovered combo chips (good) and a small set of
   single-defect chips now mis-classified as combos
   (T7c's CutMix budget at p=0.5 is the sweet spot but is not
   free). For downstream wafer-routing where combo recall is
   load-bearing the trade-off is favourable, but for use-cases
   that route on single-class top-1 the T1 result remains
   preferable.

5. **CutMix-p sweet spot is sharp.** ±0.2 in p costs 0.02–0.06
   macro-F1. Production deployments that want T7c's combo recall
   must use `p=0.5` exactly; a Phase-G ablation should test whether
   the sweet spot drifts under different LS strength or epochs.

These five are deferred to post-Phase-A/B work and do not block
the iter-8/9 paper-grade result. Iter 8/9 / 5.10 added two more
limits:

6. **Single-seed measurement is no longer sufficient.** §6.7 shows
   that at the BCE+CutMix family-mean ≈ 0.94, single-seed std ≈
   0.030 means any axis Δ below ≈3 σ ≈ 0.09 macro-F1 cannot be
   distinguished from seed noise on a single-seed sweep. The 3-seed
   T9 std confirmed in §5.10 (0.046) is consistent with this
   estimate; the marginal std of T1 across the same three seeds
   (0.030) is also non-trivial. **High single-seed variance is not
   unique to the T9 family — T1 (CE+LS=0.20) shows
   std ≈ 0.030 across {42, 43, 44}, with seed 42 (0.9268) an
   upper-tail draw and seeds 43/44 (0.8788, 0.8712) closer to the
   underlying mean (0.8923).** The headline iter-5 claim "T1+I7 =
   0.9268" was itself a single-seed upper-tail; the honest 3-seed
   mean is +0.063 lower. Phase G's first item is multi-seed (n≥3)
   confirmation of the T9 band; the variance discipline is now a
   precondition for any further headline-claim moves, and §5.10
   formalises a multi-seed reporting protocol that should be
   applied retroactively to every macro_f1 quoted above 0.92 in
   this paper.

7. **The T9 lift on bb+sr recall has high seed variance.** §5.10's
   3-seed paired comparison shows T9 wins bb+sr by +0.63 (s42),
   +0.14 (s43), and *loses* by 0.09 (s44). The mean +0.225 is
   robustly positive but the per-seed std (0.36) is comparable to
   the effect itself. The iter-6 single-seed claim that "T7c
   raises bb+sr from 0.32 to 0.96" (+0.63) was, at multi-seed mean,
   a +0.225 lift — still real, still operational, but only ≈1/3
   the magnitude of the seed=42 number. Future work that ships
   T9-class recipes for combo-decoding deployments should commit
   to n≥5 seed reporting on the bb+sr axis.

## 7.8 Why we keep the iter-by-iter narrative

We present the work as eight iterations rather than as a single
ablation table because the iters are *not* commutative. I10 only
makes sense after I7 establishes a reasonable threshold baseline.
T1 LS=0.20 only makes sense after iter 4 establishes that T1 (over
T4/T5/T6) is the right loss family. T7c's CutMix only makes sense
after T1 LS=0.20 establishes the saturated single-class regime that
*needs* combo-class supervision to break the bb+sr recall ceiling.
T9's BCE-side LS retune only makes sense after T7c establishes
that BCE+CutMix is the new base. T10/T11/T12 only make sense after
T9 establishes that the LS axis under BCE+CutMix is flat and
further gains must come from a different axis. Skipping any of
these dependencies would produce a paper with a worse final number
and weaker explanations of *why* each step worked.

Iter 6 added an important meta-iteration lesson: not every iter
has to be a forward step. Phase F (negative) and I11 (rejected)
are integral to the narrative — they constitute the search around
T7c that confirms T7c is the right answer to the right question.
**Iter 9 doubles down on this point**: three more atomic axes (drop_path,
cutmix-rect, two-LR) each tested cleanly and each rejected cleanly.
Negative results are first-class citizens of this paper — they are
what allows §7.4.4's regularisation-ceiling hypothesis and §7.5's
asymmetric-BKM-transfer claim to rest on five independent failed
axes rather than a single-iter intuition.

Iter 8 added a complementary meta-iteration lesson: **the
discipline must adapt as the macro-F1 ceiling approaches**. Below
≈0.93, single-seed sweeps were sufficient to call iters cleanly.
At ≈0.94 the per-cell measurement noise becomes comparable to the
typical iter Δ, and the budget must shift from many cells × 1 seed
to fewer cells × n seeds. We document the shift in §6.7 and
implement it for Phase G onward.

Iteration order is itself a methodological choice; we document it
explicitly in §8.
