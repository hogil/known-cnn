# Low-FAR identifiability for single-only multi-label learning (260721)

Day-1 formalization for the pivot: from an FCM-PM SOTA paper to a
**theory + protocol paper on the low-FAR recoverability of single-only
multi-label learning.** Central claim: single-only training is missing **two
independent pieces of information**, and each has a precise identifiability
status.

> 1. the **composition law** that fixes multi-defect *appearance* -> partially
>    recoverable only when an operator matches the domain's true law
>    (operator-match); irreducibly non-identifiable otherwise (appearance floor).
> 2. the **normal-score tail** that fixes the *false-alarm rate* -> provably
>    non-identifiable from defect data alone; recoverable with a *minimal*
>    amount of known-good normal calibration.

FCM-PM is retained only as the concrete method for the *partition* composition
law; summation/overlay is the method for the *superposition* law. Neither is
claimed to be universally best -- that is a feature of the theory, not a gap.

---

## 1. Setup and notation

Classes `c in {1..K}`. A wafer/image `x` carries a defect label
`y in {0,1}^K`; `y = 0` is **normal** (defect-free). The learner observes:
- a **single-defect** sample `S = {(x_i, y_i) : |y_i| = 1}` (one class each), and
- optionally a synthesizer that maps singles to pseudo-combos.

At test time the model scores `p_c(x) in [0,1]`; a decision rule flags class `c`
iff `p_c(x) >= tau`. The **sample-level normal false-alarm rate** is
`FAR(tau) = P_{x ~ P_0}[ max_c p_c(x) >= tau ]`, where `P_0` is the (unobserved)
normal distribution. The deployment goal is high bit-F1 on genuine multi-defect
inputs **subject to** `FAR(tau) <= alpha` for a small `alpha` (0.1-1%).

---

## 2. Theorem 1 (Composition non-identifiability) -- restated

Two joint appearance models can share identical single-class conditionals
`P(x | y = e_c)` yet differ on the multi-defect conditional `P(x | y = e_a+e_b)`
(the copula/interaction is a free parameter). Hence no content-blind synthesizer
is a consistent estimator of the true multi-defect appearance in general; the
excess-risk bound splits into a **geometry** term (closed when the synthesis
operator matches the domain's combination law) and an **appearance floor**
`A*(S)` that is irreducible. (Full statement + Le Cam two-point construction in
`THEORY_LOWERBOUND_260718.md`; unchanged here.) Operator-match is the
partial-recovery result: choose summation/union on superposition domains,
partition/complement (FCM-PM) on partition domains.

---

## 3. Theorem 2 (Normal-free low-FAR impossibility)

**Claim.** Let a learner observe only defect data (single-defect distribution
`P_d`, plus any synthesizer built from it) and never observe the normal
distribution `P_0`. Then for every decision rule it can produce -- i.e. every
threshold `tau` measurable w.r.t. the defect data alone -- and every target
`alpha < 1`, there exists a normal distribution `P_0` consistent with all
observed data such that `FAR(tau) = 1 > alpha`. Therefore **no defect-only
(normal-free) procedure admits a distribution-free guarantee `FAR <= alpha`**,
unless it also sacrifices all detection (`tau > 1`, flags nothing).

**Non-degeneracy.** The claim is non-vacuous only against rules with a
*detection floor*: fix a recall floor `R(tau) >= r0 > 0` on genuine defects (or,
equivalently, a nonempty accepted region on defect inputs). The escape
`tau > max score` (accept nothing) is excluded as a degenerate solution, on the
same footing as reject-all in Thm 5. Under this floor, the impossibility bites.

**Proof (two-point).** The learner's chosen `tau = tau(P_d)` does not depend on
`P_0`. Fix any such `tau` with `R(tau) >= r0` (so `tau` is below the top of the
defect score range, hence below some achievable normal scores). Consider two
worlds with the *same* `P_d` (so the learner behaves identically and picks the
same `tau`):
- World A: `P_0` places its mass on `{x : max_c p_c(x) < tau}` -> `FAR = 0`.
- World B: `P_0` places its mass on `{x : max_c p_c(x) >= tau}` -> `FAR = 1`.
Both are valid normal distributions and neither is observed by the learner, so it
cannot distinguish them. The single rule `tau` gives `FAR = 1 > alpha` in World
B. The detection floor forbids the only escape (accept nothing). QED.

**Reading.** The impossibility is **information-theoretic, not method-specific**:
it holds identically for single-only, cutmix, mixup, FCM-PM, summation, oracle.
It is a *narrow* claim -- it concerns normal-free FAR control only, and makes no
statement about whether *unlabeled* real data in general is useless (it is not
the [[project_mlsynth_paper_b_direction]] over-claim). It formalizes the
practitioner intuition that "false alarms are fatal and cannot be certified
without ever looking at good product."

---

## 4. Theorem 3 (Minimal known-good calibration)

**Claim (corollary of split-conformal, not a new theorem).** Let `m` i.i.d.
known-good normals `z_1..z_m ~ P_0` be available (no multi-label, no location --
just "this wafer is good"). Let `s_j = max_c p_c(z_j)` and set
`k = ceil((1-alpha)(m+1))`, `tau = s_(k)` (the k-th order statistic), with the
convention `tau = +inf` when `k = m+1` (i.e. `alpha < 1/(m+1)`: the sample is too
small to certify `alpha`, so accept nothing at that level). Flag iff `s > tau`
(strict), or use randomized tie-breaking, so ties do not inflate FAR.

*Two distinct guarantees, stated separately:*
1. **Marginal (exchangeability).** For a fresh exchangeable normal `z`,
   `P[ max_c p_c(z) > tau ] <= alpha`, distribution-free. The probability is over
   the joint draw of calibration + test (marginal, not conditional on the drawn
   `tau`).
2. **Calibration-conditioned population bound.** Conditioned on the realized
   calibration set, the *population* FAR `F = P_{x~P_0}[s(x) > tau]` is a random
   variable; a finite-sample binomial/DKW upper bound gives
   `F <= alpha + O(sqrt(log(1/delta)/m))` w.p. `>= 1 - delta`. Report this 95%
   upper bound, never `0/N = 0%` (0 alarms in N normals -> 95% UB ~ 3/N).

**Proof.** (1) is the standard split-conformal quantile-lemma on the scalar
nonconformity `s = max_c p_c(.)` over exchangeable normals (Vovk et al.). (2) is
Dvoretzky-Kiefer-Wolfowitz / binomial concentration for the empirical tail. Both
are cited results; we claim neither as novel (see novelty note below). QED.

**Reading.** This is the *minimal-information remedy* dual to Theorem 2: the FAR
guarantee is not an add-on but exactly the smallest supervision that defeats the
impossibility. Known-good normals are abundant in a fab (unlike multi-defect
labels, which are not -- see [[feedback_no_real_combo_label_assumption]]), so a
"practical" arm that uses them is industrially honest, not a leak.

**Novelty w.r.t. conformal risk control.** Theorem 3 itself is a *corollary* of
split-conformal / Learn-then-Test (Angelopoulos et al. 2022; Bates et al. 2021)
-- we do NOT claim the O(1/m) FAR bound as new. The contribution is the
*information-resource separation* below (Thms 4-5): which target (FAR vs
coverage) each information source (normals vs positive co-occurrence) can and
cannot control. That separation is what is new, not the rate.

---

## 4b. Theorem 4 (Normal calibration cannot recover positive appearance)

**Part (a) -- assumption-free: normals do not shrink the reachable-law radius.**
There exist two worlds `W_A, W_B` with (i) identical single-defect conditionals
`P(x|y=e_c)`, (ii) identical normal distribution `P_0`, yet (iii) different
multi-defect conditional `P(x|y=e_a+e_b)` with `TV(P_A(.|e_a+e_b),
P_B(.|e_a+e_b)) >= A*(M)`. No estimator using single-defect data plus any number
`m` of known-good normals distinguishes `W_A` from `W_B`. Hence adding normals
does **not** reduce the positive reachable-law TV radius `A*(M)`; the
identifiability gap on multi-defect appearance is invariant to `m`. *(This part
needs no margin assumption -- it is a pure non-identifiability statement.)*

**Proof of (a).** Take the Thm 1 two-point pair `(W_A, W_B)` sharing all single
conditionals and differing only in the copula of `(a,b)`. Attach the *same* `P_0`
to both (`P_0` is unconstrained by the defect conditionals). Single-defect
samples and `m` normal draws then have identical law under both worlds (singles
by (i), normals by (ii)), so the two worlds are statistically indistinguishable
from the available data and the reachable set still has TV diameter `>= A*(M)`.
QED(a).

**Part (b) -- excess-risk lower bound UNDER Assumption M.** A TV gap does not by
itself force excess *risk*: if the Bayes decision agrees on the overlapping mass,
risk can be zero despite `TV > 0`. We therefore state the risk bound only under

> **Assumption M (margin / decision-flip).** On a region of probability
> `>= p_M` under the mixture of `W_A, W_B`, the Bayes-optimal bit decision for
> pair `(a,b)` differs between the two worlds, with score margin `>= gamma`.

Under Assumption M, Le Cam's two-point method gives minimax excess multi-label
risk `>= c(p_M, gamma) * (A*(M))` with an explicit constant
`c(p_M, gamma) = (p_M * gamma / 2)` (standard testing-to-estimation reduction).
Without Assumption M the bound does **not** hold; we give a counterexample below.

**Counterexample (why M is necessary).** Let the copula difference move mass only
within a region where both worlds already assign the *same* Bayes bit-label
(e.g. both predict `{a,b}` present). Then `TV = A*(M) > 0`, yet every decision
rule incurs identical risk in both worlds, so excess risk `= 0`. Thus part (a)
(non-identifiability) can hold while the part-(b) risk bound fails -- the margin
assumption is exactly what rules this out.

**Reading.** FAR and positive-appearance are governed by *different* information
resources: normals fix the FAR axis (Thm 3), positive co-occurrence fixes the
appearance axis (Thm 1 / part (a)); neither substitutes for the other. Part (a)
is unconditional; the *risk* consequence (part (b)) needs Assumption M. In
experiments, a *practical* arm (single + known-good normals) drives realized FAR
to target yet does not close the mAP gap to the multi-label oracle -- consistent
with, but not a proof of, the separation (see the illustration caveat below).

## 4c. Corollary 5 (Selective-risk form; inherits T4(b))

*This is a corollary of Thms 3-4, not an independent theorem.*

**Definitions.** A selective rule `(g, h)` abstains when `g(x)=0` and else
predicts bits `h(x)`. **Coverage** `cov = P[g(x)=1 | y is genuine multi-positive]`.
**Selective risk** on accepted defects = bit-error on `{g=1, defect}`. **FAR** as
before, on normals.

**Statement.** Fix a coverage floor `cov >= 1-beta` (which excludes reject-all,
whose coverage is 0). With only single-defect data + known-good normals: FAR is
controllable to `alpha` (Thm 3), but the selective *bit-risk* on accepted
multi-defect inputs is bounded below by `c(p_M,gamma) * A*(M)` **under Assumption
M** (inherited from Thm 4(b)); it is not reducible by adding normals. Without
Assumption M only the non-identifiability (Thm 4(a)) transfers, not the risk
bound.

**Proof.** Immediate from Thm 3 (FAR axis) + Thm 4 (appearance axis) restricted
to the accepted set of size `>= 1-beta`. The coverage floor rules out the
degenerate `FAR=0` reject-all solution; it does **not** by itself create the risk
bound -- that comes from Thm 4(b)'s margin assumption. QED.

**Reading.** Guarantees are stated *jointly* over FAR and coverage so
"reject everything" is excluded by definition, but the lower bound's force comes
from Assumption M, not from the coverage floor alone.

---

## 5. Empirical corroboration (no new training)

Reusing the stored WM38 strict conformal summary
(`wm38_strict_all_methods_conformal_summary.csv`; 5 model seeds x 50 calibration
splits; reanalysis `analyze_low_far_transfer.py` ->
`outputs/multilabel_synth/low_far_audit_v1/transfer_table.csv`), realized
**real-normal FAR** by the source of the threshold:

```
| target alpha | SYNTHETIC (normal-free) cal | REAL-split (minimal) cal |
|--------------|----------------------------|--------------------------|
| 1%           | 86.1 - 100.0 %             | 0.9 - 1.1 %              |
| 5%           | 91.6 - 100.0 %             | 4.8 - 5.0 %              |
```

Every operator (single_only, cutmix, mixup, fcm_pm, summation_mixup_shin22,
union_mixup, ...) shows the same pattern:
- **Theorem 2 in practice**: a threshold calibrated on synthetic-zero normals
  (the best normal-free proxy) yields **86-100% real-normal FAR** -- catastrophic
  and method-independent. The synthetic normals are trivially separable, so the
  chosen `tau` sits near zero and the real normal tail sails past it (World B).
- **Theorem 3 in practice**: split-conformal on real known-good normals drives
  realized FAR to the target within +/-0.2 pp, for the *same* operators.

This is the audit's backbone: the "failed" source-only FAR numbers are not a
method failure but the empirical signature of the two-information gap. Every
matched-FAR table also reports coverage, positive-reject, worst-class recall, and
paired CIs (the joint FAR-coverage accounting of Cor. 5).

**T4/T5 illustration (NOT a proof).** At matched real-FAR 1% (real-split
calibration), FAR is pinned for all arms yet mixed-bitF1 spans 0.20-0.83 by
operator. This *illustrates* the resource separation (FAR axis fixed by normals,
appearance axis by operator) but is **not** a proof or causal confirmation of
T4/T5 -- it is consistent with them. We label it as such in the paper.

**SVHN v1 -- boundary evidence only (permanent negative).** The strict full-image
SVHN run (`svhn_full_image_operator_match_v1/`) collapsed to prevalence-level mAP
~0.22 for ALL arms (the probe never learned two-digit house numbers under a small
head). It is retained as boundary evidence for "full-image source-only transfer
collapse," NOT as fair operator ranking, for two reasons made explicit: (i)
unequal optimizer-update budget (single-only 3,150 vs synthesis arms 6,750
images), and (ii) arm-specific synthetic checkpoint-validation. GATE-2 failed
(proxy predicted summation; actual mAP winner fcm_pm; paired CI includes 0). Per
protocol we do not refit on the seen test nor run a v2 on the same test.

---

## 6. Scope and boundary (what is NOT claimed)

- Not claimed: T3 is novel -- it is a corollary of split-conformal / conformal
  risk control (Angelopoulos 2022; Bates 2021). The novelty is the *resource
  separation* (T4), not the O(1/m) rate.
- Not claimed: T4(b)/Cor.5 risk lower bounds hold unconditionally -- they require
  Assumption M (decision-flip margin); T4(a) non-identifiability is
  assumption-free. A counterexample shows M is necessary.
- Not claimed: that unlabeled real (multi or normal) data is useless in general.
  T2 is specifically about *normal-free* FAR *guarantees*, under a detection floor.
- Not claimed: FCM-PM is universally best, nor mAP superiority on public data
  (land-cover: mAP tie). FCM-PM is the partition-law *mechanism* illustrating the
  characterization, not a method-superiority claim.

## 7. Paper role (three contributions)

Retitled: *"What Can Single-Label Sources Identify for Multi-Label Recognition?
Composition, Calibration, and Coverage."*

- Contribution 1: solo-marginal minimax boundary on positive composition/
  appearance (Thm 1, + T4(a) invariance to normals).
- Contribution 2: orthogonal-resource characterization -- normal calibration
  identifies FAR (Thm 3) but not positive appearance (Thm 4); risk form under
  Assumption M (Cor. 5).
- Contribution 3: leakage-safe protocol audit + FAR-coverage accounting across
  WM38 / MNIST-family reversal / land-cover / SVHN-collapse.

FCM-PM, val-margin, NB-reject are *concrete mechanisms illustrating* the
characterization -- not method-superiority claims.

**Honest probability (post proof-audit, conservative).** ICLR ~25-40% (two-point
theorems correct but may read as "expected"; T3 is a known corollary; no clean
public method-win). TMLR ~55-70% (correctness + characterization fit) -- the
realistic primary. Depth lever for ICLR: strengthen T4 beyond two-point (e.g.
quantitative reachable-law geometry), NOT more experiments.
