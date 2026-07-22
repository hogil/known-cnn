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

## 4b. Theorem 4' (Coupled-nuisance resource separation)

*Replaces the old T4 Part (a): the free-`P_0` independence step is deleted and the
invariance is re-proven with the normal law fully COUPLED to the defect world
through a shared latent appearance nuisance `Z` (fab geometry / sensor /
background). The separation survives coupling, but now rests on a named,
falsifiable, physically-motivated condition -- not a modeling convenience.*

**Coupled model.** Latent `Z ~ P_Z`; kernels `x|(normal,Z=z)~N(z)`,
`x|(y=e_c,Z=z)~D_c(z)`, `x|(y=e_a+e_b,Z=z)~K_{ab}(z;theta)` where the copula field
`theta = {Q_z}` has `Q_z in Frechet(mu_a(z),mu_b(z))` a.e. and
`K_{ab}(z;theta) = oplus_# Q_z`. The learner observes single-defect draws from
`D_c^obs = int D_c(z)dP_Z(z)` and `m` known-good normals from
`P_0^obs = int N(z)dP_Z(z)` (never `Z`, never a pairing, never a multi-label).
Note `P_0^obs` is DERIVED from the shared `(N,P_Z)` -- it is coupled to the defect
conditionals, not attached freely. Reachable radius `A*(M) = (1/2) diam_TV` of the
`Z`-marginalized multi-defect conditional over reachable `theta` (as in Thm 1);
`A*_norm(m)` is the same after conditioning on the `m` normals.

**Assumption N-ORTH (copula variation-independence).** The parameter factors as
`(psi, theta)` with `psi = (P_Z, N, {D_c, mu_c})` generating the observables and
`theta` ranging over the full per-`z` Frechet field `Theta(psi)` *not further
constrained by* `psi`. Equivalently: the observed-data law does not depend on
`theta` (the copula is ancillary for the observables); no functional link
`theta = phi(psi)` is imposed. This is the honest replacement for "`P_0` free":
it forbids the defect-defect interaction physics from being a *known function* of
the geometry/sensor statistics a good-wafer image reveals -- falsifiable
(Thm 4'(b) builds a world where it fails), not automatic.

**Theorem 4'(a) (invariance under coupling) [PROVEN].** Under N-ORTH, for every
`m`, `A*_norm(m) = A*(M)`: the known-good normals do not shrink the copula radius
even though `P_0^obs` is fully coupled to the defect conditionals through `Z`.

*Proof.* Take diameter-realizing copula fields `theta_A, theta_B` for `A*(M)`.
Form `W_A=(psi,theta_A)`, `W_B=(psi,theta_B)` with the SAME `psi`; N-ORTH makes
both admissible (product space). Both draw normals from the identical
`P_0^obs=int N dP_Z` (shared `N,P_Z` -- the coupling is present and IDENTICAL in
both worlds, so it cannot separate them) and singles from identical `D_c^obs`;
the multi-defect conditional is unobserved. Hence `L((S),(Nm)|W_A) =
L((S),(Nm)|W_B)` exactly; the Le Cam two-point argument leaves both
`K_{ab}^obs(theta_A), K_{ab}^obs(theta_B)` reachable, TV apart by the diameter, so
`A*_norm(m) >= A*(M)`; with monotonicity, equality. The load-bearing fact is now
"`theta` variation-independent of `psi`" (N-ORTH), NOT "`P_0` free". QED.

**Lemma 4'(a') (irreducible core) [PROVEN].** Let `A_Frechet = (1/2) diam_TV` over
per-`z` Frechet freedom with `P_Z` pinned at truth. Then `A*_norm(m) >= A_Frechet`
for every `m` and every coupling: even normals identifying `P_Z` exactly cannot
touch the per-`z` couplings (no observable instantiates two defect sources at a
fixed `z`). So `A*(M)` splits into an irreducible core `A_Frechet` (always
normal-invariant) and a reducible shell `A*(M) - A_Frechet` (lives in `P_Z`-mixing,
the only thing normals can reach); N-ORTH says the shell is empty.

**Theorem 4'(b) (converse: N-ORTH is necessary) [PROVEN by construction].** There
is a coupled world violating N-ORTH where `m` normals strictly shrink the radius.
Model S-LINK: `Z in {0,1}`, `P_Z(Z=1)=w` unknown, singles uninformative about `w`,
operator OR, and a *known* copula-geometry link `p11(z)=(1/2)g(z)`, `g(0)=0,g(1)=1`
(so `A_Frechet=0`; only `w` is free), with invertible normal kernel `N(0)!=N(1)`.
Then `P(v=1)=1-w/2`: without normals `w in [0,1]` gives `A*(M)=1/4`; `m` normals
identify `w` and collapse it to `0`. Hence dropping N-ORTH lets normals constrain
`theta` through the shared `Z` -- the assumption is not free. Restoring EITHER a
per-`z` Frechet copula OR a non-invertible `N` restores invariance, so the failure
needs BOTH a modeled copula-geometry link AND normals informative about that
geometry.

**Theorem 4'(c) (quantitative reachable radius) [PROVEN for the instance,
machine-checked].** Interpolate with coupling strength `rho in [0,1]`
(`rho=0` = N-ORTH, `rho=1` = S-LINK): `p11(z)=(1/2)[(1-rho)f_z + rho z]`. With
`m` normals identifying `w` to half-width `eps_m`,

```
   A*_norm(m) = A*(M) * [ (1-rho) + 2 rho eps_m ],        A*(M) = 1/4,
   reduction  g(rho,m) = A*(M) * rho * (1 - 2 eps_m)_+ ,
   eps_m <= c sqrt(log(1/delta)/m) / kappa,   kappa = TV(N(0),N(1)),
   =>  g(rho,inf) - g(rho,m) = O( rho / (kappa sqrt(m)) ).
```

`g(0,m)=0` (orthogonality), `g(rho,0)=0` (no normals), `A*_norm(inf)=A*(M)(1-rho)
=A_Frechet` (core). This is the promised upgrade beyond two-point: a `rho`- and
`m`-explicit reachable radius, `=0` exactly at `rho=0`. The general sandwich
`A_Frechet <= A*_norm(m) <= A*(M)` and the `O(rho/sqrt(m))` scaling are proven;
the universal-linear-in-`rho` shell form is CONJECTURED (only the instance +
sandwich are proven).

**Reading.** Normals identify the FAR axis (Thm 3) and at most the `P_Z`-mixing;
they touch the copula `theta` ONLY through a modeled link `theta = phi(P_Z-facts)`.
Absent it the separation is exact under full coupling (4'(a)); present, the
reduction is capped at the geometry-tied, `N`-identifiable, finite-`m`-damped
fraction (4'(c)) and never breaches the Frechet core (4'(a')). The old free-`P_0`
step is the `rho=0` face of this -- now an assumption with a name, a necessity
proof, and a price. *Pitch: a characterization of WHEN normals can and cannot
borrow information across the shared nuisance -- not an unconditional
impossibility.*

## 4b'. Theorem 4 Part (b) (excess-risk lower bound UNDER Assumption M)

*(The risk consequence of the non-identifiability above still needs a margin
assumption -- unchanged.)*
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

---

## 8. Related work and the delta (positioning vs near-priors)

Our theorems reuse three mature bodies of machinery -- conformal calibration,
Neyman-Pearson (NP) detection, Frechet-Hoeffding/copula partial identification.
Each individual guarantee is standard; we do NOT position on the guarantees. We
position on the OBJECT they assemble into: a **source x target identifiability
map** whose off-diagonal cells (which source identifies which OTHER source's
target) no single framework defines. The one genuinely new cell -- known-good
normals cannot identify multi-defect appearance even when coupled to defects
through a shared nuisance (Thm 4') -- is invisible to every prior below because
none models a *second information source trying to identify the first's target*.

### The source x target map (the contribution is the whole map, not a cell)

```
                 | FAR axis (normal tail)          | Appearance / copula axis (multi-defect conditional)
-----------------|---------------------------------|----------------------------------------------------
S single-label   | NO (T2: defect-only cannot      | PARTIAL under operator-match; irreducible floor
  marginals      |     certify FAR)                |     A*(S) otherwise (T1)
N known-good     | YES (T3: minimal m normals,     | NO under N-ORTH (T4'a); necessity converse (4'b);
  normals        |     a conformal corollary)      |     rho-damped radius (4'c)  <-- GENUINELY NEW CELL
C positive       | NO (co-occurrence positives do  | YES (only source that fixes the copula; the
  co-occurrence  |     not reveal P_0)             |     constructive flip side of T1)
```

Single-label sources fix the marginals but not the copula; known-good normals fix
the FAR axis but not the appearance axis; positive co-occurrence is the only
source that fixes appearance. Every prior occupies ONE cell -- conformal and NP
live in `N->FAR`; Frechet-Hoeffding in the static `S->appearance` region; SPML in
a different regime. NEW = the off-diagonal `N->appearance` impossibility under
coupling (T4'a) + its necessity converse (4'b) + quantitative radius (4'c).

### Delta table

```
| Framework                              | What it gives                                  | What it does NOT give that our map does                       |
|----------------------------------------|------------------------------------------------|---------------------------------------------------------------|
| Conformal novelty / risk control       | Distribution-free finite-sample type-I / risk  | Silent on WHICH source controls which target; takes the score  |
|  (Bates+23; Angelopoulos+ CRC22;       | control on ONE nonconformity score, given      | as GIVEN; never asks if appearance/copula is recoverable;      |
|   Vovk+; Lei+18)                        | exchangeable inlier calibration data           | presupposes the inliers -> cannot even state T2 (no-normals).  |
| Neyman-Pearson / one-class / PU /       | Type-I-constrained optimal detection GIVEN     | Assumes null in hand -> silent on FAR WITHOUT it (T2), on      |
|  open-set (Scott-Nowak05; Tong13;       | null samples; contaminated-null novelty        | multi-label appearance non-ID (T1), and on a 2nd source        |
|   Blanchard-Lee-Scott10)                | detection                                       | crossing to the 1st's target (T4').                           |
| Frechet-Hoeffding / copula partial ID   | Sharp bounds on a JOINT given fixed marginals   | No learning/excess-risk under an operator; no FAR axis; no      |
|  (Frechet51; Hoeffding40; Sklar59;      | (static, single marginal-set)                  | AUXILIARY-source axis -> cannot ask if normals shrink the      |
|   Manski03)                             |                                                | copula radius (that question IS T4').                          |
| Two-sample data-combination partial ID  | Bounds a joint by FUSING two samples (the       | Fuses two MARGINAL samples of the SAME variables; no notion of  |
|  (Cross-Manski02; Ridder-Moffitt07)     | real near-prior for T4' -- see flag)           | a functionally DIFFERENT channel (normal law vs defect law) and |
|                                         |                                                | NO variation-independence condition governing cross-latent leak.|
| SPML (single-positive multi-label)      | Loss/label-estimation when a multi-label image | Different regime: SPML images already contain multiple classes; |
|  (Cole+21; Zhou+22; Liu+23)             | has ONE observed positive                      | ours are physically single-SOURCE, combos never in training.   |
```

### What we do NOT claim over prior work (pre-empt the "reassembled" reading)

(i) **T3 is a conformal corollary** -- the O(1/m) FAR bound is Lei+18 / Bates+23 /
Angelopoulos+22, not ours; we claim its *placement* in the map, not the rate.
(ii) **T1, T2 use standard two-point / Le Cam / NP-style arguments** -- textbook
machinery; T2 is close to intuitive once framed and is sold only as the cell that
makes the map's FAR column well-defined, not a standalone theorem.
(iii) **The per-slice core `A_Frechet` is classical Frechet-Hoeffding** -- we claim
only its *invariance to the auxiliary source* (Lemma 4'(a')), not the bound. The
novelty is exactly the **assembled map + Thm 4'**, nothing more.

### Honest subsumption flag (critical positioning requirement)

The sharpest subsumption threat is NOT classical Frechet-Hoeffding but the
**econometric two-sample data-combination / partial-identification** literature
(Cross-Manski 2002; Ridder-Moffitt 2007 survey; data-fusion / ecological-inference
bounds), which DOES combine samples from more than one source to bound a joint.
We MUST cite it and frame Thm 4' *inside* Manski's partial-ID language, not as if
unaware. The delta that survives: that literature fuses two marginal samples of
the SAME variables to bound their single joint, with no notion of a functionally
different channel and no variation-independence condition (N-ORTH) governing
whether the different channel leaks into the copula through a shared latent. If we
omit these citations a reviewer can say "this is data-combination partial ID"; if
we cite them and frame T4' as a new identification result within that framework,
the delta holds. (Verified: it does not actually subsume T4' -- it has no
cross-channel variation-independence axis -- but the positioning is only safe with
the explicit citation.)

Bib keys to add: bates2023outliers, angelopoulos2022crc, vovk2005algorithmic,
scott2005neyman, tong2013plugin, blanchard2010ssnd, frechet1951tableaux,
hoeffding1940, sklar1959, manski2003partial, crossmanski2002, riddermoffitt2007,
zhou2022ack, liu2023mime. Present: cole2021spml, lei2018conformal.
