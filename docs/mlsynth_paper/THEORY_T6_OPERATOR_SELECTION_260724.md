# T6 — Source-only operator-selection: the possibility/impossibility boundary (260724)

New CENTRAL theorem cluster (per the 260724 plan). Elevates the paper from
"single-label sources cannot recover the multi-defect DISTRIBUTION" (T1) to the
strictly stronger "single-label sources cannot even recover WHICH synthesis
OPERATOR to use, and here is exactly how much that costs and what buys it back."
This directly formalizes the empirically observed proxy failure (Severstal:
evidence-margin proxy picked `mixup`, `partition` had the best mean; chip/WM38:
different operators win) as an information-theoretic law, not a fixable bug.

All quantitative claims are machine-checked in
`multilabel_synth/verify_t6_selection_regret.py` (T6a game value, T6b minimax=maximin
LP duality over 200 random instances to 1e-15, T6d worst-case regret rate slope
-0.500, FCM coherence formula == brute enumeration to 0).

Positioning vs priors (must cite): Ben-David et al. 2010 (impossibility of
domain-adaptation without assumptions) and Agarwal-Zhang 2022 (minimax-regret
optimization) are the nearest frameworks; the delta is a LOW-FAR-specific,
utility-vector selection-regret with a normal-resource limit (T6c) and a
positive-resource rate (T6d) that neither states.

---

## Setup

Observed information `I = ( {P(x | y=e_c)}_c , P_0 )` = single-class marginals +
known-good normal law. Candidate synthesis operators `j = 1..K` (single_only,
mixup, cutmix, summation, partition, grid_complement, ...). A composition world
`Q in U(I)` is any joint multi-defect appearance law consistent with `I` (T1: this
set is nontrivial -- the copula is free). Training on operator `j`'s synthetic
combos and deploying at the normal-calibrated FAR-`alpha` threshold yields a bounded
**utility** `u_j(Q) in [0,1]` = FAR-`alpha`-constrained positive detection score
(bounded positive loss / recall on genuine multi-defect inputs; we prove everything
on this bounded utility and recover bit-F1 as a corollary when prevalence > 0, to
avoid F1's non-linearity). Write `u(Q) = (u_1(Q),..,u_K(Q))`.

A **selector** is a (possibly randomized) rule `p = p(I) in Delta_K` choosing an
operator from `I` alone (test-blind). Its **regret** in world `Q` is
`reg(p,Q) = max_j u_j(Q) - p . u(Q)` (gap to the best operator in hindsight).

---

## Theorem T6a (selection impossibility) [PROVEN + machine-checked]

**Claim.** There exist two worlds `Q_A, Q_B in U(I)` with IDENTICAL information `I`
(same single marginals AND same normal law `P_0`) whose best operators are
DIFFERENT: `argmax_j u_j(Q_A) = a != b = argmax_j u_j(Q_B)`, with utility gap
`Delta > 0`. Consequently:
- every DETERMINISTIC `I`-measurable selector has worst-case regret `>= Delta`;
- every RANDOMIZED selector has worst-case regret `>= Delta/2`, attained by the
  balanced mixture -- i.e. `Delta/2` is the exact value of the 2-world game.

**Construction.** By T1 the copula is free at fixed marginals; pick `Q_A` a
composition law whose interaction favours operator `a` (e.g. a superposition-like
overlap that `summation/mixup` reproduce) and `Q_B` favouring `b` (a
region-partition that `partition` reproduces), both with the SAME single marginals
and (T4'(a), N-ORTH) the SAME derived normal law. Both lie in `U(I)`; the operators'
utilities flip while `I` is fixed.

**Proof.** With `u(Q_A)=(U, U-Delta)`, `u(Q_B)=(U-Delta, U)` on the two relevant
operators, a selector `p=(p_a,p_b)` has `reg(p,Q_A)=p_b Delta`, `reg(p,Q_B)=p_a
Delta`, so worst-case `= Delta max(p_a,p_b) >= Delta/2`, minimized at `p_a=p_b=1/2`.
Deterministic `p in {0,1}^2` gives `Delta`. QED. (Machine check: `det=0.400=Delta`,
`rand=0.200=Delta/2`.)

**Reading.** Stronger than T1: even GRANTING that you will only ever *select* among
operators (not estimate the law), the selection itself is not identified from `I`.
The Severstal proxy picking the wrong operator is an instance of this `>= Delta/2`
floor, not an implementation error.

---

## Theorem T6b (exact minimax selection-regret + dual) [PROVEN + machine-checked]

**Claim.** The minimax source-only selection regret is
```
   V(I) = min_{p in Delta_K}  sup_{Q in U(I)} [ max_j u_j(Q) - p . u(Q) ],
```
and equals its convex-game DUAL (least-favourable prior over worlds)
```
   V(I) = max_{pi in P(U(I))} [ E_{Q~pi} max_j u_j(Q) - max_j E_{Q~pi} u_j(Q) ].
```

**Proof.** The map `(p, pi) -> E_pi[max_j u_j - p.u]` is bilinear; `Delta_K` is
compact convex; the utility-image `{u(Q): Q in U(I)}` has compact convex hull (work
on it, since both objectives depend on `Q` only through `u(Q)` and are linear in
it). Sion's minimax theorem gives `min_p max_pi = max_pi min_p`; `min_p E_pi[-p.u]
= -max_j E_pi u_j` (best response is a vertex), yielding the stated dual. QED.
(Machine check: for 200 random finite `U(I)` instances the primal LP value equals
the maximin dual LP value to `1.2e-15`.)

**Reading.** `V(I)` is a single scalar summarizing how ill-posed operator selection
is for this domain's information. The dual exhibits the least-favourable mixture of
composition worlds -- the adversary that any test-blind proxy must face. This is the
selection analogue of Agarwal-Zhang minimax-regret, specialized to FAR-constrained
operator utilities.

---

## Theorem T6c (normals cannot reduce the selection regret) [PROVEN]

**Claim.** If the two adversarial worlds of T6a share the normal law `P_0` (they do,
by construction under N-ORTH), then observing `m -> inf` known-good normals does NOT
decrease `V(I)`: `V(I, m normals) = V(I)` for all `m`.

**Proof.** Normals identify only `P_0` (T3) and, at most, the `P_Z`-mixing shell
(T4'(d)); by Lemma 4'(a') the per-`z` copula core -- which is what distinguishes
`Q_A` from `Q_B` -- is normal-invariant. So `Q_A, Q_B` remain both consistent for
every `m`; the T6a game is unchanged; its value `Delta/2 <= V(I)` persists. QED.

**Reading.** This is the selection-level statement of the T4' resource separation
and the exact reason MORE normals (or a better normal-calibrated proxy) cannot fix
the Severstal mis-selection: the missing information is the copula, which only
positive co-occurrence carries (T6d).

---

## Theorem T6d (minimal positive resource collapses the regret; matching bounds) [PROVEN + machine-checked]

**Claim.** Given `m` i.i.d. genuine target multi-positive validation examples (the
expensive resource -- a few real co-occurrence images, image-level, no location),
the selector "estimate each `u_j` on the `m` positives, pick the empirical argmax"
has expected regret
```
   E[reg] = O( sqrt( log K / m ) )      (UPPER),
```
and no selector can do better in the worst case:
```
   inf_selector sup_{utilities} E[reg] = Omega( sqrt( log K / m ) )   (LOWER).
```

**Proof.** Upper: each `u_j` is a bounded-`[0,1]` mean; `m` positives give
`|hat u_j - u_j| <= sqrt(log(2K/delta)/(2m))` uniformly over `K` operators
(Hoeffding + union bound); picking `argmax hat u_j` costs at most `2 max_j |hat u_j
- u_j|` regret. Lower: standard `K`-armed best-arm-identification / two-point-per-arm
Le Cam construction with gap `~ sqrt(log K/m)`. QED. (Machine check: the WORST-CASE
regret over the gap has log-log slope `-0.500` = `O(1/sqrt m)`; the fixed-gap regret
decays faster, as expected -- the rate is minimax over gaps.)

**F1 corollary.** When the target pair has prevalence `> 0`, bit-F1 is locally
Lipschitz in the bounded utility, so the same `O(sqrt(log K/m))` rate transfers to
F1 (constant depends on prevalence). We prove on the bounded utility to avoid F1's
degeneracy at zero prevalence.

**Reading.** The map is complete: single marginals + normals CANNOT select the
operator (T6a-c); a small labeled positive set CAN, at the standard `sqrt(log K/m)`
rate (T6d). This is the actionable message -- "buy a few real multi-defect labels to
pick the operator," quantified.

---

## Theorem T6-FCM (grid-complement footprint-coherence) [PROVEN + machine-checked]

*The mechanism behind WHICH operator wins per domain, as an exact combinatorial
result -- the novel piece vs the general MSDA unified analysis (Park et al.,
NeurIPS 2022), which does not model connected-evidence preservation under a grid
mask.*

**Setup.** The grid-complement operator partitions an `N = g x g` grid and assigns
a random `m`-subset of cells to source A (the rest to B). A defect with a CONNECTED
footprint occupying `r` cells is preserved iff all `r` of its cells fall on its
source's side.

**Claim.** For two disjoint footprints of sizes `r_a` (source A) and `r_b`
(source B), the probability BOTH are fully preserved is
```
   P(r_a, r_b) = C(N - r_a - r_b, m - r_a) / C(N, m).
```
For fixed `m/N`, `P` decays super-polynomially in the footprint sizes: compact
footprints (`r` small, grid-aligned -- the CHIP regime) are largely preserved, while
extended footprints (`r` large -- the continuous-defect / STEEL regime) are
destroyed.

**Proof.** The A-set is a uniform `m`-subset of `N`; it must contain all `r_a`
A-footprint cells and none of the `r_b` B-footprint cells; the remaining `m - r_a`
cells are chosen freely from the `N - r_a - r_b` non-footprint cells. Ratio to
`C(N,m)` gives the formula. QED. (Machine check: formula == exact enumeration to 0;
chip `g=9` (N=81, m=27): `P` = 0.225, 0.050, 0.011, 0.0005, ~0 for `r = 1,2,3,5,8`.)

**Reading (consistent-not-causal).** This gives the mechanism hypothesis a number:
grid-complement preserves compact grid-aligned defects (chip) but exponentially
destroys extended ones (steel), matching the observed chip-vs-Severstal reversal.
It is a CONSISTENT explanation, NOT a causal proof -- no same-domain mask ablation
was run; we label it as such.

---

## Theorem T6-HEDGE (constructive counterpart: minimax-regret operator MIXTURE) [PROVEN]

*The constructive dual to the impossibility: instead of one proxy-selected operator,
play the minimax-regret MIXTURE.*

**Construction.** On source-validation, form the `K x K` cross-operator risk matrix
`R_{jk}` = utility of operator `j` when the target is (the world induced by) candidate
synthetic law `k`. Solve the minimax-regret LP `p* = argmin_p max_k [max_j R_{jk} -
(p.R)_k ]` (the T6b game restricted to the candidate synthetic laws). Deploy the
operator MIXTURE `p*` (ensemble the arms by `p*`).

**Claim.** If the true target composition law is within TV-distance `delta` of the
convex hull of the candidate synthetic laws, the deployed target regret is bounded
by the game value plus `O(delta)`:
```
   reg_target(p*) <= V_candidates + L_u * delta,
```
`L_u` the utility's TV-Lipschitz constant.

**Proof.** `p*` achieves `V_candidates` against the worst candidate; a target within
`delta` of the hull moves every utility by `<= L_u delta` (utility TV-Lipschitz), so
the regret inflates by at most `L_u delta`. QED.

**Reading + honest scope.** This turns "you cannot pick one operator" into "then
hedge across them with the game-optimal mixture, and you pay only the hull-distance."
Validation plan: FIRST on 8 controlled composition cells (where the true law is
known and `delta` is set), Severstal as EXPLORATORY illustration only. The mixture is
NOT called confirmatory until it wins on a NEW untouched public benchmark with a
pre-frozen `p*`.

---

## Honest scope and probability

- T6a-d are the new CORE. They are honest LOWER + UPPER bounds with matching rates
  (T6d) and an exact game value (T6a, T6b), all machine-checked. They subsume the
  proxy failure as theory.
- T6-FCM is an exact combinatorial mechanism (novel vs MSDA), explicitly
  consistent-not-causal.
- T6-HEDGE is the constructive counterpart; confirmatory status withheld pending a
  new public test.
- Independent proof audit still required (per plan Section 5): T6a/T6d
  lower+upper, T6b minimax dual, T6-FCM combinatorial -- the machine checks pass; a
  human/independent-agent proof audit is the gate.

**Honest probability (decision range, not a measurement).** With soundness locked
(Severstal corrected) and T6 added but NOT yet independently proof-audited: ICLR
~**30-40%**. If T6a-d + T6-FCM + T6-HEDGE pass independent audit and the paper is
recentred on the selection boundary (title change, T1-T5 compressed to background):
~**35-45%**. Only a pre-frozen method winning on a NEW untouched public benchmark
would reach ~45-55%. No basis for 60%. This path converts the negative results
(proxy failure, FCM domain-specificity) into the paper's main theoretical
contribution rather than hiding them.
