# T6 — Source-only operator-selection: the possibility/impossibility boundary (260724)

New CENTRAL theorem cluster (per the 260724 plan). Elevates the paper from
"single-label sources cannot recover the multi-defect DISTRIBUTION" (T1) to the
strictly stronger "single-label sources cannot even recover WHICH synthesis
OPERATOR to use, and here is exactly how much that costs and what buys it back."
This directly formalizes the empirically observed proxy failure (Severstal:
evidence-margin proxy picked `mixup`, `partition` had the best mean; chip/WM38:
different operators win) as an information-theoretic law, not a fixable bug.

> **SCOPE CAVEAT (independent proof-audit, 260724 — read first).** This cluster is
> stated over an ABSTRACT utility oracle `u_j(Q)` (the FAR-`alpha` utility operator
> `j` would achieve if the true law were `Q`). At the oracle level the impossibility
> is a clean game-theoretic COROLLARY of T1 + standard best-arm. The LEARNING content
> -- that models trained on operator `j`'s synthetic combos actually realize a
> flipped, gap-`Delta` utility vector -- is imported as a modeling assumption
> (empirically MOTIVATED by the Severstal/chip/WM38 operator reversals, not proven
> from a learner). "Single-label sources cannot recover which operator" is therefore
> a theorem about the oracle game, one realizability assumption away from a statement
> about learners. The gate audit rated this cluster ICLR ~28-33% and NOT a clean
> "theory-strengthening success" until (i) T6a realizability is either proven or
> retagged as conditional [done below], (ii) T6c is stated as a FLOOR not an equality
> [done], (iii) T6d's `log K` lower bound is proven via Fano or downgraded [downgraded].
>
> **Machine-check honesty.** Of the five checks, three (`t6a` arithmetic on the
> assumed matrix, `t6b` LP duality which holds unconditionally by von Neumann,
> `t6-FCM` formula == the count it enumerates) are DERIVATION/IDENTITY checks, not
> theorem validations; only `t6d`'s `1/sqrt(m)` slope and the `t6b` hull-invariance
> check are non-trivial numerics, and `t6d` tests `K=2` so verifies the `1/sqrt(m)`
> factor ONLY. Do not read "machine-checked" as "empirically validated on learners."

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

## Theorem T6a (selection impossibility) [CONDITIONAL on realizable utility-inversion; game-value PROVEN]

*Retag (audit finding 1): the game algebra and value `Delta/2` are proven; the
EXISTENCE of two consistent worlds with FLIPPED best-operator utilities is CONDITIONAL
on operator-utility-inversion being realizable by the learner -- asserted here,
empirically motivated by the observed chip/Severstal/WM38 reversals, not derived from
a learner. The measure-theoretic half (identical marginals + identical `P_0`) DOES
follow from T1 + N-ORTH.*

**Claim (conditional).** IF there exist two worlds `Q_A, Q_B in U(I)` with IDENTICAL
information `I`
(same single marginals AND same normal law `P_0`) whose best operators are
DIFFERENT: `argmax_j u_j(Q_A) = a != b = argmax_j u_j(Q_B)`, with utility gap
`Delta > 0`. Consequently:
- every DETERMINISTIC `I`-measurable selector has worst-case regret `>= Delta`;
- every RANDOMIZED selector has worst-case regret `>= Delta/2`, attained by the
  balanced mixture -- i.e. `Delta/2` is the exact value of the 2-world game.

**What IS proven (measure half + game value).** By T1 the copula is free at fixed
marginals, and (T4'(a), N-ORTH) two copula fields share the SAME derived normal law;
so two worlds with identical `I` and different copulas exist. GIVEN a utility matrix
`u(Q_A)=(U, U-Delta)`, `u(Q_B)=(U-Delta, U)`, a selector `p=(p_a,p_b)` has
`reg(p,Q_A)=p_b Delta`, `reg(p,Q_B)=p_a Delta`, worst-case `= Delta max(p_a,p_b) >=
Delta/2` (min at `1/2,1/2`); deterministic gives `Delta`. So `Delta/2` is the exact
2-world game value. (Machine check `t6a`: `det=0.400`, `rand=0.200` -- ARITHMETIC on
the assumed matrix, constructs no world, trains nothing.)

**What is NOT proven (the realizability gap).** That the two consistent worlds
actually induce the FLIPPED utility matrix `u(Q_A), u(Q_B)` with a definite gap
`Delta` is a LEARNING claim -- it needs a model trained on operator-`a` combos to
beat one trained on operator-`b` when the truth is `Q_A`, and vice versa. This is
imported as an assumption, motivated by the empirical chip/Severstal/WM38 reversals,
NOT derived. Under this assumption T6a holds; without it, only the measure-level
non-identifiability (two `I`-consistent worlds exist) is unconditional.

**Reading.** Conditional on realizable utility-inversion, selection is not identified
from `I` -- stronger than T1. The Severstal proxy mis-selection is CONSISTENT with a
`>= Delta/2` floor; it is not, by itself, proof of one (the proxy could also fail for
finite-sample reasons). Report as "consistent with," not "instance of."

---

## Theorem T6b (exact minimax selection-regret + dual) [PROVEN; soundest piece]

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
(Machine check: primal LP == maximin dual LP to `1.2e-15` over 200 instances -- this
is unconditional von-Neumann duality, so it checks the derivation; the NON-trivial
`hull-invariance` check confirms `V` is unchanged by adding interior worlds to
`1.7e-16`, matching the convex-`reg` argument that `V(I)=V(conv U(I))`.)

**Reading.** `V(I)` is a single scalar summarizing how ill-posed operator selection
is for this domain's information. The dual exhibits the least-favourable mixture of
composition worlds -- the adversary that any test-blind proxy must face. This is the
selection analogue of Agarwal-Zhang minimax-regret, specialized to FAR-constrained
operator utilities.

---

## Theorem T6c (normals cannot push selection regret below the core FLOOR) [PROVEN as a floor]

*Corrected (audit finding 2): the earlier equality `V(I,m)=V(I) for all m` was WRONG
-- it silently imported ABSORB and contradicts 4'(d) (the reducible `P_Z`-shell IS
lowered by normals at rate `O(L_Phi/(kappa_* sqrt m))`). Only a FLOOR is
unconditional.*

**Claim (floor).** More normals never increase regret, `V(I,m) <= V(I)`; and they
cannot push it below the normal-invariant CORE game value:
```
   V(I, m normals)  >=  V_core(I)  :=  (2-world game value over core-distinguishable worlds)  >=  Delta_core/2,
```
for every `m`, where `Delta_core` is the utility gap between two worlds that differ
only in the per-`z` Frechet CORE (Lemma 4'(a'), normal-invariant).

**Proof.** Normals identify `P_0` (T3) and at most the `P_Z`-mixing shell (T4'(d));
by Lemma 4'(a') the per-`z` copula core is normal-invariant, so any two worlds
differing only in the core stay both consistent for every `m`; the T6a game over them
is unchanged, giving the floor. If `V(I)` is instead driven by SHELL worlds, normals
DO lower it toward `V_core` at the 4'(d) rate -- so `V(I,m) in [V_core, V(I)]`, not
`= V(I)`. QED.

**Reading.** The selection-level T4' separation: normals cannot reduce selection
regret below the copula-core floor, no matter how many. This is why a better
normal-calibrated proxy cannot be GUARANTEED to fix a core-driven mis-selection; the
missing information is the copula, which only positive co-occurrence carries (T6d).
(If the Severstal mis-selection were shell-driven, more normals could in principle
help -- we do not claim to know which; the floor is the guaranteed part.)

---

## Theorem T6d (minimal positive resource collapses the regret) [UPPER + Fano LOWER proven; matched Theta(sqrt(log K / m))]

**Claim.** Given `m` i.i.d. genuine target multi-positive validation examples (the
expensive resource -- a few real co-occurrence images, image-level, no location),
```
   E[reg] = O( sqrt( log K / m ) )               (UPPER, proven),
   inf_selector sup E[reg] = Omega( sqrt( log K / m ) )   (LOWER, proven by Fano below),
```
so the source-plus-`m`-positive selection regret is `Theta(sqrt(log K/m))` -- matched
in BOTH `m` and `K`.

**Proof (upper).** Each `u_j` is a bounded-`[0,1]` mean; the `K` operator MODELS are
trained on SYNTHETIC data (not on the `m` positives), so each `hat u_j` is a mean of
`m` i.i.d. bounded per-example scores -- Hoeffding + union bound give
`|hat u_j - u_j| <= sqrt(log(2K/delta)/(2m))` uniformly over `K` (no cross-arm
independence needed; shared eval samples do not break the union bound); `argmax hat
u_j` costs `<= 2 max_j |hat u_j - u_j|`.

**Proof (lower, Fano -- closes the gate-1 gap the audit flagged; regime `m >~ log K`).**
Build `K` hypotheses `H_1..H_K`; in `H_k` arm `k` has utility mean `1/2 + Delta`, all
others `1/2`. Two hypotheses `H_j, H_k` differ in exactly two arms (arm `j` and arm
`k`), each observed with `m` i.i.d. Bernoulli draws, contributing OPPOSITE-direction
KLs `KL(Ber(1/2+Delta)||Ber(1/2))` and `KL(Ber(1/2)||Ber(1/2+Delta))`; both are
`<= 4 Delta^2/(1-4 Delta^2)`, so
`KL(H_j||H_k) <= 2 m * 4 Delta^2/(1-4Delta^2) = C m Delta^2` with `C` absolute for
`Delta <= 1/4` (needs `m >~ log K` so the gap below is `< 1/4`). The construction is
symmetric, so average pairwise KL `= max` pairwise KL and Fano's `I(theta;X) <= avg
KL` loses nothing. Fano's inequality gives mis-ID probability
`P_err >= 1 - (C m Delta^2 + log 2)/log K`. Set `Delta = c sqrt(log K/m)` with `c`
small enough that `C c^2 <= 1/4`; then `P_err >= 3/4 - log2/log K >= 1/4` for
`K >= 4`. On a mis-identification the chosen arm has mean EXACTLY `1/2`, i.e. regret
`= Delta` (no case gives `< Delta`), so `E[reg] >= Delta * P_err >= (c/4)
sqrt(log K/m) = Omega(sqrt(log K/m))`. QED.

**Machine check (`verify_t6_selection_regret.py`, honesty-corrected).** (i) `t6d`
(`K=2`) log-log slope `-0.500` verifies the `1/sqrt m` factor. (ii) `t6d_fano_lower`
COMPUTES the exact Fano floor `Fano_RHS = 1 - (I + log2)/log K` at the PROOF-VALID gap
`c = 0.17` (so `C c^2 <= 1/4`; an earlier version used `c = 0.5` where `Fano_RHS < 0`
is VACUOUS -- audit catch, fixed): `Fano_RHS = 0.384, 0.551, 0.634, 0.684, 0.718` for
`K = 4..64` -- POSITIVE and GROWING with `K`; the plug-in `argmax` `P_err` (0.65-0.93)
exceeds the floor, and `reg_floor = Delta * Fano_RHS = Omega(sqrt(log K/m))`. So the
Fano LOWER bound (not just plug-in achievability) is now correctly verified.

**F1 corollary.** When the target pair has prevalence `> 0`, bit-F1 is locally
Lipschitz in the bounded utility, so the same `Theta(sqrt(log K/m))` rate transfers
(constant depends on prevalence).

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
For fixed `m/N`, `P` decays in the footprint sizes.

**Proof.** The A-set is a uniform `m`-subset of `N`; it must contain all `r_a`
A-footprint cells and none of the `r_b` B-footprint cells; the remaining `m - r_a`
cells are chosen freely from the `N - r_a - r_b` non-footprint cells. Ratio to
`C(N,m)` gives the formula. QED. (Machine check: formula == exact enumeration to 0 --
a DERIVATION identity, since the formula IS the closed form of that count; not a
domain test.)

**Honest scope of the mechanism (audit finding 5).** The DEPLOYED Severstal arm uses
`g=3` (`N=9, m=3`), NOT `g=9`. At `g=3` the decay is COARSE: `P(1,1)=0.25`,
`P(2,2)=0.06`, `P(3,3)=0.012`, and only `r <= 3` is even feasible -- the
"super-polynomial destruction of extended defects" story is WEAK at the grid actually
run. The `g=9` numbers (`0.225, 0.050, 0.011, ...`) are a hypothetical. Moreover NO
footprint-size of real steel defects was measured, and no same-domain mask ablation
was run. So T6-FCM is: an EXACT formula (proven) + an UNVALIDATED mechanism HYPOTHESIS
for the chip-vs-Severstal reversal (footprint model asserted, grid mismatched to the
deployed arm). Report the formula as the contribution; the reversal-explanation as a
hypothesis to be tested by a same-domain, same-`g` mask ablation with measured
footprints -- not as a result.

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
   reg_target(p*) <= V_candidates + 2 L_u * delta,
```
`L_u <= 1` the utility's TV-Lipschitz constant (finite and known: `u_j(Q)=E_Q[score_j]`
is LINEAR in `Q` with `score_j in [0,1]`, and the FAR threshold is fixed by `P_0`
independent of `Q`, so no discontinuity; the factor 2 is the Neyman-Pearson/TV
coupling constant -- audit finding 7).

**Proof.** `p*` achieves `V_candidates` against the worst candidate; a target within
`delta` of the hull moves every utility by `<= 2 L_u delta` (TV coupling), so the
regret inflates by at most `2 L_u delta`. QED.

**Honest scope -- HEDGE re-expresses, does not escape, the impossibility (audit
finding 7).** `delta` = TV distance of the true composition law to the candidate
synthetic-law hull IS the T1 non-identifiable appearance floor and can be
`Omega(A*)`. So in exactly the hard case T6 is about, the bound `V_candidates + 2 L_u
delta` is VACUOUS -- the mixture pays the uncontrollable `delta`. HEDGE is therefore
NOT a constructive escape from the impossibility; it re-expresses it (selection regret
= game value + uncontrollable hull-distance) and is only useful when the true law
happens to lie near the candidate hull (`delta` small), which cannot be certified
source-only. Validation plan: FIRST on 8 controlled composition cells (`delta` known);
Severstal EXPLORATORY only; NOT confirmatory until it wins on a NEW untouched public
benchmark with a pre-frozen `p*`. We call it a "hedge," not a "constructive
counterpart."

---

## Honest scope and probability (post independent proof-audit, 260724)

An independent opus proof-auditor (tasked to refute) rated the cluster and returned
**gate NOT MET**. Corrected status of each piece:
- **T6a** — game value `Delta/2` proven; the utility-inversion REALIZABILITY is a
  learning assumption, now with a NEGATIVE probe (Round 3): a natural controlled family
  did NOT realize the flip (partition-synth generalized to both worlds). So the
  impossibility is ORACLE-level and its learner-level realizability has a counter-family
  -- weaker than first written. Machine check is arithmetic on the assumed matrix.
- **T6b** — SOUND (Sion; `reg` convex in the utility vector so hull-invariant, now
  machine-checked to 1.7e-16). Strongest piece. The LP-duality check is unconditional
  (von Neumann), so it validates the derivation, not the infinite game.
- **T6c** — corrected to a FLOOR (`V(I,m) in [V_core, V(I)]`); the old equality was
  false without ABSORB and contradicted 4'(d).
- **T6d** — UPPER + Fano LOWER now BOTH proven; matched `Theta(sqrt(log K/m))` in `m`
  AND `K` (gate item 1 CLOSED: Fano construction written + machine-checked, `P_err`
  stays >0 and grows with `K`). The soundest cluster piece alongside T6b.
- **T6-FCM** — formula exact/proven; the reversal-mechanism is an unvalidated
  HYPOTHESIS (footprint model asserted, grid `g=9` mismatched to the deployed `g=3`).
- **T6-HEDGE** — bound correct with constant `2 L_u`; VACUOUS in the hard case
  (`delta = Omega(A*)`); re-expresses rather than escapes the impossibility.

**Gate (plan Section 5): 1 of 3 items CLOSED (Round 1).**
- [CLOSED] T6d Fano `log K` lower bound -- proven + machine-checked (`t6d_fano_lower`).
- [OPEN, with a NEGATIVE probe] T6a utility-inversion realizability. Round-3 attempt
  (`verify_t6a_realizability.py`) built a controlled 2-class bar family with identical
  single marginals and two co-occurrence worlds (overlay vs side-by-side), trained a
  tiny CNN on summation-synth vs partition-synth, and measured which operator wins per
  world. Result: NO inversion -- partition-synth was best or tied on BOTH worlds (World
  A overlay: 1.00 = 1.00; World B side-by-side: partition 1.00 vs summation 0.83). So
  in a natural presence-detection family the utility does NOT flip; partition-synth
  generalizes (consistent with partition winning broadly on Severstal). This is an
  honest NEGATIVE: T6a's realizability is NOT automatic and appears to fail in easy
  appearance-invariant regimes -- it would need a harder task where the co-occurrence
  APPEARANCE (not mere presence) is the label-determining cue (the T1 appearance-floor
  regime). Until such a family is exhibited, T6a's impossibility stays ORACLE-level and
  its learner-level realizability is not just unproven but has one counter-family.
- [OPEN] T6-FCM mechanism -- needs a same-domain same-`g` mask ablation with measured
  footprints (a GPU experiment, queued for the next GPU-free round).

**Honest probability (decision range, not a measurement).** With soundness locked and
T6 added but the gate NOT met: ICLR ~**28-33%** (the auditor's independent number;
the earlier "30-40% / 35-45% post-audit bump" is NOT earned as written). The cluster
is a real, honestly-scoped contribution (a clean oracle-level selection-regret
characterization + an exact combinatorial formula + a hedge), but currently sits at
the level of "corollary of T1 + best-arm with named assumptions," not a standalone
impossibility for learners. Path to ~35-45%: close the three gate items above. Only a
pre-frozen method winning on a NEW untouched public benchmark reaches ~45-55%. No
basis for 60%. The value is converting the negative results into honestly-scoped
theory, not inflating the number.
