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

**Proof (two-point).** The learner's chosen `tau = tau(P_d)` does not depend on
`P_0`. Fix any such `tau <= 1`. Consider two worlds with the *same* `P_d` (so the
learner behaves identically and picks the same `tau`):
- World A: `P_0` places its mass on `{x : max_c p_c(x) < tau}` -> `FAR = 0`.
- World B: `P_0` places its mass on `{x : max_c p_c(x) >= tau}` -> `FAR = 1`.
Both are valid normal distributions and neither is observed by the learner, so it
cannot distinguish them. The single rule `tau` gives `FAR = 1 > alpha` in World
B. The only escape, `tau > max` score so nothing is flagged, gives zero recall.
QED.

**Reading.** The impossibility is **information-theoretic, not method-specific**:
it holds identically for single-only, cutmix, mixup, FCM-PM, summation, oracle.
It is a *narrow* claim -- it concerns normal-free FAR control only, and makes no
statement about whether *unlabeled* real data in general is useless (it is not
the [[project_mlsynth_paper_b_direction]] over-claim). It formalizes the
practitioner intuition that "false alarms are fatal and cannot be certified
without ever looking at good product."

---

## 4. Theorem 3 (Minimal known-good calibration)

**Claim.** Let `m` i.i.d. known-good normals `z_1..z_m ~ P_0` be available
(no multi-label, no location -- just "this wafer is good"). Set
`tau = s_((ceil((1-alpha)(m+1)))`, the corresponding order statistic of the
normal max-scores `s_j = max_c p_c(z_j)`. Then for a fresh exchangeable normal
`z`, `P[ max_c p_c(z) >= tau ] <= alpha` (marginal, distribution-free
split-conformal). Moreover the realized population FAR obeys a finite-sample
binomial/DKW upper bound; report the 95% upper bound rather than 0/N = 0%.

**Proof.** Split-conformal on the scalar nonconformity `s = max_c p_c(.)` over
exchangeable normals (Vovk et al.); standard. QED.

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

**Claim.** Known-good normals control FAR (Thm 3) but do **not** reduce the
multi-label positive-appearance ambiguity. Formally: there exist two worlds
`W_A, W_B` with (i) identical single-defect conditionals `P(x|y=e_c)`, (ii)
identical normal distribution `P_0`, yet (iii) different multi-defect conditional
`P(x|y=e_a+e_b)`, with TV distance `>= A*(M)` (the reachable-law radius of Thm 1).
No estimator using single-defect data plus any number `m` of known-good normals
distinguishes `W_A` from `W_B`, so the minimax excess multi-label risk stays
`>= c*A*(M)` regardless of `m`.

**Proof (two-point, orthogonal resources).** Take the Thm 1 two-point pair
`(W_A, W_B)` that already share all single conditionals and differ only in the
copula/interaction of the pair `(a,b)`. Attach the *same* normal distribution
`P_0` to both (the construction is free to do so: `P_0` is unconstrained by the
defect conditionals). The learner's observations -- single-defect samples and
`m` normal draws -- have identical law under `W_A` and `W_B` (singles identical
by (i), normals identical by (ii)). Hence any decision rule is measurable w.r.t.
a statistic with the same distribution in both worlds; Le Cam's two-point bound
gives minimax excess risk `>= c * TV(P_A(x|e_a+e_b), P_B(...)) >= c*A*(M)`. The
normals add information about `P_0` only, which is orthogonal to the copula. QED.

**Reading.** FAR and positive-appearance are governed by *different* information
resources: normals fix the FAR axis (Thm 3), positive co-occurrence fixes the
appearance axis (Thm 1); neither substitutes for the other. This is exactly why
in experiments a *practical* arm (single + known-good normals) drives realized
FAR to target yet does not close the mAP gap to the multi-label oracle -- the two
gaps are provably orthogonal.

## 4c. Theorem 5 (Joint FAR-coverage resource bound; reject-all is not a solution)

**Claim.** Consider any rule that may abstain (reject). Fix a coverage floor
`cov >= 1 - beta` on genuine defects. Then with only single-defect data and
known-good normals, no rule attains both `FAR <= alpha` and defect-recall within
`A*(M)` of optimal at coverage `>= 1 - beta`, unless it observes positive
co-occurrence information. The trivial reject-all rule (`FAR = 0`) is excluded by
the coverage floor.

**Proof.** Couple Thm 3 (FAR controllable) with Thm 4 (appearance not
recoverable) under the shared coverage constraint. Reject-all has coverage `0 <
1 - beta`, violating the floor, so it is not admissible. Any admissible rule with
coverage `>= 1 - beta` must decide bit-labels on accepted defect inputs; on the
`(W_A, W_B)` pair those decisions incur excess risk `>= c*A*(M)` (Thm 4), while
FAR is separately pinned by the normals (Thm 3). Hence the joint target is
unreachable without the third resource (positive co-occurrence). QED.

**Reading.** Every guarantee here is stated *jointly* over FAR and coverage, so
the degenerate "reject everything -> FAR 0" escape is ruled out by construction.
The paper's operating-point tables (F1@FAR with reported coverage and
positive-reject) are the empirical form of this joint accounting.

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
method failure but the empirical signature of the two-information gap.

---

## 6. Scope and boundary (what is NOT claimed)

- Not claimed: that unlabeled real (multi or normal) data is useless in general.
  T2 is specifically about *normal-free* FAR *guarantees*.
- Not claimed: FCM-PM is universally best. It is the partition-law method;
  summation is the superposition-law method (WM38: summation wins, as predicted).
- Not claimed: mAP superiority. On public data FCM-PM ties CutMix on mAP; its
  edge is the FAR-controlled operating point (F1@FAR), which is exactly the
  quantity Theorems 2-3 govern.

## 7. Paper role

- Contribution 1: composition non-identifiability + operator-match (Thm 1).
- Contribution 2: normal-free low-FAR impossibility (Thm 2) -- new, narrow.
- Contribution 3: minimal known-good calibration guarantee (Thm 3) -- the remedy.
- Contribution 4: SM2-LF protocol (strict / practical / diagnostic arms) --
  separate doc `SM2_LOW_FAR_PROTOCOL_260721.md`.
- Contribution 5: cross-domain audit (WM38 done; chip aux; MNIST-family reversal;
  land-cover) reframing prior results as evidence of the two gaps.

Target: **TMLR primary** (correctness + characterization fit), **ICLR stretch**
(the two-point theorems are correct but may read as "expected"; depth is the
acceptance risk).
