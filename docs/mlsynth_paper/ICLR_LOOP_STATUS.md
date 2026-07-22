# ICLR probability-maximization loop -- status

Mode: theory-first, low-cost. GPU auto-dispatch allowed under resource gate.
Roles: JUDGE (adversarial review + next action) -> EXPERIMENT/THEORY (execute) ->
FIX (patch paper/proofs) -> repeat until ICLR probability plateaus or user stops.

## Current asset state (commit 46f0421)
- T1 composition non-identifiability + operator-match: solid.
- T2 normal-free FAR impossibility: solid (detection floor added).
- T3 minimal calibration: labeled corollary of conformal (not novel).
- T4 split: (a) assumption-free non-identifiability [solid]; (b) risk bound under
  Assumption M + counterexample [conditional].
- Cor.5 selective-risk (coverage-defined, reject-all excluded).
- Empirical: normal-free-cal FAR 86-100% vs real-split 1% (solid illustration);
  matched-FAR bitF1 0.20-0.83 spread (ILLUSTRATION only, not proof).
- SVHN v1: permanent boundary negative (degenerate, unequal budget).
- land-cover pos2: mAP tie, F1@FAR 5/5 win (partition domain).
- NB-reject (per-pattern Gaussian): FAR 22.6%->1.1%, pos_reject ~0 (works).

## Honest probability (post proof-audit): ICLR 25-40% / TMLR 55-70%.

## Round log

### Round 1 (JUDGE, opus) -- ICLR ~18% (harsh)
- R1 (dominant): all theorems are expected two-point / cited-corollary -> reads as
  position paper, not technical novelty.
- R2: near-priors unaddressed = conformal outlier/novelty detection (Bates et al.
  AoS 2023), Neyman-Pearson classification (Scott-Nowak 2005, Tong 2013), Frechet
  partial-ID (Frechet/Sklar/Hoeffding, Manski). T2/T3 literally are these.
- R3: novel axis T4 has no clean positive evidence (illustration + SVHN degenerate
  + GATE-2 fail); land-cover mAP tie.
- Q2: T4(a) is TRIVIAL -- one-line re-use of T1 pair; all substance hidden in
  "P_0 unconstrained by defect conditionals" (a free-P_0 independence assumption,
  empirically dubious: fab normal & defect share geometry/sensor/background).
- ACTION (highest leverage): (a)-sharpened -- reprove T4 WITHOUT free-P_0, under a
  SHARED appearance-nuisance Z coupling normals & defects; show reachable-law TV
  radius A*(M) still does not vanish with m normals (or explicit floor);
  characterize reachable set as function of m + coupling strength (Frechet-
  Hoeffding geometry, not two-point). Theory-only, no GPU. (c) positioning vs
  conformal-novelty/N-P/Frechet rides along as a related-work paragraph.
- Do NOT spend GPU on DTD/(b) until (a) lands.

### Round 2 (THEORY execution) -- T4' coupled-nuisance: DONE, genuine upgrade
- Replaced free-P_0 T4(a) with Theorem 4' proven UNDER coupling (P_0=int N dP_Z
  shared, not attached). N-ORTH assumption (copula variation-independent of the
  normal-identifiable geometry) is the honest, falsifiable replacement.
- 4'(a) invariance [proven], 4'(a') irreducible Frechet core [proven],
  4'(b) converse: N-ORTH necessary, radius 1/4->0 when violated [proven by
  construction], 4'(c) quantitative A*_norm(m)=A*(M)[(1-rho)+2 rho eps_m],
  reduction O(rho/sqrt(m)) [machine-checked, verify_t4_coupled_radius.py, max
  |closed-brute|=5.6e-17].
- Verdict: no longer a trivial free-P_0 restatement; still an assumption (N-ORTH)
  but substantive with named necessity + price. Defuses JUDGE R1/Q2 on T4.
- Honest: this is a real technical upgrade on the paper's most-attacked axis;
  does NOT manufacture an unconditional theorem. Re-JUDGE next (round 3).
