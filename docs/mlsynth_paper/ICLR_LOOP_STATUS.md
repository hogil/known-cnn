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

### Round 3 (JUDGE re-review) -- ICLR 18% -> 25% (+7pp)
- T4' GENUINELY defused the free-P_0 triviality (Q2): N-ORTH is a named, falsifiable
  variation-independence/ancillarity condition with a real converse (4'b, radius
  1/4->0 when violated) + quantitative price (4'c). "Most convincing part = the
  converse." Not a rename.
- Residual: 4'(a) invariance is near-definitional given N-ORTH; the novel content
  (converse + interpolation) sits in a 2-state toy; the general shell-geometry form
  is CONJECTURED (the depth lever round-1 asked for is still IOU'd). 4'(c) machine
  check moves 0 significance weight (nobody doubted the algebra).
- R1 "position paper" downgraded from dominant to strong-but-contestable (-1/3).
- CO-DOMINANT still-OPEN lever = R2: near-priors unaddressed = conformal novelty
  detection (Bates AoS 2023), Neyman-Pearson classification (Scott-Nowak 2005,
  Tong 2013), Frechet-Hoeffding partial-ID. Deepening T4' does NOT close this.
- ACTION (round 4): (b) positioning DELTA TABLE vs the R2 near-priors -- for each,
  what it gives vs what the SOURCE->TARGET resource separation adds that it does
  NOT imply. Reframes "3 known frameworks reassembled" -> "one organizing
  principle." No GPU, no downside, prerequisite for the manuscript. Then (c) write
  main_iclr.tex around it; (a) shell-geometry / (d) empirical only if a gap remains.
- Probability ledger: r1 18% -> r3 25%. TMLR remains realistic primary. 40%
  presupposes BOTH positioning AND a clean empirical win (neither exists yet).

### Round 4 (positioning delta vs near-priors) -- DONE
- Added Section 8: source x target identifiability MAP (S/N/C x FAR/appearance)
  as the organizing principle; delta table vs conformal / NP / Frechet / SPML;
  "what we do NOT claim" pre-empt paragraph.
- CRITICAL honest flag surfaced: the real subsumption threat is econometric
  two-sample DATA-COMBINATION partial-ID (Cross-Manski 2002, Ridder-Moffitt 2007),
  NOT classical Frechet. Must cite + frame T4' as a new partial-ID result within
  it (it does not actually subsume -- no cross-channel variation-independence
  axis -- but positioning is only safe with the citation). Would have been a
  reviewer kill-shot if missed.
- 16 bib keys to add (listed in doc). Re-JUDGE next (round 5).

### Round 5 (JUDGE re-review) -- ICLR 25% -> 27% (+2-3), PLATEAU approaching
- Positioning converted R2 from kill-shot to incrementality charge (+2-3). But:
  map is EXPOSITION not contribution (5/6 cells relocated known results); doc
  over-claims "map is the contribution" -> reframe to sell T4', map as frame.
- data-combination flag defused IGNORANCE kill-shot but NOT incrementality: the
  "no variation-independence axis" distinction is ASSERTED (line 446), needs a
  formal SEPARATION LEMMA. conformal/NP appearance-axis row HOLDS (real).
- R2 partially reduced (~ -1/2), NOT neutralized. R3 (no clean positive) UNTOUCHED
  and now the single largest cap: ICLR hard-capped ~33-35% by theory+positioning
  ALONE regardless of polish (empirics = tautological illustration + permanent
  negative + tie).
- ACTION (round 6, LAST confident theory increment): (a) prove general shell-
  geometry (close T4'c beyond 2-state toy) + formal data-combination separation
  lemma (harden line 446 assertion -> proposition). ~+4-6pp to the ~35% ceiling.
- After (a): theory EXHAUSTED. JUDGE: stop positioning rounds (diminishing); R3
  cannot move by prose. To exceed ~35% needs ONE pre-registered clean empirical
  win (probe that demonstrably learns + equal budget + matched-FAR bitF1 CI>0);
  base rate low (2 priors failed). 60% ICLR NOT reachable (max ceiling ~45%).
- Probability ledger: r1 18 -> r3 25 -> r5 27. Theory ceiling ~33-35%.

### Round 6 (THEORY: shell-geometry + separation lemma) -- DONE
- Thm 4'(d): general shell = L_Phi * eps_m = (copula-P_Z coupling) x (inverse
  N-injectivity)/sqrt(m); sandwich A_Frechet <= A*_norm(m) <= A*(M) + rate
  Theta(1/sqrt(m)) under N-COMPLETE (kappa_*>0). Shell conjecture CLOSED (as a
  product of two named moduli, NOT a universal linear-in-rho form).
- Prop 4'(d-necessity): universal linear form DISPROVEN (non-injective N -> floor
  above A_Frechet; singleton heterogeneous slices -> shell=100% A*(M) rho-indep).
  Toy = special case {|Z|=2 + injective N + linear coupling}.
- Prop 4'-DC: PROVEN twin (M_ORTH vs M_LINK, identical DC inputs, A*_norm 1/4 vs 0)
  -> no Cross-Manski/Ridder-Moffitt bound derives T4'; line-446 assertion -> proof.
- HONEST correctness fix: 4'(a) full-invariance needs N-ORTH AND (ABSORB)
  (slice-homogeneity), not N-ORTH alone -- scoped; core Lemma 4'(a') + sandwich
  intact. Makes hypotheses exact (strengthens).
- Applied to doc. Re-JUDGE (round 7).

### Round 7 (JUDGE re-review after 4'(d) + Prop 4'-DC) -- ICLR 27% -> 31%, THEORY EXHAUSTED
- 4'(d) genuinely closes the shell (factorization L_Phi x 1/kappa_*, two named
  moduli, universal-linear form DISPROVEN) -- real but non-constructive depth,
  reviewers discount -> +2-3pp. Prop 4'-DC + ABSORB fix = INSURANCE (removes
  subsumption + false-headline kill-shots), 0 upside. ABSORB fix net-positive
  (was a latent kill-shot; core Lemma 4'a' carries the story unconditionally).
- JUDGE: THEORY EXHAUSTED = YES. No pure-theory move clears >+1pp without diluting
  the narrative or being R3 in disguise. Stop theory rounds.
- Ceiling ~33-35% via theory alone; current 31%. Only lever above: ONE
  pre-registered clean empirical win (R3), success ~30-40% (track record 0/3:
  SVHN collapse, land-cover tie, GATE-2 fail), win-ceiling ~45% (NOT 60%).
- R3 experiment must clear ALL 4 gates at once: (1) probe DEMONSTRABLY learns
  multi-defect (mAP >> prevalence); (2) EQUAL optimizer-update budget across arms;
  (3) operator-match prediction PRE-REGISTERED before test; (4) matched-real-FAR
  bitF1, operator-match > content-blind, paired CI strictly > 0. Proceed only if
  gate (1) pre-verified on a pilot.
- FIX cleanup done: Section 8 map cell now consistent with scoped 4'(a) (core
  always irreducible; shell reducible at L_Phi/(kappa_* sqrt m)); stale prob note
  updated to 31% + exhausted.
- Probability ledger: r1 18 -> r3 25 -> r5 27 -> r7 31. THEORY CEILING reached.

### Round 8 (R3: GPU freed, ResNet-18 escalation) -- in progress
- GATE-1 PILOT PASS: ResNet-18 LEARNS SVHN full-image 2-digit (held-out synth mAP
  0.838) vs SmallRGB 0.222 (=prevalence). SVHN v1 GATE-2 collapse was a
  PROBE-CAPACITY artifact of my SmallRGB substitution, NOT fundamental. Also
  partition 0.838 >> cutmix 0.466 on the learnable synth eval (operator-match
  signal). (pilot on held-out SYNTH combos, sealed test untouched.)
- ESCALATION: added --backbone resnet18 + --outdir to the runner (v1 SmallRGB
  negative PRESERVED; ResNet-18 runs into a NEW v2 dir). Phase A proxy re-frozen
  with ResNet-18 (hash 432eb145, predicts summation again -- evidence-margin proxy
  is scale-biased, but GATE-2 judges frozen-proxy vs actual-winner, not human
  guess). Phase B sealed test (ResNet-18, test_protocol hash 8bf38c6a) RUNNING.
- This is the real R3 gate-4 attempt on the REAL sealed test with a probe that
  demonstrably learns. Outcome pending.

### Round 8b (backbone escalation results) -- pretrained learns, scratch collapses
- ResNet-18 (v2) REAL sealed test: partition mAP ~0.34 (collapse); scratch backbone
  does NOT transfer synth->real (synth 0.84 -> real 0.34). Run HUNG at 19/30
  (Windows torch hang, GPU 15GB held @2% util); killed, partial preserved as
  boundary evidence (scratch collapse + hang).
- convnextv2_tiny FCMAE (v3) gate-1 PILOT: at CORRECT lr 2e-4, LEARNS strongly --
  held-out synth partition mAP 0.882 (> ResNet 0.84), single 0.750, cutmix 0.695
  (partition > single > cutmix, operator-match signal). gate-1 PASS.
- BUG found + fixed: proxy_select trained at lr 1e-3 -> destroyed FCMAE features ->
  degenerate proxy (all arms identical margin -0.1171). Fixed proxy_select to use
  2e-4 for pretrained. The v3 frozen proxy (hash a2593f07) is INVALID (lr bug);
  will re-freeze into a fresh dir with the fix before Phase B (honest bug-fix, not
  p-hacking -- the degenerate proxy was never a valid pre-registration).
- BLOCKED: convnextv2 Phase B (real sealed test) needs GPU; external procs (codex)
  now hold GPU 100%/15GB. Run when free.
- Honest read: pretrained convnextv2 LEARNS (gate-1), but the REAL-test transfer
  (gate-4) is the open question -- both scratch backbones failed it. Outcome
  pending; ICLR still 31% until a clean real-test win lands (cap ~45%).

### Round 9 (JUDGE re-eval: METHOD-SYSTEM framing, not FCM-PM-single) -- ICLR 31% -> 40%
- User corrected the framing: paper = method-SYSTEM (operator-match+best-grid +
  val-margin + NB-reject + minimal calibration, all first-class contributions;
  theory = motivation), NOT FCM-PM-single/theory-only. Prior 31% UNDER-COUNTED
  the system.
- Recalibrated ICLR:
    theory-only (old)                                    31%
    method-system (current, honest)                      40%  (+9)
    + clean PUBLIC real-test win (full system, pre-reg)  55%  (+15)
    + 2 backbones + best-grid + full-system ablation     61%  (+6)
- Load-bearing PUBLIC+reproducible assets = NB-reject (FAR 22.6->1.1%, ~0 recall
  loss) + conformal calibration (86-100% -> ~1%); these move 31->40. Chip flagship
  (0.997/0.68 vs 0.873/16.2) is strong but PRIVATE -> cannot alone cross 60%.
- NEW dominant reject lever: "pipeline of individually-standard components." Rebuttable
  ONLY by narrative: operator-match LAW + pre-registered source-only proxy +
  identifiability map => "theory-predicted law instantiated by matched components",
  not an ad hoc stack.
- 60% REACHABLE but strictly ORDERED: (a) ONE pre-registered PUBLIC full-system
  real-test win FIRST, THEN (b) backbones + ablation. Breadth alone caps ~45%.
- HIGHEST-VALUE ACTION: full system on ONE public domain, pre-registered, to a
  sealed real-test win: best-grid/g swept on source-val + FROZEN -> proxy operator
  -> val-margin -> NB-reject -> conformal -> sealed eval ONCE. best-grid sweep is
  MANDATORY (grid9-fixed under-tunes fcm_pm = unfair to our own method) but MUST be
  source-val-only + frozen or it is a leakage/cherry-pick reject.
- BLOCKED on GPU (external procs 97%). Building the best-grid+NB+conformal runner
  extension now (GPU-free) so it fires the moment GPU frees.
- Probability ledger: r1 18 -> r3 25 -> r5 27 -> r7 31 (theory ceiling) -> r9 40
  (method-system). Path to 60 open, gated on the public full-system win.
