# Adversarial self-review — top-6 reviewer objections (2026-07-16)

Toughest-possible ICLR reviewer pass over `DRAFT.md` + `latex/main.tex`. Ranked by
likelihood of causing rejection. Each: the objection in one sharp sentence; damage
(fatal / major / minor); and answerability — (a) reframing / text already-true,
(b) a cheap experiment, (c) inherent limitation to concede.

Legend for the orchestrator: **(b)** rows are the experiments to queue; **(c)** rows
are concessions that must stay honest in the paper; **(a)** rows are handled by the
2026-07-16 text edits (see "What was added" at the bottom).

---

## 1. "You defined away the winning baseline." (rank 1)
**Objection:** Max-union / Shin et al. (2022) Summation Mixup dominates the proposed
FCM-PM on *both* axes the paper claims to care about — higher raw bit-F1 (0.80 vs
0.654) *and* lower FAR (0.010 vs 0.147) — and the paper only demotes it by inventing
a new "distributional-faithfulness / density" criterion that happens to be the one
axis where max-union loses.

**Damage:** FATAL if unaddressed — it makes a reviewer distrust the entire framing as
goalpost-moving.

**Answerable by:** (a) reframing + (b) a decisive experiment. The paper already keeps
max-union in every table with its real 0.80/0.010 and grounds the demotion in the
operator-match theorem (the density gap certifies TV >= 0.86, making max-union's
excess-risk guarantee vacuous). The honest limit: the theorem is an *upper* bound, so
it cannot by itself prove max-union is worse — only that its guarantee is uncertified.
The clean kill is the density-shift stress test (Obj. 2 (b)). Text edit added a crisp
"not a defined-away baseline" rebuttal.

## 2. "Over-dense still generalizes, so why is over-density bad?" (rank 2)
**Objection:** Max-union trained on 0.50-density wafers scores the best detector on
the 0.31-density *real* test set, which is direct empirical proof that the density
mismatch is harmless for generalization — the TV bound predicts a penalty the data
does not show.

**Damage:** MAJOR (near-fatal) — this is the internal contradiction at the heart of
the paper.

**Answerable by:** (c) concede honestly + (b) the decisive future experiment. Now
conceded in Discussion: over-density does **not** empirically penalize raw F1 in our
protocol and we do not claim it does; the faithful-operator claim rests on the
certified TV guarantee + the FAR-controlled coverage frontier (density-independent) +
the cross-regime flip, not a raw-F1 win. **(b) EXPERIMENT FOR ORCHESTRATOR — the
decisive test:** a *density-shift stress test* — train faithful (FCM-PM) vs. over-dense
(max-union) synthesis, evaluate under a deliberately shifted test density, at matched
FAR operating points, and on real higher-order (3-/4-) mixes, to see whether the
over-dense law breaks where the current same-density protocol cannot show it. Outcome
is not pre-judged in the paper.

## 3. "Modest absolute performance — not competitive with supervised SOTA." (rank 3)
**Objection:** FCM-PM recovers only 67% of the oracle (0.654 vs 0.974) and
fully-supervised MixedWM38 methods reach 98-99%, so the headline number is far below
anything usable in practice.

**Damage:** MAJOR — top venues routinely reject "interesting but weak numbers."

**Answerable by:** (a) reframing + (c) partial concession. The contribution is
annotation-free multi-label competence + a finite-sample FAR guarantee, not
leaderboard accuracy; the honest comparator is the single-only floor (0.473) and the
annotation cost removed, not the fully-supervised ceiling. Text edit added a
"Contribution type, not leaderboard SOTA" paragraph. Remaining gap (appearance
interaction of real high-order mixes) is conceded as the open problem.

## 4. "Niche / thin benchmark base for a 'general principle'." (rank 4)
**Objection:** A single public real-multi-label benchmark (MixedWM38, itself niche
wafer maps) carries the industrial claim; chip is unreleased internal data, VOC/COCO
are subsampled boundary analyses that *fail*, audio has no oracle pool, and text is a
toy — thin evidence for a claimed cross-regime law.

**Damage:** MAJOR.

**Answerable by:** (a) reframing/positioning + (c) concede scope. Reframe as a
weak-supervision + reliability contribution whose value is cross-regime generality
(5 families, 3 combination regimes) and a certified guarantee, not single-benchmark
SOTA; the VOC failure is a *predicted* boundary, not a hole. Honest venue fit (for the
orchestrator, not the paper body): TMLR, or an ICLR/NeurIPS weak-supervision /
reliable-ML framing — venues that weight method + guarantee + breadth over a
leaderboard. Text edit foregrounds this. **(b) optional strengthener:** a second
real-multi-label public benchmark would materially de-risk this.

## 5. "The theorem is a loose (here vacuous) upper bound doing persuasive work it cannot support." (rank 5)
**Objection:** Theorem 1 is an *upper* bound 2B*TV; the density corollary shows
TV >= 0.86 for max-union, i.e. the bound is vacuous (> B), and "T* = argmin TV" is
uncomputable (only a density proxy that *lower*-bounds TV is measured) — so the theory
cannot actually predict which operator generalizes better, and the one-directionality
caveat is exactly the gap max-union exploits.

**Damage:** MAJOR for a theory-forward paper (a theory reviewer will press this).

**Answerable by:** (a) reframing (partly) + (c) inherent. Downgrade the theorem's
rhetorical load from "predicts max-union is worse" to "certifies a *safety guarantee*
that is non-vacuous only for the density-matching operator" — which is exactly what is
true and is now stated in the concession paragraph. The theorem's job is the guarantee,
not the ranking; the ranking is empirical (FAR frontier + stress test). Cannot be made
into a two-sided/tight bound cheaply — concede it is a certificate, not a proof of
inferiority.

## 6. "Ranking may be a proxy-scale / weak-oracle / legacy-metric artifact." (rank 6)
**Objection:** Headline operator ranking is from SmallCNN at subsampled scale with a
retracted "statistical parity" oracle claim and a legacy macro-F1 whose denominator
was model-dependent (paper's own audit note calls pre-audit F1 "provenance, not final")
— it may not survive at real scale.

**Damage:** MAJOR (rigor / reproducibility).

**Answerable by:** (b) the queued reruns + (a) text already acknowledges. The
SmallCNN -> ResNet-18 backbone ablation shows gains transfer and grow, and the
training-size scaling is monotone (not saturated), which partly answers. **(b)
EXPERIMENT FOR ORCHESTRATOR:** finish the saved-probability, supported-class-macro
five-seed common-protocol rerun of every main arm + exact Shin22 Original/Average/
Summation arms (already in the evidence queue), so no headline number rests on the
legacy macro. Until done, keep the audit note.

---

## Verdict summary

| # | Objection (short)                          | Damage | (a) text | (b) experiment | (c) concede |
|---|--------------------------------------------|--------|----------|----------------|-------------|
| 1 | Defined away the winning baseline          | fatal  | yes      | yes (=Obj2 b)  | partial     |
| 2 | Over-dense still generalizes               | major  | yes      | YES (decisive) | yes         |
| 3 | Modest absolute performance                | major  | yes      | -              | partial     |
| 4 | Niche / thin benchmark base                | major  | yes      | optional       | partial     |
| 5 | Theorem = loose/vacuous upper bound        | major  | partial  | -              | yes         |
| 6 | Proxy-scale / weak-oracle / legacy metric  | major  | partial  | YES (queued)   | -           |

## Experiments the orchestrator should plan (the (b)/(c) residue)
1. **DECISIVE — density-shift stress test** (Obj. 1 + 2): faithful vs. over-dense
   training evaluated under a shifted test density, at matched FAR operating points,
   and on real 3-/4-mixes. This is the single experiment that converts the
   distributional-faithfulness argument from "certified guarantee" into an empirical
   win-or-honest-loss.
2. **Full-scale saved-probability rerun** (Obj. 6): supported-class macro-F1, five
   seeds, every main arm + exact Shin22 arms, so no headline rests on the legacy macro.
3. **Second public real-multi-label benchmark** (Obj. 4, optional but high-value):
   removes the single-benchmark dependency for the industrial claim.

## What was added to the paper (the (a) text fixes, 2026-07-16)
- `latex/main.tex` and `DRAFT.md`, Discussion & Limitations:
  - New paragraph **"Does density faithfulness matter if over-density does not hurt
    raw F1?"** — honest concession that over-density does not empirically penalize raw
    F1 in this protocol; reframes the claim as certified-TV-guarantee + FAR-controlled
    frontier + cross-regime flip (not a raw-F1 win); flags the density-shift stress
    test as the decisive future experiment; restates max-union is kept in tables, not
    defined away. Covers objections (i) and (ii).
  - New paragraph **"Contribution type, not leaderboard SOTA"** — positions the work
    as annotation-free weak-supervision + reliability with cross-regime generality,
    concedes the single-benchmark limitation. Covers objections (iii) and (iv).
