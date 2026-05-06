# Diary — 2026-05-05 (evening)

Append to the day's first entry (`260505_initial.md`). Phase A closes,
A3 reveals a dual-axis regime change.

## ~18:58 — Phase A3 dispatch

A3 plan: hold α=0.20, LR=1e-4 (the A1 + A2 winners), sweep epochs
∈ {3, 5, 12}. ep=8 already on the table from A1. Three sequential
trains, each with the I3 / I7 / I10 inference grid. Total ~14 min.

Hypothesis going in: ep=12 will improve marginally over ep=8, ep=3
will be too short. The default 8 was a guess; the question is
whether it is the right operating point or under-trained.

## ~19:21 — A3 done, results arrive

```
ep=3   | I3=0.8467  I7=0.8500  I10=0.8763  [I10 wins]
ep=5   | I3=0.8254  I7=0.8236  I10=0.8567  [I10 wins]
ep=8   | I3=0.9239  I7=0.9268  I10=0.8841  [I7 wins]   (A1)
ep=12  | I3=0.8926  I7=0.8872  I10=0.8351  [I3 wins]
```

**Phase A overall winner stays `T1_LS20__I7` at ep=8 = 0.9268.**

But the more interesting finding is what happened to the inference
variant ranking *across* epochs. The same I10 → I7 → I3 progression
that we saw across the LS axis (low LS → I10, LS=0.20 → I7, high LS
→ I3) reappears across the epochs axis with α held at 0.20:
- under-trained (ep=3, ep=5) → I10
- on-target (ep=8) → I7
- over-trained (ep=12) → I3

## Reading

Both axes (LS, epochs) control logit sharpness through different
mechanisms. LS distributes target mass to non-targets at fixed
gradient steps; epochs lets the gradient sharpen the target peak
progressively. Either way, the resulting softmax entropy distribution
moves on the same range, and the inference decoder's optimum tracks
it.

This means the regime change is not a quirk of LS=0.20; it's a
*general property* of the inference decoder family. The §6.2 paper
section was framed around "I10 stops winning at high LS" — we now
have to upgrade that to "the inference variant ranking is a
function of model logit sharpness, demonstrated on two independent
training axes".

Updated paper sections:
- 05_experiments.md §5.5 — added Phase A3 subsection with the four-row
  table + best-inference-per-epoch observation.
- 06_analysis.md §6.2.1 — new sub-subsection on dual-axis evidence,
  side-by-side LS vs epochs ranking.
- 07_discussion.md §7.4 — added "Unified hypothesis" paragraph
  predicting that *any* training intervention that moves logit
  sharpness will move the inference optimum along the same axis;
  testable on Phase B–F by re-running the {I3, I7, I10} grid at
  every checkpoint.

## Why this matters

Up to now I had a half-formed hunch that I10 would be the right
default decoder. After today's A3 it's clear that "the right
decoder" is not a single setting — it is a function of how
sharpened the model's logits are. We commit to re-running the
{I3, I7, I10} grid at every Phase B/C/D/E checkpoint going forward,
and to classifying each by its winning inference variant.

If the unified hypothesis holds across loss families (Phase B ASL,
Phase C Focal, Phase D BCE, Phase E BCE→ASL), then a single
"entropy regime" classifier on val could deterministically pick the
right inference at deployment, removing one degree of guesswork.
That would itself be a methodological contribution worth a section
in the next iteration of the paper.

## Phase A close

Total compute ~115 GPU-min for iters 1–5 (A1 + A2 + A3 included).
Final macro-F1 0.9268 vs argmax baseline 0.7302 = +0.1966.

A2 (LR sweep) confirmed LR=1e-4 as best; LR=3e-4 + LS=0.20 was
catastrophic (0.4155, gradient explosion destroying TAPT init). A3
(epochs sweep) confirmed ep=8 as best.

Phase A is done. Phase B (ASL γ sweep), Phase C (Focal γ), Phase D
(BCE pos_weight + LS), Phase E (BCE→ASL warmup + γ), Phase F
(combine best per family), Phase G (extended metrics + I10 H
re-tune per checkpoint) are queued.

Synthesis-side TODOs (sister repo): strong-defect filter
(`--source-strength-pct 50`) and grade-elevated chip generation
(`--grade-mode elevated_2/3`). User-queued, deferred to post-Phase-A.

End of day. The single biggest finding was the dual-axis regime
change; the single biggest measured gain was still the LS sweep
(+0.0905 vs default α=0.10). Sleep on it.
