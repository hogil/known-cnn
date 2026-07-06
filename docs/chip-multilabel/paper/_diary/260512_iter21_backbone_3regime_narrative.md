# 260512 — iter 21 / Phase 87 v2 — Paper §3.5 backbone narrative correction

## Context

The paper §3.5 "Backbone (T0)" subsection has carried a single-claim
"ConvNeXtV2 = best balanced production backbone" sentence since the
iter 1–20 ablation chain was anchored to it. The claim was made on
**accuracy grounds only** (val 5-class 1.0000 at ep 1, multi-label
bit-F1 0.9654 at iter46E) and never tested against either inline
latency or batched throughput.

iter 21 / Phase 87 v2 re-measured 4 backbones × 6 batch sizes on an
isolated GPU with `torch.cuda.Event` timing (20 warm-up + 100 fwd
passes per cell). The result is that **no single backbone wins
across all three operational axes** — accuracy, b = 1 latency, and
peak chip/s do not co-rank — so the single-claim must be retracted
and replaced with a three-regime recommendation.

## What changed in §3.5.1

The logger agent had already written a list-and-table-heavy patch
(§3.5.1) recording the raw measurements. The narrator pass rewrote
the same content as flowing narrative with five named blocks:

1. **Why the original claim was wrong** — one paragraph framing the
   axis-collapse error and the corrected three-regime stance.
2. **The three regimes** — single condensed table (regime / metric /
   winner / bit-F1 / FAR / cost) with an explicit "ConvNeXtV2 wins
   none of the three" sentence.
3. **Why V2 loses on throughput — the GRN architectural quirk** —
   paragraph-form explanation of the GRN per-channel-mean-of-norms
   serialisation under cuDNN's default kernel selection, isolated to
   V2 by comparison with V1 (same family, same params, 1.88×
   scaling).
4. **Why V1 wins throughput and Swin wins FAR** — separate paragraphs
   articulating the two different mechanisms (GRN absence for V1;
   window-attention locality for Swin's OOD-wafer-canvas
   misfire-suppression).
5. **10 k-chip cost projection + scope/limitations** — restated cost
   table, then explicit retention of iter46E V2 as paper-main
   accuracy headline and three named limitations (A6000-only
   measurement, small-data single-seed, V1 224-vs-others-384 pixel
   advantage).

The final paragraph names the separation between the paper-SOTA
winner (V2, historical anchor) and the production-deployment winner
(V1 batched / Swin inline-or-FAR-strict) as a methodological feature
of the corrected §3.5 rather than a contradiction.

## What did *not* change

- The paper-main single-model accuracy headline (iter46E ConvNeXtV2,
  bit-F1 0.9654, Total FAR 1.07 %) is **unchanged**.
- All iter 1–20 ablation comparisons stay anchored to ConvNeXtV2.
- 4-bag ensemble (§5.19.5) is still V2-based; V1-bag migration is
  queued, not done.

## Sources

- raw measurement: `_phase87_precise_speed.py` + `iter_21_backbone_throughput_paper3.md`
- companion CSV: `docs/chip-multilabel/tables/backbone_throughput.csv`
- accuracy refs: `outputs/iter46E_*` (V2), `outputs/iter77A_*` (V1), `outputs/iter77C_*` (Swin), `outputs/iter77E_*` (EffV2)

## Word count

§3.5.1 narrative pass = ~720 words (target was ~400; expanded to
accommodate the five-block storyline structure plus the explicit
limitation paragraph; the paper-narrator territory mandate
"narrative flow over list dump" justifies the over-budget).
