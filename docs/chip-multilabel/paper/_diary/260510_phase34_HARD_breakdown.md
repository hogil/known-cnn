# 260510 — Phase 31b HARD050 saturation breakdown

## Trigger

§5.26 (Phase 28 n = 500 robust eval) had finalised the
4-bag headline at 0.9953 ± 0.0002 / 0 % across pure-hard,
hard + KD, alt seed 7, iter-33 alt, iter-34 KD + asym. The
spread across composition types compressed to 0.0014 (n =
200) / 0.0031 (n = 500). §6.18 was rewritten to claim "KD
axis is interchangeable". The compression smelled like
saturation rather than a structural invariant — easy chips
all decoded perfectly, masking real composition
differences.

## Hypothesis

Filter the 9 model `preds_chip.parquet` files to
`source-strength-pct ≤ 0.50` chips only (HARD050) and
re-evaluate the same 4-bag composition types. If the
headline numbers compress on easy chips because the
defect signal is over-strong relative to the model's
threshold margin, then HARD050 should expose any genuine
ranking among bag types.

## Result

**Saturation broken, hard + KD wins.** Single-model
HARD050 (2 003 chips intersection, vs 7 080 at FULL n =
500) shows:

- 24_LS030 cells take **single-model top-2** at 0.9767 /
  0.9707 — despite being FAR-fragile-alone at FULL eval
  (best 22.5 % ni_FAR). The cell is a HARD-chip specialist.
- 4-bag composition spread expands 6.7 × (0.0031 →
  0.0208).
- Hard + KD 4-bag {24_LS030_seed42 + 26 B + 26 H + 33 D}
  reaches **0.9689 / 0 %**, beating pure-hard NEW HEADLINE
  by **+0.0019**. Per-class lift is sc +0.0024, bb
  +0.0063; fk / sr saturated.

## Interpretation

Two findings consolidate paper-grade:

1. **"FAR-fragile = HARD-chip specialist"** — the §6.17.2
   ensemble-from-fragility thesis is strengthened. The
   24_LS030 high-LS hard-label cell over-fits Normal
   boundaries (over-firing) and, by the same mechanism,
   extracts maximal signal from low-strength defect chips.
   Both behaviours are absorbed by the bag's majority vote.

2. **KD axis is scope-bounded interchangeable.** The §6.18
   claim is correct on EASY chips and wrong on HARD chips.
   Soft-target gradient signal is redundant when the
   empirical loss saturates and dominant when it does not.
   This is the cleanest distillation-value finding in the
   paper.

## Paper updates

- §5.27 — new subsection with single-model + 4-bag
  HARD050 tables, cross-eval comparison FULL n = 50 / 200
  / 500 vs HARD050.
- §6.17.3 — saturation breakdown via strength filter.
  Cross-eval composition spread table, per-class HARD050
  delta (pure-hard vs hard + KD), 24_LS030 specialist
  reading.
- §6.18.1 — KD axis interchangeable on EASY, dominant on
  HARD. Refines §6.18 textbook-counter framing on the
  axis-composition sub-question.
- §7.6.4 — limitations subsection: saturation hides
  composition differences; strength-stratified evaluation
  is necessary.
- §7 cost frontier — HARD050 addendum table; deployment
  recommendation revised to prefer hard + KD for production.
- abstract — HARD-eval refinement paragraph.
- §9 conclusion — KD-axis nuance paragraph.

## Methodological lesson

Reporting only saturating numbers (FULL v15direct n =
200 / 500 = 0.9953) is methodologically incomplete. A
strength-stratified slice (or equivalent
difficulty-conditioning) should accompany any
"interchangeable axes" claim. The paper's controlled
benchmark already provides such a slice via the
`source-strength-pct` synthesis knob; production-grade
benchmarks elsewhere should adopt analogous stratification
or risk under-stating composition-relevant differences.

## Decisions logged

- The **headline number 0.9953 / 0 %** at FULL n = 500
  remains the primary claim — easy-chip saturated benchmark.
- The **production deployment recommendation** flips from
  "any well-spread 4-bag" to "prefer hard + KD"
  {24_LS030_seed42, 26 B, 26 H, 33 D}.
- The **KD axis** is rehabilitated: not free-and-redundant,
  but EASY-redundant / HARD-dominant.

_Sources: Phase 31b HARD050 block in
`docs/chip-multilabel/iters/iter_39_purehard_4bag.md`;
strength-filtered intersection on 9 model
`preds_chip.parquet`._
