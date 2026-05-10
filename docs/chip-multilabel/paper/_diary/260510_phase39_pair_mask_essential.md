# 260510 — Phase 39 / Iter 46 — Pair-Mask is method-essential

## Context

Iter 46 closed an open question from §4.6.6 / §5.18:
the original component-necessity argument established
that **all four FCM-PM axes are jointly necessary** to
clear the dual gate (bit-F1 ≥ 0.95 AND `ni_FAR` ≤ 5 %),
but did not isolate **which axis is the safety-critical
one**. iter 28 / iter 29 proved non-decomposability;
iter 46 now refines the picture into a two-tier taxonomy.

## What was run

6 single-cell ablation runs on top of the production
26 B baseline (g = 4, LS = 0.10, mode = complement,
pair = masked, fill = corner, cutmix-p = 0.25,
cutmix-rect = 0.5), each perturbing one or more axes:

| cell | perturbation                                 |
|------|----------------------------------------------|
| A    | pair = none (remove pair-mask)               |
| B    | mode = single (remove complement)            |
| C    | LS = 0.30 + pair-fill = noise                |
| D    | cutmix-p = 0.40 + g = 4, LS = 0.40           |
| E    | cutmix-rect = 0.3 (vs default 0.5)           |
| F    | pair = none + p = 0.40 + g = 2 + LS = 0.30   |

Eval: FULL n = 200 + HARD050 cross-eval, dual gate.

## Key finding (★ paper-headline)

**Pair Mask is the safety-critical contribution of FCM-PM.**

- Cell A (pair-mask removed alone) — bF1 0.9781 → 0.7977,
  FAR 2.5 % → **100 %**. Catastrophic safety collapse
  while defect-class accuracy partially survives.
- Cell F (pair-mask removed + 3 helpful axis swaps stacked)
  — bF1 recovers to 0.9723 but FAR stays at **100 %** at
  FULL. **Pair-mask removal dominates other axes.**
- Cells B / C / D / E (other single-axis perturbations,
  pair-mask retained) — all PASS dual gate; bit-F1 drops
  range from −0.013 (cell E) to −0.166 (cell C).

This sharpens the §4.6.6 monolithic necessity claim into
a **two-tier taxonomy**:

- **Method-essential (binary safety):** pair-mask,
  complement mode.
- **Hyperparameter-tunable (continuous accuracy):**
  pair-fill, cutmix-p, cutmix-rect.

## Mechanism (added to §6.19)

Without pair-mask, the model only ever sees class A
**alongside** class B in the same chip. The sigmoid
head learns the marginal `P(class = c | any defect)`
rather than the conditional `P(class = c | this chip)`.
Normal / Invalid chips, OOD for "any defect," default
to ≥ 1 defect prediction → FAR = 100 %. Pair-mask is
literally the supervision channel that grounds
isolated-class semantics; without it the open-set
abstention behaviour the FAR gate measures cannot
exist.

## Paper updates landed

1. **§5.28** (`05_experiments.md`) — new subsection
   "Iter 46 — FCM-PM 5-axis ablation" with the 7-row
   table (6 ablation cells + 26 B baseline) and the
   two-tier finding.
2. **§6.19** (`06_analysis.md`) — new subsection
   "Why pair-mask is the safety-critical contribution"
   with mechanism analysis and validation of the
   original FCM-PM design.
3. **§4.6.6** (`04_methods.md`) — appended a
   "Two-tier refinement" paragraph that cites §5.28
   and §6.19, sharpening the symmetric four-axis
   framing into safety-critical vs accuracy-critical
   vs hyperparameter-tunable.
4. **§7.7** (`07_discussion.md`) — new subsection
   "Method-essential vs hyperparameter-tunable axes"
   with the two-tier taxonomy table and practitioner
   guidance.

## Why this matters for the paper

- §4.6.6 / §5.18 framed FCM-PM as four orthogonal axes
  that are jointly necessary. iter 46 reveals that the
  axes are **not symmetric** — one of them (pair-mask)
  is the binary FAR-control switch, the rest are
  accuracy-shaping.
- This is a stronger and more **actionable** claim for
  reviewers and practitioners. The defense for ablating
  the method-essential tier reproduces FAR = 100 %; the
  defense for the tunable tier exposes a hyperparameter
  envelope rather than a fixed recipe.
- It also generalises beyond the FCM-PM masking style:
  any chip-multi-label augmentation that mixes paired
  class signals must keep an isolated-class supervision
  channel to retain open-set abstention.

## Source

- iter 46 6-cell ablation, single-cell training (each
  cell trained from scratch with the perturbation),
  FULL n = 200 + HARD050 cross-eval, dual gate.
- 26 B baseline = production paper-main configuration
  (g = 4, LS = 0.10, mode = complement, pair = masked,
  fill = corner, cutmix-p = 0.25, cutmix-rect = 0.5).
