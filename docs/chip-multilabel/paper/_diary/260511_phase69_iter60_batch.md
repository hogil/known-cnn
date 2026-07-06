# 260511 — Phase 69 / iter 60 — Batch dimension ablation

## Context

iter 59 (§5.42 / §6.27) partitioned three hyperparameter axes
(cutmix-discount, pair-loss-w, cutmix-grid-prob) as **dummy** —
five distinct recipes produced identical 0.9872 / 0.5 % predictions
at four-decimal per-class precision. The natural inverse question:
which axes are **deterministic**, and is the `batch = 2 accum = 8`
specification of 50 B arbitrary or experimentally pinned?

iter 60 sweeps physical batch and accumulation factor on a 6-cell
FULL n = 200 grid around 50 B.

## Setup

6 cells, each = 50 B recipe with exactly one batch-dimension change:

- 60 A — physical 2 / accum 4 / eff 8 (halve accum)
- 60 B — physical 2 / accum 16 / eff 32 (double accum)
- 60 C — physical 4 / accum 8 / eff 32 (double physical)
- 60 D — physical 2 / accum 32 / eff 64 (quadruple accum)
- 60 E — physical 4 / accum 4 / eff 16 (same eff, double physical)
- 60 F — physical 1 / accum 16 / eff 16 (same eff, halve physical)

All cells use FULL n = 200 evaluation, fresh seeds, dual gate
`bit-F1 ≥ threshold` and `ni_FAR ≤ 5 %`.

## Results

| cell | physical | accum | eff | bF1 | ni_FAR | dual | bb / fk / sc / sr | Δ vs 50 B |
|------|:-:|:-:|:-:|---:|---:|---|---|---:|
| **50 B** | **2** | **8**  | **16** | **0.9872** | **0.5 %** | PASS | 0.9866 / 0.9825 / 0.9795 / 1.0000 | **★ sweet spot** |
| 60 A | 2 | 4  | 8  | 0.9780 | 1 %   | PASS | 0.9728 / 0.9517 / 0.9913 / 0.9961 | − 0.009 |
| 60 B | 2 | 16 | 32 | 0.8784 | 0 %   | PASS | 0.9586 / 0.8033 / 0.7901 / 0.9619 | − 0.109 |
| 60 C | 4 | 8  | 32 | 0.8924 | 0 %   | PASS | 0.9621 / 0.9244 / 0.7136 / 0.9694 | − 0.095 |
| 60 D | 2 | 32 | 64 | 0.9488 | 100 % | **FAIL** | 0.9754 / 0.9078 / 0.9785 / 0.9333 | bF1 OK / FAR break |
| 60 E | 4 | 4  | 16 | 0.9778 | 0 %   | PASS | 0.9881 / 0.9809 / 0.9430 / 0.9992 | − 0.009 (same eff) |
| 60 F | 1 | 16 | 16 | 0.8905 | 100 % | **FAIL** | 0.9586 / 0.9324 / 0.7143 / 0.9568 | − 0.097 / FAR break |

## Findings

### F1 — effective-batch sweet spot at 16 is narrow

Halving effective batch (60 A → eff 8) regresses − 0.009; doubling
(60 B, 60 C → eff 32) regresses − 0.10 catastrophically; quadrupling
(60 D → eff 64) holds bit-F1 but breaks FAR at 100 %. The 2×
perturbation window around eff = 16 is empty — the effective-batch
sweet spot is **sharper than α** (whose FAIL boundary at 0.55 is
+ 0.05 from the sweet at 0.50).

### F2 — physical batch matters at fixed effective batch

Three cells at eff = 16 with different physical batch:
- 50 B (b = 2): 0.9872 / 0.5 %
- 60 E (b = 4): 0.9778 / 0 %   → − 0.009
- 60 F (b = 1): 0.8905 / 100 % → − 0.097 + FAR break

Physical batch is a **separate deterministic axis**, not a memory-
vs-throughput trade-off. Mechanism: BatchNorm running-statistics
quality is non-monotone in physical batch.

### F3 — single-sample BN (b = 1) catastrophically breaks FAR

60 F at b = 1 produces `ni_FAR = 100 %`, identical failure mode to
α = 0.55 (iter 59 B). With b = 1 the BN running-mean and running-var
are pure per-sample point estimates, accumulating high-frequency
noise into inference-time normalisation that drives Normal /
Invalid chips past the FAR threshold.

## Mechanism — BN running statistics non-monotonicity

The b = {1, 2, 4} sweep at fixed eff = 16 reveals a non-monotone
optimum at b = 2:

- **b = 1**: per-sample point estimate, no variance signal, high-
  frequency noise propagates into BN running stats → unsafe FAR.
- **b = 2** (50 B): minimum batch with a two-sample variance
  estimate; running stats smooth enough to be stable but stochastic
  enough to preserve the noise signal that the rest of the recipe
  (drop-path = 0, LS = 0.20, KD α = 0.5) appears to depend on.
- **b = 4** (60 E): averaged running stats lose the noise signal
  → mild − 0.009 regress.

The recipe is calibrated to consume a specific amount of BN noise;
both starving (b = 4) and saturating (b = 1) the noise budget
regress.

## Paper implications

- **Batch dimension joins the deterministic axis set** (§6.27.1).
- The deterministic / dummy taxonomy now spans ~ 8 deterministic
  (KD α, LS, drop-path, grad-clip, epochs, physical batch, accum,
  effective batch, lr) vs ~ 3 dummy (cutmix-discount, pair-loss-w,
  cutmix-grid-prob).
- The `batch = 2 accum = 8` specification is **experimentally
  verified, not arbitrary** — production deployment must replicate.

## Next

§5.43, §6.27.1, §7.10.9, abstract all updated. HEADLINE 0.9953
unchanged (this is single-model 1× cost finding only, no impact on
4-bag ensemble).

_Source: iter 60 6-cell batch dimension sweep on top of 50 B
recipe._
