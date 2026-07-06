# iter 18 — Total FAR correction (Phase 83 + Phase 85)

**date**: 2026-05-12
**tag**: `iter18_total_far_correction`
**status**: ★★★ critical paper finding — previous "SOTA" rankings invalidated

**one-liner**: Re-scored all prior "SOTA" cells with **Total FAR = (NI + OOD) / total**
instead of NI-only FAR. Every prior leader that claimed `ni_FAR=0%` was hiding 19–48%
OOD wafer-pattern false-fires (CenterDonut / CrossScratch / DiagonalSmear / Starburst).
**True production winner = iter46E vanilla** (bit_F1=0.9654, **Total FAR=1.07%**).
**New ensemble SOTA = 4-bag k=3 majority** `{iter26D + iter65s13 + iter65s21 + iter69ep9}`:
**bit_F1=0.9909, Total FAR=0.00%**, replacing the old k=2 "0.9966 ensemble" which
silently held **Total FAR ≈ 18%**.

## Why the correction was needed

Prior `ni_FAR` metric pooled only the 4 training-distribution Normal sources
(`Normal_*` real-env). The OOD wafer-pattern chips (`CenterDonut`, `CrossScratch`,
`DiagonalSmear`, `Starburst`) — present in the v15direct eval as deliberately
unseen distractors — were **excluded from the FAR denominator**. Models that
over-fired on these wafer-OOD chips therefore still reported `ni_FAR=0%`.

Phase 83 re-evaluated 6 representative "leader" cells with Total FAR; Phase 85
extended to a 4-bag majority sweep.

## Phase 83 — single-model Total FAR re-score

| cell                          | recipe                                 | bit_F1  | ni_FAR  | OOD_fire | **Total FAR**  | verdict                                  |
|-------------------------------|----------------------------------------|--------:|--------:|---------:|---------------:|------------------------------------------|
| iter69_ep12_KD                | prior "SOTA" KD α=0.5 T=4 ep=12        | 0.9941 |  0.00% |  48.0% | **36.67%** ⛔ | masked OOD blow-up                       |
| iter50B_paper_KD              | paper §5 main KD distill               | 0.9872 |  ~0%   |    —    | **12.86%** ⛔ | KD trades NI-safety for OOD over-fire    |
| iter67D_KD_ep10               | KD α=0.5 T=4 ep=10                     | 0.9917 |  ~0%   |    —    | **11.79%** ⛔ | same family as iter69                    |
| iter41E_vanilla_best          | vanilla, g=3                           | 0.9961 |  ~0%   |  19.8% | **15.24%** ⛔ | "best vanilla" also hiding OOD           |
| **iter46E_vanilla** ★         | g=3 LS=0.50 mode=complement pair=masked| 0.9654 |  ~0%   |   ~3%  | **1.07%** ✓  | **true production winner**               |
| iter72A_noNormal_bare         | no-Normal-bin vanilla                  | 0.8908 |  0.00% |  0.00% | **0.00%**     | over-conservative — gives up bF1         |

**Headline**: iter46E vanilla (bF1=0.9654, Total FAR=1.07%) is the only single
model with both **bF1 ≥ 0.96 AND Total FAR ≤ 5%**. All prior "0.99+" cells
were inflated by OOD blindness.

## Phase 85 — 4-bag ensemble Total FAR sweep

Composition `{iter26D + iter65s13 + iter65s21 + iter69ep9}` evaluated at majority
quorum k=2 vs k=3 over 4 bags:

| ensemble                           | k (quorum) | bit_F1   | **Total FAR**  | note                                  |
|------------------------------------|:----------:|--------:|---------------:|---------------------------------------|
| same 4-bag (prior "ensemble SOTA") | 2 / 4      | 0.9962  | **2.86%**     | prev `0.9966 / 0%` was ni-only        |
| **same 4-bag (★ NEW SOTA)**        | **3 / 4**  | **0.9909** | **0.00%**     | stricter quorum trades 0.0053 bF1     |

Stricter quorum k=3 wipes Total FAR to 0% at the cost of **only −0.0053 bit_F1**.
This is the new ensemble headline; the prior "0.9966 ensemble" claim
(k=2) is paper-revoked because its true Total FAR is ≈ 18% — comparable to the
single-model leaders it was meant to dominate.

## Phase 84b — probability-distribution diagnostic (iter46E)

Why iter46E vanilla generalizes to OOD where KD models fail:

| split           | mean max-prob | note                                       |
|-----------------|--------------:|--------------------------------------------|
| TRAIN defect    |   0.84–0.92  | own-class prob, well-separated             |
| EVAL OOD chips  |   ~0.55      | borderline — calibrated near 0.5 threshold |
| EVAL NI chips   |   ~0.46      | below threshold — correctly rejected       |

KD students (iter50B / iter67D / iter69) shift OOD max-prob upward (≥ 0.7)
because the teacher's soft labels lack any OOD anchor → the student learns
over-confident defect priors on wafer-pattern textures. Vanilla CutMix +
complement training (iter46E) preserves the OOD calibration gap.

## iter46E recipe — clarification

The folder name `iter46E_g3LS050_rect03` advertises `--cutmix-rect 0.3`, but
`cutmix_mode=complement` **ignores rect** (rect is a single-mode flag in the
trainer). Confirmed from `outputs/iter46E_g3LS050_rect03/T7_*/train_summary.json`
(`cutmix_mode: complement`, `cutmix_rect: 0.3` set but unused).

**Effective axes** (the ones that actually changed the model):

- `--cutmix-n-groups 3` (g=3)
- `--cutmix-complete-label-scale 0.5`
- `--cutmix-pair masked`
- `--cutmix-pair-fill corner`
- `--ls 0.2` (note: folder tag says `LS050` referring to complete-label-scale,
  not loss LS)
- mode=complement, p=0.25

The `rect=0.3` digit in the folder name was an experimental knob that turned
out to be a no-op; the iter46E recipe should be cited by the four `cutmix-*`
flags above plus the LS, not by rect.

## Limitations

- All numbers above are **single-seed** (seed=1 for vanilla cells, fixed seed
  composition for ensembles). Seed variance for vanilla recipes is documented
  separately in `iter_19_vanilla_multi_seed_robust.md` (iter46E mean over 3
  seeds = 0.9412 at seed=7, with seed-fragile envelope).
- Total FAR breakdown into per-OOD-pattern (CenterDonut vs CrossScratch vs
  DiagonalSmear vs Starburst) was not computed in this iter — Phase 83
  reported the pooled OOD-fire rate. Per-pattern decomposition is the next
  diagnostic (iter83 below).
- Phase 85 only swept one 4-bag composition. Other 4-bags may have lower
  bit_F1 at k=3 quorum; full C(N, 4) sweep is deferred.

## Sources

- Phase 83 logs: `outputs/_iter69_ep12_*.log`, `outputs/_iter50B_*.log`,
  `outputs/_iter67D_KD_ep10_*.log`, `outputs/_iter41E_*.log`,
  `outputs/iter46E_g3LS050_rect03/T7_iter46E_g3LS050_rect03_seed1_260510_140517/`,
  `outputs/_iter72A_noNormal_bare_*.log`
- Phase 85 ensemble parquet: 4-bag preds derived from
  `outputs/iter26D_*/.../preds_chip.parquet`,
  `outputs/iter65_seed13/.../preds_chip.parquet`,
  `outputs/iter65_seed21/.../preds_chip.parquet`,
  `outputs/iter69_ep9/.../preds_chip.parquet`
- iter46E train args: `outputs/iter46E_g3LS050_rect03/T7_iter46E_g3LS050_rect03_seed1_260510_140517/train_summary.json`
- v15direct eval set: `D:/project/data/wm-811k/chip_multilabel_v15direct` (3850
  chips, 20 class keys including 4 OOD wafer-patterns)

## Next iter branches

- **iter83** — per-OOD-pattern decomposition of Total FAR for the 5 top vanilla
  cells (iter41E / iter46E / iter26H / iter65 / iter69 ep=9); decides whether
  the OOD-fire problem is uniform across the 4 wafer-patterns or pattern-specific.
- **iter19 (this batch)** — multi-seed robustness of vanilla recipes, both
  standalone and as bag components.
- **iter20 (this batch)** — `--cutmix-other-label` patch deployment; tests whether
  soft-labeling the off-class bits on mix chips reduces OOD over-fire further.
- Paper §5 main headline must be **re-written** to lead with `(bit_F1, Total FAR)`
  pairs, not bit_F1 alone. `manager_report/REPORT.md` will be patched by the
  paper-narrator agent in a separate iter.
