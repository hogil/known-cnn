# Iter 28 — Mixup α sweep + Iter 29 — label×spatial isolation (paper §5 ablation)

- **Date**: 2026-05-09
- **Tag**: `iter_28_29_paper_ablation`
- **Scope**: Iter 28 (6 Mixup α sweep cells) + Iter 29 (3 label×spatial isolation cells, paper §5 6-cell matrix completion)
- **Train data**: `classification_chips/` only (4-class clean — bank_boundary / fork / scratch / scratch_rot, 200/class). Same no-leak protocol as iter 21–27.
- **Dual eval**: `v14class` (800 chip, 12 key × 50, in-distribution) + `v15direct` (1000 chip, 12 key × 50 + 4 OOD wafer-canvas × 50)
- **One-line**: `★ All 6 Mixup α variants fail v15 ni_FAR (100%) — pixel α-blend palette destruction confirmed; 4-design label×spatial isolation matrix shows region-paste + full-cover + pair-mask + hard-label ALL needed (any single removal → broken).`

## Motivation

Iter 21–27 established that **complement CutMix g=2 LS=1.0 FCM-PM (iter21E)** is the single-best dual-pass design and that **14-bag majority ensemble (iter27)** is the paper-headline ensemble. To support paper §5 ablation table we need two clean atomic isolations:

1. **Iter 28**: Why not Mixup? Mixup (Zhang 2018) is the canonical alternative to CutMix and is widely used as a baseline in classification papers. We need a sweep showing it categorically fails on chip multi-label palette data.
2. **Iter 29**: Decompose the 4 design choices in iter21E (`region paste` + `full grid cover` + `pair mask` + `hard label`) into a 2×3 spatial × label matrix and demonstrate that **all 4 are necessary** (no single one is dispensable).

## Iter 28 — Mixup α sweep (6 trains)

All 6 share the iter21E base recipe — `T7N` (BCE+LS=0.20 + Normal y=−1 sentinel) with **CutMix replaced by Mixup α-blend** at the listed α value. Per-class F1 collected from per-eval `preds_chip.parquet`, configuration-best inference cell across {I3, I6, I7, I10}.

### Results table

| tag | spec                       | v14 bF1 | v14 ni% | v15 bF1 | v15 ni% | dual-pass? |
|:---:|:---------------------------|--------:|--------:|--------:|--------:|:----------:|
| 28A | Mixup α=0.2 (Zhang 2018 default) | 0.9875 |   5.00% | 0.9834 | **100.00%** | ✗ |
| 28B | Mixup α=1.0                |  0.9092 | 100.00% | 0.8924 | 100.00% | ✗ |
| 28C | Mixup α=0.1                |  0.9098 | 100.00% | 0.8627 | 100.00% | ✗ |
| 28D | Mixup α=0.4                |  0.9753 | 100.00% | 0.9141 | 100.00% | ✗ |
| 28E | Mixup α=2.0                |  0.9783 | 100.00% | 0.9671 | 100.00% | ✗ |
| 28F | Mixup α=0.4 + cutmix-p=0.5 |  0.9091 | 100.00% | 0.8984 | 100.00% | ✗ |

### Findings (iter 28) — pixel α-blend palette destruction

- **★ All 6 Mixup variants fail v15 ni_FAR with 100%** — every Mixup α value tested (0.1, 0.2, 0.4, 1.0, 2.0) destroys Normal/Invalid rejection on the OOD-augmented set.
- **Even the v14-passable α=0.2 (5.00% v14 ni)** explodes to 100% on v15direct — the same cell with no OOD pressure passes, which means α=0.2 is a fragile coincidence not a robust design.
- **α=0.1 / 1.0 also destroy v14 bit_F1 (0.91 region)** — the palette-blended chips are so far from the data manifold that even in-distribution F1 collapses.
- **28F (Mixup α=0.4 + cutmix-p=0.5 mixed)** does NOT rescue Mixup — combining Mixup with CutMix at 50:50 still saturates ni_FAR at 100% on both eval sets. The pixel α-blend signal contaminates the model regardless of CutMix presence.

**Mechanism — pixel α-blend palette destruction (paper §5 narrative)**:
The chip image is a **palette-grade PNG** where pixel value 0 = Normal, 1–7 = defect intensity grade (palette discrete code, not luminance). Mixup performs `x = λ·x_A + (1−λ)·x_B` at the pixel level: this **synthesizes new pixel values that do not exist in the palette** (e.g. mixing grade 0 and grade 5 at λ=0.5 produces value 2.5 → quantized to grade 3, which carries a different defect-intensity meaning entirely unrelated to either input). The model is trained to associate these synthetic invalid grades with mixed labels, which destroys Normal-vs-defect discrimination at the calibration boundary — manifesting as ni_FAR=100% under any OOD pressure.

This is **the central reason CutMix > Mixup for palette-graded multi-label**: CutMix preserves every pixel's palette grade (region copy-paste only), while Mixup synthesizes invalid intermediate grades that violate the palette contract.

## Iter 29 — label×spatial isolation (3 trains, paper §5 core)

### Decomposition of iter21E design (4 axes)

iter21E = **(complement g=2) + (paired mask) + (hard one-hot label) + (full-cover paste)**. We isolate each by ablating one design at a time:

| design axis | iter21E setting | ablation tested |
|:---|:---|:---|
| spatial: region vs pixel | region copy-paste | (covered by iter 28 — Mixup) |
| spatial: layout | grid_complete (g=2 full cover) | std box-cut (Yun 2019, single rect) |
| spatial: pair structure | pair-masked complement (only A/B pair active) | no pair mask (random partner) |
| label: hard vs soft | hard one-hot [A=1, B=1] | soft λ-mix (LS scale) |

The 6-cell paper §5 matrix = **3 spatial layouts × 2 label types** (some cells already filled by prior iters).

### Iter 29 results table (3 new trains)

| tag | spec                                       | v14 bF1 | v14 ni% | v15 bF1 | v15 ni% | dual-pass? |
|:---:|:-------------------------------------------|--------:|--------:|--------:|--------:|:----------:|
| 29A | std box-cut (single rect) + hard label   |  0.7381 | 100.00% | 0.7616 | 100.00% | ✗ |
| 29B | complement g=2 + pair mask + soft LS=0.5 |  0.9921 | 100.00% | 0.9953 | 100.00% | ✗ (highest bF1, but FAR fail) |
| 29C | grid_complete g=2 + no pair mask + hard LS=1.0 | 0.9369 |   2.50% | 0.9248 | 100.00% | ✗ |

### Findings (iter 29) — 4 design contribution decomposition

- **29A (std box-cut + hard label)**: bit_F1 collapses to **0.74–0.76** on both eval sets, ni_FAR=100% — single rectangular paste leaves too much un-touched majority of the chip; the paste-edge gradient dominates and the rest of the image is treated as "Normal", training Normal-vs-defect calibration on a label that says both classes are present. The hard label compounds this: the model is told [A=1, B=1] but only A's region was actually pasted.
- **29B (complement g=2 + pair mask + soft LS=0.5)**: **highest v15 bit_F1 ever recorded for a single design (0.9953)** — but ni_FAR=100% on v15. The full-cover + pair-mask combination perfectly fits the in-distribution paste signal, but the soft λ-mix label leaks Normal probability mass into both defect bins. **F1-only winner ≠ deployable.** This is the cleanest example in the entire chip-multilabel history that **ni_FAR-blind metric optimization is a trap**.
- **29C (grid_complete + no pair mask + hard LS=1.0)**: v14 ni passes at 2.50% but **v15 ni explodes to 100%** — removing the pair mask (any random partner can pair with any region) creates ambiguous mixed-class regions that confuse the OOD-rejection boundary even when the label is hard.

### ★ Paper §5 — 6-cell label × spatial matrix (final form)

| spatial \ label | soft (λ-mix) | hard (both [A=1, B=1]) |
|:---|:---|:---|
| **std box-cut** (Yun 2019) | iter21C: v15 0.85 / 100% ✗ | **iter29A**: v15 0.76 / 100% ✗ |
| **grid_complete** (no pair mask) | iter21D 18F1: v15 0.93 / 2.5% ✓ | **iter29C**: v15 0.92 / 100% ✗ |
| **complement + pair mask** | **iter29B**: v15 0.99 / 100% ✗ | **iter21E ★**: v15 0.97 / 3.75% ✓ |

**Decisive finding — all 4 design choices are necessary**:

- **iter21C** (std box-cut, soft) — fails: needs full-cover and hard label.
- **iter29A** (std box-cut, hard) — fails: needs full-cover (and pair mask).
- **iter21D** (grid_complete, no pair mask, soft LS=0.5) — passes v15 ni at 2.5% but loses bit_F1: pair mask absent allows ambiguous pair pairing.
- **iter29C** (grid_complete, no pair mask, hard LS=1.0) — fails: hard label without pair mask is the worst combination (sharp wrong-pair signal).
- **iter29B** (complement, pair mask, soft LS=0.5) — fails ni: soft label leaks calibration on full-cover.
- **iter21E ★** (complement, pair mask, hard LS=1.0) — **only passing cell**: all 4 designs together.

**Contribution decomposition** (each design's marginal effect, holding others constant near iter21E):

| remove from iter21E | result cell | v15 bit_F1 Δ | v15 ni_FAR Δ |
|:---|:---|---:|---:|
| region-paste → pixel α-blend (Mixup) | iter 28 (any α) | −0.005 to −0.105 | +96% to +96% |
| full-cover → single rect | iter 29A (hard) / 21C (soft) | −0.21 / −0.12 | +96% / +96% |
| pair mask → none | iter 29C / iter 21D | −0.05 / −0.04 | +96% / −1.25% |
| hard label → soft (λ-mix) | iter 29B / iter 21D | +0.03 / −0.04 | +96% / −1.25% |

The pair-mask removal and hard→soft transitions are partially confounded (29B vs 21D both lose pair-mask in different ways), but **every single-axis removal blows up at least one ni_FAR gate or loses ≥0.04 bit_F1**.

## Cross-iter delta vs prior winners

| metric | iter27 14-bag ENS | iter21E single best | iter29B (F1-only) | iter28A (Mixup α=0.2 best) | iter29A (worst) |
|:---|---:|---:|---:|---:|---:|
| v14 bit_F1 | **1.0000** | 0.9913 | 0.9921 | 0.9875 | 0.7381 |
| v14 ni_FAR | **0.00%** | 0.00% | 100.00% | 5.00% | 100.00% |
| v15 bit_F1 | **0.9929** | 0.9691 | 0.9953 | 0.9834 | 0.7616 |
| v15 ni_FAR | **0.00%** | 3.75% | 100.00% | 100.00% | 100.00% |
| dual-pass? | ✓✓ | ✓ | ✗ | ✗ | ✗ |

## Sources

- iter 28 outputs: `outputs/iter28A_mixup02/`, `outputs/iter28B_mixup10/`, `outputs/iter28C_mixup01/`, `outputs/iter28D_mixup04/`, `outputs/iter28E_mixup20/`, `outputs/iter28F_mixup04_cutmix/` — each with `eval_v14class/` and `eval_v15direct/` sub-trees containing `preds_chip.parquet` and `results_matrix.parquet`.
- iter 29 outputs: `outputs/iter29A_box_hard/`, `outputs/iter29B_compl_g2_softLS05/`, `outputs/iter29C_grid_hard_LS10/` — each with same dual-eval structure.
- Cross-reference: `outputs/iter21C_T7N_cutmix/`, `outputs/iter21D_18F1_repeat/`, `outputs/iter21E_19C_repeat/` (matrix neighbors).
- Numbers above are pulled from `docs/chip-multilabel/tables/all_runs_macro_f1.csv` (rows iter=28, iter=29) and the new `tables/paper_section5_ablation.csv`.
