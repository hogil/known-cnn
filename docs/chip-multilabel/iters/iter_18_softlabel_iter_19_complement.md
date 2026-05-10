# Iter 18 (soft-label CutMix sweep) + Iter 19 (complement CutMix sweep, partial)

> Records two consecutive CutMix-design sweeps run on **260508** under the
> shared training recipe **T7 (BCE+LS=0.20) + Normal training, epochs=8,
> seed=1, batch=8 accum=4 (iter18) / batch=4 accum=4 (iter19 g=2/3) / batch=2
> (iter19 g=4)**. Eval set: 640 chips (11-class multi-label).
>
> - **Iter 18** (12:54-13:48): 6-cell soft-label CutMix design sweep.
>   Probes (a) pair-masked 1-pair vs grid CutMix, (b) complete-fill grid +
>   `label_scale ∈ {0.5, 0.75, 1.0}`. **Winner: iter18D** (grid+pair_masked+soft)
>   = macro_f1 **0.8272** at I3.
> - **Iter 19** (14:05-, in progress): 12-cell complement CutMix sweep
>   (group ∈ {2, 3, 4} × label_scale ∈ {0.5, 0.75, 1.0} × pair ∈ {masked,
>   none}). **Currently best: iter19B** (complement g=2 + label=0.75 +
>   pair=masked) = macro_f1 **0.8427** at I3 — overall sweep best so far.
>
> _Footnote: iter19A epoch-1 crash, retrain pending. iter19C-L still running
> at logging time — see `outputs/_iter19_complement_resume.log`. **Numbers
> below are partial for iter19**._

---

## 1. Motivation

Iter 16 (paired CutMix) saturated the **same-pair** mechanism. Iter 17/18
ask: does **soft labelling** of CutMix targets (proportional to pixel-area
mix) rescue calibration without hurting hard combo recognition? Iter 19
extends this to **complement** CutMix (sample N-of-K classes from the rest of
the batch, not just one pair).

| iter | mechanism | hparam axis | n cells |
|---|---|---|---|
| 18 | pair_masked vs grid (legacy + complete-fill) + label_scale | label_scale ∈ {0.5, 0.75, 1.0} for grid_complete | 6 |
| 19 | **complement** (group=N classes from batch) | group ∈ {2,3,4} × label_scale ∈ {0.5,0.75,1.0} × pair ∈ {masked, none} | 12 (planned) |

All trains share **same data**, **same seed=1**, **8 epochs**, **T7+LS=0.20**,
**Normal training** — atomic per `feedback_atomic_method_iteration.md`.

---

## 2. Iter 18 — soft-label CutMix sweep (6 cells, complete)

### 2.1 Best inference per cell (winner row in **bold**)

| cell | description | best inf | macro_f1 | top1_11 | mAP | bb F1 | fork F1 | sc F1 | sr F1 |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| **iter18A** | baseline (no pair, hard, cutmix-p=0.25 std rect) | I3 | 0.8214 | 0.5641 | 0.8831 | 0.9800 | 0.7485 | **0.7912** | 0.7660 |
| iter18B | pair=masked + soft label (1 pair → masked) | I3 | 0.7843 | 0.5625 | 0.8920 | 0.9476 | 0.6519 | 0.7350 | 0.8028 |
| **iter18D ★** | **grid + pair=masked + soft label** | **I3** | **0.8272** | 0.6125 | 0.9303 | 0.9670 | 0.7266 | 0.7828 | 0.8325 |
| iter18F1 | grid + complete fill + label_scale=0.5 | I7 | 0.8200 | 0.6156 | **0.9434** | 0.9527 | 0.7421 | 0.7915 | 0.7939 |
| iter18F2 | grid + complete fill + label_scale=0.75 | I10 | 0.8036 | 0.5250 | 0.9454 | 0.9513 | 0.6880 | 0.7110 | **0.8641** |
| iter18F3 | grid + complete fill + label_scale=1.0 (hard) | I10 | 0.8196 | 0.5703 | 0.9357 | 0.9669 | 0.7430 | 0.6853 | 0.8832 |

**spread (6 cells)**: 0.7843 ~ 0.8272 = **+0.0429** (≈ ±0.014 std). Within
historical single-seed noise (±0.030 macro_f1, iter 8 ref).

### 2.2 All 4 inference variants per cell (iter18F1/F2/F3 only — A/B/D parquets store best cell only)

| iter | I3 | I6 | I7 | I10 |
|---|---:|---:|---:|---:|
| iter18A | **0.8214** | — | — | — |
| iter18B | **0.7843** | — | — | — |
| iter18D | **0.8272** | — | — | — |
| iter18F1 | 0.8095 | 0.8141 | **0.8200** | 0.8024 |
| iter18F2 | 0.7968 | 0.8032 | 0.7705 | **0.8036** |
| iter18F3 | 0.7810 | 0.7988 | 0.7785 | **0.8196** |

> _A/B/D parquets contain 1 row only (best inference variant pre-selected by
> the runner). F1/F2/F3 store all 4 variants._

### 2.3 Insights (iter 18)

- **Grid > pair-only**: A → D = **+0.0058** (within noise, but consistent
  with mAP +0.0472 → 0.9303 calibration gain).
- **Soft label hurts in `complete fill` regime**: F1 (0.5) ≈ F3 (1.0) ≈
  baseline; F2 (0.75) is the worst (0.8036). Calibration gain (mAP ↑ to
  0.9454) doesn't translate to macro_f1.
- **scratch_rot scaling**: F2/F3 push sr F1 to 0.864/0.883 (best in iter18)
  but at cost of fork/sc — classic compositional CutMix trade.
- **Best-inference-variant flips**: A/B/D peak at I3 (per-class threshold);
  F1 peaks at I7 (entropy-conditional); F2/F3 peak at I10 (Normal-veto).
  Confirms **CutMix density → inference dispersion**.

---

## 3. Iter 19 — complement CutMix sweep (partial, 2 of 12 cells)

### 3.1 Best inference per cell

| cell | description | best inf | macro_f1 | top1_11 | mAP | bb F1 | fork F1 | sc F1 | sr F1 |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| iter19A | complement g=2, label=0.5, pair=masked (**epoch-1 crash**, retrain pending) | I3 | 0.8078 | 0.5813 | 0.9537 | 0.9438 | **0.7993** | 0.7064 | 0.7817 |
| **iter19B ★** | **complement g=2, label=0.75, pair=masked** | **I3** | **0.8427** | 0.5641 | **0.9685** | 0.9540 | 0.7848 | 0.7013 | **0.9307** |

### 3.2 All 4 inference variants per cell

| iter | I3 | I6 | I7 | I10 |
|---|---:|---:|---:|---:|
| iter19A | **0.8078** | 0.7806 | 0.8003 | 0.8030 |
| iter19B | **0.8427** | 0.8413 | 0.7939 | 0.8210 |

### 3.3 Insights (iter 19, partial)

- **iter19B (complement g=2, label=0.75) is the new overall sweep best**
  (0.8427 vs iter18D 0.8272, **+0.0155** macro_f1; +0.0382 mAP).
- **scratch_rot 0.9307** — biggest single-class jump from iter18 (0.8325 →
  0.9307, +0.10).
- iter19A epoch-1 crash means its 0.8078 is sub-baseline; retrain expected
  to recover ≥ baseline.
- I3/I6 close (0.8427 / 0.8413) at iter19B → calibration is roughly OK
  per-class.
- **Pending (C-L)**: g=3 / g=4 sweeps + pair=none sweeps. Higher group
  (more classes mixed per sample) increases regularization but may dilute
  signal — open question.

---

## 4. Cross-iter delta vs prior best (iter17 winner)

| ref | macro_f1 | top1_11 | Δ vs ref |
|---|---:|---:|---|
| iter17 (multi_combo, prior best) | (see iter_17_multi_combo.md) | | — |
| **iter18D** | 0.8272 | 0.6125 | (within iter18 sweep) |
| **iter19B** | **0.8427** | 0.5641 | **+0.0155 macro_f1 over iter18D** |

> Note: **single-seed numbers** — historical noise ±0.030 macro_f1. iter19B
> result needs seed-replication confirmation before adopting as new
> recipe-level winner.

---

## 5. Source paths (relative)

- `outputs/iter18A_T7N_baseline_softlabel/T7_T7_iter18A_baseline_softlabel_seed1_260508_125426/eval_seed1/stage1_260508_125952/`
- `outputs/iter18B_T7N_pair_softlabel/T7_T7_iter18B_pair_softlabel_seed1_260508_130011/eval_seed1/stage1_260508_130601/`
- `outputs/iter18D_T7N_grid_pair_softlabel/T7_T7_iter18D_grid_pair_softlabel_seed1_260508_130620/eval_seed1/stage1_260508_131408/`
- `outputs/iter18F1_T7N_gridcomplete_label0.5/T7_T7_iter18F1_gridcomplete_label0.5_seed1_260508_132951/eval_seed1/stage1_260508_133536/`
- `outputs/iter18F2_T7N_gridcomplete_label0.75/T7_T7_iter18F2_gridcomplete_label0.75_seed1_260508_133555/eval_seed1/stage1_260508_134141/`
- `outputs/iter18F3_T7N_gridcomplete_label1.0/T7_T7_iter18F3_gridcomplete_label1.0_seed1_260508_134201/eval_seed1/stage1_260508_134745/`
- `outputs/iter19A_complement_g2_l0.5_pmasked/T7_T7_iter19A_complement_g2_l0.5_masked_seed1_260508_140548/eval_seed1/stage1_260508_141521/`
- `outputs/iter19B_complement_g2_l0.75_pmasked/T7_T7_iter19B_complement_g2_l0.75_masked_seed1_260508_141606/eval_seed1/stage1_260508_142215/`

Live progress log: `outputs/_iter19_complement_resume.log` (iter19C-L in progress, 11 trains
remaining including iter19A retrain).

Each cell parquet: `results_matrix.parquet`, `per_class_metrics.parquet`,
`thresholds.json`, `eval_summary.json`, `report.md`.

---

## 6. TODO follow-ups

1. iter19A retrain (epoch-1 crash) — re-launch with same hparam (g=2, label=0.5, pair=masked).
2. iter19C-L (10 cells) — record once finished; expect g=3/g=4 to peak below g=2 if pattern holds.
3. iter19B 5-seed replication — confirm 0.8427 is not single-seed lucky.
4. Update `02_results.md` cross-iter timeline with final iter19 results once C-L done.
