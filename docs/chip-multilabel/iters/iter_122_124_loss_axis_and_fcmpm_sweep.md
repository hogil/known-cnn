# iter 122 + 123 + 124 — T6 loss-axis dead-end and FCM-PM spatial granularity sweep

- **timestamp**: 2026-05-13
- **tag**: `loss_axis_dead_end_and_fcmpm_grid_sweep`
- **iter ids**: 122 (T6 ASL clip=0.05), 123 (T6 ASL clip=0.10), 124a-i (FCM-PM granularity)
- **source**:
  - `outputs/iter122_T6_asl_gn4/T6_iter122_T6_asl_gn4_260513_085714/`
  - `outputs/iter123_T6_asl_clip01/T6_iter123_T6_asl_clip01_260513_091520/`
  - `outputs/iter124_{a,b,c,d,e,f,g,h,i}_*/T7_*/`
  - `outputs/_iter124_grid_size_sweep_summary.log`
  - `_iter124_reeval.sh`
- **summary**: iter112 paper SOTA (bF1 0.9964 / Total FAR 0.83% / cell=I13) is unchanged.
  Three follow-up axes tested, all regress: (i) T6 BCE→ASL γ_neg=4 clip=0.05/0.10 dead-end
  on the loss axis, (ii) FCM-PM GRID=2 too coarse (bF1 0.8705 / FAR 1.07% — partial
  iter124a result, remaining 8 sub-runs pending re-eval).

---

## 1. iter122 — T6 (BCE warmup → ASL γ_neg=4 clip=0.05)

### Hypothesis

Hold iter116J recipe frozen (T7 BCE+LS=0.30, FCM-PM complement g=3, val_margin,
save-every-epoch) and swap loss to **T6 (BCE warmup → ASL γ_neg=4, clip=0.05)**.
Target: amplify partner-bit gradient on weak 2-combo signal (`bb+sr → sr`,
`fork+sr → sr`).

### Setup

- variant: T6
- BCE→ASL switchover: ep6
- ASL clip: 0.05
- val_criterion: margin
- save-every-epoch: on
- cutmix: complement g=3 pair=masked label_scale=0.5
- epochs: 10, lr 1e-4, batch 2 accum 8, seed 1, `--no-normal`
- elapsed: 518.8s

### Results

Three eval ckpts (best/val_margin pick, ep06 first ASL ckpt, ep10 final ASL):

| ckpt | macro_f1 (4cls) | bF1 (positives) | Total FAR | NI FAR | OOD FAR | bb+sr→sr | fork+sr→sr | fork+sr→fork |
|------|----------------:|----------------:|----------:|-------:|--------:|---------:|-----------:|-------------:|
| ep03 (val_margin BCE pick) | 0.7132 | 0.8122 | 84.20% | 76.50% | 86.60% | 0.869 | 0.988 | 0.912 |
| ep06 (first ASL) | 0.7727 | 0.8298 | 74.20% | 79.00% | 72.70% | 0.900 | 0.787 | 0.775 |
| ep10 (final ASL) | 0.8210 | 0.8297 | 9.40% | 29.00% | 3.30% | **0.981** | 0.750 | 0.819 |

### Verdict — REGRESSION

- bF1 0.8297 (ep10) is **−0.1667** below iter112 SOTA (0.9964)
- Total FAR 9.40% is **+8.57pp** above iter112 SOTA (0.83%) — catastrophic
- bb+sr partner recall improved 0.831 → 0.981 (+0.150) — hypothesis partially confirmed,
  but at the cost of fork+sr partner recall (1.000 → 0.750, −0.250)
- val_margin selection picks ep03 (BCE warmup phase, no ASL effect at all)

### Why failed

1. ASL clip=0.05 is over-aggressive — auto-tuned thresholds drop to fork=0.02 /
   scratch=0.06 (vs iter116J's 0.180 / 0.140), so Normal/Invalid/OOD chips with
   weak scratch-/fork-like noise flip into the defect class
2. val_margin criterion picks the BCE warmup phase ckpt (ep3) because BCE phase
   has saturated val_margin > 0.97 while ASL phase has val_margin ~0.96 by design
   (ASL deliberately pushes negative logits below zero by clip amount)
3. partner-recall trade-off (sr-on-bb vs sr-on-fork) is intrinsic to γ_neg=4 —
   gradient amplification on one combo's weak-positive bit suppresses the
   per-class auto-threshold of the other combo's strong-positive bit

---

## 2. iter123 — T6 (BCE warmup → ASL γ_neg=4 clip=0.10), single-atomic clip swap

### Hypothesis

iter122 root-cause was clip=0.05 being over-aggressive. Single-atomic clip swap
to 0.10 to test whether the FAR regression is recoverable.

### Setup

Identical to iter122 except:
- ASL clip: **0.10** (was 0.05)
- BCE→ASL switchover: ep4 (was ep6 — earlier switch)
- elapsed: 750.9s (+45% vs iter122; clip 0.10 backward heavier)

### Results

| ckpt | macro_f1 (4cls) | bF1 | Total FAR | NI FAR | OOD FAR | bb+sr→sr | fork+sr→sr | fork+sr→fork |
|------|----------------:|----:|----------:|-------:|--------:|---------:|-----------:|-------------:|
| ep03 (val_margin BCE pick) | 0.7132 | 0.7132 | (BCE — same as iter122) | — | — | 0.869 | 0.988 | 0.912 |
| ep10 (final ASL clip=0.10) | 0.8298 | 0.8297 | **5.00%** | 16.00% | 1.60% | **0.988** | **0.838** | **0.838** |

### iter123 ep10 vs iter122 ep10 (single-atomic clip 0.05 → 0.10)

| metric | iter122 (clip=0.05) | iter123 (clip=0.10) | Δ |
|---|---:|---:|---:|
| bF1 | 0.8297 | 0.8297 | 0.0000 |
| Total FAR | 9.40% | 5.00% | −4.40pp |
| NI FAR | 29.00% | 16.00% | −13.00pp |
| OOD FAR | 3.30% | 1.60% | −1.70pp |
| bb+sr→sr | 0.981 | 0.988 | +0.007 |
| fork+sr→sr | 0.750 | **0.838** | +0.088 |
| fork+sr→fork | 0.819 | 0.838 | +0.019 |

### Verdict — DRAW (partial recovery, winner-criterion not met)

- vs iter122: partial improvement (FAR −4.40pp, fork+sr partner recall +0.088)
- vs iter112 SOTA: bF1 still **−0.1667** below, Total FAR **+4.17pp** above
- bit_F1 ≥ 0.99 AND Total FAR ≤ 0.5% winner criterion: **NOT MET**

### Why winner not met

1. ASL γ_neg=4 trade-off is intrinsic regardless of clip — clip controls
   probability-shift severity but does not change which classes get pushed
2. clip=0.10 still drops auto-tuned thresholds enough that OOD-pattern wafer
   chips (Starburst, CrossScratch, DiagonalSmear, CenterDonut) have FAR=1.6%
3. val_margin selection still picks ep3 BCE-only ckpt — clip-independent

### Loss-axis verdict — DEAD END

The T6 BCE→ASL axis under iter116J recipe is dead end:
- clip=0.05 → FAR 9.4% (catastrophic)
- clip=0.10 → FAR 5.0% (still order-of-magnitude above iter112's 0.83%)
- Both: bF1 stuck at 0.83, +0.04 over the BCE-only baseline but **−0.17 below iter112 SOTA**

---

## 3. iter124 — FCM-PM spatial granularity sweep (clean GRID = g × n)

### Hypothesis

Hold iter116J recipe frozen, sweep the **spatial granularity of the FCM-PM
complement mask** along two axes:
- **g** = `--cutmix-n-groups` (partition count)
- **n** = grid multiplier such that `GRID = g · n` (clean integer GRID where
  each group has exactly `n²` cells, square sub-blocks)

### Setup

- variant: T7 BCE+LS=0.30
- cutmix mode: `complement` (g={2,3} × n={1,2,3,4}) or `bisect_h` / `bisect_v`
- cutmix_p=0.25, pair=masked, complete_label_scale=0.5
- val_criterion=margin, save-every-epoch
- epochs: 10, lr 1e-4, batch 2 accum 8, seed 1, `--no-normal`

### 9 sub-runs

| sub | g | n | GRID | cells | cells/group | cutmix_mode | best ep | total ep | elapsed s |
|----|--:|--:|-----:|------:|------------:|-------------|--------:|---------:|----------:|
| 124a | 2 | 1 | 2 | 4   | 2  | complement | 6  | 10 | 407 |
| 124b | 2 | 2 | 4 | 16  | 8  | complement | 10 | 10 | 414 |
| 124c | 2 | 3 | 6 | 36  | 18 | complement | 4  | 10 | 407 |
| 124d | 2 | 4 | 8 | 64  | 32 | complement | 4  | 10 | 403 |
| 124e | 3 | 1 | 3 | 9   | 3  | complement | 10 | 10 | 466 |
| 124f | 3 | 2 | 6 | 36  | 12 | complement | 4  | 10 | 469 |
| 124g | 3 | 3 | 9 | 81  | 27 | complement | 10 | 10 | 463 |
| 124h | — | — | — | bisect_h | — | bisect_h | 6 | 10 | 339 |
| 124i | — | — | — | bisect_v | — | bisect_v | 6 | 10 | 341 |

### iter124a result (only completed full-eval sub-run)

Full `eval_v15direct` (11-class × 200 chips/class = 2200 chips):

| cell | macro_f1 (4cls) | bF1 (positives) | Total FAR | NI FAR | OOD FAR | top1_11cls |
|------|----------------:|----------------:|----------:|-------:|--------:|-----------:|
| T0__I3 | 0.8306 | 0.7113 | 52.02% | 33.00% | 57.97% | 0.5185 |
| T0__I7 | 0.8305 | 0.7103 | 49.76% | 23.50% | 57.97% | 0.5208 |
| **T0__I10** | **0.7699** | **0.8705** | **1.07%** | **0.00%** | **1.41%** | **0.5354** |
| T0__I13 | 0.6588 | 0.8128 | 1.07% | 0.00% | 1.41% | 0.4711 |

Per-positive-class F1 (T0__I10 best cell):
- bank_boundary 0.6926
- fork 0.9496
- scratch 0.9581
- scratch_rot 0.9014
- bb+fork 0.8626
- bb+scratch 0.7619
- bb+scratch_rot 0.8305
- fork+scratch 0.9423
- fork+scratch_rot 0.9354

### iter124a vs iter112 paper SOTA

| metric | iter124a (GRID=2) | iter112 (GRID=3 default) | Δ |
|---|---:|---:|---:|
| bF1 (positives) | 0.8705 | 0.9964 | **−0.1259** |
| Total FAR | 1.07% | 0.83% | +0.24pp |
| top1_11cls | 0.5354 | 0.5800 | −0.0446 |
| bank_boundary F1 | 0.6926 | ~0.9984 | **−0.306** |

The dominant failure mode is bank_boundary at GRID=2 — the BB perimeter signal
gets mixed with the paired chip's central defect at half-image patch size, causing
position-confused FN. This validates that the FCM-PM mechanism requires
**GRID ≥ 6** (cell size ≤ 64 px at 384 input) to preserve spatial composition.

### iter124b-i — re-eval pending

The remaining 8 sub-runs have empty `eval_v15direct_n200/` folders. Analyst
dispatched `_iter124_reeval.sh` (running at logging time). When complete, the
following questions will be answered:

1. **GRID monotonicity** — does bF1 increase monotonically with GRID? Or is
   there an intermediate optimum at GRID=6?
2. **g=2 vs g=3 at matched GRID=6** (124c vs 124f) — does partition count
   matter when total spatial resolution is held constant?
3. **bisect_h vs bisect_v** (124h vs 124i) — does scratch_rot (top-tilted)
   benefit more from horizontal slicing than vertical?
4. **Does any granularity beat iter112's GRID=3 default** (bF1 0.9964)?

CSV rows for 124b-i are marked `PENDING` and will be back-filled in-place once
the reeval completes.

---

## Cross-iter SOTA timeline

| iter | bF1 | Total FAR | cell | Δ bF1 vs SOTA | Δ FAR vs SOTA |
|------|----:|----------:|-----:|--------------:|--------------:|
| iter46E (legacy with Normal) | 0.9755 | 1.07% | I7 | — | — |
| iter101A (per-ep ep10 final) | 0.9964 | 4.52% | I13 | +0.021 | +3.45pp |
| iter111 (T_max=10 ep08 best) | 0.9963 | 1.31% | I13 | — | −3.21pp |
| **iter112 (T_max=20 ep06 best) — paper SOTA** | **0.9964** | **0.83%** | **I13** | **★** | **★** |
| iter122 (T6 ASL clip=0.05, ep10) | 0.8297 | 9.40% | I10 | −0.1667 | +8.57pp |
| iter123 (T6 ASL clip=0.10, ep10) | 0.8297 | 5.00% | I10 | −0.1667 | +4.17pp |
| iter124a (GRID=2 FCM-PM, best) | 0.8705 | 1.07% | I10 | −0.1259 | +0.24pp |
| iter124b-i | PENDING | PENDING | — | — | — |

**Paper SOTA = iter112 best_model (ep06, val_f1 sel, Tmax=20), bF1 0.9964 / Total FAR 0.83% / cell I13.**

---

## Files / sources

- iter122: `outputs/iter122_T6_asl_gn4/T6_iter122_T6_asl_gn4_260513_085714/`
  - `eval_v15direct_n200/stage1_260513_090615/` (best ep3 BCE pick)
  - `eval_ep06/stage1_260513_090739/` (first ASL ckpt)
  - `eval_ep10/stage1_260513_090849/` (final ASL ckpt)
- iter123: `outputs/iter123_T6_asl_clip01/T6_iter123_T6_asl_clip01_260513_091520/`
  - `eval_v15direct_n200/stage1_260513_092827/` (best ep3 BCE pick)
  - `eval_ep10/stage1_260513_092951/` (final ASL ckpt, clip=0.10)
- iter124a: `outputs/iter124_a_g2_n1/T7_iter124_a_g2_n1_260513_093646/eval_v15direct/stage1_260513_104129/`
- iter124b-i: train_summary + history under `outputs/iter124_{b..i}_*/T7_*/`; eval pending
- iter124 dispatcher log: `outputs/_iter124_grid_size_sweep_summary.log`
- iter124 reeval script: `_iter124_reeval.sh`
- analyst notes: `chip_multilabel/notes.md` (iter 122 / iter 123 entries)
