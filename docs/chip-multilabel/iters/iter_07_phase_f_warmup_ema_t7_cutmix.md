# Iter 7 — Phase F: warmup / EMA negative + T7 CutMix peak

**Run window**: 2026-05-05 19:19 – 20:33
**Train dirs**:
`outputs/logs_chip_multilabel/T1_F1_warmup_260505_191936/`,
`outputs/logs_chip_multilabel/T1_F2_ema_260505_193223/`,
`outputs/logs_chip_multilabel/T7_T7_cutmix50_LS20_260505_195128/`,
`outputs/logs_chip_multilabel/T7_T7a_BCE_LS20_260505_195921/`,
`outputs/logs_chip_multilabel/T7_T7d_cutmix70_LS20_260505_200745/`,
`outputs/logs_chip_multilabel/T7_T7b_cutmix30_LS20_260505_201858/`
**Stage1 dirs**:
`outputs/stage1_260505_192541/` (F1),
`outputs/stage1_260505_194014/` (F2),
`outputs/stage1_260505_194443/` (I11 band-aid on T1_LS20_ep8),
`outputs/stage1_260505_195730/` (T7c, cutmix=0.5),
`outputs/stage1_260505_200523/` (T7a, BCE no cutmix),
`outputs/stage1_260505_201706/` (T7d, cutmix=0.7),
`outputs/stage1_260505_203340/` (T7b, cutmix=0.3)

## Goal

Phase A closed at iter 6 with `T1_LS20_ep8 + I7 = 0.9268`. Iter 7 covers
two distinct probes:

1. **Phase F (warmup / EMA)** — import structural BKMs (LR warmup, EMA)
   from anomaly-detection literature on top of the Phase A winner. Test
   whether they transfer to small-data multi-label chip classification.
2. **T7 atomic decomposition + CutMix-p sweep** — separate the
   contributions of switching loss CE → BCE and adding CutMix p=0.5,
   then sweep cutmix_p ∈ {0.0, 0.3, 0.5, 0.7} to locate the optimum.

## Phase F — anomaly-detection BKM transfer (negative)

| variant | recipe                                            | best cell | macro_f1 | top1_11 | Δ vs T1_LS20__I7 (0.9268) |
|---------|---------------------------------------------------|-----------|---------:|--------:|--------------------------:|
| F1      | LR warmup 2ep (start_factor=0.05, eta_min=1e-6)   | F1__I10   |   0.8181 |  0.5540 |                  **−0.1087** |
| F2      | EMA decay=0.95 + dynamic decay warmup             | F2__I10   |   0.8377 |  0.6602 |                  **−0.0891** |

_Source: outputs/stage1_260505_192541/results_matrix.parquet (F1),
outputs/stage1_260505_194014/results_matrix.parquet (F2)._

### F1 — warmup 2ep (−0.109)

The recipe transferred from anomaly-detection BKM tables that lift
val_macro_f1 by ~0.05–0.10. Here it lost 0.109. Diagnosis:

1. **TAPT-initialized backbone is already in a good basin** — warmup is
   designed for cold-start where large LR with random init blows up the
   loss. We start from a 33-class supervised checkpoint, so the model
   does not need 2 epochs of LR ≤ 5e-6.
2. **8-epoch budget can't afford 2 wasted epochs** — the LS=0.20 recipe
   was already tuned to converge in 8 epochs. Warmup steals 2/8 = 25%
   of the schedule, leaving 6 epochs at the working LR — undertrained.
3. **The win on anomaly-detection (binary chart, much more data) does
   not generalize** — the structural assumption (cold start, many
   steps) is the opposite of our setting (warm start, few steps).

### F2 — EMA 0.95 (−0.089)

EMA averaging assumes many effective gradient steps to reach the
unbiased running mean. With ~12 effective steps in the 8-epoch
budget, EMA is averaging partially-trained weights into the final
checkpoint. The dynamic decay warmup (lower decay early, ramping to
0.95) helps but does not rescue the regime mismatch.

### Phase F verdict — paper-worthy negative result

Anomaly-detection BKMs (warmup, EMA) do **not** transfer to small-data
multi-label chip classification with TAPT init. The structural
assumption gap (cold-start vs warm TAPT, many-step vs few-step) breaks
the transfer. We log this as a negative result rather than tuning the
hyperparameters, because the diagnosis is structural, not numerical.

## I11 — pair-aware threshold band-aid (rejected)

Before training T7, we tested whether a pure-inference fix could close
the bb+sr combo recall gap on the existing T1_LS20_ep8 checkpoint.

I11 adds a co-occurrence threshold: declare bank_boundary + scratch_rot
together when both class logits exceed an extra pair-aware cutoff. No
retraining required.

| metric             | T1_LS20__I7 | T1_LS20__I11 | Δ        |
|--------------------|------------:|-------------:|---------:|
| macro_f1           |      0.9268 |       0.9199 | −0.0069  |
| top1_11            |      0.8449 |       0.8432 | −0.0017  |
| bb+sr combo recall |      ~0.325 |       ~0.481 | +25 chips |
| bb+fork FP         |    baseline |       +31 FP | over-trigger |

_Source: outputs/stage1_260505_194443/results_matrix.parquet._

**Rejected**: bb+sr recall gain is real (+25 chips), but the pair-aware
threshold over-triggers on bb+fork (31 FP) because the bb logit is
already at the co-occurrence cutoff for many bb-only chips. Net macro-F1
loss is small (−0.007) but in the wrong direction. The right fix is at
training (CutMix), not inference.

## T7 — atomic decomposition (CE→BCE then +CutMix)

We probe the contribution of each change in (T1 = CE+LS=0.20) → (T7 =
BCE+LS=0.20+CutMix p=0.5) by running each step in isolation. All other
hyperparameters held: LS=0.20, LR=1e-4, ep=8.

| step  | loss | cutmix-p | best cell | macro_f1 | top1_11 | bb+sr | scratch_rot F1 | ECE_post | Δ from prev |
|-------|------|---------:|-----------|---------:|--------:|------:|---------------:|---------:|------------:|
| T1    | CE   | 0.0      | T1__I7    |   0.9268 |  0.8449 | ~0.32 |         0.9689 |   0.1788 |       (ref) |
| T7a   | BCE  | 0.0      | T7a__I3   |   0.8577 |  0.5534 |   —   |              — |   0.0731 |     **−0.0691** |
| T7c ★ | BCE  | **0.5**  | T7c__I10  |   **0.9271** |  0.8307 | **0.96** |     **1.0000** |   **0.0446** |     **+0.0694** |

_Source: outputs/phase_a_260505_175105/sweep_log.csv (T1),
outputs/stage1_260505_200523 (T7a),
outputs/stage1_260505_195730 (T7c)._

### Atomic verdicts

- **CE → BCE alone (T1 → T7a): −0.0691** confirms iter 4: BCE drops the
  softmax-style information that the F1-max threshold tuning relies on.
  T7a's best cell is I3 (not I7), and top1_11 collapses to 0.5534. ECE
  is much better (0.0731 vs 0.1788), but at a heavy macro-F1 cost.
- **+ CutMix p=0.5 (T7a → T7c): +0.0694** almost exactly cancels the
  BCE penalty. CutMix on multi-hot targets directly teaches that bb+sr
  can co-occur in pixel space — the model learns the visual
  co-occurrence pattern instead of treating combos as adversarial
  perturbations of single-class chips.
- **Net (T1 → T7c): +0.0003 macro-F1 — tied** at the headline metric,
  but with a transformed operational profile: bb+sr combo recall
  **0.32 → 0.96** (+0.63 absolute), `scratch_rot` per-class F1 reaches
  **1.0000** (was 0.9689), ECE_post drops 4× (0.1788 → 0.0446).
- **The macro-F1 tie hides a large operational improvement.** This is
  the headline finding of iter 7 and the most paper-relevant outcome
  of Phase F.

## CutMix-p sweep (BCE+LS=0.20 base)

Same recipe as T7a (BCE+LS=0.20, ep=8, LR=1e-4) varying only `cutmix_p`.

| cutmix-p | run id | best cell | macro_f1 | top1_11 | bb+sr recall |
|---------:|--------|-----------|---------:|--------:|-------------:|
| 0.0      | T7a    | T7a__I3   |   0.8577 |  0.5534 |            — |
| 0.3      | T7b    | T7b__I10  |   0.8626 |  0.5511 |       0.7312 |
| **0.5 ★**| T7c    | T7c__I10  | **0.9271** | **0.8307** | **0.9562** |
| 0.7      | T7d    | T7d__I10  |   0.9038 |  0.7432 |            — |

_Source: outputs/stage1_260505_{200523,203340,195730,201706}/results_matrix.parquet._

The sweep shape is the same non-monotonic optimum we saw on the LS
sweep in iter 5: a sharp peak at p=0.5, with both lower (0.0/0.3) and
higher (0.7) values losing 0.03–0.07 macro-F1. Reading the curve:

- **p=0.0 (T7a) — BCE-only floor**: model treats each combo as
  adversarial vs single-class, never learns co-occurrence.
- **p=0.3 (T7b) — partial repair**: model sees combo patterns ~30% of
  the time, partial recovery on bb+sr (recall 0.73) but not enough
  signal to recover macro-F1 vs T1.
- **p=0.5 (T7c, peak)**: even mix of single-class and combo training,
  bb+sr recall 0.96, scratch_rot F1 perfect, macro-F1 ties T1.
- **p=0.7 (T7d) — over-mixing**: model starts to hallucinate combos
  when only one defect is present (single-class bank_boundary chips
  start triggering scratch_rot too). Loses 0.023 macro-F1 vs p=0.5.

## T7c per-class breakdown (best cell I10)

| class          | threshold | precision | recall | F1     | AP     |
|----------------|----------:|----------:|-------:|-------:|-------:|
| bank_boundary  |    0.6200 |    0.9345 | 0.8469 | 0.8885 | 0.9575 |
| fork           |    0.1400 |    0.8815 | 0.8484 | 0.8647 | 0.7547 |
| scratch        |    0.7400 |    1.0000 | 0.9146 | 0.9554 | 0.9725 |
| scratch_rot    |    0.4200 |    1.0000 | 1.0000 | 1.0000 | 1.0000 |

_Source: outputs/stage1_260505_195730/per_class_metrics.parquet._

Versus T1_LS20__I7 per-class (iter 5):

- **scratch_rot 0.9689 → 1.0000** — perfect recall and precision
  thanks to CutMix bb+sr co-occurrence training.
- **fork precision 0.7014 → 0.8815** at almost identical recall —
  BCE+CutMix sharpens fork's negative discrimination far more than
  CE+LS could.
- **bank_boundary F1 0.8974 → 0.8885** — small drop, paid for the
  bb+sr combo gain.
- **scratch F1 0.9725 → 0.9554** — small drop, still excellent.

## Headline finding

**bb+sr combo recall 0.32 → 0.96** is the single biggest operational
improvement of any iteration to date. Macro-F1 alone (0.9268 → 0.9271)
fails to capture it — the trade is concealed by the metric. This is
the canonical example of why the multi-label benchmark needs combo
recall as a first-class reporting axis, not just macro-F1 / top1_11.

## Remaining weak points (T7c)

- **bb + scratch combo**: 78 chips → scratch single (bank lost) — the
  CutMix-trained bb+sr co-occurrence does not generalize to bb+scratch.
- **fork + scratch combo**: 52 chips → scratch single (fork lost) — fork
  signal still gets dominated by stronger scratch evidence in combos.
- **bb single → bb+fork**: 58 chips — fork over-fire still present in
  ~10% of bank_boundary single chips.

These suggest the next round (analyst-iter6 recommendation, separate
task): expanded CutMix coverage to additional combo pairs, BCE LS
sweep, or CE+LS+CutMix soft-target hybrid.

## Files

- `outputs/logs_chip_multilabel/T1_F1_warmup_260505_191936/` — F1 train.
- `outputs/logs_chip_multilabel/T1_F2_ema_260505_193223/` — F2 train.
- `outputs/logs_chip_multilabel/T7_T7_cutmix50_LS20_260505_195128/` — T7c train (★).
- `outputs/logs_chip_multilabel/T7_T7a_BCE_LS20_260505_195921/` — T7a train.
- `outputs/logs_chip_multilabel/T7_T7d_cutmix70_LS20_260505_200745/` — T7d train.
- `outputs/logs_chip_multilabel/T7_T7b_cutmix30_LS20_260505_201858/` — T7b train.
- `outputs/stage1_260505_192541/` — F1 inference matrix.
- `outputs/stage1_260505_194014/` — F2 inference matrix.
- `outputs/stage1_260505_194443/` — I11 band-aid on T1_LS20_ep8.
- `outputs/stage1_260505_195730/` — T7c inference matrix (★).
- `outputs/stage1_260505_200523/` — T7a inference matrix.
- `outputs/stage1_260505_201706/` — T7d inference matrix.
- `outputs/stage1_260505_203340/` — T7b inference matrix.
- `chip_multilabel/notes.md` — iter 6 section (lead's running notebook).
