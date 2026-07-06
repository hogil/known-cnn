# iter 20 — `--cutmix-other-label` trainer patch (260512) + iter83 design

**date**: 2026-05-12
**tag**: `iter20_other_label_patch`
**status**: trainer patch landed; iter83 sweep designed but not yet executed

**one-liner**: Trainer gains a new flag `--cutmix-other-label` (default 0.0)
that label-smooths the **off-class bits** of CutMix-generated mix chips.
Motivated by Phase 84b's probability-distribution diagnostic on iter46E:
TRAIN defect own-class prob = 0.84–0.92, but EVAL OOD max-prob = ~0.55 (near
threshold) and EVAL NI max-prob = ~0.46. The hypothesis is that hard
zero-labeling of off-class bits on mix chips teaches the model an
over-confident "absent class = 0" prior that fails to generalize to OOD
chips with mid-range max-prob. Soft off-class labels (e.g., 0.1) should
preserve calibration uncertainty.

## Patch detail

File: `chip_multilabel/_train_chip_variant.py` (260512)

CLI addition (line 410):

```python
ap.add_argument("--cutmix-other-label", type=float, default=0.0,
                help="Label value for off-class bits in mix chip (260512). "
                     "Default 0 (hard zero). Non-zero = label smoothing for "
                     "'neutral' bits — chip mix 의 A/B class 가 아닌 2 bits 에 "
                     "적용. e.g., 0.1 → 'soft uncertain' for off-class.")
```

Implementation (line 838–841):

```python
# 260512: other-label LS support. Initialize with cutmix_other_label
# for all bits, then overwrite A/B class bits below.
other_lbl = float(args.cutmix_other_label)
mix_t = torch.full((len(TRAIN_CLASSES),), other_lbl, device=device)
```

The change inverts the previous mix-target initialization: was
`torch.zeros(...)`, is now `torch.full(..., other_lbl)`. A/B class bits are
subsequently overwritten by either `--cutmix-ab-labels` (paper §6.16
asymmetric) or `--cutmix-label-area-prop` / `--cutmix-complete-label-scale`
(paper §6.13 area-proportional). Off-class bits **retain** `other_lbl`.

Default `0.0` reproduces pre-patch behavior exactly — patch is non-breaking.

## Motivation — Phase 84b probability-distribution diagnostic (iter46E)

The diagnostic that motivated this patch:

| split           | mean max-prob | interpretation                                |
|-----------------|--------------:|-----------------------------------------------|
| TRAIN defect    |   0.84–0.92  | own-class prob — well separated               |
| EVAL OOD chips  |   ~0.55      | near 0.5 threshold — borderline correct       |
| EVAL NI chips   |   ~0.46      | below threshold — correctly rejected          |

The OOD/NI max-prob ~0.55 / ~0.46 gap is only **0.09**. Any seed-noise or
small recipe perturbation can flip OOD chips above threshold, causing the
OOD over-fire that Phase 83 documented as ~3–48% Total FAR depending on
the cell.

Hypothesis: the **hard-zero off-class bit labeling on mix chips** teaches a
"sharp absent class" prior. Mix chips show {A, B} class textures and the
model is told the other two classes are exactly 0. For OOD chips that look
like *partial* mixtures of the trained patterns, the model has no calibration
signal to distinguish "this is a real partial defect" from "this is OOD
noise" — both sit at mid-range probability. Softening the off-class label
to e.g. 0.1 should bias the model toward outputting `(0.1, 0.1, 0.1, 0.1)`
for ambiguous patches, pushing OOD predictions **below** threshold rather
than near it.

## iter83 sweep design (planned, not yet executed)

Goal: measure Total FAR vs `--cutmix-other-label` value at fixed best-vanilla
recipe.

Base recipe (from iter18 winner iter46E):

- T7 BCE+LS=0.20
- `--cutmix-mode complement`
- `--cutmix-n-groups 3`
- `--cutmix-complete-label-scale 0.5`
- `--cutmix-pair masked`
- `--cutmix-pair-fill corner`
- `--cutmix-p 0.25`
- 8 epochs, cosine, single LR=1e-4
- seed=1 (then expand to 3 seeds {1, 7, 13} for variance)

Sweep axis `--cutmix-other-label ∈ {0.0, 0.05, 0.1, 0.15, 0.2}`:

| cell    | other-label | hypothesis                                                                  |
|---------|------------:|-----------------------------------------------------------------------------|
| 83A     |       0.0   | baseline (= iter46E exact reproduction)                                     |
| **83B** |    **0.05** | mild softening; expected: small Total FAR drop, small bit_F1 hit            |
| 83C     |       0.10  | matched to BCE+LS=0.20 floor; expected: best Total FAR / bit_F1 trade       |
| 83D     |       0.15  | over-smoothed; expected: bit_F1 collapse on edge defects                    |
| 83E     |       0.20  | match-LS limit; expected: model can't separate A/B from off-class at all    |

Eval: v15direct n=200 + Total FAR (NI + OOD) + per-OOD-pattern breakdown
(CenterDonut / CrossScratch / DiagonalSmear / Starburst).

Decision rule for paper §5 main: pick the largest `other_label` whose
**bit_F1 stays within −0.005 of baseline (83A)** AND **Total FAR ≤ 5%**. If
even 83B (0.05) violates either gate, the patch is paper-negative and we
report it as a counter-example (which is itself paper-valuable —
"calibration via off-class smoothing does not help").

## Limitations / open issues

- Patch is landed but **no training run yet** — iter83 sweep is on the queue,
  not in `outputs/`. This iter doc registers the intent and the recipe
  template; results come in a follow-up iter file once the 5-cell sweep
  finishes.
- Total FAR measurement infrastructure relies on having the 4 OOD wafer-pattern
  classes (CenterDonut, CrossScratch, DiagonalSmear, Starburst) present in
  v15direct. The intersection eval already includes them (3850 chips, 20
  class keys, see iter79 log line `[stage1] discovered 3850 chips across 20
  class keys`). Total FAR can be computed offline from existing
  `preds_chip.parquet` files for any prior cell.
- The patch is **only relevant for mix chips** (CutMix-generated synthetic
  composites). Original chips (the 100 ImageFolder samples per class)
  continue to use their raw multi-hot label. So `--cutmix-other-label` only
  has effect when `--cutmix-p > 0`.
- The patch does not affect single-mode CutMix (`--cutmix-mode single` — the
  rect-region path); only the complement / scattered / pair-masked code paths
  initialize `mix_t` from `other_lbl`. Need to verify single-mode path also
  uses the same init point if we ever sweep mode=single.

## Sources

- Patch file: `chip_multilabel/_train_chip_variant.py` (lines 410–413 CLI,
  lines 838–841 init)
- Motivating diagnostic: Phase 84b prob-distribution matrix (see
  `iter_18_total_far_correction.md` for the table)
- Base recipe: `outputs/iter46E_g3LS050_rect03/T7_iter46E_g3LS050_rect03_seed1_260510_140517/train_summary.json`

## Next iter branches

- **iter83** (this design) — 5-cell `cutmix-other-label` sweep at seed=1, then
  3-seed expansion {1, 7, 13} on the winning cell.
- **iter84** — if iter83 finds a winner, paired with `--cutmix-pair-fill` axis
  (corner vs noise vs alt) to test second-order interaction.
- **Paper §5 update** — iter18 + iter19 + iter20 collectively constitute the
  "Total FAR correction + seed-robustness + calibration" section. Manager
  report and paper main headline must move to `(bit_F1, Total FAR)` pairs
  with seed-variance bands. Narrator agent handles in separate iter.
