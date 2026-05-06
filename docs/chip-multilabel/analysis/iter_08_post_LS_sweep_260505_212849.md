# Iter 08 — post-LS-sweep next-axis analysis

**TS**: 260505_212849
**Author**: analyst-iter8 (opus)
**Anchors**:
- T9d (LS=0.07 seed=42, lucky outlier): macro_f1 **0.9705**, top1_11 **0.9267**.
  Source: `outputs/stage1_260505_211334/`.
- T9g (LS=0.07 seed=43, realistic): macro_f1 **0.9408**, top1_11 **0.8307**.
  Source: `outputs/stage1_260505_212557/`.
- T7c (LS=0.20 seed=42): macro_f1 0.9271, top1_11 0.8307. Source:
  `outputs/stage1_260505_195730/`.

LS sweep verdict: BCE+LS in [0.05, 0.10] + CutMix p=0.5 plateaus at
**macro_f1 ~0.94 mean ±0.03 single-seed**. Single-LS gradient is exhausted.

---

## 1. Re-frame the variance — it is *not* uniform across classes

Per-class metrics under best inference cell (`T0__I7`, comparable thresholds):

| class | T9d (lucky)   | T9g (realistic) | Δ (lucky − realistic) |
|---|---|---|---:|
| bank_boundary | P=0.992 R=0.958 **F1=0.974** AP=0.982 | P=0.998 R=0.964 **F1=0.981** AP=0.982 | **−0.007** |
| fork          | P=1.000 R=0.895 **F1=0.945** AP=0.988 | P=0.739 R=0.908 **F1=0.815** AP=0.825 | **+0.130** |
| scratch       | P=0.991 R=0.935 **F1=0.962** AP=0.976 | P=0.973 R=0.962 **F1=0.968** AP=0.980 | **−0.006** |
| scratch_rot   | P=1.000 R=1.000 **F1=1.000** AP=1.000 | P=1.000 R=1.000 **F1=1.000** AP=1.000 | 0.000 |

**Reading**: bank_boundary, scratch, scratch_rot are seed-stable across
seed=42→43. The entire ±0.030 macro_f1 single-seed swing comes from **one
class — fork** (F1: 0.945 vs 0.815, ΔF1 = 0.130; AP: 0.988 vs 0.825,
ΔAP = 0.163).

Mechanism: fork's F1-max threshold sits at **0.10–0.22** (lowest of all
four classes — see iter 1-7 per-class tables), with **fork support of 640
chips** (the largest, since fork appears in {fork, bb+fork, fork+sr,
fork+scratch} = 4 of 11 eval classes). The combination of (i) lowest
threshold, (ii) most multi-positive eval support, and (iii) most diffuse
spatial pattern means small shifts in fork's logit distribution from one
seed initialization to another flip ~50–80 chips between TP and FP.

**This is the right framing for "next axis":** the question is not "how do
we lift the 0.94 mean", it is "how do we shrink fork's seed sensitivity
without sacrificing the other three already-near-perfect classes". Any
candidate that does not touch fork's logit distribution is unlikely to
move the needle.

## 2. Operating context (constraints)

- 4-class × ~100 chips classification_chips/, eval = 2200 chips × 11
  classes (chip_multilabel_eval_full).
- Same epochs=8, batch=8, accum=4, base LR=1e-4.
- No rotation/flip aug — only translate(0.05) + scale(0.95–1.05).
- No TTA, no warmup, no EMA (iter 7 confirmed both regress).
- Each candidate = one GPU job ~6–10 min on chip_multilabel/run_stage1
  pipeline.

## 3. Candidate axes — graded against fork-variance reduction

Each entry: **mechanism → expected fork ΔAP, expected macro_f1 (opt /
pess), bb+sr risk, citation**.

### 3.1 ★★★ drop_path_rate=0.05 (timm stochastic depth) — TOP RECOMMENDATION

CFG flag already exists at `chip_multilabel/_train_chip_variant.py:206-207`
(`--drop-path-rate`, currently default 0.0). One-line dispatch change.

**Mechanism**: stochastic depth randomly drops residual blocks in
ConvNeXtV2 during training. Each forward pass sees a different sub-network
and the loss must work for the ensemble. This is **explicit model
ensembling baked into a single set of weights** — the canonical recipe for
seed-variance reduction in modern conv architectures (it is the only
hparam that varies with model size in the original ConvNeXt paper).

**Why it directly attacks the fork-variance failure mode**: fork's per-seed
flip is driven by the most-diffuse class anchoring on whichever minor
pattern dominated the early-epoch gradient on that seed. drop_path forces
the head to make a fork decision from many different sub-network feature
extractors over training, averaging out the early-epoch lucky-feature
attraction. The bias-variance theory result (Hayou 2023) shows
explicitly that as drop probability increases from 0 toward a small value
~0.05, **variance decreases monotonically** with negligible bias cost —
exactly the regime we want.

**Expected effect**:
- Fork AP: 0.825 → 0.92–0.96 (recovers most of the 0.163 seed gap)
- Mean macro_f1 (across 2 seeds): 0.94 → **0.95–0.96**
- Pessimistic: 0.93 (small under-fit if 0.05 too aggressive on 327
  training chips)
- bb+sr risk: zero (scratch_rot already at F1=1.0 for both seeds)
- Single-seed variance: **the operationally meaningful improvement** —
  paper-narrator can claim "honest mean improves; seed std halves".

**Citations**:
- Huang et al., "Deep Networks with Stochastic Depth", ECCV 2016 (arxiv 1603.09382) — seminal.
- Liu et al., "A ConvNet for the 2020s" (ConvNeXt paper) — confirms
  drop_path_rate is the only hparam tuned per scale.
- Hayou, "Regularization in ResNet with Stochastic Depth" (TMLR 2023) —
  theoretical bias-variance proof.

### 3.2 ★★ Multi-seed averaging (n=3) on T9d-style config

Cheap (3× the same config, average sigmoid logits at inference). Does not
push the **mean** but pushes the **honestly-reportable single number**.

**Mechanism**: independent seed ensembles average out the fork
init-flip variance directly.

**Expected effect**:
- macro_f1: 0.94 → 0.95–0.96 (mean ensemble usually beats best single seed
  by ~0.5σ)
- Variance: drops to ~σ/√n
- bb+sr: unchanged (already 0.96 on each seed).

**Citations**: standard ML, no specific paper needed.

**Why ranked below #3.1**: averaging is bookkeeping — it does not change
the underlying model's training dynamics. drop_path improves a single
training run (cheaper at deploy), and the two are stackable.

### 3.3 ★★ Two-LR group (backbone=5e-5, head=2e-4) at LS=0.07

The CFG flags `--lr-backbone` / `--lr-head` already exist at
`_train_chip_variant.py:198-201`. The TAPT backbone is strongly initialized
on the same chip data; further fine-tuning at LR=1e-4 may be over-fitting
the backbone's already-correct features and starving the 4-class head.

**Mechanism**: keep TAPT backbone almost frozen (5e-5), let the multi-label
classification head adapt to BCE+CutMix+LS surface. The fork class —
because of its diffuse spatial pattern — most likely has its
discrimination signal already in the TAPT representation; the variance
comes from re-shuffling those features under the head.

**Expected effect**:
- Mean macro_f1: 0.94 → 0.94–0.95 (small)
- Pessimistic: 0.92 if backbone too cold to integrate CutMix-mixed pixels
- Variance: probably reduced (head sees less init-shuffle from backbone
  drift)
- Citation: anomaly-detection BKM (sister repo, validated). Same loss
  family, different domain.

**Why ranked below #3.1**: less direct attack on fork-variance specifically,
and the BKM source domain is binary anomaly (different from 4-class +
CutMix). Phase F already showed warmup transferred poorly from the same
source.

### 3.4 ★ cutmix-rect sweep at LS=0.07 (rect ∈ {0.25, 0.35, 0.65})

CFG flag exists (`--cutmix-rect`, default 0.5). cutmix-rect controls the
paste area; smaller rect → smaller patch → less label corruption per chip
but less combo signal. Larger rect → more combo signal but more label
ambiguity.

**Mechanism**: rect=0.5 was tuned on T7 with LS=0.20. With LS=0.07 the
target distribution is sharper, so the tolerable rect could differ.

**Expected effect**:
- Mean macro_f1: 0.94 → 0.93–0.95 (likely flat — rect=0.5 is the natural
  midpoint and we have no evidence it's wrong here)
- Variance: unclear; smaller rect could increase variance by giving fork
  fewer negative pixels per batch.
- Citation: Yun et al., "CutMix" (ICCV 2019, arxiv 1905.04899) — rect
  uniform from Beta(α=1) is the original; varying rect is a reasonable
  hparam.

**Why mid-tier**: orthogonal to the fork-variance issue. Could improve, but
no targeted mechanism.

### 3.5 ★ grad_clip=0.5 (was 1.0)

CFG flag exists (`--grad-clip`). Anomaly-detection BKM. Cheaper to test
than EMA/warmup, and unlike those, grad_clip is **not** an averaging
mechanism, so the over-smoothing failure mode of F2 doesn't apply.

**Mechanism**: BCE+CutMix produces occasional spike gradients on
high-confidence-wrong batches (where one chip was CutMix-pasted into
another and the BCE targets disagree sharply). Tighter clip stabilizes
these, which should reduce the seed-driven lucky/unlucky early-batch
swings.

**Expected effect**:
- Mean macro_f1: 0.94 → 0.94–0.95
- Variance: ~half if the source of variance was early-batch gradient
  spikes (plausible but unverified).
- Citation: anomaly-detection BKM.

### 3.6 — (skip) Per-class LS

The reformulated problem (only fork is variance-sensitive) is exactly what
this targets, but per-class LS is **not currently implemented** —
`losses.py` would need a new branch and the CFG plumbing. That violates
the "single atomic axis, one CLI flag" constraint. Leave for iter 9 if
drop_path doesn't fix.

### 3.7 — (skip) Class-balanced CutMix forcing bb+sr pair

bb+sr is already at recall 0.96 / F1 1.0 on both seeds. The constraint to
target instead is **fork pairs** (fork+scratch, fork+sr, bb+fork). Same
implementation cost as per-class LS, and the upside is small because
fork-pair recall is already high; only the precision (and fork
single-class TP/FP) matters. Leave for iter 9.

### 3.8 — (skip) ASL+CutMix retry

T4 (ASL γ_neg=4) regressed −0.078 in iter 4. Adding CutMix on top is
fighting two opposing forces (ASL pulling fork's negative logits very low,
CutMix injecting partial-positive labels for fork). High experiment risk
for unclear payoff.

### 3.9 — (skip) I12 inference ensemble (T1 + T9d sigmoid average)

This is bookkeeping at the inference layer (no retrain). Useful for
**reporting** but does not fix the underlying single-seed variance and
double-counts T9d's outlier luck (averaging an outlier seed with a
non-outlier just regresses toward the mean — which is what we already
have).

### 3.10 — (skip) Mixup α=0.05

Mixup pixel-blends, while CutMix region-pastes. They are different
augmentations and ablating across both at α=0.05 is two atomic axes
collapsed. CutMix p=0.5 is the established peak; replacing it with mixup
is a regression bet, not an additive bet. Could be worth iter 10 but not
the next axis.

## 4. Top recommendation

**Single atomic axis next: `--drop-path-rate=0.05`** on the T9d-equivalent
config (BCE+LS=0.07+CutMix p=0.5 rect=0.5, ep=8, bs=8 accum=4, LR=1e-4).
Run on **two seeds (42, 43)** so the variance claim is honest.

Expected outcome per seed: macro_f1 0.94–0.96; expected mean across the
two seeds: **0.95**, with a tighter spread than the current ±0.030.

Implementation:
```bash
python -m chip_multilabel._train_chip_variant \
  --variant T7 --ls 0.07 --cutmix-p 0.5 --cutmix-rect 0.5 \
  --drop-path-rate 0.05 --epochs 8 --batch 8 --accum 4 --lr 1e-4 \
  --seed 42 --tag T9h_dpr05_s42

python -m chip_multilabel._train_chip_variant \
  --variant T7 --ls 0.07 --cutmix-p 0.5 --cutmix-rect 0.5 \
  --drop-path-rate 0.05 --epochs 8 --batch 8 --accum 4 --lr 1e-4 \
  --seed 43 --tag T9h_dpr05_s43
```

Then `chip_multilabel/run_stage1.py` over both checkpoints, same eval set.

If both seeds land within ±0.010 of each other (whether at 0.95 or
elsewhere), drop_path was the right axis and the variance story closes.
If they still spread ±0.030, the variance is not init-route but
data-shuffle (which is the next probe — fixed batch order or larger
effective batch via accum=8).

## 5. Two-line rationale for paper-narrator

> Iter 8 LS sweep mean macro_f1 plateaus at 0.94 ±0.030 single-seed,
> with the entire ±0.03 swing localized to **fork** (ΔF1=0.13 between
> seeds 42, 43). Adding stochastic depth `drop_path_rate=0.05` is the
> theoretically grounded, single-flag, paper-citable next axis: it
> targets seed-variance directly without disturbing the three already-
> stable classes (bank_boundary, scratch, scratch_rot).

---

_Sources cited:_
- [Deep Networks with Stochastic Depth](https://arxiv.org/abs/1603.09382)
- [Regularization in ResNet with Stochastic Depth (Hayou 2023)](https://openreview.net/pdf?id=8v4Sev9pXv)
- [CutMix: Regularization Strategy (Yun 2019)](https://arxiv.org/pdf/1905.04899)
- [ConvNeXt — A ConvNet for the 2020s](https://arxiv.org/abs/2201.03545)
- iter 7 atomic decomposition: `outputs/stage1_260505_195730/per_class_metrics.parquet` (T7c)
- iter 8 LS sweep variance evidence: `outputs/stage1_260505_211334/per_class_metrics.parquet` (T9d), `outputs/stage1_260505_212557/per_class_metrics.parquet` (T9g)
- chip_multilabel/notes.md (full iter history)
- chip_multilabel/_train_chip_variant.py:206 (`--drop-path-rate` CFG flag already wired)
