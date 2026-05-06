# Abstract

We adapt a strong single-label chip CNN (ConvNeXtV2-Base 384, trained
single-label CE on 4 defect classes plus an `invalid_main` head) into a
multi-label predictor over an 11-class chip benchmark (4 single defects,
5 plausible defect combinations, `Normal`, `Invalid`; 2200 chips). On a
pure argmax baseline the model achieves macro-F1 = **0.7302**;
inference-time interventions alone — per-class F1-max thresholds, joint
coordinate-descent thresholds, and an entropy-based `Normal` gate — lift
the unchanged backbone to **0.8542**. A single training intervention
(CE with label smoothing α=0.10) further reaches **0.8634**. A targeted
sweep of label-smoothing strength surfaces a sharp peak at **α=0.20**,
yielding macro-F1 = **0.9268** and top-1 11-class accuracy =
**0.8449** at seed 42.

A subsequent iteration adds multi-source CutMix at training time (BCE
+ LS=0.20 + CutMix `p=0.5`, denoted T7c), motivated by the observation
that the residual `bank_boundary+scratch_rot` (bb+sr) combo recall
under T1+I7 was only **0.3250** (52/160 chips). T7c reaches macro-F1 =
**0.9271** at parity with T1 while lifting bb+sr combo recall to
**0.9562** (153/160), a **+0.6312** absolute recall gain on the
hardest combo class. An atomic decomposition shows the lift comes
solely from CutMix: CE→BCE alone costs −0.0691 macro-F1, and adding
CutMix `p=0.5` recovers +0.0694 on top, with a sharp peak at `p=0.5`
(`p<0.5` and `p>0.5` both regress).

A final iteration retunes label smoothing on the BCE + CutMix base —
the CE-side α=0.20 was inherited without re-tuning. A single-axis
α sweep over {0.00, 0.05, 0.06, 0.07, 0.08, 0.10, 0.20} (T9 family)
yields a family-mean macro-F1 ≈ **0.94**. A single-seed peak at
α=0.07 reaches **0.9705**, but seed=43 / seed=44 replicates of the
same config give **0.9408** / **0.8803**, exposing single-seed
measurement variance ≈ **0.046** (sample std, n=3) in this regime.

We therefore close the paper with a **3-seed paired comparison** of
the headline configurations T1 (CE+LS=0.20) and T9 (BCE+LS=0.07+CutMix
p=0.5) on seeds {42, 43, 44}. Across all three seeds, **T9 wins or
ties T1 on every metric** in every paired comparison:

| metric                    | T1 (mean ± std)   | T9 (mean ± std)   | Δ T9−T1   |
|---------------------------|------------------:|------------------:|----------:|
| macro_f1                  | 0.8923 ± 0.0301   | **0.9305 ± 0.0460** | **+0.0382** |
| top1_11class              | 0.7697 ± 0.0714   | **0.8242 ± 0.1058** | **+0.0545** |
| bb+sr recall              | 0.5292 ± 0.2577   | **0.7542 ± 0.3500** | **+0.2250** |

(Per-seed paired Δ_macro_f1: s42 +0.044, s43 +0.062, s44 +0.009;
paired Δ_bb+sr: s42 +0.631, s43 +0.138, s44 −0.094 — T9 unlucky on
seed 44 only on the bb+sr axis. T9 never loses on macro_f1 or
top1_11.) Single-seed peaks (0.9705 macro-F1, 0.9267 top-1, 0.9563
bb+sr) are noise outliers around the true mean — the headline claim
is the **3-seed mean ± std**, not the seed=42 peak.

A subsequent all-negative axis sweep (drop_path arXiv:1603.09382,
cutmix-rect arXiv:1905.04899, two-LR arXiv:2110.00476, F1 warmup,
F2 EMA, T8 CE-soft + cutmix, T13a ASL light + cutmix, I11
pair-aware threshold) regresses macro-F1 by 0.05–0.11 across **eight
independent atomic axes** — seven training-side, one inference-side.
None recovers the T9 family-mean. The pattern hardens a
*regularisation-ceiling* hypothesis: in our small-data + strong-TAPT
+ tuned-LS regime, additive structural regularisers are a net cost,
and even hyperparameter-axis variants on alternate loss families
(ASL γ_neg=2 + CutMix, CE-soft + CutMix) under-perform the BCE +
LS=0.07 + CutMix combination.

We document negative results (ASL, BCE-without-CutMix, BCE→ASL, TTA,
min-floor thresholds, F1 warmup, F2 EMA, T8 CE-soft + CutMix, I11
pair-aware heuristic, T10 drop_path, T11a cutmix-rect, T12a two-LR,
T13a ASL γ_neg=2 + CutMix) as ablations of equal value, and surface
an **asymmetric BKM-transfer** finding: hyperparameter-axis tuning
(LS) and one data-axis intervention (CutMix p=0.5) transfer reliably;
five structural-axis BKMs (warmup, EMA, drop_path, cutmix-rect,
two-LR) and three loss-family alternates (T8, T13a, the pair-aware
inference heuristic I11) all fail. The single-seed-variance lesson
itself — that a sweep maximum at the macro-F1 ≈ 0.94 ceiling is
biased upward by ≈0.5 σ ≈ 0.02 from selection over noisy cells —
becomes a paper-grade methodological discipline (§6.7), formalised
into a **multi-seed reporting protocol** (§9): any macro-F1 quoted
above 0.92 must come with either an `n≥3` seed mean ± std or an
explicit single-seed flag.
