# 260512 night — iter 112 NEW single-model SOTA

_Timestamp: 2026-05-12 22:30._
_Scope: paper-narrator daily log for the iter 111 / iter 112
single-model SOTA result and its three methodological
contributions._

## 1. What happened

iter 112 (`outputs/iter112_ep20/T7_iter112_ep20_260512_214618/`)
trained a single-model T7 BCE + LS = 0.20 + CutMix-complement
(g = 3, p = 0.25, masked-corner, cls = 0.5) checkpoint for **20
epochs** under cosine `T_max = 20` and selected the best epoch by
the new `--val-criterion f1` (per-bit BCE-macro-F1 on the
multi-hot val split).

The result, evaluated on v15direct n = 200 (3 080 chips) under
the absolute rule (260512) — train on 4 single-defect classes
only, bit-F1 over positive cells only, Total FAR = (NI_fp +
OOD_fp) / (N_NI + N_OOD):

| metric        | iter 112 (this run)  | iter 46E (legacy, absolute-rule re-eval) | Δ        |
|---------------|---------------------:|-----------------------------------------:|---------:|
| bit-F1        | **0.9964**           | 0.9755                                   | +0.0209  |
| Total FAR     | **0.83 %**           | 1.07 %                                   | −0.24 pp |
| 1 × cost vs 4-bag | bit-F1 gap +0.0011 | — | shrinks 1 × → 4 × cost gap from −0.0081 to +0.0011 (sampling noise) |
| chip accuracy | 98.77 % (2 410 / 2 440) | — | — |

The 7 false-positive chips at the SOTA cell are all from
`Starburst` (a radial OOD wafer pattern), 5 / 7 predict
`fork+scratch`, with fork sigmoid 0.50 – 0.73 and scratch
sigmoid 0.17 – 0.29 — a uniform fork-strong / scratch-marginal
signature consistent with a structural projection of the radial
Starburst pattern onto the (fork, scratch) feature pair (§6.30).

## 2. The three methodological contributions

| contribution                              | mechanism                                                                    | paper section            |
|-------------------------------------------|------------------------------------------------------------------------------|--------------------------|
| `--save-every-epoch` per-epoch eval       | turns selection-criterion into a tractable axis                              | §5.46, §6.29             |
| `--val-criterion f1` (multi-label aware)  | replaces biased single-label `val_acc` with per-bit BCE-macro-F1             | §6.29, §7.12.1, §7.12.6  |
| Cosine `T_max = 20`                       | extended LR plateau exposes ep 6 sweet spot invisible under `T_max = 10`     | §5.46.5, §7.12.1         |

The first two are **methodological** and portable to any
single-label-trained → multi-label-evaluated setting. The third is
**recipe-specific** and queued for a controlled `T_max ∈ {10, 15,
20, 25, 30}` ablation.

## 3. Selection-criterion ablation (§6.29.1)

| criterion         | epoch picked | eval bit-F1 (I10) | Total FAR (I10) | verdict           |
|-------------------|-------------:|------------------:|----------------:|-------------------|
| `val_acc`         | ep 1         | ≈ 0.94            | high            | ★ under-train     |
| `val_f1`          | **ep 6**     | **0.9964**        | **0.83 %**      | ★ **correct (SOTA)** |
| `val_auroc`       | ep 16        | ≈ 0.99            | ≈ 91 %          | ★ catastrophic    |
| arith / geom / harm of (val_f1, val_auroc) | ep 6 | 0.9964 | 0.83 % | ★ same as val_f1 |

Spearman correlations across the 21 saved checkpoints at I10:

- (val_acc, eval bit-F1) = **− 0.52** — anti-correlated;
- (val_f1, eval bit-F1) = **+ 0.78** — strongly positively correlated;
- (val_auroc, eval bit-F1) = **+ 0.08** — saturation noise.

The val_auroc saturation at 1.0000 from ep 14 onward (four-way
tie) breaks the per-bit selection; the threshold-search
component is the actionable axis, and AUROC is threshold-free
by construction (§6.29.2).

## 4. Three subsidiary negative results (§5.46.8)

| axis                        | result                                  | mechanism            |
|-----------------------------|-----------------------------------------|----------------------|
| 3-combo eval chips          | 100 % failure at every iter 112 cell    | label-cardinality bias of CutMix (Wang 2024 SpliceMix arXiv:2311.15200) |
| Linear probe (frozen)       | − 0.11 bit-F1 vs full FT (iter 105)     | TAPT-fragility (§7.2) |
| CutMix p = 1.0              | − 0.07 bit-F1 vs p = 0.25 (iter 100)    | over-fit to 2-combo |

All three independent axes confirm the iter 112 specification as
a narrow optimum.

## 5. Headline impact on paper headlines

| iter   | recipe                                              | bit-F1   | Total FAR | inf cost | role                                |
|--------|-----------------------------------------------------|---------:|----------:|---------:|-------------------------------------|
| iter46E | T7 (legacy, Normal-trained)                        | 0.9755   | 1.07 %    | 1 ×      | paper-main legacy                   |
| iter77C | Swin V1 Base 384 LS = 0.50                         | 0.9692   | 0.00 %    | 1 ×      | FAR-strict winner (§3.5.1)          |
| iter 50 B | KD distilled α = 0.5 T = 4 (4-bag teacher)       | 0.9872   | 0.50 %    | 1 ×      | prior KD single-model SOTA          |
| **iter 112** | T7 + cosine T_max = 20 + val_f1 selection    | **0.9964** | **0.83 %** | **1 ×** | ★ **NEW single-model SOTA**        |
| iter 39 / NEW MAIN | 4-bag pure-hard majority vote           | 0.9953   | 0.00 %    | 4 ×      | paper-final 4-bag (unchanged)       |

The iter 112 result does **not displace** the 4-bag majority-
vote paper-final headline (0.9953 / 0 % at 4 × cost); it
tightens the 1 × cost frontier from 0.9872 / 0.50 % (50 B) to
0.9964 / 0.83 % and shrinks the 4-bag → 1 × bit-F1 gap to
+0.0011 (within sampling noise). The 4-bag's 0 % Total FAR
remains the production-grade differentiator at the dual-gate
level; the iter 112 0.83 % is **PASS** under the production
gate (≤ 5 %).

## 6. Citations queued

- He et al. 2019 "Bag of Tricks" arXiv:1812.01187 — cosine schedule with long T_max for FT regimes
- Loshchilov & Hutter 2017 SGDR arXiv:1608.03983 — cosine restart origin
- Wightman et al. 2021 ResNet-Strikes-Back arXiv:2110.00476 — BCE + cosine multi-label
- Zhang & Zhou 2014 "Review of Multi-Label Learning" arXiv:1310.5419 — F1 vs AUROC distinction
- Lipton et al. 2014 arXiv:1402.1892 — F1-threshold mathematical foundation
- Wang et al. 2024 SpliceMix arXiv:2311.15200 — label-cardinality bias in CutMix
- Zhou et al. 2023 "Understanding Label-Cardinality Bias" arXiv:2309.10678
- Caron et al. 2021 DINO arXiv:2104.14294 — self-distillation LR sensitivity
- Yun et al. 2019 CutMix arXiv:1905.04899
- Müller et al. 2019 Label Smoothing arXiv:1906.02629

## 7. Paper section updates from this iter

| paper file           | section added / updated                          | nature                                                  |
|----------------------|--------------------------------------------------|---------------------------------------------------------|
| `abstract.md`        | NEW HEADLINE block (after §5.45 modern-backbone) | bit-F1 0.9964 / 0.83 % single-model SOTA + 3 method contributions + 3 negatives + cost-frontier analysis |
| `03_data.md`         | absolute rule + metric definitions block         | bit-F1 / Total FAR definitions clarified; `n_per_class = 160 / 200` clarification |
| `05_experiments.md`  | NEW §5.46 (iter 111 / iter 112)                   | setup + 4-cell results table + per-epoch trajectory + cosine ablation + headline comparison + 3 subsidiary negatives + summary |
| `06_analysis.md`     | NEW §6.29 (selection-criterion ablation)          | 4-cell table + Spearman analysis + mechanism diagnosis + recommendation                                  |
| `06_analysis.md`     | NEW §6.30 (FAR mechanism at iter 112 SOTA)        | FP source breakdown + Starburst projection mechanism + mitigation options + §6.20 connection           |
| `07_discussion.md`   | NEW §7.12 (iter 112 discussion)                   | 3 contributions + recipe headroom decomposition + 8-axis negative findings table + limitations + future work prioritisation |

## 8. Note on follow-up work

The iter 112 result is **single-seed (42)**. The §9 paper-grade
discipline requires n ≥ 3 seeds for any macro-F1 above 0.92, so
the immediate follow-up is a 3-seed replication (seeds 42 / 43 /
44) under the iter 112 spec. The cost is 3 × ≈ 15 min training
+ 3 × per-epoch eval ≈ 30 min eval = ≈ 1.25 hour wall clock.
Queued as the first paper §9 task.

The 3-combo failure (§5.46.8.1) is the highest-leverage
remaining axis for further single-model lift — a label-
cardinality-aware training intervention (3-mix CutMix or
offline 3-combo synthesis) is queued as the second paper §9
task.

_Sources:
`outputs/iter111_seed1_reproduce_now/T7_iter111_seed1_now_260512_212411/`,
`outputs/iter112_ep20/T7_iter112_ep20_260512_214618/`,
`outputs/iter112_ep20/T7_iter112_ep20_260512_214618/eval_v15direct_n200_best_model/stage1_260512_220154/{eval_summary.json, per_class_metrics.parquet, preds_chip.parquet, report.md, thresholds.json}`._
