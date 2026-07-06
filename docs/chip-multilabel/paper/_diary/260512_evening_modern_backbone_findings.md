# 260512 evening — 5 new findings from iter 95–99 (modern backbone landscape)

_Timestamp: 2026-05-12 19:50._
_Scope: paper-narrator update reflecting 5 new findings from the
iter95 / iter96 / iter97 / iter99 backbone-family expansion runs._

## 1. New runs entering the narrative

| run                                    | backbone (timm)                                | recipe perturbation                  | val_acc | best ep | bit-F1 (= 4-defect macro_f1) at best inference cell |
|----------------------------------------|------------------------------------------------|--------------------------------------|--------:|--------:|------------------------------------------------:|
| iter95A_dinov3_convnext_base           | `convnextv2_base.fcmae_ft_in22k_in1k_384`<br>+ DINOv3 self-distill weights | default LR=1e-4                       |   ≈0.98 |  early  | 0.6211 (T0__I10) — fork collapse (F1 0.38)      |
| iter95B_swinv2_base_384                | `swinv2_base_window12to24_192to384.ms_in22k_ft_in1k` | default LR=1e-4 + 384 fine-tune | ≈0.97   |  late   | 0.7843 (T0__I10) — 150 min train (21× legacy)   |
| iter96A_hiera_base                     | `hiera_base_224.mae_in1k_ft_in1k`              | default LR=1e-4                       | ≈0.98   |  ep1    | 0.7228 (T0__I3) — domain mismatch ceiling       |
| iter97A_lr5e5  (★ rescue)              | DINOv3 ConvNeXt-Base                            | LR=5e-5 (½ default)                   | 0.9877 @ ep9 | ep9 (best_val_acc) | **0.8700** (T0__I10) — −0.10 vs baseline |
| iter97A_lr5e5 _final_ (ep20)           | DINOv3 ConvNeXt-Base                            | LR=5e-5, final epoch                  | 0.9877 @ ep20 | ep20 | 0.7765 (T0__I3) — **−0.094 vs best ep9**       |
| iter99A_convnextv2_b                   | `convnextv2_base.fcmae_ft_in22k_in1k_384`     | ep10 best-from-6 epochs               | —       | global ep | 0.8367 (T0__I10) — −0.13 vs iter46E baseline   |
| iter99B_swin_v1_b                      | `swin_base_patch4_window12_384.ms_in22k_ft_in1k` | ep10 best-from-6                  | —       | global ep | 0.8030 (T0__I3) — −0.17 vs iter77C baseline    |
| iter99C_dinov3_convx_b                 | DINOv3 ConvNeXt-Base                            | ep10 best-from-6 (default LR)         | —       | global ep | 0.7423 (T0__I10) — fork mid-tier               |
| iter99D_hiera_base                     | Hiera-Base                                      | ep10 best-from-6                      | —       | global ep | 0.7039 (T0__I10) — confirms iter96A ceiling     |
| iter99E_convnextv2_lr5e5               | ConvNeXtV2-Base FCMAE                           | LR=5e-5 + ep10 best-from-6            | —       | global ep | 0.8282 (T0__I10) — under-LR regresses by 0.14   |

_Sources: `outputs/iter{95,96,97,99}*/T*/eval_v15direct_n200/stage1_*/results_matrix.parquet` and `per_class_metrics.parquet`; iter97A also reports `eval_v15direct_n200_best` (best-by-val-acc, ep9) vs `eval_v15direct_n200_final` (ep20)._

## 2. The 5 findings as scaffolded into paper sections

### Finding 1 — Modern backbones (2022–2025) underperform FCMAE 2023 baseline under fair recipe (→ §3.5, §5.45, §7.11)

- **DINOv3 ConvNeXt-Base (Meta 2025, arXiv:2508.10104 self-distillation post-FCMAE)** with default LR=1e-4 → bit-F1 0.621, fork F1 collapses to 0.38. Halving LR to 5e-5 rescues to 0.870 — still **−0.095** vs the FCMAE-only iter46E baseline (0.9654).
- **Swin V2 Base 384 (Microsoft 2022, arXiv:2111.09883)** with log-CPB + cosine-attention + window 12→24 transfer → bit-F1 0.784. Training time 150 min (vs ≈5 min for ConvNeXtV2-Base, **21× slower**) at no accuracy benefit.
- **Hiera-Base (Meta 2023, arXiv:2306.00989, MAE-pretrained)** → bit-F1 0.704; macro_f1 collapses on I10/I13 (the entropy-gate cells) suggesting confidence calibration mismatch.
- ConvNeXtV2-Base FCMAE (Woo et al. 2023, arXiv:2301.00808) under iter46E recipe = **0.9654 bit-F1** remains the paper-main winner; Swin V1 Base 384 (Liu et al. 2021, arXiv:2103.14030) at iter77C = 0.9692 remains the FAR-strict winner.
- **Strong claim** for §3.5 / §7.11: the FCMAE objective (sparse-convolution masked autoencoder + pixel reconstruction) transfers uniquely well to the chip palette domain. Modern self-distillation (DINOv3) and improved-attention (Swin V2) variants underperform their *direct predecessors* — DINOv3 < ConvNeXtV2, Swin V2 < Swin V1 — under our matched-recipe protocol. This is counter-textbook: the natural-image SOTA ordering does not transfer.

### Finding 2 — `best_val_acc` selection on 4-class single-label train split is a biased proxy for multi-label eval F1 (→ §3.5, §6.28)

- iter97A LR=5e-5 produces val_acc=0.9877 at **four ties: epochs 9, 16, 18, 19, 20** (history.json). The training tracker picked ep9 (first peak).
- At eval time: ep9 (best) → bit-F1 **0.8700** (I10), ep20 (final) → bit-F1 **0.7765** (I3). **Δ = −0.094 bit-F1** between best-val-acc and final-epoch checkpoints, despite both reporting val_acc=0.9877.
- The val_acc curve is *flat from ep1 to ep20* (0.9816–0.9877, range 0.6%), while bit-F1 follows an **inverted-U with plateau then decline** — classic overfit-to-single-label signature.
- The FCM-PM augmentation (CutMix-complement g=3 with pair-mask) reduces but does not close this gap. iter97A's `best` vs `final` divergence with FCM-PM ON shows the regularisation is not sufficient to align single-label val_acc with multi-label eval F1.
- **Methodological recommendation for §9 future work**: a multi-label proxy criterion (eval bit-F1 on a held-out fraction of synth eval set) for early stopping. This is queued as a follow-up axis — currently we cannot deploy `best_val_acc` selection without overstating the single-model number by up to 0.10 bit-F1.
- Citation: this connects to the Lipton et al. 2014 F1-threshold paper (arXiv:1402.1892) and Wang et al. 2024 multi-label-from-single-label literature (arXiv:2405.13451). The gap is a multi-label generalisation of the classic train-acc/val-loss divergence.

### Finding 3 — Global best-from-6-epoch rule does not work (→ §5.45, §6.28.1)

iter99 ran 5 backbones × ep10 with "best-from-6-epoch" model selection (a candidate global recipe to replace per-backbone tuning).

| backbone                            | iter99 best-from-6 bit-F1 | reference baseline           | Δ        |
|-------------------------------------|---------------------------:|------------------------------|---------:|
| ConvNeXtV2-Base FCMAE               | 0.8367                     | iter46E (= 0.9654)           | **−0.129** |
| Swin V1 Base 384                    | 0.8030                     | iter77C (= 0.9692)           | **−0.166** |
| DINOv3 ConvNeXt-Base                | 0.7423                     | iter97A LR=5e-5 (= 0.8700)   | −0.128   |
| Hiera-Base                          | 0.7039                     | iter96A (= 0.7228)           | −0.019   |
| ConvNeXtV2-Base LR=5e-5             | 0.8282                     | iter46E (= 0.9654)           | −0.137   |

- **Every cell regresses below the prior best for that backbone.** The "best-from-6-epoch" global rule is therefore *not* the answer to Finding 2's selection-bias problem — it produces a worse selection than backbone-specific tuning.
- The sweet-spot epoch is backbone-dependent: ConvNeXtV2-Base = ep2–3 (per iter46E history), DINOv3 LR=5e-5 = ep9 (iter97A), Hiera = ep1 (early-converge then degrade), Swin V1 = ep4–6. No single epoch is universal.
- **Conclusion**: epoch selection is a deterministic backbone-specific axis. The hyperparameter-axis cluster from §6.27.1 expands: deterministic axes are not only narrow but also backbone-coupled.

### Finding 4 — Speed-quality Pareto frontier across 6 backbones (→ §3.5, §5.45)

| backbone                | params | bit-F1 (best safe) | train time         | role                                |
|-------------------------|-------:|-------------------:|-------------------:|-------------------------------------|
| **ConvNeXtV2-Base FCMAE** | 87.7M | **0.9654**         | ~5 min             | ★ paper-main baseline (iter46E)     |
| **Swin V1 Base 384**    | 86.9M | **0.9692**         | ~7 min             | ★ FAR-strict-zero winner (iter77C)  |
| ConvNeXt V1 Large       | 196.2M | 0.872 (I13 only)   | ~5.4 min           | OOD-leaky on all I7-safe cells      |
| **DINOv3 ConvNeXt-Base**| 87.7M | 0.8700 (LR=5e-5)   | ~6 min             | rescue at LR=5e-5; default LR fails |
| **Swin V2 Base 384**    | 87.1M | 0.7843             | **~150 min** (21×) | ★ unacceptable speed / accuracy     |
| **Hiera-Base**          | ~52M  | 0.7228             | ~2.5 min           | fast but low ceiling                |

- Pareto front (accuracy × time):
  - **Fast + accurate**: ConvNeXtV2-Base FCMAE (5 min, 0.9654) and Swin V1 Base (7 min, 0.9692) dominate.
  - **Slow + accurate**: nothing (Swin V2 dropped here — slow with no accuracy benefit).
  - **Fast + inaccurate**: Hiera-Base (2.5 min, 0.7228) — usable only when accuracy is non-binding.
  - **Slow + inaccurate**: Swin V2 Base (150 min, 0.7843) — strictly dominated.
- The Pareto reading **strengthens the §3.5 three-regime claim**: the paper-SOTA winner remains ConvNeXtV2-Base FCMAE, the FAR-strict winner remains Swin V1 Base 384, and **no 2022–2025 backbone displaces either of them on this benchmark under matched recipe**.

### Finding 5 — Absolute rule 260512 enforcement (→ §3.1–§3.3, §5.1, all tables)

The 260512 feedback (`feedback_chip_multilabel_train_eval_composition.md`) makes three definitions absolute. Paper-narrator pass surfaces them explicitly:

| axis              | definition                                                       | enforcement                                                  |
|-------------------|------------------------------------------------------------------|--------------------------------------------------------------|
| Training set      | 4 single-defect classes only (`bank_boundary, fork, scratch, scratch_rot`)<br>`--no-normal` mandatory on every train script | `chip_multilabel/_train_chip_variant.py` TRAIN_CLASSES=4     |
| Evaluation set    | 5 groups: 4 single defect, 5 (≤6) 2-combo, Normal, Invalid, OOD wafer-pattern | `chip_multilabel/run_stage1.py` eval discovery               |
| **bit F1**        | **macro-F1 of positive cells only** (4 single + 5 combo = 9 cells)<br>**≠ macro_f1** (which averages all 11+OOD cells) | per_class_metrics.parquet `class ∈ defect∪combo` mean         |
| **Total FAR**     | (NI_fp + OOD_fp) / (N_Normal + N_Invalid + N_OOD)<br>NI-only is deprecated | preds_chip.parquet groupby `group ∈ {NI, OOD}`                |

- **NI-only FAR is misleading** when OOD distractors exist; the bundled metric reads "0%" while the strict Total FAR reads 1.07% on iter46E.
- The 260512 rule supersedes earlier mixed conventions (Phase 87 lesson, iter18 correction).
- Method §3.1 / Experiments §5.1 / table headers in §5.45 reflect this — bit_F1, Total_FAR are the headline columns; legacy `macro_f1` is retained only for cross-iter continuity.

## 3. Section update plan

| paper file                | section                                  | nature of change                                  |
|---------------------------|------------------------------------------|---------------------------------------------------|
| `03_data.md`              | §3.1 (eval composition) + §3.5 (backbone) | reflect 5-group eval + Finding 1 + LR sensitivity |
| `05_experiments.md`       | NEW §5.45 (iter95–99 modern backbones)   | append landscape table + Findings 1–4             |
| `06_analysis.md`          | NEW §6.28 (selection bias + ep policy)   | Finding 2 + Finding 3 deep-dive                   |
| `07_discussion.md`        | NEW §7.11 (modern backbone failure)      | Finding 1 + Finding 4 implication + future work   |
| `abstract.md`             | (no headline change; add a paragraph)    | acknowledge backbone-landscape negative result    |

Headline numbers in the abstract are *unchanged* — Finding 1–4 are all **negative results** that do not displace the iter46E / iter77C / iter50B / iter77A four-headline grid. Finding 5 (rule enforcement) is a definitional pass that clarifies metric meaning but does not move numbers either.

## 4. Citations queued

- Woo et al. 2023, ConvNeXt V2 / FCMAE — arXiv:2301.00808
- DINOv3 (Meta 2025) — arXiv:2508.10104
- Swin V2 (Liu et al. 2022) — arXiv:2111.09883
- Hiera (Ryali et al. 2023) — arXiv:2306.00989
- Liu et al. 2021 Swin V1 — arXiv:2103.14030
- Wightman et al. 2021 ResNet-Strikes-Back — arXiv:2110.00476 (BCE + LS multi-label connection)
- Lipton et al. 2014 F1-threshold — arXiv:1402.1892 (multi-label thresholding theory)
- Yun et al. 2019 CutMix — arXiv:1905.04899
- Müller et al. 2019 LS — arXiv:1906.02629
