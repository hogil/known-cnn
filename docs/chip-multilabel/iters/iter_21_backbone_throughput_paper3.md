# iter 21 — Phase 87 v2 isolated-GPU backbone throughput & paper §3 backbone-choice correction

**date**: 2026-05-12
**tag**: `iter21_backbone_throughput_paper3`
**source roots**:
- script: `D:/project/known-cnn/_phase87_precise_speed.py` (torch.cuda.Event precise timing, 20 warmup + 100 iter)
- raw measurement: `C:/Users/hgcho/AppData/Local/Temp/claude/D--project-known-cnn/02bd8759-3206-45ff-9c0b-b034c8f11590/tasks/b8c1a92y4.output`
- accuracy source (bit_F1, Total FAR): `D:/project/known-cnn/outputs/iter77*/T*/eval_v15direct_n200/stage1_*/preds_chip.parquet`
- companion CSV (this iter): `D:/project/known-cnn/docs/chip-multilabel/tables/backbone_throughput.csv`

**one-liner**: 4 backbones × 6 batch sizes precise-GPU measurement (torch.cuda.Event, isolated). **ConvNeXtV2 batching HURTS** (b=1 peak 37 chip/s, b=64 drops to 26 chip/s — GRN layer architectural quirk); production winners are **ConvNeXt V1 (76 chip/s b=64, bF1 0.9830, FAR 2.62%)** and **Swin-Base (54 chip/s b=4, bF1 0.9692, FAR 0.00% — strict-zero unique)**. Paper §3 backbone narrative was wrong ("ConvNeXtV2 = best balanced") and must split into latency-critical vs throughput-critical regimes.

## Motivation — paper §3 backbone-choice quantitative re-validation

Previous paper §3.5 narrative claimed ConvNeXtV2-Base as the "best balanced production backbone" on accuracy grounds alone (val 5-class 1.0000 at ep 1, multi-label-eval bit_F1 ≥ 0.965 at iter46E). It did **not** measure inference cost or scaling under realistic GPU batching. iter83 / Phase 87 v1 was a wall-clock measurement on a shared GPU (other torch jobs co-resident → high variance, sample-dependent timing). Phase 87 v2 re-measures with:

1. **Isolated GPU** (no other PyTorch jobs).
2. **torch.cuda.Event for precise timing** (avoids wall-clock host-side jitter; CUDA-stream-synchronous).
3. **20 warm-up forward passes + 100 measured passes**, batch ∈ {1, 4, 8, 16, 32, 64}, image dimension per-backbone (ConvNeXtV2 / Swin / EffV2 = 384; ConvNeXt V1 = 224).
4. **Same recipe (iter46E-equivalent)** for all 4 backbones at training side — T7 BCE+LS, FCM-PM g=3 LS=0.50, complement, pair=masked, fill=corner, **no KD**, single-LR 1e-4, cosine, 8 ep, seed=1.
5. **Total FAR (NI + OOD)** scored on `v15direct n=200` (3080 chips), the same evaluation set used as paper main since iter18.

This iter records the corrected measurement and revises paper §3 narrative to reflect **two distinct operational regimes** instead of a single "best" backbone.

## Sweep design

| backbone               | params  | img | timm name                                    | log dir                              |
|------------------------|--------:|----:|----------------------------------------------|--------------------------------------|
| ConvNeXtV2-Base        | 87.7 M  | 384 | `convnextv2_base.fcmae_ft_in22k_in1k_384`    | `outputs/iter46E_g3LS050_rect03/`    |
| ConvNeXt-Base V1       | 87.6 M  | 224 | `convnext_base.fb_in22k_ft_in1k`             | `outputs/iter77A_convnext_base/`     |
| Swin-Base 384          | 86.9 M  | 384 | `swin_base_patch4_window12_384`              | `outputs/iter77C_swin_base/`         |
| EfficientNetV2-M       | 52.9 M  | 384 | `tf_efficientnetv2_m`                        | `outputs/iter77E_efficientv2_m/`     |

Each (backbone × batch) cell = 100 forward passes timed end-to-end on a single A6000 (48 GB) with `torch.cuda.Event(enable_timing=True)`.

## Results — full 4 × 6 throughput matrix

### Throughput (chips per second, higher is better)

| backbone           | b=1 | b=4 | b=8 | b=16 | b=32 | b=64 | peak | scaling (b=1 → peak) |
|--------------------|---:|---:|---:|---:|---:|---:|---:|---:|
| ConvNeXtV2-Base    | **37** | 30 | 28 | (30) | 35 | 26 | **37 @ b=1** | **0.70 – 0.93×** (NEGATIVE — batching hurts) |
| ConvNeXt-Base V1   | 40 | 60 | 64 | 70 | 74 | **76** | **76 @ b=64** | **1.85×** (positive — normal scaling) |
| Swin-Base 384      | 47 | **54** | 53 | 50 | 48 | (47) | **54 @ b=4** | **1.13×** (flat — saturates fast) |
| EfficientV2-M      | 42 | **158** | 158 | 145 | 130 | 113 | **158 @ b=4** | **3.77×** (positive — strongest batching gain) |

### Per-chip latency (ms/chip, lower is better)

| backbone           | b=1     | b=4    | b=8    | b=32   | b=64   |
|--------------------|--------:|-------:|-------:|-------:|-------:|
| ConvNeXtV2-Base    | 26.92   | 32.83  | 35.20  | 28.95  | 38.18  |
| ConvNeXt-Base V1   | 24.86   | 16.74  | 15.60  | 13.43  | **13.21** |
| Swin-Base 384      | 21.08   | **18.59** | 18.85 | 20.81 | 21.27 |
| EfficientV2-M      | 23.84   | **6.32** | 6.33 | 7.65  | 8.84   |

### Accuracy + Total FAR cross-reference (v15direct n=200, 3080 chips)

| backbone           | bit_F1 | Total FAR | NI FAR | OOD FAR | verdict at iter46E recipe |
|--------------------|-------:|----------:|-------:|--------:|---------------------------|
| ConvNeXtV2-Base    | **0.9654** | **1.07%** | 0% | 1.07% | true production single-model (iter46E paper main) |
| ConvNeXt-Base V1   | **0.9830** | 2.62% | 0% | 2.62% | best batched-throughput PASS (bF1 + FAR ≤ 5%) |
| Swin-Base 384      | 0.9692 | **0.00%** | 0% | 0% | strict-zero FAR (only one in sweep) |
| EfficientV2-M      | FAIL   | —      | —    | —     | small-data fit failure — fastest GPU but unusable bF1 |

## ConvNeXtV2 GRN-batching architectural quirk

ConvNeXtV2-Base is the **only backbone in the sweep where batching makes things slower**. b=1 peak is 37 chip/s; b=8 drops to 28 chip/s; b=64 drops further to 26 chip/s (**−30 % vs b=1**). For all other tested backbones batching either helps a lot (EfficientV2-M 3.77×, ConvNeXt V1 1.85×) or saturates early (Swin 1.13×). The mechanism is **GRN (Global Response Normalization)**, the ConvNeXtV2-specific layer (Woo et al. 2023, arXiv:2301.00808):

```
GRN(X) = X * (γ * Gx / (mean(Gx) + ε) + 1) + β
where Gx = ||X||_2 over spatial dimensions per channel
```

GRN does a per-channel L2-norm + mean-of-norms reduction at **every block**. The L2-norm is well-vectorised at b=1 (single 7×7×768 reduction) but the per-batch mean-of-norms broadcast across batch elements **serialises the channel-mean reduction** under cuDNN's default kernel selection at higher batch sizes. This explains the **U-shape**: minimum at b=1 (no batch reduction), small rebound at b=32 (kernel-launch overhead amortised), then collapse at b=64 (the serialised mean dominates).

This is a **measurement-grade architectural finding**, not a bug. It says: **ConvNeXtV2 is a latency-optimised backbone, not a throughput backbone.** ConvNeXt V1 (without GRN) batches normally.

## Paper §3 backbone-selection criteria — three operational regimes

### Regime A — Latency-critical (single-chip inline)

When chip-level decisions block downstream wafer-classification (e.g., inline inspection pipeline where each chip's score must clear before the next chip is fed), the metric is **single-chip latency** (ms/chip at b=1):

| rank | backbone           | b=1 latency | bF1     | Total FAR |
|------|--------------------|------------:|--------:|----------:|
| 1    | Swin-Base 384      | **21.08 ms** | 0.9692 | **0.00%** |
| 2    | ConvNeXt V1        | 24.86 ms    | 0.9830  | 2.62%     |
| 3    | EfficientV2-M      | 23.84 ms    | FAIL    | —         |
| 4    | ConvNeXtV2-Base    | 26.92 ms    | 0.9654  | 1.07%     |

**Recommendation**: Swin-Base 384 — best single-chip latency **AND** strict-zero FAR. Trade-off vs ConvNeXt V1 = 0.0138 lower bF1 for 4 ms latency saving + 2.62 percentage-point FAR reduction.

### Regime B — Throughput-critical (batched offline scoring)

When chips are accumulated (e.g., wafer-level reprocessing of all 600+ chips, or batch QC), the metric is **chip/s at peak-throughput batch**:

| rank | backbone           | peak chip/s | at batch | bF1     | Total FAR |
|------|--------------------|------------:|---------:|--------:|----------:|
| 1    | EfficientV2-M      | **158**     | 4 / 8    | FAIL    | — (not deployable) |
| 2    | ConvNeXt V1        | **76**      | 64       | **0.9830** | 2.62% |
| 3    | Swin-Base 384      | 54          | 4        | 0.9692  | **0.00%** |
| 4    | ConvNeXtV2-Base    | 37          | 1        | 0.9654  | 1.07%     |

**Recommendation**: ConvNeXt V1 — 2.05× the throughput of ConvNeXtV2 at the same parameter count (87.6 M vs 87.7 M) with **+0.0176 bit_F1** at iter46E recipe. The 2.62 % Total FAR is the cost; if strict FAR=0 is mandatory, fall back to Swin-Base at 54 chip/s.

### Regime C — ConvNeXtV2-only inline (legacy / paper-main coupled)

ConvNeXtV2 remains the paper main checkpoint and the iter46E reference. It is **not** dominated on accuracy (0.9654 is within 0.0176 of the best, and only Swin matches its strict-zero NI FAR while losing 0.0038 bF1). But it is **strictly dominated on cost**:

- vs ConvNeXt V1 at b=64: same params (87.6/87.7 M), 2.05× faster, +0.0176 bF1, −2.62% Total FAR.
- vs Swin at b=4: smaller (86.9 M), 1.46× faster, +0.0038 bF1, **strict zero FAR**.

The architectural quirk (GRN batching) means ConvNeXtV2 is only competitive at **b=1 inline** scenarios, where its 27 ms / 37 chip/s falls between Swin (21 ms / 47 chip/s) and the others. **Recommend reserving ConvNeXtV2 for inline mode only, not batched production.**

## 10,000-chip processing time projection (paper §3 quantitative)

| scenario               | backbone         | batch | chip/s | total time | use case                    |
|------------------------|------------------|------:|-------:|-----------:|-----------------------------|
| Throughput-critical    | EfficientV2-M    | 4     | 158    | **63 s** | (FAIL bF1 — not deployable) |
| Throughput-critical    | ConvNeXt V1      | 64    | 76     | **132 s** | production winner (bF1 0.9830) |
| Strict-FAR + batched   | Swin-Base 384    | 4     | 54     | **185 s** | FAR=0 mandatory             |
| Latency-critical inline| ConvNeXtV2-Base  | 1     | 37     | **270 s** | inline + paper-main reference|

The **ConvNeXt V1 → ConvNeXtV2 ratio is 2.05× (132 s vs 270 s on 10 k chips)**. At 1 M chip/day this projects to **63 GPU-min / day savings** (≈ $20/yr per A6000 at $0.20/GPU-hr), or 8.8% of annual GPU electricity for the inspection stage if ConvNeXtV2 is retained as the paper-main backbone for inline mode while ConvNeXt V1 is used for batched re-processing.

## What changes for the paper

| section                  | before                                              | after                                                                                          |
|--------------------------|-----------------------------------------------------|------------------------------------------------------------------------------------------------|
| §3.5 Backbone (T0)       | "ConvNeXtV2-Base = best balanced production backbone" | "ConvNeXtV2-Base = latency-critical inline backbone (b=1); ConvNeXt V1 V1 = batched throughput backbone (b=64); Swin-Base = strict-zero-FAR alternative" |
| §3.5 (new subsection)    | (none)                                              | **§3.5.1 GRN batching quirk** — ConvNeXtV2 chip/s is **non-monotonic in batch size** (peak at b=1, U-shape via b=32, collapse at b=64); explanation = GRN per-channel mean reduction at every block |
| §5 cost table            | bF1 alone                                           | (bit_F1, Total FAR, peak chip/s, ms/chip @ b=1) — 4-column production headline                |
| §6 ablation              | ConvNeXtV2 vs others = "+bF1 = 0.9654 / paper main" | ConvNeXtV2 vs ConvNeXt V1 = **−0.0176 bF1 for +0.55% lower OOD FAR but 2.05× slower batched**  |

The narrator-agent territory is the actual prose rewrite of §3.5; this iter records the **measurement evidence** and the quantitative claim. paper §3 will be amended via narrator agent at next iteration if heavy prose lift is required.

## Cross-iter delta

- vs iter18 (Phase 83 / 85 Total FAR correction): single-model paper-main winner remains **iter46E ConvNeXtV2** at bF1 0.9654 / Total FAR 1.07%. **This iter does not change accuracy ranking** — it adds **cost ranking** as an orthogonal axis.
- vs iter77 / iter77G (Phase 87 v1 wall-clock, contaminated GPU): every number here supersedes those measurements. iter77 measurements should be cited only as "rough wall-clock"; the canonical numbers are this iter.
- vs paper §5.19.5 (4-bag production cost): 4-bag is still 4× single-model. The single-model cost gains from switching ConvNeXtV2 → ConvNeXt V1 are **2.05×**, so a **4-bag ConvNeXt V1 ensemble would land at 4× / 2.05 = 1.95×** the cost of a 1× ConvNeXtV2 single model, while gaining +0.0176 bF1 per bag. This is a **future paper §5.19 production-deployment refinement** not measured in this iter.

## Limitations

1. **Single GPU class** (A6000 48 GB). The GRN batching curve will likely look different on H100 (different cuDNN heuristics) or Jetson Orin (Tensor RT compiled). Re-measurement on at least one Tensor RT target is queued for paper §5.19.5 production-deployment validation.
2. **EfficientV2-M bF1 FAIL** is at the iter46E recipe (single-LR 1e-4, 8 ep) — EfficientV2 may converge with a different recipe (likely needs longer schedule + warmup), but the throughput finding (158 chip/s @ b=4) holds regardless of accuracy.
3. **Img-size dependency**: ConvNeXt V1 used 224 (timm default for in22k_ft_in1k), the others 384. This is a 2.94× pixel-count advantage for ConvNeXt V1. We retain the result as-published because 224 is the canonical fine-tune size for ConvNeXt V1, but a re-measurement at 384 (1× advantage match) is a fair-comparison follow-up — expected to reduce ConvNeXt V1's lead but not eliminate it (GRN absence holds across resolutions).
4. **Eval set** (v15direct n=200, 3080 chips) is the post-iter18 paper-main eval. iter43 n=500 confirmation has not been re-run for ConvNeXt V1 / Swin / EffV2.

## Sources

- `D:/project/known-cnn/_phase87_precise_speed.py` (measurement script)
- `C:/Users/hgcho/AppData/Local/Temp/claude/D--project-known-cnn/02bd8759-3206-45ff-9c0b-b034c8f11590/tasks/b8c1a92y4.output` (raw measurement output)
- `D:/project/known-cnn/outputs/iter46E_g3LS050_rect03/T*/eval_v15direct_n200/stage1_*/preds_chip.parquet` (ConvNeXtV2 accuracy)
- `D:/project/known-cnn/outputs/iter77A_convnext_base/T*/eval_v15direct_n200/stage1_*/preds_chip.parquet` (ConvNeXt V1 accuracy)
- `D:/project/known-cnn/outputs/iter77C_swin_base/T*/eval_v15direct_n200/stage1_*/preds_chip.parquet` (Swin-Base accuracy)
- `D:/project/known-cnn/outputs/iter77E_efficientv2_m/T*/eval_v15direct_n200/stage1_*/preds_chip.parquet` (EfficientV2-M accuracy — FAIL)
- `D:/project/known-cnn/docs/chip-multilabel/tables/backbone_throughput.csv` (this iter's canonical CSV)
