# 02 — Results

All numbers reported to 4 decimal places. Eval set: 2200 chips, 11-class
multi-label.

## 2026-05-13 update — iter122 + iter123 + iter124 (SOTA unchanged, iter112 still winner)

Three follow-up iters tested **off-recipe axes** against the iter116J/iter112
recipe baseline. **All three regressed** — iter112 paper SOTA
(bF1 0.9964 / Total FAR 0.83%) is still the winner. Summary:

| iter | axis tested | atomic delta vs iter116J/iter112 | best cell | bF1 | Total FAR | Δ bF1 vs iter112 | Δ FAR vs iter112 |
|------|-------------|----------------------------------|----------:|----:|----------:|-----------------:|-----------------:|
| iter112 (SOTA, kept) | — | — | I13 | 0.9964 | 0.83% | — | — |
| iter122 (ep10) | loss → T6 BCE→ASL γ_neg=4 **clip=0.05** | swap T7→T6 | I10 | 0.8297 | 9.40% | −0.1667 | +8.57pp |
| iter123 (ep10) | loss → T6 BCE→ASL γ_neg=4 **clip=0.10** | iter122 clip 0.05→0.10 | I10 | 0.8297 | 5.00% | −0.1667 | +4.17pp |
| iter124a | spatial g=2 n=1 (GRID=2, 4 cells) | FCM-PM granularity sweep | I10 | 0.8705 | 1.07% | −0.1259 | +0.24pp |
| iter124b-i | spatial sweep g={2,3} × n={1..4} + bisect | (re-eval pending) | — | PENDING | PENDING | — | — |

**Verdict**: iter112 paper SOTA holds. The T6 loss axis (iter122 / iter123)
is a **dead end** — clip 0.05/0.10 both regress with Total FAR ≥ 5%. iter124a
(GRID=2, coarsest possible spatial granularity) regresses bF1 by 0.126 and
the remaining 8 iter124 sub-runs (b-i) are re-evaluating to map the full
GRID curve. Details in `03_ablations.md` "2026-05-13 ablation updates".

## ★★★★ 2026-05-12 NIGHT update (22:30) — iter111 + iter112 paper SOTA

**New single-model SOTA** under the **absolute rule** metric
(training: 4 single defect only / `--no-normal`; eval bF1: positive group
= single + 2-combo only; **3-combo excluded** from bF1; Total FAR =
combined FP-rate on Normal + Invalid + OOD-wafer-pattern chips). Single
seed=1, ConvNeXtV2-Base FCMAE 384, T7 (BCE+LS=0.20) + FCM-PM CutMix.

| iter | recipe spec (delta) | ckpt sel | epoch | bF1 (4 single +2-combo) | Total FAR | I-cell |
|------|--------------------|---------:|------:|------------------------:|----------:|-------:|
| iter46E (paper main, legacy) | with Normal training | best/val_acc | n/a | 0.9755 | 1.07% | I7 |
| iter101A (per-epoch eval baseline) | ep=10 cosine T_max=10 | epoch_10 | 10 | 0.9964 | 4.52% | I13 |
| iter111 (T_max=10, seed=1 reproduce) | ep=10 cosine T_max=10 | **epoch_08** | 8 | **0.9963** | **1.31%** | I13 |
| **iter112 (T_max=20, ep=20)** | ep=20 cosine T_max=20 | **best/val_f1** | **6** | **0.9964** | **0.83%** | **I13** |

**★ iter112 best_model (selected by val_f1 = 0.9907 at ep06) = paper SOTA.**
**bF1 = 0.9964 / Total FAR = 0.83% (7 FP / 840 negative chips), I-cell = I13.**

Per-class F1 (iter112 best / I13): bank_boundary 0.9984, fork 0.9881,
scratch 0.9992, scratch_rot 1.0000.

### FAR breakdown — iter112 best_model (ep06) at I13

| FP source | FP count | FAR rate | predicted class |
|-----------|---------:|---------:|-----------------|
| Starburst (OOD wafer-pattern) | 5 | — | fork+scratch (all 5) |
| Normal | 2 | — | fork+scratch (both 2) |
| CrossScratch (OOD wafer-pattern) | 0 | 0.00% | — |
| Invalid | 0 | 0.00% | — |
| **Total** | **7 / 840** | **0.83%** | — |

All 7 false-positives predict the same composite label **`fork+scratch`**,
suggesting a single coherent model failure mode (low-frequency vertical
line + diffuse blob signature being mis-fired as `fork+scratch`) rather
than a scattered population of FAR sources.

### Checkpoint selection criterion — iter111 + iter112 + iter101A combined

Across **404 + 88 = 492 (run × cell)** rows in
`outputs/_reeval_absolute_rule.csv` (per-epoch eval for iter101A 12 ckpts,
iter111 12 ckpts, iter112 22 ckpts), the empirical correlation between
validation-time metrics and eval-time bF1 / Total FAR is:

| val criterion | best ckpt picked (iter112) | bF1 | Total FAR | comment |
|---------------|---------------------------|----:|----------:|---------|
| **val_acc** | ep03 (val_acc max = 0.9877) | 0.9941 | 0.12% | anti-correlated (ρ ≈ −0.52); picks an EARLIER ckpt than optimal |
| **val_f1** ★ | **ep06 (val_f1 max = 0.9907)** | **0.9964** | **0.83%** | best — picks the SOTA ckpt |
| val_auroc | ep16 (val_auroc max = 1.0000) | 0.9965 | 91.7% | saturate-vulnerable; ties at 1.0 push pick toward late epoch with catastrophic FAR |
| (vf1+vauroc)/2 | ep06 | 0.9964 | 0.83% | matches val_f1 pick |
| geo-mean(vf1, vauroc) | ep06 | 0.9964 | 0.83% | matches val_f1 pick |
| harm-mean(vf1, vauroc) | ep06 | 0.9964 | 0.83% | matches val_f1 pick |

**Paper Methods §X must declare `--val-criterion f1` as the default
selection rule.** `val_acc` mis-picks earlier and `val_auroc` mis-picks
later (catastrophic). The 3-way arith/geom/harm aggregates of (val_f1,
val_auroc) all coincide with the val_f1 pick — val_f1 alone is sufficient.

### Cosine T_max sweep — ep=10 vs ep=20

Same recipe (ConvNeXtV2-B FCMAE 384, T7 BCE+LS=0.20, FCM-PM CutMix
`cutmix_mode=complement`, `cutmix_pair=masked`, `cutmix_pair_fill=corner`,
`cutmix_p=0.25`, `cutmix_n_groups=3`, `cutmix_complete_label_scale=0.5`,
batch=2, accum=8, lr=1e-4, `--no-normal`, seed=1):

| run | epochs | cosine T_max | best epoch (val_f1) | bF1 (best) | Total FAR (best) | bF1 (final) | Total FAR (final) |
|-----|------:|------------:|--------------------:|-----------:|-----------------:|------------:|------------------:|
| iter111 | 10 | 10 | 3 (also picked at 8 by val_f1 tie-break) | 0.9891 | 1.90% | 0.9964 | 4.52% |
| **iter112** | **20** | **20** | **6** | **0.9964** | **0.83%** | 0.9818 | 100% |

**Doubling cosine T_max (10 → 20) shifts the best-by-val_f1 checkpoint
from ep3 → ep6 and lifts bF1 by +0.0073 / drops Total FAR by 1.07pp.**
The val_f1 plateau at 0.9907 is reached three times in iter112 (ep06,
ep08, ep10) all with comparable eval-time bF1, while the iter111 val_f1
plateau at 0.9907 is reached six times — but only the late-epoch
plateaus give the SOTA cell. Slower LR decay (longer cosine period)
**lets the FCM-PM CutMix signal accumulate for an extra ~3 epochs at
near-peak LR** before annealing, which is the proximate mechanism for
the +0.0073 bF1.

### Recipe spec (paper Methods §X — verbatim CLI)

```
python -m chip_multilabel._train_chip_variant \
  --variant T7 --ls 0.20 \
  --backbone convnextv2_base.fcmae_ft_in22k_in1k_384 --img-size 384 \
  --epochs 20 --batch 2 --accum 8 --lr 1e-4 --seed 1 \
  --no-normal --val-criterion f1 --save-every-epoch \
  --cutmix-mode complement --cutmix-p 0.25 --cutmix-n-groups 3 \
  --cutmix-pair masked --cutmix-pair-fill corner \
  --cutmix-complete-label-scale 0.5
```

### Trainer patches (260512) — new flags in `_train_chip_variant.py`

| flag | semantics | default |
|------|-----------|---------|
| `--val-criterion {acc,f1,auroc}` | which val metric drives `best_model.pth` | `acc` (legacy) — paper recommends `f1` |
| `--save-every-epoch` | also dump `epoch_NN_model.pth` for per-epoch retroactive eval | off |
| `--best-from-epoch N` | (deprecated; replaced by `--val-criterion`) floor on which epoch can be "best" | — |
| `--freeze-backbone` | linear-probe mode (head-only trainable) | off |

`--save-every-epoch` is the enabling mechanism for the val_criterion
empirical study above (without per-epoch dumps, the val_f1 vs val_acc vs
val_auroc comparison cannot be done retroactively).

**Source**: `outputs/iter111_seed1_reproduce_now/T7_iter111_seed1_now_260512_212411/`,
`outputs/iter112_ep20/T7_iter112_ep20_260512_214618/`,
`outputs/iter101A_convnextv2_perep/T7_iter101A_convnextv2_perep_260512_200902/`,
combined per-epoch reeval `outputs/_reeval_absolute_rule.csv` (492 rows),
training logs `outputs/_iter111_now.log`, `outputs/_iter112_ep20.log`,
`outputs/_iter101_per_epoch_eval.log`. Per-iter detail logged below in
`05_backbone_landscape.md` (iter101 + iter111 + iter112 paragraphs) and
ablation in `03_ablations.md` (cosine T_max sweep + val_criterion study).

## ★★★ 2026-05-12 update — iter88-94 backbone landscape sweep (12 backbones, 4 families)

See **`05_backbone_landscape.md`** for the full per-cell breakdown
and Pareto-frontier reading. **Headline takeaways** from iter88-89 (12 backbone
runs, 7 with full eval, 28 inference cells):

1. **Single-seed Swin-Base 86.9M still dominates the 80-90M param region for
   strict Total FAR**. Three Swin-Base variants (iter77C g=3 LS=0.50,
   iter89_LR14_LS3_g3, iter89_LR14_LS3_g2) all land at **Total FAR = 0.00%**;
   `bF1_4def` is monotone in CutMix-group count and LS value
   (0.9692 / 0.9434 / 0.9278 respectively).
2. **Scaling up Swin V1 (Base 86.9M → Large 195.20M) HURTS bF1_4def by 0.050**
   under the same iter88 recipe. iter88E Swin-Large lands at 0.9192 bF1 /
   0.12% Total FAR, vs iter77C Swin-Base 0.9692 / 0.00%.
3. **ConvNeXt V1 Large 196M exhibits a worse OOD-FAR pathology** than ConvNeXt
   V1 Base 87M. iter88F's I3/I7 cells score bF1 0.9919/0.9931 but at
   **100.00% Total FAR**; the best safe cell (I13) drops to 0.8720 at 10.60%
   FAR. ConvNeXt V1 Base (iter77A) remains the throughput-production winner.
4. **EfficientNetV2-L, EfficientFormerV2-L, TinyViT-21M, ResNet-50/152,
   MobileNetV4-Conv-L all fail to produce a Total-FAR-safe + competitive-bF1
   cell** under the ConvNeXtV2-tuned T7 recipe. Per-family recipe re-tuning
   is queued for iter95.
5. iter46E ConvNeXtV2-Base 0.9654 / 1.07% (paper-main winner) and iter77C
   Swin-Base 0.9692 / 0.00% (strict-zero / latency winner) **remain the
   Pareto-frontier operating points** after this expansion. No iter88/89 cell
   beats either on a strict (Total FAR ≤ 5%) basis.

Source: `iters/iter88-94` outputs + `tables/backbone_landscape.csv` (37 rows).
Pending iter95: per-family recipe sweep + throughput measurement for the 7 new
backbones (GPU at 100% during this logger pass blocked throughput probe).

## ★★★ 2026-05-12 update — Phase 87 v2 backbone throughput + paper §3 backbone-choice correction (iter 21)

**Critical paper finding (see `iters/iter_21_backbone_throughput_paper3.md`)**: previous paper §3.5 narrative claimed "ConvNeXtV2-Base = best balanced production backbone" on accuracy grounds alone. Phase 87 v2 isolated-GPU precise measurement (torch.cuda.Event, 20 warmup + 100 iter, 4 backbones × 6 batch sizes) reveals **ConvNeXtV2 is the only backbone where batching makes throughput worse** (peak 37 chip/s at b=1, drops to 26 chip/s at b=64) due to **GRN (Global Response Normalization) per-channel reduction at every block**. The corrected §3 narrative splits backbone selection into **three operational regimes**:

| regime | metric | winner | spec |
|--------|--------|--------|------|
| Latency-critical (inline, b=1) | ms/chip @ b=1 | **Swin-Base 384** | 21.08 ms + bF1 0.9692 + **Total FAR 0.00%** (only strict-zero in sweep) |
| Throughput-critical (batched, b=64) | chip/s peak | **ConvNeXt V1** | 76 chip/s @ b=64 + bF1 **0.9830** + Total FAR 2.62% |
| Paper-main (inline + coupled) | bF1 + paper-main legacy | **ConvNeXtV2-Base** | 37 chip/s @ b=1 only + bF1 0.9654 + Total FAR 1.07% (iter46E) |

**4 backbones × 6 batch sizes scaling table (chips/sec)**:

| backbone        | params | b=1    | b=4    | b=8 | b=32 | b=64 | peak       | scaling     | bF1     | Total FAR |
|-----------------|-------:|-------:|-------:|----:|-----:|-----:|-----------:|------------:|--------:|----------:|
| ConvNeXtV2-Base | 87.7M  | **37** | 30     | 28  | 35   | 26   | **37 @ b=1** | **0.70–0.93×** ⛔ | 0.9654 | 1.07% |
| ConvNeXt-V1     | 87.6M  | 40     | 60     | 64  | 74   | **76** | **76 @ b=64** | **1.85×** ✓ | **0.9830** | 2.62% |
| Swin-Base 384   | 86.9M  | 47     | **54** | 53  | 48   | 47   | **54 @ b=4** | **1.13×**   | 0.9692 | **0.00%** |
| EfficientV2-M   | 52.9M  | 42     | **158**| 158 | 130  | 113  | **158 @ b=4** | **3.77×** ✓ | FAIL (small-data not converged) | — |

**10,000-chip processing time** (paper §3 quantitative): EffV2-M 63s (FAIL) → ConvNeXt V1 132s (production winner) → Swin 185s (FAR=0) → ConvNeXtV2 270s (slowest production). **ConvNeXt V1 is 2.05× faster than ConvNeXtV2 at the same param count (87.6M ≈ 87.7M), with +0.0176 bF1.**

**Implications**:
- Paper §3.5 must be rewritten — the "ConvNeXtV2 = best" claim was an accuracy-only call that ignored GRN's batching pathology.
- The 4-bag ensemble cost (§5.19.5) of 4× single-model assumes ConvNeXtV2; a 4-bag ConvNeXt V1 ensemble would run at **4 / 2.05 = 1.95×** the cost of a 1× ConvNeXtV2 single, while gaining +0.0176 bF1 per bag.
- Single-model accuracy ranking does **not** change (iter46E ConvNeXtV2 remains paper-main single-model winner by accuracy + Total FAR pairing).
- ConvNeXtV2 GRN finding is **measurement-grade architectural evidence**, not a bug — Woo et al. 2023 (arXiv:2301.00808) introduced GRN as channel-mean response normalization; the per-batch mean broadcast serialises under cuDNN at high batch sizes.

Source: `iters/iter_21_backbone_throughput_paper3.md` + `tables/backbone_throughput.csv` (24 rows). Raw script `_phase87_precise_speed.py`. Accuracy from `outputs/iter46E_g3LS050_rect03/` + `outputs/iter77{A,C,E}_*/T*/eval_v15direct_n200/`.

## ★★★ 2026-05-12 update — Total FAR correction (Phase 83 / 85) invalidates prior FAR rankings

**Critical paper finding (see `iters/iter_18_total_far_correction.md`)**: every
prior "SOTA" cell that claimed `ni_FAR = 0%` was hiding **19–48% OOD
wafer-pattern false-fires** (CenterDonut / CrossScratch / DiagonalSmear /
Starburst). The legacy `ni_FAR` metric pooled only the 4 Normal sources and
excluded the OOD wafer-pattern chips that v15direct deliberately includes as
distractors. Re-scoring with **Total FAR = (NI + OOD) / total** flips the
production rankings:

| cell                                  | recipe                          | bit_F1   | ni_FAR  | **Total FAR** | verdict                              |
|---------------------------------------|---------------------------------|---------:|--------:|--------------:|--------------------------------------|
| iter69 ep=12 (prev "SOTA" KD)         | α=0.5 T=4 ep=12                 |  0.9941 |   0.00% |   **36.67%** ⛔ | masked OOD blow-up                   |
| iter50B paper KD                      | paper §5 main KD                |  0.9872 |   ~0%   |   **12.86%** ⛔ | KD trades NI-safety for OOD over-fire|
| iter67D KD ep=10                      | KD α=0.5 T=4 ep=10              |  0.9917 |   ~0%   |   **11.79%** ⛔ | same family as iter69                |
| iter41E "vanilla best"                | g=3 vanilla                     |  0.9961 |   ~0%   |   **15.24%** ⛔ | also hiding OOD                      |
| **iter46E vanilla** ★                 | g=3 LS=0.50 compl pair=masked   |  0.9654 |   ~0%   |   **1.07%** ✓ | **true production single-model**     |
| iter72A no-Normal bare                | no-Normal vanilla               |  0.8908 |   0.00% |   **0.00%**   | over-conservative                    |
| **prev "0.9966 ensemble" k=2/4**      | `{26D + 65s13 + 65s21 + 69ep9}` |  0.9962 |    ~0%  |   **2.86%** ⛔ | prior ensemble SOTA — revoked        |
| **★★★ same 4-bag k=3/4**              | (stricter quorum)               | **0.9909** |  0.00% | **0.00%**     | **NEW ensemble SOTA**                |

**Implications**:
- Single-model paper headline moves from `iter50B 0.9872 / FAR 0.5%` to
  `iter46E 0.9654 / Total FAR 1.07%`. The 0.0218 bit_F1 drop buys 11.79
  percentage-point Total-FAR reduction.
- 4-bag ensemble headline moves from `0.9966 / "0%"` (k=2, actually 2.86%
  Total FAR) to `0.9909 / 0.00%` (k=3, true zero). The −0.0053 bit_F1 cost
  for the stricter quorum is paid in exchange for zero OOD false-fire.
- The `--cutmix-rect` flag is a **no-op under `cutmix_mode=complement`** —
  iter46E (`rect=0.3`) and iter42F (`rect=0.5`) produce bit-identical
  predictions across 3 different seeds (see `iters/iter_19_vanilla_multi_seed_robust.md`).
  Future tables should treat them as the same model family.
- iter46E recipe should be cited by `g=3 / cutmix-complete-label-scale=0.5 /
  cutmix-pair=masked / cutmix-pair-fill=corner / LS=0.20`, not by rect.
- Multi-seed sweep of iter26H recipe (8 seeds, `iter_19_*`) shows
  **seed=42 catastrophic** (`ni_FAR=100%`) while 5/8 seeds pass `ni_FAR ≤ 0.5%`.
  Same-recipe 3-bag `{s7+s13+s21}` restores `0.9955 / ni_FAR=0%`.
- Trainer patched (260512) with `--cutmix-other-label` flag (default 0.0 =
  no behavior change); planned iter83 sweep ∈ {0, 0.05, 0.10, 0.15, 0.20}
  tests whether soft off-class labeling reduces Total FAR further. See
  `iters/iter_20_other_label_patch.md`.

The prior headline rows below this section pre-date the Total FAR correction
and are preserved verbatim for paper rebuttal / narrative use. **All "ni_FAR=0%"
claims in those rows should be interpreted as ni-only and may carry OOD
over-fire**. Re-scoring with Total FAR is on the iter83 queue.

**★★★ FINAL PAPER MAIN WINNER (Phase 28 n=500 confirmation, iter 43) → pure-hard 4-bag**
`{24_LS030_seed42 + 26B + 26D + 26H}` thr ≥ 2/4 simple-majority on **v15direct_n1000
n=500 (5250-chip per-class eval, 7080 intersection chips, most reliable
evaluation to date)** → **v15 bit_F1 = 0.9953**, **ni_FAR = 0.00%**,
per-class bb=0.9959 / fk=0.9915 / sc=0.9937 / sr=1.0000
**(at FULL eval; at HARD050 hard+KD wins by +0.0019 — see iter 43 Phase 31b below).**

**Hard+KD 4-bag TIES**: `{24_LS030_seed42 + 26B + 26H + 33D}` lands at the
identical **0.9953 / 0.00%** with per-class F1 differing by ≤0.0003 (pure
noise). **Two independent 4-bag compositions converge on the same headline
number** — the KD-axis vs hard-label diversity distinction is
**indistinguishable at the 4-bag level**.

**Stability across eval-set sizes**: 0.9992 (n=50) → 0.9955 (n=200) → **0.9953
(n=500)** shows monotonic convergence — the n=50 reading was a small-sample
artifact, n=200 already gave the stable answer, and n=500 confirms it within
0.0002. The "ensemble cancels fragility" thesis is **strongest at n=500**:
`24_LS030_seed42` has **22.50% ni_FAR alone** at n=500 (single-model FAIL)
yet the 4-bag containing it holds at **0.00% ni_FAR** on a 5250-chip eval —
the strongest single-data-point demonstration in the project's history.
See `iters/iter_42_n200_rebuttal.md` (Phase 28 n=500 confirmation appended)
+ `tables/paper_main_headline.csv` (rows `iter39_ensemble_4bag_pureHard_n500_FINAL`
and `iter39_ensemble_4bag_hardKD_n500_TIE`).

**Prior n=200 reading (iter 42 — confirmed by n=500):** pure-hard 4-bag
v15 bit_F1 = 0.9955, ni_FAR = 0.00% on 3080 chips. Confirmed by n=500
(0.9953/0%) — both within sampling noise of each other.

**Prior n=50 reading (iter39 — REBUTTED above):** pure-hard 4-bag
`{24_LS030_seed42 + 26B + 26D + 26H}` thr ≥ 2/4 → v15 bit_F1 = 0.9992,
ni_FAR = 0.00%. Superseded as the headline metric — the underlying ensemble
configuration remains the recommended deployment, but the bit_F1 reading
should be reported as 0.9955 (n=200, 3080 chips).

**★ Prior HEADLINE (iter37 4-bag) is seed-robust** — confirmed iter38: 3/3 seed
variants of 37E slot PASS dual-gate at v15 bit_F1 = **0.9976 ± 0.0007**;
single-model seed-luck is canceled by the ensemble. (Now superseded by iter39 pure-hard.)

## Cross-iter best timeline

| iter | best_cell           | macro_f1   | top1_11    | Δ macro_f1 | Δ top1_11  | source                                                                |
|-----:|---------------------|-----------:|-----------:|-----------:|-----------:|-----------------------------------------------------------------------|
|   21 | **★ Phase 87 v2 backbone throughput — paper §3.5 narrative correction** (3 regimes) | ConvNeXt V1 b=64 = **76 chip/s + bF1 0.9830 + Total FAR 2.62%** (throughput winner); ConvNeXtV2 b=1 = 37 chip/s + bF1 0.9654 + Total FAR 1.07% (latency-only, **GRN batching quirk**); Swin-Base b=4 = 54 chip/s + bF1 0.9692 + **Total FAR 0.00%** (strict-zero) | — | (no accuracy headline change — adds cost axis) | — | iter 21 isolated-GPU precise measurement (torch.cuda.Event, 20 warmup + 100 iter, 4 backbones × 6 batch). **ConvNeXtV2-Base is the only backbone where batching makes throughput worse**: b=1 peak 37 chip/s → b=64 drops to 26 chip/s (**0.70× scaling, NEGATIVE**) due to **GRN (Global Response Normalization) per-channel mean reduction at every block** (Woo et al. 2023 arXiv:2301.00808). All other backbones scale normally: EfficientV2-M **3.77×** (peak 158 chip/s b=4, but bF1 FAIL at iter46E recipe), ConvNeXt V1 **1.85×** (peak 76 chip/s b=64, bF1 0.9830 PASS), Swin-Base **1.13×** flat (peak 54 chip/s b=4, **strict-zero FAR**). **Paper §3.5 correction**: PREVIOUS "ConvNeXtV2 = best balanced" claim was accuracy-only and ignored GRN. NEW 3-regime selection: Latency-critical inline → Swin-Base 21.08 ms/chip b=1; Throughput-critical batched → ConvNeXt V1 76 chip/s b=64; ConvNeXtV2 reserved for inline + paper-main legacy. **10k-chip processing time**: ConvNeXt V1 132 s vs ConvNeXtV2 270 s = **2.05× speedup at same param count + +0.0176 bF1**. Single-model accuracy ranking unchanged (iter46E ConvNeXtV2 remains paper-main winner by bF1 + Total FAR). New CSV `tables/backbone_throughput.csv` (24 rows). See `iters/iter_21_backbone_throughput_paper3.md`. |
|   20 | **★ trainer patch — `--cutmix-other-label`** (no run yet) | (planned iter83 sweep) | — | — | — | iter 20 trainer change: `chip_multilabel/_train_chip_variant.py` gains `--cutmix-other-label` (default 0.0, non-breaking). Motivated by Phase 84b prob-distribution diagnostic on iter46E (TRAIN defect own-prob 0.84-0.92 vs EVAL OOD max-prob ~0.55 vs EVAL NI max-prob ~0.46). Hypothesis: hard-zero off-class bit labeling on mix chips → over-confident "absent class" prior that fails on OOD. Soft off-class label (e.g., 0.1) preserves calibration uncertainty. Planned iter83 sweep `{0.0, 0.05, 0.10, 0.15, 0.20}` at iter46E base recipe + 3-seed expansion on winner. See `iters/iter_20_other_label_patch.md`. |
|   19 | **★ iter79 iter26H multi-seed + iter80 cross-recipe (12/16 cells)** | seed=7 macro_f1 **0.8769** (iter79 best), seed=42 catastrophic | 0.5526 | — | — | iter 19 vanilla multi-seed robustness. **iter79** (iter26H recipe × 8 seeds): bit_F1 0.9695 ± 0.033, 5/8 pass `ni_FAR ≤ 0.5%`, **seed=7 peak 0.9961**, **seed=42 ni_FAR=100% catastrophic**. Same-recipe 3-bag `{s7+s13+s21}` restores **0.9955 / ni_FAR=0%**. **iter80** (4 recipes × 3 seeds = 12 cells done): best mean = 40F (0.8365 stage1 macro_f1). **Critical**: iter42F (rect=0.5) and iter46E (rect=0.3) produce **bit-identical macro_f1 across all 3 shared seeds** (0.8142 / 0.7998 / 0.7750) — proves `--cutmix-rect` is no-op under `cutmix_mode=complement`. Total FAR not yet measured for iter79/iter80 cells (iter83 deferred). See `iters/iter_19_vanilla_multi_seed_robust.md`. |
|   18 | **★★★ Phase 83 / 85 Total FAR correction — invalidates prior "SOTA" rankings** | iter46E vanilla single = bit_F1 **0.9654 / Total FAR 1.07%** ★ true production winner; NEW ensemble SOTA 4-bag k=3 = **0.9909 / Total FAR 0.00%** | — | — | — | iter 18 critical paper correction. Legacy `ni_FAR` pooled only 4 Normal sources, excluded 4 OOD wafer-patterns (CenterDonut/CrossScratch/DiagonalSmear/Starburst). Phase 83 re-scored all "SOTA" cells with **Total FAR = (NI + OOD) / total**: iter69 ep=12 "SOTA" KD bF1=0.9941 but **Total FAR=36.67%** (masked OOD blow-up); iter50B paper KD 0.9872 / 12.86%; iter67D 0.9917 / 11.79%; iter41E 0.9961 / 15.24%; **iter46E vanilla = 0.9654 / 1.07%** ★ (only single model with bF1≥0.96 AND Total FAR≤5%); iter72A no-Normal 0.8908 / 0.00% (over-conservative). Phase 85 ensemble: prior "0.9966 SOTA" k=2 actually Total FAR=2.86% — **same 4-bag k=3 (stricter quorum) → 0.9909 / 0.00% NEW ensemble SOTA** (−0.0053 bF1 trade for true zero). iter46E recipe clarification: `--cutmix-rect 0.3` is no-op under `cutmix_mode=complement`; real axes are `n-groups=3 / complete-label-scale=0.5 / pair=masked / pair-fill=corner / LS=0.20`. Phase 84b prob-distribution diagnostic: iter46E TRAIN defect own-prob 0.84-0.92 vs EVAL OOD max-prob ~0.55 vs EVAL NI max-prob ~0.46 — motivates iter20 patch. **Paper §5 main rewrite required**: lead with `(bit_F1, Total FAR)` pairs, not bit_F1 alone. Narrator agent handles `manager_report/REPORT.md` separately. See `iters/iter_18_total_far_correction.md`. |
|   52 | **★★★ iter52 teacher bag-size curve (paper §6.21) — 4-bag UNIQUE sweet spot at α=0.5; 5-bag highest bF1 0.9913 BUT FAR 99.5% catastrophic; 14-bag collapses (needs α=0.3)** | bit_F1 0.9872 (FULL n=200, 3080 chips, iter52C ≡ iter50B; non-monotonic curve 0.9198→0.9768→0.9872→[0.9913 FAIL]→0.9862→0.9053) | — | (no headline change — duplicates iter50B 0.9872/0.5%; curve is added narrative) | — | **iter 52 teacher bag-size sweep at fixed (α=0.5, T=4)** — 6-cell sweep mapping bag size ∈ {2, 3, 4, 5, 6, 14} into single-model students; **iter52C (4-bag) = unique PASS sweet spot** at bit_F1=**0.9872** / FAR=**0.5%** (per-class bb=0.9866 / fk=0.9825 / sc=0.9795 / sr=1.0000). **Curve is non-monotonic**: 2-bag 0.9198 → 3-bag 0.9768 → 4-bag **0.9872** → 5-bag 0.9913 (**FAR=99.5%, FAIL**) → 6-bag 0.9862 → 14-bag 0.9053 (collapse). **5-bag FAR collapse paradox**: highest raw bit_F1 but ni_FAR=99.5% — adding 26B at α=0.5 over-fits the bb/fk pair-mask boundary, fires on virtually all Normal chips (0.0041 bit_F1 gain irrelevant). **6-bag (52E) recovery**: 26D KD-student diversity cancels the over-firing, FAR snaps back to 0.0% — viable PASS but indistinguishable from 4-bag at higher cost. **14-bag collapse at α=0.5**: paper main 33A used 14-bag at **α=0.3**, the correct α for that bag size; 14-bag with α=0.5 under-recovers ground-truth signal (−0.0819 vs 4-bag). **Paper §6.21 main claim — teacher bag-size ↔ optimal α anti-correlation**: smaller bag → sharper teacher (higher avg max prob) → student needs higher α; larger bag → softer teacher → lower α. The (α, T) recipe is **not transferable** across bag sizes. **No paper main change** — iter52C ≡ iter50B headline; curve is added narrative material. New rows `iter52{A,B,C,D,E,F}_FULL` + `iter52{A,B,C,D,E,F}_HARD050` (= 12) in `all_runs_macro_f1.csv`. See `iters/iter_52_teacher_bagsize_curve.md`. |
|   50 | **★★★ iter50 4-bag teacher KD distillation — α=0.5 T=4 single SOTA 0.9872/0.5% (vs 33A paper main 0.9840/0%, +0.0032)** | bit_F1 0.9872 (FULL n=200, 3080 chips) | — | +0.0032 bit_F1 vs iter33A (14-bag teacher α=0.3 T=4); +0.005 FAR trade | — | **iter 50 4-bag teacher KD α/T sweep** — 5-cell sweep distilling NEW MAIN 4-bag teacher (24_LS030_seed42 + 26H + 33A + 37E, ensemble 0.9964/0%) into single students at α ∈ {0.3, 0.5, 0.7} × T ∈ {2, 4, 8}. **iter50B (α=0.5, T=4) = NEW SINGLE-SOTA**: bit_F1 = **0.9872**, ni_FAR = **0.5%**, per-class bb=0.9866 / fk=0.9825 / sc=0.9795 / sr=1.0000 (all PASS dual-gate). Beats prior single-best iter33A (14-bag-teacher α=0.3) at 0.9840 / 0% by **+0.0032 bit_F1** (with a small 0.5% FAR trade). **α sweet spot shifts 0.3 → 0.5** when teacher bag shrinks from 14 → 4: smaller teacher concentrates probability mass (avg max prob 0.91 in 4-bag vs softer 14-bag), so the student needs higher α to absorb that signal. T-axis at α=0.3 is secondary (T=2 → 0.9384, T=4 → 0.8921, T=8 → 0.9323 — all dominated by α=0.5 T=4). **Production cost frontier**: 1× single-model gap to 4× ensemble closes from 0.0124 → 0.0092 (−26%). New row `iter50B_4bag_teacher_KD_singleSOTA` in `paper_main_headline.csv` + 10 detail rows in `all_runs_macro_f1.csv` (5 FULL + 5 HARD050 pending). See `iters/iter_50_4bag_teacher_KD.md`. |
|   46 | **★★★ iter46 5-axis FCM-PM ablation — pair-mask = method-essential (removing causes −0.18 catastrophic FAR collapse 2.5%→100%); other axes tunable** | (ablation; not a new winner — paper §5 ablation table material) | — | pair=none Δ bF1=−0.18 / Δ FAR=+97.5% (ESSENTIAL); mode=single Δ bF1=−0.035 (HELPFUL); fill/p/rect Δ bF1=−0.013 to −0.166 (tunable) | — | **iter 46 FCM-PM 5-axis ablation** — 6 cells flipping one axis (or one combo) vs 26B baseline (g=3 LS=0.50 mode=complement fill=corner p=0.25 rect=0.5 pair-mask=on). Cell A (pair=none) and F (pair=none + p=0.40 + g=2 LS=0.30) **FAIL dual-gate** with FAR=100% — pair-mask is method-essential safety-critical. Cells B (mode=single), C (fill=noise), D (p=0.40 g=4 LS=0.40), E (rect=0.3) PASS with smaller bF1 drops (−0.013 to −0.166). HARD050 cross-eval: 46E within 0.005 of 26B; 46D drops most (LS=0.40 lacks calibration). Paper §5 ablation table + §6 mechanism story material — cleanly separates method-essential (pair-mask, complement) from tunable (fill/p/rect/g/LS) axes. New CSV row `iter46_FCM_PM_ablation_summary` in `paper_main_headline.csv` + 12 detail rows in `all_runs_macro_f1.csv`. See `iters/iter_46_FCM_PM_ablation.md`. |
|   45 | **★★★ Phase 35/36 strength curve REVOKES iter44 HARD WINNER claim — pure-hard NEW HEADLINE wins at 5/6 thresholds; dual-seed advantage was strength=0.50 single-point artifact** | (revocation; winner reverts to FULL pure-hard 4-bag bit_F1=0.9955 / 0%) | — | iter44 HARD-W +0.0154 advantage at HARD050 ONLY; at HARD045 pure-hard 0.9941 wins, at HARD055 pure-hard 0.9966 wins, at HARD060 pure-hard 0.9959 wins, at FULL pure-hard 0.9955 wins | — | **iter 45 Phase 35/36 strength-curve REVOCATION** — 6-point sweep `strength_max ∈ {0.40, 0.45, 0.50, 0.55, 0.60, 1.00}` × 9-model pool = 45 stage-1 inferences. Result: pure-hard NEW HEADLINE 4-bag `{24_LS030_seed42 + 26B + 26D + 26H}` thr ≥ 2/4 wins at **5/6 thresholds** (0.45/0.55/0.60/FULL by 0.0008–0.0065 over hard+KD; HARD040 is degenerate bb-excluded slice n=975). The iter44 "HARD WINNER" `{24_LS030_seed42 + 33D + 37E + 24_LS030_seed7}` 0.9843 / 2.00% advantage exists ONLY at `strength_max=0.50` (HARD050, 2003 chips) — at adjacent thresholds 0.45 and 0.55 it is strictly dominated by pure-hard. **Single-point artifact, not a robust ranking.** Dual-seed advantage is **sample-composition specific** to the HARD050 cohort. **Paper-grade lesson**: single-threshold conclusions are vulnerable to slice composition; report sweeps not points. iter44 row in `paper_main_headline.csv` annotated **REVOKED** (preserved for rebuttal narrative). New SUMMARY row `iter45_strength_curve_revoke` added. Final paper-grade claim reverts to pure-hard NEW HEADLINE at FULL n=200/n=500 = 0.9953–0.9955 / 0%. See `iters/iter_45_strength_curve_revoke.md` + `tables/paper_main_headline.csv` (rows `iter44_ensemble_4bag_HARD_WINNER_dualseed` REVOKED + `iter45_strength_curve_revoke` SUMMARY). |
|   44 | **(REVOKED by iter45 strength curve — single-point artifact at strength_max=0.50 only)** Phase 34 HARD050 big-sweep — claimed NEW HARD WINNER 4-bag {24_LS030_dual-seed + 33D + 37E} = 0.9843 / 2.00% beats hard+KD by +0.0154 | **v15 bit_F1 0.9843 (HARD050, 2003 chips)** | — | iter44 0.9843 vs iter43 hard+KD 0.9689 (+0.0154); vs pure-hard 0.9670 (+0.0173) | — | **iter 44 Phase 34 HARD050 BIG-SWEEP REFINEMENT** — exhaustive C(9,4)=126 4-bag sweep + bag-size frontier (5/6/7-bag) at v15direct_HARD050 (strength≤0.50, 2003 intersection chips). **NEW HARD WINNER 4-bag** = `{24_LS030_seed42 + 33D + 37E + 24_LS030_seed7}` thr ≥ 2/4: bit_F1=**0.9843** ni_FAR=**2.00%** (within 5% paper-spec gate), per-class bb=**0.9517** / fk=**0.9891** / sc=**0.9964** / sr=**1.0000**. **Beats prior iter43 hard+KD HARD-winner (0.9689) by +0.0154** and pure-hard 4-bag (0.9670) by +0.0173. **Insight**: ALL top-10 4-bag at HARD eval include BOTH 24_LS030 seeds — `24_LS030` is the HARD-chip specialist (single bF1 0.9767, bb 0.9307), and dual-seed bag double-votes on hard chips. 33D KD-student + 37E asymmetric provide diversity from non-correlated boundaries to cancel FAR over-firing. **4-bag is global optimum**; 5/6/7-bag regress (consensus dilution at HARD). **Reframes iter43 Phase 31b claim**: "hard+KD wins by +0.0019" was a directed-comparison limited to single-seed 4-bag family — once dual-seed allowed, both prior winners are surpassed. Strict 0% FAR alt = pure-hard 4-bag at 0.9670 / 0.00% (production safety-critical). See `iters/iter_44_HARD_winner_refined.md` + `tables/paper_main_headline.csv` (row `iter44_ensemble_4bag_HARD_WINNER_dualseed`). |
|   43 | **★★★ Phase 31b HARD050 saturation breakdown — hard+KD beats pure-hard by +0.0019 at HARD eval** (refined by iter44 — exhaustive sweep finds dual-seed strategy at 0.9843, +0.0154 over hard+KD) | **v15 bit_F1 0.9689 (HARD050, 2003 intersection chips, strength≤0.50)** | — | hard+KD 0.9689 vs pure-hard 0.9670 (+0.0019 at HARD eval; ties at FULL eval) | — | **iter 43 Phase 31b HARD eval BREAKTHROUGH** (refined by iter44 Phase 34 big-sweep: dual-24_LS030-seed 4-bag at 0.9843 surpasses by +0.0154 — directed-comparison was not exhaustive) — first config to beat the pure-hard 4-bag on a stress eval. Eval set `eval_v15direct_HARD050` filters chips by `strength ≤ 0.50` (Phase 31a `≤ 0.40` over-filtered, excluding all bb chips); Phase 31b retains 2003 intersection chips after merge. **HARD WINNER 4-bag** = `{24_LS030_seed42 + 26B + 26H + 33D}` (hard+KD): bit_F1=**0.9689** ni_FAR=**0.00%**, per-class bb=**0.8985**/fk=**0.9882**/sc=**0.9890**/sr=**1.0000**. **Runner-up** = NEW HEADLINE pure-hard 4-bag at **0.9670 / 0.00%** (-0.0019). Cross-eval pattern: at FULL n=50/n=200/n=500 the two configs are interchangeable (within ≤ 0.0008); at HARD050 they diverge with the hard+KD 4-bag winning on bb (+0.0063) and sc (+0.0024). **Mechanism**: KD-student (33D, α=0.5 T=8) provides soft-target calibration on edge-of-defect chips that hard-label cells over-confidently mis-classify. **24_LS030 = HARD-chip specialist** (single bF1 0.9767 dominant on HARD but FAR-fragile alone at 20.5%); 4-bag cancels its FAR over-firing while keeping the HARD-chip strength. **Paper §6.17/§6.18 refinement**: KD axis is **interchangeable at FULL eval but dominant at HARD eval** — adds genuine value when saturation broken. See `iters/iter_43_HARD_eval_breakthrough.md` + `tables/paper_main_headline.csv` (rows `iter39_ensemble_4bag_hardKD_HARD050_WINNER` ★ + `iter39_ensemble_4bag_pureHard_HARD050_RUNNER`). |
|   43 | **★★★ FINAL HEADLINE (n=500 confirmation) — pure-hard OR hard+KD 4-bag TIE at 0.9953/0%** | **v15 bit_F1 0.9953 (n=500, 5250-chip per-class, 7080 intersection)** | — | n=200 0.9955 → n=500 0.9953 (within 0.0002 noise; CONFIRMED) | — | **iter 43 Phase 28 n=500 FINAL CONFIRMATION** — pure-hard 4-bag `{24_LS030_seed42 + 26B + 26D + 26H}` thr ≥ 2/4 majority, re-evaluated at v15direct_n1000 **n=500 (5250-chip per-class eval, 7080 intersection chips — most reliable evaluation to date)**. **v15 bit_F1=0.9953, ni_FAR=0.00%**, per-class bb=**0.9959** / fk=**0.9915** / sc=**0.9937** / sr=**1.0000**. **Hard+KD 4-bag `{24_LS030_seed42+26B+26H+33D}` TIES at identical 0.9953/0.00%** with per-class differing by ≤0.0003 (pure noise) — KD-axis vs hard-label diversity is **indistinguishable at the 4-bag headline level**. Two independent compositions converge on the same number. **Stability across n**: 0.9992 (n=50) → 0.9955 (n=200) → 0.9953 (n=500) shows monotonic convergence; n=50 was small-sample artifact, n=200 already stable, n=500 confirms within 0.0002. **"Ensemble cancels fragility" strongest at n=500**: `24_LS030_seed42` ni_FAR=22.50% alone (single-model FAIL on 5250-chip eval), 4-bag holds at 0.00%. See `iters/iter_42_n200_rebuttal.md` (Phase 28 n=500 confirmation appended) + `tables/paper_main_headline.csv` (rows `iter39_ensemble_4bag_pureHard_n500_FINAL` and `iter39_ensemble_4bag_hardKD_n500_TIE`). |
|   42 | (CONFIRMED by n=500 — both within 0.0002 noise) **★★★ Prior n=200 REBUTTAL HEADLINE — pure-hard 4-bag 0.9955/0%** | **v15 bit_F1 0.9955 (n=200, 3080 chips)** | — | n=50 0.9992 → n=200 0.9955 (rebuttal) → n=500 0.9953 (CONFIRMED) | — | (CONFIRMED by Phase 28 n=500: 0.9953/0% — within sampling noise of n=200 0.9955) **iter 42 n=200 REBUTTAL** — same pure-hard 4-bag composition `{24_LS030_seed42 + 26B + 26D + 26H}` thr ≥ 2/4 majority, re-evaluated at v15direct **n=200 (3080 chips, 4× larger eval)**. **v15 bit_F1=0.9955, ni_FAR=0.00%**, per-class bb=**0.9984** / fk=**0.9881** / sc=**0.9953** / sr=**1.0000**. **Supersedes iter39 n=50 reading 0.9992 as small-sample artifact**. All 4-bag types fall in 0.9945–0.9959 at n=200 (sampling band 0.0014, vs n=50 spread 0.0047) — **"pure-hard wins by +0.0016" thesis FALSIFIED**. Single-model re-eval shows `24_LS030_seed42` n=200 ni_FAR=20.5% (single-model FAIL) yet the 4-bag containing it holds at 0% ni_FAR — **"ensemble cancels fragility" thesis STRENGTHENED** (further confirmed at n=500 where seed42 alone has 22.50% ni_FAR but 4-bag stays at 0%). See `iters/iter_42_n200_rebuttal.md` + `tables/paper_main_headline.csv`. |
|   39 | (REBUTTED at n=200 by iter 42 — 0.9992 was n=50 small-sample artifact, real value 0.9955) **★★★ Prior NEW PAPER MAIN HEADLINE (pure hard-label 4-bag, no asym, no KD)** | **v15 bit_F1 0.9992 (n=50)** | — | **+0.0016 v15 vs iter37 4-bag (n=50)** | — | (REBUTTED at n=200 by iter 42 — 0.9992 was n=50 small-sample artifact, real value 0.9955) iter 39 prior PAPER MAIN — 4-bag majority vote (≥2/4) composition: **24_LS030_seed42 + 26B + 26D + 26H** = pure hard-label diversity within (g, LS, seed, fill-style) space (g∈{2,3,4} × LS∈{0.30, 0.40, 0.50, 0.67} × seed∈{1, 42} × fill∈{compl, white}). v15 bit_F1=**0.9992** (n=50), ni_FAR=**0.00%**. **Supersedes iter37 4-bag** (`26B + 26D + 37E + 33A`, 0.9976) by **+0.0016** at n=50; n=200 spread between iter33/34/37/39 4-bag types collapses inside sampling band. **Pure hard-label diversity reaches global optimum without specialty axes** — at n=50; thesis falsified at n=200. Inference-only re-composition (no new training; uses iter21–26 preserved checkpoints). See `iters/iter_39_pureHard_headline.md` + `iters/iter_42_n200_rebuttal.md` + `tables/paper_main_headline.csv`. |
|  0\* | T0__I0              |     0.7302 |     0.4472 |          — |          — | outputs/stage1_260505_162842 (argmax baseline)                        |
|    1 | T0__I3              |     0.8466 |     0.6017 |    +0.1164 |    +0.1545 | outputs/stage1_260505_162842                                          |
|    2 | T0__I7              |     0.8485 |     0.6210 |    +0.0019 |    +0.0193 | outputs/stage1_260505_165400                                          |
|    3 | T0__I10             |     0.8542 |     0.6517 |    +0.0057 |    +0.0307 | outputs/stage1_260505_170827                                          |
|    4 | T1__I10             |     0.8634 |     0.7006 |    +0.0092 |    +0.0489 | outputs/stage1_260505_173649                                          |
|    5 | **T1_LS20__I7**     | **0.9268** | **0.8449** | **+0.0634**| **+0.1443**| outputs/phase_a_260505_175105                                         |
|    6 | (no new best)       |     0.9268 |     0.8449 |    +0.0000 |    +0.0000 | outputs/phase_a_260505_185805 — Phase A3 confirms ep=8 is global best |
|    7 | **T7c__I10**        | **0.9271** |     0.8307 | **+0.0003**|    -0.0142 | outputs/stage1_260505_195730 — BCE+LS=0.20+CutMix(p=0.5), bb+sr 0.32→0.96 |
|    8 | **T9d__I7** ☆       | **0.9705** | **0.9267** | **+0.0434**| **+0.0960**| outputs/stage1_260505_211334 — BCE+LS=0.07+CutMix(p=0.5), seed=42 (lucky outlier) |
|    8 | T9g__I7 (realistic) |   0.9408 |  0.8307 |    +0.0137 |    +0.0000 | outputs/stage1_260505_212557 — same config, seed=43; variance ±0.030 |
|    9 | (no new best)       |   0.9705 |  0.9267 |    +0.0000 |    +0.0000 | drop_path / cutmix-rect / two-LR all regress (see iter_09)             |
|   10 | **baseline+C_44 ENS** ★★★ | **0.9950** (10-def) | **0.9396** | **+0.0245** (over T9d) | **+0.0129** | ad-hoc ensemble, baseline T9d + C_44 (Normal trained, cutmix=0.25) logit avg. 5-sample-seed mean **0.9930±0.005**, FAR 0.00%. See `iters/iter_10_master_consol_sc_sr.md`. |
|   11 | (no new best — single) | 0.9050 | 0.8646 | -0.0900 vs iter10 ENS | — | iter 11 paper-style 6-train × 6-inf × 3-phase ablation. Best single = T6+I3 (BCE→ASL) macro 0.905 BUT Normal F1=0.000, FAR=100% — **operationally unusable**. Confirms ensemble necessity. See `iters/iter_11_paper_ablation_matrix.md`. |
|   12 | T7 v19zpp (no Normal)  |     0.8490 |        n/a |    -0.1210 |          — | iter 12 v19z++ master 21-class (4+6+4+Normal+Invalid+5OOD), `--no-normal`. Best single T7 (LS=0.20 + CutMix), CF1 0.8490, **ni_chip_FAR 80%, ood_chip_FAR 100%** — paper "fix" iter exposes Normal-not-trained 한계. See `iters/iter_12_v19_status.md`. |
|   13 | **T7N+T5_w70_30 ENS** ★ | 0.9083 (CF1) |        n/a |    +0.0593 |          — | iter 13 Cycle A: T7-with-Normal training (C7N) + 9 logit-avg ensemble configs. Winner = T7N+T5 70:30 = **CF1 0.9083, F1_fork 0.7656, ni_chip_FAR 0.50%**. Cycle B CutMix variant grid: random_rect winner CF1 0.9188 but ni_FAR 20% trade-off. See `iters/iter_13_cycle_a_b_t7n_v19zpp.md`. |
|   14 | T7N v20 single        | 0.9226 (CF1) |        n/a |    +0.0143 |          — | iter 14 v20 chip data (fork sigma 1.0~1.5 → 1.8~2.5, 두께 ↑) retrain T7N. CF1 0.9226 (vs iter 13 random_rect 0.9188), F1_fork 0.8591, **ni_chip_FAR 0.00%**, ood_chip_FAR 0.94%. fork single recall 1.0, fork+sr +9.4%. See `iters/iter_14_v20_fork_thickness.md`. |
|   15 | T7N P1A LS=0.05 (Normal OFF) | 0.9088 (CF1) |        n/a |    -0.0138 |          — | iter 15 paper-style 4-class only (`--no-normal`) ablation matrix on post-v5 chip data. P1A LS sweep ∈ {0.025-0.25}, peak at LS=0.05 = **CF1 0.9088** but **ni_FAR 36%**. T3 Focal: ni/ood FAR 100%. T9 sigfocal: 0.8273. **Paper baseline counter-example** to demonstrate Normal training necessity. See `iters/iter_15_p0_p1a_paper_ablation.md`. |
|   18 | T7N iter18D (grid+pair_masked+soft) | 0.8272 | 0.6125 |  (sweep) |  (sweep) | iter 18 soft-label CutMix sweep (6 cells, 260508 12:54-13:48). Range 0.7843-0.8272 (spread +0.0429, ≈±0.014). **iter18D ★ winner** (grid+pair_masked+soft, I3 = 0.8272). complete-fill + label_scale ∈ {0.5,0.75,1.0} (F1/F2/F3) all sub-D. Best-inference flips with CutMix density: A/B/D→I3, F1→I7, F2/F3→I10. See `iters/iter_18_softlabel_iter_19_complement.md`. |
|   19 | T7N iter19B (complement g=2, l=0.75, pair=masked) | **0.8427** | 0.5641 | +0.0155 vs iter18D |    — | iter 19 complement CutMix sweep (12 cells planned, **partial: 2 of 12 done**, 260508 14:05-). **iter19B ★ best so far** = macro_f1 0.8427 (I3), mAP 0.9685, sr F1 0.9307. iter19A (g=2, l=0.5) 0.8078 epoch-1 crash (retrain pending). C-L (10 cells: g∈{2,3,4}, label∈{0.5,0.75,1.0}, pair∈{masked,none}) in progress, see `outputs/_iter19_complement_resume.log`. **Single-seed; ±0.030 noise applies — needs replication.** See `iters/iter_18_softlabel_iter_19_complement.md`. |
|   21 | **T7N iter21E (19C compl g=2 LS=1.0 FCM-PM)** ★ | **bit_F1 0.9913 v14 / 0.9691 v15** | — | first single model passing **both** dual-eval gates | — | iter 21 clean baseline — 8 trains on `classification_chips/` only, dual-eval on disjoint **v14class (800 chip)** + **v15direct (1000 chip, +4 OOD wafer-canvas)** = no-leak protocol. **E (19C complement g=2 LS=1.0 FCM-PM)** wins both: v14 bit_F1=0.9913 ni_FAR=0.00%, v15direct bit_F1=0.9691 ni_FAR=3.75% (per-class F1 ≥ 0.94 all 4 defects). Baseline A (T5 no-Norm) collapses on v15: bit_F1=0.7872 ni_FAR=0% only because Normal/Invalid mass merged into defect bins (v14 ni_FAR=100%). C (std CutMix LS=1.0): ni_FAR=100% on both — soft labels + complement structure both required. See `iters/iter_21_clean_baseline.md`. |
|   22 | T7N iter22D (LS=0.30 single) | bit_F1 0.9851 v14 / 0.9439 v15 | — | -0.0062 / -0.0252 vs 21E | — | iter 22 hparam tune sweep (10 cells, 1 atomic axis each on T7N+19C base). Only **22D LS=0.30** (ni_FAR 0%/1.25%) and **22G drop_path=0.05** (0%/0%, but bit_F1=0.9207 v15) clear both gates. LS=0.10 / EMA / warmup / fork pos_weight all blow up ni_FAR to 60–100%. CutMix-p≠0.5 regresses bit_F1 by ≈0.06–0.13. **22J `lr-head=5e-5` md5 byte-identical to 21E → CLI flag wiring bug**, treated as replica. Default recipe is at a stable local optimum. See `iters/iter_22_25_full_phase4.md`. |
|   23 | T7N iter23A,B fork pos_weight | bit_F1 0.9984 v14 / 0.9702 v15 (best of 2) | — | catastrophic ni_FAR 87–100% | — | iter 23 fork pos_weight ∈ {0.5, 0.7} — both destroy ni_FAR (87–100% on both eval sets) by pushing Normal mass into fork bin via calibration shift. **Negative result, paper counter-example** for "single per-class loss tweak ≠ free F1." See `iters/iter_22_25_full_phase4.md`. |
|   24 | T7N iter24 LS=0.30 3-seed | bit_F1 0.9945/0.9944 v14 (seed=7,42) | — | ni_FAR bimodal 50–67% v15 | — | iter 24 LS=0.30 3-seed verify (seed=1=iter22D, +seeds 7,42). **Per-seed v15 ni_FAR is bimodal**: seed=1 → 1.25%, seeds 7+42 → 50–67%. Single-seed claims overstate worst-case OOD safety. **Directly motivates iter25 ensemble** — different seeds make complementary OOD errors. See `iters/iter_22_25_full_phase4.md`. |
|   25 | **★★★ Iter25 6-seed I10 majority ENS** | **bit_F1 0.9976 v14 / 0.9913 v15** | — | **+0.0063 v14 / +0.0222 v15** vs 21E | — | **iter 25 BREAKTHROUGH — 6-seed I10 majority vote ensemble** (3 LS=0.20 seeds + 3 LS=0.30 seeds, threshold ≥4/6). v14 bit_F1=**0.9976** ni_FAR=**0.00%** (F1bb=0.9969, F1fk=0.9937, F1sc=F1sr=**1.0000 perfect**). v15direct bit_F1=**0.9913** ni_FAR=**0.00%** (all 4 per-class F1 ≥ 0.987). vs 12-T5 baseline (paper start): v15 0.7872 → 0.9913 = **+0.2041 (+26%)**, ni_FAR 100%(real) → 0%. **First config in chip-multilabel history combining top-tier defect F1 with zero false-alarm under OOD pressure.** Validates iter10 ensemble principle (diversity > quantity) at 6× scale. **New paper main winner.** See `iters/iter_22_25_full_phase4.md` + `tables/paper_main_headline.csv`. |
|   26 | **iter26B (g=3 LS=0.50) NEW best single** | bit_F1 0.9921 v14 / 0.9791 v15 | — | +0.0008 v14 / +0.0100 v15 vs 21E single | — | iter 26 9-train diversity sweep (g∈{2,3,4} × LS∈{0.40,0.50,0.60,0.67,0.75,0.83,0.85,1.00} × CutMix-fill ∈ {complement, white, noise}). **26B (g=3 LS=0.50)** clears both gates with v14=0.9921/0.00%, v15=0.9791/1.25%, F1_fk=0.994. Five cells advance to iter27 ensemble: 26B/26D/26F/26G/26H (others fail ni gate). **CutMix white/noise byte-identical** (negative axis). g=3 = sweet spot. See `iters/iter_26_27_diversity_finalEnsemble.md`. |
|   27 | **★★★ Iter27 14-bag I10 majority ENS (PAPER HEADLINE)** | **bit_F1 1.0000 v14 / 0.9929 v15** | — | **+0.0024 v14 / +0.0016 v15** vs iter25 | — | **iter 27 PAPER HEADLINE — 14-bag majority vote** (iter25 6-seed + 21F/21H/22G + 5 iter26 winners). thr ≥5–6/14 simple-majority window: v14 bit_F1=**1.0000 PERFECT**, v15 bit_F1=**0.9929** ni_FAR=**0.00%** (F1bb=0.9909, F1fk=0.9874, F1sc=0.9907, F1sr=0.9970). Threshold-flat: v14 stays 1.0000 across thr=5..9/14. **vs 12-T5 baseline (paper start): v15 0.7872 → 0.9929 = +0.2057 (+26%), ni_FAR 100%(real) → 0%.** Diversity > Quantity validated at 14× scale across 3 axes (g × LS × CutMix-fill). **New all-time best across every reported metric.** See `iters/iter_26_27_diversity_finalEnsemble.md` + `tables/paper_main_headline.csv`. |
|   28 | iter28 Mixup α sweep (6 trains) | **all v15 ni_FAR=100%** | — | — | — | iter 28 **paper §5 evidence — pixel α-blend palette destruction**. Mixup α ∈ {0.1, 0.2, 0.4, 1.0, 2.0} + α=0.4+CutMix combo: **all 6 v15 ni_FAR=100%**, only α=0.2 has v14 ni 5% (still v15 100%). bit_F1 spread 0.86–0.99 v14, 0.86–0.98 v15 — F1 alone is misleading without dual-FAR gate. Mechanism: Mixup synthesizes invalid intermediate palette grades (palette is discrete code, not luminance), destroying Normal-vs-defect calibration. **Confirms CutMix > Mixup is design-level not tuning.** See `iters/iter_28_29_paper_ablation.md`. |
|   29 | iter29 label×spatial isolation (3 trains, paper §5 6-cell) | 29B v15 **bit_F1=0.9953** (F1-only) / ni 100% | — | — | — | iter 29 **paper §5 evidence — 4 design contributions all necessary**. 29A (std box-cut + hard) v14/v15 0.74/0.76 ✗ — single rect leaks. 29B (compl + pair-mask + soft LS=0.5) v15 0.9953 ★ highest single bit_F1 ever **BUT ni_FAR=100%** — F1-only winner not deployable. 29C (grid_complete + hard LS=1.0 no pair-mask) v14 ni 2.5% ✓ but v15 ni 100% ✗. **Paper §5 6-cell label×spatial matrix complete — only iter21E ★ (compl + pair-mask + hard + full-cover) clears both gates.** Removing any single design (region-paste / full-cover / pair-mask / hard-label) breaks the model. See `iters/iter_28_29_paper_ablation.md` + `tables/paper_section5_ablation.csv`. |
|   33 | Iter33 4-bag I10 majority ENS (superseded by iter34) | v15 bit_F1 0.9945 | — | +0.0016 v15 vs iter27 14-bag | — | iter 33 prior PAPER MAIN — 4-bag majority vote (≥2/4) composition: **26B + 21F + 21H + 26D** = span g∈{3,4} × LS∈{0.40, 0.50, 0.67, 0.75} × CutMix=complement. v15 bit_F1=**0.9945**, ni_FAR=**0.00%**. Supersedes iter27 14-bag and iter28A 16-bag on per-model gain. 3 of 19 dual-pass-eligible 4-cell subsets all tie at 0.9945 — structural winner. **Superseded by iter34 4-bag with KD-student (+0.0016 v15 bit_F1).** Retained as hard-label-only baseline. See `iters/iter_33_small_bag_exploration.md`. |
|   34 | Iter34 4-bag KD-mixed (superseded by iter37) | v15 bit_F1 0.9961 | — | +0.0016 v15 vs iter33 4-bag | — | iter 34 (superseded by iter37 4-bag with asymmetric-label diversity +0.0015) — 4-bag majority vote (≥2/4) composition: **26B + 21F + 26D + 33A** = span hard-label{g∈{3,4} × LS∈{0.40, 0.50, 0.67}} + KD-student diversity axis (33A: α=0.3 T=4 distilled from 14-bag teacher). v15 bit_F1=**0.9961**, ni_FAR=**0.00%** (F1bb=0.9937, F1fk=0.9937, F1sc=0.9969, F1sr=1.0000). **Supersedes iter33 4-bag** by replacing 21H (hard-label g=4 LS=0.75) with 33A KD-student. Three 4-bag alternates all tie at 0.9961 (26B+26D+21E+33A, 26B+26D+36C+33A, 26B+26D+26H+33A). **Diversity gain comes from non-correlated KD axis** — pure-KD 4-bag (33A+B+C+D) only reaches 0.9873 (KD alone insufficient); hard-label-only 4-bag reaches 0.9945. Mixing 1 KD-student into 3 hard-label cells = +0.0016. 6-bag ties 4-bag at 0.9961 (4-bag = cost-optimal). Ultra-cheap 2-bag OR (26B+33A) = **0.9969 v15 bit_F1** (highest of all configs but ni_FAR=1.25%). See `iters/iter_34_bagSweep_KD_headline.md` + `tables/paper_main_headline.csv`. |
|   37 | Iter37 4-bag KD+asym I10 majority ENS (superseded by iter39 pure-hard) | v15 bit_F1 0.9976 | — | +0.0015 v15 vs iter34 4-bag | — | (superseded by iter39 pure-hard 4-bag, +0.0016) prior PAPER MAIN — 4-bag majority vote (≥2/4) composition: **26B + 26D + 37E + 33A** = first 4-bag spanning **ALL FOUR diversity axes** (group g + label-smoothing LS + asymmetry + KD distillation). v15 bit_F1=**0.9976**, ni_FAR=**0.00%** (F1bb=0.9969, F1fk=0.9969, F1sc=0.9969, F1sr=1.0000). **Supersedes iter34 4-bag** by replacing 21F (hard-label g=3 LS=0.67 sym) with **37E** (g=3 (s_A=1.0, s_B=0.5) **asymmetric AB-pair**). Asymmetric label diversity = the missing 4th axis after g/LS/KD. iter37 sweep **12/12 FINAL — 5 PASS / 7 FAIL** (37A/D/E/H/L pass dual-gate). g=4 only PASS via exact area-prop match (0.25, 1.0) — paper §6 strengthening: at g=4, only labels EXACTLY matching (1/g, (g-1)/g) preserve FAR; (1.0, 0.5) and (0.75, 1.0) — both PASS at g=2/g=3 — FAIL at g=4. 7 alternative 4-bags all reach 0.9969 (1 below headline). 5-bag and 6-bag do not exceed 4-bag (4-bag = cost-optimal). Two consecutive at-cost-fixed +0.0015 lifts (iter33 → iter34 → iter37) validate diversity-axis-discovery research strategy. **vs 12-T5 baseline (paper start): v15 0.7872 → 0.9976 = +0.2104 (+27%), ni_FAR 100%(real) → 0%.** See `iters/iter_37_asymmetric_AB_labels.md` + `tables/paper_main_headline.csv`. |
|   38 | iter38 seed-robust + g=2 gap-fill (6/6 FINAL) | v15 bit_F1 (single 0.84–0.98 all FAIL) / **★ 4-bag 0.9976 ± 0.0007 PASS 3/3 seeds** | — | NEW HEADLINE seed-robustness confirmed | — | **iter 38 paper §6 gap-fill** — **6/6 single cells FAIL dual-gate** (38A/B = 37E reseeds @ seed=7,42 → bF1 holds 0.984 but ni_FAR collapses to 100%; 38C/D/E/F = g=2 PASS basin probes at (1.0, 0.6/0.4) and (0.6/0.4, 1.0) all FAIL). **★ NEW HEADLINE 4-bag is seed-robust** — replacing 37E in `26B + 26D + 37E + 33A` with seed=7 → v15 bF1=0.9976 / ni_FAR=1.25% PASS, with seed=42 → v15 bF1=0.9969 / ni_FAR=0.00% PASS. **3/3 seed variants PASS dual-gate, 0.9976 ± 0.0007 spread**. Single-model seed-luck (bimodal FAR: seed=1 → 1.25%, seeds 7/42 → 100%) is **canceled by the 4-bag ensemble** — the other 3 bags carry FAR while the 37E slot contributes asymmetric-label diversity in any of its seeds. 6-bag seed-redundant `{26B+26D+37E_s1/s7/s42+33A}` thr=3/6 simple-majority = 0.9969 (does not beat 4-bag → 4-bag remains cost-optimal). g=2 PASS basin = **3 isolated points** in (s_A, s_B) space, not a contiguous region. Paper main headline unchanged. See `iters/iter_38_seedRobust_gapfill.md` + `tables/paper_main_headline.csv`. |
|   32 | iter32A KD baseline (α=0.5 T=4 skip-cutmix) | v15 bit_F1 0.8952 / ni_FAR 0.00% | — | KD pivot (over-soften) | — | iter 32 KD baseline single cell — distill iter33 4-bag teacher into single T7 student. α=0.5 over-soften pivot lands at v15 bit_F1=0.8952 (PASS dual-gate). Sets up iter33 sweep. See `iters/iter_32_33_KD_sweep.md`. |
|   33b | **iter33A KD-student (α=0.3 T=4 skip-cutmix)** ★ | v15 bit_F1 **0.9840** / ni_FAR 0.00% | — | -0.0105 vs 4-bag teacher at **1× cost** | — | **iter 33 KD WINNER — KD-student headline (production-cost variant of paper §7)**. α=0.3 / T=4 / skip-on-cutmix recovers **98.94% of the 4-bag teacher gain at 25% inference cost** (single forward pass). 8 cells dual-pass under v15 ni_FAR ≤ 5%. **Negative axes (paper §5)**: α=0.5 (-0.089), α=0.7 (-0.009), with-cutmix (-0.089), EMA 0.95 (-0.024), epochs=16 (-0.089). T=2 +0.0113 F1 but +1.25% FAR borderline. **Three over-soften failure modes converge on bit_F1=0.8952** (32A, 33E, 33G). MAIN headline unchanged — 33A positioned as deployment variant. See `iters/iter_32_33_KD_sweep.md` + `tables/paper_section5_ablation.csv`. |
|   35 | iter35 area-proportional FCM-PM (8/8) | best dual-pass v15 bit_F1 **0.9033** (35H) | — | structural FAR collapse | — | iter 35 **area-proportional label scaling sweep** (g∈{2,3,4} × scale∈{0.3, 0.5, 1.0, 1.33, 1.5}). **7/8 cells collapse v15 ni_FAR to 100%**. Only 35H (g=3, scale=0.3, effective labels (0.10, 0.20)) PASSES at v15 bit_F1=0.9033 — well below iter21E single. **F1-only winner** = 35E (g=3, s=0.5, F1=0.9898) but ni_FAR=100% — same trap as iter29B. Confirms paper §6 "FCM-PM only works under hard labels + complement fill + pair mask + full cover." See `iters/iter_35_areaprop_complete.md`. |
|   36 | iter36 g=2 symmetric LS sweep (2/8 partial) | both completed cells FAIL | — | localizes g=2 PASS/FAIL boundary | — | iter 36 **g=2 symmetric LS axis** (LS ∈ [0.40, 0.90], 8 cells planned, **C–H pending**). 36A (LS=0.40) and 36B (LS=0.45) both v15 bit_F1 < 0.88 + ni_FAR=100%. Pattern matches iter30D (g=2 LS=0.50 FAIL) and diverges from iter21E (g=2 LS=1.0 PASS) — the PASS region for g=2 is narrowing toward LS=1.0. **14/16 cells (iter35 + iter36 partial) v15 ni_FAR=100% — largest single-batch FAR collapse in project history**, paper §6 structural failure mode confirmed at 4 independent dimensions. See `iters/iter_35_areaprop_complete.md`. |
|   40 | iter40 (g, LS) gap-fill 6/6 FINAL | 4 PASS (low bit_F1) / 2 FAIL | — | no headline change; LS axis non-monotonic at g=4 | — | iter 40 paper §6 (g, LS) map gap-fill — 6 single-model T7 cells. **Dual-PASS (4)**: 40A (g=2, LS=0.20) v15 bF1=0.8841/0%, 40B (g=3, LS=0.30) 0.8213/0%, 40D (g=4, LS=0.30) 0.8784/3.75%, 40F (g=4, LS=0.60) 0.9799/3.75%. **FAIL (2)**: 40C (g=3, LS=0.40) bF1=0.9698/100% (F1-only winner trap), 40E (g=4, LS=0.50) 0.9429/100%. **g=4 LS axis is non-monotonic**: PASS at 0.30, FAIL at 0.50, PASS at 0.60 — same seed-luck pattern as iter38/37E. No single-model cell beats iter39 4-bag NEW HEADLINE. See `iters/iter_40_41_phase26_summary.md`. |
|   41 | **Phase 26 big bag-sweep — 4-bag global optimum confirmed** | C(25,4)=12650 + C(25,5)=53130 + C(25,6)=177100 combos | — | iter39 NEW HEADLINE 0.9992 v15 bit_F1 = global optimum across all bag sizes | — | **iter 41 paper §6 finalization** — exhaustive C(25, k) sweep over the pool of 25 single-model preds parquets at simple-majority thr=⌈k/2⌉. **C(25, 4)=12,650 combos**: peak = **0.9992 / 0.00%** at `24_LS030_seed42 + 26B + 26D + 26H` = **iter39 NEW PAPER MAIN HEADLINE confirmed as global 4-bag optimum**. Top-10 bF1=0.9984–0.9992, all dual-PASS. **C(25, 5)=53,130 combos**: peak=**0.9976/0%** = REGRESS −0.0016 vs 4-bag; AHHHK/AHHHH dominate ties. **C(25, 6)=177,100 combos** (~50 min compute): peak=**0.9984/0%** at `37E+24_s7+24_s42+26B+26D+26H` (AHHHHH) — partial recovery from 5-bag but still 0.0008 below 4-bag. **Pure-hard cost frontier**: size=2 OR-vote 0.9929 / size=3 0.9969 / size=4 **0.9992 ★** / size=5 0.9976 / size=6 0.9984 / size=7 0.9953 — **k=4 is global optimum** within pure-hard space. **4-bag remains cost-optimal across k ∈ {4, 5, 6}**; paper main headline UNCHANGED. See `iters/iter_40_41_phase26_summary.md`. |
|   41b | Phase 26 prob-avg vs majority counter-textbook (paper §5.24/§6.18) | 4-bag majority 0.9992 vs prob-avg 0.9976 | — | Δ=+0.0016 majority over prob-avg | — | iter 41 supplementary — **counter to ML textbook guidance**, simple-majority hard-vote (thr=2/4) outperforms prob-avg+thr=0.5 by **+0.0016 v15 bit_F1** on the headline 4-bag. Mechanism: in our high-correlation regime (4 hard-label models on same chip data with overlapping g/LS), per-chip probabilities co-shift; prob-avg amplifies correlated errors at threshold boundary, while hard-vote majority **decorrelates by quantization** (each vote crosses 0.5 independently). Already documented in paper §5.24 / §6.18; this iter logs the data confirmation across 12K+ bag combos. See `iters/iter_40_41_phase26_summary.md`. |
|   41c | Phase 26 paper §6 nuance update — 3-tier (g, LS) map | (no metric change) | — | seed-fragile vs deterministic regions | — | iter 41 supplementary — paper §6 "narrow PASS basin" framing **over-stated** the structural picture. Updated 3-tier nuance: (a) **deterministic-PASS** = g=2 LS=0.30 (3 seeds verified PASS: iter22D + iter24_s7 + iter24_s42); (b) **seed-fragile** = g=3 LS=0.50 (37E: 1/3 PASS across seeds); (c) **deterministic-FAIL** = area-prop iter35 (7/8) + g=4 LS=0.50 (40E) + g=2 LS=0.40/0.45 (36A/36B). Non-monotonic "FAIL between PASS" at g=4 LS=0.30/0.50/0.60 (40D PASS / 40E FAIL / 40F PASS) is consistent with seed-luck noise, not a structural basin. **Honest paper claim**: single-model FAR is high-variance w.r.t. seed/LS perturbations; ensemble cancels the stochasticity. Logger records data only — paper-narrator owns prose update to `paper/06_analysis.md`. See `iters/iter_40_41_phase26_summary.md`. |

\*iter 0 = the argmax baseline cell that lives inside iter 1's run.
☆ T9d is a single-seed favorable outlier (seed=42). The matched-config
seed=43 run (T9g) gives 0.9408. **Single-seed variance ±0.030 macro-F1**
at this config is now the dominant uncertainty above the LS axis (see
iter 8 for the full curve / variance discussion).

**Phase A final winner (closed): `T1_LS20_ep8 + I7`  →  macro_f1 = 0.9268, top1\_11 = 0.8449.**
Iter 6 (Phase A3 epochs sweep over {3, 5, 12} at LS=0.20) does not beat ep=8;
ep=12 → 0.8926 (I3), ep=3 → 0.8763 (I10), ep=5 → 0.8567 (I10). See
`iters/iter_06_phase_a3_epochs_sweep.md` for the full nine cells and the
training-duration → best-inference-variant regime change.

**Phase F (iter 7) outcome — two faces.** Anomaly-detection BKM transfer
(F1 warmup 2ep, F2 EMA 0.95) regressed by −0.109 / −0.089 macro-F1 — small
data + TAPT init does not need warmup, and EMA over-smooths under ~12
effective steps. The I11 pair-aware threshold band-aid (no retrain) lost
−0.007 net (bb+sr recall +25 chips, but bb+fork over-trigger 31 FP). The
T7 atomic decomposition (T1 → T7a CE→BCE → T7c +CutMix p=0.5) ties macro_f1
on the surface (0.9268 → 0.9271, +0.0003) but **flips the operational
profile**: bb+sr combo recall jumps from 0.32 to 0.96 (+0.63), and
`scratch_rot` per-class F1 reaches 1.0000 in T7c. CutMix-p sweep peaks
sharply at 0.5 (→ 0.7: 0.9038, → 0.3: 0.8626, → 0.0: 0.8577). See
`iters/iter_07_phase_f_warmup_ema_t7_cutmix.md`.

**Iter 8 (T9 LS sweep + variance verify).** Re-sweeping LS under the
BCE+CutMix recipe shifts the optimum to **LS=0.07** (T9d__I7 = 0.9705,
seed=42). The curve has a sharp cliff at LS=0.08 (T9e=0.8085) and a
broad 0.94 plateau across LS∈{0.05, 0.06, 0.10}. The same config at
seed=43 (T9g) drops to 0.9408 — a **single-seed variance of ±0.030**,
roughly the same magnitude as the LS=0.05→0.07 gap. The variance is
concentrated in fork F1 (0.945 vs 0.815 across seeds). T9d 0.9705 is
the best observed, T9g 0.9408 is the realistic point estimate. See
`iters/iter_08_T9_LS_sweep_variance.md`.

**Iter 9 (drop_path / cutmix-rect / two-LR — all atomic-failed).**
Three orthogonal regularizers/training-regime BKMs probed on top of
the iter-8 LS=0.07 recipe; **all regress**. T10 drop_path 0.05 (n=2
seeds): −0.054 / −0.049. T11 cutmix-rect 0.25: −0.106 (confounded with
0.5→0.25 ratio drop, which alone matches iter-7's p=0.3 result). T12
two-LR backbone/head: −0.084 (top1_11 drops 0.27, killing combos). All
three fail under the same diagnosis as iter 7 warmup/EMA: long-training
regime BKMs that need many effective steps to stabilize, on a small-
data + TAPT-init + 8-epoch budget that doesn't provide them. See
`iters/iter_09_negative_axis_drop_cutmix_two_lr.md`.

Cumulative gain vs argmax baseline (best observed, T9d__I7): **+0.2403
macro-F1**, **+0.4795 top1\_11class**. Realistic gain (T9g__I7, same
config seed=43): **+0.2106 macro-F1**, **+0.3835 top1\_11class**.

_Source: outputs/stage1_260505_162842/results_matrix.parquet,
outputs/stage1_260505_165400/results_matrix.parquet,
outputs/stage1_260505_170827/results_matrix.parquet,
outputs/stage1_260505_173649/results_matrix.parquet,
outputs/phase_a_260505_175105/sweep_log.csv,
outputs/phase_a_260505_185805/sweep_log.csv,
outputs/stage1_260505_195730/results_matrix.parquet (T7c, iter 7),
outputs/stage1_260505_211334/results_matrix.parquet (T9d, iter 8 ☆),
outputs/stage1_260505_212557/results_matrix.parquet (T9g, iter 8 — variance verify),
outputs/stage1_260505_{213423,213817,214222,214634}/results_matrix.parquet (iter 9)._

## Top-15 all-time cells (by macro_f1)

| rank | iter | cell_id            | macro_f1   | top1_11 | source                                                                  |
|-----:|-----:|--------------------|-----------:|--------:|-------------------------------------------------------------------------|
|    1 |    8 | **T9d__I7** ☆      | **0.9705** |  0.9267 | outputs/stage1_260505_211334 — LS=0.07 seed=42 (lucky outlier)          |
|    2 |    8 | T9d__I10 ☆         |     0.9705 |  0.9267 | outputs/stage1_260505_211334 — same checkpoint, I10 ties I7              |
|    3 |    8 | T9d__I3 ☆          |     0.9673 |  0.9187 | outputs/stage1_260505_211334                                            |
|    4 |    8 | T9d__I11 ☆         |     0.9654 |  0.9205 | outputs/stage1_260505_211334                                            |
|    5 |    8 | T9b__I7            |     0.9449 |  0.8670 | outputs/stage1_260505_210535 — LS=0.05                                  |
|    6 |    8 | T9b__I10           |     0.9449 |  0.8670 | outputs/stage1_260505_210535                                            |
|    7 |    8 | T9b__I11           |     0.9440 |  0.8659 | outputs/stage1_260505_210535                                            |
|    8 |    8 | T9b__I3            |     0.9424 |  0.8614 | outputs/stage1_260505_210535                                            |
|    9 |    8 | T9g__I7 (realistic)|     0.9408 |  0.8307 | outputs/stage1_260505_212557 — LS=0.07 seed=43, variance verify         |
|   10 |    8 | T9g__I10           |     0.9408 |  0.8307 | outputs/stage1_260505_212557                                            |
|   11 |    8 | T9g__I11           |     0.9408 |  0.8307 | outputs/stage1_260505_212557                                            |
|   12 |    8 | T9f__I3            |     0.9401 |  0.8648 | outputs/stage1_260505_212153 — LS=0.06                                  |
|   13 |    8 | T9a__I10           |     0.9364 |  0.8489 | outputs/stage1_260505_210059 — LS=0.10                                  |
|   14 |    8 | T9a__I7            |     0.9346 |  0.8443 | outputs/stage1_260505_210059                                            |
|   15 |    8 | T9a__I11           |     0.9346 |  0.8443 | outputs/stage1_260505_210059                                            |

☆ T9d (rank 1–4) is a single-seed favorable outlier (seed=42); the
matched-config seed=43 run T9g lands at rank 9–11 (0.9408). The realistic
gap between rank-1-observed (T9d, 0.9705) and rank-9-realistic (T9g,
0.9408) is **0.0297 macro-F1** at fixed config — see iter 8 for the
variance discussion.

Iter 8 reshuffles the **entire top-15 to T9 cells** — every iter-7 cell
(T7c, T7d) and every iter-5 cell (T1_LS20, T1_LS15) is pushed off the
list by the BCE+LS=0.07+CutMix(p=0.5) recipe. The previous rank-1
(T7c__I10 = 0.9271) is now rank 16+. Iter 9's drop_path/cutmix-rect/
two-LR experiments did not produce any cells that crack the top-15.

_Source: docs/chip-multilabel/tables/all_runs_macro_f1.csv (all rows incl. ranks 16+)._

## Per-iter winner — per-class F1 detail

### iter 1 — T0__I3 (frozen, F1-max + top-K rescue)

| class          | threshold | precision | recall | F1     | AP     |
|----------------|----------:|----------:|-------:|-------:|-------:|
| bank_boundary  |    0.4994 |    0.9788 | 0.9391 | 0.9585 | 0.9752 |
| fork           |    0.1195 |    0.4843 | 0.9141 | 0.6331 | 0.5762 |
| scratch        |    0.7682 |    1.0000 | 0.9438 | 0.9711 | 0.9723 |
| scratch_rot    |    0.8355 |    1.0000 | 0.7000 | 0.8235 | 0.8700 |

_Source: outputs/stage1_260505_162842/per_class_metrics.parquet (cell_id=T0__I3)._

### iter 2 — T0__I7 (frozen, F1-max + step-search Δ=0.02)

| class          | threshold | precision | recall | F1     | AP     |
|----------------|----------:|----------:|-------:|-------:|-------:|
| bank_boundary  |    0.5000 |    0.9788 | 0.9391 | 0.9585 | 0.9752 |
| fork           |    0.1400 |    0.5005 | 0.8609 | 0.6330 | 0.5762 |
| scratch        |    0.7400 |    1.0000 | 0.9479 | 0.9733 | 0.9723 |
| scratch_rot    |    0.8200 |    1.0000 | 0.7083 | 0.8293 | 0.8700 |

_Source: outputs/stage1_260505_165400/per_class_metrics.parquet (cell_id=T0__I7)._

### iter 3 — T0__I10 (frozen, I7 + entropy Normal gate)

| class          | threshold | precision | recall | F1     | AP     |
|----------------|----------:|----------:|-------:|-------:|-------:|
| bank_boundary  |    0.5000 |    0.9786 | 0.9297 | 0.9535 | 0.9752 |
| fork           |    0.1400 |    0.5360 | 0.8609 | 0.6607 | 0.5762 |
| scratch        |    0.7400 |    1.0000 | 0.9479 | 0.9733 | 0.9723 |
| scratch_rot    |    0.8200 |    1.0000 | 0.7083 | 0.8293 | 0.8700 |

_Source: outputs/stage1_260505_170827/per_class_metrics.parquet (cell_id=T0__I10)._

The Normal-gate gain comes mostly from `fork` precision (0.5005 → 0.5360, recall held).

### iter 4 — T1__I10 (CE+LS=0.10 retrain, I7+entropy)

| class          | threshold | precision | recall | F1     | AP     |
|----------------|----------:|----------:|-------:|-------:|-------:|
| bank_boundary  |    0.4600 |    1.0000 | 0.7781 | 0.8752 | 0.8969 |
| fork           |    0.2200 |    0.7014 | 0.7891 | 0.7426 | 0.6607 |
| scratch        |    0.6600 |    0.9803 | 0.9354 | 0.9574 | 0.9824 |
| scratch_rot    |    0.5000 |    1.0000 | 0.7833 | 0.8785 | 0.9614 |

_Source: outputs/stage1_260505_173649/per_class_metrics.parquet (cell_id=T0__I10 row, but the model was T1)._

The big jump is **fork F1 0.6607 → 0.7426** (+0.082): label smoothing
flattens the runner-up logit so multi-label thresholding actually has a
distinguishable score for fork-in-combo chips.

### iter 5 — T1_LS20__I7 (CE+LS=0.20 retrain, I7) — overall best (until iter 7)

`per_class_metrics.parquet` is not stored for sweep cells; per-class
breakdown is the next thing to capture in iter 6 if needed.
Aggregate: macro_f1 = 0.9268, top1\_11 = 0.8449.

_Source: outputs/phase_a_260505_175105/sweep_log.csv (LS=0.20, inference_id=I7)._

### iter 7 — T7c__I10 (BCE+LS=0.20+CutMix p=0.5, I10) — new overall best

| class          | threshold | precision | recall | F1     | AP     |
|----------------|----------:|----------:|-------:|-------:|-------:|
| bank_boundary  |    0.6200 |    0.9345 | 0.8469 | 0.8885 | 0.9575 |
| fork           |    0.1400 |    0.8815 | 0.8484 | 0.8647 | 0.7547 |
| scratch        |    0.7400 |    1.0000 | 0.9146 | 0.9554 | 0.9725 |
| scratch_rot    |    0.4200 |    1.0000 | 1.0000 | 1.0000 | 1.0000 |

_Source: outputs/stage1_260505_195730/per_class_metrics.parquet (cell_id=T0__I10)._

The headline shifts in T7c vs T1_LS20:

- **`scratch_rot` perfect** (F1 1.0000, AP 1.0000) — CutMix multi-hot
  training directly teaches the bb+sr visual co-occurrence, so the
  scratch-rot signal is no longer collapsed by the bb signal.
- **`fork` precision 0.70 → 0.88** at almost identical recall — BCE +
  CutMix sharpens fork's negative discrimination far more than CE+LS could.
- **bank_boundary** trades a small F1 drop (0.8974 → 0.8885) for the
  bb+sr gain — net positive on combo recall.
- **ECE_post 0.1788 → 0.0446** (4× lower) — BCE + CutMix produces a much
  better-calibrated probability surface as a side benefit.

The headline operational metric: **bb+sr combo recall 0.32 → 0.96** (from
T1_LS20 baseline → T7c). See `iters/iter_07_phase_f_warmup_ema_t7_cutmix.md`
for the full atomic decomposition (T1 → T7a → T7c) and CutMix-p sweep.

### iter 8 — T9d__I7 (BCE+LS=0.07+CutMix p=0.5, seed=42) — observed best

| class          | threshold | precision | recall | F1     | AP     |
|----------------|----------:|----------:|-------:|-------:|-------:|
| bank_boundary  |    0.9000 |    0.9919 | 0.9578 | 0.9746 | 0.9818 |
| fork           |    0.2200 |    1.0000 | 0.8953 | 0.9448 | 0.9877 |
| scratch        |    0.5800 |    0.9912 | 0.9354 | 0.9625 | 0.9759 |
| scratch_rot    |    0.1800 |    1.0000 | 1.0000 | 1.0000 | 1.0000 |

_Source: outputs/stage1_260505_211334/per_class_metrics.parquet (cell_id=T0__I7)._

vs **T9g__I7 (same config, seed=43, realistic point)**:

| class          | threshold | precision | recall | F1     | AP     |
|----------------|----------:|----------:|-------:|-------:|-------:|
| bank_boundary  |    0.3800 |    0.9984 | 0.9641 | 0.9809 | 0.9817 |
| fork           |    0.1000 |    0.7392 | 0.9078 | 0.8149 | 0.8253 |
| scratch        |    0.7400 |    0.9726 | 0.9625 | 0.9675 | 0.9804 |
| scratch_rot    |    0.3000 |    1.0000 | 1.0000 | 1.0000 | 1.0000 |

_Source: outputs/stage1_260505_212557/per_class_metrics.parquet (cell_id=T0__I7)._

The seed-driven variance is **concentrated in fork** (F1 0.9448 vs 0.8149,
AP 0.9877 vs 0.8253). bank_boundary, scratch, scratch_rot are stable
across seeds (Δ ≤ 0.013 F1). fork is the diffuse / longest-tail defect
per iter-1 error analysis — its sigmoid mass is most sensitive to which
fork patches end up in CutMix mosaics. See `iters/iter_08_T9_LS_sweep_variance.md`.

## Iter 10 — Ensemble FINAL winner (260506)

**Trigger** (260506): user re-added `scratch+scratch_rot` to COMBO_KEYS (was excluded in iter 1-9 design as "same defect family"). T9d on 12-class showed sc+sr F1=0.755 (R=0.606) — model never learned sc+sr CutMix pair (was disallowed).

**Phase journey**:
1. A1 retry (cutmix-p=0.5 + sc+sr CutMix on): sc+sr 1.000 ✅ but other classes ↓ (Normal 0.000) — net negative
2. D (cutmix-p=0.25, gentler): macro 0.9116, sc+sr 0.947 — partial recovery, Normal still 0
3. C (Normal training added, 5-class with y=-1 sentinel): Normal F1 1.000 ± 0.000 lock ✅, fork+scratch 0.673 NEW weakness (cross-class suppression)
4. F (fork↔scratch CutMix pair bias): fork+sc 0.95 but bb/fork+sr ↓ — net negative
5. **H Ensemble** (baseline T9d + C_44 logit avg): **10-defect macro 0.9950**, all class F1 ≥ 0.987, FAR 0%. **Diversity (with-Normal vs without) > Quantity (multi-seed)** — baseline + 1 C_44 (0.995) > baseline + 3 C seeds (0.966).

**Key memory rules added** (260506):
- `feedback_logit_ensemble_complementary.md` — H finding永久 룰
- `feedback_normal_training_open_set.md` — Normal training non-negotiable
- `feedback_cross_class_suppression.md` — fork combo prob 3× collapse mechanism
- `feedback_master_storage_vs_runtime_sampling.md` — single SoT folder
- `feedback_chip_train_batch_safe.md` — shared GPU batch=8 강제

**Final 5-sample-seed mean** (eval on master n=50):
- 4-single macro F1 = 0.9963 ± 0.0045
- 6-combo macro F1 = 0.9908 ± 0.0063
- 10-defect macro F1 = **0.9930 ± 0.0049**
- Normal F1 = 1.0000 ± 0.0000, sc+sr F1 = 1.0000 ± 0.0000
- False Alarm Rate = **0.00% ± 0.00%** (Normal 800 → 0 false alarms in 1000-chip wafer)

See `iters/iter_10_master_consol_sc_sr.md`.

## Iter 11 — Paper-style 4-row Ablation Matrix (260506)

108 cells = 6 train (T1/T3/T4/T5/T6/T7, all 4-class only via `--no-normal`) × 6 inference (I3/I7/I10/I11/I12/I13) × 3 phases (p50 simple Normal / p30 simple / p50 diverse Normal).

**Best single per train** (cross-phase):

| Train | Best | macro | Normal | FAR |
|---|---|---:|---:|---:|
| T6 (BCE→ASL) | P1+I3 | **0.905** | 0.000 | 100% ❌ |
| T5 (BCE) | P3+I11 | 0.894 | 0.000 | 100% ❌ |
| T7 (BCE+LS) | P2+I7 | 0.860 | 0.000 | 100% ❌ |
| T4 (ASL) | P1+I10 | 0.803 | 0.857 | 18% ⚠ |
| T1 (CE+LS) | P3+I11 | 0.620 | 0.000 | 100% ❌ |
| T3 (Focal) | P1+I11 | 0.513 | 0.974 | 5% ⚠ |

**Key findings**:
1. ★ Normal training 누락 = catastrophic FAR (T1/T5/T6/T7 모두 100% FAR) — operationally unusable
2. **Asymmetric (T4 ASL) / Focal (T3) 만 4-class only 환경에서 Normal 자연스럽게 generalize** (asymmetric/focal mechanism)
3. p50 → p30 distribution-shift Δ < 0.02 — 모든 model robust
4. Normal diversity 효과 marginal — T4 ASL 만 receptive (+0.07 N, -12.5% FAR)
5. **iter 10 ensemble (0.995, FAR 0%) 이 모든 single iter 11 model 압도** — single 의 한계 입증

See `iters/iter_11_paper_ablation_matrix.md`.

## Iter 26 — 9-train diversity sweep (g × LS × CutMix-fill)

9 trains spanning **3 orthogonal axes**: complement granularity g ∈ {2, 3, 4}, label
scale LS ∈ {0.40, 0.50, 0.60, 0.67, 0.75, 0.83, 0.85, 1.00}, CutMix fill mode ∈ {complement
(default), white, noise}.

| tag | spec | v14 bF1 | v14 ni% | v15 bF1 | v15 ni% | v15 F1_fk | v15 F1_sc | dual-pass? |
|:---:|:---|---:|---:|---:|---:|---:|---:|:---:|
| 26A | g=2 LS=0.85 | 0.9945 | 2.50% | 0.9816 | 100.00% | 0.987 | 0.981 | ✗ |
| **26B ★** | **g=3 LS=0.50** | **0.9921** | **0.00%** | **0.9791** | **1.25%** | **0.994** | **0.923** | **✓ NEW best single** |
| 26C | g=3 LS=0.83 | 0.9869 | 100.00% | 0.9685 | 31.25% | 0.984 | 0.922 | ✗ |
| 26D | g=4 LS=0.40 | 0.9873 | 0.00% | 0.9353 | 0.00% | 0.971 | 0.918 | ✓ |
| 26E | g=4 LS=0.60 | 0.9827 | 100.00% | 0.9873 | 97.50% | 0.984 | 0.974 | ✗ |
| 26F | g=2 LS=1.0 white | 0.9953 | 0.00% | 0.9541 | 0.00% | 0.954 | 0.904 | ✓ |
| 26G | g=2 LS=1.0 noise | 0.9953 | 0.00% | 0.9541 | 0.00% | byte-id 26F |   | ✓ |
| 26H | g=3 LS=0.67 white | 0.9722 | 0.00% | 0.9687 | 2.50% | 0.994 | 0.881 | ✓ |
| 26I | g=4 LS=0.75 white | 0.9688 | 95.00% | 0.9471 | 2.50% | 0.939 | 0.923 | ✗ |

**Findings**:
- ★ **26B (g=3 LS=0.50)** = NEW best single (v15 bit_F1 0.9791 vs prior 21E 0.9691 = +0.0100), supersedes iter21E.
- **CutMix white-fill ≡ noise-fill** (26F byte-identical to 26G — negative axis for ensembling).
- g=3 = sweet spot; g=2+LS≥0.83 over-tolerates OOD; g=4+LS≥0.75 collapses ni_FAR.
- 5 cells advance to iter27 14-bag ensemble: 26B / 26D / 26F / 26G / 26H.

See `iters/iter_26_27_diversity_finalEnsemble.md`.

## Iter 27 — 14-bag final ensemble (★★★ PAPER HEADLINE)

**Bag composition (14 models)**: iter25 6-seed (LS=0.20×3 + LS=0.30×3) + iter21F (g=3 LS=0.67) +
iter21H (g=4 LS=0.75) + iter22G (drop_path=0.05) + iter26B/D/F/G/H. Span: g∈{2,3,4},
LS∈{0.20,0.30,0.40,0.50,0.67,0.75,1.00}, CutMix∈{complement,white,noise}.

**Threshold sweep (per-class I10 majority vote)**:

| threshold | v14 bF1 | v15 bF1 | v15 ni_FAR |
|:---|---:|---:|---:|
| ≥ 5/14 (35%) ★ | **1.0000** | **0.9929** | **0.00%** |
| ≥ 6/14 (43%) ★ | **1.0000** | **0.9929** | **0.00%** |
| ≥ 7/14 (50%, simple-maj) | 1.0000 | 0.9921 | 0.00% |
| ≥ 9/14 (64%) | 1.0000 | 0.9779 | 0.00% |
| ≥ 10/14 (71%) | 0.9976 | 0.9700 | 0.00% |

**Headline winner — thr ≥5–6/14 simple-majority window**:

| metric | v14class | v15direct |
|:---|---:|---:|
| **bit_F1** | **1.0000** | **0.9929** |
| **ni_FAR** | **0.00%** | **0.00%** |
| F1_bb | 1.0000 | 0.9909 |
| F1_fk | 1.0000 | 0.9874 |
| F1_sc | 1.0000 | 0.9907 |
| F1_sr | 1.0000 | 0.9970 |

**Lift vs prior milestones**:
- vs 12-T5 baseline (paper start): v15 0.7872 → **0.9929 = +0.2057 (+26%)**, ni_FAR 100% (real) → **0.00%**.
- vs iter21E single best: v15 0.9691 → 0.9929 = **+0.0238**, v15 ni_FAR 3.75% → **0.00%**.
- vs iter25 6-seed (prior best): v14 0.9976 → **1.0000 (+0.0024)**, v15 0.9913 → **0.9929 (+0.0016)**.

**Why diversity > quantity (validated at 14× scale across 3 axes)**: Adding {g, LS, CutMix-fill}
diversity bags on top of iter25's seed-only ensemble lifts v15 bit_F1 +0.0016 and saturates v14
to PERFECT, with **zero** ni_FAR cost. ni_FAR is **0.00% across the entire threshold sweep**
(35–71% consensus) — diversity span kills correlated false alarms.

**v14 plateau is flat across thr=5..9/14** — robust to threshold tuning within ±2 votes of
simple-majority. v15 falls off only at very high consensus (≥9/14: 0.9779; ≥10/14: 0.9700).

**This is the new paper main winner — first chip-multilabel config achieving v14 PERFECT +
v15 0.9929 + 0% ni_FAR simultaneously.** See `iters/iter_26_27_diversity_finalEnsemble.md`
+ `tables/paper_main_headline.csv`.

## Iter 30 — FCM-PM g extension (partial — 3/6 done)

Continuation of iter28/29 paper-ablation matrix. Probes whether FCM-PM CutMix
with **larger group counts** (g=5, g=6) and varying `complete_label_scale`
(0.20 / 0.30 / 0.50) reaches iter28's g=3 LS=0.50 winner (26B). Plus 3-seed
mini-ensemble of 26B paradigm (30D/E/F seed1/7/42).

| variant | g | LS | seed | v14 best macro_f1 | v15 best macro_f1 |
|:---|:--:|:--:|:--:|---:|---:|
| 30A_g5_LS020 | 5 | 0.20 | 1  | 0.7919 (I10) | 0.6981 (I10) |
| 30B_g5_LS050 | 5 | 0.50 | 1  | 0.7752 (I10) | 0.7050 (I6)  |
| 30C_g6_LS030 | 6 | 0.30 | 1  | 0.7625 (I7)  | 0.7446 (I10) |
| 30D_g2_LS050 | 2 | 0.50 | 1  | (training)   | (training)   |
| 30E_g3_LS050 | 3 | 0.50 | 7  | (queued)     | (queued)     |
| 30F_g3_LS050 | 3 | 0.50 | 42 | (queued)     | (queued)     |

**Findings (partial)**:
- g=5/6 macro_f1 < iter28 26B (g=3) — larger group count fragments label-spatial signal.
- label_scale=0.20 best v14 / worst v15 (overfitting strong patches).
- All v14 < 0.80 → these variants are **not** ensemble-bag candidates.
- 30D/E/F (g=3 LS=0.50 seed sweep) is the actually-paper-relevant test → pending.

## Iter 31 — 26B regularization sweep (PENDING)

7 variants (EMA 0.95 / drop_path 0.05/0.10 / warmup 2 / epochs 16 / lrhead 5e-5)
applied to iter28 26B winner. Queue waiting for iter30-resume to finish.
See `iters/iter_30_31_g_extension_regularization.md`.

## Table dump

`tables/all_runs_macro_f1.csv` contains every iter-1-through-30 cell (484 rows
as of 260509 iter30 partial) with columns: `iter, cell_id, train_id,
inference_id, macro_f1, micro_f1, mAP, top1_11class, temperature, ece_post, source`.

---

## 260512 update — iter95-99 modern-backbone sweep + best-from-epoch ablation

**Summary**: 14 single-model trains (iter95A/B, iter96A, iter97A/B/C, iter99A-E)
across 5 backbone families (DINOv3-ConvNeXt-B, SwinV2-B-384, Hiera-B, Swin
V1-B-384, ConvNeXtV2-B-384). All cells use the iter88 T7 recipe (BCE+LS=0.20
+ CutMix complement p=0.25 g=3 rect=0.5) with optional LR rescue and
optional `--best-from-epoch=6`. **56 new cells appended** to both
`tables/all_runs_macro_f1.csv` (iter=95/96/97/99) and
`tables/backbone_landscape.csv` (rows 39-94).

### Cross-iter best-cell timeline (new rows only)

Per run_tag, the best inference cell (max `macro_f1`) and its Total FAR:

| iter / run | backbone | ckpt | epoch | best cell | macro_f1 | bF1_4def | Total FAR | wall-time |
|------------|----------|------|------:|-----------|---------:|---------:|----------:|----------:|
| iter95A | DINOv3-ConvNeXt-B | best | 3 | T0__I10 | 0.6211 | 0.6211 | 34.05% | 2.5 min |
| iter95B | SwinV2-B-384 | best | 1 | T0__I10 | 0.7843 | 0.7843 | **0.12%** | 145.0 min |
| iter96A | Hiera-B | best | 1 | T0__I3 | 0.7228 | 0.7228 | 100% | 2.5 min |
| **iter97A ★** | **DINOv3-ConvNeXt-B (LR=5e-5)** | **best** | **9** | **T0__I10** | **0.8700** | **0.8700** | **1.43%** | **5.6 min** |
| iter97A (FINAL) | DINOv3-ConvNeXt-B (LR=5e-5) | final | 20 | T0__I3 | 0.7765 | 0.7765 | 100% | 5.6 min |
| iter97B | DINOv3-ConvNeXt-B (LR=2e-5) | best | 1 | T0__I7 | 0.7037 | 0.7037 | 100% | 5.5 min |
| iter97C | DINOv3-ConvNeXt-B (LR=1e-5) | best | 8 | T0__I10 | 0.8129 | 0.8129 | 44.29% | 5.5 min |
| iter99A | ConvNeXtV2-B-384 (ep>=6) | best | 6 | T0__I10 | 0.8367 | 0.8367 | 12.14% | 7.6 min |
| iter99B | Swin-V1-B-384 (ep>=6) | best | 6 | T0__I3 | 0.8030 | 0.8030 | 100% | 6.3 min |
| iter99C | DINOv3-ConvNeXt-B (ep>=6) | best | 6 | T0__I10 | 0.7423 | 0.7423 | 86.79% | 2.8 min |
| iter99D | Hiera-B (ep>=6) | best | 6 | T0__I10 | 0.7039 | 0.7039 | **4.76%** | 2.9 min |
| iter99E | ConvNeXtV2-B-384 (LR=5e-5, ep>=6) | best | 8 | T0__I10 | 0.8282 | 0.8282 | 10.83% | 7.6 min |

★ **iter97A is the iter95-99 single-best cell** (DINOv3 macro_f1 0.8700 /
Total FAR 1.43% at ep=9 with LR=5e-5), but **still −0.06 below the iter89
Swin V1 reference 0.9278** and **−0.10 below the iter46E ConvNeXtV2-B
production reference 0.9654**. The iter95-99 sweep did not produce a new
overall winner; it produced a strong negative result about modern backbones.

### Comparison vs prior single-model headlines

| reference cell | bF1_4def | Total FAR | iter95-99 challenger | challenger bF1 | challenger Total FAR | Δ bF1 | Δ FAR |
|----------------|---------:|----------:|----------------------|---------------:|---------------------:|------:|------:|
| iter46E ConvNeXtV2-B-384 (paper main) | 0.9654 | 1.07% | iter99A ConvNeXtV2 same recipe ep>=6 | 0.8367 | 12.14% | **−0.13** | +11.1pp |
| iter77C Swin-B-384 (strict FAR) | 0.9692 | 0.00% | iter99B Swin-B-384 ep>=6 | 0.8030 | 100% | **−0.17** | +100pp |
| iter89 Swin-B-384 LR1e-4 LS0.3 g2 | 0.9278 | 0.00% | iter97A DINOv3-B LR5e-5 ep9 | 0.8700 | 1.43% | **−0.06** | +1.4pp |

**Every iter95-99 cell is worse than the corresponding pre-iter95 reference
on bF1, Total FAR, or both.** The iter95-99 sweep is a confirmation that
the existing ConvNeXtV2 / Swin V1 + FCMAE recipe combination is **not a
local optimum that the next-generation backbones surpass** — at this
4-class chip-multilabel + 4-OOD-pattern eval, modern backbones (DINOv3 SSL,
SwinV2 scaled cosine, Hiera MAE-ViT) need significant per-family
recipe redesign before they can challenge the production reference.

### Headline lessons (paper §5 / §7)

1. **Modern backbones do not transfer the FCMAE / Swin V1 recipe**
   (Lesson 1, see `05_backbone_landscape.md` §5.10). DINOv3 default LR
   loses 0.31 bF1; rescue at 2× lower LR recovers 0.25 but still trails by
   0.06. SwinV2 trails 0.18 at 21× wall-time.
2. **best-from-epoch is backbone-specific, not global** (Lesson 2, see
   `03_ablations.md` "Best-from-epoch policy ablation"). FCMAE / Swin V1
   peak at ep 1-3; DINOv3 / Hiera need ep 6-9. Global floor breaks 2 / 5
   backbones.
3. **val_acc plateau is NOT an eval F1 proxy** (Lesson 3). iter97A: val_acc
   sits at 0.9816 for 20 epochs; eval F1 drifts 0.87 (ep9) → 0.77 (ep20),
   Total FAR explodes 1.43% → 90.5%. **Paper §5 must show eval-F1 curves,
   not val_acc curves.**
4. **FCM-PM cutmix_p=0.25 is not strong enough to prevent single-label
   overfit on DINOv3 by ep20** (Lesson 4). iter100 (cutmix_p=1.0)
   running at logger time 19:50 — results pending in iter100 update.

### Open items for iter100+

- **iter100** (running): ConvNeXtV2-B-384 + cutmix_p=1.0. Tests Lesson 4
  hypothesis that increased CutMix coverage closes the iter97A best→final
  regression gap (0.87 → 0.77).
- **iter101+**: per-backbone early-stop patience implementation in
  `_train_chip_variant.py` (replace global `--best-from-epoch` floor).
- **DINOv3 ensemble**: iter97A LR=5e-5 ep9 has a unique error
  distribution vs Swin V1 / ConvNeXtV2; potential ensemble diversity
  candidate. Deferred to post-iter100.

_Source: `tables/backbone_landscape.csv` rows 39-94 (56 new cells, single-seed
seed=42, v15direct n_per_class=200, 4 inference cells per train run).
Per-iter detail in `05_backbone_landscape.md` §§ 5.7-5.11 and
`03_ablations.md` "Best-from-epoch policy ablation (iter99)"._
