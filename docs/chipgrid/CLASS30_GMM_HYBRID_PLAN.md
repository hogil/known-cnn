# Class-30 + GMM Hybrid Plan

Updated: 2026-05-03

This note records the current project-specific plan for reducing the saturated wafer classes, adding object-less wafer-canvas classes, and using distribution models as features for CNN fusion.

## Resource Rule

Before any long run:

- Check GPU util, VRAM, temperature with `nvidia-smi`.
- Check CPU memory with `Get-CimInstance Win32_OperatingSystem`.
- During long runs, kill only the process tree that this queue started if GPU memory, CPU memory, or GPU temperature crosses the configured limit.
- Do not kill unrelated user/Codex/Claude processes unless explicitly requested.

Current guarded queue example: `experiments/run_chipgrid_object_plan.ps1`.

## Project Files Actually Relevant

| file | why it matters |
|---|---|
| `_sample_gen.py` | CPU generator, object distributions, `DEFECT_BUDGET`, current base `build_tasks`. It is not the active path for the new wafer-canvas classes. |
| `_sample_gen_gpu.py` | Active GPU generator. Contains 10 wafer-canvas freehand classes via `WAFER_CANVAS_PATTERNS`. |
| `_dist_learn_per_class.py` | Existing per-class distribution-learning ablation. Important rule: if class name contains object, use that object; otherwise use all defect chips. |
| `_chipgrid_kde_gmm.py` | Existing Stage A proof: KDE on chip positions + GMM on object-count vector. This is not the full four-option GMM plan. |
| `_chipgrid_gmm_options.py` | Stage A implementation for alpha/beta/gamma/delta GMM feature options. Saves per-option score matrices for later fusion. |
| `cnn_train_chipgrid_fusion.py` | Existing Stage B proof: V3 image branch + KDE/GMM aux vector. It does not cover R-only/wafer CNN fusion yet. |
| `configs/chipgrid_class20_hard.yaml` | Runnable active-class config for the existing hard/special 20 classes. |
| `configs/chipgrid_class30_target.yaml` | Target 30-class active list. It should fail until the 8 new wafer-canvas classes are generated. |
| `docs/chipgrid/RESULTS.md` | V3 full-data and per-class weakness. Edge-Top/Bottom remain the main same-distribution object confusion. |
| `results_intra_dist/summary.json` | Intra-distribution object metrics. V3 is strong overall but Edge-Bottom still lower than saturated groups. |
| `results_disagree/summary.json` | Confusion evidence for deciding which existing object classes to keep. |

## Current Data Reality

`D:/project/data/wm-811k/unknown` currently has 34 trainable folders used by chipgrid:

- 30 regular distribution-object classes: `Center/Donut/Edge-Ring/Edge-Bottom/Edge-Top/Full × 5 objects`.
- 1 special object class: `Thick-Edge_invalid_main`.
- 1 object class with tiny support: `Normal_bank_boundary` with 20 images.
- 2 object-less wafer-canvas classes: `Starburst`, `CommaCluster`.
- `Normal` exists with 5000 images but is excluded from chipgrid training.

Correction: `Normal_bank_boundary` is not object-less. It contains `bank_boundary`. `Thick-Edge_invalid_main` contains `invalid_main`, but should stay because its distribution is special.

## Target Dataset Shape

User target:

- Keep 20 existing object-bearing or special classes.
- Add/keep 10 object-less wafer-canvas classes.
- Total: 30 classes.

This makes the task less dominated by saturated easy object classes and adds true wafer-pattern classes where object ID is not the shortcut.

## Proposed 20 Existing Classes

Keep classes that are hard, confusion-relevant, or structurally special:

1. `Edge-Bottom_bank_boundary`
2. `Edge-Bottom_invalid_main`
3. `Edge-Bottom_particle_blast`
4. `Edge-Bottom_scratch`
5. `Edge-Bottom_scratch_21deg`
6. `Edge-Top_bank_boundary`
7. `Edge-Top_invalid_main`
8. `Edge-Top_particle_blast`
9. `Edge-Top_scratch`
10. `Edge-Top_scratch_21deg`
11. `Donut_bank_boundary`
12. `Donut_invalid_main`
13. `Donut_particle_blast`
14. `Donut_scratch`
15. `Donut_scratch_21deg`
16. `Center_invalid_main`
17. `Edge-Ring_invalid_main`
18. `Full_invalid_main`
19. `Thick-Edge_invalid_main`
20. `Edge-Ring_particle_blast`

Rationale:

- Edge-Top/Bottom all stay because same-distribution different-object confusion is the known weak point.
- Donut all stay because it is a non-trivial non-Gaussian spatial distribution and still tests object separation.
- `Thick-Edge_invalid_main` stays regardless of current score because it is distribution-special.
- Additional invalid classes keep invalid-vs-defect signal from collapsing into one distribution.
- `Edge-Ring_particle_blast` is a non-edge-row object-bearing sanity anchor from disagreement evidence.

Do not hard-delete data. Implement this as a subset config first, then regenerate a clean dataset only after the subset result is useful.

Runnable config:

```powershell
python cnn_eval_chipgrid.py --variant V3 --no-r-channel --active-classes-yaml configs/chipgrid_class20_hard.yaml
python _chipgrid_kde_gmm.py --active-classes-yaml configs/chipgrid_class20_hard.yaml
python cnn_train_chipgrid_fusion.py --active-classes-yaml configs/chipgrid_class20_hard.yaml
```

The chipgrid, KDE/GMM, and fusion scripts now default to strict active-class checking. If a listed class is missing, the run stops instead of silently becoming the wrong class count.

## Object-Less 10 Wafer-Canvas Classes

Existing:

1. `Starburst`
2. `CommaCluster`

Implemented in `_sample_gen_gpu.py` using full-wafer alpha fields, not chip-grid object patterns:

3. `DiagonalSmear` - long diagonal contact/smear across the wafer.
4. `CrossScratch` - two or more crossing full-wafer scratches.
5. `CrescentArc` - broad crescent edge contact.
6. `SpiralTrail` - curved rotational drag mark.
7. `ParallelScratches` - several freehand near-parallel wafer-scale scratches.
8. `EdgeSmudge` - broad fuzzy edge-localized smudge.
9. `BlobChain` - chain of soft round contacts along a path.
10. `BrokenRing` - large discontinuous ring segments, distinct from small comma arcs.

Generation rule:

- These classes should set `chip_meta[(gy,gx)]["obj"] = None`.
- They should not save object-specific chip crops as if they were one of the 5 registered chip objects.
- Classification should rely on wafer geometry/palette, not object ID.

Generation examples:

```powershell
# one class smoke generation after GPU is free
python _sample_gen_gpu.py --only-class DiagonalSmear --n 5 --seed-offset 9000000

# full 10 object-less class generation uses the normal generator task list.
# Use a seed offset if appending to an existing data root to avoid filename collision.
python _sample_gen_gpu.py --n 200 --seed-offset 9000000
```

## GMM/KDE Feature Options To Actually Compare

The current `_chipgrid_kde_gmm.py` has only:

- KDE over all non-zero chip positions.
- GMM over 5-D object-count vectors.

That is useful, but incomplete. The real comparison should implement:

| option | feature | purpose |
|---|---|---|
| alpha | 2-D chip positions, all defect chips | position-only baseline |
| beta | 2-D positions filtered by target object from class name | user-preferred object-aware distribution with low dimension |
| gamma | wafer-level summary: chip count, mean/std x/y, dominant object, entropy | robust low-dimensional wafer descriptor |
| delta | object-wise binary-map moments | explicit per-object spatial shape |

For beta, if class name has no object, use all defect chips. For object-bearing classes, only use the target object where possible. If JSON lacks chip object, fall back to obj_id map.

Implemented runner:

```powershell
python _chipgrid_gmm_options.py `
  --active-classes-yaml configs/chipgrid_class20_hard.yaml `
  --n-per-class 100 `
  --n-components 4 `
  --save-features
```

Outputs go to `results_gmm_options/<tag>_<timestamp>/` with `summary.json` and optional `<option>_features.npz`.

## Fusion Backbones

Do not restrict the hybrid feature to V3 only.

Priority comparison:

1. R-only ConvNeXt + GMM/KDE feature: wafer CNN carries palette geometry; GMM carries explicit distribution likelihood.
2. Chipgrid proof fusion with selectable `--image-branch r-only|v3|r-plus-v3`: fast sanity check before ConvNeXt-scale runs.
3. V3 chipgrid + GMM/KDE feature: object-grid CNN carries object pattern; GMM feature may regularize sparse Edge-Top/Bottom cases.
4. 3-channel FCMAE input: channel 1 = palette grayscale, channel 2 = obj_id map expanded, channel 3 = zero dummy. Use the pretrained 3-channel stem as-is, then fine-tune.
5. Adapter input: multi-channel maps -> small adapter -> pretrained ConvNeXt 3-channel stem.

## Literature Anchors Checked

- [ConvNeXt V2 / FCMAE, CVPR 2023](https://arxiv.org/abs/2301.00808): supports keeping the pretrained 3-channel stem/adapters as a serious route, rather than randomizing the first layer.
- [DINOv2, 2023](https://arxiv.org/abs/2304.07193): supports the self-supervised backbone idea, but it is a larger follow-up because this project's labeled set is small and current GPU is already busy.
- [CLIP, ICML 2021](https://proceedings.mlr.press/v139/radford21a.html): motivates multi-modal contrastive alignment; in this project that maps to wafer image branch + object-grid branch.
- [Geometric transformation-invariant CNN for wafer maps, Scientific Reports 2023](https://www.nature.com/articles/s41598-023-34147-2): supports preserving wafer-level geometry instead of relying only on object ID.
- [Noise-robust CNN-ESN hybrid wafer map classification, 2026](https://pmc.ncbi.nlm.nih.gov/articles/PMC13029249/): supports hybrid representations for perturbation/noise robustness, matching the GMM/CNN fusion direction.

## Existing Result To Keep In Mind

`logs_chipgrid_kde_gmm/kdegmm_n100_bw1.0_k2_260503_190108`:

- VAL F1: 86.48%
- TEST F1: 89.48%
- KDE-only VAL F1: 26.13%
- GMM-only VAL F1: 81.56%

Interpretation:

- Distribution priors alone are weaker than V3.
- They can still be useful as auxiliary features.
- KDE position-only is too weak in the current formulation; do not over-invest in plain surface lookup.

`logs_chipgrid_fusion/fusion_seed42_260503_190759_running` is an active V3+KDE/GMM fusion run. Treat it as a proof run, not the final design.

## Next Implementation Order

1. Record the above in docs/skills and stop launching new GPU runs until active runs finish or resources are safe.
2. Use `configs/chipgrid_class20_hard.yaml` for immediate class-pruning checks without deleting data.
3. Generate the 8 new object-less wafer-canvas patterns when GPU is free.
4. Switch to `configs/chipgrid_class30_target.yaml` after the generated class folders exist.
5. Run `_chipgrid_gmm_options.py` for alpha/beta/gamma/delta using split-safe train-only fitting.
6. Implement hybrid features for R-only and V3 separately.
7. Compare on paired seeds and report per-class confusion, especially Edge-Top/Bottom and new object-less classes.
