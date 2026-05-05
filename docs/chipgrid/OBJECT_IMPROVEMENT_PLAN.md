# Same-Distribution Object Improvement

This queue targets the remaining failure mode where the wafer distribution is correct but the object type changes, especially Edge-Top/Edge-Bottom object swaps.

## Added paths

- `_build_obj_id_maps.py --save-prob-maps` writes chip-CNN softmax maps to `D:/project/data/wm-811k/obj_prob_maps`.
- `cnn_eval_chipgrid.py --variant V4` reads those soft maps as five object-probability channels.
- `cnn_eval_chipgrid.py --aux-heads factorized` adds distribution/object auxiliary heads.
- `--hard-contrastive-weight` adds a supervised contrastive loss on selected hard classes.

## Build V4 maps

```powershell
python _build_obj_id_maps.py `
  --chip-model logs_chip/overall/best_model.pth `
  --batch 128 `
  --device cuda `
  --save-prob-maps
```

Existing hard obj_id maps are not rewritten when only V4 probability maps are missing.

## Main queue

```powershell
.\experiments\run_chipgrid_object_plan.ps1 -BuildProbMaps
```

Smoke check:

```powershell
.\experiments\run_chipgrid_object_plan.ps1 -Smoke
```

Smoke mode builds a tiny temporary hard/prob map set under `logs_chipgrid_smoke` and passes those paths into V3/V4.

The queue logs resource snapshots before, during, and after each Python process. Defaults:

- GPU memory kill threshold: `92%`
- CPU memory kill threshold: `90%`
- GPU temperature kill threshold: `82C`
- GPU utilization is logged; utilization kill is disabled by default with `-GpuUtilMaxPct 100`

Example with stricter memory limits:

```powershell
.\experiments\run_chipgrid_object_plan.ps1 -BuildProbMaps -GpuMemMaxPct 85 -CpuMemMaxPct 85
```

Primary comparison should use paired seeds `42,1,2,3,4` and report:

- `macro_f1`
- `object_acc_intra_dist`
- `edge_object_acc`
- Edge-Top/Edge-Bottom confusion count from per-class reports or `_objid_diagnosis.py`

Recommended decision order:

1. `v3_objonly` vs `v4_soft_objonly`: does preserving chip uncertainty help?
2. `v4_soft_objonly` vs `v4_factorized`: does explicit dist/object decomposition help?
3. `v4_factorized` vs `v4_factorized_edge_supcon`: does hard edge contrastive separation reduce same-distribution object swaps?
