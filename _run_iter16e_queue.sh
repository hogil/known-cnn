#!/bin/bash
# 260508 iter 16-E queue: wait for 16-D train to finish, then train + eval 16-E.
# Also run eval on 16-C and 16-D after 16-E since neither has eval yet.
set -e
cd /d/project/known-cnn

LOG_DIR=outputs
QUEUE_LOG="${LOG_DIR}/_iter16E_queue.log"
echo "$(date) [queue] start, waiting for 16-D train to finish" > "${QUEUE_LOG}"

# Poll: 16-D done when train_summary.json + final_epoch_model.pth both exist
target_dir="outputs/iter16D_T7N_pair_grid_v52/T7_T7_iter16D_pair_grid_seed1_260508_093633"
while true; do
  if [ -f "${target_dir}/train_summary.json" ] && [ -f "${target_dir}/final_epoch_model.pth" ]; then
    echo "$(date) [queue] 16-D train DONE" >> "${QUEUE_LOG}"
    break
  fi
  sleep 30
done

# Step 1 — train iter 16-E (grid_sparse + paired CutMix, seed 1)
echo "$(date) [queue] train 16-E grid_sparse+pair seed 1" >> "${QUEUE_LOG}"
TRAIN_LOG="${LOG_DIR}/_iter16E.log"
python -m chip_multilabel._train_chip_variant \
  --variant T7 --ls 0.20 \
  --epochs 8 --batch 8 --accum 4 --seed 1 \
  --cutmix-p 0.25 --cutmix-rect 0.5 \
  --cutmix-mode grid_sparse --cutmix-grid-k 8 \
  --cutmix-pair masked --cutmix-pair-loss-w 1.0 --cutmix-pair-fill corner \
  --out-root outputs/iter16E_T7N_pair_grid_sparse_v52 \
  --tag T7_iter16E_pair_grid_sparse_seed1 \
  > "${TRAIN_LOG}" 2>&1
echo "$(date) [queue] train 16-E DONE" >> "${QUEUE_LOG}"

# Step 2 — eval 16-C, 16-D, 16-E on multi-label master
EVAL_SET=/d/project/data/wm-811k/chip_multilabel
for run_glob in \
  "outputs/iter16C_T7N_pair_scattered_v52/T7_T7_iter16C_pair_scattered_seed1_260508_093103" \
  "outputs/iter16D_T7N_pair_grid_v52/T7_T7_iter16D_pair_grid_seed1_260508_093633" \
  "outputs/iter16E_T7N_pair_grid_sparse_v52/T7_iter16E_pair_grid_sparse_seed1*"; do
  RUN_DIR=$(ls -d ${run_glob} 2>/dev/null | head -1)
  if [ -z "${RUN_DIR}" ]; then
    echo "$(date) [queue] WARN no run dir for ${run_glob}" >> "${QUEUE_LOG}"
    continue
  fi
  echo "$(date) [queue] eval ${RUN_DIR}" >> "${QUEUE_LOG}"
  python -m chip_multilabel.run_stage1 \
    --model "${RUN_DIR}/best_model.pth" \
    --eval-set "${EVAL_SET}" \
    --out-root "${RUN_DIR}/eval_seed1" \
    --variants I3 \
    --n-per-class 50 \
    --strength-min 0.5 \
    --seed 42 \
    >> "${QUEUE_LOG}" 2>&1
done
echo "$(date) [queue] all eval DONE" >> "${QUEUE_LOG}"

# Step 3 — print 4-row summary
echo "" >> "${QUEUE_LOG}"
echo "=== 4-row CutMix mode × pair comparison (seed 1) ===" >> "${QUEUE_LOG}"
python -c "
import json, glob, os
runs = {
    '16-A single no_pair':   'outputs/iter16A_T7N_baseline_v52/T7_T7_iter16A_baseline_seed1_260508_074646',
    '16-B single + pair':    'outputs/iter16B_T7N_pair_v52/T7_T7_iter16B_paired_seed1_260508_074809',
    '16-C scattered + pair': 'outputs/iter16C_T7N_pair_scattered_v52/T7_T7_iter16C_pair_scattered_seed1_260508_093103',
    '16-D grid + pair':      'outputs/iter16D_T7N_pair_grid_v52/T7_T7_iter16D_pair_grid_seed1_260508_093633',
}
e_glob = glob.glob('outputs/iter16E_T7N_pair_grid_sparse_v52/T7_iter16E_pair_grid_sparse_seed1*')
if e_glob: runs['16-E grid_sparse + pair'] = e_glob[0]
print(f'{\"iter\":<28} {\"macro_f1\":>10} {\"top1_11c\":>10}')
for label, rd in runs.items():
    eg = sorted(glob.glob(f'{rd}/eval_seed1/stage1_*/eval_summary.json'))
    if not eg:
        print(f'{label:<28} {\"(no eval)\":>10} {\"\":>10}')
        continue
    with open(eg[-1]) as f: d = json.load(f)
    mf1 = d.get('stage1', {}).get('best_macro_f1', float('nan'))
    cells = d.get('stage1', {}).get('all_cells', [])
    top = cells[0].get('top1_11class', float('nan')) if cells else float('nan')
    print(f'{label:<28} {mf1:>10.4f} {top:>10.4f}')
" >> "${QUEUE_LOG}"

echo "$(date) [queue] DONE" >> "${QUEUE_LOG}"
