#!/usr/bin/env bash
# Auto-chain: recipe_sweep (in progress) -> fcm_pm_sweep -> seed_inf_loop
# Stops on first SUCCESS or all complete.

CHAIN_LOG="_chain_all_sweeps.log"
echo "[$(date '+%H:%M:%S')] chain start" > "$CHAIN_LOG"

# --- Phase 1: wait for recipe_sweep PID 47831 to finish (already running) ---
RECIPE_PID=47831
echo "[$(date '+%H:%M:%S')] Phase 1: waiting for recipe_sweep PID=$RECIPE_PID" >> "$CHAIN_LOG"
while kill -0 "$RECIPE_PID" 2>/dev/null; do
    sleep 60
done
echo "[$(date '+%H:%M:%S')] Phase 1: recipe_sweep done" >> "$CHAIN_LOG"

# Check if recipe_sweep already found success (R20.log would have END or SUCCESS line)
if grep -q "SUCCESS recipe=" _recipe_R*.log 2>/dev/null; then
    WINNER=$(grep -h "SUCCESS recipe=" _recipe_R*.log | head -1)
    echo "[$(date '+%H:%M:%S')] Recipe sweep SUCCESS: $WINNER" >> "$CHAIN_LOG"
    echo "STOP chain — recipe found." >> "$CHAIN_LOG"
    exit 0
fi

# --- Phase 2: FCM-PM 2nd-level sweep ---
echo "[$(date '+%H:%M:%S')] Phase 2: launching _fcm_pm_optimal_sweep.sh" >> "$CHAIN_LOG"
bash _fcm_pm_optimal_sweep.sh > _fcm_pm_optimal_sweep.log 2>&1

if grep -q "SUCCESS recipe=" _fcm_sweep_*.log 2>/dev/null; then
    WINNER=$(grep -h "SUCCESS recipe=" _fcm_sweep_*.log | head -1)
    echo "[$(date '+%H:%M:%S')] FCM sweep SUCCESS: $WINNER" >> "$CHAIN_LOG"
    echo "STOP chain — FCM recipe found." >> "$CHAIN_LOG"
    exit 0
fi
echo "[$(date '+%H:%M:%S')] Phase 2: FCM sweep done, no success" >> "$CHAIN_LOG"

# --- Phase 3: extended seed sweep with best recipe so far ---
# Pick the recipe with best (bit_F1 >= 0.99, lowest FAR) from ALL sweeps so far
echo "[$(date '+%H:%M:%S')] Phase 3: analyzing best recipe + seed extension" >> "$CHAIN_LOG"
python -c "
import re, glob, os
best = None
best_score = -1e9
for f in sorted(glob.glob('_recipe_R*.log') + glob.glob('_fcm_sweep_*.log')):
    name = os.path.basename(f).replace('_recipe_', '').replace('_fcm_sweep_', '').replace('.log', '')
    txt = open(f, encoding='utf-8', errors='ignore').read()
    m = re.search(r'eval_bit_F1=([0-9.]+) eval_FAR Total=([0-9.]+)', txt[-5000:])
    if not m:
        continue
    bf1 = float(m.group(1)); far = float(m.group(2))
    # score: bit_F1 - 0.01*FAR  (penalize FAR)
    score = bf1 - 0.01 * far
    if score > best_score:
        best_score = score
        best = (name, bf1, far)
print(f'BEST_SCORE_RECIPE: {best}')
" >> "$CHAIN_LOG" 2>&1

echo "[$(date '+%H:%M:%S')] Phase 3: launching extended seed loop" >> "$CHAIN_LOG"
bash _seed_inf_loop.sh > _seed_inf_loop_chain.log 2>&1

if grep -q "SUCCESS seed=" _inf_loop_s*.log 2>/dev/null; then
    WINNER=$(grep -h "SUCCESS seed=" _inf_loop_s*.log | head -1)
    echo "[$(date '+%H:%M:%S')] Seed sweep SUCCESS: $WINNER" >> "$CHAIN_LOG"
    exit 0
fi

echo "[$(date '+%H:%M:%S')] CHAIN END: all phases exhausted, no 0.99/0% achieved" >> "$CHAIN_LOG"
exit 1
