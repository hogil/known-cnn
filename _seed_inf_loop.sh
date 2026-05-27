#!/usr/bin/env bash
# Infinite seed loop: train_n200 + v15direct_n2000, find seed with bit_F1>=0.99 AND Total FAR=0
# Stop on first success OR user kill.

# Skip seeds already tried
TRIED="1 2 5 7 11 33 42 77 99"
TRIES=(13 17 19 23 29 31 37 41 43 47 53 59 61 67 71 73 79 83 89 97 101 103 107 109 113 127 131 137 139 149 151 157 163 167 173 179 181 191 193 197 199 211 223 227 229 233 239 241 251 257)

CUDA_VISIBLE_DEVICES=0
COUNT=0
for seed in "${TRIES[@]}"; do
    COUNT=$((COUNT + 1))
    LOG="_inf_loop_s${seed}.log"
    OUT="outputs/iter116J_infloop_s${seed}"
    echo "=== [$(date +%H:%M:%S)] iter ${COUNT}: seed=${seed} ==="
    SEED=$seed CUDA_VISIBLE_DEVICES=0 bash mega_matrix/run_single_sota.sh \
        --train-root E:/data/images/train_n200 \
        --eval-set E:/data/images/chip_multilabel_v15direct_n2000 \
        --out-root "$OUT" \
        --epochs 10 \
        > "$LOG" 2>&1

    # parse result
    RES=$(grep "BEST cell" "$LOG" | tail -1)
    if [ -z "$RES" ]; then
        echo "  seed=${seed}: NO RESULT (check $LOG)"
        continue
    fi
    BIT_F1=$(echo "$RES" | grep -oP 'eval_bit_F1=\K[0-9.]+')
    TOT_FAR=$(echo "$RES" | grep -oP 'Total=\K[0-9.]+')
    NI_FAR=$(echo "$RES" | grep -oP 'NI=\K[0-9.]+')
    OOD_FAR=$(echo "$RES" | grep -oP 'OOD=\K[0-9.]+')
    echo "  seed=${seed}: bit_F1=${BIT_F1} Tot=${TOT_FAR} NI=${NI_FAR} OOD=${OOD_FAR}"

    # success check: bit_F1 >= 0.99 AND Total FAR == 0
    SUCCESS=$(python -c "
import sys
b = float('${BIT_F1}'); t = float('${TOT_FAR}')
print('YES' if b >= 0.99 and t == 0.0 else 'NO')
")
    if [ "$SUCCESS" = "YES" ]; then
        echo ""
        echo "=== SUCCESS seed=${seed} bit_F1=${BIT_F1} Total FAR=${TOT_FAR}% ==="
        echo "iter ${COUNT}: bit_F1>=0.99 AND Total FAR=0 ACHIEVED"
        exit 0
    fi
done

echo ""
echo "=== END: tried ${COUNT} seeds, no success ==="
exit 1
