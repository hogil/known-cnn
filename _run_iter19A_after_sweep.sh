#!/bin/bash
# 260508 — iter19A retrain AFTER sweep B-L done
# Polls outputs/_iter19_complement_resume.log for "DONE" line, then runs 19A retrain.
set -e
cd /d/project/known-cnn
WAIT_LOG=outputs/_iter19_complement_resume.log
RUN_LOG=outputs/_iter19A_retrain.log

echo "$(date) [iter19A-after-sweep] waiting for sweep B-L to finish ..." > "$RUN_LOG"
until grep -q "\[iter19-resume\] DONE" "$WAIT_LOG" 2>/dev/null; do
    sleep 30
done
echo "$(date) [iter19A-after-sweep] sweep done — starting 19A retrain" >> "$RUN_LOG"

bash _run_iter19A_retrain.sh
echo "$(date) [iter19A-after-sweep] 19A retrain DONE" >> "$RUN_LOG"
