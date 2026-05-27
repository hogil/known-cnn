#!/usr/bin/env bash
# Resource watchdog v3 — PER-PROCESS project budget (260527)
#
# 측정 대상: 이 프로젝트의 python 학습/평가 프로세스만
#   (chip_multilabel._train_chip_variant / run_stage1)
#   공유 baseline (다른 사람 30-40%) 은 제외 — 프로젝트 자기 사용량만 본다.
#
# 한계 (사용자 directive 260527):
#   GPU : 40% of 16 GB = 6400 MiB  (프로젝트 python 합)
#   RAM : 20% of 64 GB = 13000 MiB (프로젝트 python RSS 합)
#   CPU : 20% of 16 cores          (프로젝트 python %CPU 합 / nproc)
#
# 위반 시: 해당 프로세스 kill -9. 매 30 sec.
# 시작: nohup bash _resource_watchdog.sh > _resource_watchdog.log 2>&1 &
# 종료: pkill -f "_resource_watchdog.sh"

GPU_LIMIT_MIB=6400
RAM_LIMIT_MIB=13000
CPU_LIMIT_PCT=20
NCORES=$(nproc 2>/dev/null || echo 16)
CHECK_INTERVAL=30
PATTERN="chip_multilabel\.(_train_chip_variant|run_stage1)"

echo "[$(date +%Y%m%d_%H%M%S)] watchdog v3 start GPU=${GPU_LIMIT_MIB}MiB RAM=${RAM_LIMIT_MIB}MiB CPU=${CPU_LIMIT_PCT}%(of ${NCORES}core) per-process INTERVAL=${CHECK_INTERVAL}s"

while true; do
    ts=$(date +%H:%M:%S)

    # project pids
    project_pids=$(ps -ef 2>/dev/null | grep -E "$PATTERN" | grep -v grep | awk '{print $2}')
    n_proj=$(echo "$project_pids" | grep -v '^$' | wc -l)

    if [ -z "$project_pids" ]; then
        sleep $CHECK_INTERVAL; continue
    fi

    # --- GPU: sum project pids' GPU mem from compute-apps ---
    gpu_proj_mib=0
    apps=$(nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader,nounits 2>/dev/null)
    for pid in $project_pids; do
        m=$(echo "$apps" | awk -F',' -v p="$pid" '{gsub(/ /,"",$1); if($1==p){gsub(/ /,"",$2); print $2}}')
        [ -n "$m" ] && gpu_proj_mib=$((gpu_proj_mib + m))
    done

    # --- RAM: sum project pids' VmRSS ---
    ram_kb=0
    for pid in $project_pids; do
        rss=$(grep VmRSS /proc/$pid/status 2>/dev/null | awk '{print $2}')
        [ -n "$rss" ] && ram_kb=$((ram_kb + rss))
    done
    ram_mib=$((ram_kb / 1024))

    # --- CPU: PowerShell %Processor Time of all python procs, normalized by cores (best-effort) ---
    cpu_pct=$(powershell -NoProfile -Command "
        try {
          \$s=Get-Counter '\Process(python*)\% Processor Time' -EA Stop;
          \$t=(\$s.CounterSamples | Measure-Object -Property CookedValue -Sum).Sum / $NCORES;
        } catch { \$t=0 }
        [math]::Round(\$t,1)
    " 2>/dev/null)
    [ -z "$cpu_pct" ] && cpu_pct=0
    cpu_int=$(printf "%.0f" "$cpu_pct" 2>/dev/null || echo 0)

    action=""; violation=0
    if [ "$gpu_proj_mib" -gt "$GPU_LIMIT_MIB" ]; then action="${action} GPU_OVER=${gpu_proj_mib}MiB"; violation=1; fi
    if [ "$ram_mib" -gt "$RAM_LIMIT_MIB" ]; then action="${action} RAM_OVER=${ram_mib}MiB"; violation=1; fi
    if [ "$cpu_int" -gt "$CPU_LIMIT_PCT" ]; then action="${action} CPU_OVER=${cpu_int}%"; violation=1; fi

    if [ "$violation" -eq 1 ]; then
        kill -9 $project_pids 2>/dev/null
        action="${action} KILLED=$(echo $project_pids | tr '\n' ',')"
    fi

    # 2+ train running → kill all but oldest
    train_pids=$(ps -ef 2>/dev/null | grep -E "chip_multilabel._train_chip_variant" | grep -v grep | awk '{print $2}' | sort -n)
    n_train=$(echo "$train_pids" | grep -v '^$' | wc -l)
    if [ "$n_train" -ge 2 ]; then
        to_kill=$(echo "$train_pids" | tail -n +2 | tr '\n' ' ')
        kill -9 $to_kill 2>/dev/null
        action="${action} DUP_KILL=$to_kill"
    fi

    if [ -n "$action" ]; then
        echo "[$ts] GPUproj=${gpu_proj_mib}MiB RAM=${ram_mib}MiB CPU=${cpu_int}% n=${n_proj} ACTION:${action}"
    fi
    sleep $CHECK_INTERVAL
done
