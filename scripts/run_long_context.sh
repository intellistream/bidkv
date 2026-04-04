#!/bin/bash
# Run all 7 strategies for long_context workload
# Sequential execution to avoid GPU/port conflicts
set -euo pipefail

DIR=/home/cyb/bidkv
OUT=$DIR/results/vllm_v8_long_context
LOG=$DIR/results/long_context_progress.log
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
CMD="conda run --no-capture-output -n sagellm python -m bidkv.experiments.vllm.runner"
COMMON="--workloads long_context --runs 3 --num-gpu-blocks-override 600 --gpu-memory-utilization 0.5 --output-dir $OUT"

mkdir -p "$OUT"
echo "[$(date)] Starting long-context experiments (7 strategies × 3 rates × 3 runs = 63)" | tee -a $LOG

STRATEGIES=(bidkv preempt-evict preempt-evict-sjf largest-first static-random uniform slack-aware)

for strat in "${STRATEGIES[@]}"; do
    echo "[$(date)] Starting $strat (9 runs)..." | tee -a $LOG
    cd $DIR && $CMD --strategies "$strat" $COMMON 2>&1 | tee -a $LOG
    echo "[$(date)] $strat done" | tee -a $LOG

    # Kill any leftover GPU processes between strategies
    sleep 5
    kill -9 $(ps aux | grep 'vllm\.experiments\.vllm\.serve' | grep -v grep | awk '{print $2}') 2>/dev/null || true
    sleep 10
done

echo "[$(date)] ALL 7 long-context strategies complete! (63/63)" | tee -a $LOG
