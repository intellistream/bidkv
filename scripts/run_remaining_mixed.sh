#!/bin/bash
# Run remaining 2 strategies: uniform → slack-aware
# Sequential execution to avoid GPU/port conflicts
set -euo pipefail

DIR=/home/cyb/bidkv
OUT=$DIR/results/vllm_v8_full_validation
LOG=$DIR/results/remaining_mixed_progress.log
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
CMD="conda run --no-capture-output -n sagellm python -m bidkv.experiments.vllm.runner"
COMMON="--workloads mixed --runs 3 --num-gpu-blocks-override 600 --gpu-memory-utilization 0.5 --output-dir $OUT"

echo "[$(date)] Starting remaining mixed strategies..." | tee -a $LOG

# --- uniform ---
echo "[$(date)] Starting uniform (9 runs)..." | tee -a $LOG
cd $DIR && $CMD --strategies uniform $COMMON 2>&1 | tee -a $LOG
echo "[$(date)] uniform done" | tee -a $LOG

# Kill any leftover GPU processes
sleep 5
kill -9 $(ps aux | grep 'vllm\.experiments\.vllm\.serve' | grep -v grep | awk '{print $2}') 2>/dev/null || true
sleep 10

# --- slack-aware ---
echo "[$(date)] Starting slack-aware (9 runs)..." | tee -a $LOG
cd $DIR && $CMD --strategies slack-aware $COMMON 2>&1 | tee -a $LOG
echo "[$(date)] slack-aware done" | tee -a $LOG

echo "[$(date)] ALL 7 mixed strategies complete! (63/63)" | tee -a $LOG
