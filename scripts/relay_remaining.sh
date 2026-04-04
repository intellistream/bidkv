#!/bin/bash
set -e
DIR=/home/cyb/bidkv
OUT=$DIR/results/vllm_v8_full_validation
LOG=$DIR/results/relay_progress.log
ENV="HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1"
CMD="conda run --no-capture-output -n sagellm python -m bidkv.experiments.vllm.runner"
COMMON="--workloads mixed --runs 3 --num-gpu-blocks-override 600 --gpu-memory-utilization 0.5 --output-dir $OUT"

echo "[$(date)] Relay script started, waiting for static-random to finish..." >> $LOG

# Wait for static-random (9 JSONs)
while true; do
    COUNT=$(ls $OUT/static-random__*.json 2>/dev/null | wc -l)
    if [ "$COUNT" -ge 9 ]; then
        echo "[$(date)] static-random done ($COUNT files)" >> $LOG
        break
    fi
    sleep 60
done

# Kill leftover vllm processes
sleep 10
kill -9 $(ps aux | grep 'vllm\.experiments\.vllm\.serve' | grep -v grep | awk '{print $2}') 2>/dev/null || true
sleep 5

# Run uniform
echo "[$(date)] Starting uniform..." >> $LOG
cd $DIR && HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 $CMD --strategies uniform $COMMON >> $LOG 2>&1
echo "[$(date)] uniform done" >> $LOG

sleep 5
kill -9 $(ps aux | grep 'vllm\.experiments\.vllm\.serve' | grep -v grep | awk '{print $2}') 2>/dev/null || true
sleep 5

# Run slack-aware
echo "[$(date)] Starting slack-aware..." >> $LOG
cd $DIR && HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 $CMD --strategies slack-aware $COMMON >> $LOG 2>&1
echo "[$(date)] slack-aware done" >> $LOG

echo "[$(date)] ALL 7 strategies complete!" >> $LOG
