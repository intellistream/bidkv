#!/bin/bash
# Full long_context experiment: 7 strategies × 3 rates × 3 runs = 63 runs
# Frozen params (v8-frozen): KV=600, GPU=0.5, max_seqs=32, block_size=16, max_model_len=8192
# Frozen rates: 0.35, 0.5, 0.7 req/s
# Requests per run: 500

set -e

cd /home/cyb/bidkv

echo "=== BidKV Long-Context Full Experiment ==="
echo "Start time: $(date)"
echo "Matrix: 7 strategies × 3 rates × 3 runs = 63 runs"
echo ""

export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

conda run -n sagellm python -m bidkv.experiments.vllm.runner \
  --strategies "preempt-evict,preempt-evict-sjf,static-random,largest-first,uniform,slack-aware,bidkv" \
  --workloads "long_context" \
  --long-rates "0.35,0.5,0.7" \
  --runs 3 \
  --output-dir results/vllm_v8_long_context \
  --gpu-memory-utilization 0.5 \
  --num-gpu-blocks-override 600 \
  --max-num-seqs 32 \
  --block-size 16 \
  --max-model-len 8192 \
  2>&1

echo ""
echo "=== Experiment Complete ==="
echo "End time: $(date)"
echo "Results: results/vllm_v8_long_context/"
ls results/vllm_v8_long_context/*.json 2>/dev/null | wc -l
