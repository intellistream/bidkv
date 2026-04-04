#!/usr/bin/env bash
# Run BidKV-only long_context experiment with v9 fix (avg>500 gate removed).
# Prerequisite: uniform + slack-aware results already in results/vllm_v8_long_context/
#
# This script:
# 1. Removes any old BidKV long_context results (from pre-fix code)
# 2. Runs BidKV × 3 rates × 3 runs = 9 runs
# 3. Results go to same directory as other strategies

set -euo pipefail

RESULTS_DIR="results/vllm_v8_long_context"

echo "=== BidKV Long-Context Experiment (v9 fix: avg>500 gate removed) ==="
echo "Date: $(date)"
echo ""

# Remove old BidKV results (from pre-fix code)
OLD_BIDKV=$(ls ${RESULTS_DIR}/bidkv__long_context__*.json 2>/dev/null || true)
if [ -n "$OLD_BIDKV" ]; then
    echo "Removing old BidKV results (pre-fix):"
    echo "$OLD_BIDKV"
    rm -f ${RESULTS_DIR}/bidkv__long_context__*.json
    echo ""
fi

echo "Starting BidKV runs: 1 strategy × 3 rates × 3 runs = 9 runs"
echo "Output: ${RESULTS_DIR}/"
echo ""

conda run -n sagellm python -m bidkv.experiments.vllm.runner \
    --strategies bidkv \
    --workloads long_context \
    --long-rates "0.35,0.5,0.7" \
    --runs 3 \
    --output-dir "${RESULTS_DIR}" \
    --gpu-memory-utilization 0.5 \
    --num-gpu-blocks-override 600 \
    --max-num-seqs 32 \
    --block-size 16 \
    --max-model-len 8192

echo ""
echo "=== BidKV Long-Context Complete ==="
echo "Results: $(ls ${RESULTS_DIR}/bidkv__long_context__*.json 2>/dev/null | wc -l) files"
