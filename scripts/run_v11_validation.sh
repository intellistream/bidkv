#!/bin/bash
# v11 validation experiment: bidkv (v11 selective protection) vs pe-sjf + h2o
# Output to a separate directory to preserve v8 results

export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

cd /home/cyb/bidkv

echo "=== v11 Validation Experiment ==="
echo "Start: $(date)"
echo ""

# Only run bidkv — pe-sjf and h2o haven't changed, reuse v8 data
for STRAT in bidkv; do
  echo "--- Running strategy: $STRAT ($(date)) ---"
  conda run -n sagellm python -m bidkv.experiments.vllm.runner \
    --strategies "$STRAT" \
    --workloads "mixed" \
    --mixed-rates "2.0,3.8,5.7" \
    --runs 3 \
    --num-gpu-blocks-override 600 \
    --output-dir "results/vllm_v11_validation" \
    --ttft-target-ms 1000 \
    --resume || echo "WARNING: $STRAT runner exited with error, continuing..."
  echo "--- $STRAT done ($(date)) ---"
  echo ""
done

echo "=== All done: $(date) ==="
