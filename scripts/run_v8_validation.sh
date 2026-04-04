#!/bin/bash
# v8 validation experiment: bidkv + pe-sjf + h2o, mixed workload, 3 rates × 3 runs
# Run from bidkv repo root: nohup bash scripts/run_v8_validation.sh > results/v8_validation.log 2>&1 &

export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

cd /home/cyb/bidkv

echo "=== v8 Validation Experiment ==="
echo "Start: $(date)"
echo ""

# Run each strategy separately so one crash doesn't block others
for STRAT in bidkv preempt-evict-sjf largest-first; do
  echo "--- Running strategy: $STRAT ($(date)) ---"
  conda run -n sagellm python -m bidkv.experiments.vllm.runner \
    --strategies "$STRAT" \
    --workloads "mixed" \
    --mixed-rates "2.0,3.8,5.7" \
    --runs 3 \
    --num-gpu-blocks-override 600 \
    --output-dir "results/vllm_v8_analysis" \
    --ttft-target-ms 1000 \
    --resume || echo "WARNING: $STRAT runner exited with error, continuing..."
  echo "--- $STRAT done ($(date)) ---"
  echo ""
done

echo "=== All done: $(date) ==="
