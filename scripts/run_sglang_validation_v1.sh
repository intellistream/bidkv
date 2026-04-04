#!/bin/bash
# SGLang 验证性实验: bidkv + preempt-evict-sjf + h2o-style
# mixed workload, rate=3.8, 1 run per strategy
#
# 运行方法:
#   cd /home/bidkv
#   bash scripts/run_sglang_validation_v1.sh 2>&1 | tee results/sglang_validation_v1/run.log

set -e

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

export PYTHONPATH="${REPO_ROOT}/src:${PYTHONPATH:-}"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export BIDKV_MODEL="/home/models/Llama-3.1-8B-Instruct"
export no_proxy="127.0.0.1,localhost"
export NO_PROXY="127.0.0.1,localhost"
export CUDA_HOME=/usr/local/cuda-12.8
export PATH="/usr/local/cuda-12.8/bin:$PATH"

if command -v conda &>/dev/null && conda run -n sagellm python --version &>/dev/null 2>&1; then
    PYTHON="conda run -n sagellm python"
else
    PYTHON="python3"
fi

OUTPUT_DIR="results/sglang_validation_v1"
mkdir -p "$OUTPUT_DIR"

echo "=== SGLang Validation v1 ==="
echo "Strategies: bidkv, preempt-evict-sjf, h2o-style"
echo "Workload: mixed, Rate: 3.8, Runs: 1"
echo "max-total-tokens: 9600 (≈ vLLM 600 blocks × 16)"
echo "Python: $PYTHON"
echo "Start: $(date)"
echo "Output: $OUTPUT_DIR"
echo ""

for STRAT in bidkv preempt-evict-sjf h2o-style; do
  echo "--- Running strategy: $STRAT ($(date)) ---"
  $PYTHON -m bidkv.experiments.sglang.runner \
    --strategies "$STRAT" \
    --workloads "mixed" \
    --mixed-rates "3.8" \
    --runs 1 \
    --max-total-tokens 9600 \
    --output-dir "$OUTPUT_DIR" \
    --traces-dir "experiments/vllm/traces" \
    --resume \
    2>&1 | tee -a "$OUTPUT_DIR/runner_${STRAT}.log" \
    || echo "WARNING: $STRAT runner exited with error (see log), continuing..."
  echo "--- $STRAT done ($(date)) ---"
  echo ""
done

echo "=== All strategies done: $(date) ==="
echo ""
echo "Results:"
ls -1 "$OUTPUT_DIR"/*.json 2>/dev/null | head -20 || echo "(no JSON results yet)"
