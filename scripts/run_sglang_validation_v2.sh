#!/bin/bash
# SGLang 验证性实验 v2: bidkv + preempt-evict-sjf + h2o-style
# mixed workload, rate=5.7 (高负载), 3 runs per strategy
#
# 运行方法:
#   tmux new -s sglang_v2
#   bash scripts/run_sglang_validation_v2.sh 2>&1 | tee results/sglang_validation_v2/run.log

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

OUTPUT_DIR="results/sglang_validation_v2"
mkdir -p "$OUTPUT_DIR"

echo "=== SGLang Validation v2 (高负载) ==="
echo "Strategies: bidkv, preempt-evict-sjf, h2o-style"
echo "Workload: mixed, Rate: 5.7, Runs: 3"
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
    --mixed-rates "5.7" \
    --runs 3 \
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
echo ""
echo "Summary:"
python3 -c "
import json, pathlib, sys
d = pathlib.Path('$OUTPUT_DIR')
rows = []
for f in sorted(d.glob('*.json')):
    r = json.loads(f.read_text())
    s = r['summary']
    rows.append((r['strategy'], r['run_index'], s['ttft_ms_p50'], s['ttft_ms_p95'], s['ttft_ms_p99'], s['throughput_rps'], s['successful_requests'], s['total_requests']))
print(f\"{'strategy':22} {'run':3} {'p50':>8} {'p95':>8} {'p99':>8} {'tput':>7} {'ok/tot':>9}\")
print('-'*75)
for row in rows:
    print(f\"{row[0]:22} {row[1]:3d} {row[2]:7.0f}ms {row[3]:7.0f}ms {row[4]:7.0f}ms {row[5]:6.2f}rps {row[6]:4d}/{row[7]:4d}\")
" 2>/dev/null || true
