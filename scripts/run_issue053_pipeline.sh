#!/bin/bash
# Issue #053 Full Experiment Pipeline
# 
# Prerequisites:
#   - P1 (4 core strategies × 2 workloads) must have completed first
#   - GPU must be free
#
# This script runs:
#   1. P2: 4 secondary strategies × mixed workload (36 runs)
#   2. Merge all vLLM results into results/vllm_full/
#   3. SGLang full experiment (72 runs)
#   4. vLLM + SGLang analysis
#
# Usage:
#   conda run -n sagellm bash scripts/run_issue053_pipeline.sh
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
CONDA_RUN=(conda run -n sagellm python)
MODEL="${BIDKV_MODEL:-meta-llama/Llama-3.1-8B-Instruct}"
TRACES="results/formal/traces"

cd "$REPO_DIR"

echo "=========================================="
echo "Issue #053 Full Experiment Pipeline"
echo "=========================================="
date

# ──────────────────────────────────────────────
# Step 1: P2 — Secondary strategies × mixed
# ──────────────────────────────────────────────
echo ""
echo "[Step 1/4] P2: 4 secondary strategies × mixed (36 runs)"
echo "Strategies: static-random, uniform, global-nobid, slack-aware"
echo ""

"${CONDA_RUN[@]}" -m bidkv.experiments.vllm.runner \
  --strategies "static-random,uniform,global-nobid,slack-aware" \
  --workloads "mixed" \
  --runs 3 \
  --traces-dir "$TRACES" \
  --output-dir results/vllm_full_p2 \
  --model "$MODEL" \
  --gpu-memory-utilization 0.85 \
  --max-model-len 8192 \
  --port 8000 \
  2>&1 | tee results/vllm_full_p2_log.txt

echo "[Step 1/4] P2 completed."
echo "P2 results: $(find results/vllm_full_p2 -name '*.json' | wc -l) files"

# ──────────────────────────────────────────────
# Step 2: Merge all vLLM results
# ──────────────────────────────────────────────
echo ""
echo "[Step 2/4] Merging vLLM results into results/vllm_full/"
echo ""

mkdir -p results/vllm_full

# Copy formal long_context results (8 strategies × 3 rates × 3 runs = 72)
cp results/formal/long_context/*.json results/vllm_full/ 2>/dev/null || true
# Remove candidate_consistency_report from merged dir
rm -f results/vllm_full/candidate_consistency_report.json

# Copy P1 mixed results only (skip long_context duplicates)
for f in results/vllm_full_p1/*__mixed__*.json; do
  [ -f "$f" ] && cp "$f" results/vllm_full/
done

# Copy P2 mixed results
for f in results/vllm_full_p2/*__mixed__*.json; do
  [ -f "$f" ] && cp "$f" results/vllm_full/
done

echo "Merged results: $(find results/vllm_full -name '*.json' | wc -l) files"
echo "Expected: 144 (72 long_context + 72 mixed)"

# ──────────────────────────────────────────────
# Step 3: SGLang full experiment (72 runs)
# ──────────────────────────────────────────────
echo ""
echo "[Step 3/4] SGLang full experiment (3 strategies × 2 workloads × 3 rates × 3 runs = 54)"
echo ""

"${CONDA_RUN[@]}" -m bidkv.experiments.sglang.runner \
  --strategies "sglang_default,slack_aware,bidkv" \
  --workloads "mixed,long_context" \
  --runs 3 \
  --traces-dir "$TRACES" \
  --output-dir results/sglang_full \
  --model "$MODEL" \
  --port 30000 \
  2>&1 | tee results/sglang_full_log.txt

echo "[Step 3/4] SGLang completed."
echo "SGLang results: $(find results/sglang_full -name '*.json' | wc -l) files"

# ──────────────────────────────────────────────
# Step 4: Analysis
# ──────────────────────────────────────────────
echo ""
echo "[Step 4/4] Running analysis..."
echo ""

# vLLM analysis
echo "Running vLLM analysis..."
"${CONDA_RUN[@]}" -m bidkv.experiments.vllm.analysis \
  --results-dir results/vllm_full \
  --output-dir results/vllm_full/analysis

# SGLang analysis (with cross-framework comparison)
echo "Running SGLang analysis..."
"${CONDA_RUN[@]}" -m bidkv.experiments.sglang.analysis \
  --sglang-results-dir results/sglang_full \
  --vllm-results-dir results/vllm_full \
  --output-dir results/sglang_full/analysis

echo ""
echo "=========================================="
echo "Issue #053 Pipeline Complete!"
echo "=========================================="
echo ""
echo "Outputs:"
echo "  vLLM results:    results/vllm_full/ ($(find results/vllm_full -name '*.json' -not -path '*/analysis/*' | wc -l) runs)"
echo "  vLLM analysis:   results/vllm_full/analysis/"
echo "  SGLang results:  results/sglang_full/ ($(find results/sglang_full -name '*.json' -not -path '*/analysis/*' | wc -l) runs)"
echo "  SGLang analysis: results/sglang_full/analysis/"
echo ""
date
