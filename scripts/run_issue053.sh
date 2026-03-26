#!/usr/bin/env bash
# =============================================================================
# Issue #053: 全量实验运行脚本 (vLLM 144 + SGLang 72 runs)
#
# 用法:
#   bash scripts/run_issue053.sh           # 从头开始运行全量
#   bash scripts/run_issue053.sh --resume  # 断点续跑（跳过已完成的 runs）
#
# 本脚本执行:
#   Step 1: 环境验证
#   Step 2: vLLM P1 — 4 核心策略 (72 runs)
#   Step 3: vLLM P2 — 4 次级策略 (72 runs)
#   Step 4: 合并 vLLM 结果 → results/vllm_full/
#   Step 5: SGLang — 4 策略 (72 runs)
#   Step 6: 数据分析 — Table 1/2 + Figure 3-7
#   Step 7: 最终验收
# =============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_DIR"

# 参数解析
RESUME_FLAG=""
if [[ "${1:-}" == "--resume" ]]; then
    RESUME_FLAG="--resume"
    echo "[INFO] Resume mode: will skip runs with existing result files."
fi

CONDA_RUN="conda run -n sagellm"

# 冻结参数
MODEL="/home/cyb/Llama-3.1-8B-Instruct"
TRACES_DIR="results/formal/traces"
VLLM_P1_DIR="results/vllm_full_p1"
VLLM_P2_DIR="results/vllm_full_p2"
VLLM_FULL_DIR="results/vllm_full"
SGLANG_DIR="results/sglang_full"
VLLM_PORT=8000
SGLANG_PORT=30000

# ─── Step 1: 环境验证 ─────────────────────────────────────────────────
echo "============================================================"
echo "Step 1: 环境验证"
echo "============================================================"

echo "[CHECK] Frozen traces..."
if [[ ! -f "$TRACES_DIR/manifest.json" ]]; then
    echo "[ERROR] Frozen traces not found at $TRACES_DIR"
    exit 1
fi
echo "[OK] Traces directory exists with manifest"

echo "[CHECK] Tests..."
$CONDA_RUN python -m pytest tests/ -q --tb=line 2>&1 | tail -3
echo "[OK] Tests passed"

echo "[CHECK] Frozen rates consistency..."
$CONDA_RUN python -c "
from bidkv.experiments.vllm.config import WORKLOAD_REQUEST_RATES
from bidkv.experiments.sglang.config import WORKLOAD_REQUEST_RATES as SGLANG_RATES
assert WORKLOAD_REQUEST_RATES == SGLANG_RATES, 'Rate mismatch!'
print('[OK] vLLM and SGLang rates match:', WORKLOAD_REQUEST_RATES)
"

# ─── Step 2: vLLM P1 — 4 核心策略 ─────────────────────────────────────
echo ""
echo "============================================================"
echo "Step 2: vLLM P1 — 3 核心策略 (preempt-evict, h2o-style, bidkv)"
echo "============================================================"

$CONDA_RUN python -m bidkv.experiments.vllm.runner \
    --strategies "preempt-evict,h2o-style,bidkv" \
    --workloads "mixed,long_context" \
    --runs 3 \
    --traces-dir "$TRACES_DIR" \
    --output-dir "$VLLM_P1_DIR" \
    --model "$MODEL" \
    --gpu-memory-utilization 0.85 \
    --max-model-len 8192 \
    --port "$VLLM_PORT" \
    $RESUME_FLAG \
    2>&1 | tee results/vllm_p1_log_053.txt

P1_COUNT=$(ls "$VLLM_P1_DIR"/*.json 2>/dev/null | grep -v consistency | wc -l)
echo "[P1 DONE] $P1_COUNT result files"

# ─── Step 3: vLLM P2 — 4 次级策略 ─────────────────────────────────────
echo ""
echo "============================================================"
echo "Step 3: vLLM P2 — 4 次级策略 (static-random, uniform, global-nobid, slack-aware)"
echo "============================================================"

$CONDA_RUN python -m bidkv.experiments.vllm.runner \
    --strategies "static-random,uniform,global-nobid,slack-aware" \
    --workloads "mixed,long_context" \
    --runs 3 \
    --traces-dir "$TRACES_DIR" \
    --output-dir "$VLLM_P2_DIR" \
    --model "$MODEL" \
    --gpu-memory-utilization 0.85 \
    --max-model-len 8192 \
    --port "$VLLM_PORT" \
    $RESUME_FLAG \
    2>&1 | tee results/vllm_p2_log_053.txt

P2_COUNT=$(ls "$VLLM_P2_DIR"/*.json 2>/dev/null | grep -v consistency | wc -l)
echo "[P2 DONE] $P2_COUNT result files"

# ─── Step 4: 合并 vLLM 结果 ───────────────────────────────────────────
echo ""
echo "============================================================"
echo "Step 4: 合并 vLLM P1 + P2 → $VLLM_FULL_DIR"
echo "============================================================"

mkdir -p "$VLLM_FULL_DIR"
cp "$VLLM_P1_DIR"/*.json "$VLLM_FULL_DIR/"
cp "$VLLM_P2_DIR"/*.json "$VLLM_FULL_DIR/"
FULL_COUNT=$(ls "$VLLM_FULL_DIR"/*.json 2>/dev/null | grep -v consistency | wc -l)
echo "[MERGE] $FULL_COUNT total result files (expect ≥144)"

# ─── Step 5: SGLang — 4 策略 ──────────────────────────────────────────
echo ""
echo "============================================================"
echo "Step 5: SGLang — 3 策略 (sglang_default, slack_aware, bidkv)"
echo "============================================================"

$CONDA_RUN python -m bidkv.experiments.sglang.runner \
    --strategies "sglang_default,slack_aware,bidkv" \
    --workloads "mixed,long_context" \
    --runs 3 \
    --traces-dir "$TRACES_DIR" \
    --output-dir "$SGLANG_DIR" \
    --model "$MODEL" \
    --port "$SGLANG_PORT" \
    $RESUME_FLAG \
    2>&1 | tee results/sglang_log_053.txt

SG_COUNT=$(ls "$SGLANG_DIR"/*.json 2>/dev/null | wc -l)
echo "[SGLANG DONE] $SG_COUNT result files (expect ≥72)"

# ─── Step 6: 数据分析 ─────────────────────────────────────────────────
echo ""
echo "============================================================"
echo "Step 6: 数据分析 — Table 1/2 + Figure 3-7"
echo "============================================================"

echo "[ANALYSIS] vLLM → Table 1 + Figure 3-6..."
$CONDA_RUN python -m bidkv.experiments.vllm.analysis \
    --results-dir "$VLLM_FULL_DIR" \
    --output-dir "$VLLM_FULL_DIR/analysis" \
    2>&1 | tee results/vllm_analysis_log.txt

echo ""
echo "[ANALYSIS] SGLang → Table 2 + Figure 7 + DC..."
$CONDA_RUN python -m bidkv.experiments.sglang.analysis \
    --results-dir "$SGLANG_DIR" \
    --output-dir "$SGLANG_DIR/analysis" \
    2>&1 | tee results/sglang_analysis_log.txt

# ─── Step 7: 最终验收 ─────────────────────────────────────────────────
echo ""
echo "============================================================"
echo "Step 7: 最终验收"
echo "============================================================"

echo "[CHECK] vLLM result count..."
VLLM_FINAL=$(ls "$VLLM_FULL_DIR"/*.json 2>/dev/null | grep -v consistency | grep -v analysis | wc -l)
echo "  vLLM: $VLLM_FINAL files (expect ≥144)"

echo "[CHECK] SGLang result count..."
SG_FINAL=$(ls "$SGLANG_DIR"/*.json 2>/dev/null | wc -l)
echo "  SGLang: $SG_FINAL files (expect ≥72)"

echo "[CHECK] Crash/abort scan..."
echo "  vLLM aborts: $(grep -l 'aborted\|OVERLOAD_FAILURE' "$VLLM_FULL_DIR"/*.json 2>/dev/null | wc -l)"
echo "  SGLang aborts: $(grep -l 'aborted\|OVERLOAD_FAILURE' "$SGLANG_DIR"/*.json 2>/dev/null | wc -l)"

echo "[CHECK] Directional consistency..."
if [[ -f "$SGLANG_DIR/analysis/directional_consistency.json" ]]; then
    $CONDA_RUN python -c "
import json
dc = json.load(open('$SGLANG_DIR/analysis/directional_consistency.json'))
print('  DC result:', json.dumps(dc, indent=2)[:500])
"
else
    echo "  [WARN] DC report not found yet"
fi

echo ""
echo "============================================================"
echo "Issue #053 全量实验运行完成！"
echo "============================================================"
echo "产出物:"
echo "  vLLM:  $VLLM_FULL_DIR/analysis/ (Table 1 + Figure 3-6)"
echo "  SGLang: $SGLANG_DIR/analysis/ (Table 2 + Figure 7)"
echo ""
echo "下一步: 更新 CHANGELOG.md 并提交"
