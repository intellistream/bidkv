#!/usr/bin/env bash
# =============================================================================
# 前置实验：驱逐候选异质性数据采集
#
# 目标：在同一个 vLLM 运行中，于每次 KV 压力事件同时模拟 4 种策略的
#       受害者选择，记录候选分布 + 各策略选择 → 用于论文 Motivation 图
#
# 产出：
#   results/preliminary_motivation/victim_heterogeneity.jsonl
#       每行一个事件，包含 candidates 列表 + strategy_choices
#   results/preliminary_motivation/completions.jsonl
#       每行一个请求完成记录 {request_id, final_output_tokens}
#       用于 post-hoc join 计算真实 completion_ratio
#   paper/figures/fig_victim_heterogeneity.pdf / .png
#
# 配置选择：
#   - 策略: bidkv（走全 hook 路径，确保 logger 被调用）
#   - 工作负载: mixed, rate=5.7（高压力，KV 压力最充分，候选集最丰富）
#   - num-gpu-blocks-override=300（加大 KV 压力，捕获更多事件）
#   - 仅 1 run，约 300 个请求（~3~5min 实验时间）
#
# 用法：
#   bash scripts/run_motivation_capture.sh
#   # 仅绘图（已有数据时）
#   bash scripts/run_motivation_capture.sh --plot-only
# =============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_DIR"

CONDA_PY="python3.10"
export PYTHONPATH="/home/bidkv/src${PYTHONPATH:+:$PYTHONPATH}"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
# flashinfer/tvm_ffi .so ABI 兼容性修复（OSError: undefined symbol _ZNK3c106Device3strB5cxx11Ev）
export TVM_FFI_DISABLE_TORCH_C_DLPACK=1
MODEL="${BIDKV_MODEL:-/home/models/Llama-3.1-8B-Instruct}"
TRACES_DIR="experiments/vllm/traces"
OUTPUT_DIR="results/preliminary_motivation"
LOG_FILE="$OUTPUT_DIR/victim_heterogeneity.jsonl"
COMP_FILE="$OUTPUT_DIR/completions.jsonl"
PLOT_OUT_DIR="$OUTPUT_DIR"
PORT=8000

# 标准实验规格（与冒烟冻结环境完全一致）
GPU_UTIL="0.5"
GPU_BLOCKS="600"
MAX_SEQS="32"
MAX_MODEL_LEN="8192"
RUNS="3"
RATE="5.7"

# ── 参数解析 ────────────────────────────────────────────────────────
PLOT_ONLY=false
for arg in "$@"; do
    case "$arg" in
        --plot-only) PLOT_ONLY=true ;;
    esac
done

mkdir -p "$OUTPUT_DIR"

# ──────────────────────────────────────────────────────────────────
# Step 0: 说明
# ──────────────────────────────────────────────────────────────────
echo "=================================================================="
echo "Motivation Experiment: Victim Heterogeneity Data Capture"
echo "=================================================================="
echo "  Output dir  : $OUTPUT_DIR"
echo "  Log file    : $LOG_FILE"
echo "  Comp file   : $COMP_FILE"
echo "  Config      : mixed/rate=$RATE | $GPU_BLOCKS blocks | $RUNS runs | bidkv"
echo "=================================================================="
echo ""

if $PLOT_ONLY; then
    echo "[INFO] --plot-only: skipping data capture, going directly to plotting"
    if [[ ! -f "$LOG_FILE" ]]; then
        echo "[ERROR] Log file not found: $LOG_FILE"
        echo "        Run without --plot-only first to capture data."
        exit 1
    fi
    EVENT_COUNT=$(wc -l < "$LOG_FILE")
    echo "[INFO] Found $EVENT_COUNT events in $LOG_FILE"
else
    # ── Step 1: 直接使用 frozen trace ──────────────────────────────
    echo "----------------------------------------------------------------"
    echo "Step 1: 使用 frozen trace: $TRACES_DIR/mixed_rate${RATE}.json (1000 requests)"
    echo "----------------------------------------------------------------"
    echo "[OK] Trace: $TRACES_DIR/mixed_rate${RATE}.json"
    echo ""

    # ── Step 2: 清除旧日志（若存在）────────────────────────────────
    if [[ -f "$LOG_FILE" ]]; then
        echo "[INFO] Removing old log file: $LOG_FILE"
        rm -f "$LOG_FILE"
    fi
    if [[ -f "$COMP_FILE" ]]; then
        echo "[INFO] Removing old completions file: $COMP_FILE"
        rm -f "$COMP_FILE"
    fi
    # ── Step 3: 运行 vLLM + BidKV（BIDKV_LOG_PREEMPTION_EVENTS 激活 logger）──
    echo "----------------------------------------------------------------"
    echo "Step 2: Run vLLM with bidkv strategy + preemption event logger"
    echo "         BIDKV_LOG_PREEMPTION_EVENTS=$LOG_FILE"
    echo "         num-gpu-blocks-override=$GPU_BLOCKS (standard frozen env)"
    echo "         runs=$RUNS  requests=1000  rate=$RATE"
    echo "----------------------------------------------------------------"
    echo ""

    export BIDKV_LOG_PREEMPTION_EVENTS="$LOG_FILE"
    export BIDKV_LOG_COMPLETIONS="$COMP_FILE"
    export BIDKV_LOG_KV_THRESHOLD="0.80"

    $CONDA_PY -m bidkv.experiments.vllm.runner \
        --strategies "bidkv" \
        --workloads "mixed" \
        --runs $RUNS \
        --mixed-rates "$RATE" \
        --output-dir "$OUTPUT_DIR" \
        --traces-dir "$TRACES_DIR" \
        --model "$MODEL" \
        --gpu-memory-utilization $GPU_UTIL \
        --max-model-len $MAX_MODEL_LEN \
        --num-gpu-blocks-override $GPU_BLOCKS \
        --max-num-seqs $MAX_SEQS \
        --block-size 16 \
        --port $PORT \
        2>&1 | tee "$OUTPUT_DIR/capture_run.log"

    unset BIDKV_LOG_PREEMPTION_EVENTS
    unset BIDKV_LOG_COMPLETIONS
    unset BIDKV_LOG_KV_THRESHOLD

    echo ""
    echo "----------------------------------------------------------------"
    if [[ -f "$LOG_FILE" ]]; then
        EVENT_COUNT=$(wc -l < "$LOG_FILE")
        echo "[OK] Preemption event log captured: $EVENT_COUNT events"
        echo "     File: $LOG_FILE"
    else
        echo "[WARN] Log file not created — no preemption events triggered"
        echo "       Check that KV pressure was sufficient."
        echo "       Try reducing num-gpu-blocks-override further."
        exit 1
    fi
    if [[ -f "$COMP_FILE" ]]; then
        COMP_COUNT=$(wc -l < "$COMP_FILE")
        echo "[OK] Completions log captured: $COMP_COUNT records"
        echo "     File: $COMP_FILE"
    else
        echo "[WARN] Completions file not created (completion hook may not have fired)"
    fi
    echo "----------------------------------------------------------------"
    echo ""
fi

# ──────────────────────────────────────────────────────────────────
# Step 3 (or 2 in plot-only mode): 绘图
# ──────────────────────────────────────────────────────────────────
echo "----------------------------------------------------------------"
echo "Step 3: Plotting victim heterogeneity figure"
echo "        Input : $LOG_FILE"
echo "        Output: $PLOT_OUT_DIR/"
echo "----------------------------------------------------------------"
echo ""

$CONDA_PY scripts/plot_victim_heterogeneity.py \
    --log-file "$LOG_FILE" \
    --completions-file "$COMP_FILE" \
    --output-dir "$PLOT_OUT_DIR" \
    --format "pdf,png"

echo ""
echo "=================================================================="
echo "DONE"
echo "  Figure files:"
ls -lh "$PLOT_OUT_DIR"/fig_victim_*.{pdf,png} 2>/dev/null || \
    echo "  (check $PLOT_OUT_DIR for output files)"
echo "=================================================================="
