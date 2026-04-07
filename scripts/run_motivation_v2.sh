#!/usr/bin/env bash
# =============================================================================
# 前置实验 v2：Victim Selection Heterogeneity 数据采集
#
# 修复了 v1 中 preemption_logger.py 因 h2o_style → largest_first 重命名
# 导致的 ModuleNotFoundError。
#
# 产出：
#   results/preliminary_motivation/victim_heterogeneity.jsonl
#   results/preliminary_motivation/completions.jsonl
#   paper/figures/fig_preliminary_motivation.pdf (3-panel figure)
#
# 配置（frozen env）：
#   策略: bidkv（触发完整 hook + logger）
#   工作负载: mixed, rate=3.8（主分化 regime）
#   600 blocks, gpu-mem-util=0.5, max-num-seqs=32
#   1 run, 1000 requests
#
# 用法：
#   bash scripts/run_motivation_v2.sh            # 完整采集 + 绘图
#   bash scripts/run_motivation_v2.sh --plot-only # 仅绘图（已有数据）
# =============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_DIR"

CONDA_PY="python3.10"
export PYTHONPATH="/home/bidkv/src${PYTHONPATH:+:$PYTHONPATH}"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TVM_FFI_DISABLE_TORCH_C_DLPACK=1

MODEL="${BIDKV_MODEL:-/home/models/Llama-3.1-8B-Instruct}"
TRACES_DIR="experiments/vllm/traces"
OUTPUT_DIR="results/preliminary_motivation"
LOG_FILE="$OUTPUT_DIR/victim_heterogeneity.jsonl"
COMP_FILE="$OUTPUT_DIR/completions.jsonl"
PORT=8000

# Frozen env parameters
GPU_UTIL="0.5"
GPU_BLOCKS="600"
MAX_SEQS="32"
MAX_MODEL_LEN="8192"
RATE="3.8"

# ── 参数解析 ──
PLOT_ONLY=false
for arg in "$@"; do
    case "$arg" in
        --plot-only) PLOT_ONLY=true ;;
    esac
done

mkdir -p "$OUTPUT_DIR"

echo "=================================================================="
echo "Motivation Experiment v2: Victim Heterogeneity Data Capture"
echo "=================================================================="
echo "  Output dir : $OUTPUT_DIR"
echo "  Log file   : $LOG_FILE"
echo "  Rate       : $RATE req/s (medium pressure, main differentiation regime)"
echo "  Config     : $GPU_BLOCKS blocks | $MAX_SEQS max-seqs | bidkv strategy"
echo "=================================================================="

if $PLOT_ONLY; then
    echo "[INFO] --plot-only mode"
    if [[ ! -f "$LOG_FILE" ]]; then
        echo "[ERROR] Log file not found: $LOG_FILE"
        echo "        Run without --plot-only first."
        exit 1
    fi
    EVENT_COUNT=$(wc -l < "$LOG_FILE")
    echo "[INFO] Found $EVENT_COUNT events"
else
    # ── Step 1: Verify logger code is fixed ──
    echo ""
    echo "[CHECK] Verifying preemption_logger imports..."
    $CONDA_PY -c "
from bidkv.adapters.vllm.preemption_logger import _get_strategies
s = _get_strategies()
print(f'  OK: {len(s)} strategies loaded: {list(s.keys())}')
" || {
        echo "[ERROR] preemption_logger import check failed."
        echo "        The h2o_style → largest_first rename may not be complete."
        exit 1
    }

    # ── Step 2: Clean old data ──
    for f in "$LOG_FILE" "$COMP_FILE"; do
        if [[ -f "$f" ]]; then
            echo "[INFO] Removing old: $f"
            rm -f "$f"
        fi
    done

    # ── Step 3: Run experiment with preemption logging ──
    echo ""
    echo "----------------------------------------------------------------"
    echo "Step 3: vLLM + BidKV with preemption event logger"
    echo "  BIDKV_LOG_PREEMPTION_EVENTS=$LOG_FILE"
    echo "  BIDKV_LOG_KV_THRESHOLD=0.80"
    echo "  Rate: $RATE | Blocks: $GPU_BLOCKS | Runs: 1"
    echo "----------------------------------------------------------------"

    export BIDKV_LOG_PREEMPTION_EVENTS="$LOG_FILE"
    export BIDKV_LOG_COMPLETIONS="$COMP_FILE"
    export BIDKV_LOG_KV_THRESHOLD="0.80"

    $CONDA_PY -m bidkv.experiments.vllm.runner \
        --strategies "bidkv" \
        --workloads "mixed" \
        --runs 1 \
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
        2>&1 | tee "$OUTPUT_DIR/capture_v2.log"

    unset BIDKV_LOG_PREEMPTION_EVENTS
    unset BIDKV_LOG_COMPLETIONS
    unset BIDKV_LOG_KV_THRESHOLD

    echo ""
    if [[ -f "$LOG_FILE" ]]; then
        EVENT_COUNT=$(wc -l < "$LOG_FILE")
        echo "[OK] Captured $EVENT_COUNT preemption events → $LOG_FILE"
    else
        echo "[WARN] No log file created."
        echo "       Server log: $OUTPUT_DIR/server_bidkv.log"
        echo "       Check for errors: grep -i 'error\|exception' $OUTPUT_DIR/server_bidkv.log | tail -5"
        exit 1
    fi
fi

# ── Step 4: Generate 3-panel figure ──
echo ""
echo "----------------------------------------------------------------"
echo "Step 4: Generating 3-panel motivation figure"
echo "----------------------------------------------------------------"

$CONDA_PY scripts/plot_motivation_3panel.py \
    --log-file "$LOG_FILE" \
    --output "paper/figures/fig_preliminary_motivation.pdf"

echo ""
echo "=================================================================="
echo "DONE — Outputs:"
echo "  Data:   $LOG_FILE"
echo "  Figure: paper/figures/fig_preliminary_motivation.pdf"
echo "  PNG:    paper/figures/fig_preliminary_motivation.png"
echo "=================================================================="
