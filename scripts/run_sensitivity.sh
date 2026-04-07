#!/bin/bash
# BidKV Sensitivity Analysis — KV Gate Threshold + Weight Robustness
#
# Pure sensitivity analysis: varies parameters around their defaults.
# No ablation (component removal) experiments.
#
# Axes:
#   1. KV gate threshold: 0.85, 0.90, [0.95], 0.98
#   2. Completion weight w_c: 0.5, 1.0, [2.0], 4.0
#   3. Starvation weight w_s: 0.1, 0.25, [0.5], 1.0
#
# 9 variants × 3 runs = 27 runs at rate=3.8 (sustained KV pressure regime).
#
# Usage:
#   cd /home/bidkv
#   nohup bash scripts/run_sensitivity.sh > results/vllm_sensitivity/run.log 2>&1 &
#
# Output: results/vllm_sensitivity/

set -euo pipefail

BIDKV_DIR="/home/bidkv"
OUTPUT_BASE="$BIDKV_DIR/results/vllm_sensitivity"
LOG_DIR="$OUTPUT_BASE/logs"
mkdir -p "$OUTPUT_BASE" "$LOG_DIR"

# Frozen experiment params (v8-frozen, must match copilot-instructions.md)
MODEL="${BIDKV_MODEL:-/home/models/Llama-3.1-8B-Instruct}"
RATE="3.8"
WORKLOAD="mixed"
RUNS=3
GPU_MEM="0.5"
GPU_BLOCKS="600"
MAX_SEQS="32"
BLOCK_SIZE="16"
MAX_MODEL_LEN="8192"

# Common runner args (frozen server params from v8)
COMMON_ARGS="--strategies bidkv \
  --workloads $WORKLOAD \
  --mixed-rates $RATE \
  --runs $RUNS \
  --gpu-memory-utilization $GPU_MEM \
  --num-gpu-blocks-override $GPU_BLOCKS \
  --max-num-seqs $MAX_SEQS \
  --block-size $BLOCK_SIZE \
  --max-model-len $MAX_MODEL_LEN \
  --model $MODEL \
  --resume"

# Sensitivity variants: NAME -> ENV_VARS
# Only vary one parameter at a time; defaults: gate=0.95, w_c=2.0, w_s=0.5
declare -A VARIANTS

VARIANTS["default"]=""                                    # baseline (all defaults)

# === Axis 1: KV Gate Threshold ===
VARIANTS["gate-85"]="BIDKV_KV_GATE=0.85"                 # early trigger
VARIANTS["gate-90"]="BIDKV_KV_GATE=0.90"                 # moderate
# gate-95 = default (0.95)
VARIANTS["gate-98"]="BIDKV_KV_GATE=0.98"                 # late trigger

# === Axis 2: Completion Weight (w_c) ===
VARIANTS["wc-05"]="BIDKV_COMPLETION_WEIGHT=0.5"           # weak completion protection
VARIANTS["wc-10"]="BIDKV_COMPLETION_WEIGHT=1.0"           # moderate
# wc-20 = default (2.0)
VARIANTS["wc-40"]="BIDKV_COMPLETION_WEIGHT=4.0"           # strong completion protection

# === Axis 3: Starvation Weight (w_s) ===
VARIANTS["ws-01"]="BIDKV_STARVATION_WEIGHT=0.1"           # weak anti-starvation
VARIANTS["ws-025"]="BIDKV_STARVATION_WEIGHT=0.25"         # moderate
# ws-05 = default (0.5)
VARIANTS["ws-10"]="BIDKV_STARVATION_WEIGHT=1.0"           # strong anti-starvation

# Run order: default first, then gate, then weights
RUN_ORDER=(
  "default"
  "gate-85" "gate-90" "gate-98"
  "wc-05" "wc-10" "wc-40"
  "ws-01" "ws-025" "ws-10"
)

echo "========================================"
echo "BidKV Sensitivity Analysis"
echo "Rate=$RATE, Workload=$WORKLOAD, Runs=$RUNS"
echo "Total variants: ${#RUN_ORDER[@]}"
echo "Output: $OUTPUT_BASE"
echo "========================================"

COMPLETED=0
FAILED=0

for VARIANT in "${RUN_ORDER[@]}"; do
  VARIANT_DIR="$OUTPUT_BASE/$VARIANT"
  VARIANT_LOG="$LOG_DIR/${VARIANT}.log"
  ENV_VARS="${VARIANTS[$VARIANT]}"

  # Check if already completed (3 result files exist)
  EXPECTED_FILES=0
  for r in $(seq 0 $((RUNS - 1))); do
    if [[ -f "$VARIANT_DIR/bidkv__${WORKLOAD}__rate${RATE}__r${r}.json" ]]; then
      EXPECTED_FILES=$((EXPECTED_FILES + 1))
    fi
  done
  if [[ $EXPECTED_FILES -ge $RUNS ]]; then
    echo "[SKIP] $VARIANT — already completed ($EXPECTED_FILES/$RUNS files)"
    COMPLETED=$((COMPLETED + 1))
    continue
  fi

  echo ""
  echo "----------------------------------------"
  echo "[RUN] Variant: $VARIANT"
  echo "  ENV: $ENV_VARS"
  echo "  Output: $VARIANT_DIR"
  echo "  Log: $VARIANT_LOG"
  echo "----------------------------------------"

  mkdir -p "$VARIANT_DIR"

  # Build env command prefix
  ENV_CMD="HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 BIDKV_SERVER_STARTUP_TIMEOUT=600"
  if [[ -n "$ENV_VARS" ]]; then
    ENV_CMD="$ENV_CMD $ENV_VARS"
  fi

  # Run experiment
  START_TIME=$(date +%s)
  if eval "$ENV_CMD python3 -m bidkv.experiments.vllm.runner \
    $COMMON_ARGS \
    --output-dir $VARIANT_DIR" > "$VARIANT_LOG" 2>&1; then
    END_TIME=$(date +%s)
    ELAPSED=$((END_TIME - START_TIME))
    echo "[DONE] $VARIANT — ${ELAPSED}s"
    COMPLETED=$((COMPLETED + 1))
  else
    END_TIME=$(date +%s)
    ELAPSED=$((END_TIME - START_TIME))
    echo "[FAIL] $VARIANT — ${ELAPSED}s (see $VARIANT_LOG)"
    FAILED=$((FAILED + 1))
  fi

  # Thorough cleanup between variants: kill any orphan GPU/vLLM processes
  # and wait for port + GPU memory release
  echo "  [CLEANUP] Killing orphan processes..."
  pkill -9 -f "bidkv.experiments.vllm.serve" 2>/dev/null || true
  pkill -9 -f "multiprocessing.spawn" 2>/dev/null || true
  pkill -9 -f "vllm.entrypoints" 2>/dev/null || true
  sleep 5

  # Wait for GPU memory to stabilize (max 60s)
  for i in $(seq 1 20); do
    GPU_USED=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | head -1)
    if [[ "${GPU_USED:-99999}" -lt 3000 ]]; then
      echo "  [CLEANUP] GPU stable at ${GPU_USED} MiB after $((i*3))s"
      break
    fi
    sleep 3
  done

  # Wait for port 8000 to be fully released
  for i in $(seq 1 10); do
    if ! (cat /proc/net/tcp 2>/dev/null | grep -q ":1F40 "); then
      break
    fi
    sleep 2
  done

  echo "  [CLEANUP] Done"
done

echo ""
echo "========================================"
echo "Sensitivity Analysis Complete"
echo "  Completed: $COMPLETED"
echo "  Failed:    $FAILED"
echo "  Results:   $OUTPUT_BASE"
echo "========================================"
