#!/bin/bash
# Sensitivity analysis for BidKV v8 formula parameters.
# Tests 10 variants (1 default + 9 non-default) × 3 runs = 30 runs total.
# Rate fixed at 3.8 req/s (mid-high pressure, max strategy differentiation).
#
# Axes:
#   completion_weight: BIDKV_COMPLETION_WEIGHT  default=0.5  test=0.25,1.0,2.0
#   starvation_weight: BIDKV_STARVATION_WEIGHT  default=0.3  test=0.1,0.6,1.0
#   kv_gate:           BIDKV_KV_GATE            default=0.95 test=0.85,0.90,0.98
#
# Usage:
#   nohup bash scripts/run_sensitivity_v2.sh > results/vllm_sensitivity_v2/run.log 2>&1 &

set -euo pipefail

export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

RESULTS_DIR="results/vllm_sensitivity_v2"
cd /home/cyb/bidkv

mkdir -p "$RESULTS_DIR"

echo "=== BidKV Sensitivity Analysis v2 ==="
echo "Start: $(date)"
echo "Output dir: $RESULTS_DIR"
echo ""

# GPU cleanup helper
gpu_cleanup() {
    echo "[cleanup] Waiting 10s for GPU memory release..."
    sleep 10
    echo "[cleanup] Done"
}

# Run a single variant.
# Args: label  completion_weight  starvation_weight  kv_gate
run_variant() {
    local LABEL=$1
    local CW=$2
    local SW=$3
    local KG=$4

    echo "--- Variant: $LABEL  CW=$CW  SW=$SW  KG=$KG  ($(date)) ---"

    export BIDKV_COMPLETION_WEIGHT="$CW"
    export BIDKV_STARVATION_WEIGHT="$SW"
    export BIDKV_KV_GATE="$KG"

    # custom output sub-dir so files don't collide across variants
    local VARIANT_DIR="${RESULTS_DIR}/${LABEL}"
    mkdir -p "$VARIANT_DIR"

    conda run -n sagellm python -m bidkv.experiments.vllm.runner \
        --strategies "bidkv" \
        --workloads "mixed" \
        --mixed-rates "3.8" \
        --runs 3 \
        --num-gpu-blocks-override 600 \
        --output-dir "$VARIANT_DIR" \
        --ttft-target-ms 1000 \
        --resume || echo "WARNING: variant $LABEL exited with error, continuing..."

    echo "--- $LABEL done ($(date)) ---"
    echo ""
    gpu_cleanup
}

# ── 1. Default (w_c=0.5  w_s=0.3  gate=0.95) ─────────────────────────────────
run_variant "default"        0.5   0.3   0.95

# ── 2. Completion weight axis ─────────────────────────────────────────────────
run_variant "cw_0.25"        0.25  0.3   0.95
run_variant "cw_1.0"         1.0   0.3   0.95
run_variant "cw_2.0"         2.0   0.3   0.95

# ── 3. Starvation weight axis ─────────────────────────────────────────────────
run_variant "sw_0.1"         0.5   0.1   0.95
run_variant "sw_0.6"         0.5   0.6   0.95
run_variant "sw_1.0"         0.5   1.0   0.95

# ── 4. KV gate axis ───────────────────────────────────────────────────────────
run_variant "gate_0.85"      0.5   0.3   0.85
run_variant "gate_0.90"      0.5   0.3   0.90
run_variant "gate_0.98"      0.5   0.3   0.98

echo "=== All variants complete: $(date) ==="
echo "Results saved to: $RESULTS_DIR"
