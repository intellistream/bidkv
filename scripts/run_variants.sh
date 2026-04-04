#!/usr/bin/env bash
# Quick A/B test: BidKV parameter variants at rate=5.7 (weakest point)
# Each variant ~5min. Total ~30min for 6 variants.
set -euo pipefail

OUTBASE="results/vllm_v10_variants"
COMMON_ARGS=(
    --strategies "bidkv"
    --workloads "mixed"
    --mixed-rates "5.7"
    --runs 1
    --gpu-memory-utilization 0.5
    --num-gpu-blocks-override 600
    --max-num-seqs 32
    --block-size 16
    --max-model-len 8192
)

run_variant() {
    local name="$1"
    shift
    local outdir="${OUTBASE}/${name}"
    echo "================================================================"
    echo "=== Running variant: ${name}"
    echo "=== Env: $@"
    echo "================================================================"
    mkdir -p "${outdir}"
    env "$@" \
        HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
        conda run -n sagellm python -m bidkv.experiments.vllm.runner \
        "${COMMON_ARGS[@]}" \
        --output-dir "${outdir}" \
        2>&1 | tee "${outdir}/run.log"
    echo "=== Completed: ${name}"
    echo ""
}

# v8 baseline (default params): gate=0.95, prompt_guard=500, w_c=0.5, w_s=0.3
run_variant "v8_baseline" \
    BIDKV_REORDER_GATE=0.95 BIDKV_PROMPT_GUARD=500 \
    BIDKV_COMPLETION_W=0.5 BIDKV_STARVATION_W=0.3

# Angle 2a: Lower reorder gate to 85%
run_variant "gate85" \
    BIDKV_REORDER_GATE=0.85 BIDKV_PROMPT_GUARD=500 \
    BIDKV_COMPLETION_W=0.5 BIDKV_STARVATION_W=0.3

# Angle 2b: Remove avg_prompt guard (set to 0)
run_variant "no_prompt_guard" \
    BIDKV_REORDER_GATE=0.95 BIDKV_PROMPT_GUARD=0 \
    BIDKV_COMPLETION_W=0.5 BIDKV_STARVATION_W=0.3

# Angle 2c: Lower gate + remove prompt guard
run_variant "gate85_no_guard" \
    BIDKV_REORDER_GATE=0.85 BIDKV_PROMPT_GUARD=0 \
    BIDKV_COMPLETION_W=0.5 BIDKV_STARVATION_W=0.3

# Angle 3a: Stronger completion protection (w_c=2.0 quadratic-like effect)
run_variant "strong_completion" \
    BIDKV_REORDER_GATE=0.95 BIDKV_PROMPT_GUARD=500 \
    BIDKV_COMPLETION_W=2.0 BIDKV_STARVATION_W=0.3

# Angle 3b: Stronger starvation penalty
run_variant "strong_starvation" \
    BIDKV_REORDER_GATE=0.95 BIDKV_PROMPT_GUARD=500 \
    BIDKV_COMPLETION_W=0.5 BIDKV_STARVATION_W=1.0

# Angle combined: best gate + strong completion
run_variant "gate85_strong_completion" \
    BIDKV_REORDER_GATE=0.85 BIDKV_PROMPT_GUARD=0 \
    BIDKV_COMPLETION_W=2.0 BIDKV_STARVATION_W=0.5

echo "All variants completed!"
echo "Analyze with: python3 scripts/analyze_variants.py"
