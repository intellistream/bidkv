#!/bin/bash
# SGLang 验证实验: bidkv + preempt-evict-sjf + h2o-style (修复后版本)
# mixed workload, rate=5.7, 3 runs per strategy
#
# 运行方法:
#   bash scripts/run_sglang_validation_rate57.sh 2>&1 | tee results/sglang_validation_rate57/run.log

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

PYTHON="python3"

OUTPUT_DIR="results/sglang_validation_rate57"
mkdir -p "$OUTPUT_DIR"

echo "=== SGLang Validation: rate=5.7, 3 runs (post KV-stats fix) ==="
echo "Strategies: bidkv, preempt-evict-sjf, h2o-style"
echo "Workload: mixed | Rate: 5.7 | Runs: 3"
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
    2>&1 | tee "$OUTPUT_DIR/runner_${STRAT}.log" \
    || echo "WARNING: $STRAT exited with error, continuing..."
  echo "--- $STRAT done ($(date)) ---"
  echo ""
done

echo "=== All done: $(date) ==="
echo ""
echo "Results in $OUTPUT_DIR:"
ls -lh "$OUTPUT_DIR"/*.json 2>/dev/null | head -20

echo ""
echo "=== Summary (TTFT p95 / Throughput) ==="
$PYTHON - << 'PYEOF'
import json, glob, sys, os

out = os.environ.get("OUTPUT_DIR", "results/sglang_validation_rate57")
files = sorted(glob.glob(f"{out}/sglang__*.json"))
rows = []
for f in files:
    try:
        d = json.load(open(f))
        s = d.get("summary", d)
        strat = d.get("strategy", "?")
        run = d.get("run_index", d.get("run_id", "?"))
        p50 = s.get("ttft_ms_p50", "?")
        p95 = s.get("ttft_ms_p95", "?")
        tput = s.get("throughput_rps", "?")
        ok = s.get("successful_requests", "?")
        total = s.get("total_requests", "?")
        rows.append((strat, run, p50, p95, tput, ok, total))
    except Exception as e:
        print(f"  [warn] {f}: {e}")

rows.sort(key=lambda x: (x[0], x[1]))
print(f"{'strategy':<22} {'run':>3} {'p50_ttft':>10} {'p95_ttft':>10} {'tput':>9} {'ok/total':>12}")
print("-" * 72)
for r in rows:
    strat, run, p50, p95, tput, ok, total = r
    p50s = f"{p50:.0f}ms" if isinstance(p50, (int, float)) else str(p50)
    p95s = f"{p95:.0f}ms" if isinstance(p95, (int, float)) else str(p95)
    tputs = f"{tput:.2f}rps" if isinstance(tput, (int, float)) else str(tput)
    print(f"{strat:<22} {run:>3} {p50s:>10} {p95s:>10} {tputs:>9} {ok!s:>5}/{total!s}")

# Per-strategy mean
print("")
print("=== 3-run mean ===")
from collections import defaultdict
agg = defaultdict(lambda: {"p50":[], "p95":[], "tput":[], "ok":[], "total":[]})
for strat, run, p50, p95, tput, ok, total in rows:
    g = agg[strat]
    if isinstance(p50, float): g["p50"].append(p50)
    if isinstance(p95, float): g["p95"].append(p95)
    if isinstance(tput, float): g["tput"].append(tput)
    if isinstance(ok, int): g["ok"].append(ok)
    if isinstance(total, int): g["total"].append(total)
print(f"{'strategy':<22} {'p50_ttft':>10} {'p95_ttft':>10} {'tput':>9} {'ok/total':>12}")
print("-" * 68)
for strat in sorted(agg):
    g = agg[strat]
    p50m = sum(g["p50"])/len(g["p50"]) if g["p50"] else float("nan")
    p95m = sum(g["p95"])/len(g["p95"]) if g["p95"] else float("nan")
    tputm = sum(g["tput"])/len(g["tput"]) if g["tput"] else float("nan")
    okm = sum(g["ok"]) // len(g["ok"]) if g["ok"] else "?"
    totm = sum(g["total"]) // len(g["total"]) if g["total"] else "?"
    print(f"{strat:<22} {p50m:>9.0f}ms {p95m:>9.0f}ms {tputm:>8.2f}rps {okm!s:>5}/{totm!s}")
PYEOF
