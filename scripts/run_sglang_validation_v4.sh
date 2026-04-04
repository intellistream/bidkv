#!/bin/bash
# SGLang 验证实验 v4: 更新后方法（双受害者，阈值优化）
# bidkv + preempt-evict-sjf + h2o-style(=largest-first)
# mixed workload, rate=5.7, 3 runs per strategy
#
# v4 主要变更（相对于 v3）:
#   - 双受害者选择：KV>95% 时同时驱逐 2 个请求
#   - proactive preempt 阈值: 90%→85%, cooldown: 5s→2s
#   - SRPT 阈值: 80%→75%, cooldown: 1.5s→0.8s
#   - preempt-evict-sjf 加入 LIFO 组（无 running-reorder）
#   - h2o_style → largest_first 重命名（registry 对齐）
#
# 运行方法:
#   bash scripts/run_sglang_validation_v4.sh 2>&1 | tee results/sglang_validation_v4/run.log

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
OUTPUT_DIR="results/sglang_validation_v4"
mkdir -p "$OUTPUT_DIR"

echo "=== SGLang Validation v4: rate=5.7, 3 runs ==="
echo "Strategies: bidkv, preempt-evict-sjf, h2o-style(largest-first)"
echo "v4 changes: dual-victim(>95%), proactive 85%/2s, SRPT 75%/0.8s"
echo "Start: $(date)"
echo "Output: $OUTPUT_DIR"
echo ""

for STRAT in bidkv preempt-evict-sjf h2o-style; do
  echo "--- Strategy: $STRAT ($(date)) ---"
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

export OUTPUT_DIR
python3 - << 'PYEOF'
import json, glob, os
from collections import defaultdict

out = os.environ.get("OUTPUT_DIR", "results/sglang_validation_v4")
files = sorted(glob.glob(f"{out}/sglang__*.json"))
rows = []
for f in files:
    d = json.load(open(f))
    s = d.get("summary", d)
    strat = d.get("strategy", "?")
    run = d.get("run_index", "?")
    p50 = s.get("ttft_ms_p50", float("nan"))
    p95 = s.get("ttft_ms_p95", float("nan"))
    tput = s.get("throughput_rps", float("nan"))
    ok = s.get("successful_requests", "?")
    total = s.get("total_requests", "?")
    rows.append((strat, run, p50, p95, tput, ok, total))

rows.sort(key=lambda x: (x[0], x[1]))
print(f"\n=== v4 Per-run Results (mixed, rate=5.7) ===")
print(f"{'strategy':<22} {'run':>3} {'p50_ttft':>10} {'p95_ttft':>10} {'tput':>9} {'ok/total':>12}")
print("-" * 72)
for strat, run, p50, p95, tput, ok, total in rows:
    print(f"{strat:<22} {run:>3} {p50:>9.0f}ms {p95:>9.0f}ms {tput:>8.2f}rps {ok!s:>5}/{total!s}")

agg = defaultdict(lambda: {"p50":[], "p95":[], "tput":[]})
for strat, run, p50, p95, tput, ok, total in rows:
    g = agg[strat]
    if isinstance(p50, float): g["p50"].append(p50)
    if isinstance(p95, float): g["p95"].append(p95)
    if isinstance(tput, float): g["tput"].append(tput)

print(f"\n=== 3-run Mean ===")
print(f"{'strategy':<22} {'p50_ttft':>10} {'p95_ttft':>10} {'tput':>9} {'nruns':>5}")
print("-" * 60)
for strat in sorted(agg):
    g = agg[strat]
    p50m = sum(g["p50"])/len(g["p50"]) if g["p50"] else float("nan")
    p95m = sum(g["p95"])/len(g["p95"]) if g["p95"] else float("nan")
    tputm = sum(g["tput"])/len(g["tput"]) if g["tput"] else float("nan")
    n = len(g["p50"])
    print(f"{strat:<22} {p50m:>9.0f}ms {p95m:>9.0f}ms {tputm:>8.2f}rps {n:>5}")

print("""
=== v3 Baseline (3-run mean, same sampling fallback) ===
bidkv                   144ms     4791ms   4.13rps
h2o-style               115ms      655ms   5.45rps  (PyTorch sampler; shorter outputs)
preempt-evict-sjf       196ms     5691ms   3.42rps  (高方差: run0=4.16rps, run1/2=3.05rps)
""")
PYEOF
