#!/bin/bash
# v3 resume: 补跑 pe-sjf run1/2 + h2o-style run0/1/2
# 所有 run 写入 results/sglang_validation_v3（--resume 跳过已存在文件）

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

OUTPUT_DIR="results/sglang_validation_v3"
mkdir -p "$OUTPUT_DIR"

echo "=== v3 resume: pe-sjf + h2o-style ($(date)) ==="

# 补跑 preempt-evict-sjf（run0 已存在，--resume 会跳过）
echo "--- preempt-evict-sjf ($(date)) ---"
sleep 5
python3 -m bidkv.experiments.sglang.runner \
  --strategies "preempt-evict-sjf" \
  --workloads "mixed" \
  --mixed-rates "5.7" \
  --runs 3 \
  --max-total-tokens 9600 \
  --output-dir "$OUTPUT_DIR" \
  --traces-dir "experiments/vllm/traces" \
  --resume \
  2>&1 | tee "$OUTPUT_DIR/runner_preempt-evict-sjf_resume.log" \
  || echo "WARNING: preempt-evict-sjf exited with error"

echo "--- pe-sjf done ($(date)) ---"
sleep 15   # 等 GPU 内存完全释放，避免 transformers.AutoProcessor 缓存污染

# 补跑 h2o-style
echo "--- h2o-style ($(date)) ---"
python3 -m bidkv.experiments.sglang.runner \
  --strategies "h2o-style" \
  --workloads "mixed" \
  --mixed-rates "5.7" \
  --runs 3 \
  --max-total-tokens 9600 \
  --output-dir "$OUTPUT_DIR" \
  --traces-dir "experiments/vllm/traces" \
  --resume \
  2>&1 | tee "$OUTPUT_DIR/runner_h2o-style_resume.log" \
  || echo "WARNING: h2o-style exited with error"

echo "--- h2o-style done ($(date)) ---"

echo ""
echo "=== v3 Resume complete: $(date) ==="
echo ""

python3 - << 'PYEOF'
import json, glob
from collections import defaultdict

out = "results/sglang_validation_v3"
files = sorted(glob.glob(f"{out}/sglang__*.json"))
print(f"Total result files: {len(files)}")
rows = []
for f in files:
    d = json.load(open(f))
    s = d.get("summary", d)
    strat = d.get("strategy", "?")
    run   = d.get("run_index", "?")
    p50   = s.get("ttft_ms_p50", 0)
    p95   = s.get("ttft_ms_p95", 0)
    p99   = s.get("ttft_ms_p99", 0)
    tput  = s.get("throughput_rps", 0)
    ok    = s.get("successful_requests", 0)
    total = s.get("total_requests", 0)
    rows.append((strat, run, p50, p95, p99, tput, ok, total))

rows.sort(key=lambda x: (x[0], x[1]))
print(f"\n{'strategy':<22} {'run':>3} {'p50':>8} {'p95':>8} {'p99':>8} {'tput':>9} {'ok/total':>12}")
print("-" * 80)
for strat, run, p50, p95, p99, tput, ok, total in rows:
    print(f"{strat:<22} {run:>3} {p50:>7.0f}ms {p95:>7.0f}ms {p99:>7.0f}ms {tput:>8.2f}rps {ok:>5}/{total}")

print("\n=== 3-run mean ===")
agg = defaultdict(lambda: {"p50":[], "p95":[], "p99":[], "tput":[]})
for strat, run, p50, p95, p99, tput, ok, total in rows:
    agg[strat]["p50"].append(p50)
    agg[strat]["p95"].append(p95)
    agg[strat]["p99"].append(p99)
    agg[strat]["tput"].append(tput)

# Load v2 for comparison
v2_files = sorted(glob.glob("results/sglang_validation_v2/sglang__*.json"))
v2_agg = defaultdict(lambda: {"p95":[]})
for f in v2_files:
    d = json.load(open(f))
    s = d.get("summary", d)
    v2_agg[d.get("strategy","?")]["p95"].append(s.get("ttft_ms_p95", 0))

print(f"\n{'strategy':<22} {'p50':>8} {'p95':>8} {'p99':>8} {'tput':>9} {'vs_v2_p95':>12}")
print("-" * 75)
for strat in sorted(agg):
    g = agg[strat]
    n = len(g["p95"])
    if n == 0: continue
    p50m = sum(g["p50"])/n; p95m = sum(g["p95"])/n
    p99m = sum(g["p99"])/n; tputm = sum(g["tput"])/n
    v2g = v2_agg.get(strat, {}).get("p95", [])
    if v2g:
        v2p95 = sum(v2g)/len(v2g)
        delta = (p95m - v2p95) / v2p95 * 100
        vs = f"{delta:+.1f}% ({n} runs)"
    else:
        vs = f"({n} runs)"
    print(f"{strat:<22} {p50m:>7.0f}ms {p95m:>7.0f}ms {p99m:>7.0f}ms {tputm:>8.2f}rps {vs:>14}")
PYEOF
