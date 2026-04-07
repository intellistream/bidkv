#!/bin/bash
# SGLang 消融实验 v7: vanilla_sglang vs random_evict vs bidkv @ rate=3.8
#
# v7 变更（相对于 v6）:
#   - BidKV 公式更新：late_penalty = c × 2（原 c² × 2，改为线性）
#   - _compute_keep_score 同步更新：(0.1 + c) × efficiency × starvation
#
# 运行方法:
#   tmux new-session -d -s sglang_v7 && tmux send-keys -t sglang_v7 \
#   "bash scripts/run_sglang_v7_rate38.sh 2>&1 | tee results/sglang_v7_rate38/run.log" Enter

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
OUTPUT_DIR="results/sglang_v7_rate38"
mkdir -p "$OUTPUT_DIR"

echo "=== SGLang v7 Native Ablation @ rate=3.8: vanilla → random → bidkv ==="
echo "BidKV formula: late_penalty = c × 2 (linear, updated from c²)"
echo "Strategies: vanilla_sglang, random_evict, bidkv"
echo "Workload: mixed, rate=3.8, 3 runs per strategy"
echo "Start: $(date)"
echo "Output: $OUTPUT_DIR"
echo ""

for STRAT in vanilla_sglang random_evict bidkv; do
  echo "--- Strategy: $STRAT ($(date)) ---"
  $PYTHON -m bidkv.experiments.sglang.runner \
    --strategies "$STRAT" \
    --workloads "mixed" \
    --mixed-rates "3.8" \
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
import json, glob, os, statistics
from collections import defaultdict

out = os.environ.get("OUTPUT_DIR", "results/sglang_v7_rate38")
files = sorted(glob.glob(f"{out}/sglang__*.json"))

def pct(data, p):
    if not data: return float('nan')
    s = sorted(data)
    return s[min(int(len(s)*p/100), len(s)-1)]

rows_by_strat = defaultdict(list)
for f in files:
    try:
        d = json.load(open(f))
    except Exception as e:
        print(f"  skip {f}: {e}")
        continue
    ok = [r for r in d['request_results'] if not r.get('error')]
    ttft = sorted(r['ttft_ms'] for r in ok if r.get('ttft_ms') is not None)
    tpot = []
    for r in ok:
        if r.get('completion_tokens',0)>1 and r.get('ttft_ms') and r.get('total_latency_ms'):
            tpot.append((r['total_latency_ms']-r['ttft_ms'])/(r['completion_tokens']-1))
    tpot.sort()
    slo = sum(1 for t in ttft if t<=300)/len(ttft)*100 if ttft else 0
    strat = d.get('strategy','?')
    rows_by_strat[strat].append({
        'p50': pct(ttft,50), 'p95': pct(ttft,95),
        'tpot95': pct(tpot,95), 'tput': d['summary']['throughput_rps'],
        'slo': slo, 'ok': len(ok), 'total': len(d['request_results'])
    })

print(f"\n{'Strategy':<20} {'TTFT-p50':>9} {'TTFT-p95':>9} {'TPOT-p95':>9} {'Tput':>7} {'SLO300':>7} {'OK/N':>6}")
print("-"*75)
for strat in ['vanilla_sglang','random_evict','bidkv']:
    rows = rows_by_strat.get(strat, [])
    if not rows:
        print(f"{strat:<20}  (no data)")
        continue
    p50 = statistics.mean(r['p50'] for r in rows)
    p95 = statistics.mean(r['p95'] for r in rows)
    tp95 = statistics.mean(r['tpot95'] for r in rows)
    tput = statistics.mean(r['tput'] for r in rows)
    slo = statistics.mean(r['slo'] for r in rows)
    ok = sum(r['ok'] for r in rows)
    tot = sum(r['total'] for r in rows)
    print(f"{strat:<20} {p50:>8.0f}ms {p95:>8.0f}ms {tp95:>8.1f}ms {tput:>6.2f}r/s {slo:>6.1f}% {ok:>3}/{tot}")

PYEOF
