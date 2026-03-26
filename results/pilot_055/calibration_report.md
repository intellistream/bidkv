# Issue #055 Calibration Report

**日期**: 2026-03-19
**操作者**: Agent
**状态**: ✅ 完成

## 1. 数据来源

校准基于以下已有实验数据（无需额外 pilot 运行）：

| 数据集 | 路径 | 内容 | 策略 | Workload | Rates |
|--------|------|------|------|----------|-------|
| Formal long_context | `results/formal/long_context/` | 72 runs (8策略×3rate×3repeat) | 全部 8 策略 | long_context | 0.35, 0.5, 0.7 |
| Pilot V3 | `results/pilot_v3/` | 12 runs | preempt-evict, h2o-style | mixed + long_context | 0.5, 0.7, 1.0 |
| Pilot V3 Mixed High | `results/pilot_v3_mixed_high/` | 6 runs | preempt-evict, h2o-style | mixed | 2.0, 3.8, 5.7 |

总计 **90 runs** 可用于校准。

## 2. Rate 选择依据

### 2.1 Mixed Workload

数据来源：`results/pilot_v3/` (6 runs) + `results/pilot_v3_mixed_high/` (6 runs)

| Rate (req/s) | Strategy | Total | Success | Throughput (rps) | P99 TTFT (ms) | 压力分级 |
|---|---|---|---|---|---|---|
| 0.5 | preempt-evict | 50 | 50 | 0.679 | 343 | 极低 |
| 0.7 | preempt-evict | 50 | 50 | 0.913 | 355 | 低 |
| 1.0 | preempt-evict | 50 | 50 | 1.231 | 364 | 低 |
| **2.0** | preempt-evict | 80 | 80 | **2.064** | 356 | **低压 ✓** |
| **3.8** | preempt-evict | 80 | 80 | **3.318** | 450 | **中压 ✓** |
| **5.7** | preempt-evict | 80 | 80 | **3.810** | 440 | **高压 ✓** |

**选择理由**：
- **rate_low = 2.0**: 吞吐量 2.06 rps，P99 TTFT 356ms，系统从容处理
- **rate_mid = 3.8**: 吞吐量 3.32 rps（开始饱和），P99 TTFT 升至 450ms
- **rate_high = 5.7**: 吞吐量 3.81 rps（完全饱和，请求速率 5.7 vs 实际 3.8），策略差异最大化

### 2.2 Long Context Workload

数据来源：`results/formal/long_context/` (72 runs) + `results/pilot_v3/` (6 runs)

| Rate (req/s) | Strategy | Total | Success | Throughput (rps) | P99 TTFT (ms) | 压力分级 |
|---|---|---|---|---|---|---|
| **0.35** | preempt-evict (formal) | 150 | 150 | **0.403** | - | **低压 ✓** |
| 0.5 | preempt-evict (pilot) | 25 | 25 | 0.503 | 3221 | 中低 |
| **0.5** | preempt-evict (formal) | 150 | 150 | **0.558** | - | **中压 ✓** |
| **0.7** | preempt-evict (formal) | 75 | 75 | **0.672** | - | **高压 ✓** |
| 0.7 | preempt-evict (pilot) | 25 | 25 | 0.628 | 4863 | 中高 |
| 1.0 | preempt-evict (pilot) | 25 | 25 | 0.668 | 10127 | 极高 |

**选择理由**：
- **rate_low = 0.35**: 吞吐量 0.40 rps，全部请求成功，baseline 无压力
- **rate_mid = 0.5**: 吞吐量 0.56 rps，中等压力，compression 策略开始分化
- **rate_high = 0.7**: 吞吐量 0.67 rps（饱和），P99 TTFT >4.8s，高压下策略差异最大

## 3. 最终冻结值

⚠️ **FROZEN — RULE RATE-FREEZE: 冻结后不可基于策略表现调整**

```python
# src/bidkv/experiments/vllm/config.py
WORKLOAD_REQUEST_RATES = {
    "mixed": (2.0, 3.8, 5.7),        # req/s
    "long_context": (0.35, 0.5, 0.7), # req/s
}

# src/bidkv/experiments/sglang/config.py — 与 vLLM 共享冻结 rates
WORKLOAD_REQUEST_RATES = {
    "mixed": (2.0, 3.8, 5.7),
    "long_context": (0.35, 0.5, 0.7),
}
```

## 4. Trace 冻结状态

所有 frozen traces 使用 **seed=42**，存储于 `results/formal/traces/`。

| 文件名 | Workload | Rate | Requests | SHA-256 (前16位) |
|--------|----------|------|----------|-----------------|
| mixed_rate2.0.json | mixed | 2.0 | 1000 | 6221f218b1a5... |
| mixed_rate3.8.json | mixed | 3.8 | 1000 | 67f67193b7e6... |
| mixed_rate5.7.json | mixed | 5.7 | 1000 | c1020d5edaf3... |
| long_rate0.35.json | long_context | 0.35 | 500 | 37ec72f2151b... |
| long_rate0.5.json | long_context | 0.5 | 500 | 1480e30b3175... |
| long_rate0.7.json | long_context | 0.7 | 500 | b4febee0fd36... |

可复现性验证：`python -m bidkv.experiments.vllm.freeze_traces --mode verify --use-frozen-rates`
验证结果：**PASSED** (6/6 hashes match)

## 5. Config 更新记录

| 文件 | 变更 |
|------|------|
| `src/bidkv/experiments/vllm/config.py` | 新增 `WORKLOAD_REQUEST_RATES` dict，更新 `DEFAULT_REQUEST_RATES` 为 `(2.0, 3.8, 5.7)` |
| `src/bidkv/experiments/vllm/config.py` | `ExperimentConfig` 新增 `workload_rates` 字段 + `get_rates_for_workload()` |
| `src/bidkv/experiments/sglang/config.py` | 同步新增 `WORKLOAD_REQUEST_RATES`，更新 `DEFAULT_REQUEST_RATES` |
| `src/bidkv/experiments/sglang/config.py` | `SGLangExperimentConfig` 新增 `workload_rates` 字段 + `get_rates_for_workload()` |
| `src/bidkv/experiments/vllm/runner.py` | 使用 `get_rates_for_workload()` 替代 `config.request_rates` |
| `src/bidkv/experiments/sglang/runner.py` | 同步更新 per-workload rates |
| `src/bidkv/experiments/vllm/freeze_traces.py` | 支持 `--use-frozen-rates` 和 per-workload rates |

## 6. 全量实验矩阵（#053）

### vLLM：8 策略 × 2 workloads × 3 rates × 3 runs
- mixed: 8 × 3 × 3 = 72 runs (rates: 2.0, 3.8, 5.7)
- long_context: 8 × 3 × 3 = 72 runs (rates: 0.35, 0.5, 0.7)
- **总计: 144 runs**

### SGLang：4 策略 × 2 workloads × 3 rates × 3 runs
- mixed: 4 × 3 × 3 = 36 runs (rates: 2.0, 3.8, 5.7)
- long_context: 4 × 3 × 3 = 36 runs (rates: 0.35, 0.5, 0.7)
- **总计: 72 runs**
