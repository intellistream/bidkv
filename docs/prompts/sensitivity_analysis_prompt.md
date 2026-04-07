# BidKV 敏感性分析实验

## 目标

验证 BidKV v8 公式参数的鲁棒性。公式为：

```
U = tokens_freed / (δ + ε)
δ = 1 + w_c · completion + w_s · num_preemptions
```

v8 默认值：`w_c=0.5`, `w_s=0.3`。
scheduler_hook 中 KV gate 默认值：`0.95`（KV 使用率低于此值时不激活 quality-aware reorder）。

## 敏感性轴

每次只变一个参数，其余保持默认。

| 轴 | 环境变量 | 默认 | 测试值 |
|----|---------|------|--------|
| Completion weight | `BIDKV_COMPLETION_WEIGHT` | 0.5 | 0.25, 1.0, 2.0 |
| Starvation weight | `BIDKV_STARVATION_WEIGHT` | 0.3 | 0.1, 0.6, 1.0 |
| KV gate threshold | `BIDKV_KV_GATE` | 0.95 | 0.85, 0.90, 0.98 |

共 10 组（9 个非默认 + 1 个 default），每组 3 runs = **30 runs**。

## 实验参数

- Rate: **3.8** req/s（中高压，策略分化明显）
- Workload: mixed
- Trace: seed=42 frozen
- 服务端参数：v8-frozen（gpu_mem=0.5, blocks=600, seqs=32, block_size=16, model_len=8192）

## 前置：代码修改

当前 `bidkv_strategy.py` 中 `0.5` 和 `0.3` 是硬编码的，`scheduler_hook.py` 中 `0.95` 也是硬编码的。需要改成读取环境变量，默认值不变：

**bidkv_strategy.py**（将已有的环境变量定义的默认值改为 v8 值，并在公式中使用）：
```python
_COMPLETION_WEIGHT = float(os.environ.get("BIDKV_COMPLETION_WEIGHT", "0.5"))
_STARVATION_WEIGHT = float(os.environ.get("BIDKV_STARVATION_WEIGHT", "0.3"))
```
然后把 `select_victims()` 中的：
```python
quality_delta = 1.0 + 0.5 * completion
quality_delta += req.num_preemptions * 0.3
```
改为：
```python
quality_delta = 1.0 + _COMPLETION_WEIGHT * completion
quality_delta += req.num_preemptions * _STARVATION_WEIGHT
```

**scheduler_hook.py**（在文件顶部添加，然后替换硬编码值）：
```python
_KV_GATE = float(os.environ.get("BIDKV_KV_GATE", "0.95"))
```
把 `if usage < 0.95:` 改为 `if usage < _KV_GATE:`。

修改后跑一遍 `pytest tests/ -v -x` 确认全部通过。

## 运行

```bash
cd /home/bidkv
nohup bash scripts/run_sensitivity_v2.sh > results/vllm_sensitivity_v2/run.log 2>&1 &
```

运行脚本逻辑：遍历 10 组变体，每组通过环境变量设置对应参数值，调用 `python3 -m bidkv.experiments.vllm.runner`，每组之间做 GPU cleanup。

## 验证

实验完成后，首先检查 **default 变体的 TTFT P95 是否在 550–750ms 范围**（v8 参考值 631ms）。
若偏差过大（>1000ms），说明代码改动引入了问题。

## 分析

用分析脚本对每个轴计算 SLO/TTFT 的 span（最大值 - 最小值），判断鲁棒性：
- SLO span < 5pp 且 TTFT span < 20%：**鲁棒**
- SLO span < 10pp 且 TTFT span < 40%：**中等敏感**
- 否则：**敏感**，需在论文中讨论
