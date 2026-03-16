"""bidkv.experiments.vllm — vLLM 7-baseline + Oracle 完整实验框架。

论文 §6 Evaluation 的主要数据来源。

模块结构
--------
- ``config``: 实验参数配置
- ``workload``: 工作负载加载与冻结
- ``collector``: 运行时指标收集
- ``runner``: 实验执行编排
- ``analysis``: 结果分析与论文 Figure 生成
"""

from __future__ import annotations
