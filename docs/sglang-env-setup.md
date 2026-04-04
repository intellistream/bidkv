# SGLang 实验环境搭建指南

本文档记录在 **不含 CUDA 工具链的裸环境** 中成功跑通 BidKV SGLang 验证实验的完整过程，
包括所有踩过的坑和最终解法。

---

## 一、验证成功的环境配置

| 组件 | 版本 |
|------|------|
| OS | Ubuntu 22.04.5 LTS |
| Python | 3.10.12（系统 python3） |
| GPU | NVIDIA RTX A6000 48GB |
| CUDA Runtime | 12.8（`libcuda.so.580`） |
| CUDA Toolkit / nvcc | **12.8**（`cuda-nvcc-12-8` via NVIDIA apt） |
| SGLang | 0.5.9 |
| flashinfer | 0.6.3 |
| BidKV model | Llama-3.1-8B-Instruct（本地路径） |

---

## 二、关键依赖说明

### 为什么需要 CUDA 12 nvcc？

SGLang 0.5.9 用两套 JIT 编译路径，两者都需要 nvcc：

1. **flashinfer JIT**（attention kernel）：调用 nvcc 编译 `.cu` 文件；
   flashinfer 0.6.3 的 CUDA 代码中使用了 `cudaLaunchKernelEx` API 和 `cuda_fp8.h`，
   这两者都是 **CUDA ≥ 12.0** 引入的，nvcc 11.x 无法编译。

2. **TVM JIT rope kernel**（`sglang.jit_kernel.rope`）：无论任何 attention backend
   都会触发，同样需要 `nvcc` 被 `CUDA_HOME` 正确指向。

> ⚠️ 使用 `--attention-backend triton` 或 `--attention-backend torch_native` **不能绕过**
> 这两个 JIT 编译步骤——它们与 attention backend 无关。

---

## 三、常见错误与原因

### 错误 1：`cuda_fp8.h: No such file or directory`

```
fatal error: cuda_fp8.h: No such file or directory
```

**原因**：`nvidia-cuda-toolkit`（apt）安装的是 nvcc **11.5**，不含 `cuda_fp8.h`
（该头文件从 CUDA 11.8 才引入）。

**错误解法（不可用）**：手动把 `cuda_fp8.h` 从 pip 包
（`nvidia/cuda_runtime/include/`）复制到 `/usr/include/` → 文件存在了，
但 nvcc 11.5 编译时报 `cuda_fp8.hpp` 里的 `__CUDA_ARCH_HAS_FEATURE__(SM100_ALL)` 语法错误。

**根本原因**：pip 里的 `cuda_fp8.h` 是 CUDA 12.8 版本，nvcc 11.5 无法解析。

---

### 错误 2：`identifier "cudaLaunchKernelEx" is undefined`

```
error: identifier "cudaLaunchKernelEx" is undefined
```

**原因**：`cudaLaunchKernelEx` 是 CUDA 12.0 引入的 API，nvcc 11.5 不认识。
即便头文件问题解决，编译仍会失败。

---

### 错误 3：`RuntimeError: Could not find nvcc and default cuda_home='/usr/local/cuda' doesn't exist`

**原因**：
- `nvidia-cuda-toolkit` 只安装 nvcc 到 `/usr/bin/nvcc`，
  但 `/usr/local/cuda` symlink 不存在；
- flashinfer 先检查 `CUDA_HOME` 环境变量，找不到则报错。

---

## 四、正确解法：安装 CUDA 12.8 nvcc

### 步骤 1：添加 NVIDIA 官方 apt 仓库

```bash
wget -q https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb \
     -O /tmp/cuda-keyring.deb
dpkg -i /tmp/cuda-keyring.deb
apt-get update
```

### 步骤 2：安装 cuda-nvcc-12-8

```bash
apt-get install -y cuda-nvcc-12-8
```

安装后，nvcc 12.8 位于 `/usr/local/cuda-12.8/bin/nvcc`，
apt 会自动创建 `/usr/local/cuda` → `/usr/local/cuda-12.8` 的 symlink。

验证：
```bash
/usr/local/cuda-12.8/bin/nvcc --version
# Cuda compilation tools, release 12.8, V12.8.93
```

### 步骤 3：清理旧的 flashinfer JIT 缓存

如果之前已经尝试过编译（留有损坏的 cache）：

```bash
rm -rf /root/.cache/flashinfer
```

---

## 五、运行实验

### 5.1 所需环境变量

在运行脚本前，需设置以下变量（已集成在 `scripts/run_sglang_validation_v1.sh`）：

| 变量 | 值 | 说明 |
|------|----|------|
| `PYTHONPATH` | `<repo>/src` | bidkv 包路径（无需 editable install） |
| `HF_HUB_OFFLINE` | `1` | 禁止从 HuggingFace 下载模型 |
| `TRANSFORMERS_OFFLINE` | `1` | 同上 |
| `BIDKV_MODEL` | `/home/models/Llama-3.1-8B-Instruct` | 本地模型路径（默认 auto-detect 失败时必须设置） |
| `no_proxy` / `NO_PROXY` | `127.0.0.1,localhost` | **必须设置**，否则系统 HTTP 代理会拦截健康检查请求 |
| `CUDA_HOME` | `/usr/local/cuda-12.8` | **必须设置**，确保 flashinfer/TVM 找到正确的 nvcc |
| `PATH` | `/usr/local/cuda-12.8/bin:$PATH` | 将 nvcc 12.8 置于搜索路径首位 |

### 5.2 运行验证实验（3 策略，1 run）

```bash
cd /home/bidkv
bash scripts/run_sglang_validation_v1.sh 2>&1 | tee results/sglang_validation_v1/run.log
```

脚本会依次运行三个策略，每个都：
1. 启动独立的 SGLang server（`bidkv.experiments.sglang.serve_entry`）；
2. 发送 5 条 warmup 请求；
3. 按 Poisson 到达模型发送 1000 条 trace 请求；
4. 保存 JSON 结果，停止 server。

首次运行时，flashinfer JIT 编译约需 **5–15 分钟**（编译 ~80 个 CUDA kernel），
之后有缓存（`/root/.cache/flashinfer/`），每次复用。

### 5.3 完整矩阵实验（v2.3 冻结 54 runs）

```bash
cd /home/bidkv
export CUDA_HOME=/usr/local/cuda-12.8
export PATH="/usr/local/cuda-12.8/bin:$PATH"
export PYTHONPATH=src
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export BIDKV_MODEL=/home/models/Llama-3.1-8B-Instruct
export no_proxy="127.0.0.1,localhost"
export NO_PROXY="127.0.0.1,localhost"

python3 -m bidkv.experiments.sglang.runner \
    --strategies "sglang_default,slack_aware,bidkv" \
    --workloads "mixed,long_context" \
    --mixed-rates "2.0,3.8,5.7" \
    --long-rates "0.35,0.5,0.7" \
    --runs 3 \
    --max-total-tokens 9600 \
    --output-dir results/sglang_full \
    --traces-dir experiments/vllm/traces \
    --resume
```

---

## 六、验证结果（2026-04-02）

实验：3 策略 × mixed workload × rate=3.8 × 1 run

| 策略 | TTFT p50 | TTFT p95 | TTFT p99 | tput | success |
|------|---------|---------|---------|------|--------|
| bidkv | 101 ms | 4729 ms | 6002 ms | 3.61 rps | 1000/1000 |
| preempt-evict-sjf | 100 ms | 4363 ms | 6921 ms | 3.62 rps | 1000/1000 |
| h2o-style | 100 ms | 5269 ms | 7703 ms | 3.60 rps | 1000/1000 |

结果文件：`results/sglang_validation_v1/`

---

## 七、常见配置问题 Q&A

**Q: 为什么不能用 `apt-get install nvidia-cuda-toolkit`？**

A: 这个包安装的是 nvcc **11.5**，不支持 CUDA 12 API。
必须从 NVIDIA 官方 CUDA 仓库安装 `cuda-nvcc-12-8`。

**Q: 为什么加了 `--attention-backend triton` 还是报错？**

A: triton 后端通过 TVM 进行 JIT 编译，TVM 的 `_find_cuda_home()` 同样
需要 nvcc，且会用于编译 rope kernel。attention backend 与 rope JIT 是两条独立的路径。

**Q: 为什么健康检查一直 hang/超时？**

A: 系统设置了 HTTP 代理（如 `http_proxy=http://127.0.0.1:7890`），它会拦截
对 `127.0.0.1:30000` 的健康检查请求。必须设置 `no_proxy=127.0.0.1,localhost`。

**Q: 没有 sagellm conda 环境，能跑吗？**

A: 可以。BidKV 实验框架只需系统 `python3`，通过 `PYTHONPATH=src` 加载。
系统 Python 3.10 + pip 安装的 sglang/flashinfer/torch 即可。

**Q: 模型加载失败（OSError / HuggingFace 404）怎么办？**

A: 设置 `BIDKV_MODEL=/path/to/local/model`。当 `HF_HUB_OFFLINE=1` 时，
SGLang 必须能在本地找到模型目录，不会联网下载。
