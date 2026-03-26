================================================================================
  BidKV vLLM 实现路线重构方案
  Issue-047 · Phase C crash 修复与实验策略调整
  版本：v1.0-draft（待 reviewer 冻结）
  日期：2026-03-19
================================================================================

目录
  §1  问题定性
  §2  方案对比与抉择
  §3  推荐路线（双轨方案）
  §4  禁止路线
  §5  最小 vLLM 内核扩展 API 设计
  §6  Recompute Fallback 执行器设计
  §7  实验执行策略（Mode A / Mode B）
  §8  论文叙事建议
  §9  实施排期与优先级
  §10 风险评估

================================================================================
§1  问题定性
================================================================================

1.1 直接现象

    所有 BidKV 注入策略（h2o-style, uniform, global-nobid, slack-aware, bidkv）
    在 long_context workload 下运行约 180 秒后 crash。
    Preempt-evict（无 BidKV 注入）150/150 成功。
    所有注入策略共享同一 crash 路径：execute_compression() → _free_tail_blocks()。

1.2 直接根因（三层）

    Bug 1：block ownership violation
        _free_tail_blocks() 调用 block_pool.free_blocks(tail_blocks) 直接绕过
        coordinator / SingleTypeKVCacheManager 的 req_to_blocks 映射和 refcount
        管理。被释放的 block 进入 free list，但 coordinator 仍认为该 block 属于
        原请求。

    Bug 2：coordinator 内部状态不一致
        coordinator.get_blocks(request_id) 返回的是 req_to_blocks 内部 list 的
        直接引用。del group_blocks[-n:] 直接缩短了 coordinator 持有的列表。
        后续 allocate_slots() 依赖 len(req_to_blocks[req_id]) 计算
        num_new_blocks = cdiv(num_tokens, block_size) - len(req_blocks)，
        列表缩短后会导致多分配 block（分配到已被其他请求复用的地址上）。

    Bug 3：CUDA 内存损坏 → stale block reuse
        被释放的 block 被新请求通过正常分配流程获取。旧请求继续 decode 时写入
        该 block 的 KV cache 地址，新请求同时也在读/写同一地址。
        结果：约 180 秒后累积足够冲突，触发 CUDA memory corruption 或 segfault。

1.3 核心定性判断

    ┌─────────────────────────────────────────────────────────────────┐
    │  这不是"实现还不够完善"的 bug，                                │
    │  而是"vLLM V1 当前公开 KV 生命周期不支持该状态转换"的          │
    │  设计层面冲突。                                                │
    └─────────────────────────────────────────────────────────────────┘

    理由：

    a) vLLM v1 的 KVCacheManager 提供的公开语义只有两种：
       - allocate_slots(request)：为请求分配新 block
       - free(request)：释放请求的全部 block
       不存在 "partial free / partial truncation" 的公开接口。

    b) coordinator 内部的 req_to_blocks 映射是 all-or-nothing 语义：
       请求要么占有全部 block，要么在 free() 后一次性释放全部。
       没有中间状态（"请求占有前 N 个 block，后 M 个已释放"）。

    c) block_pool 的 free_blocks() 是底层操作，不更新上层 coordinator
       的 req_to_blocks 映射。在 vLLM 内部，只有 SingleTypeKVCacheManager
       的 free() 方法才会同时更新两层状态。

    d) Request.num_computed_tokens 是外部可写字段，但 allocate_slots()
       依赖 len(req_to_blocks) 而非 num_computed_tokens 来计算新 block
       需求。即使正确更新 num_computed_tokens，block 层面的不一致仍然存在。

1.4 为什么之前的 null-block 方案也不可接受

    之前提出的 null-block 替换方案（借鉴 vLLM 的 remove_skipped_blocks()
    模式）看似保持了列表长度不变，但仍有根本缺陷：

    a) 仍然绕过 coordinator：block_pool.free_blocks(removed) 直接操作底层，
       coordinator 的状态（如 prefix hash、block ref_cnt 统计）未更新。

    b) null_block 是共享单例：所有被替换位置指向同一 null_block 实例。
       若两个请求同时有 null_block 占位，allocate_slots 对 block 唯一性的
       隐式假设可能被违反。

    c) 语义不明确：null_block 的 KV data 是 stale/zero，请求继续 decode
       时 attention 计算会读取这些无意义数据。这不是"压缩"，而是"静默数据
       损坏"——只是不会 crash 而已。

    d) remove_skipped_blocks() 的真实使用场景是 speculative decoding 的
       "丢弃未命中的投机 token"，其前提是这些 block 从未被 attention
       计算引用过。将其应用于"正在被 decode 引用的 KV block"是语义误用。

    e) vLLM 版本升级风险高：null_block 的行为未被公开文档化，任何 minor
       version 变化都可能使该 hack 失效。

    结论：null-block 方案虽然比原始 _free_tail_blocks() "不那么危险"，
    但仍属于插件层 unsafe hack，必须与原方案一起放弃。

================================================================================
§2  方案对比与抉择
================================================================================

    ┌─────────────────────┬──────────────────┬──────────────────┬──────────────────┐
    │ 维度                │ 原始方案          │ Null-block 方案   │ Reviewer 双轨方案 │
    │                     │ (已废弃)          │ (本 agent 提出)   │ (采纳)           │
    ├─────────────────────┼──────────────────┼──────────────────┼──────────────────┤
    │ 操作层              │ 插件层            │ 插件层            │ 主线:内核扩展     │
    │                     │                  │                  │ 保底:原生语义     │
    ├─────────────────────┼──────────────────┼──────────────────┼──────────────────┤
    │ coordinator 一致性  │ ❌ 直接破坏       │ ⚠️ 表面一致       │ ✅ 内核保证       │
    │                     │                  │  实际未更新       │                  │
    ├─────────────────────┼──────────────────┼──────────────────┼──────────────────┤
    │ block 真实释放      │ ✅ 但不安全       │ ✅ + null 占位    │ ✅ 原子操作       │
    ├─────────────────────┼──────────────────┼──────────────────┼──────────────────┤
    │ 请求继续 decode     │ ✅               │ ✅               │ 主线:✅           │
    │                     │                  │                  │ 保底:需 recompute │
    ├─────────────────────┼──────────────────┼──────────────────┼──────────────────┤
    │ KV 数据正确性       │ ❌ stale reuse   │ ❌ null data      │ ✅ 一致           │
    ├─────────────────────┼──────────────────┼──────────────────┼──────────────────┤
    │ CUDA safety         │ ❌ crash ≈180s   │ ⚠️ 可能不 crash   │ ✅ 安全           │
    │                     │                  │  但数据错误       │                  │
    ├─────────────────────┼──────────────────┼──────────────────┼──────────────────┤
    │ vLLM 版本耦合度     │ 高（内部结构）    │ 高（null_block）  │ 低（正式 API）    │
    ├─────────────────────┼──────────────────┼──────────────────┼──────────────────┤
    │ 论文可信度          │ ❌ unsafe hack   │ ⚠️ 勉强可辩护     │ ✅ 正规扩展       │
    ├─────────────────────┼──────────────────┼──────────────────┼──────────────────┤
    │ 实验可稳定复现      │ ❌ 随机 crash    │ ⚠️ 结果不可信     │ ✅               │
    └─────────────────────┴──────────────────┴──────────────────┴──────────────────┘

    最终决策：完全采纳 reviewer 的双轨方案。

    1. 放弃原始 _free_tail_blocks() 实现
    2. 放弃 null-block 替换方案
    3. 主线：最小 vLLM 内核扩展（tail truncation API）
    4. 保底：recompute fallback（使用 vLLM 原生 free + recompute 语义）
    5. 实验优先级：Mode A（recompute fallback）先行，Mode B（tail truncation）增强

================================================================================
§3  推荐路线（双轨方案）
================================================================================

3.1 主线：最小 vLLM 内核扩展 — tail truncation API

    目标：在 vLLM v1 的 KVCacheManager / coordinator 层新增一个最小原子操作，
    支持"释放请求尾部连续 N 个 block"，同时原子更新所有相关状态。

    扩展范围：
    - SingleTypeKVCacheManager: 新增 truncate_tail_blocks() 方法
    - KVCacheCoordinator: 新增 truncate_tail() 转发方法
    - KVCacheManager: 新增 truncate_request_tail() 公开 API

    改动量估算：约 60-80 行新增代码（3 个方法 + TruncateResult dataclass）。
    不修改任何现有方法的行为，不改变任何现有 API 的签名。

    详见 §5 完整 API 设计。

3.2 保底线：recompute fallback

    目标：利用 vLLM 原生的 request-level free + recompute 机制，实现
    "BidKV 决策 + vLLM 执行"的安全压缩路径。

    语义：
    - BidKV 负责：选择哪个请求需要"压缩"（selection / ranking / decision）
    - vLLM 负责：将被选中的请求从 running 移到 waiting 队列（preempt）
    - 请求后续通过 vLLM 原生 recompute 路径重新进入 running
    - 结果：请求的全部 KV 被释放，后续 decode 需要重算 prefill

    优势：
    - 100% 使用 vLLM 原生公开 API，零内部状态操作
    - 无 crash 风险，可立即跑通全部 216 runs
    - 论文可表述为 "bid-guided safe release / recompute semantics"

    详见 §6 完整执行器设计。

================================================================================
§4  禁止路线
================================================================================

以下路线明确禁止，不得作为主线或备选方案继续推进：

    ┌─────────────────────────────────────────────────────────────────┐
    │ 4.1 插件层 _free_tail_blocks() 直接调用                        │
    │     block_pool.free_blocks() / del group_blocks[...]           │
    │     理由：绕过 coordinator，导致 CUDA memory corruption         │
    │     状态：代码保留但标记 @deprecated，不得在任何 mode 中调用     │
    ├─────────────────────────────────────────────────────────────────┤
    │ 4.2 Null-block 替换方案                                        │
    │     理由：仍然绕过 coordinator，null_block 语义误用，            │
    │     KV 数据不正确，version fragile                              │
    │     状态：不实现                                                │
    ├─────────────────────────────────────────────────────────────────┤
    │ 4.3 未经源码级审计使用 remove_skipped_blocks()                  │
    │     理由：该方法为 speculative decoding 设计，                   │
    │     对正在被 attention 引用的 block 使用是语义误用               │
    │     状态：不采用                                                │
    ├─────────────────────────────────────────────────────────────────┤
    │ 4.4 任何绕过 coordinator 的内部结构修改                         │
    │     包括但不限于：直接操作 req_to_blocks dict、                  │
    │     直接操作 block_pool free list、直接修改 block ref_cnt        │
    │     状态：绝对禁止                                              │
    ├─────────────────────────────────────────────────────────────────┤
    │ 4.5 在不改 vLLM 源码前提下追求：                                │
    │     真正释放部分 KV + 请求不中断 decode + 不重算                 │
    │     + 保持内部一致性                                            │
    │     理由：这组目标在当前 vLLM v1 公开 API 下不成立               │
    │     状态：明确标记为 infeasible                                  │
    └─────────────────────────────────────────────────────────────────┘

================================================================================
§5  最小 vLLM 内核扩展 API 设计
================================================================================

5.1 API 签名

    扩展点 1：SingleTypeKVCacheManager（最底层，实际执行者）
    ─────────────────────────────────────────────────────────────
    def truncate_tail_blocks(
        self,
        request_id: str,
        num_blocks_to_free: int,
    ) -> TruncateResult:
        """释放请求尾部连续 num_blocks_to_free 个 block。

        原子操作：更新 req_to_blocks + block refcount + free list。
        不允许释放后 req_to_blocks 为空（至少保留 1 个 block）。
        """

    扩展点 2：KVCacheCoordinator（中间层，转发 + 跨 group 协调）
    ─────────────────────────────────────────────────────────────
    def truncate_tail(
        self,
        request_id: str,
        num_blocks_to_free: int,
    ) -> TruncateResult:
        """对所有 kv_cache_group 执行 tail truncation。"""

    扩展点 3：KVCacheManager（公开层，BidKV 唯一应调用的接口）
    ─────────────────────────────────────────────────────────────
    def truncate_request_tail(
        self,
        request_id: str,
        num_blocks_to_free: int,
    ) -> TruncateResult:
        """公开 API：安全释放请求尾部 KV block。"""

5.2 TruncateResult 结构

    @dataclass(frozen=True)
    class TruncateResult:
        success: bool                   # 是否成功执行
        actual_freed_blocks: int        # 实际释放的 block 数
        actual_freed_tokens: int        # 实际释放的 token 数
        new_num_blocks: int             # truncation 后请求的 block 数
        new_computed_token_boundary: int # truncation 后的有效 token 边界
        fallback_required: bool         # 是否需要 fallback 到 recompute
        reason: str                     # 失败原因或操作说明

5.3 不变量（Invariants）

    所有不变量必须在 truncate 前后同时成立：

    INV-1  req_to_blocks[request_id] 长度 = 请求实际占有的 block 数
           且与 coordinator 的视图一致。

    INV-2  每个 block 的 ref_cnt 与"依赖该 block 的 request 集合"一致。
           truncation 释放的 block，其 ref_cnt 必须递减（若降到 0 则入 free list）。

    INV-3  block_pool.free_blocks 仅在 ref_cnt = 0 时将 block 归还 free list。

    INV-4  truncation 后 len(req_to_blocks[req_id]) >= 1。
           不允许将请求的 block 全部释放（那是 free() 的语义）。

    INV-5  truncation 后 new_computed_token_boundary =
           new_num_blocks * block_size（向下对齐）。
           调用方（BidKV）负责同步更新 Request.num_computed_tokens。

    INV-6  仅允许释放尾部连续 block。不支持中间块裁剪。
           理由：中间块裁剪会破坏 attention 的因果性（causal mask）。

    INV-7  如果被请求的 block 涉及 prefix cache（共享 block），
           则该 block 不得释放（ref_cnt > 1）。
           此时返回 actual_freed_blocks < num_blocks_to_free。

5.4 安全条件 / 前置检查

    - request_id 必须存在于 req_to_blocks 中
    - num_blocks_to_free > 0
    - num_blocks_to_free < len(req_to_blocks[request_id])（不允许全部释放）
    - 不在 allocate_slots() 执行期间调用（单线程模型，天然满足）

5.5 Fallback 触发条件

    以下情况 truncate_request_tail() 返回 fallback_required=True：

    a) request_id 不存在于 coordinator
    b) 请求只有 1 个 block（无法再 truncate）
    c) 尾部 block 涉及共享 prefix cache（ref_cnt > 1），无法安全
       释放任何 block
    d) 内部异常（defensive，理论上不应发生）

    BidKV adapter 收到 fallback_required=True 后，应调用 recompute
    fallback 路径（§6）。

5.6 SingleTypeKVCacheManager.truncate_tail_blocks() 核心伪代码

    def truncate_tail_blocks(self, request_id, num_blocks_to_free):
        blocks = self.req_to_blocks.get(request_id)
        if blocks is None:
            return TruncateResult(success=False, ..., fallback_required=True,
                                  reason="request not found")
        if len(blocks) <= 1:
            return TruncateResult(success=False, ..., fallback_required=True,
                                  reason="only 1 block, cannot truncate")

        # 限制：最多释放到只剩 1 个 block
        max_freeable = len(blocks) - 1
        actual_free = min(num_blocks_to_free, max_freeable)

        # 取尾部 block
        tail = blocks[-actual_free:]

        # 检查 prefix cache 安全性（ref_cnt > 1 的 block 不释放）
        safe_tail = []
        for blk in reversed(tail):
            if blk.ref_cnt > 1:
                break  # 遇到共享 block，停止
            safe_tail.append(blk)
        safe_tail.reverse()

        if not safe_tail:
            return TruncateResult(success=False, ..., fallback_required=True,
                                  reason="tail blocks shared by prefix cache")

        # 原子操作：从 req_to_blocks 中移除 + 释放到 block_pool
        del blocks[-len(safe_tail):]
        self.block_pool.free_blocks(safe_tail)

        new_num_blocks = len(blocks)
        freed_tokens = len(safe_tail) * self.block_size

        return TruncateResult(
            success=True,
            actual_freed_blocks=len(safe_tail),
            actual_freed_tokens=freed_tokens,
            new_num_blocks=new_num_blocks,
            new_computed_token_boundary=new_num_blocks * self.block_size,
            fallback_required=False,
            reason="tail truncated",
        )

5.7 关键区别：为什么内核扩展安全而插件层 hack 不安全

    在内核扩展中，同一个方法同时执行：
    a) del blocks[-n:]          — 更新 req_to_blocks 映射
    b) block_pool.free_blocks() — 更新 block_pool free list + ref_cnt

    这两步在同一个方法中原子完成（vLLM v1 是单线程 EngineCore），
    因此不存在中间不一致状态。

    在插件层 hack 中，这两步分别由不同代码执行，且插件无法确保
    coordinator 的其他内部状态（如 unhashed block tracking）同步更新。

================================================================================
§6  Recompute Fallback 执行器设计
================================================================================

6.1 概述

    Recompute fallback 是 BidKV 在 vLLM 上的安全保底执行路径。
    BidKV 仍然负责"选谁"（selection），但执行方式改为 vLLM 原生的
    request-level preempt + recompute。

6.2 执行流程

    ┌──────────────┐     ┌───────────────────┐     ┌──────────────┐
    │  BidKV 决策  │     │  vLLM Scheduler    │     │  vLLM Engine │
    │  select victim│────▶│  preempt(request)  │────▶│  free all KV │
    │  + bid ranking│     │  move → waiting    │     │  recompute   │
    └──────────────┘     └───────────────────┘     └──────────────┘

    步骤：
    1. BidKV PressureDetector 检测到 KV pressure
    2. BidKV Solver 选出 victim request（s）
    3. BidKV adapter 调用 recompute_fallback_executor:
       a) 将 victim request 标记为 preempted
       b) 调用 scheduler 的原生 preempt 路径（将 request 从 running
          移到 waiting 队列）
       c) vLLM 的 kv_cache_manager.free(request) 释放全部 KV
       d) request 后续被 scheduler 重新调度时，从头 recompute prefill
    4. 返回 FallbackResult

6.3 FallbackResult 结构

    @dataclass(frozen=True)
    class FallbackResult:
        request_id: str
        actual_freed_tokens: int        # 所有 KV token 被释放
        execution_mode: str             # 固定为 "recompute_fallback"
        preempt_success: bool           # 是否成功 preempt
        reason: str

6.4 与 vLLM 原生 preempt 的区别

    vLLM 原生 preempt：
    - 触发条件：allocate_slots() 失败
    - 选择策略：lowest priority (FIFO/user-specified)
    - BidKV 无参与

    BidKV recompute fallback：
    - 触发条件：BidKV PressureDetector 提前检测到压力
    - 选择策略：BidKV bid-based ranking（quality-aware）
    - 执行：复用 vLLM 原生 preempt 机制
    - 核心价值：victim 选择更智能（考虑压缩 budget / quality impact）

    论文角度：BidKV 的贡献在于 selection intelligence，而非 execution mechanics。
    即使 execution 是 recompute，selection 质量仍然可以产生显著差异。

6.5 Adapter 层接口

    class VLLMAdapter:
        def execute_compression(self, request_id: str, target_tokens: int) -> int:
            """执行压缩。根据当前 mode 选择执行路径。

            Mode A: recompute fallback → preempt + full free
            Mode B: tail truncation → partial KV release
            """
            if self._config.execution_mode == "recompute_fallback":
                return self._execute_recompute_fallback(request_id)
            elif self._config.execution_mode == "tail_truncation":
                result = self._execute_tail_truncation(request_id, target_tokens)
                if result.fallback_required:
                    return self._execute_recompute_fallback(request_id)
                return result.actual_freed_tokens
            else:
                raise ValueError(f"Unknown execution mode: {self._config.execution_mode}")

================================================================================
§7  实验执行策略（Mode A / Mode B）
================================================================================

7.1 Mode A — Native Safe Fallback Mode（当前最高优先级）

    ┌─────────────────────────────────────────────────────────────────┐
    │ 优先级：P0                                                      │
    │ 状态  ：立即可实现，不依赖任何 vLLM 源码修改                      │
    │ 目标  ：跑通全部 8 × 3 × 3 × 3 = 216 runs，零 crash             │
    └─────────────────────────────────────────────────────────────────┘

    执行语义：
    - BidKV 负责：pressure detection → bid collection → solver ranking
    - vLLM 负责：request-level preempt + full KV free + recompute
    - 请求被 preempt 后全部 KV 释放，重新进入 waiting 队列

    指标含义调整：
    - "compression" 在 Mode A 中等价于 "bid-guided preemption"
    - tokens_freed = 被选中请求的全部 KV token（而非 partial）
    - 质量影响：重算延迟（TTFT regression），但输出质量不受影响
      （因为 KV 是完整重算的，无信息损失）

    论文表述：
    - "Bid-guided safe release with recompute semantics"
    - "BidKV selection layer + framework-native safe execution"
    - 优势：selection 智能性仍可对比（BidKV vs random vs h2o vs uniform）
    - 不足：无法展示 partial eviction 的效率优势

    实验矩阵：完整 216 runs，与 Issue-047 原始规划一致。
    区别：所有 injected 策略使用 recompute fallback 执行。

7.2 Mode B — Tail Truncation Mode（增强项）

    ┌─────────────────────────────────────────────────────────────────┐
    │ 优先级：P1（Mode A 稳定跑通后再推进）                            │
    │ 依赖  ：vLLM 最小内核扩展已实现 + 通过单元测试                    │
    │ 目标  ：在部分 workload 上展示 partial eviction 的效率优势        │
    └─────────────────────────────────────────────────────────────────┘

    执行语义：
    - BidKV 负责：pressure detection → bid collection → solver ranking
    - vLLM 扩展负责：truncate_request_tail() 安全释放尾部 block
    - 请求在满足安全条件时继续 decode（无需 recompute）
    - 安全条件不满足时自动 fallback 到 Mode A

    指标含义：
    - tokens_freed = 实际被 truncate 的 KV token（partial）
    - 质量影响：尾部 KV 丢失可能导致输出质量下降（ROUGE/EM 退化）
      这正是 BidKV quality-aware bidding 的价值所在

    论文表述：
    - "Minimal kernel extension for safe partial KV truncation"
    - "BidKV full pipeline: quality-aware bidding + safe partial eviction"
    - 优势：展示 partial eviction 的真正效率优势
    - 说明：需明确标注使用了最小内核扩展，非纯插件能力

    实验矩阵：在 Mode A 数据基础上，对核心策略子集重新运行。

7.3 双 Mode 对比价值

    Mode A vs Mode B 的对比本身是一个有价值的实验贡献：

    a) Mode A 展示：BidKV 的 selection intelligence
       → "选谁压缩"比"随机选"好多少？
       → 即使执行方式相同（全部 recompute），选择质量仍有差异

    b) Mode B 展示：BidKV 的 execution efficiency
       → "部分释放"比"全部释放 + recompute"好多少？
       → partial eviction 的效率优势（更少 recompute 开销）

    c) Mode A → Mode B 的增量：
       → 论文可以展示从 safe fallback 到 efficient truncation 的渐进收益
       → 这是一个非常自然的 论文叙事结构

7.4 Mode 切换机制

    在 BidKVConfig 中新增 execution_mode 字段：

    @dataclass
    class BidKVConfig:
        ...
        execution_mode: str = "recompute_fallback"
        # 可选值: "recompute_fallback" | "tail_truncation"

    实验脚本通过环境变量或配置文件切换：
        BIDKV_EXECUTION_MODE=recompute_fallback  # Mode A（默认）
        BIDKV_EXECUTION_MODE=tail_truncation      # Mode B

================================================================================
§8  论文叙事建议
================================================================================

8.1 若只完成 Mode A（recompute fallback）

    §1 Introduction:
        BidKV introduces a market-inspired quality-aware compression scheduling
        framework for KV cache management in LLM serving systems.

    §4 System Design:
        BidKV separates compression decision (selection) from execution:
        - Selection layer: bid-based quality-aware ranking
        - Execution layer: framework-native safe release semantics

        In vLLM, compression execution is realized through framework-native
        request preemption and recompute. BidKV's quality-aware bid ranking
        determines which requests to preempt, while vLLM manages the safe
        lifecycle transition (free → recompute → resume).

    §6 Evaluation:
        - Table 1 仍然有效：8 strategies 的 SLO violation / P99 TTFT / throughput 对比
        - Figure 3a/b：质量指标方面，Mode A 中输出质量不受影响（因为是完整 recompute），
          因此质量图需要调整为"recompute latency overhead"而非"quality degradation"
        - Figure 4：BidKV selection 的效率差距仍可衡量
        - 核心论点转向：selection intelligence 本身的价值

    §7 Discussion:
        Partial KV eviction without recompute requires kernel-level support
        from the serving framework. We demonstrate that even with
        recompute-based execution, quality-aware bid-driven selection
        significantly outperforms system-inferred heuristics.

    不可声称的：
        ❌ "true uninterrupted partial KV eviction"
        ❌ "zero recompute overhead"
        ❌ "pure plugin-layer compression"

8.2 若完成 Mode B（tail truncation）

    §4 System Design 可升级为：
        BidKV provides two execution modes:
        (1) Safe fallback: framework-native preempt + recompute
        (2) Efficient mode: minimal kernel extension for safe partial
            KV tail truncation (~60 LOC addition to vLLM)

        The kernel extension provides a truncate_request_tail() API
        that atomically manages block ownership, coordinator state,
        and block pool free lists, enabling safe partial KV release
        without recomputing the request from scratch.

    §6 Evaluation 可新增：
        - Mode A vs Mode B 的直接对比
        - Partial eviction 的效率增益（更少 recompute 开销 → 更高 throughput）
        - Quality degradation 分析（Mode B 的 ROUGE/EM 下降量 = BidKV 的 quality budget 管理能力）

    §7 Discussion 可升级为：
        Our minimal kernel extension demonstrates that safe partial KV
        truncation is achievable with approximately 60 lines of code
        addition to vLLM's KV cache manager. This extension maintains
        all coordinator invariants and provides a foundation for
        future framework-level support of partial KV lifecycle management.

8.3 为什么"修改 vLLM 源码"对系统论文不是坏事

    a) 系统论文常见的贡献模式：
       - 识别现有系统的 abstraction gap
       - 提出最小扩展来弥补该 gap
       - 在扩展后的系统上验证新设计
       典型例子：vLLM 自身就是对 ORCA 的扩展；PagedAttention 是对 GPU memory
       管理的扩展。

    b) BidKV 的扩展具有通用价值：
       - tail truncation 不仅服务于 BidKV
       - 任何需要 partial KV release 的研究（如 context compression、
         attention sparsification、memory-efficient long-context serving）
         都能受益于这个 API
       - 这本身可以作为论文的一个 contribution

    c) 比 unsafe hack 更可信：
       - reviewer 更愿意看到"我们扩展了 60 行代码来支持新语义"
       - 而非"我们用 monkey-patch 绕过了框架的内部状态管理"
       - 前者展示了对系统的理解，后者暴露了对系统的误用

    d) 复现性更好：
       - 60 行 patch 可以作为 supplementary material 提供
       - 其他研究者可以在 1 分钟内 apply patch 并复现实验

================================================================================
§9  实施排期与优先级
================================================================================

9.1 Phase C 修订后的执行顺序

    ┌─────┬────────────────────────────────────────┬───────┬──────────┐
    │ 步骤 │ 任务                                   │ 依赖  │ 优先级   │
    ├─────┼────────────────────────────────────────┼───────┼──────────┤
    │ C.1 │ 废弃 _free_tail_blocks() unsafe 实现    │ 无    │ P0       │
    │     │ 标记 @deprecated，替换为 mode dispatcher │       │          │
    ├─────┼────────────────────────────────────────┼───────┼──────────┤
    │ C.2 │ 实现 recompute fallback executor        │ C.1   │ P0       │
    │     │ FallbackResult + 与 scheduler preempt    │       │          │
    │     │ 的集成                                   │       │          │
    ├─────┼────────────────────────────────────────┼───────┼──────────┤
    │ C.3 │ BidKVConfig 新增 execution_mode 字段    │ C.2   │ P0       │
    │     │ 默认 "recompute_fallback"               │       │          │
    ├─────┼────────────────────────────────────────┼───────┼──────────┤
    │ C.4 │ Mode A 全量实验：216 runs               │ C.3   │ P0       │
    │     │ 确认 0 crash + 生成 Table 1 数据        │       │          │
    ├─────┼────────────────────────────────────────┼───────┼──────────┤
    │ C.5 │ 实现 vLLM tail truncation 内核扩展      │ C.1   │ P1       │
    │     │ TruncateResult + 3 层 API              │       │（可并行）│
    ├─────┼────────────────────────────────────────┼───────┼──────────┤
    │ C.6 │ tail truncation 单元测试 + 集成测试     │ C.5   │ P1       │
    ├─────┼────────────────────────────────────────┼───────┼──────────┤
    │ C.7 │ Mode B 对比实验（核心策略子集）          │ C.6   │ P1       │
    │     │ Mode A vs Mode B 效率对比               │       │          │
    ├─────┼────────────────────────────────────────┼───────┼──────────┤
    │ C.8 │ #048 SGLang 实验（独立推进）            │ 无    │ P0       │
    │     │ SGLang adapter 使用 recompute fallback  │       │（可并行）│
    └─────┴────────────────────────────────────────┴───────┴──────────┘

9.2 关键约束

    - C.1–C.4 为 critical path，任何延期直接影响论文数据交付
    - C.5–C.7 为增强路径，不阻塞 C.4 的数据交付
    - C.8 可与 C.1–C.4 并行推进
    - Mode B 若在 Gate-B deadline (2026-03-31) 前未稳定，
      论文直接使用 Mode A 数据

================================================================================
§10  风险评估
================================================================================

    ┌──────────────────────┬────────────┬────────────────────────────────┐
    │ 风险                 │ 概率       │ 缓解措施                       │
    ├──────────────────────┼────────────┼────────────────────────────────┤
    │ Mode A recompute     │ 中         │ 这是预期行为，Mode A 的价值在   │
    │ 开销过大，SLO        │            │ selection intelligence，而非   │
    │ violation 过高       │            │ execution efficiency。若过高，  │
    │                      │            │ 可调整 pressure threshold。     │
    ├──────────────────────┼────────────┼────────────────────────────────┤
    │ Mode A 中 BidKV 与   │ 低         │ 即使在 recompute 语义下，        │
    │ baseline 差异不显著  │            │ quality-aware selection 仍应    │
    │                      │            │ 优于 random/uniform。           │
    │                      │            │ 若确实不显著 → Scenario B pivot │
    ├──────────────────────┼────────────┼────────────────────────────────┤
    │ vLLM tail truncation │ 中-高      │ Mode A 已提供完整数据保底。      │
    │ 扩展在 Gate-B 前     │            │ Mode B 降级为论文附录/未来工作。 │
    │ 未稳定               │            │                                │
    ├──────────────────────┼────────────┼────────────────────────────────┤
    │ SGLang adapter 也    │ 低         │ SGLang 的 scheduler 架构更开放，│
    │ 需要类似修改         │            │ 但若需要，同样采用 recompute    │
    │                      │            │ fallback 优先策略。             │
    ├──────────────────────┼────────────┼────────────────────────────────┤
    │ Recompute fallback   │ 低         │ 直接使用 vLLM 自身 preempt     │
    │ 实现困难             │            │ 机制，BidKV 只需触发信号。      │
    │                      │            │ 几乎等同于"BidKV 选择 victim   │
    │                      │            │ 后让 vLLM 自己 preempt"。      │
    └──────────────────────┴────────────┴────────────────────────────────┘

================================================================================
END OF DOCUMENT — 待 reviewer 审核冻结
================================================================================
