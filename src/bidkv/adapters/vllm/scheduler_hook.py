"""Scheduler Hook — vLLM Scheduler monkey-patch 注入。

将 BidKV 智能 victim 选择注入到 vLLM v1 Scheduler 的调度路径中。

注入点（v2 — victim reorder + proactive preemption）：
- ``schedule()``：
  1. 重排 running 列表（victim reorder）— 当 vLLM 需要 preempt 时，
     选择 BidKV 评分最低的请求。
  2. 在 schedule() 之前检查 KV 利用率，当利用率超过阈值且有 waiting
     请求时，主动 preempt 最低质量请求，防止级联 preemption。
     有严格 cooldown 防止 thrashing。
- ``update_from_output()``：decode step 完成后，更新 token tracking。
- ``_free_request()``：请求完成时，清理 BidKV 内部状态。
- ``update_from_output()``：decode step 完成后，更新 token tracking。
- ``_free_request()``：请求完成时，清理 BidKV 内部状态。

设计原则：
- **质量感知**：BidKV scoring 选择最可牺牲的请求（recompute cost × anti-starvation）
- **主动压缩**：在 KV 高压时 proactively preempt，减少级联 preemption
- **Feature OFF 零开销**：BidKV 未激活时，monkey-patch 方法直接调用原始方法
- **可逆**：``uninstall_scheduler_hook()`` 可恢复原始方法
"""

from __future__ import annotations

import functools
import logging
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from bidkv.adapters.vllm.adapter import VLLMAdapter

logger = logging.getLogger(__name__)

# 用于存储原始方法的属性名前缀
_ORIG_PREFIX = "_bidkv_orig_"


def install_scheduler_hook(scheduler: Any, adapter: VLLMAdapter) -> None:
    """将 BidKV 压缩逻辑注入到 vLLM Scheduler。

    Monkey-patch 三个方法：
    1. ``schedule()`` — preemption 前压缩
    2. ``update_from_output()`` — decode step 后 H2O 更新
    3. ``_free_request()`` — 请求完成时 cleanup

    Parameters
    ----------
    scheduler:
        vLLM v1 Scheduler 实例。
    adapter:
        VLLMAdapter 实例。
    """
    # Install KV truncation support (token-level block release)
    kv_cache_manager = getattr(scheduler, "kv_cache_manager", None)
    if kv_cache_manager is not None:
        from bidkv.adapters.vllm.truncation_hook import install_truncation_support

        install_truncation_support(kv_cache_manager)

    # 保存原始方法
    setattr(scheduler, f"{_ORIG_PREFIX}schedule", scheduler.schedule)
    setattr(scheduler, f"{_ORIG_PREFIX}update_from_output", scheduler.update_from_output)
    if hasattr(scheduler, "_free_request"):
        setattr(scheduler, f"{_ORIG_PREFIX}_free_request", scheduler._free_request)

    # Patch schedule()
    scheduler.schedule = functools.partial(_patched_schedule, scheduler, adapter)

    # Patch update_from_output()
    scheduler.update_from_output = functools.partial(
        _patched_update_from_output, scheduler, adapter
    )

    # Patch _free_request()
    if hasattr(scheduler, f"{_ORIG_PREFIX}_free_request"):
        scheduler._free_request = functools.partial(_patched_free_request, scheduler, adapter)

    # 保存 adapter 引用
    scheduler._bidkv_adapter = adapter

    logger.info("BidKV scheduler hooks installed on vLLM Scheduler")


def uninstall_scheduler_hook(scheduler: Any, adapter: VLLMAdapter) -> None:  # noqa: ARG001
    """移除 BidKV 注入，恢复 vLLM 原始行为。

    Parameters
    ----------
    scheduler:
        vLLM v1 Scheduler 实例。
    adapter:
        VLLMAdapter 实例。
    """
    # 恢复原始方法
    orig_schedule = getattr(scheduler, f"{_ORIG_PREFIX}schedule", None)
    if orig_schedule is not None:
        scheduler.schedule = orig_schedule

    orig_update = getattr(scheduler, f"{_ORIG_PREFIX}update_from_output", None)
    if orig_update is not None:
        scheduler.update_from_output = orig_update

    orig_free = getattr(scheduler, f"{_ORIG_PREFIX}_free_request", None)
    if orig_free is not None:
        scheduler._free_request = orig_free

    # 清理属性
    for attr in list(vars(scheduler)):
        if attr.startswith(_ORIG_PREFIX) or attr == "_bidkv_adapter":
            delattr(scheduler, attr)

    logger.info("BidKV scheduler hooks removed from vLLM Scheduler")


_SCHEDULE_CALL_COUNT = 0
_DIAG_LOG = "/tmp/bidkv_diag.log"
_METRICS_FILE = "/tmp/bidkv_metrics_latest.json"


def _diag(msg: str) -> None:
    """Write diagnostic message to a file (works in subprocesses)."""
    import os

    with open(_DIAG_LOG, "a") as f:
        f.write(f"[{os.getpid()}] {msg}\n")


def _dump_metrics(adapter: VLLMAdapter) -> None:
    """Atomically dump adapter metrics to a well-known JSON file."""
    import json
    import os

    metrics = adapter.metrics.as_dict()
    # Tag with strategy + PID so runner can validate provenance
    metrics["_strategy"] = os.environ.get("BIDKV_STRATEGY", "")
    metrics["_pid"] = os.getpid()
    tmp = _METRICS_FILE + f".{os.getpid()}.tmp"
    with open(tmp, "w") as f:
        json.dump(metrics, f)
    os.replace(tmp, _METRICS_FILE)


def _compute_keep_score(req: Any) -> float:
    """Compute keep score for a request: higher = more valuable to keep running.

    Scoring considers three factors:
    1. **Completion progress**: requests close to finishing will free KV soon
       naturally. Preempting them wastes invested compute for minimal gain.
    2. **Recompute cost vs remaining work**: ratio of tokens already computed
       to tokens still needed. High ratio = expensive to redo, little benefit.
    3. **Anti-starvation**: previously preempted requests get a strong keep
       bonus to prevent wasteful repeated preemption cycles.
    """
    num_computed = getattr(req, "num_computed_tokens", 0)
    num_prompt = getattr(req, "num_prompt_tokens", 0)
    num_preemptions = getattr(req, "num_preemptions", 0)

    # Get max output tokens from sampling params
    max_tokens = 256  # conservative fallback
    sp = getattr(req, "sampling_params", None)
    if sp is not None:
        mt = getattr(sp, "max_tokens", None)
        if mt is not None and mt > 0:
            max_tokens = mt

    # Output progress
    num_output = max(0, num_computed - num_prompt)
    remaining_output = max(1, max_tokens - num_output)

    # Completion ratio: 0.0 (just started) → 1.0 (done)
    completion = min(1.0, num_output / max_tokens) if max_tokens > 0 else 0.0

    # Efficiency ratio: how expensive is preemption relative to remaining work?
    # High = costly to preempt per unit of remaining work → KEEP
    # Low = cheap to preempt per unit of remaining work → expendable
    efficiency = (num_computed + 1) / remaining_output

    # Anti-starvation: each prior preemption strongly increases keep score
    starvation_mult = 1.0 + num_preemptions * 2.0

    # Combined score: completion amplifies efficiency + starvation
    # Requests close to finishing get massive boost (completion^2 for emphasis)
    keep_score = (0.1 + completion * completion) * efficiency * starvation_mult

    return keep_score


def _build_running_candidates(running: Any, adapter: VLLMAdapter) -> list[tuple[Any, Any]]:
    """Build (RequestState, vllm_request) pairs from scheduler.running.

    Returns a list so callers can map strategy decisions back to vLLM objects.
    """
    from bidkv.baselines.base import RequestState

    pairs: list[tuple[Any, Any]] = []
    for req in running:
        rid = getattr(req, "request_id", None)
        if rid is None:
            continue
        token_ids = adapter._request_tokens.get(rid)
        pairs.append(
            (
                RequestState(
                    request_id=rid,
                    current_tokens=len(token_ids) if token_ids else 0,
                    token_ids=tuple(token_ids) if token_ids else (),
                ),
                req,
            )
        )
    return pairs


def _reorder_running_for_preemption(scheduler: Any, adapter: VLLMAdapter) -> None:  # noqa: ARG001
    """Reorder scheduler.running so the strategy-preferred victim is at the end.

    For FCFS policy, vLLM preempts with self.running.pop() — the last element.
    Routes victim selection through the experiment strategy when available.
    """
    running = getattr(scheduler, "running", None)
    if running is None or len(running) <= 1:
        return

    # Only reorder when preemption is imminent (waiting requests exist).
    waiting = getattr(scheduler, "waiting", None)
    if not waiting:
        return

    strategy = adapter._experiment_strategy
    strategy_name = adapter._experiment_strategy_name

    if strategy is not None and strategy_name != "bidkv":
        # Route through baseline strategy's select_victims()
        pairs = _build_running_candidates(running, adapter)
        candidates = [p[0] for p in pairs]
        if candidates:
            actions = strategy.select_victims(candidates, 1)
            if actions:
                victim_rid = actions[0].request_id
                # Move victim to end (popped first by vLLM FCFS)
                for i, req in enumerate(running):
                    if getattr(req, "request_id", None) == victim_rid:
                        victim_req = running.pop(i)
                        running.append(victim_req)
                        return

    # BidKV / default: use completion-aware keep_score
    scored: list[tuple[float, int, Any]] = []
    for idx, req in enumerate(running):
        keep_score = _compute_keep_score(req)
        scored.append((keep_score, idx, req))

    # Sort: highest keep_score first, lowest last (preemption target)
    scored.sort(key=lambda x: -x[0])

    running.clear()
    for _, _, req in scored:
        running.append(req)


def _proactive_preempt(scheduler: Any, adapter: VLLMAdapter) -> None:
    """Proactively preempt+truncate lowest-quality request under KV pressure.

    Fires BEFORE vLLM's native schedule(), so freed blocks become available
    for waiting requests without triggering vLLM's own cascade preemption.

    Conservative guards to avoid creating excessive recompute overhead:
    - KV block utilization must exceed 88% (moderate headroom before cascade)
    - Waiting queue must be non-empty (there's demand for blocks)
    - At least 3 running requests (keep at least 2 running after preemption)
    - Cooldown of 3 seconds between proactive preemptions

    Uses pure recompute (no output truncation) to avoid TPOT degradation.
    The benefit comes from quality-aware victim selection + earlier intervention
    that prevents costly cascade preemptions.
    """
    import time

    cooldown_s = 3.0
    now = time.monotonic()
    last_proactive = getattr(scheduler, "_bidkv_last_proactive", 0.0)
    if now - last_proactive < cooldown_s:
        return

    # Must have waiting requests (demand for blocks)
    waiting = getattr(scheduler, "waiting", None)
    if not waiting:
        return

    # Must have >=3 running requests (keep at least 2 after preemption)
    running = getattr(scheduler, "running", None)
    if running is None or len(running) < 3:
        return

    # Check KV utilization — 88% threshold (moderate headroom before cascade)
    kv_mgr = getattr(scheduler, "kv_cache_manager", None)
    if kv_mgr is None:
        return
    block_pool = getattr(kv_mgr, "block_pool", None)
    if block_pool is None:
        return
    usage = block_pool.get_usage()
    if usage < 0.88:
        return

    # --- Compute how many tokens to free (target: bring usage down to 80%) ---
    used, total = adapter.get_kv_stats()
    target_usage = int(total * 0.80)
    needed_tokens = max(1, used - target_usage)

    # --- Strategy-aware victim selection ---
    strategy = adapter._experiment_strategy
    strategy_name = adapter._experiment_strategy_name

    victim_id: str | None = None
    diag_detail = ""

    if strategy is not None:
        # Route ALL strategies (including BidKV) through select_victims()
        # for clean ablation: each strategy is evaluated by its own logic.
        pairs = _build_running_candidates(running, adapter)
        candidates = [p[0] for p in pairs]
        if candidates:
            actions = strategy.select_victims(candidates, needed_tokens)
            if actions:
                victim_id = actions[0].request_id
                diag_detail = f"strategy={strategy_name}"
    else:
        # No strategy set (standalone adapter): use completion-aware keep_score
        best_victim = None
        best_score = float("inf")
        for req in running:
            keep_score = _compute_keep_score(req)
            if keep_score < best_score:
                best_score = keep_score
                best_victim = req

        if best_victim is not None:
            victim_id = getattr(best_victim, "request_id", None)
            diag_detail = f"score={best_score:.2f}"

    if victim_id is None:
        return

    # Quality-aware victim selection + truncation.
    freed = adapter.execute_compression(victim_id, needed_tokens)
    if freed > 0:
        scheduler._bidkv_last_proactive = now
        _diag(f"proactive truncate: {victim_id} ({diag_detail}, usage={usage:.2f}, freed={freed})")


_MODEL_EXECUTOR_RESOLVED = False


def _resolve_model_executor(scheduler: Any, adapter: VLLMAdapter) -> None:
    """Lazily discover model_executor via gc.get_referrers.

    Plugin cannot patch EngineCore.__init__ effectively because
    load_general_plugins() runs INSIDE the already-executing __init__.
    Instead, on the first schedule() call (after __init__ completes),
    walk gc.get_referrers(scheduler) to find EngineCore and grab its
    model_executor for Mode B block-table sync.
    """
    global _MODEL_EXECUTOR_RESOLVED
    if _MODEL_EXECUTOR_RESOLVED:
        return
    _MODEL_EXECUTOR_RESOLVED = True

    # gc.get_referrers returns 0 in vLLM's spawned EngineCore subprocess.
    # Walk the call stack instead: schedule() is called by EngineCore.step()
    # or similar, whose `self` has model_executor.
    import sys

    frame = sys._getframe(1)  # caller of _patched_schedule
    while frame is not None:
        self_obj = frame.f_locals.get("self")
        if self_obj is not None:
            me = getattr(self_obj, "model_executor", None)
            if me is not None:
                adapter._model_executor = me
                _diag(
                    f"resolved model_executor via stack walk "
                    f"(frame={frame.f_code.co_name})"
                )
                return
        frame = frame.f_back
    _diag("WARN: could not resolve model_executor via stack walk")


def _patched_schedule(scheduler: Any, adapter: VLLMAdapter) -> Any:
    """Patched schedule() — 智能 victim 选择 + 主动压缩。

    v2 approach:
    1. Reorder scheduler.running (victim reorder) — when vLLM's native
       preemption fires (FCFS: self.running.pop()), it picks the request
       with the lowest BidKV quality score.
    2. Proactively preempt the lowest-quality request when KV utilization
       is high, freeing blocks BEFORE vLLM's native preemption cascades.
    """
    global _SCHEDULE_CALL_COUNT
    _SCHEDULE_CALL_COUNT += 1

    # Lazy-resolve model_executor on first call (after EngineCore.__init__)
    if adapter._model_executor is None:
        _resolve_model_executor(scheduler, adapter)
    if _SCHEDULE_CALL_COUNT <= 3 or _SCHEDULE_CALL_COUNT % 1000 == 0:
        used, total = adapter.get_kv_stats()
        tracked = len(adapter._request_tokens)
        running = len(getattr(scheduler, "running", []))
        pct = (used / total * 100) if total > 0 else 0.0
        _diag(
            f"schedule() #{_SCHEDULE_CALL_COUNT}: "
            f"kv={used}/{total} ({pct:.1f}%) "
            f"running={running} tracked={tracked}"
        )
        _dump_metrics(adapter)
    # Feature OFF 快速路径
    if not adapter.config.is_active:
        orig = getattr(scheduler, f"{_ORIG_PREFIX}schedule")
        return orig()

    # 同步 request tracking
    _sync_request_tracking(scheduler, adapter)

    # Proactively preempt lowest-quality request when KV pressure is high
    _proactive_preempt(scheduler, adapter)

    # Reorder running list: put most expendable request at the end
    # (FCFS preemption uses self.running.pop() — last element)
    _reorder_running_for_preemption(scheduler, adapter)

    # Let vLLM handle everything — including preemption with OUR ordering
    orig = getattr(scheduler, f"{_ORIG_PREFIX}schedule")
    result = orig()

    return result


def _patched_update_from_output(
    scheduler: Any,
    adapter: VLLMAdapter,
    scheduler_output: Any,
    model_runner_output: Any,
) -> Any:
    """Patched update_from_output() — decode step 后更新 H2O scoring。

    在调用原始 update_from_output 后，遍历 running requests，
    为每个请求更新 token tracking。
    """
    orig = getattr(scheduler, f"{_ORIG_PREFIX}update_from_output")
    result = orig(scheduler_output, model_runner_output)

    # Feature OFF 快速路径
    if not adapter.config.is_active:
        return result

    # 更新追踪信息：将新生成的 token 加入追踪
    _update_token_tracking_from_output(scheduler, adapter, model_runner_output)

    return result


def _patched_free_request(scheduler: Any, adapter: VLLMAdapter, request: Any, **kwargs: Any) -> Any:
    """Patched _free_request() — 请求完成时清理 BidKV 状态。"""
    # 先清理 BidKV 状态
    request_id = getattr(request, "request_id", None)
    if request_id is not None:
        adapter.on_request_complete(request_id)

    # 调用原始方法（透传所有额外参数，如 delay_free_blocks）
    orig = getattr(scheduler, f"{_ORIG_PREFIX}_free_request")
    return orig(request, **kwargs)


def _sync_request_tracking(scheduler: Any, adapter: VLLMAdapter) -> None:
    """同步 vLLM 的活跃请求到 adapter 的 tracking。

    遍历 scheduler.running，确保所有 running request 都被追踪。
    """
    running = getattr(scheduler, "running", [])
    tracked = set(adapter.get_tracked_requests())

    for request in running:
        req_id = getattr(request, "request_id", None)
        if req_id is None:
            continue
        if req_id not in tracked:
            # 新请求：从 request 中提取 token ids
            token_ids = _extract_token_ids(request)
            if token_ids:
                adapter.track_request(req_id, token_ids)


def _extract_token_ids(request: Any) -> list[int]:
    """从 vLLM Request 对象中提取 token IDs。

    vLLM v1 Request 有 ``prompt_token_ids`` 和 ``output_token_ids`` 属性。
    """
    prompt_ids = getattr(request, "prompt_token_ids", None)
    output_ids = getattr(request, "output_token_ids", None)

    token_ids: list[int] = []
    if prompt_ids is not None:
        token_ids.extend(prompt_ids)
    if output_ids is not None:
        token_ids.extend(output_ids)

    return token_ids


def _update_token_tracking_from_output(
    scheduler: Any,  # noqa: ARG001
    adapter: VLLMAdapter,
    model_runner_output: Any,
) -> None:
    """从 model_runner_output 中更新 token tracking。

    在每个 decode step 后，将新生成的 token 加入追踪。
    """
    sampled_token_ids = getattr(model_runner_output, "sampled_token_ids", None)
    if sampled_token_ids is None:
        return

    req_id_to_index = getattr(model_runner_output, "req_id_to_index", None)
    if req_id_to_index is None:
        return

    for req_id, req_index in req_id_to_index.items():
        tokens = adapter._request_tokens.get(req_id)
        if tokens is None:
            continue
        # 添加新 token
        if req_index < len(sampled_token_ids):
            new_token_ids = sampled_token_ids[req_index]
            if new_token_ids:
                tokens.extend(new_token_ids)
