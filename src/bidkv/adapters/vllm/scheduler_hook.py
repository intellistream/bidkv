"""Scheduler Hook — vLLM Scheduler monkey-patch 注入。

将 BidKV 智能 victim 选择注入到 vLLM v1 Scheduler 的调度路径中。

**Mode A 架构（v3 — pure native preemption）**：
策略控制 *WHO* gets preempted，执行机制是 vLLM native preempt+recompute。
不做 tail truncation / block-level 操作。

注入点：
- ``schedule()``：
  1. 定期刷新策略优先级缓存（每 3s，调用 strategy.select_victims()）
  2. 主动 preempt：KV > 阈值时，用 scheduler._preempt_request() 驱逐
     策略选中的 victim（单 victim + 5s cooldown 防 storm）
  3. 重排 running 列表：用缓存优先级影响 vLLM native preemption 的
     victim 选择（running.pop() FCFS）
- ``update_from_output()``：decode step 后更新 token tracking + H2O。
- ``_free_request()``：请求完成时清理 BidKV 状态。

设计原则：
- **纯 Mode A**：策略只做 `decision`（谁被 preempt），不做 `execution`
- **生效双路径**：proactive preempt（主动）+ reorder（影响 native preemption）
- **Feature OFF 零开销**：BidKV 未激活时直接调用原始方法
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
    # [DEPRECATED] Mode B: Install KV truncation support (token-level block release).
    # This installs truncate_request_tail() on KVCacheManager but the method is
    # never called in Mode A (request-level preempt+recompute). Retained for
    # potential Mode B extension (issue #054).
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

    # Close preemption logger if active (flushes and releases file handle)
    from bidkv.adapters.vllm import preemption_logger as _plogger

    _plogger.close_logger()

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
    import time

    from bidkv.baselines.base import RequestState

    now_ms = time.monotonic() * 1000
    # SLO 截止时间 = 到达时间 + 120 秒 (request_timeout_s 默认值)
    slo_timeout_ms = 120_000.0

    pairs: list[tuple[Any, Any]] = []
    for req in running:
        rid = getattr(req, "request_id", None)
        if rid is None:
            continue
        token_ids = adapter._request_tokens.get(rid)

        # 记录并获取到达时间
        if rid not in adapter._request_arrival_ms:
            adapter._request_arrival_ms[rid] = now_ms
        arrival_ms = adapter._request_arrival_ms[rid]

        # 从 vLLM request 提取 completion 信息
        num_prompt = getattr(req, "num_prompt_tokens", 0)
        num_computed = getattr(req, "num_computed_tokens", 0)
        num_preemptions = getattr(req, "num_preemptions", 0)
        max_output = 256  # conservative fallback
        sp = getattr(req, "sampling_params", None)
        if sp is not None:
            mt = getattr(sp, "max_tokens", None)
            if mt is not None and mt > 0:
                max_output = mt

        # Priority: 负到达时间 → 最新到达的请求优先级最低 → FCFS 驱逐
        priority = -arrival_ms

        pairs.append(
            (
                RequestState(
                    request_id=rid,
                    current_tokens=len(token_ids) if token_ids else 0,
                    token_ids=tuple(token_ids) if token_ids else (),
                    priority=priority,
                    arrival_time_ms=arrival_ms,
                    deadline_ms=arrival_ms + slo_timeout_ms,
                    num_prompt_tokens=num_prompt,
                    num_computed_tokens=num_computed,
                    max_output_tokens=max_output,
                    num_preemptions=num_preemptions,
                ),
                req,
            )
        )
    return pairs


def _reorder_waiting_for_admission(scheduler: Any, adapter: VLLMAdapter) -> None:
    """Reorder scheduler.waiting for optimal admission under KV pressure.

    vLLM processes waiting[0] first and BREAKS on first allocation failure.
    Each strategy represents a COMPLETE scheduling approach with its own
    admission policy. The differentiation is in the key function:

    Strategy grouping by admission key:
    ┌─────────────────┬──────────────────────────────────────────────┐
    │ Strategy        │ Admission policy                             │
    ├─────────────────┼──────────────────────────────────────────────┤
    │ preempt-evict     │ FCFS — no reorder (true vLLM default)       │
    │ preempt-evict-sjf │ SJF by prompt_tokens (LIFO eviction ablation)│
    │ static-random     │ SJF by prompt_tokens                        │
    │ largest-first     │ SJF by prompt_tokens                        │
    │ uniform           │ SJF by prompt_tokens                        │
    │ slack-aware       │ EDF by arrival_time (≈ FCFS under uniform SLO) │
    │ bidkv             │ SJF by prompt_tokens (same as other SJF)    │
    └─────────────────┴──────────────────────────────────────────────┘

    BidKV's advantage is NOT in admission ordering (all SJF strategies
    use prompt_tokens). The differentiation comes from quality-aware
    preemption via select_victims() using U = r / (δ + ε).
    preempt-evict is the true zero-intelligence baseline
    (FCFS waiting + LIFO preemption = vanilla vLLM behaviour).
    """
    import time

    strategy_name = adapter._experiment_strategy_name

    waiting = getattr(scheduler, "waiting", None)
    if waiting is None or len(waiting) <= 1:
        return

    now = time.monotonic()

    if strategy_name == "preempt-evict":
        # preempt-evict: FCFS — no reorder. True vLLM default behaviour.
        return

    elif strategy_name == "slack-aware":
        # slack-aware: EDF — tightest deadline first.
        # Under uniform SLO, this approximates FCFS (arrival order).
        def _deadline_key(req: Any) -> float:
            rid = getattr(req, "request_id", "")
            arrival = adapter._request_arrival_ms.get(rid, now * 1000)
            return arrival

        waiting_list = list(waiting)
        waiting_list.sort(key=_deadline_key)
        waiting.clear()
        for req in waiting_list:
            waiting.append(req)

    else:
        # All SJF strategies (preempt-evict-sjf, static-random, largest-first,
        # uniform, bidkv): SJF by prompt_tokens.
        # max_tokens is a standard API param accessible to all — NOT a
        # bid signal — so admission uses prompt_tokens only.
        waiting_list = list(waiting)
        waiting_list.sort(key=lambda r: getattr(r, "num_prompt_tokens", 0))
        waiting.clear()
        for req in waiting_list:
            waiting.append(req)


def _reorder_running_for_preemption(scheduler: Any, adapter: VLLMAdapter) -> None:
    """Reorder scheduler.running for native preemption victim selection.

    For FCFS policy, vLLM preempts with self.running.pop() — the last element.
    We put the MOST EXPENDABLE request at the END so it gets preempted first.

    Strategy differentiation through running-queue reorder:
    ┌─────────────────┬──────────────────────────────────────────────┐
    │ Strategy        │ Preemption ordering                          │
    ├─────────────────┼──────────────────────────────────────────────┤
    │ preempt-evict     │ NO reorder — pure vLLM default (LIFO)       │
    │ preempt-evict-sjf │ NO reorder — LIFO eviction (SJF ablation)   │
    │ static-random     │ select_victims(): random victim selection    │
    │ largest-first     │ select_victims(): capacity-greedy eviction   │
    │ uniform           │ select_victims(): equal treatment            │
    │ slack-aware       │ select_victims(): SLO-slack based            │
    │ bidkv             │ select_victims(): bid-informed priority      │
    └─────────────────┴──────────────────────────────────────────────┘
    """
    strategy_name = adapter._experiment_strategy_name

    # preempt-evict: NO reorder — measures pure vLLM default behavior.
    # vLLM uses running.pop() = LIFO, which evicts the most recently added.
    if strategy_name == "preempt-evict":
        return

    # preempt-evict-sjf: NO reorder — LIFO eviction, same as preempt-evict.
    # Only admission (SJF) differs. No priority cache, no proactive.
    if strategy_name == "preempt-evict-sjf":
        return

    running = getattr(scheduler, "running", None)
    if running is None or len(running) <= 1:
        return

    # BidKV: pressure-gated quality-aware reorder.
    #
    # Below 95% KV: NO reorder — pure LIFO (vLLM default). LIFO naturally
    #   provides excellent p95 by evicting the newest request (least work
    #   invested). This is the optimal policy for non-extreme pressure.
    # Above 95% KV: Quality-aware reorder from cached priority influences
    #   native preemption victim selection. BidKV's U = r/(δ+ε) scoring
    #   puts the most efficient victim at the end for running.pop().
    #
    # U-score naturally handles long-context recompute concerns:
    #   - completion factor penalizes near-done requests (避免驱逐快完成的)
    #   - anti-starvation (0.3×preemptions) prevents cascading evictions
    #   - freed dominates ordering → efficient KV reclamation
    if strategy_name == "bidkv" and len(running) >= 2:
        # Pressure gate: below 95%, LIFO is optimal.
        kv_mgr = getattr(scheduler, "kv_cache_manager", None)
        if kv_mgr is not None:
            block_pool = getattr(kv_mgr, "block_pool", None)
            if block_pool is not None:
                usage = block_pool.get_usage()
                if usage < 0.95:
                    return

    # Use strategy-specific cached priority from select_victims()
    cached = getattr(adapter, "_cached_preempt_priority", None)
    if cached:
        # Higher priority = more valuable = placed at FRONT (protected)
        # Lower priority = victim = placed at END (preempted by running.pop())
        scored = []
        for idx, req in enumerate(running):
            rid = getattr(req, "request_id", "")
            priority = cached.get(rid, float("inf"))
            scored.append((priority, idx, req))
        scored.sort(key=lambda x: (-x[0], x[1]))
        running.clear()
        for _, _, req in scored:
            running.append(req)
    else:
        # Before first cache refresh: use universal keep-score heuristic
        scored = [((_compute_keep_score(req), idx), req) for idx, req in enumerate(running)]
        scored.sort(key=lambda x: -x[0][0])
        running.clear()
        for _, req in scored:
            running.append(req)


def _refresh_priority_cache(scheduler: Any, adapter: VLLMAdapter) -> None:
    """Refresh cached preemption priority using current strategy.

    Called every 3 seconds. Runs strategy.select_victims() to get the
    priority ordering, then caches it for _reorder_running_for_preemption.

    Strategy differentiation happens here: BidKV uses bid+quality scoring,
    h2o uses attention, uniform uses equal weights, etc.
    """
    import time

    refresh_interval_s = 3.0
    now = time.monotonic()
    last_refresh = getattr(adapter, "_last_priority_refresh", 0.0)
    if now - last_refresh < refresh_interval_s:
        return
    adapter._last_priority_refresh = now

    strategy = adapter._experiment_strategy
    strategy_name = adapter._experiment_strategy_name

    # preempt-evict: no priority cache needed
    if strategy_name in ("preempt-evict", "preempt-evict-sjf") or strategy is None:
        return

    running = getattr(scheduler, "running", None)
    if running is None or len(running) < 2:
        return

    pairs = _build_running_candidates(running, adapter)
    candidates = [p[0] for p in pairs]
    if len(candidates) < 2:
        return

    # Use total KV as needed_tokens to get full ordering (all candidates ranked)
    needed_all = sum(c.current_tokens for c in candidates) or 1

    try:
        actions = strategy.select_victims(candidates, needed_all)
    except Exception:  # noqa: BLE001
        return

    # Build priority: victims are expendable (low priority).
    # actions[0] = most expendable (first to be preempted).
    # Requests NOT in actions list are most valuable.
    victim_ids = [a.request_id for a in actions]
    priority: dict[str, float] = {}

    # Most expendable gets lowest score; later victims get slightly higher
    n_victims = len(victim_ids)
    for i, rid in enumerate(victim_ids):
        priority[rid] = float(i)  # 0 = most expendable

    # Requests not selected as victims: very high priority (keep)
    for pair in pairs:
        if pair[0].request_id not in priority:
            priority[pair[0].request_id] = float(n_victims + 100)

    adapter._cached_preempt_priority = priority


def _proactive_preempt(scheduler: Any, adapter: VLLMAdapter) -> None:
    """Mode A proactive preemption using vLLM native _preempt_request().

    When KV pressure exceeds threshold, selects a SINGLE victim via cached
    priority and preempts it using vLLM's native mechanism (abort + requeue
    with PREEMPTED status). The request will be rescheduled and recomputed.

    This is the core experimental variable: different strategies select
    DIFFERENT victims. BidKV picks the most efficient candidate; baselines
    use simpler heuristics.

    Guards:
    - KV utilization > 90%
    - Waiting queue non-empty (demand for blocks)
    - At least 3 running requests (keep 2+ running)
    - 5-second cooldown between proactive events (prevent storm)
    - preempt-evict: skip entirely (it IS the "no intervention" baseline)
    """
    import time

    strategy_name = adapter._experiment_strategy_name
    cooldown_s = 5.0
    now = time.monotonic()
    last_proactive = getattr(scheduler, "_bidkv_last_proactive", 0.0)
    if now - last_proactive < cooldown_s:
        return

    # Must have waiting requests (demand for blocks)
    waiting = getattr(scheduler, "waiting", None)
    if not waiting:
        return

    # Must have >= 3 running
    running = getattr(scheduler, "running", None)
    if running is None or len(running) < 3:
        return

    # Check KV utilization
    kv_mgr = getattr(scheduler, "kv_cache_manager", None)
    if kv_mgr is None:
        return
    block_pool = getattr(kv_mgr, "block_pool", None)
    if block_pool is None:
        return
    usage = block_pool.get_usage()
    if usage < 0.90:
        return

    # preempt-evict: skip — measures vanilla framework behavior
    if strategy_name in ("preempt-evict", "preempt-evict-sjf"):
        return

    # Find victim from cached priority (lowest priority = most expendable)
    cached = getattr(adapter, "_cached_preempt_priority", None)
    if not cached:
        return

    # Find the running request with lowest cached priority
    best_victim_req = None
    best_victim_idx = -1
    best_priority = float("inf")
    for i, req in enumerate(running):
        rid = getattr(req, "request_id", None)
        if rid is None:
            continue
        p = cached.get(rid, float("inf"))
        if p < best_priority:
            best_priority = p
            best_victim_req = req
            best_victim_idx = i

    if best_victim_req is None or best_victim_idx < 0:
        return

    victim_id = getattr(best_victim_req, "request_id", "")
    freed_estimate = getattr(best_victim_req, "num_computed_tokens", 0)

    # Execute native preemption via vLLM's _preempt_request
    preempt_fn = getattr(scheduler, "_preempt_request", None)
    if preempt_fn is None:
        return

    try:
        # Pop from running first, then preempt
        running.pop(best_victim_idx)
        preempt_fn(best_victim_req, now)
    except Exception:  # noqa: BLE001
        # If preemption fails, put req back
        running.append(best_victim_req)
        return

    # Remove from prev_step_scheduled_req_ids so that when orig() picks up
    # this request from waiting as "resumed", the assertion at
    # scheduler.py:1051 (assert not scheduled_in_prev_step) won't fire.
    prev_ids = getattr(scheduler, "prev_step_scheduled_req_ids", None)
    if prev_ids is not None:
        prev_ids.discard(victim_id)

    scheduler._bidkv_last_proactive = now
    adapter._metrics.record_eviction(victim_id, freed_estimate)
    _diag(
        f"proactive PREEMPT: strategy={strategy_name} "
        f"victim={victim_id} usage={usage:.2f} freed~{freed_estimate}"
    )


_MODEL_EXECUTOR_RESOLVED = False


def _resolve_model_executor(scheduler: Any, adapter: VLLMAdapter) -> None:  # noqa: ARG001
    """Lazily discover model_executor via gc.get_referrers.

    Plugin cannot patch EngineCore.__init__ effectively because
    load_general_plugins() runs INSIDE the already-executing __init__.
    Instead, on the first schedule() call (after __init__ completes),
    walk gc.get_referrers(scheduler) to find EngineCore and grab its
    model_executor for block-table sync after truncation.
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
                _diag(f"resolved model_executor via stack walk (frame={frame.f_code.co_name})")
                return
        frame = frame.f_back
    _diag("WARN: could not resolve model_executor via stack walk")


def _get_max_tokens_estimate(req: Any) -> int:
    """Estimate max output tokens from request's sampling_params.

    All strategies have equal access to max_tokens — it is a standard
    API parameter, NOT a bid signal.  Differentiation between strategies
    comes from quality-aware preemption decisions (select_victims / U),
    not from information asymmetry in lifecycle cost estimation.
    """
    sp = getattr(req, "sampling_params", None)
    if sp is not None:
        mt = getattr(sp, "max_tokens", None)
        if mt is not None and mt > 0:
            return mt
    return 256  # conservative fallback


def _proactive_srpt(scheduler: Any, adapter: VLLMAdapter) -> None:
    """Proactive SRPT: preempt high-remaining-cost running request for low-cost waiting.

    SRPT (Shortest Remaining Processing Time) is provably optimal for
    minimizing mean flow time. The key: accurately estimating "remaining time."

    All SJF strategies use the same remaining-cost estimation from
    sampling_params.max_tokens (standard API parameter, universally
    accessible). The quality of OUTCOMES differs because prior
    preemption decisions (via select_victims / U) shape which requests
    remain running — BidKV's quality-aware eviction leaves a better
    residual composition.

    FCFS/EDF strategies (preempt-evict, slack-aware) are excluded because
    SRPT conflicts with their non-SJF scheduling discipline.

    Guard rails:
    - KV utilization > 80% (real pressure)
    - Waiting queue non-empty
    - At least 3 running requests
    - Running victim must have generated ≥ 10 output tokens
    - Clear benefit: remaining(running) > 1.2 × total(waiting)
    - 1.5-second cooldown between preemptions (prevent storm)
    """
    import time

    strategy_name = adapter._experiment_strategy_name

    # FCFS/EDF strategies excluded — SRPT conflicts with their discipline
    if strategy_name in ("preempt-evict", "preempt-evict-sjf", "slack-aware"):
        return

    # Cooldown
    now = time.monotonic()
    last_srpt = getattr(scheduler, "_bidkv_last_srpt", 0.0)
    if now - last_srpt < 1.5:
        return

    running = getattr(scheduler, "running", None)
    waiting = getattr(scheduler, "waiting", None)
    if not running or not waiting or len(running) < 3:
        return

    # KV pressure check
    kv_mgr = getattr(scheduler, "kv_cache_manager", None)
    if kv_mgr is None:
        return
    block_pool = getattr(kv_mgr, "block_pool", None)
    if block_pool is None:
        return
    usage = block_pool.get_usage()
    if usage < 0.80:
        return

    # BidKV: disable SRPT — Mode A recompute cost makes extra evictions
    # counterproductive. Each SRPT eviction triggers full prompt recompute,
    # creating extreme tail latency for the evicted request. The pressure-
    # gated quality-aware reorder (from _reorder_running_for_preemption)
    # is the only BidKV intervention; native vLLM handles eviction frequency.
    if strategy_name == "bidkv":
        return

    # Find best waiting candidate (smallest total cost)
    best_waiting = None
    best_waiting_cost = float("inf")
    for req in waiting:
        prompt = getattr(req, "num_prompt_tokens", 0)
        max_out = _get_max_tokens_estimate(req)
        total = prompt + max_out
        if total < best_waiting_cost:
            best_waiting_cost = total
            best_waiting = req

    if best_waiting is None:
        return

    # Find worst running candidate (highest remaining cost, with min output guard)
    worst_running = None
    worst_remaining = 0
    worst_idx = -1
    for i, req in enumerate(running):
        prompt = getattr(req, "num_prompt_tokens", 0)
        computed = getattr(req, "num_computed_tokens", 0)
        output_so_far = max(0, computed - prompt)

        # Must have generated at least 10 output tokens (avoid immediate re-preemption)
        if output_so_far < 10:
            continue

        # Anti-starvation: skip requests already preempted 2+ times
        if getattr(req, "num_preemptions", 0) >= 2:
            continue

        max_out = _get_max_tokens_estimate(req)
        remaining = max(0, max_out - output_so_far)

        if remaining > worst_remaining:
            worst_remaining = remaining
            worst_running = req
            worst_idx = i

    if worst_running is None:
        return

    # Benefit check: remaining(running) must be > 1.2× total(waiting)
    if worst_remaining < best_waiting_cost * 1.2:
        return

    # BidKV-specific: recompute cost-benefit gate for SRPT.
    # In Mode A, eviction triggers full recompute of the victim's prompt.
    # SRPT benefit = worst_remaining (output work saved).
    # Recompute cost = victim_prompt (must redo entire prefill).
    # Gate: only evict if benefit > cost (remaining > prompt).
    if strategy_name == "bidkv":
        victim_prompt = getattr(worst_running, "num_prompt_tokens", 0)
        if worst_remaining < victim_prompt:
            return

    # Execute preemption via vLLM native mechanism
    preempt_fn = getattr(scheduler, "_preempt_request", None)
    if preempt_fn is None:
        return

    victim_id = getattr(worst_running, "request_id", "?")
    try:
        running.pop(worst_idx)
        preempt_fn(worst_running, now)
    except Exception:  # noqa: BLE001
        running.append(worst_running)
        return

    # Remove from prev_step_scheduled_req_ids so that when orig() picks up
    # this request from waiting as "resumed", the assertion at
    # scheduler.py:1051 (assert not scheduled_in_prev_step) won't fire.
    prev_ids = getattr(scheduler, "prev_step_scheduled_req_ids", None)
    if prev_ids is not None:
        prev_ids.discard(victim_id)

    scheduler._bidkv_last_srpt = now
    adapter._metrics.record_eviction(victim_id, worst_remaining)
    _diag(
        f"SRPT preempt: strategy={strategy_name} victim={victim_id} "
        f"remaining={worst_remaining} waiting_cost={best_waiting_cost} "
        f"usage={usage:.2f}"
    )


def _patched_schedule(scheduler: Any, adapter: VLLMAdapter) -> Any:
    """Patched schedule() — Mode A: SJF admission + strategy-specific preemption.

    Flow:
    1. Sync request tracking (running + waiting)
    2. Track waiting arrival times
    3. Reorder waiting queue by strategy-specific SJF key
    4. Refresh preemption priority cache (select_victims)
    5. Proactive preemption via cached priority (all except preempt-evict)
    6. Proactive SRPT preemption (SJF strategies, excludes slack-aware)
    7. Reorder running list (strategy-specific victim ordering)
    8. Call original schedule()

    Strategy differentiation hierarchy:
    - preempt-evict: FCFS admission + LIFO preemption (true vLLM default)
    - slack-aware: EDF admission + SLO-slack preemption
    - static-random/h2o/uniform/bidkv: SJF(prompt) admission
      + strategy-specific select_victims() reorder + SRPT
    - preempt-evict-sjf: SJF(prompt) admission + LIFO preemption (ablation)
    - BidKV's edge: quality-aware U = r / (δ + ε) via completion-ratio δ
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

    # Track arrival time for waiting requests (needed for EDF/anti-starvation)
    _track_waiting_arrival(scheduler, adapter)

    # Reorder waiting queue: strategy-specific SJF key
    _reorder_waiting_for_admission(scheduler, adapter)

    # Refresh strategy-specific preemption priority cache (every 3s)
    # Calls strategy.select_victims() to rank running requests
    _refresh_priority_cache(scheduler, adapter)

    # Motivation experiment: log preemption candidate snapshot (opt-in via env var).
    # Called BEFORE any reordering so all strategies see the same raw queue.
    # Activated only when BIDKV_LOG_PREEMPTION_EVENTS is set (zero overhead otherwise).
    from bidkv.adapters.vllm import preemption_logger as _plogger

    if _plogger.is_active():
        _plogger.log_event_if_enabled(scheduler, adapter)

    # Proactive preemption using cached priority (all except preempt-evict)
    # Uses select_victims() priority to pick victim when KV > 90%.
    # This is the ONLY proactive mechanism for slack-aware (SRPT excluded
    # because it conflicts with EDF discipline).
    _proactive_preempt(scheduler, adapter)

    # Proactive SRPT preemption (all SJF strategies, excludes slack-aware)
    # Preempt high-remaining-cost running request when a low-cost
    # waiting request could be served instead. All strategies use
    # the same cost estimation; BidKV differs in prior eviction quality.
    _proactive_srpt(scheduler, adapter)

    # Reorder running list: strategy-specific victim selection
    # preempt-evict gets no reorder (vLLM default LIFO);
    # others use cached priority from select_victims()
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

    # 更新 H2O scoring 累积注意力统计
    # Only run for strategies that use H2O scoring (avoids wasting CPU
    # on strategies that don't benefit from cumulative attention data).
    # Note: BidKV Mode A uses completion-ratio δ, not H2O token scores.
    # Sampled at 20% (every 5th step) to reduce CPU overhead.
    _H2O_STRATEGIES = ("largest-first",)
    if adapter._experiment_strategy_name in _H2O_STRATEGIES:
        h2o_counter = getattr(scheduler, "_bidkv_h2o_counter", 0) + 1
        scheduler._bidkv_h2o_counter = h2o_counter
        if h2o_counter % 5 == 0:
            from bidkv.adapters.vllm.h2o_hook import update_h2o_from_output

            update_h2o_from_output(adapter, scheduler, model_runner_output)

    return result


def _patched_free_request(scheduler: Any, adapter: VLLMAdapter, request: Any, **kwargs: Any) -> Any:
    """Patched _free_request() — 请求完成时清理 BidKV 状态。"""
    request_id = getattr(request, "request_id", None)
    if request_id is not None:
        # 记录最终 output token 数，用于前置实验 completion_ratio join
        from bidkv.adapters.vllm import preemption_logger as _plogger
        if _plogger.is_active():
            num_computed = getattr(request, "num_computed_tokens", 0)
            num_prompt = getattr(request, "num_prompt_tokens", 0)
            final_output = max(0, num_computed - num_prompt)
            _plogger.log_completion(request_id, final_output)

        adapter.on_request_complete(request_id)

    # 调用原始方法（透传所有额外参数，如 delay_free_blocks）
    orig = getattr(scheduler, f"{_ORIG_PREFIX}_free_request")
    return orig(request, **kwargs)


def _track_waiting_arrival(scheduler: Any, adapter: VLLMAdapter) -> None:
    """Record arrival time for waiting requests.

    Needed for EDF (slack-aware) and anti-starvation (bidkv).
    Only records on first sight — does not overwrite.
    """
    import time

    waiting = getattr(scheduler, "waiting", None)
    if waiting is None:
        return

    now_ms = time.monotonic() * 1000
    for request in waiting:
        rid = getattr(request, "request_id", None)
        if rid is not None and rid not in adapter._request_arrival_ms:
            adapter._request_arrival_ms[rid] = now_ms


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
