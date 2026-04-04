#!/usr/bin/env python3
"""绘制驱逐候选异质性图 — 论文前置实验 v2。

BidKVStrategy.select_victims 对齐：
  X = Re-prefill Cost = prompt_tokens / 256
  Y = Output Tokens Freed = num_output_tokens（仅解码阶段 KV）
  U = output_tokens / (recompute_norm + late_penalty + starvation)

git clone https://github.com/cybber695/bidkv.git
  LIFO  → 选最新到达 → Y 小（刚开始解码）→ 左下
  H2O   → 选 total tokens 最多 → long-prompt → X 大 → 右下
  BidKV → 选真实U最高 → Y 大 + X 小 → 左上  (理想区域)

git clone https://github.com/cybber695/bidkv.git
    python scripts/plot_victim_heterogeneity.py \\
        --log-file results/preliminary_motivation/victim_heterogeneity.jsonl \\
        --output-dir results/preliminary_motivation/ \\
        --format "pdf,png"
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np

STRATEGY_STYLE: dict[str, dict[str, Any]] = {
    "pe-lifo": {
        "label": "vLLM Default (LIFO)",
        "color": "#e74c3c",
        "marker": "X",
        "size": 100,
        "zorder": 5,
        "alpha": 0.85,
    },
    "largest-first": {
        "label": "Largest-First",
        "color": "#f39c12",
        "marker": "^",
        "size": 90,
        "zorder": 4,
        "alpha": 0.85,
    },
    "pe-sjf": {
        "label": "PE-SJF",
        "color": "#27ae60",
        "marker": "s",
        "size": 90,
        "zorder": 4,
        "alpha": 0.85,
    },
    "bidkv": {
        "label": "BidKV (Ours)",
        "color": "#2980b9",
        "marker": "o",
        "size": 110,
        "zorder": 6,
        "alpha": 0.90,
    },
}
PLOT_STRATEGIES = ["pe-lifo", "largest-first", "pe-sjf", "bidkv"]


def true_bidkv_u(c: dict) -> float:
    """BidKV Mode A utility (与 bidkv_strategy.py 完全对齐)."""
    output_tokens = c.get("num_output_tokens", 0)
    num_preemptions = c.get("num_preemptions", 0)
    if output_tokens <= 2:
        current = c.get("tokens_freed", c.get("num_computed_tokens", 0))
        return current / max(0.1, 1.0 + num_preemptions * 0.5)
    num_prompt = c.get("num_prompt_tokens", 0)
    recompute_norm = max(0.5, num_prompt / 256.0)
    comp = c.get("completion_ratio", 0.0)
    late_penalty = comp * comp * 2.0
    starvation = num_preemptions * 0.5
    quality_delta = max(0.1, recompute_norm + late_penalty + starvation)
    return output_tokens / quality_delta


def load_completions(comp_path: Path | None) -> dict[str, int]:
    """Load completions.jsonl → {request_id: final_output_tokens}."""
    if comp_path is None or not comp_path.exists():
        return {}
    result: dict[str, int] = {}
    with open(comp_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
                rid = rec.get("request_id")
                fot = rec.get("final_output_tokens")
                if rid and fot is not None:
                    result[rid] = int(fot)
            except (json.JSONDecodeError, ValueError):
                continue
    return result


def load_events(log_path: Path) -> list[dict]:
    events: list[dict] = []
    with open(log_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                events.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return events


def extract_data(events: list[dict], completions: dict[str, int] | None = None) -> dict[str, Any]:
    """提取数据，使用正确的轴。

    X = true completion_ratio = num_output_tokens_at_snapshot / final_output_tokens
        若 completions 为空，回退到 output / max_output
    Y = num_computed_tokens = prompt + output_so_far（当前 KV cache 占用量）
    U = true BidKV utility
    rank = 候选集内按 U 排名 (1=最优)
    """
    all_x: list[float] = []
    all_y: list[float] = []
    all_u: list[float] = []
    strat_x: dict[str, list[float]] = {s: [] for s in PLOT_STRATEGIES}
    strat_y: dict[str, list[float]] = {s: [] for s in PLOT_STRATEGIES}
    strat_u: dict[str, list[float]] = {s: [] for s in PLOT_STRATEGIES}
    strat_rank: dict[str, list[int]] = {s: [] for s in PLOT_STRATEGIES}
    kv_usage_list: list[float] = []
    valid_events = 0

    for event in events:
        candidates = event.get("candidates", [])
        choices = event.get("strategy_choices", {})
        if not candidates or not choices:
            continue
        valid_events += 1
        kv_usage_list.append(event.get("kv_usage", 0.0))

        # 第一步：用 completions join 更新每个候选的真实 completion_ratio
        for c in candidates:
            rid2 = c["request_id"]
            num_output = c.get("num_output_tokens", 0)
            if completions and rid2 in completions and completions[rid2] > 0:
                comp = min(1.0, num_output / completions[rid2])
            else:
                comp = c.get("completion_ratio", 0.0)  # fallback: output/max_output
            c["completion_ratio"] = comp  # 更新后 true_bidkv_u 也用真实分母

        # 用更新后的 completion_ratio 重新计算 U
        cand_u_list = [(c, true_bidkv_u(c)) for c in candidates]
        cand_u_sorted = sorted(cand_u_list, key=lambda t: t[1], reverse=True)
        rank_map = {c["request_id"]: i + 1 for i, (c, _) in enumerate(cand_u_sorted)}
        cand_map = {c["request_id"]: (c, u) for c, u in cand_u_list}

        for c, u in cand_u_list:
            comp = c.get("completion_ratio", 0.0)
            cached = c.get("num_computed_tokens", 0)  # KV cache footprint = prompt + output_so_far
            all_x.append(comp)
            all_y.append(float(cached))
            all_u.append(u)

        for strat in PLOT_STRATEGIES:
            rid = choices.get(strat)
            if rid and rid in cand_map:
                c, u = cand_map[rid]
                comp = c.get("completion_ratio", 0.0)
                cached = c.get("num_computed_tokens", 0)
                strat_x[strat].append(comp)
                strat_y[strat].append(float(cached))
                strat_u[strat].append(u)
                strat_rank[strat].append(rank_map[rid])

    return {
        "all_x": np.array(all_x),
        "all_y": np.array(all_y),
        "all_u": np.array(all_u),
        "strat_x": {s: np.array(v) for s, v in strat_x.items()},
        "strat_y": {s: np.array(v) for s, v in strat_y.items()},
        "strat_u": {s: np.array(v) for s, v in strat_u.items()},
        "strat_rank": {s: np.array(v) for s, v in strat_rank.items()},
        "kv_usage": np.array(kv_usage_list),
        "event_count": valid_events,
    }


def plot_main(data: dict[str, Any], output_dir: Path, formats: list[str]) -> None:
    """双栏主图：(a) 2D scatter  (b) rank violin。"""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5),
                             gridspec_kw={"width_ratios": [2.2, 1]})
    rng = np.random.default_rng(42)

    # ── Panel A: 2D Scatter ─────────────────────────────────────────
    ax = axes[0]
    cx = data["all_x"]
    cy = data["all_y"]

    jx = rng.normal(0, 0.015, size=len(cx))
    cy_std = float(cy.std())
    jy = rng.normal(0, max(cy_std * 0.015, 1.0), size=len(cy))

    ax.scatter(cx + jx, cy + jy, s=14, color="#bbbbbb", alpha=0.15,
               linewidths=0, zorder=1, rasterized=True, label="All candidates")

    y_max = float(cy.max()) if len(cy) > 0 else 800

    for strat in PLOT_STRATEGIES:
        sx = data["strat_x"][strat]
        sy = data["strat_y"][strat]
        if len(sx) == 0:
            continue
        style = STRATEGY_STYLE[strat]
        ax.scatter(sx, sy, s=style["size"], c=style["color"],
                   marker=style["marker"], alpha=style["alpha"],
                   zorder=style["zorder"], label=style["label"],
                   linewidths=0.5, edgecolors="white")

    # 均值十字 (+)
    for strat in PLOT_STRATEGIES:
        sx = data["strat_x"][strat]
        sy = data["strat_y"][strat]
        if len(sx) < 3:
            continue
        style = STRATEGY_STYLE[strat]
        mx, my = float(np.mean(sx)), float(np.mean(sy))
        ax.plot(mx, my, marker="+", markersize=20, color=style["color"],
                markeredgewidth=2.8, zorder=10, alpha=0.95)

    # Ideal zone：低完成度 + 大 KV footprint → 左上角
    ax.annotate(
        "Ideal victim:\nhigh KV footprint,\nlow decode progress",
        xy=(0.06, y_max * 0.82),
        xytext=(0.35, y_max * 0.68),
        fontsize=8, color="#2c3e50", ha="center",
        arrowprops=dict(arrowstyle="->", color="#2c3e50", lw=1.0),
        bbox=dict(boxstyle="round,pad=0.3", facecolor="#ecf0f1",
                  alpha=0.75, edgecolor="none"),
    )

    ax.set_xlabel("Decode Completion Ratio  (output tokens / final output tokens)", fontsize=10)
    ax.set_ylabel("KV Cache Footprint  (prompt + generated tokens)", fontsize=10)
    avg_kv = float(data["kv_usage"].mean()) * 100
    ax.set_title(
        f"(a) Eviction Candidates & Strategy Selections\n"
        f"({data['event_count']} pressure events, avg KV = {avg_kv:.0f}%)",
        fontsize=10, pad=8)
    ax.set_xlim(-0.02, 1.05)
    ax.set_ylim(bottom=0)
    ax.xaxis.set_major_formatter(ticker.PercentFormatter(xmax=1))
    ax.yaxis.set_major_formatter(
        ticker.FuncFormatter(lambda v, _: f"{int(v):,}"))
    ax.grid(True, linestyle="--", linewidth=0.4, alpha=0.4)

    handles, labels = ax.get_legend_handles_labels()
    gi = labels.index("All candidates") if "All candidates" in labels else None
    if gi is not None:
        handles = handles[:gi] + handles[gi+1:] + [handles[gi]]
        labels = labels[:gi] + labels[gi+1:] + [labels[gi]]
    ax.legend(handles, labels, fontsize=8.5, loc="upper right",
              framealpha=0.85, edgecolor="#bdc3c7")

    # ── Panel B: Rank Violin ────────────────────────────────────────
    ax2 = axes[1]
    rank_data_list = []
    rank_labels_list = []
    rank_colors_list = []
    rank_means_list = []
    rank_pct1_list = []

    all_r_flat: list[int] = []
    for strat in PLOT_STRATEGIES:
        r = data["strat_rank"][strat]
        if len(r) > 0:
            all_r_flat.extend(r.tolist())
    max_rank = max(all_r_flat) if all_r_flat else 15

    for strat in PLOT_STRATEGIES:
        r = data["strat_rank"][strat]
        if len(r) == 0:
            continue
        style = STRATEGY_STYLE[strat]
        rank_data_list.append(r)
        rank_labels_list.append(style["label"])
        rank_colors_list.append(style["color"])
        rank_means_list.append(float(np.mean(r)))
        rank_pct1_list.append(float(np.sum(r == 1) / len(r) * 100))

    if rank_data_list:
        positions = list(range(1, len(rank_data_list) + 1))
        try:
            vp = ax2.violinplot(rank_data_list, positions=positions,
                                widths=0.65, showmedians=True, showextrema=True)
            for body, color in zip(vp["bodies"], rank_colors_list):
                body.set_facecolor(color)
                body.set_alpha(0.55)
                body.set_edgecolor(color)
                body.set_linewidth(0.8)
            for part in ("cmedians", "cmins", "cmaxes", "cbars"):
                if part in vp:
                    vp[part].set_color("#2c3e50")
                    vp[part].set_linewidth(1.2)
        except Exception:
            bp = ax2.boxplot(
                rank_data_list, positions=positions, patch_artist=True,
                widths=0.55,
                medianprops=dict(color="black", linewidth=2),
                whiskerprops=dict(linewidth=1.2),
                capprops=dict(linewidth=1.2),
                flierprops=dict(marker=".", markersize=4, alpha=0.4),
            )
            for patch, color in zip(bp["boxes"], rank_colors_list):
                patch.set_facecolor(color)
                patch.set_alpha(0.65)

        rng2 = np.random.default_rng(7)
        for i, (r, color) in enumerate(zip(rank_data_list, rank_colors_list)):
            jitter = rng2.uniform(-0.18, 0.18, size=len(r))
            ax2.scatter(np.full(len(r), i + 1) + jitter, r, s=16, c=color,
                        alpha=0.35, zorder=5, linewidths=0)

        y_top = max_rank * 1.06
        for i, (mean_r, pct1, color) in enumerate(
                zip(rank_means_list, rank_pct1_list, rank_colors_list)):
            ax2.text(i + 1, y_top, f"mu={mean_r:.1f}",
                     ha="center", va="bottom", fontsize=7.5,
                     color=color, fontweight="bold")
            ax2.text(i + 1, y_top - max_rank * 0.08,
                     f"rank-1: {pct1:.0f}%",
                     ha="center", va="bottom", fontsize=6.5, color=color)

        ax2.set_xticks(positions)
        ax2.set_xticklabels(rank_labels_list, fontsize=7.5, rotation=20, ha="right")
        ax2.set_ylabel(f"Victim Rank by U  (1=best, {max_rank}=worst)", fontsize=9)
        ax2.set_title(
            "(b) Quality of Chosen Victim\n(lower rank = smarter eviction choice)",
            fontsize=10, pad=8)
        ax2.set_ylim(0.5, y_top + max_rank * 0.16)
        ax2.invert_yaxis()
        ax2.grid(True, axis="y", linestyle="--", linewidth=0.4, alpha=0.4)
        ax2.axhline(1, color="#27ae60", lw=1.2, ls="--", alpha=0.6, zorder=1)
        ax2.text(len(rank_data_list) + 0.45, 1, "optimal",
                 fontsize=6.5, color="#27ae60", va="center", alpha=0.8)

    fig.tight_layout(pad=1.5)
    for fmt in formats:
        out = output_dir / f"fig_victim_heterogeneity.{fmt}"
        fig.savefig(out, dpi=200, bbox_inches="tight")
        print(f"[OK] Saved: {out}")
    plt.close(fig)
    _print_stats(data)


def plot_centroid_comparison(
        data: dict[str, Any], output_dir: Path, formats: list[str]) -> None:
    """质心对比图（论文附图或 motivation 主图）。"""
    fig, ax = plt.subplots(figsize=(6.5, 5))
    cx, cy = data["all_x"], data["all_y"]
    ax.scatter(cx, cy, s=10, color="#cccccc", alpha=0.12, linewidths=0,
               zorder=1, rasterized=True)

    y_max = float(cy.max()) if len(cy) > 0 else 800

    for strat in PLOT_STRATEGIES:
        sx = data["strat_x"][strat]
        sy = data["strat_y"][strat]
        if len(sx) < 5:
            continue
        style = STRATEGY_STYLE[strat]
        ax.scatter(sx, sy, s=50, c=style["color"], marker=style["marker"],
                   alpha=0.40, zorder=4, linewidths=0.4, edgecolors="white")
        mx, my = float(np.mean(sx)), float(np.mean(sy))
        ax.plot(mx, my, marker="*", markersize=18, color=style["color"],
                markeredgecolor="white", markeredgewidth=0.8, zorder=10)
        # 根据策略位置选择偏移方向避免重叠
        dir_x = -0.12 if mx > 0.5 else 0.04
        dir_y = y_max * 0.10
        ax.annotate(
            f"{style['label']}\n(comp={mx:.0%}, KV={my:.0f})",
            xy=(mx, my),
            xytext=(mx + dir_x, my + dir_y),
            fontsize=7.5, color=style["color"], fontweight="bold",
            arrowprops=dict(arrowstyle="-", color=style["color"],
                            lw=0.8, alpha=0.7),
        )

    ax.annotate(
        "<- Ideal victim\n   (high KV, low completion)",
        xy=(0.05, y_max * 0.82),
        fontsize=8.5, color="#2c3e50", ha="left",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="#ecf0f1",
                  alpha=0.8, edgecolor="none"),
    )
    ax.set_xlabel("Decode Completion Ratio  (output tokens / final output tokens)", fontsize=11)
    ax.set_ylabel("KV Cache Footprint  (prompt + generated tokens)", fontsize=11)
    ax.set_title("Eviction Strategy Centroids\n(star = mean choice per strategy)",
                 fontsize=10)
    ax.set_xlim(-0.02, 1.05)
    ax.set_ylim(bottom=0)
    ax.xaxis.set_major_formatter(ticker.PercentFormatter(xmax=1))
    ax.yaxis.set_major_formatter(
        ticker.FuncFormatter(lambda v, _: f"{int(v):,}"))
    ax.grid(True, linestyle="--", linewidth=0.4, alpha=0.4)
    fig.tight_layout()
    for fmt in formats:
        out = output_dir / f"fig_victim_centroid.{fmt}"
        fig.savefig(out, dpi=200, bbox_inches="tight")
        print(f"[OK] Saved: {out}")
    plt.close(fig)


def _print_stats(data: dict[str, Any]) -> None:
    cx, cy = data["all_x"], data["all_y"]
    print()
    print("=" * 65)
    print("  Preemption Candidate Statistics  (v2 axes)")
    print("=" * 65)
    print(f"  Events     : {data['event_count']}")
    print(f"  Candidates : {len(cx)}")
    print(f"  KV usage   : {float(data['kv_usage'].mean()) * 100:.1f}%")
    print(f"  X (completion_ratio)  : mean={float(cx.mean()):.3f}, "
          f"std={float(cx.std()):.3f}")
    print(f"  Y (kv_footprint)    : mean={float(cy.mean()):.0f}, "
          f"std={float(cy.std()):.0f}")
    print()
    print(f"  {'Strategy':<24} {'N':>4}  {'Avg comp':>9}  {'Avg Y(kv)':>10}  "
          f"{'Avg U':>8}  {'Rank-1%':>8}")
    print("  " + "-" * 65)
    for strat in PLOT_STRATEGIES:
        sx = data["strat_x"][strat]
        sy = data["strat_y"][strat]
        su = data["strat_u"][strat]
        sr = data["strat_rank"][strat]
        if len(sx) == 0:
            continue
        label = STRATEGY_STYLE[strat]["label"]
        pct1 = float(np.sum(sr == 1) / len(sr) * 100)
        print(f"  {label:<24} {len(sx):>4}  {float(np.mean(sx)):>8.1%}  "
              f"{float(np.mean(sy)):>10.1f}  {float(np.mean(su)):>8.1f}  "
              f"{pct1:>7.1f}%")
    print("=" * 65)
    lifo_u_arr = data["strat_u"]["pe-lifo"]
    lf_u_arr = data["strat_u"]["largest-first"]
    sjf_u_arr = data["strat_u"]["pe-sjf"]
    bidkv_u_arr = data["strat_u"]["bidkv"]
    lifo_u = float(np.mean(lifo_u_arr)) if len(lifo_u_arr) > 0 else 1.0
    lf_u = float(np.mean(lf_u_arr)) if len(lf_u_arr) > 0 else 1.0
    sjf_u = float(np.mean(sjf_u_arr)) if len(sjf_u_arr) > 0 else 1.0
    bidkv_u = float(np.mean(bidkv_u_arr)) if len(bidkv_u_arr) > 0 else 1.0
    print(f"  BidKV / LIFO = {bidkv_u/lifo_u:.1f}x  "
          f"BidKV / Largest-First = {bidkv_u/lf_u:.1f}x  "
          f"BidKV / PE-SJF = {bidkv_u/sjf_u:.1f}x")
    print()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot victim heterogeneity figure for BidKV motivation (v3)")
    parser.add_argument("--log-file", type=Path,
                        default=Path(
                            "results/preliminary_motivation/victim_heterogeneity.jsonl"))
    parser.add_argument("--completions-file", type=Path, default=None,
                        help="completions.jsonl 记录每个 request 的最终 output 长度"
                             "（默认与 --log-file 同目录）")
    parser.add_argument("--output-dir", type=Path,
                        default=Path("results/preliminary_motivation"))
    parser.add_argument("--format", type=str, default="pdf,png")
    parser.add_argument("--no-centroid", action="store_true",
                        help="Skip centroid summary figure")
    args = parser.parse_args()

    if not args.log_file.exists():
        print(f"[ERROR] Log file not found: {args.log_file}")
        sys.exit(1)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    formats = [f.strip() for f in args.format.split(",")]

    # 推断 completions 文件路径
    comp_path: Path | None = args.completions_file
    if comp_path is None:
        comp_path = args.log_file.parent / "completions.jsonl"

    print(f"[INFO] Loading: {args.log_file}")
    events = load_events(args.log_file)
    print(f"[INFO] Loaded {len(events)} raw events")
    if not events:
        print("[ERROR] No events found")
        sys.exit(1)

    completions = load_completions(comp_path)
    if completions:
        print(f"[INFO] Completions loaded: {len(completions)} records from {comp_path}")
    else:
        print(f"[WARN] No completions found at {comp_path}, using max_output as fallback")

    data = extract_data(events, completions)
    print(f"[INFO] {data['event_count']} valid events, "
          f"{len(data['all_x'])} observations")

    print("[INFO] Rendering main figure...")
    plot_main(data, args.output_dir, formats)

    if not args.no_centroid:
        print("[INFO] Rendering centroid figure...")
        plot_centroid_comparison(data, args.output_dir, formats)

    print("[INFO] Done ->", args.output_dir)


if __name__ == "__main__":
    main()
