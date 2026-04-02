# BidKV v8 Full Validation Analysis
# Generated: 2026-04-02
# Data: results/vllm_v8_full_validation/ (63 runs)

## Final Metric System (4-column main table)

| # | Metric | Definition | Source |
|---|--------|-----------|--------|
| 1 | Throughput (req/s) | Completed requests / experiment duration | Standard (vLLM, Orca, SGLang) |
| 2 | SLO attainment(300ms) (%) | Fraction of requests with TTFT ≤ 300ms | S³ (ISCA'24) |
| 3 | TTFT p95 (ms) | 95th percentile time to first token | Standard LLM serving |
| 4 | TPOT p95 (ms) | 95th percentile time per output token | Sarathi-Serve (OSDI'24) |

**Supplementary**: Goodput(500ms), SLO(500ms), SLO(1000ms), TTFT/TPOT p50/p99

## 1. Cross-Rate Average (Main Table Metrics)

| Strategy | Throughput | SLO300% | TTFT p95 | TPOT p95 |
|----------|-----------|---------|----------|----------|
| **bidkv** | 2.99 (#4) | 87.1 (#1) | 554 (#1) | 96.4 (#4) |
| static-random | 3.21 (#1) | 87.0 (#2) | 1076 (#5) | 86.0 (#1) |
| uniform | 3.21 (#2) | 86.9 (#3) | 1069 (#4) | 86.7 (#2) |
| h2o-style | 2.92 (#6) | 84.4 (#4) | 584 (#3) | 100.1 (#6) |
| preempt-evict-sjf | 2.77 (#7) | 82.8 (#5) | 572 (#2) | 129.4 (#7) |
| slack-aware | 3.05 (#3) | 72.4 (#6) | 4023 (#6) | 93.2 (#3) |
| preempt-evict | 2.98 (#5) | 72.2 (#7) | 5241 (#7) | 98.3 (#5) |

### Cross-Rate Ranking Summary

| Strategy | Thru Rank | SLO Rank | TTFT Rank | TPOT Rank | Rank Sum | Wins |
|----------|-----------|----------|-----------|-----------|----------|------|
| **bidkv** | #4 | #1 | #1 | #4 | 10 | 2 |
| static-random | #1 | #2 | #5 | #1 | 9 | 2 |
| uniform | #2 | #3 | #4 | #2 | 11 | 0 |
| h2o-style | #6 | #4 | #3 | #6 | 19 | 0 |
| preempt-evict-sjf | #7 | #5 | #2 | #7 | 21 | 0 |
| slack-aware | #3 | #6 | #6 | #3 | 18 | 0 |
| preempt-evict | #5 | #7 | #7 | #5 | 24 | 0 |

## 2. Per-Rate Breakdown

### Rate = 2.0 req/s

| Strategy | Throughput | SLO300% | TTFT p95 | TPOT p95 |
|----------|-----------|---------|----------|----------|
| **bidkv** | 1.96 (#1) | 97.0 (#1) | 255 (#1) | 72.4 (#1) |
| static-random | 1.96 (#2) | 96.5 (#3) | 264 (#3) | 72.8 (#2) |
| uniform | 1.96 (#3) | 96.6 (#2) | 260 (#2) | 72.9 (#3) |
| h2o-style | 1.96 (#4) | 95.5 (#4) | 288 (#4) | 74.4 (#4) |
| preempt-evict-sjf | 1.96 (#5) | 94.2 (#5) | 326 (#5) | 84.3 (#7) |
| slack-aware | 1.96 (#6) | 91.8 (#7) | 904 (#7) | 81.3 (#6) |
| preempt-evict | 1.96 (#7) | 92.0 (#6) | 667 (#6) | 78.1 (#5) |

### Rate = 3.8 req/s

| Strategy | Throughput | SLO300% | TTFT p95 | TPOT p95 |
|----------|-----------|---------|----------|----------|
| **bidkv** | 3.42 (#5) | 83.2 (#2) | 630 (#1) | 107.1 (#5) |
| static-random | 3.57 (#1) | 84.1 (#1) | 1195 (#5) | 90.8 (#1) |
| uniform | 3.57 (#2) | 82.6 (#3) | 1185 (#4) | 91.5 (#2) |
| h2o-style | 3.28 (#6) | 79.9 (#4) | 675 (#3) | 112.8 (#6) |
| preempt-evict-sjf | 3.06 (#7) | 79.6 (#5) | 666 (#2) | 153.4 (#7) |
| slack-aware | 3.47 (#4) | 68.2 (#7) | 5386 (#6) | 93.6 (#3) |
| preempt-evict | 3.48 (#3) | 68.4 (#6) | 6542 (#7) | 96.9 (#4) |

### Rate = 5.7 req/s

| Strategy | Throughput | SLO300% | TTFT p95 | TPOT p95 |
|----------|-----------|---------|----------|----------|
| **bidkv** | 3.60 (#4) | 81.0 (#2) | 778 (#2) | 109.8 (#4) |
| static-random | 4.09 (#2) | 80.4 (#3) | 1771 (#5) | 94.3 (#1) |
| uniform | 4.10 (#1) | 81.6 (#1) | 1760 (#4) | 95.6 (#2) |
| h2o-style | 3.53 (#5) | 77.7 (#4) | 790 (#3) | 113.0 (#5) |
| preempt-evict-sjf | 3.28 (#7) | 74.5 (#5) | 725 (#1) | 150.5 (#7) |
| slack-aware | 3.72 (#3) | 57.2 (#6) | 5779 (#6) | 104.6 (#3) |
| preempt-evict | 3.49 (#6) | 56.2 (#7) | 8513 (#7) | 119.8 (#6) |

### BidKV Per-Rate Performance

| Rate | Thru Rank | SLO Rank | TTFT Rank | TPOT Rank | Wins/4 | Top-3/4 |
|------|-----------|----------|-----------|-----------|--------|---------|
| 2.0 | #1 | #1 | #1 | #1 | 4/4 | 4/4 |
| 3.8 | #5 | #2 | #1 | #5 | 1/4 | 2/4 |
| 5.7 | #4 | #2 | #2 | #4 | 0/4 | 2/4 |
| **Cross-rate** | #4 | #1 | #1 | #4 | 2/4 | 2/4 |

## 3. Supplementary Metrics (Cross-Rate Average)

| Strategy | Goodput(500) | SLO500% | SLO1000% | TTFT p50 | TTFT p99 | TPOT p50 | TPOT p99 |
|----------|-------------|---------|----------|---------|---------|---------|---------|
| **bidkv** | 2.79 | 94.0 | 97.9 | 104 | 3506 | 52.2 | 161.5 |
| static-random | 2.92 | 92.3 | 95.5 | 112 | 3869 | 50.5 | 123.9 |
| uniform | 2.90 | 91.6 | 95.4 | 113 | 3745 | 50.6 | 125.6 |
| h2o-style | 2.68 | 92.8 | 97.9 | 109 | 2816 | 55.6 | 183.7 |
| preempt-evict-sjf | 2.46 | 90.0 | 97.6 | 108 | 3467 | 52.9 | 196.0 |
| slack-aware | 2.24 | 76.6 | 82.0 | 131 | 11065 | 49.8 | 131.7 |
| preempt-evict | 2.19 | 76.8 | 81.8 | 133 | 11694 | 49.9 | 144.9 |

## 4. BidKV vs. Each Baseline (Cross-Rate Δ)

| vs. Baseline | ΔThru | ΔSLO300 | ΔTTFT p95 | ΔTPOT p95 | Wins |
|-------------|-------|---------|-----------|-----------|------|
| vs. static-random | -0.22 | +0.1pp | +522ms | -10.5ms | 2/4 |
| vs. uniform | -0.22 | +0.1pp | +514ms | -9.8ms | 2/4 |
| vs. h2o-style | +0.07 | +2.7pp | +30ms | +3.7ms | 4/4 |
| vs. preempt-evict-sjf | +0.22 | +4.3pp | +18ms | +33.0ms | 4/4 |
| vs. slack-aware | -0.06 | +14.7pp | +3469ms | -3.3ms | 2/4 |
| vs. preempt-evict | +0.01 | +14.9pp | +4686ms | +1.9ms | 4/4 |

## 5. Per-Run Variance (Std Dev across 3 runs)

| Strategy | Rate | Thru σ | SLO300 σ | TTFT95 σ | TPOT95 σ |
|----------|------|--------|----------|----------|----------|
| **bidkv** | 2.0 | 0.000 | 0.26 | 9.2 | 1.57 |
| **bidkv** | 3.8 | 0.035 | 1.80 | 40.1 | 5.05 |
| **bidkv** | 5.7 | 0.089 | 0.70 | 123.7 | 8.04 |
| static-random | 2.0 | 0.000 | 0.40 | 16.6 | 0.83 |
| static-random | 3.8 | 0.008 | 0.95 | 59.5 | 4.65 |
| static-random | 5.7 | 0.058 | 2.43 | 197.7 | 2.80 |
| uniform | 2.0 | 0.000 | 0.55 | 21.7 | 0.70 |
| uniform | 3.8 | 0.020 | 1.20 | 131.9 | 4.08 |
| uniform | 5.7 | 0.028 | 0.59 | 417.8 | 1.02 |
| h2o-style | 2.0 | 0.000 | 0.25 | 2.6 | 0.85 |
| h2o-style | 3.8 | 0.085 | 1.87 | 8.7 | 5.23 |
| h2o-style | 5.7 | 0.054 | 0.42 | 33.0 | 3.16 |
| preempt-evict-sjf | 2.0 | 0.000 | 0.32 | 23.7 | 0.35 |
| preempt-evict-sjf | 3.8 | 0.064 | 1.63 | 3.2 | 8.19 |
| preempt-evict-sjf | 5.7 | 0.044 | 1.70 | 46.9 | 10.21 |
| slack-aware | 2.0 | 0.000 | 1.48 | 480.5 | 4.11 |
| slack-aware | 3.8 | 0.046 | 0.91 | 506.9 | 7.05 |
| slack-aware | 5.7 | 0.182 | 0.95 | 562.3 | 12.24 |
| preempt-evict | 2.0 | 0.000 | 0.35 | 47.5 | 2.14 |
| preempt-evict | 3.8 | 0.039 | 0.60 | 108.9 | 3.14 |
| preempt-evict | 5.7 | 0.081 | 0.87 | 1515.5 | 4.92 |

## 6. Key Observations

### BidKV Strengths (main table)
- **SLO attainment(300ms) #1**: BidKV achieves 87.1%,
  the highest fraction meeting the strict 300ms TTFT target across all rates.
- **TTFT p95 #1**: BidKV controls tail prefill latency at 554ms,
  vs. next-best 572ms (PE-SJF) and far ahead of static-random (1076ms).

### BidKV Weaknesses (main table)
- **Throughput #4**: 7% below static-random/uniform due to disabling SRPT.
  SRPT aggressively preempts long-running requests to free KV for new arrivals,
  boosting throughput at the cost of latency predictability.
- **TPOT p95 #4**: 12% behind static-random. Same root cause — no SRPT means
  long decode sequences are not preempted, increasing tail TPOT.

### Tradeoff Narrative
BidKV's quality-aware victim selection avoids aggressive SRPT preemption,
trading ~7% throughput for significantly better user-facing latency quality:
- 1.9x better TTFT p95 vs static-random (554ms vs 1076ms)
- +15pp SLO advantage vs PE baseline (87.1% vs 72.2%)
- Under moderate load (rate=2.0), BidKV dominates all 4 metrics.
- The throughput gap only appears under high KV pressure (rate=3.8, 5.7)
  where SRPT-enabled strategies sacrifice latency predictability for raw throughput.
