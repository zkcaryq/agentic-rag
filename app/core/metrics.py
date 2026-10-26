"""流式性能指标：TTFT（首 Token 延迟）与总延迟的进程内滑动窗口聚合。

简历里的「TTFT 指标埋点」「支撑并发查询」需要能拿出数字，所以这里按 endpoint
分桶保存最近 window_size 个样本，`GET /metrics` 直接暴露 P50/P95/P99，
`scripts/bench_concurrency.py` 压测时也用它做服务端侧的对照
（压测脚本量的是客户端 TTFT，两者口径不同，分开命名才不糊弄）。

注意：窗口只活在单个进程里。多 worker 部署时每个进程各算各的，
要跨进程聚合得换成 Redis，这里没做——演示与压测都是单进程。
"""
from __future__ import annotations

import threading
import time
from collections import deque
from dataclasses import dataclass, field

from app.conf.infra_config import infra_config_metrics


@dataclass
class Sample:
    endpoint: str
    ttft_ms: float | None
    total_ms: float
    tokens: int | None
    ts: float = field(default_factory=time.monotonic)


_buckets: dict[str, deque] = {}
_lock = threading.Lock()


def _window_size() -> int:
    return max(1, int(infra_config_metrics.window_size or 200))


def _bucket(endpoint: str) -> deque:
    bucket = _buckets.get(endpoint)
    if bucket is None:
        bucket = _buckets[endpoint] = deque(maxlen=_window_size())
    return bucket


def record(
    endpoint: str,
    *,
    total_ms: float,
    ttft_ms: float | None = None,
    tokens: int | None = None,
) -> None:
    """记一个样本。endpoint 分桶：query（整轮问答）/ answer_generate（纯生成）。"""
    if not infra_config_metrics.enabled:
        return
    with _lock:
        _bucket(endpoint).append(Sample(endpoint, ttft_ms, total_ms, tokens))


def _percentile(sorted_vals: list[float], p: float) -> float:
    """线性插值百分位：样本少时比"取第 N 个"更稳，也不会把 p50 报成 p0。"""
    if not sorted_vals:
        return 0.0
    k = (len(sorted_vals) - 1) * p
    low = int(k)
    high = min(low + 1, len(sorted_vals) - 1)
    return round(sorted_vals[low] + (sorted_vals[high] - sorted_vals[low]) * (k - low), 1)


def snapshot() -> dict:
    """各 endpoint 的窗口快照，供 `GET /metrics` 和压测脚本读取。"""
    with _lock:
        buckets = {ep: list(bucket) for ep, bucket in sorted(_buckets.items()) if bucket}

    out: dict = {}
    for ep, samples in buckets.items():
        ttfts = sorted(s.ttft_ms for s in samples if s.ttft_ms is not None)
        totals = sorted(s.total_ms for s in samples)
        tokens = [s.tokens for s in samples if s.tokens is not None]
        avg_total = sum(totals) / len(totals)
        avg_tokens = sum(tokens) / len(tokens) if tokens else 0.0
        out[ep] = {
            "count": len(samples),
            # ttft_count 单独给：非流式请求没有首 Token，混进来会虚高样本数
            "ttft_count": len(ttfts),
            "ttft_p50": _percentile(ttfts, 0.5),
            "ttft_p95": _percentile(ttfts, 0.95),
            "ttft_p99": _percentile(ttfts, 0.99),
            "total_p50": _percentile(totals, 0.5),
            "total_p95": _percentile(totals, 0.95),
            "total_p99": _percentile(totals, 0.99),
            "avg_total_ms": round(avg_total, 1),
            "avg_tokens": round(avg_tokens, 1),
            "tokens_per_sec": round(avg_tokens / (avg_total / 1000), 1)
            if tokens and avg_total > 0
            else 0.0,
        }
    return out


def reset() -> None:
    """清空窗口（压测脚本跑前重置、单测隔离用）。"""
    with _lock:
        _buckets.clear()
