"""流式性能指标：TTFT（首 Token 延迟）与总延迟的进程内滑动窗口聚合。

简历里的「TTFT 指标埋点」「支撑并发查询」需要能拿出数字，所以这里按 endpoint
分桶保存最近 window_size 个样本，`GET /metrics` 直接暴露 P50/P95/P99，
`scripts/bench_concurrency.py` 压测时也用它做服务端侧的对照
（压测脚本量的是客户端 TTFT，两者口径不同，分开命名才不糊弄）。

注意：窗口本身是**进程内**的。查询派发到 MQ worker 后样本会落在 worker 进程，
所以 record 之后调用 export_to_bus() 把样本镜像到 Redis `rag:metrics:{pid}`，
`GET /metrics` 读取时合并所有进程窗口——否则多 worker 部署下接口只会看到一个空窗口。
"""
from __future__ import annotations

import os
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


def record(endpoint: str, *, total_ms: float, ttft_ms: float | None = None, tokens: int | None = None) -> None:
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


def _aggregate(samples: list) -> dict:
    ttfts = sorted(s.ttft_ms for s in samples if s.ttft_ms is not None)
    totals = sorted(s.total_ms for s in samples)
    tokens = [s.tokens for s in samples if s.tokens is not None]
    avg_total = sum(totals) / len(totals)
    avg_tokens = sum(tokens) / len(tokens) if tokens else 0.0
    return {
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


def snapshot(extra: dict[str, list] | None = None) -> dict:
    """各 endpoint 的窗口快照，供 `GET /metrics` 和压测脚本读取。

    extra：其它进程导出后合并进来的样本列表（同 endpoint 拼在一起再算分位数），
    这样多 worker 部署时接口仍能看到全局数字。
    """
    with _lock:
        buckets = {ep: list(bucket) for ep, bucket in sorted(_buckets.items()) if bucket}

    for ep, samples in (extra or {}).items():
        buckets.setdefault(ep, [])
        buckets[ep] = buckets[ep] + list(samples)

    return {ep: _aggregate(samples) for ep, samples in sorted(buckets.items()) if samples}


def dump_samples() -> dict[str, list[list]]:
    """导出本进程窗口样本（给别的进程合并用）。"""
    with _lock:
        return {
            ep: [[s.ttft_ms, s.total_ms, s.tokens] for s in bucket]
            for ep, bucket in _buckets.items()
            if bucket
        }


def load_samples(raw: dict[str, list[list]]) -> dict[str, list]:
    """把 dump_samples 的形态还原成 Sample 列表。"""
    out: dict[str, list] = {}
    for ep, rows in (raw or {}).items():
        out[ep] = [Sample(ep, row[0], row[1], row[2]) for row in rows if len(row) == 3]
    return out


def export_to_bus() -> bool:
    """把本进程样本写到 Redis（`rag:metrics:{pid}`），供 API 进程合并读取。

    为什么需要：查询派发到 MQ worker 后，样本记在 worker 进程的窗口里，
    HTTP 侧的 /metrics 永远读不到——多 worker 部署时"有埋点"就成了空话。
    """
    if not infra_config_metrics.enabled:
        return False
    raw = dump_samples()
    if not raw:
        return False
    from app.clients import redis_utils

    return redis_utils.store_metrics(os.getpid(), raw)


def reset() -> None:
    """清空窗口（压测脚本跑前重置、单测隔离用）。"""
    with _lock:
        _buckets.clear()


def record_and_export(endpoint: str, **kwargs) -> None:
    """记一个样本并立刻镜像到 Redis。

    埋点调用点都用这个：一次查询只多一次 Redis SET，换来的是
    "MQ worker 里的样本也能被 /metrics 读到"，值得。
    """
    record(endpoint, **kwargs)
    export_to_bus()
