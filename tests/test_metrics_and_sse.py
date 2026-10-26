"""TTFT 指标窗口与 SSE 跨进程事件通道的单测。

不依赖 Redis / MQ / 网络：Redis 通道用 monkeypatch 替掉真实客户端，
CI 里可直接跑。
"""
import pytest

from app.core import metrics
from app.utils import sse_utils


@pytest.fixture(autouse=True)
def clean_window():
    metrics.reset()
    yield
    metrics.reset()


# ==================== 滑动窗口聚合 ====================
def test_record_and_snapshot():
    metrics.record("answer_generate", ttft_ms=100.0, total_ms=500.0, tokens=50)
    metrics.record("answer_generate", ttft_ms=300.0, total_ms=900.0, tokens=120)
    snap = metrics.snapshot()["answer_generate"]
    assert snap["count"] == 2
    assert snap["ttft_count"] == 2
    assert snap["ttft_p50"] == 200.0  # (100+300)/2
    assert snap["total_p95"] == 880.0  # 2 样本插值: 500+(900-500)*0.95
    assert snap["avg_tokens"] == 85.0


def test_percentile_interpolation():
    for value in [10.0, 20.0, 30.0, 40.0]:
        metrics.record("query", total_ms=value)
    snap = metrics.snapshot()["query"]
    assert snap["count"] == 4
    # 4 样本：p50 → k=1.5 → 20+(30-20)*0.5
    assert snap["total_p50"] == 25.0
    # p99 → k=2.97 → 30+(40-30)*0.97
    assert snap["total_p99"] == 39.7


def test_samples_without_ttft_are_not_counted_as_ttft():
    # 非流式请求没有首 Token，不能把它们的 total 混进 TTFT 分位数
    metrics.record("query", total_ms=1000.0)
    metrics.record("query", total_ms=2000.0)
    snap = metrics.snapshot()["query"]
    assert snap["ttft_count"] == 0
    assert snap["ttft_p50"] == 0.0
    assert snap["total_p50"] == 1500.0


def test_disabled_records_nothing(monkeypatch):
    monkeypatch.setattr(metrics.infra_config_metrics, "enabled", False)
    metrics.record("query", total_ms=1.0)
    assert metrics.snapshot() == {}


def test_window_size_caps_bucket(monkeypatch):
    monkeypatch.setattr(metrics.infra_config_metrics, "window_size", 3)
    for value in range(10):
        metrics.record("query", total_ms=float(value))
    assert metrics.snapshot()["query"]["count"] == 3


def test_snapshot_merges_remote_samples():
    """多进程合并：worker 镜像回来的样本要和本进程窗口一起算分位数。"""
    metrics.record("query", total_ms=100.0)
    raw = metrics.dump_samples()
    metrics.reset()

    metrics.record("query", total_ms=300.0)
    merged = metrics.snapshot(extra=metrics.load_samples(raw))
    assert merged["query"]["count"] == 2
    assert merged["query"]["total_p50"] == 200.0


def test_dump_load_roundtrip_keeps_ttft_and_tokens():
    metrics.record("answer_generate", ttft_ms=120.5, total_ms=800.0, tokens=42)
    restored = metrics.load_samples(metrics.dump_samples())
    sample = restored["answer_generate"][0]
    assert (sample.ttft_ms, sample.total_ms, sample.tokens) == (120.5, 800.0, 42)


def test_record_and_export_without_redis_is_safe(monkeypatch):
    """没开 Redis 时导出只是 no-op，不能把埋点本身炸掉（单测环境必须能裸跑）。"""
    from app.clients import redis_utils

    monkeypatch.setattr(metrics.infra_config_metrics, "enabled", True)
    monkeypatch.setattr(redis_utils, "is_enabled", lambda: False)
    monkeypatch.setattr(redis_utils, "store_metrics", lambda *a: False)
    metrics.record_and_export("query", total_ms=50.0)
    assert metrics.snapshot()["query"]["count"] == 1


def test_export_sends_samples_to_bus(monkeypatch):
    """导出送出去的必须是本窗口的样本，否则 /metrics 合并不到东西。"""
    from app.clients import redis_utils

    sent = {}
    monkeypatch.setattr(metrics.infra_config_metrics, "enabled", True)
    monkeypatch.setattr(redis_utils, "is_enabled", lambda: True)
    monkeypatch.setattr(
        redis_utils, "store_metrics", lambda pid, raw: sent.update(pid=pid, raw=raw) or True
    )
    metrics.record_and_export("answer_generate", ttft_ms=10.0, total_ms=90.0, tokens=5)
    assert sent["raw"]["answer_generate"] == [[10.0, 90.0, 5]]


# ==================== SSE 传输通道 ====================
def test_push_goes_to_local_queue_by_default():
    session_id = "s-local"
    queue = sse_utils.create_sse_queue(session_id)
    sse_utils.push_to_session(session_id, sse_utils.SSEEvent.DELTA, {"delta": "x"})
    assert queue.get_nowait()["event"] == "delta"
    sse_utils.remove_sse_queue(session_id)


def test_remote_session_uses_redis_bus(monkeypatch):
    # worker 进程里没有本地队列，被标成 remote 的会话必须写 Redis，否则事件全丢
    written = []
    monkeypatch.setattr(
        sse_utils.redis_utils,
        "publish_event",
        lambda sid, event, data: written.append((sid, event, data)) or True,
    )
    session_id = "s-remote"
    sse_utils.mark_remote_transport(session_id)
    sse_utils.push_to_session(session_id, sse_utils.SSEEvent.FINAL, {"answer": "a"})
    assert written == [(session_id, "final", {"answer": "a"})]
    sse_utils.remove_sse_queue(session_id)


def test_remote_session_ignores_local_queue(monkeypatch):
    monkeypatch.setattr(sse_utils.redis_utils, "publish_event", lambda *a: True)
    session_id = "s-both"
    queue = sse_utils.create_sse_queue(session_id)
    sse_utils.mark_remote_transport(session_id)
    sse_utils.push_to_session(session_id, sse_utils.SSEEvent.PROGRESS, {"status": "processing"})
    assert queue.empty()
    sse_utils.remove_sse_queue(session_id)


def test_local_push_without_queue_is_dropped_quietly():
    # 不该抛异常：客户端已经断开时队列会被 finally 清掉
    sse_utils.push_to_session("s-missing", sse_utils.SSEEvent.DELTA, {"delta": "x"})


def test_remove_sse_queue_clears_transport():
    session_id = "s-cleanup"
    sse_utils.create_sse_queue(session_id)
    sse_utils.mark_remote_transport(session_id)
    sse_utils.remove_sse_queue(session_id)
    assert sse_utils.get_sse_queue(session_id) is None
    assert sse_utils.is_remote_transport(session_id) is False


def test_create_sse_queue_is_idempotent():
    session_id = "s-idem"
    first = sse_utils.create_sse_queue(session_id)
    second = sse_utils.create_sse_queue(session_id)
    assert first is second
    sse_utils.remove_sse_queue(session_id)
