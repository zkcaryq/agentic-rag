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
