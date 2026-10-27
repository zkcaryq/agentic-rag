"""SSE 事件推送。

两种传输通道：
  - 进程内队列（默认）：查询图和执行它的 FastAPI 进程在同一个进程里；
  - Redis 列表（跨进程）：查询任务被 RabbitMQ worker 执行时，事件写进 Redis，
    API 进程读出来转发给前端。会话被标记为 remote 就走这条通道。

事件类型见 SSEEvent：ready 由 generator 建立连接时发，progress/delta/final/error
由业务节点推，__close__ 是收尾信号。
"""
import asyncio
import json
import queue
from typing import Any, AsyncGenerator, Dict, Optional

from fastapi import Request

from app.clients import redis_utils
from app.core.logger import logger


class SSEEvent:
    READY = "ready"         # 连接建立
    PROGRESS = "progress"   # 任务节点进度
    DELTA = "delta"         # LLM 流式输出增量
    FINAL = "final"         # 最终完整答案
    ERROR = "error"         # 错误信息
    CLOSE = "__close__"     # 关闭连接信号


# 全局 SSE 会话队列存储
# Key: 事件通道 id, Value: queue.Queue
_session_stream: Dict[str, queue.Queue] = {}

# 传输通道：channel_id -> "remote" 表示事件由别的进程生产，读 Redis
_transport: Dict[str, str] = {}

# session_id -> 本次提问的事件通道。一次提问一个通道：
# 通道若按 session 复用，上一问残留的 __close__ 会被下一问的连接读到，
# 表现是"七个节点全绿却显示服务出现问题"（实测踩过，见 docs/RESUME-ALIGNMENT.md）
_stream_of: Dict[str, str] = {}


def bind_stream(session_id: str, stream_id: str) -> str:
    """把 session 绑定到本次提问的通道 id，并返回通道 id。"""
    _stream_of[session_id] = stream_id
    return stream_id


def channel_of(key: str) -> str:
    """session_id 或通道 id -> 真正的事件通道。"""
    return _stream_of.get(key, key)


def get_sse_queue(session_id: str) -> Optional["queue.Queue"]:
    """获取指定通道的队列（传 session_id 时自动解析到当前通道）"""
    return _session_stream.get(channel_of(session_id))


def create_sse_queue(session_id: str) -> "queue.Queue":
    """创建并注册一个新的 SSE 队列。

    幂等：已存在则直接复用。前端可能先连 /stream 再发 /query，
    如果这里覆盖字典条目，generator 会继续读旧队列导致事件丢失。
    """
    channel = channel_of(session_id)
    existing = _session_stream.get(channel)
    if existing is not None:
        return existing
    q = queue.Queue()
    _session_stream[channel] = q
    return q


def mark_remote_transport(session_id: str) -> None:
    """标记该通道的事件由其他进程（MQ worker）生产，走 Redis 通道。"""
    _transport[channel_of(session_id)] = "remote"


def is_remote_transport(session_id: str) -> bool:
    return _transport.get(channel_of(session_id)) == "remote"


def remove_sse_queue(session_id: str):
    """移除指定通道的队列与通道标记"""
    channel = channel_of(session_id)
    _session_stream.pop(channel, None)
    _transport.pop(channel, None)


def _sse_pack(event: str, data: Dict[str, Any]) -> str:
    """打包 SSE 消息格式"""
    payload = json.dumps(data, ensure_ascii=False)
    return f"event: {event}\ndata: {payload}\n\n"


def push_to_session(session_id: str, event: str, data: Dict[str, Any]):
    """通过 session_id 推送事件（自动解析到本次提问的通道）。

    被标记为 remote 的通道写 Redis（本进程的队列对它没有意义），
    其余走进程内队列。
    """
    channel = channel_of(session_id)
    if is_remote_transport(channel):
        if redis_utils.publish_event(channel, event, data):
            return
        logger.warning(f"[SSE] {channel} 事件写入 Redis 失败: {event}")
        return

    stream_queue = _session_stream.get(channel)
    if stream_queue:
        stream_queue.put({"event": event, "data": data})
    else:
        logger.warning(f"[SSE] 推送 {event} 时找不到 {channel} 的队列")


async def sse_generator(session_id: str, request: Request):
    """SSE 生成器，用于 FastAPI 的 StreamingResponse。

    入参既可能是本次提问的通道 id，也可能是老的 session_id（兼容直接订阅会话）；
    统一解析成通道后再读写，避免读到上一问残留的事件。
    """
    channel = channel_of(session_id)
    stream_queue = _session_stream.get(channel)
    if stream_queue is None and not is_remote_transport(channel):
        # 没有本地队列且不是跨进程通道，确实无源可读，直接结束
        logger.warning(f"[SSE] {channel} 没有可用队列，结束推流")
        return

    loop = asyncio.get_running_loop()
    try:
        yield _sse_pack(SSEEvent.READY, {})

        while True:
            if await request.is_disconnected():
                logger.info(f"[SSE] 客户端断开: {channel}")
                break

            if is_remote_transport(channel):
                msg = await loop.run_in_executor(None, redis_utils.pop_event, channel)
                if msg is None:
                    continue
            else:
                try:
                    # 用 run_in_executor 避免同步队列堵住事件循环
                    msg = await loop.run_in_executor(None, stream_queue.get, True, 1.0)
                except queue.Empty:
                    continue

            event = msg.get("event")
            data = msg.get("data")

            # 特殊关闭事件
            if event == SSEEvent.CLOSE:
                logger.info(f"[SSE] {channel} 收到关闭信号")
                break

            yield _sse_pack(event, data)
    except (asyncio.CancelledError, ConnectionResetError, BrokenPipeError):
        # 生成器被取消 / 对端断开：静默退出
        return
    except Exception as e:
        logger.error(f"[SSE] {channel} 推流异常: {e}")
    finally:
        if is_remote_transport(channel):
            redis_utils.clear_events(channel)
        remove_sse_queue(channel)
