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
# Key: session_id, Value: queue.Queue
_session_stream: Dict[str, queue.Queue] = {}

# 会话传输通道：session_id -> "remote" 表示事件由别的进程生产，读 Redis
_transport: Dict[str, str] = {}


def get_sse_queue(session_id: str) -> Optional["queue.Queue"]:
    """获取指定 session 的队列"""
    return _session_stream.get(session_id)


def create_sse_queue(session_id: str) -> "queue.Queue":
    """创建并注册一个新的 SSE 队列。

    幂等：已存在则直接复用。前端可能先连 /stream 再发 /query，
    如果这里覆盖字典条目，generator 会继续读旧队列导致事件丢失。
    """
    existing = _session_stream.get(session_id)
    if existing is not None:
        return existing
    q = queue.Queue()
    _session_stream[session_id] = q
    return q


def mark_remote_transport(session_id: str) -> None:
    """标记该会话的事件由其他进程（MQ worker）生产，走 Redis 通道。"""
    _transport[session_id] = "remote"


def is_remote_transport(session_id: str) -> bool:
    return _transport.get(session_id) == "remote"


def remove_sse_queue(session_id: str):
    """移除指定 session 的队列与通道标记"""
    _session_stream.pop(session_id, None)
    _transport.pop(session_id, None)


def _sse_pack(event: str, data: Dict[str, Any]) -> str:
    """打包 SSE 消息格式"""
    payload = json.dumps(data, ensure_ascii=False)
    return f"event: {event}\ndata: {payload}\n\n"


def push_to_session(session_id: str, event: str, data: Dict[str, Any]):
    """通过 session_id 推送事件。

    被标记为 remote 的会话写 Redis（本进程的队列对它没有意义），
    其余走进程内队列。
    """
    if is_remote_transport(session_id):
        if redis_utils.publish_event(session_id, event, data):
            return
        logger.warning(f"[SSE] {session_id} 事件写入 Redis 失败: {event}")
        return

    stream_queue = get_sse_queue(session_id)
    if stream_queue:
        stream_queue.put({"event": event, "data": data})
    else:
        logger.warning(f"[SSE] 推送 {event} 时找不到 {session_id} 的队列")


async def sse_generator(session_id: str, request: Request):
    """SSE 生成器，用于 FastAPI 的 StreamingResponse。"""
    stream_queue = get_sse_queue(session_id)
    if stream_queue is None and not is_remote_transport(session_id):
        # 没有本地队列且不是跨进程会话，确实无源可读，直接结束
        logger.warning(f"[SSE] {session_id} 没有可用队列，结束推流")
        return

    loop = asyncio.get_running_loop()
    try:
        yield _sse_pack(SSEEvent.READY, {})

        while True:
            if await request.is_disconnected():
                logger.info(f"[SSE] 客户端断开: {session_id}")
                break

            if is_remote_transport(session_id):
                msg = await loop.run_in_executor(None, redis_utils.pop_event, session_id)
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
                logger.info(f"[SSE] {session_id} 收到关闭信号")
                break

            yield _sse_pack(event, data)
    except (asyncio.CancelledError, ConnectionResetError, BrokenPipeError):
        # 生成器被取消 / 对端断开：静默退出
        return
    except Exception as e:
        logger.error(f"[SSE] {session_id} 推流异常: {e}")
    finally:
        if is_remote_transport(session_id):
            redis_utils.clear_events(session_id)
        remove_sse_queue(session_id)
