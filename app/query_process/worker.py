"""查询 worker：消费 arag.query.task 队列。

启动（仓库根目录，需 RABBITMQ_ENABLED=true 且 REDIS_ENABLED=true）：
    .venv/Scripts/python.exe app/query_process/worker.py

拓扑见 app/clients/mq_utils.declare_topology：
  arag.query -> arag.query.task

为什么要 Redis 事件总线：查询图跑在本进程，EventSource 却连在 API 进程，
progress/delta/final 事件必须跨进程送达，所以这里把会话标成 remote 传输，
push_to_session 会写进 rag:sse:{session_id}:events 列表，由 API 进程 BLPOP 转发。

任务跑在子线程、ack 在主线程结算，原因见 mq_utils.consume_with_heartbeat。
"""
import os
import sys
import time
from concurrent.futures import Future, ThreadPoolExecutor

sys.path.insert(
    0,
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)

from app.clients import mq_utils
from app.clients import redis_utils
from app.conf.infra_config import infra_config_mq as mq_conf
from app.core import metrics
from app.core.logger import logger
from app.query_process.agent.main_graph import query_app
from app.query_process.agent.state import create_default_query_state
from app.utils.sse_utils import (
    SSEEvent,
    bind_stream,
    mark_remote_transport,
    push_to_session,
)
from app.utils.task_utils import set_task_result, update_task_status

QUERY_ENDPOINT = "query"


def execute_query(
    session_id: str,
    query_text: str,
    is_stream: bool = True,
    stream_id: str = "",
) -> dict:
    """执行一次查询并推送事件，返回 {status, answer}。"""
    # 必须在图开始跑之前绑定通道并标记 remote：事件按"这次提问的通道"投递，
    # 否则同一会话的上一问残留事件会被下一问的连接读走
    if stream_id:
        bind_stream(session_id, stream_id)
    mark_remote_transport(stream_id or session_id)
    update_task_status(session_id, "processing", push_queue=True)

    state = create_default_query_state(
        session_id=session_id,
        original_query=query_text,
        is_stream=is_stream,
    )
    started = time.monotonic()
    try:
        final_state = query_app.invoke(state)
        metrics.record_and_export(QUERY_ENDPOINT, total_ms=(time.monotonic() - started) * 1000)
        update_task_status(session_id, "completed", push_queue=True)
        return {"status": "completed", "answer": final_state.get("answer", "")}
    except Exception as e:
        metrics.record_and_export(QUERY_ENDPOINT, total_ms=(time.monotonic() - started) * 1000)
        logger.exception(f"[{session_id}] worker 执行查询失败: {e}")
        set_task_result(session_id, "error", str(e))
        update_task_status(session_id, "failed", push_queue=True)
        push_to_session(session_id, SSEEvent.ERROR, {"message": "处理失败，请稍后重试"})
        return {"status": "failed", "answer": ""}
    finally:
        push_to_session(session_id, SSEEvent.CLOSE, {})


def _submit(pool: ThreadPoolExecutor):
    def submit(payload: dict) -> Future:
        session_id = payload.get("session_id")
        query_text = payload.get("query", "")
        if not session_id or not query_text:
            raise ValueError(f"查询消息缺少 session_id/query: {payload}")

        logger.info(f"[{session_id}] worker 收到查询任务，交给执行线程")
        return pool.submit(
            execute_query,
            session_id=session_id,
            query_text=query_text,
            is_stream=bool(payload.get("is_stream", True)),
            stream_id=payload.get("stream_id", ""),
        )

    return submit


def _settle(future: Future, payload: dict, channel, delivery_tag) -> None:
    """查询失败不重投：重试一次问答对用户没意义（他会重新问），错误已经通过 SSE
    error 事件回给前端，所以一律 ack，避免毒消息在队列里打转。"""
    session_id = payload.get("session_id", "?")
    try:
        result = future.result()
        status = result.get("status")
    except Exception as e:
        logger.exception(f"[{session_id}] worker 执行查询异常: {e}")
        status = "error"
    channel.basic_ack(delivery_tag=delivery_tag)
    logger.info(f"[{session_id}] 查询任务结算: {status}")


def _serve_once() -> None:
    workers = max(int(mq_conf.prefetch_count or 1), 1)
    connection = mq_utils.get_connection()
    channel = connection.channel()
    mq_utils.declare_topology(channel)
    channel.basic_qos(prefetch_count=workers)

    logger.info(f"查询 worker 已启动，监听队列 {mq_conf.query_queue}（并发 {workers}）")
    pool = ThreadPoolExecutor(max_workers=workers, thread_name_prefix="query")
    try:
        mq_utils.consume_with_heartbeat(
            connection,
            channel,
            mq_conf.query_queue,
            _submit(pool),
            _settle,
        )
    finally:
        pool.shutdown(wait=False)
        mq_utils.safe_close(connection)


def main() -> int:
    if not mq_utils.is_enabled():
        logger.error("RABBITMQ_ENABLED=false 或缺少 RABBITMQ_URL，worker 无法启动")
        return 1
    if not redis_utils.is_enabled():
        logger.error("REDIS_ENABLED=false：事件无法回传 API 进程，worker 拒绝启动")
        return 1
    return mq_utils.run_with_reconnect(_serve_once)


if __name__ == "__main__":
    raise SystemExit(main())
