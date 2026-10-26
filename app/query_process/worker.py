"""查询 worker：消费 zhanggui.query.task 队列。

启动（仓库根目录，需 RABBITMQ_ENABLED=true 且 REDIS_ENABLED=true）：
    .venv/Scripts/python.exe app/query_process/worker.py

拓扑见 app/clients/mq_utils.declare_topology：
  zhanggui.query -> zhanggui.query.task

为什么要 Redis 事件总线：查询图跑在本进程，EventSource 却连在 API 进程，
progress/delta/final 事件必须跨进程送达，所以这里把会话标成 remote 传输，
push_to_session 会写进 rag:sse:{session_id}:events 列表，由 API 进程 BLPOP 转发。

失败不重投：一次问答失败重试的语义对用户没意义（用户会重新问），
所以错误通过 SSE error 事件回给前端后直接 ack，避免毒消息打转。
"""
import json
import os
import sys
import time

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
    mark_remote_transport,
    push_to_session,
)
from app.utils.task_utils import set_task_result, update_task_status

QUERY_ENDPOINT = "query"


def execute_query(session_id: str, query_text: str, is_stream: bool = True) -> dict:
    """执行一次查询并推送事件，返回 {status, answer}。"""
    # 必须在图开始跑之前标记，否则第一个 progress 事件会落进本进程的内存队列
    mark_remote_transport(session_id)
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


def _on_message(channel, method, properties, body) -> None:
    try:
        payload = json.loads(body.decode("utf-8"))
    except Exception as e:
        logger.error(f"查询消息解析失败，丢弃: {e}")
        channel.basic_ack(delivery_tag=method.delivery_tag)
        return

    session_id = payload.get("session_id")
    query_text = payload.get("query", "")
    if not session_id or not query_text:
        logger.warning(f"查询消息缺少 session_id/query，丢弃: {payload}")
        channel.basic_ack(delivery_tag=method.delivery_tag)
        return

    logger.info(f"[{session_id}] worker 收到查询任务")
    execute_query(
        session_id=session_id,
        query_text=query_text,
        is_stream=bool(payload.get("is_stream", True)),
    )
    channel.basic_ack(delivery_tag=method.delivery_tag)


def main() -> int:
    if not mq_utils.is_enabled():
        logger.error("RABBITMQ_ENABLED=false 或缺少 RABBITMQ_URL，worker 无法启动")
        return 1
    if not redis_utils.is_enabled():
        logger.error("REDIS_ENABLED=false：事件无法回传 API 进程，worker 拒绝启动")
        return 1

    connection = mq_utils.get_connection()
    channel = connection.channel()
    mq_utils.declare_topology(channel)
    channel.basic_qos(prefetch_count=mq_conf.prefetch_count)
    channel.basic_consume(queue=mq_conf.query_queue, on_message_callback=_on_message)

    logger.info(f"查询 worker 已启动，监听队列 {mq_conf.query_queue}")
    try:
        channel.start_consuming()
    except KeyboardInterrupt:
        logger.info("查询 worker 收到中断信号，停止消费")
        channel.stop_consuming()
    finally:
        connection.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
