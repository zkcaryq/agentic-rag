"""导入 worker：消费 arag.import.task 队列。

启动（仓库根目录）：
    .venv/Scripts/python.exe app/import_process/worker.py

队列拓扑见 app/clients/mq_utils.declare_topology：
  arag.import  -> arag.import.task
  arag.import.dlx -> arag.import.dlq（处理失败的消息进死信）

任务跑在子线程、ack 在主线程结算，原因写在 mq_utils.consume_with_heartbeat 里：
导入图远超 60 秒心跳窗口，在回调里同步跑完会被服务端判死并重投给别的 consumer。
"""
import os
import sys
from concurrent.futures import Future, ThreadPoolExecutor

sys.path.insert(
    0,
    os.path.dirname(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    ),
)

from app.clients import mq_utils
from app.conf.infra_config import infra_config_mq as mq_conf
from app.core.logger import logger
from app.import_process.task_runner import execute_import

_PAYLOAD_KEYS = (
    "task_id",
    "local_dir",
    "local_file_path",
    "file_name",
    "file_type",
    "file_size",
)


def _submit(pool: ThreadPoolExecutor):
    def submit(payload: dict) -> Future:
        kwargs = {key: payload[key] for key in _PAYLOAD_KEYS if key in payload}
        task_id = payload.get("task_id", "?")
        logger.info(f"[{task_id}] worker 收到导入任务，交给执行线程")
        return pool.submit(execute_import, **kwargs)

    return submit


def _settle(future: Future, payload: dict, channel, delivery_tag) -> None:
    task_id = payload.get("task_id", "?")
    try:
        result = future.result()
    except Exception as e:  # execute_import 内部已兜底，这里防御编程
        logger.exception(f"[{task_id}] worker 执行异常: {e}")
        channel.basic_nack(delivery_tag=delivery_tag, requeue=False)
        return

    if result.get("status") == "failed":
        # 进死信，避免毒消息无限重投把队列堵死
        channel.basic_nack(delivery_tag=delivery_tag, requeue=False)
    else:
        channel.basic_ack(delivery_tag=delivery_tag)
    logger.info(f"[{task_id}] 导入任务结算: {result.get('status')}")


def _serve_once() -> None:
    workers = max(int(mq_conf.prefetch_count or 1), 1)
    connection = mq_utils.get_connection()
    channel = connection.channel()
    mq_utils.declare_topology(channel)
    channel.basic_qos(prefetch_count=workers)

    logger.info(f"worker 已启动，监听队列 {mq_conf.import_queue}（并发 {workers}）")
    pool = ThreadPoolExecutor(max_workers=workers, thread_name_prefix="import")
    try:
        mq_utils.consume_with_heartbeat(
            connection,
            channel,
            mq_conf.import_queue,
            _submit(pool),
            _settle,
        )
    finally:
        # 不等待在跑的任务：进程退出时未 ack 的消息由服务端重投，断点快照会让重跑很便宜
        pool.shutdown(wait=False)
        mq_utils.safe_close(connection)


def main() -> int:
    if not mq_utils.is_enabled():
        logger.error("RABBITMQ_ENABLED=false 或缺少 RABBITMQ_URL，worker 无法启动")
        return 1
    return mq_utils.run_with_reconnect(_serve_once)


if __name__ == "__main__":
    raise SystemExit(main())
