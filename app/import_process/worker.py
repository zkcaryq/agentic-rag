"""导入 worker：消费 zhanggui.import.task 队列。

启动（仓库根目录）：
    .venv/Scripts/python.exe app/import_process/worker.py

队列拓扑见 app/clients/mq_utils.declare_topology：
  zhanggui.import  -> zhanggui.import.task
  zhanggui.import.dlx -> zhanggui.import.dlq（处理失败的消息进死信）
"""
import json
import os
import sys

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


def _on_message(channel, method, properties, body) -> None:
    try:
        payload = json.loads(body.decode("utf-8"))
    except Exception as e:
        logger.error(f"消息解析失败，直接丢进死信: {e}")
        channel.basic_nack(delivery_tag=method.delivery_tag, requeue=False)
        return

    task_id = payload.get("task_id", "?")
    logger.info(f"[{task_id}] worker 收到导入任务")

    kwargs = {key: payload[key] for key in _PAYLOAD_KEYS if key in payload}
    try:
        result = execute_import(**kwargs)
    except Exception as e:  # execute_import 内部已兜底，这里防御编程
        logger.error(f"[{task_id}] worker 执行异常: {e}", exc_info=True)
        channel.basic_nack(delivery_tag=method.delivery_tag, requeue=False)
        return

    if result.get("status") == "failed":
        # 进死信，避免毒消息无限重投把队列堵死
        channel.basic_nack(delivery_tag=method.delivery_tag, requeue=False)
    else:
        channel.basic_ack(delivery_tag=method.delivery_tag)


def main() -> int:
    if not mq_utils.is_enabled():
        logger.error("RABBITMQ_ENABLED=false 或缺少 RABBITMQ_URL，worker 无法启动")
        return 1

    connection = mq_utils.get_connection()
    channel = connection.channel()
    mq_utils.declare_topology(channel)
    channel.basic_qos(prefetch_count=mq_conf.prefetch_count)
    channel.basic_consume(
        queue=mq_conf.import_queue, on_message_callback=_on_message
    )

    logger.info(f"worker 已启动，监听队列 {mq_conf.import_queue}")
    try:
        channel.start_consuming()
    except KeyboardInterrupt:
        logger.info("worker 收到中断信号，停止消费")
        channel.stop_consuming()
    finally:
        connection.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
