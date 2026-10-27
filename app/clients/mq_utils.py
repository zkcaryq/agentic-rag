"""RabbitMQ 客户端（pika，同步，与项目风格一致）。

交换机 / 队列 / routing key 全部以 arag. 开头，匹配软路由上该账号的
^arag\\..* 权限；导入与查询分属不同队列，就是简历里说的「任务资源隔离」。
"""
import json
from typing import Any, Dict

from app.conf.infra_config import infra_config_mq as mq_conf
from app.core.logger import logger


def is_enabled() -> bool:
    return bool(mq_conf.enabled and mq_conf.url)


def _parameters():
    import pika

    params = pika.URLParameters(mq_conf.url)
    params.heartbeat = 60
    params.blocked_connection_timeout = 30
    return params


def get_connection():
    import pika

    return pika.BlockingConnection(_parameters())


def declare_topology(channel) -> None:
    """幂等声明交换机与队列（含导入死信队列）。"""
    # 导入主链路
    channel.exchange_declare(
        exchange=mq_conf.import_exchange, exchange_type="direct", durable=True
    )
    channel.exchange_declare(
        exchange=mq_conf.import_dlx, exchange_type="direct", durable=True
    )
    channel.queue_declare(
        queue=mq_conf.import_queue,
        durable=True,
        arguments={
            "x-dead-letter-exchange": mq_conf.import_dlx,
            "x-dead-letter-routing-key": f"{mq_conf.import_queue}.dead",
        },
    )
    channel.queue_bind(
        queue=mq_conf.import_queue,
        exchange=mq_conf.import_exchange,
        routing_key=mq_conf.import_routing_key,
    )
    channel.queue_declare(queue=mq_conf.import_dlq, durable=True)
    channel.queue_bind(
        queue=mq_conf.import_dlq,
        exchange=mq_conf.import_dlx,
        routing_key=f"{mq_conf.import_queue}.dead",
    )

    # 查询链路（与导入隔离，互不堆积）
    channel.exchange_declare(
        exchange=mq_conf.query_exchange, exchange_type="direct", durable=True
    )
    channel.queue_declare(queue=mq_conf.query_queue, durable=True)
    channel.queue_bind(
        queue=mq_conf.query_queue,
        exchange=mq_conf.query_exchange,
        routing_key=mq_conf.query_routing_key,
    )


def _publish(exchange: str, routing_key: str, payload: Dict[str, Any], label: str) -> bool:
    """投递一条持久化 JSON 消息。失败返回 False，由调用方决定退回进程内执行。"""
    import pika

    if not is_enabled():
        return False
    try:
        connection = get_connection()
        try:
            channel = connection.channel()
            declare_topology(channel)
            channel.basic_publish(
                exchange=exchange,
                routing_key=routing_key,
                body=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
                properties=pika.BasicProperties(
                    delivery_mode=2,  # 持久化，broker 重启不丢
                    content_type="application/json",
                ),
            )
        finally:
            connection.close()
        logger.info(f"{label}已投递: {payload.get('task_id') or payload.get('session_id')}")
        return True
    except Exception as e:
        logger.exception(f"投递{label}失败: {e}")
        return False


def publish_import_task(payload: Dict[str, Any]) -> bool:
    """投递导入任务到 arag.import.task。"""
    return _publish(
        mq_conf.import_exchange, mq_conf.import_routing_key, payload, "导入任务"
    )


def publish_query_task(payload: Dict[str, Any]) -> bool:
    """投递查询任务到 arag.query.task。

    查询走独立队列，和导入互不堆积：批量灌文档时问答不会被排在几千个导入任务后面，
    这就是简历里的「任务资源隔离」。
    """
    return _publish(
        mq_conf.query_exchange, mq_conf.query_routing_key, payload, "查询任务"
    )


def consume_with_heartbeat(
    connection,
    channel,
    queue: str,
    submit,
    settle,
    *,
    poll_seconds: float = 1.0,
) -> None:
    """长任务消费循环：任务丢线程池，主线程只驱动 AMQP I/O。

    为什么不能直接在回调里跑完再 ack：pika 的心跳只在 I/O 循环里发，而导入图要跑
    几十分钟、一次问答也要十几到几十秒，都超过 heartbeat=60 的两拍。服务端连续两次
    收不到心跳就判定连接死亡，把未 ack 的消息重投给别的 consumer——实测同一个
    task_id 被两个 worker 各跑一遍，MinerU 白解析一次、图片摘要白调一次 VLM。
    所以：回调只做「解析 + 提交」，ack/nack 由主线程在任务结束后结算。

    submit(payload) -> Future；settle(future, payload, channel, delivery_tag)。
    """
    active: list = []

    def _on_message(ch, method, _properties, body) -> None:
        try:
            payload = json.loads(body.decode("utf-8"))
        except Exception as e:
            logger.error(f"[{queue}] 消息解析失败，进死信: {e}")
            ch.basic_nack(delivery_tag=method.delivery_tag, requeue=False)
            return
        try:
            future = submit(payload)
        except Exception as e:
            logger.exception(f"[{queue}] 任务提交失败: {e}")
            ch.basic_nack(delivery_tag=method.delivery_tag, requeue=False)
            return
        active.append((future, payload, method.delivery_tag))

    channel.basic_consume(queue=queue, on_message_callback=_on_message)
    try:
        while True:
            connection.process_data_events(time_limit=poll_seconds)
            for item in list(active):
                future, payload, delivery_tag = item
                if not future.done():
                    continue
                active.remove(item)
                try:
                    settle(future, payload, channel, delivery_tag)
                except Exception as e:
                    logger.exception(f"[{queue}] 结算失败: {e}")
    except KeyboardInterrupt:
        logger.info(f"[{queue}] 收到中断信号，停止消费（未 ack 的消息由服务端重投）")
        channel.stop_consuming()
