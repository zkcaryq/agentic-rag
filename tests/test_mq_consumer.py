"""worker 消费循环的回归用例：长任务不能在回调里同步跑完再 ack。

真实事故：pika heartbeat=60，而导入图要跑几十分钟，回调阻塞期间主线程不发心跳，
服务端判连接死亡后把未 ack 的消息重投给别的 consumer——同一个 task_id 被两个
worker 各跑一遍。这里的假连接复现的就是"心跳有没有被持续驱动"这件事。
"""
from concurrent.futures import Future

from app.clients.mq_utils import consume_with_heartbeat


class FakeChannel:
    def __init__(self):
        self.actions = []
        self.callback = None

    def basic_consume(self, queue, on_message_callback):
        self.callback = on_message_callback

    def basic_ack(self, delivery_tag):
        self.actions.append(("ack", delivery_tag))

    def basic_nack(self, delivery_tag, requeue):
        self.actions.append(("nack", delivery_tag, requeue))

    def stop_consuming(self):
        self.actions.append(("stop", None))

    def settled(self):
        """只看 ack/nack，忽略退出时的 stop。"""
        return [a for a in self.actions if a[0] != "stop"]


class FakeConnection:
    """每驱动一次 I/O 就执行一帧；帧跑完抛 KeyboardInterrupt 结束循环。"""

    def __init__(self, frames):
        self.frames = list(frames)
        self.io_driven = 0

    def process_data_events(self, time_limit=None):
        self.io_driven += 1
        if self.frames:
            self.frames.pop(0)()
            return
        raise KeyboardInterrupt


def _deliver(channel, body: bytes, tag=1):
    method = type("Method", (), {"delivery_tag": tag})()
    channel.callback(channel, method, None, body)


def test_no_ack_while_the_task_is_still_running():
    """投递那一帧绝不能 ack——同步 ack 就是旧写法，它会阻塞心跳导致消息被重投。"""
    channel = FakeChannel()
    future = Future()
    observed = []

    connection = FakeConnection(
        frames=[
            lambda: _deliver(channel, b'{"task_id": "t1"}'),
            lambda: observed.append(channel.settled()),   # 任务未完成的时刻
            lambda: future.set_result({"status": "completed"}),
        ]
    )
    consume_with_heartbeat(
        connection,
        channel,
        "arag.import.task",
        submit=lambda payload: future,
        settle=lambda f, p, ch, tag: ch.basic_ack(delivery_tag=tag),
    )

    assert observed == [[]]
    assert channel.settled() == [("ack", 1)]
    assert connection.io_driven >= 3          # 主线程一直在驱动 I/O（心跳）


def test_parse_failure_goes_to_dead_letter_without_submitting():
    channel = FakeChannel()
    executed = []
    connection = FakeConnection(
        frames=[lambda: _deliver(channel, b"not-json", tag=7), lambda: executed.append(1)]
    )

    consume_with_heartbeat(
        connection,
        channel,
        "arag.import.task",
        submit=lambda payload: executed.append("submit"),
        settle=lambda *a: None,
    )

    assert channel.settled() == [("nack", 7, False)]
    assert executed == [1]                    # 坏消息不该被执行


def test_submit_failure_nacks_so_poison_message_cannot_loop():
    channel = FakeChannel()

    def boom(payload):
        raise ValueError("missing session_id")

    connection = FakeConnection(
        frames=[lambda: _deliver(channel, b'{"session_id": ""}', tag=3), lambda: None]
    )
    consume_with_heartbeat(
        connection, channel, "arag.query.task", submit=boom, settle=lambda *a: None
    )

    assert channel.settled() == [("nack", 3, False)]


def test_settle_failure_does_not_break_the_loop():
    channel = FakeChannel()
    future = Future()
    future.set_result({"status": "completed"})
    settled = []

    def angry_settle(f, payload, ch, tag):
        settled.append(tag)
        raise RuntimeError("connection closed")

    connection = FakeConnection(
        frames=[
            lambda: _deliver(channel, b'{"task_id": "t9"}'),
            lambda: None,
            lambda: None,
        ]
    )
    # 结算抛错必须被吞掉并继续循环：循环一崩就不再驱动 I/O，心跳断供，
    # 而这正是本用例要防的那个缺陷的形状
    consume_with_heartbeat(
        connection,
        channel,
        "arag.import.task",
        submit=lambda payload: future,
        settle=angry_settle,
    )

    assert settled == [1]
    assert channel.settled() == []
    assert connection.io_driven >= 3
