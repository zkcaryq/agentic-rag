"""Redis 客户端与 key 约定。

软路由上的 Redis 是 ACL 模式，应用账号 `arag` 只有 `~rag:*` 权限，
所以所有 key 必须由 key() 统一加前缀，业务代码不要自己拼字符串。
"""
import threading

from app.conf.infra_config import REDIS_KEY_PREFIX
from app.conf.infra_config import infra_config_redis as redis_conf
from app.core.logger import logger

_client = None
_lock = threading.Lock()


def key(*parts: str) -> str:
    """拼一个带 ACL 前缀的 key，例如 key("task", tid, "done")。"""
    return REDIS_KEY_PREFIX + ":".join(p for p in parts if p)


def is_enabled() -> bool:
    return bool(redis_conf.enabled and redis_conf.url)


def get_redis():
    """Redis 单例；未启用或连不上时返回 None，调用方走进程内兜底。"""
    global _client
    if not is_enabled():
        return None
    if _client is not None:
        return _client

    with _lock:
        if _client is not None:
            return _client
        try:
            import redis

            client = redis.from_url(
                redis_conf.url,
                decode_responses=True,
                socket_connect_timeout=3,
                socket_timeout=3,
            )
            client.ping()
            _client = client
            logger.info("Redis 连接成功")
        except Exception as e:
            logger.error(f"Redis 连接失败，任务状态将退回进程内存: {e}")
            _client = None
    return _client


def ttl_seconds() -> int:
    return redis_conf.default_ttl_seconds


# ---------------------------------------------------------------------------
# SSE 跨进程事件通道
#
# 查询任务被投递到 RabbitMQ 后，执行图跑在 worker 进程里，而 EventSource 连的是
# API 进程的内存队列——两边看不见彼此。所以 worker 把事件 RPUSH 进 Redis 列表，
# API 进程 BLPOP 出来转发给前端。用列表而不是 pubsub：ACL 账号只有 ~rag:* 的
# key 权限，没申请频道权限，列表在现有授权下就能用。
# ---------------------------------------------------------------------------


def sse_events_key(session_id: str) -> str:
    return key("sse", session_id, "events")


def publish_event(session_id: str, event: str, data: dict) -> bool:
    """写一条 SSE 事件到 Redis；未启用或写失败返回 False。"""
    import json

    client = get_redis()
    if client is None:
        return False
    events_key = sse_events_key(session_id)
    try:
        client.rpush(events_key, json.dumps({"event": event, "data": data}, ensure_ascii=False))
        client.expire(events_key, ttl_seconds())
        return True
    except Exception as e:
        logger.warning(f"写入 SSE 事件失败 {session_id}/{event}: {e}")
        return False


def pop_event(session_id: str, timeout: float = 1.0) -> dict | None:
    """从 Redis 取一条事件（阻塞至多 timeout 秒），没有返回 None。

    走独立客户端：BLPOP 会占住连接直到超时，用共享单例去阻塞会把别的
    调用（任务状态、检查点）一起拖住。
    """
    client = get_blocking_client()
    if client is None:
        return None
    try:
        popped = client.blpop(sse_events_key(session_id), timeout=timeout)
    except Exception as e:
        logger.warning(f"读取 SSE 事件失败 {session_id}: {e}")
        return None
    if not popped:
        return None
    import json

    try:
        return json.loads(popped[1])
    except Exception as e:
        logger.warning(f"SSE 事件反序列化失败 {session_id}: {e}")
        return None


def clear_events(session_id: str) -> None:
    client = get_redis()
    if client is None:
        return
    try:
        client.delete(sse_events_key(session_id))
    except Exception as e:
        logger.warning(f"清理 SSE 事件失败 {session_id}: {e}")


# ---------------------------------------------------------------------------
# 跨进程指标镜像（rag:metrics:{pid}）
#
# TTFT 窗口活在各自进程里：查询走 MQ 时样本记在 worker，API 进程的 /metrics
# 读不到。worker 每次做完查询把自己的样本 dump 进来，接口读取时合并所有进程。
# ---------------------------------------------------------------------------


def store_metrics(pid: int, raw: dict) -> bool:
    client = get_redis()
    if client is None:
        return False
    import json

    try:
        client.set(key("metrics", str(pid)), json.dumps(raw, ensure_ascii=False), ex=ttl_seconds())
        return True
    except Exception as e:
        logger.warning(f"写入指标快照失败: {e}")
        return False


def load_all_metrics(exclude_pid: int | None = None) -> dict:
    """读回所有进程 dump 出来的样本，合并成 {endpoint: [样本...]}。

    exclude_pid：本进程自己的镜像 key 要跳过，否则本地窗口 + Redis 会重复计数。
    """
    client = get_redis()
    if client is None:
        return {}
    import json

    skip = key("metrics", str(exclude_pid)) if exclude_pid else None
    merged: dict[str, list] = {}
    try:
        for metric_key in client.scan_iter(match=key("metrics", "*"), count=100):
            if skip and metric_key == skip:
                continue
            payload = client.get(metric_key)
            if not payload:
                continue
            for endpoint, rows in json.loads(payload).items():
                merged.setdefault(endpoint, []).extend(rows)
    except Exception as e:
        logger.warning(f"读取指标快照失败: {e}")
    return merged


_blocking_client = None


def get_blocking_client():
    """只给 BLPOP 用的连接，socket_timeout 必须为 None，否则超时会被客户端先掐断。"""
    global _blocking_client
    if not is_enabled():
        return None
    if _blocking_client is not None:
        return _blocking_client

    with _lock:
        if _blocking_client is not None:
            return _blocking_client
        try:
            import redis

            client = redis.from_url(
                redis_conf.url,
                decode_responses=True,
                socket_connect_timeout=3,
                socket_timeout=None,
                health_check_interval=30,
            )
            client.ping()
            _blocking_client = client
        except Exception as e:
            logger.error(f"Redis 阻塞连接创建失败，跨进程 SSE 不可用: {e}")
            _blocking_client = None
    return _blocking_client
