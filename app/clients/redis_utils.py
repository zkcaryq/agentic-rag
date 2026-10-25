"""Redis 客户端与 key 约定。

软路由上的 Redis 是 ACL 模式，应用账号 `zhanggui` 只有 `~rag:*` 权限，
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
