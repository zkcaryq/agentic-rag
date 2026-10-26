"""基础设施连接配置：PostgreSQL / Redis / RabbitMQ / LangSmith。

全部走 .env，业务代码不写死任何地址与凭据。
软路由上的 Redis 与 RabbitMQ 都做了前缀隔离：
  - Redis ACL 用户 zhanggui 只有 ~rag:* 权限，所以 key 必须以 REDIS_KEY_PREFIX 开头；
  - RabbitMQ 用户 zhanggui 只有 ^zhanggui\\..* 权限，所以队列/交换机/routing key
    必须以 MQ_PREFIX 开头，否则会因为权限不足被拒。
上层拼名字时统一引用这两个常量，不要在业务代码里各写各的。
"""
from dataclasses import dataclass
import os

from dotenv import load_dotenv

load_dotenv()

REDIS_KEY_PREFIX = "rag:"
MQ_PREFIX = "zhanggui."


def _env_bool(name: str, default: bool = False) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in ("1", "true", "yes", "on")


@dataclass
class PostgresConfig:
    dsn: str
    enabled: bool


@dataclass
class RedisConfig:
    url: str
    enabled: bool
    default_ttl_seconds: int


@dataclass
class RabbitMQConfig:
    url: str
    enabled: bool
    prefetch_count: int

    # 导入与查询分队列，就是简历里说的「任务资源隔离」
    @property
    def import_exchange(self) -> str:
        return f"{MQ_PREFIX}import"

    @property
    def import_queue(self) -> str:
        return f"{MQ_PREFIX}import.task"

    @property
    def import_routing_key(self) -> str:
        return f"{MQ_PREFIX}import.task"

    @property
    def import_dlq(self) -> str:
        return f"{MQ_PREFIX}import.dlq"

    @property
    def import_dlx(self) -> str:
        return f"{MQ_PREFIX}import.dlx"

    @property
    def query_exchange(self) -> str:
        return f"{MQ_PREFIX}query"

    @property
    def query_queue(self) -> str:
        return f"{MQ_PREFIX}query.task"

    @property
    def query_routing_key(self) -> str:
        return f"{MQ_PREFIX}query.task"


@dataclass
class LangSmithConfig:
    enabled: bool
    project: str
    # api_key 只用于判断是否配置齐全，不在任何位置打印

    # langsmith SDK 在不同版本里读的名字不一样：LANGCHAIN_TRACING_V2（旧）、
    # LANGSMITH_TRACING / LANGSMITH_TRACING_V2（新）都出现过。简历里写了 LangSmith
    # 链路观测，所以不能把开关押在某一个变量名上——升级 SDK 就会静默失效。
    TRACING_ALIASES = ("LANGCHAIN_TRACING_V2", "LANGSMITH_TRACING", "LANGSMITH_TRACING_V2")
    KEY_ALIASES = ("LANGCHAIN_API_KEY", "LANGSMITH_API_KEY")
    PROJECT_ALIASES = ("LANGCHAIN_PROJECT", "LANGSMITH_PROJECT")


@dataclass
class MetricsConfig:
    """流式性能指标（TTFT / 总延迟）的进程内滑动窗口聚合配置。"""

    enabled: bool
    window_size: int


infra_config_pg = PostgresConfig(
    dsn=os.getenv("PG_DSN", ""),
    enabled=_env_bool("PG_ENABLED", False),
)

infra_config_redis = RedisConfig(
    url=os.getenv("REDIS_URL", ""),
    enabled=_env_bool("REDIS_ENABLED", False),
    default_ttl_seconds=int(os.getenv("REDIS_TASK_TTL_SECONDS", "86400")),
)

infra_config_mq = RabbitMQConfig(
    url=os.getenv("RABBITMQ_URL", ""),
    enabled=_env_bool("RABBITMQ_ENABLED", False),
    prefetch_count=int(os.getenv("RABBITMQ_PREFETCH", "1")),
)

def _first_env(names: tuple[str, ...]) -> str:
    for name in names:
        value = (os.getenv(name) or "").strip()
        if value:
            return value
    return ""


def _any_env_bool(names: tuple[str, ...]) -> bool:
    """任一别名置真即算开启（只看显式设过的，没设的按 False 处理）。"""
    return any(_env_bool(name, False) for name in names if os.getenv(name) is not None)


infra_config_langsmith = LangSmithConfig(
    enabled=_any_env_bool(LangSmithConfig.TRACING_ALIASES),
    project=_first_env(LangSmithConfig.PROJECT_ALIASES) or "agentic-rag",
)


def apply_langsmith_env() -> bool:
    """把 tracing 开关与项目名写全所有别名，返回是否处于开启状态。

    由 LLM 客户端在初始化前调用：langchain 是否上报完全看进程环境变量，
    这里统一一次，避免"配置里开了但 SDK 读的是另一个名字"。
    """
    if not infra_config_langsmith.enabled:
        return False
    for name in LangSmithConfig.TRACING_ALIASES:
        os.environ[name] = "true"
    for name in LangSmithConfig.PROJECT_ALIASES:
        os.environ[name] = infra_config_langsmith.project
    return True

infra_config_metrics = MetricsConfig(
    enabled=_env_bool("METRICS_ENABLED", True),
    window_size=int(os.getenv("METRICS_WINDOW_SIZE", "200")),
)
