"""PostgreSQL 连接与建表。

两个注意点（都是这台软路由上实测踩过的坑）：
  1. 路由 PG 的会话时区默认 Asia/Shanghai，而表里是 naive TIMESTAMP。
     连接必须带 options="-c TimeZone=UTC"（psycopg 不认 server_settings），
     否则写进去的时间会比 UTC 快 8 小时。
  2. 路由与开发机之间有约 1 分钟时钟漂移，所以时间一律用数据库自己的 now()，
     不要在 Python 侧取时间再和库里的时间比较。
"""
import threading

from app.conf.infra_config import infra_config_pg as pg_conf
from app.core.logger import logger

_schema_ready = False
_lock = threading.Lock()

_DDL = [
    """
    CREATE TABLE IF NOT EXISTS chat_message (
        id BIGSERIAL PRIMARY KEY,
        session_id TEXT NOT NULL,
        role TEXT NOT NULL,
        text TEXT NOT NULL,
        rewritten_query TEXT NOT NULL DEFAULT '',
        product_names JSONB,
        image_urls JSONB,
        ts TIMESTAMP NOT NULL DEFAULT (now() AT TIME ZONE 'UTC')
    )
    """,
    """
    CREATE INDEX IF NOT EXISTS idx_chat_message_session_ts
        ON chat_message (session_id, ts DESC)
    """,
    """
    CREATE TABLE IF NOT EXISTS import_tasks (
        task_id TEXT PRIMARY KEY,
        file_name TEXT NOT NULL DEFAULT '',
        file_type TEXT NOT NULL DEFAULT '',
        file_size BIGINT NOT NULL DEFAULT 0,
        status TEXT NOT NULL DEFAULT 'pending',
        failed_stage TEXT,
        error_type TEXT,
        error_message TEXT,
        retry_count INTEGER NOT NULL DEFAULT 0,
        chunk_count INTEGER NOT NULL DEFAULT 0,
        duration_ms BIGINT,
        started_at TIMESTAMP,
        finished_at TIMESTAMP,
        created_at TIMESTAMP NOT NULL DEFAULT (now() AT TIME ZONE 'UTC')
    )
    """,
    """
    CREATE INDEX IF NOT EXISTS idx_import_tasks_status
        ON import_tasks (status, created_at DESC)
    """,
]


def _connect():
    if not pg_conf.enabled:
        raise RuntimeError("PG_ENABLED=false，未启用 PostgreSQL")
    if not pg_conf.dsn:
        raise RuntimeError("未配置 PG_DSN")
    return psycopg_connect()


def psycopg_connect():
    import psycopg

    return psycopg.connect(
        pg_conf.dsn,
        connect_timeout=5,
        # 锁 UTC，业务表里是 naive TIMESTAMP
        options="-c TimeZone=UTC",
    )


def get_connection():
    """取一个已确保建表的连接（调用方负责 close）。"""
    conn = _connect()
    ensure_schema(conn)
    return conn


def ensure_schema(conn=None):
    """幂等建表。首次调用会执行全部 DDL 并提交。"""
    global _schema_ready
    if _schema_ready:
        return

    with _lock:
        if _schema_ready:
            return
        own_connection = conn is None
        target = conn or _connect()
        try:
            with target.cursor() as cur:
                for statement in _DDL:
                    cur.execute(statement)
            target.commit()
            _schema_ready = True
            logger.info("PostgreSQL 表结构检查完成（chat_message / import_tasks）")
        finally:
            if own_connection:
                target.close()
