"""会话历史存储入口。

`HISTORY_BACKEND`：
  - `auto`（默认）：PG_ENABLED=true 用 PostgreSQL，否则用 SQLite；
  - `postgres`：强制走软路由上的 PostgreSQL；
  - `sqlite`：本地演示用文件库（output/demo/history.db）。

业务代码只 import 这里，不直接依赖具体驱动。
"""
import os
from typing import Any, Dict, List


def _backend():
    name = os.getenv("HISTORY_BACKEND", "auto").strip().lower()

    if name == "sqlite":
        from app.clients import sqlite_history_utils

        return sqlite_history_utils

    if name == "postgres":
        from app.clients import pg_history_utils

        return pg_history_utils

    from app.conf.infra_config import infra_config_pg

    if infra_config_pg.enabled:
        from app.clients import pg_history_utils

        return pg_history_utils

    from app.clients import sqlite_history_utils

    return sqlite_history_utils


def save_chat_message(
    session_id: str,
    role: str,
    text: str,
    rewritten_query: str = "",
    product_names: List[str] = None,
    image_urls: List[str] = None,
    message_id: str = None,
) -> str:
    return _backend().save_chat_message(
        session_id=session_id,
        role=role,
        text=text,
        rewritten_query=rewritten_query,
        product_names=product_names,
        image_urls=image_urls,
        message_id=message_id,
    )


def get_recent_messages(session_id: str, limit: int = 10) -> List[Dict[str, Any]]:
    return _backend().get_recent_messages(session_id, limit=limit)


def clear_history(session_id: str) -> int:
    return _backend().clear_history(session_id)


def update_message_product_names(ids: List[str], product_names: List[str]) -> int:
    return _backend().update_message_product_names(ids, product_names)
