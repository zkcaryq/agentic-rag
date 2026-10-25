"""对话历史存储入口。

已从 MongoDB 迁到 PostgreSQL（软路由 192.168.100.1 上的 agentic_rag 库）。
保留这一层薄封装，业务代码只 import 这里，不再直接依赖具体驱动。
"""
from typing import Any, Dict, List

from app.clients import pg_history_utils

save_chat_message = pg_history_utils.save_chat_message
get_recent_messages = pg_history_utils.get_recent_messages
clear_history = pg_history_utils.clear_history
update_message_product_names = pg_history_utils.update_message_product_names

__all__ = [
    "save_chat_message",
    "get_recent_messages",
    "clear_history",
    "update_message_product_names",
]
