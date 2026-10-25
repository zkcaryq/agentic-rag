"""SQLite 版会话历史（本地演示用；生产走 PostgreSQL）。

对外签名与 pg_history_utils 完全一致，业务侧通过 app.clients.history 门面调用。
"""
import json
import os
import sqlite3
import threading
from pathlib import Path
from typing import Any, Dict, List

from app.core.logger import logger

_lock = threading.Lock()
_initialised_paths: set[str] = set()

_DDL = [
    """
    CREATE TABLE IF NOT EXISTS chat_message (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        session_id TEXT NOT NULL,
        role TEXT NOT NULL,
        text TEXT NOT NULL,
        rewritten_query TEXT NOT NULL DEFAULT '',
        product_names TEXT,
        image_urls TEXT,
        ts TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ','now'))
    )
    """,
    """
    CREATE INDEX IF NOT EXISTS idx_chat_message_session_ts
        ON chat_message (session_id, ts DESC)
    """,
]


def _db_path() -> Path:
    return Path(os.getenv("SQLITE_HISTORY_PATH", "output/demo/history.db"))


def _connect() -> sqlite3.Connection:
    path = _db_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path, timeout=10)
    conn.row_factory = sqlite3.Row
    key = str(path)
    if key not in _initialised_paths:
        with _lock:
            if key not in _initialised_paths:
                with conn:
                    for statement in _DDL:
                        conn.execute(statement)
                _initialised_paths.add(key)
                logger.info(f"SQLite 历史库已就绪: {path}")
    return conn


def save_chat_message(
    session_id: str,
    role: str,
    text: str,
    rewritten_query: str = "",
    product_names: List[str] = None,
    image_urls: List[str] = None,
    message_id: str = None,
) -> str:
    payload = (
        json.dumps(product_names or [], ensure_ascii=False),
        json.dumps(image_urls or [], ensure_ascii=False),
    )
    conn = _connect()
    try:
        with conn:
            if message_id:
                conn.execute(
                    """
                    UPDATE chat_message
                       SET session_id = ?, role = ?, text = ?, rewritten_query = ?,
                           product_names = ?, image_urls = ?
                     WHERE id = ?
                    """,
                    (session_id, role, text, rewritten_query or "", payload[0],
                     payload[1], int(message_id)),
                )
                return message_id
            cursor = conn.execute(
                """
                INSERT INTO chat_message
                    (session_id, role, text, rewritten_query, product_names, image_urls)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (session_id, role, text, rewritten_query or "", payload[0], payload[1]),
            )
            return str(cursor.lastrowid)
    finally:
        conn.close()


def get_recent_messages(session_id: str, limit: int = 10) -> List[Dict[str, Any]]:
    conn = _connect()
    try:
        rows = conn.execute(
            """
            SELECT * FROM (
                SELECT id, session_id, role, text, rewritten_query,
                       product_names, image_urls, ts
                  FROM chat_message
                 WHERE session_id = ?
              ORDER BY ts DESC, id DESC
                 LIMIT ?
            ) recent
            ORDER BY ts ASC, id ASC
            """,
            (session_id, limit),
        ).fetchall()
    finally:
        conn.close()

    messages = []
    for row in rows:
        messages.append(
            {
                "id": str(row["id"]),
                "session_id": row["session_id"],
                "role": row["role"],
                "text": row["text"],
                "rewritten_query": row["rewritten_query"],
                "product_names": json.loads(row["product_names"] or "[]"),
                "image_urls": json.loads(row["image_urls"] or "[]"),
                "ts": row["ts"],
            }
        )
    return messages


def clear_history(session_id: str) -> int:
    conn = _connect()
    try:
        with conn:
            cursor = conn.execute(
                "DELETE FROM chat_message WHERE session_id = ?", (session_id,)
            )
            return cursor.rowcount
    finally:
        conn.close()


def update_message_product_names(ids: List[str], product_names: List[str]) -> int:
    if not ids:
        return 0
    payload = json.dumps(product_names or [], ensure_ascii=False)
    conn = _connect()
    try:
        with conn:
            placeholders = ",".join("?" for _ in ids)
            cursor = conn.execute(
                f"UPDATE chat_message SET product_names = ? WHERE id IN ({placeholders})",
                [payload] + [int(i) for i in ids],
            )
            return cursor.rowcount
    finally:
        conn.close()
