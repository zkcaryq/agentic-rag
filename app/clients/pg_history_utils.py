"""对话历史读写（PostgreSQL 实现，替换原 MongoDB）。

对外签名与 mongo_history_utils 保持一致，业务侧通过 app.clients.history 门面调用。
"""
import json
from datetime import timezone
from typing import Any, Dict, List

from app.clients.pg_client import get_connection
from app.core.logger import logger


def _row_to_dict(row) -> Dict[str, Any]:
    created_at = row[7]
    return {
        "id": str(row[0]),
        "session_id": row[1],
        "role": row[2],
        "text": row[3],
        "rewritten_query": row[4],
        "product_names": row[5] or [],
        "image_urls": row[6] or [],
        # 保持与原 Mongo 版本一致的字段形式：epoch 秒（float）
        "ts": created_at.replace(tzinfo=timezone.utc).timestamp() if created_at else 0.0,
    }


def save_chat_message(
    session_id: str,
    role: str,
    text: str,
    rewritten_query: str = "",
    product_names: List[str] = None,
    image_urls: List[str] = None,
    message_id: str = None,
) -> str:
    """写入或更新一条会话记录。message_id 有值时做更新。"""
    payload = (
        json.dumps(product_names or [], ensure_ascii=False),
        json.dumps(image_urls or [], ensure_ascii=False),
    )

    with get_connection() as conn:
        with conn.cursor() as cur:
            if message_id:
                cur.execute(
                    """
                    UPDATE chat_message
                       SET session_id = %s, role = %s, text = %s,
                           rewritten_query = %s, product_names = %s::jsonb, image_urls = %s::jsonb
                     WHERE id = %s
                 RETURNING id
                    """,
                    (session_id, role, text, rewritten_query or "",
                     payload[0], payload[1], int(message_id)),
                )
                row = cur.fetchone()
                conn.commit()
                return str(row[0]) if row else message_id

            cur.execute(
                """
                INSERT INTO chat_message
                    (session_id, role, text, rewritten_query, product_names, image_urls)
                VALUES (%s, %s, %s, %s, %s::jsonb, %s::jsonb)
             RETURNING id
                """,
                (session_id, role, text, rewritten_query or "", payload[0], payload[1]),
            )
            row = cur.fetchone()
            conn.commit()
            return str(row[0])


def get_recent_messages(session_id: str, limit: int = 10) -> List[Dict[str, Any]]:
    """取最近 N 条，按时间正序返回。

    注意：原 Mongo 实现是 find().sort(ts ASC).limit(N)，实际取到的是最早的 N 条；
    这里改成先按时间倒序取 N 条再正序输出，才是真正的"最近 N 条"。
    """
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT id, session_id, role, text, rewritten_query, product_names, image_urls, ts
                  FROM (
                        SELECT id, session_id, role, text, rewritten_query,
                               product_names, image_urls, ts
                          FROM chat_message
                         WHERE session_id = %s
                      ORDER BY ts DESC, id DESC
                         LIMIT %s
                       ) recent
              ORDER BY ts ASC, id ASC
                """,
                (session_id, limit),
            )
            rows = cur.fetchall()

    return [_row_to_dict(row) for row in rows]


def clear_history(session_id: str) -> int:
    """清空指定会话的所有记录，返回删除条数。"""
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute("DELETE FROM chat_message WHERE session_id = %s", (session_id,))
            deleted = cur.rowcount
            conn.commit()
    logger.info(f"已清空会话 {session_id} 的 {deleted} 条历史")
    return deleted


def update_message_product_names(ids: List[str], product_names: List[str]) -> int:
    """批量更新历史记录的关联产品名。"""
    if not ids:
        return 0
    payload = json.dumps(product_names or [], ensure_ascii=False)
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "UPDATE chat_message SET product_names = %s::jsonb WHERE id = ANY(%s)",
                (payload, [int(i) for i in ids]),
            )
            updated = cur.rowcount
            conn.commit()
    return updated
