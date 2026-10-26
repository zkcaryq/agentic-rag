"""导入任务表读写。

这张表是「导入失败率 <4%」这个指标的唯一数据来源：
分母是文件数，失败按 failed_stage 归因，部分成功（有 chunk 但节点报错）单独记。

时长用数据库自己的 now() 计算，避开路由与开发机之间约 1 分钟的时钟漂移。
"""
from typing import Any, Dict, Optional

from app.clients.pg_client import get_connection
from app.core.logger import logger

STATUS_PENDING = "pending"
STATUS_PROCESSING = "processing"
STATUS_COMPLETED = "completed"
STATUS_FAILED = "failed"
STATUS_PARTIAL = "partial_success"


def register_task(
    task_id: str,
    file_name: str,
    file_type: str = "",
    file_size: int = 0,
) -> None:
    """任务入队时登记一行（幂等）。"""
    try:
        with get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO import_tasks (task_id, file_name, file_type, file_size, status)
                    VALUES (%s, %s, %s, %s, %s)
                    ON CONFLICT (task_id) DO NOTHING
                    """,
                    (task_id, file_name, file_type, int(file_size), STATUS_PENDING),
                )
            conn.commit()
    except Exception as e:
        logger.warning(f"登记导入任务失败（不影响主流程）: {e}")


def mark_started(task_id: str) -> None:
    try:
        with get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    UPDATE import_tasks
                       SET status = %s,
                           started_at = COALESCE(started_at, now() AT TIME ZONE 'UTC')
                     WHERE task_id = %s
                    """,
                    (STATUS_PROCESSING, task_id),
                )
            conn.commit()
    except Exception as e:
        logger.warning(f"标记任务开始失败: {e}")


def mark_finished(
    task_id: str,
    status: str,
    chunk_count: int = 0,
    failed_stage: Optional[str] = None,
    error_type: Optional[str] = None,
    error_message: Optional[str] = None,
) -> None:
    """标记任务结束。duration_ms 由数据库计算，避免两端时钟不一致。"""
    try:
        with get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    UPDATE import_tasks
                       SET status = %s,
                           chunk_count = %s,
                           failed_stage = %s,
                           error_type = %s,
                           error_message = %s,
                           finished_at = now() AT TIME ZONE 'UTC',
                           duration_ms = CASE
                               WHEN started_at IS NULL THEN NULL
                               ELSE (EXTRACT(EPOCH FROM (now() - started_at)) * 1000)::BIGINT
                           END
                     WHERE task_id = %s
                    """,
                    (status, int(chunk_count), failed_stage, error_type,
                     (error_message or "")[:1000] or None, task_id),
                )
            conn.commit()
    except Exception as e:
        logger.warning(f"标记任务结束失败: {e}")


def get_task(task_id: str) -> Dict[str, Any]:
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT task_id, file_name, file_type, file_size, status, failed_stage,
                       error_type, chunk_count, duration_ms, started_at, finished_at
                  FROM import_tasks WHERE task_id = %s
                """,
                (task_id,),
            )
            row = cur.fetchone()
    if not row:
        return {}
    keys = ["task_id", "file_name", "file_type", "file_size", "status", "failed_stage",
            "error_type", "chunk_count", "duration_ms", "started_at", "finished_at"]
    return dict(zip(keys, row))


def list_tasks(limit: int = 50) -> list[Dict[str, Any]]:
    """最近的任务列表（按创建时间倒序）。"""
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT task_id, file_name, status, failed_stage, chunk_count,
                       duration_ms, created_at, finished_at
                  FROM import_tasks
              ORDER BY created_at DESC
                 LIMIT %s
                """,
                (limit,),
            )
            rows = cur.fetchall()
    keys = ["task_id", "file_name", "status", "failed_stage", "chunk_count",
            "duration_ms", "created_at", "finished_at"]
    return [dict(zip(keys, row)) for row in rows]


def aggregate_stats() -> Dict[str, Any]:
    """按状态聚合，给出失败率口径。

    分母 = 已完成 + 失败（不含 pending/processing，避免把还没跑完的算成失败）。
    部分成功（有内容但节点报错）单列，不计入分母。
    """
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT status, COUNT(*), COALESCE(SUM(chunk_count), 0)
                  FROM import_tasks
              GROUP BY status
                """
            )
            rows = cur.fetchall()
            cur.execute(
                """
                SELECT failed_stage, COUNT(*)
                  FROM import_tasks
                 WHERE status = %s AND failed_stage IS NOT NULL
              GROUP BY failed_stage
              ORDER BY COUNT(*) DESC
                """,
                (STATUS_FAILED,),
            )
            stage_rows = cur.fetchall()

    by_status = {row[0]: {"count": row[1], "chunks": int(row[2])} for row in rows}
    decided = by_status.get(STATUS_COMPLETED, {}).get("count", 0) + \
        by_status.get(STATUS_FAILED, {}).get("count", 0)
    failed = by_status.get(STATUS_FAILED, {}).get("count", 0)

    return {
        "by_status": by_status,
        "failed_stage_distribution": {row[0]: row[1] for row in stage_rows},
        "denominator": decided,
        "failed": failed,
        "failure_rate": (failed / decided) if decided else 0.0,
    }
