"""任务状态追踪。

对外函数签名保持不变（22 个调用点零改动），内部从「进程内存 dict」换成 Redis：
  - Redis 可用：状态落 rag:task:{id}，进程重启不丢，支持断点续跑；
  - Redis 不可用：自动退回进程内 dict，任务照常跑，只是不能跨进程续跑。

key 统一走 redis_utils.key()，保证符合 ACL 的 ~rag:* 前缀。
"""
import json
from typing import Any, Dict, List, Optional

from .sse_utils import push_to_session

try:  # 允许在没有 redis 依赖的环境下 import 本模块
    from app.clients.redis_utils import get_redis, key as redis_key, ttl_seconds
except Exception:  # pragma: no cover - 仅兜底
    get_redis = lambda: None  # type: ignore
    redis_key = lambda *parts: ":".join(parts)  # type: ignore
    ttl_seconds = lambda: 86400  # type: ignore

# ---------------------------
# 进程内兜底存储（Redis 不可用时使用）
# ---------------------------
_tasks_running_list: Dict[str, List[str]] = {}
_tasks_done_list: Dict[str, List[str]] = {}
_tasks_status: Dict[str, str] = {}
_tasks_result: Dict[str, Dict[str, str]] = {}

TASK_STATUS_PENDING = "pending"
TASK_STATUS_PROCESSING = "processing"
TASK_STATUS_COMPLETED = "completed"
TASK_STATUS_FAILED = "failed"
TASK_STATUS_PARTIAL = "partial_success"

# 节点名 -> 中文名映射（用于前端展示）
_NODE_NAME_TO_CN: Dict[str, str] = {
    "upload_file": "开始上传文件",
    "node_entry": "检查文件",
    "node_pdf_to_md": "PDF转Markdown",
    "node_office_to_md": "Office转Markdown",
    "node_md_img": "Markdown图片处理",
    "node_product_name_recognition": "主体名称识别",
    "node_document_split": "文档切分",
    "node_bge_embedding": "向量生成",
    "node_import_milvus": "导入向量库",
    "__end__": "处理完成",
    "END": "处理完成",
    # --- Query 流程节点 ---
    "node_product_confirm": "确认问题产品",
    "node_answer_output": "生成答案",
    "node_rerank": "重排序",
    "node_rrf": "倒排融合",
    "node_web_search_mcp": "网络搜索",
    "node_search_embedding": "切片搜索",
    "node_search_embedding_hyde": "切片搜索(假设性文档)",
    "node_multi_search": "多路搜索",
    "node_join": "多路搜索合并",
}


def _to_cn(node_name: str) -> str:
    return _NODE_NAME_TO_CN.get(node_name, node_name)


def _status_key(task_id: str) -> str:
    return redis_key("task", task_id)


def _key(task_id: str, suffix: str) -> str:
    return redis_key("task", task_id, suffix)


def _touch(client, *keys: str) -> None:
    for item in keys:
        try:
            client.expire(item, ttl_seconds())
        except Exception:
            pass


# ---------------------------
# 写：running / done
# ---------------------------
def add_running_task(task_id: str, node_name: str, is_stream: bool = False) -> None:
    """标记节点开始执行。"""
    client = get_redis()
    if client is not None:
        try:
            running = _key(task_id, "running")
            client.lrem(running, 0, node_name)
            client.rpush(running, node_name)
            _touch(client, running)
        except Exception:
            pass
    else:
        running_list = _tasks_running_list.setdefault(task_id, [])
        if node_name not in running_list:
            running_list.append(node_name)

    if is_stream:
        task_push_queue(task_id)


def add_done_task(task_id: str, node_name: str, is_stream: bool = False) -> None:
    """标记节点完成，并从 running 里摘掉。"""
    client = get_redis()
    if client is not None:
        try:
            running = _key(task_id, "running")
            done = _key(task_id, "done")
            client.lrem(running, 0, node_name)
            # 先删再加，保证顺序是「最近完成在最后」且不重复
            client.lrem(done, 0, node_name)
            client.rpush(done, node_name)
            _touch(client, running, done)
        except Exception:
            pass
    else:
        running_list = _tasks_running_list.setdefault(task_id, [])
        _tasks_running_list[task_id] = [n for n in running_list if n != node_name]
        done_list = _tasks_done_list.setdefault(task_id, [])
        if node_name not in done_list:
            done_list.append(node_name)

    if is_stream:
        task_push_queue(task_id)


def is_node_done(task_id: str, node_name: str) -> bool:
    """断点续跑用：判断某个节点是否已经完成。"""
    return node_name in _raw_done_list(task_id)


def _raw_done_list(task_id: str) -> List[str]:
    client = get_redis()
    if client is not None:
        try:
            return client.lrange(_key(task_id, "done"), 0, -1) or []
        except Exception:
            return []
    return list(_tasks_done_list.get(task_id, []))


# ---------------------------
# 结果 / 状态
# ---------------------------
def set_task_result(task_id: str, name: str, value: str) -> None:
    client = get_redis()
    if client is not None:
        try:
            result_key = _key(task_id, "result")
            client.hset(result_key, name, value)
            _touch(client, result_key)
        except Exception:
            pass
    else:
        _tasks_result.setdefault(task_id, {})[name] = value


def get_task_result(task_id: str, name: str, default: str = "") -> str:
    client = get_redis()
    if client is not None:
        try:
            value = client.hget(_key(task_id, "result"), name)
            return value if value is not None else default
        except Exception:
            return default
    return _tasks_result.get(task_id, {}).get(name, default)


def get_task_status(task_id: str) -> str:
    client = get_redis()
    if client is not None:
        try:
            return client.hget(_status_key(task_id), "status") or ""
        except Exception:
            return ""
    return _tasks_status.get(task_id, "")


def update_task_status(task_id: str, status_name: str, push_queue: bool = False) -> None:
    client = get_redis()
    if client is not None:
        try:
            status_key = _status_key(task_id)
            client.hset(status_key, "status", status_name)
            _touch(client, status_key)
        except Exception:
            pass
    else:
        _tasks_status[task_id] = status_name

    if push_queue:
        task_push_queue(task_id)


def get_done_task_list(task_id: str) -> List[str]:
    return [_to_cn(n) for n in _raw_done_list(task_id)]


def get_running_task_list(task_id: str) -> List[str]:
    client = get_redis()
    if client is not None:
        try:
            return [_to_cn(n) for n in client.lrange(_key(task_id, "running"), 0, -1)]
        except Exception:
            return []
    return [_to_cn(n) for n in _tasks_running_list.get(task_id, [])]


def clear_checkpoint(task_id: str) -> None:
    """只删断点快照，保留任务状态与结果（/status 仍可查）。"""
    client = get_redis()
    if client is None:
        return
    try:
        client.delete(_key(task_id, "checkpoint"))
    except Exception:
        pass


def clear_task(task_id: str) -> None:
    client = get_redis()
    if client is not None:
        try:
            client.delete(
                _status_key(task_id),
                _key(task_id, "running"),
                _key(task_id, "done"),
                _key(task_id, "result"),
                _key(task_id, "checkpoint"),
            )
        except Exception:
            pass
    _tasks_running_list.pop(task_id, None)
    _tasks_done_list.pop(task_id, None)
    _tasks_status.pop(task_id, None)
    _tasks_result.pop(task_id, None)


# ---------------------------
# 断点续跑：轻量状态快照
# ---------------------------
_SKIP_CHECKPOINT_FIELDS = {"dense_vector", "sparse_vector"}


def save_checkpoint(task_id: str, state: Dict[str, Any]) -> None:
    """把解析阶段的中间结果存到 Redis，供进程重启后续跑。

    不含向量（体积大且可由 embedding 节点重算）。
    """
    client = get_redis()
    if client is None:
        return
    try:
        payload = {
            "file_title": state.get("file_title", ""),
            "local_dir": state.get("local_dir", ""),
            "local_file_path": state.get("local_file_path", ""),
            "pdf_path": state.get("pdf_path", ""),
            "md_path": state.get("md_path", ""),
            "md_content": state.get("md_content", ""),
            "product_name": state.get("product_name", ""),
            "is_md_read_enabled": state.get("is_md_read_enabled", False),
            "is_pdf_read_enabled": state.get("is_pdf_read_enabled", False),
            "chunks": [
                {k: v for k, v in chunk.items() if k not in _SKIP_CHECKPOINT_FIELDS}
                for chunk in (state.get("chunks") or [])
            ],
        }
        checkpoint_key = _key(task_id, "checkpoint")
        client.set(checkpoint_key, json.dumps(payload, ensure_ascii=False))
        _touch(client, checkpoint_key)
    except Exception as e:
        from app.core.logger import logger

        logger.warning(f"保存断点失败: {e}")


def load_checkpoint(task_id: str) -> Optional[Dict[str, Any]]:
    client = get_redis()
    if client is None:
        return None
    try:
        raw = client.get(_key(task_id, "checkpoint"))
        if not raw:
            return None
        return json.loads(raw)
    except Exception:
        return None


# ---------------------------
# SSE 进度推送
# ---------------------------
def task_push_queue(task_id: str):
    push_to_session(task_id, "progress", {
        "status": get_task_status(task_id),
        "done_list": get_done_task_list(task_id),
        "running_list": get_running_task_list(task_id),
    })
