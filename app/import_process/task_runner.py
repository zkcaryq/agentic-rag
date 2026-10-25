"""导入任务执行器。

HTTP 服务与 MQ worker 共用这一份逻辑：HTTP 收到文件后投递 MQ，
worker 消费后调用 execute_import；MQ 不可用时 HTTP 侧用线程兜底直接调用。
"""
from app.clients import pg_task_utils as tasks
from app.core.logger import logger
from app.import_process.agent.main_graph import kb_import_app
from app.import_process.agent.state import create_default_state
from app.utils.task_utils import (
    TASK_STATUS_COMPLETED,
    TASK_STATUS_FAILED,
    TASK_STATUS_PARTIAL,
    clear_checkpoint,
    load_checkpoint,
    update_task_status,
)

# 这些字段从 checkpoint 里恢复时允许为空但不覆盖已有值
_RESTORE_FIELDS = (
    "file_title",
    "local_dir",
    "local_file_path",
    "pdf_path",
    "md_path",
    "md_content",
    "product_name",
    "chunks",
    "is_md_read_enabled",
    "is_pdf_read_enabled",
    "is_office_read_enabled",
)


def execute_import(
    task_id: str,
    local_dir: str,
    local_file_path: str,
    file_name: str = "",
    file_type: str = "",
    file_size: int = 0,
) -> dict:
    """执行一次导入。异常会被吞掉但会写进任务表，便于统计失败率。"""
    tasks.mark_started(task_id)
    update_task_status(task_id, "processing")
    logger.info(f"[{task_id}] 开始执行导入任务: {local_file_path}")

    init_state = create_default_state(
        task_id=task_id,
        local_dir=local_dir,
        local_file_path=local_file_path,
    )

    # 断点续跑：恢复解析阶段的中间结果，已完成节点由图的包装器跳过
    checkpoint = load_checkpoint(task_id)
    if checkpoint:
        restored = [field for field in _RESTORE_FIELDS if checkpoint.get(field)]
        for field in restored:
            init_state[field] = checkpoint[field]
        logger.info(f"[{task_id}] 从断点续跑，已恢复字段: {restored}")

    try:
        final_state = kb_import_app.invoke(init_state)
        chunk_count = len(final_state.get("chunks", []))
        product_name = final_state.get("product_name", "")
        failed_stage = final_state.get("failed_stage")
        duration_status = TASK_STATUS_PARTIAL if failed_stage else TASK_STATUS_COMPLETED

        update_task_status(task_id, duration_status)
        tasks.mark_finished(
            task_id, duration_status, chunk_count=chunk_count, failed_stage=failed_stage
        )
        logger.info(
            f"[{task_id}] 导入任务完成: 产品={product_name}, 切片数={chunk_count}"
            + (f", 存在节点失败: {failed_stage}" if failed_stage else "")
        )
        # 成功后清掉断点快照，保留任务状态供 /status 查询
        clear_checkpoint(task_id)
        return {"status": duration_status, "chunk_count": chunk_count}

    except Exception as e:
        failed_stage = init_state.get("failed_stage")
        update_task_status(task_id, TASK_STATUS_FAILED)
        tasks.mark_finished(
            task_id,
            TASK_STATUS_FAILED,
            failed_stage=failed_stage,
            error_type=e.__class__.__name__,
            error_message=str(e),
        )
        logger.error(f"[{task_id}] 导入任务失败（阶段 {failed_stage}）: {e}", exc_info=True)
        return {"status": TASK_STATUS_FAILED, "failed_stage": failed_stage}
