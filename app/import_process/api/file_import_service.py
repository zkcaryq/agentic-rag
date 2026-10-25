"""
文件导入 Web 服务
提供文件上传、任务状态查询、静态页面服务。

上传后：登记任务 → 原文件存 OSS → 投递 RabbitMQ → 由 worker 消费执行。
MQ 不可用时退回进程内线程池执行，功能不受影响，只是失去跨进程隔离。
"""
import os
import sys
import uuid
import shutil
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path
from typing import List

sys.path.insert(
    0,
    os.path.dirname(
        os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    ),
)

import uvicorn
from fastapi import BackgroundTasks, FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse

from app.clients import mq_utils
from app.clients import oss_utils
from app.clients import pg_task_utils as tasks
from app.core.logger import logger
from app.import_process.task_runner import execute_import
from app.utils.path_util import PROJECT_ROOT
from app.utils.task_utils import (
    get_done_task_list,
    get_running_task_list,
    get_task_status,
)

SUPPORTED_EXTENSIONS = [".pdf", ".md", ".docx", ".xlsx"]

# MQ 不可用时的本地兜底执行器
_fallback_executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix="import")

app = FastAPI(
    title="File Import Service",
    description="知识库文件导入服务 - 支持 PDF/MD/DOCX/XLSX 上传与自动处理",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/import.html", response_class=HTMLResponse)
async def get_import_page():
    html_path = PROJECT_ROOT / "app" / "import_process" / "page" / "import.html"
    if not html_path.exists():
        return HTMLResponse(content="<h1>404 - import.html not found</h1>", status_code=404)
    with open(html_path, "r", encoding="utf-8") as f:
        return HTMLResponse(content=f.read())


@app.get("/health")
async def health_check():
    return {"status": "ok", "service": "file_import_service"}


@app.post("/upload")
async def upload_files(
    background_tasks: BackgroundTasks,
    files: List[UploadFile] = File(..., description="要上传的文件（PDF/MD/DOCX/XLSX）"),
):
    date_str = datetime.now().strftime("%Y%m%d")
    base_output_dir = PROJECT_ROOT / "output" / date_str
    base_output_dir.mkdir(parents=True, exist_ok=True)

    results = []
    for file in files:
        file_ext = os.path.splitext(file.filename)[1].lower()
        if file_ext not in SUPPORTED_EXTENSIONS:
            results.append(
                {
                    "filename": file.filename,
                    "success": False,
                    "error": f"不支持的文件类型: {file_ext}，仅支持 {SUPPORTED_EXTENSIONS}",
                }
            )
            continue

        task_id = str(uuid.uuid4())
        task_dir = base_output_dir / task_id
        task_dir.mkdir(parents=True, exist_ok=True)
        file_path = task_dir / file.filename

        try:
            with open(file_path, "wb") as buffer:
                shutil.copyfileobj(file.file, buffer)
        except Exception as e:
            logger.error(f"[{task_id}] 文件保存失败: {e}")
            results.append(
                {"filename": file.filename, "success": False, "error": f"文件保存失败: {e}"}
            )
            continue

        file_size = file_path.stat().st_size
        tasks.register_task(
            task_id=task_id,
            file_name=file.filename,
            file_type=file_ext,
            file_size=file_size,
        )

        # 原文件存 OSS（私有 bucket，只记 storage_key）
        storage_key = oss_utils.build_storage_key(
            Path(file.filename).stem, file.filename, sub_dir="sources"
        )
        if not oss_utils.upload_file(str(file_path), storage_key):
            logger.warning(f"[{task_id}] 原文件未上传 OSS，仅保留本地副本")

        payload = {
            "task_id": task_id,
            "local_dir": str(task_dir),
            "local_file_path": str(file_path),
            "file_name": file.filename,
            "file_type": file_ext,
            "file_size": file_size,
        }

        if mq_utils.publish_import_task(payload):
            dispatch = "rabbitmq"
        else:
            # MQ 不可用：退回进程内线程池，保证功能可用
            _fallback_executor.submit(execute_import, **payload)
            dispatch = "local_thread"
            logger.warning(f"[{task_id}] RabbitMQ 不可用，改用本地线程执行")

        results.append(
            {
                "filename": file.filename,
                "success": True,
                "task_id": task_id,
                "dispatch": dispatch,
                "message": "文件已接收，正在处理中",
            }
        )

    return JSONResponse(
        content={
            "code": 200,
            "message": f"收到 {len(files)} 个文件",
            "results": results,
        }
    )


@app.get("/status/{task_id}")
async def get_task_status_info(task_id: str):
    status = get_task_status(task_id)
    if not status:
        raise HTTPException(status_code=404, detail=f"任务 {task_id} 不存在")
    return {
        "code": 200,
        "task_id": task_id,
        "status": status,
        "done_list": get_done_task_list(task_id),
        "running_list": get_running_task_list(task_id),
    }


@app.get("/tasks")
async def list_tasks(limit: int = 50):
    """任务列表（来自 PostgreSQL import_tasks 表）。"""
    try:
        return {"code": 200, "tasks": tasks.list_tasks(limit=limit)}
    except Exception as e:
        logger.warning(f"读取任务列表失败: {e}")
        return {"code": 200, "tasks": [], "message": f"任务表不可用: {e}"}


@app.get("/tasks/{task_id}")
async def get_task_detail(task_id: str):
    detail = tasks.get_task(task_id)
    if not detail:
        raise HTTPException(status_code=404, detail=f"任务 {task_id} 不存在")
    return {"code": 200, "task": detail}


@app.get("/stats/import")
async def import_stats():
    """导入成功率 / 失败率聚合。"""
    try:
        return {"code": 200, "stats": tasks.aggregate_stats()}
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"任务表不可用: {e}")


if __name__ == "__main__":
    logger.info("启动文件导入服务...")
    uvicorn.run(app, host="127.0.0.1", port=8002)
