"""
检索服务端

路由：
  GET  /chat.html            聊天页面（读 app/query_process/page/chat.html）
  POST /query                提问；is_stream=true 时立即返回 session_id 并后台执行
  GET  /stream/{session_id}  SSE 事件流（ready / progress / delta / final / error）
  GET  /history/{session_id} 会话历史
  GET  /result/{session_id}  会话结果快照（状态/答案/引用校验/错误，SSE 断开时兜底取回）
  GET  /media/{storage_key}  图片签名 URL 重定向（私有 OSS bucket）
  GET  /metrics              TTFT / 总延迟滑动窗口快照
  GET  /health               健康检查

流式查询的派发：优先投 RabbitMQ 由 worker 进程执行（与导入任务资源隔离），
MQ 或 Redis 不可用时退回本进程线程池，功能不变、只是失去跨进程隔离。
"""
import os
import sys
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(
    0,
    os.path.dirname(
        os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    ),
)

import uvicorn
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse, RedirectResponse, StreamingResponse
from pydantic import BaseModel

from app.clients import mq_utils
from app.clients import redis_utils
from app.clients.history import clear_history, get_recent_messages, save_chat_message
from app.conf.infra_config import infra_config_metrics
from app.core import metrics
from app.core.logger import logger
from app.query_process.agent.main_graph import query_app
from app.query_process.agent.state import create_default_query_state
from app.utils.sse_utils import (
    SSEEvent,
    create_sse_queue,
    get_sse_queue,
    mark_remote_transport,
    push_to_session,
    sse_generator,
)
from app.utils.task_utils import (
    get_done_task_list,
    get_running_task_list,
    get_task_result,
    get_task_status,
    set_task_result,
    update_task_status,
)

PAGE_DIR = Path(__file__).resolve().parents[1] / "page"

# 整轮问答（含检索、重排、生成）的耗时桶；生成节点自己的模型 TTFT 见 answer_generate
QUERY_ENDPOINT = "query"

# 查询图是同步阻塞的，放到线程池里跑，避免堵住 FastAPI 的事件循环
_executor = ThreadPoolExecutor(max_workers=4, thread_name_prefix="query")

app = FastAPI(title="Query Service")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class QueryRequest(BaseModel):
    query: str
    session_id: str | None = None
    is_stream: bool = False


def _mq_dispatch_enabled() -> bool:
    """跨进程执行需要两样东西：MQ 派活 + Redis 把事件送回本进程。缺一个就退回线程池。"""
    return mq_utils.is_enabled() and redis_utils.is_enabled()


@app.get("/chat.html", response_class=HTMLResponse)
async def get_chat_page():
    page = PAGE_DIR / "chat.html"
    if page.exists():
        return FileResponse(page, media_type="text/html")
    return HTMLResponse("<h1>chat.html 缺失</h1>", status_code=404)


@app.post("/query")
async def query(request: QueryRequest):
    session_id = request.session_id or str(uuid.uuid4())
    logger.info(f"[{session_id}] 收到查询: {request.query}")

    # 用户消息在入口落一次，节点内不再重复写，避免历史里出现两条同样的提问
    save_chat_message(session_id, "user", request.query, "", [])

    init_state = create_default_query_state(
        session_id=session_id,
        original_query=request.query,
        is_stream=request.is_stream,
    )

    if not request.is_stream:
        started = time.monotonic()
        final_state = query_app.invoke(init_state)
        metrics.record(QUERY_ENDPOINT, total_ms=(time.monotonic() - started) * 1000)
        return {
            "code": 200,
            "session_id": session_id,
            "answer": final_state.get("answer", "抱歉，无法生成答案"),
            "references": final_state.get("references", []),
        }

    # 流式：先建队列再派发，客户端随后连 /stream 就能读到缓冲的事件
    create_sse_queue(session_id)

    if _mq_dispatch_enabled() and mq_utils.publish_query_task(
        {"session_id": session_id, "query": request.query, "is_stream": True}
    ):
        # 必须在任何 push 之前标记：标记后事件统一走 Redis，本进程队列不再参与
        mark_remote_transport(session_id)
        update_task_status(session_id, "processing", push_queue=True)
        return {"code": 200, "session_id": session_id, "stream": True, "dispatch": "rabbitmq"}

    update_task_status(session_id, "processing", push_queue=True)
    _executor.submit(_run_query_async, init_state, session_id)
    return {"code": 200, "session_id": session_id, "stream": True, "dispatch": "local_thread"}


def _run_query_async(state, session_id: str) -> None:
    """线程池分支：在本进程执行查询图，结果通过 SSE 事件推给前端。"""
    started = time.monotonic()
    try:
        query_app.invoke(state)
        metrics.record(QUERY_ENDPOINT, total_ms=(time.monotonic() - started) * 1000)
        update_task_status(session_id, "completed", push_queue=True)
    except Exception as e:
        logger.exception(f"[{session_id}] 查询失败: {e}")
        set_task_result(session_id, "error", str(e))
        update_task_status(session_id, "failed", push_queue=True)
        push_to_session(
            session_id, SSEEvent.ERROR, {"message": "处理失败，请稍后重试"}
        )
    finally:
        push_to_session(session_id, SSEEvent.CLOSE, {})


@app.get("/stream/{session_id}")
async def stream(session_id: str, request: Request):
    # 允许前端先连流再发问：没有队列就补一个
    if get_sse_queue(session_id) is None:
        create_sse_queue(session_id)
    return StreamingResponse(
        sse_generator(session_id, request),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


@app.get("/history/{session_id}")
async def history(session_id: str, limit: int = 20):
    return {
        "code": 200,
        "session_id": session_id,
        "messages": get_recent_messages(session_id, limit=limit),
    }


@app.delete("/history/{session_id}")
async def delete_history(session_id: str):
    deleted = clear_history(session_id)
    return {"code": 200, "deleted": deleted}


@app.get("/media/{storage_key:path}")
async def media(storage_key: str):
    """图片访问入口：本地后端直接回文件，OSS 后端 302 到签名 URL。"""
    from app.clients import storage

    local_path = storage.resolve_local(storage_key)
    if local_path is not None:
        return FileResponse(local_path)

    url = storage.sign_url(storage_key)
    if not url:
        raise HTTPException(status_code=404, detail="对象不存在或存储未启用")
    return RedirectResponse(url, status_code=302)


@app.get("/result/{session_id}")
async def query_result(session_id: str):
    """读会话的落库结果（状态 + 答案 + 引用校验结论 + 错误）。

    节点本来就把这些写进 Redis 的 `rag:task:{id}:result`，但之前没有任何读接口，
    等于写了没人取。SSE 断开时（或查询由 worker 进程执行、流没接上）前端就靠这个
    把结果取回来。
    """
    return {
        "code": 200,
        "session_id": session_id,
        "status": get_task_status(session_id),
        "answer": get_task_result(session_id, "answer"),
        "citation_validation": get_task_result(session_id, "citation_validation"),
        "error": get_task_result(session_id, "error"),
        "done_list": get_done_task_list(session_id),
        "running_list": get_running_task_list(session_id),
    }


@app.get("/metrics")
async def metrics_snapshot():
    """TTFT / 总延迟滑动窗口快照。

    窗口是进程内的：MQ 派发出去的查询样本记在 worker 进程，这里看不到。
    要端到端的并发数字请用 scripts/bench_concurrency.py（客户端口径，跨进程都算）。
    """
    return {
        "code": 200,
        "scope": "本进程窗口，不含 worker 进程的样本",
        "window_enabled": infra_config_metrics.enabled,
        "query_dispatch": "rabbitmq" if _mq_dispatch_enabled() else "local_thread",
        "endpoints": metrics.snapshot(),
    }


@app.get("/health")
async def health():
    return {"status": "ok"}


if __name__ == "__main__":
    print("启动查询服务... http://127.0.0.1:8001/chat.html")
    uvicorn.run(app, host="127.0.0.1", port=8001)
