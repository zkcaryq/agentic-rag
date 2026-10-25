"""
检索服务端

路由：
  GET  /chat.html            聊天页面（读 app/query_process/page/chat.html）
  POST /query                提问；is_stream=true 时立即返回 session_id 并后台执行
  GET  /stream/{session_id}  SSE 事件流（ready / progress / delta / final / error）
  GET  /history/{session_id} 会话历史
  GET  /media/{storage_key}  图片签名 URL 重定向（私有 OSS bucket）
  GET  /health               健康检查
"""
import os
import sys
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

from app.clients.history import clear_history, get_recent_messages, save_chat_message
from app.core.logger import logger
from app.query_process.agent.main_graph import query_app
from app.query_process.agent.state import create_default_query_state
from app.utils.sse_utils import (
    SSEEvent,
    create_sse_queue,
    get_sse_queue,
    push_to_session,
    sse_generator,
)
from app.utils.task_utils import set_task_result, update_task_status

PAGE_DIR = Path(__file__).resolve().parents[1] / "page"

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
        final_state = query_app.invoke(init_state)
        return {
            "code": 200,
            "session_id": session_id,
            "answer": final_state.get("answer", "抱歉，无法生成答案"),
            "references": final_state.get("references", []),
        }

    # 流式：先建队列再派发，客户端随后连 /stream 就能读到缓冲的事件
    create_sse_queue(session_id)
    update_task_status(session_id, "processing", push_queue=True)
    _executor.submit(_run_query_async, init_state, session_id)
    return {"code": 200, "session_id": session_id, "stream": True}


def _run_query_async(state, session_id: str) -> None:
    """后台执行查询图；所有结果通过 SSE 事件推给前端。"""
    try:
        query_app.invoke(state)
        update_task_status(session_id, "completed", push_queue=True)
    except Exception as e:
        logger.error(f"[{session_id}] 查询失败: {e}", exc_info=True)
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
    """私有 bucket 的图片访问入口：校验后 302 到带签名的 URL。"""
    from app.clients.oss_utils import sign_url

    url = sign_url(storage_key)
    if not url:
        raise HTTPException(status_code=404, detail="对象不存在或 OSS 未启用")
    return RedirectResponse(url, status_code=302)


@app.get("/health")
async def health():
    return {"status": "ok"}


if __name__ == "__main__":
    print("启动查询服务... http://127.0.0.1:8001/chat.html")
    uvicorn.run(app, host="127.0.0.1", port=8001)
