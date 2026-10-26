"""
联网搜索节点
使用百炼 MCP 服务进行联网搜索
"""
import sys
import json
import asyncio
import threading

from app.core.logger import logger
from app.query_process.agent.state import QueryGraphState
from app.utils.task_utils import add_running_task, add_done_task
from app.conf.bailian_mcp_config import mcp_config


def node_web_search_mcp(state: QueryGraphState) -> QueryGraphState:
    """
    联网搜索节点
    """
    func_name = sys._getframe().f_code.co_name
    logger.info(f">>> 开始执行节点: {func_name}")

    add_running_task(state.get("session_id", ""), func_name, state.get("is_stream", False))

    query = state.get("rewritten_query", "") or state.get("original_query", "")

    if not query:
        logger.warning("查询为空，跳过联网搜索")
        state["web_search_docs"] = []
        add_done_task(state.get("session_id", ""), func_name, state.get("is_stream", False))
        return state

    try:
        results = _run_async(_async_web_search(query))
        docs = _format_search_results(results or [])
        state["web_search_docs"] = docs
        logger.info(f"联网搜索完成，获取 {len(docs)} 条结果")

    except Exception as e:
        logger.error(f"联网搜索失败: {e}", exc_info=True)
        state["web_search_docs"] = []

    add_done_task(state.get("session_id", ""), func_name, state.get("is_stream", False))
    logger.info(f">>> 节点执行完成: {func_name}")

    return state


def _run_async(coro):
    """在独立线程里跑协程。

    直接在 FastAPI 的请求线程里 new_event_loop 会与已有事件循环冲突
    （日志里表现为 "Cannot run the event loop"），所以放到子线程里 asyncio.run。
    """
    box: dict = {}

    def runner():
        try:
            box["value"] = asyncio.run(coro)
        except Exception as exc:  # noqa: BLE001 - 交给调用线程处理
            box["error"] = exc

    thread = threading.Thread(target=runner, daemon=True)
    thread.start()
    thread.join(timeout=40)
    if "error" in box:
        raise box["error"]
    return box.get("value")


async def _async_web_search(query: str):
    """异步执行联网搜索（百炼 MCP 的 EnhancedSearch 服务）。

    2026-10-26 实测确认的三件事，别再照旧代码猜：
      - 传输是 Streamable HTTP，`.../mcps/WebSearch/sse` 已 404；
      - 服务名是 `EnhancedSearch`（不是 `WebSearch`），工具名是 `search_pro`；
      - 工具入参**只有 query**（多传 count 会被拒：`Only param:query is supported`）；
      - 鉴权头要 `Bearer <key>`，返回体是 CallToolResult.is_error（新版 mcp 已不用 isError）。
    """
    # 没配置就直接跳过：避免在没有 key 的环境里每次查询都等 30 秒超时
    if not mcp_config.mcp_base_url or not mcp_config.api_key:
        logger.warning(
            "未配置百炼 MCP（MCP_DASHSCOPE_BASE_URL / OPENAI_API_KEY），本次跳过联网搜索"
        )
        return []

    try:
        from agents.mcp import MCPServerStreamableHttp

        search_mcp = MCPServerStreamableHttp(
            name="web_search",
            params={
                "url": mcp_config.mcp_base_url,
                "headers": {"Authorization": f"Bearer {mcp_config.api_key}"},
                "timeout": 20,
            },
        )

        await search_mcp.connect()
        try:
            result = await search_mcp.call_tool(
                tool_name="search_pro",
                arguments={"query": query},
            )
        finally:
            await search_mcp.cleanup()

        if getattr(result, "is_error", False):
            text = result.content[0].text if result.content else ""
            logger.error(f"联网搜索被服务端拒绝: {text[:200]}")
            return []

        if result and result.content:
            raw_text = result.content[0].text
            data = json.loads(raw_text)
            return data.get("pages", [])

    except ImportError:
        # 不再返回假数据：mock 结果会污染 RRF 融合，让检索指标失去意义
        logger.warning(
            "MCP 客户端未安装（需要 openai-agents），本次联网搜索跳过；"
            "RRF 将只融合向量与 HyDE 两路"
        )
        return []
    except Exception as e:
        logger.error(f"MCP搜索失败: {e}")
        return []

    return []


def _format_search_results(results):
    """格式化搜索结果"""
    docs = []
    for item in results:
        snippet = item.get("snippet", "").strip()
        if snippet:
            docs.append({
                "title": item.get("title", ""),
                "url": item.get("url", ""),
                "snippet": snippet,
                "source": "web",
                "content": snippet
            })
    return docs