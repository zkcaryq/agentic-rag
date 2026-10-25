"""
产品确认节点
提取产品名称，改写用户问题
"""
import sys
import json
from typing import List, Dict, Any

from app.core.load_prompt import load_prompt
from app.core.logger import logger
from app.query_process.agent.state import QueryGraphState
from app.utils.task_utils import add_running_task, add_done_task
from app.lm.lm_utils import get_llm_client
from app.clients.history import get_recent_messages, save_chat_message
from app.clients.milvus_utils import get_milvus_client


def node_product_confirm(state: QueryGraphState) -> QueryGraphState:
    """
    产品确认节点

    功能：
        1. 从用户问题中提取产品名称
        2. 改写问题为独立完整的查询语句
        3. 在Milvus中确认产品是否存在
        4. 如果无法确认，生成反问让用户选择
    """
    func_name = sys._getframe().f_code.co_name
    logger.info(f">>> 开始执行节点: {func_name}")

    add_running_task(state.get("session_id", ""), func_name, state.get("is_stream", False))

    session_id = state.get("session_id", "")
    original_query = state.get("original_query", "")
    is_stream = state.get("is_stream", False)

    # 步骤1：获取历史记录（同时存进 state，供答案节点拼多轮上下文）
    history = get_recent_messages(session_id, limit=10)
    state["history"] = [
        {"role": item.get("role", ""), "text": item.get("text", "")} for item in history
    ]

    # 步骤2：提取产品名称和改写问题
    extract_result = step_extract_info(original_query, history)
    product_names = extract_result.get("product_names", [])
    rewritten_query = extract_result.get("rewritten_query", original_query)

    state["rewritten_query"] = rewritten_query

    # 步骤3：如果有产品名称，在向量库中确认
    if product_names:
        confirmed_items = step_confirm_items(product_names)

        if confirmed_items:
            # 确认成功
            state["product_names"] = confirmed_items
            logger.info(f"确认产品: {confirmed_items}")
        else:
            # 无法确认，生成反问
            answer = f"您是想问以下哪个产品：{'、'.join(product_names[:3])}？请明确一下型号。"
            state["answer"] = answer
            state["product_names"] = []
            logger.info(f"需要用户确认: {answer}")

            # 保存助手回复
            save_chat_message(session_id, "assistant", answer, "", [])
            add_done_task(session_id, func_name, is_stream)
            return state
    else:
        # 无产品名称，使用文件标题作为兜底（简化处理）
        state["product_names"] = ["通用"]

    # 用户问题由 /query 入口统一落库，这里不再重复写，避免历史里出现两条同样的提问

    add_done_task(session_id, func_name, is_stream)
    logger.info(f">>> 节点执行完成: {func_name}")

    return state


def step_extract_info(query: str, history: List[Dict]) -> Dict[str, Any]:
    """提取产品名称和改写问题"""
    try:
        llm = get_llm_client(json_mode=True)

        # 构建历史文本
        history_text = ""
        for msg in history[-5:]:  # 最近5条
            history_text += f"{msg.get('role', '')}: {msg.get('text', '')}\n"

        prompt = load_prompt(
            "rewritten_query_and_productnames",
            history_text=history_text or "（无）",
            query=query,
        )

        response = llm.invoke(prompt)
        content = response.content.strip()

        # 清理JSON
        if content.startswith("```json"):
            content = content.replace("```json", "").replace("```", "")

        result = json.loads(content)
        return result

    except Exception as e:
        logger.error(f"提取信息失败: {e}")
        return {"product_names": [], "rewritten_query": query}


def step_confirm_items(product_names: List[str]) -> List[str]:
    """在 chunk 集合里按 product_name 模糊确认产品是否存在。

    之前这里查的是一个从未建过的 kb_product_names 集合，每次都走到"集合不存在"的兜底分支，
    等于没做确认。现在直接查真实的 chunk 集合，并回填库里的规范叫法。
    """
    try:
        client = get_milvus_client()
        if not client:
            return product_names

        from app.conf.milvus_config import milvus_config

        collection_name = milvus_config.chunks_collection
        if not client.has_collection(collection_name):
            logger.warning(f"集合 {collection_name} 不存在，跳过产品确认")
            return product_names

        confirmed: List[str] = []
        for item in product_names:
            safe = str(item).replace('"', "").strip()
            if not safe:
                continue
            rows = client.query(
                collection_name=collection_name,
                filter=f'product_name like "%{safe}%"',
                output_fields=["product_name"],
                limit=50,
            )
            if rows:
                names = {row.get("product_name") for row in rows if row.get("product_name")}
                # 优先用库里完全一致的叫法，找不到就保留模型提取的名称
                confirmed.append(next((n for n in names if safe in n), item))
            else:
                confirmed.append(item)

        return list(dict.fromkeys(confirmed))

    except Exception as e:
        logger.error(f"确认产品失败: {e}")
        return product_names