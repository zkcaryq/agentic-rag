"""
HyDE (Hypothetical Document Embeddings) 检索节点
先让 LLM 生成假设性答案，再用答案的向量去检索真实文档。
"""
import sys

from app.conf.milvus_config import milvus_config
from app.core.logger import logger
from app.query_process.agent.state import QueryGraphState
from app.utils.task_utils import add_running_task, add_done_task
from app.utils.escape_milvus_string_utils import escape_milvus_string
from app.lm.lm_utils import get_llm_client
from app.clients.milvus_utils import get_milvus_client, search_chunks_with_fallback
from app.lm.provider_factory import get_embedding_provider


def node_search_embedding_hyde(state: QueryGraphState) -> QueryGraphState:
    """
    HyDE 检索节点

    原理：
        1. 让 LLM 根据用户问题生成一段假设性的理想答案
        2. 将假设答案向量化
        3. 用假设答案的向量去检索真实文档
        4. 返回检索结果

    优点：解决短查询语义稀疏问题，提高召回率
    """
    func_name = sys._getframe().f_code.co_name
    logger.info(f">>> 开始执行节点: {func_name}")

    session_id = state.get("session_id", "")
    is_stream = state.get("is_stream", False)
    add_running_task(session_id, func_name, is_stream)

    query = state.get("rewritten_query", "") or state.get("original_query", "")
    product_names = state.get("product_names", [])

    if not query:
        logger.warning("查询为空，跳过HyDE检索")
        state["hyde_embedding_chunks"] = []
        state["hyde_doc"] = ""
        add_done_task(session_id, func_name, is_stream)
        return state

    try:
        # 1. 生成假设性文档
        logger.info(f"生成假设性文档: {query[:50]}...")
        hyde_doc = step_generate_hypothetical_doc(query)
        if not hyde_doc:
            logger.warning("假设性文档生成失败，跳过HyDE检索")
            state["hyde_embedding_chunks"] = []
            state["hyde_doc"] = ""
            add_done_task(session_id, func_name, is_stream)
            return state

        state["hyde_doc"] = hyde_doc
        logger.info(f"假设性文档长度: {len(hyde_doc)} 字符")

        # 2. 向量化假设性文档
        provider = get_embedding_provider()
        record = provider.embed([hyde_doc], text_type="query")[0]

        # 3. 构建过滤条件
        expr = None
        if product_names and product_names != ["通用"]:
            quoted = ", ".join(f'"{escape_milvus_string(v)}"' for v in product_names)
            expr = f"product_name in [{quoted}]"
            logger.info(f"过滤条件: {expr}")

        # 4. 连接 Milvus
        client = get_milvus_client()
        if not client:
            logger.warning("Milvus未连接，返回空结果")
            state["hyde_embedding_chunks"] = []
            add_done_task(session_id, func_name, is_stream)
            return state

        # 5. 混合检索（无稀疏向量时自动降级为纯稠密；产品过滤落空则退回全库）
        chunks = search_chunks_with_fallback(
            client=client,
            collection_name=milvus_config.chunks_collection,
            dense_vector=record.dense,
            sparse_vector=record.sparse,
            expr=expr,
            limit=milvus_config.search_top_k,
            ranker_weights=(milvus_config.dense_weight, milvus_config.sparse_weight),
            label="HyDE检索",
        )

        state["hyde_embedding_chunks"] = chunks
        logger.info(f"HyDE检索到 {len(chunks)} 个相关切片")

    except Exception as e:
        logger.error(f"HyDE检索失败: {e}", exc_info=True)
        state["hyde_embedding_chunks"] = []
        state["hyde_doc"] = ""

    add_done_task(session_id, func_name, is_stream)
    logger.info(f">>> 节点执行完成: {func_name}")

    return state


def step_generate_hypothetical_doc(query: str) -> str:
    """生成假设性文档（HyDE 的第一步）。提示词在 prompts/hyde_prompt.prompt。"""
    try:
        from app.core.load_prompt import load_prompt

        llm = get_llm_client()
        prompt = load_prompt("hyde_prompt", rewritten_query=query)
        response = llm.invoke(prompt)
        return response.content.strip()
    except Exception as e:
        # 无 LLM 属于预期降级，不按错误级别刷屏
        logger.warning(f"生成假设性文档失败，本次跳过 HyDE: {e}")
        return ""
