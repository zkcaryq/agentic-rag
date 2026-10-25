"""
向量检索节点
在 Milvus 中搜索相关文档切片。

向量由 app.lm.providers 提供，本节点不感知具体供应商；
供应商不支持稀疏向量时，search_with_optional_sparse 会自动降级为纯稠密检索。
"""
import sys

from app.conf.milvus_config import milvus_config
from app.core.logger import logger
from app.query_process.agent.state import QueryGraphState
from app.utils.task_utils import add_running_task, add_done_task
from app.clients.milvus_utils import get_milvus_client, search_with_optional_sparse
from app.lm.provider_factory import get_embedding_provider


def node_search_embedding(state: QueryGraphState) -> QueryGraphState:
    """
    向量检索节点

    功能：
        1. 将改写后的问题向量化（query 侧，部分供应商会与 document 侧区分编码）
        2. 在 Milvus 中做稠密 + 稀疏混合检索
        3. 返回 Top-K 结果
    """
    func_name = sys._getframe().f_code.co_name
    logger.info(f">>> 开始执行节点: {func_name}")

    session_id = state.get("session_id", "")
    is_stream = state.get("is_stream", False)
    add_running_task(session_id, func_name, is_stream)

    query = state.get("rewritten_query", "") or state.get("original_query", "")
    product_names = state.get("product_names", [])

    if not query:
        logger.warning("查询为空，跳过检索")
        add_done_task(session_id, func_name, is_stream)
        return state

    try:
        # 1. 向量化查询
        logger.info(f"向量化查询: {query[:50]}...")
        provider = get_embedding_provider()
        record = provider.embed([query], text_type="query")[0]

        # 2. 构建过滤条件
        expr = None
        if product_names and product_names != ["通用"]:
            quoted = ", ".join(f'"{v}"' for v in product_names)
            expr = f"product_name in [{quoted}]"
            logger.info(f"过滤条件: {expr}")

        # 3. 连接 Milvus
        client = get_milvus_client()
        if not client:
            logger.warning("Milvus未连接，返回空结果")
            state["embedding_chunks"] = []
            add_done_task(session_id, func_name, is_stream)
            return state

        # 4. 混合检索（无稀疏向量时自动降级为纯稠密）
        res = search_with_optional_sparse(
            client=client,
            collection_name=milvus_config.chunks_collection,
            dense_vector=record.dense,
            sparse_vector=record.sparse,
            expr=expr,
            limit=milvus_config.search_top_k,
            ranker_weights=(milvus_config.dense_weight, milvus_config.sparse_weight),
        )

        chunks = res[0] if res else []
        state["embedding_chunks"] = chunks
        logger.info(f"检索到 {len(chunks)} 个相关切片")

    except Exception as e:
        logger.error(f"向量检索失败: {e}", exc_info=True)
        state["embedding_chunks"] = []

    add_done_task(session_id, func_name, is_stream)
    logger.info(f">>> 节点执行完成: {func_name}")

    return state
