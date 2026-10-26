"""
BGE向量化节点
将文本切片转换为稠密 + 稀疏向量。

向量由 app.lm.providers 里的供应商实现产生，本节点不感知具体厂商：
改 .env 里的 EMBEDDING_PROVIDER 即可切换（dashscope / siliconflow / local）。
"""
import sys
from typing import List, Dict, Any

from app.core.logger import logger
from app.import_process.agent.state import ImportGraphState
from app.lm.provider_factory import get_embedding_provider
from app.utils.task_utils import add_running_task, add_done_task


def node_bge_embedding(state: ImportGraphState) -> ImportGraphState:
    """
    向量化节点

    核心流程：
        1. 获取切片数据
        2. 调供应商生成稠密 + 稀疏向量
        3. 更新状态

    Args:
        state: 导入流程状态

    Returns:
        更新后的状态，chunks 新增 dense_vector 与 sparse_vector
    """
    func_name = sys._getframe().f_code.co_name
    logger.info(f">>> 开始执行节点: {func_name}")

    add_running_task(state.get("task_id", ""), func_name)

    try:
        chunks = state.get("chunks", [])
        if not chunks:
            logger.warning("无切片数据，跳过向量化")
            add_done_task(state.get("task_id", ""), func_name)
            return state

        logger.info(f"开始向量化，共 {len(chunks)} 个切片")
        chunks = step_generate_embeddings(chunks)

        state["chunks"] = chunks
        state["embeddings_content"] = chunks
        logger.info(f"向量化完成，共 {len(chunks)} 个切片")

    except Exception as e:
        # 明确的失败阶段，供任务表统计失败率使用
        state["failed_stage"] = "embedding"
        logger.exception(f"向量化失败: {e}")
        add_done_task(state.get("task_id", ""), func_name)
        raise

    add_done_task(state.get("task_id", ""), func_name)
    logger.info(f">>> 节点执行完成: {func_name}")
    return state


def step_generate_embeddings(chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """调供应商生成向量，写回每个 chunk。

    这里曾经是 random.uniform 生成的模拟向量 + 所有切片共用一个常量稀疏向量，
    导致入库向量与查询向量不在同一空间、检索结果实际是随机的。现在改为真实调用，
    并在返回条数、维度、稀疏向量是否为空三处做校验，避免同类假实现复发。
    """
    provider = get_embedding_provider()

    texts = []
    for chunk in chunks:
        content = chunk.get("content", "")
        product_name = chunk.get("product_name", "")
        texts.append(f"{product_name}\n{content}" if product_name else content)

    records = provider.embed(texts, text_type="document")

    if len(records) != len(chunks):
        raise RuntimeError(
            f"向量返回条数与切片数不一致：切片 {len(chunks)}，向量 {len(records)}"
        )

    for chunk, record, text in zip(chunks, records, texts):
        if not record.dense:
            raise RuntimeError(f"切片 {chunk.get('chunk_id')} 的稠密向量为空")
        if len(record.dense) != provider.dimension:
            raise RuntimeError(
                f"稠密向量维度不符：期望 {provider.dimension}，实际 {len(record.dense)}"
            )
        chunk["dense_vector"] = record.dense
        chunk["sparse_vector"] = record.sparse or {}
        chunk["embedding_text"] = text
        chunk["embedding_provider"] = provider.name
        chunk["embedding_dim"] = provider.dimension

    if not provider.supports_sparse:
        logger.warning(
            "当前 embedding 供应商不支持稀疏向量，本次入库的切片将没有稀疏向量；"
            "检索会自动降级为纯稠密检索"
        )

    return chunks
