"""
重排序节点
用 Cross-Encoder（当前默认百炼 gte-rerank-v2）对 RRF 融合后的候选精排，
再用「断崖检测」自适应决定保留几条，而不是写死取前 5。
"""
import sys
from typing import List, Tuple

from app.core.logger import logger
from app.query_process.agent.state import QueryGraphState
from app.utils.task_utils import add_running_task, add_done_task

# 断崖检测参数集中在这里，方便调参与在面试时解释依据
MIN_SCORE = 0.30      # 绝对阈值：低于它直接丢弃
CLIFF_RATIO = 0.60    # 相邻分数比低于它视为断崖，就此截断
MAX_KEEP = 8          # 上限，避免上下文被撑爆
FALLBACK_FLOOR = 0.15  # 阈值过严导致一条不剩时，退一步保底取 2 条的软下限


def node_rerank(state: QueryGraphState) -> QueryGraphState:
    """重排序节点"""
    func_name = sys._getframe().f_code.co_name
    logger.info(f">>> 开始执行节点: {func_name}")

    session_id = state.get("session_id", "")
    is_stream = state.get("is_stream", False)
    add_running_task(session_id, func_name, is_stream)

    rrf_chunks = state.get("rrf_chunks", [])
    query = state.get("rewritten_query", "") or state.get("original_query", "")

    if not rrf_chunks or not query:
        logger.warning("无待重排文档或查询为空")
        state["reranked_docs"] = rrf_chunks[:MAX_KEEP]
        add_done_task(session_id, func_name, is_stream)
        return state

    logger.info(f"开始重排序: {len(rrf_chunks)} 个候选")

    try:
        from app.lm.provider_factory import get_rerank_provider

        provider = get_rerank_provider()
        documents = [chunk.get("content", "") for chunk in rrf_chunks]
        ranked = provider.rerank(query, documents, top_n=len(documents))
        if not ranked:
            raise RuntimeError("重排供应商返回空结果")

        for index, score in ranked:
            rrf_chunks[index]["rerank_score"] = score

        keep_indexes = select_by_cliff(ranked)
        selected = []
        for index in keep_indexes:
            doc = dict(rrf_chunks[index])
            doc["rerank_score"] = next(
                (score for i, score in ranked if i == index), 0.0
            )
            selected.append(doc)

        state["reranked_docs"] = selected
        if selected:
            logger.info(
                f"重排序完成: 候选 {len(rrf_chunks)} -> 保留 {len(selected)}，"
                f"最高分 {ranked[0][1]:.3f}，最低入选分 {selected[-1]['rerank_score']:.3f}"
            )
        else:
            logger.info("重排序完成: 无达标候选")

    except Exception as e:
        # 故障降级：重排不可用时按 RRF 顺序保底，保证问答链路不断
        logger.error(f"重排序失败，降级为 RRF 顺序: {e}", exc_info=True)
        state["reranked_docs"] = rrf_chunks[:MAX_KEEP]

    add_done_task(session_id, func_name, is_stream)
    logger.info(f">>> 节点执行完成: {func_name}")

    return state


def select_by_cliff(
    ranked: List[Tuple[int, float]],
    min_score: float = MIN_SCORE,
    ratio: float = CLIFF_RATIO,
    max_keep: int = MAX_KEEP,
    fallback_floor: float = FALLBACK_FLOOR,
) -> List[int]:
    """断崖检测式动态 Top-K。

    ranked 已按分数降序（供应商实现负责统一到 0~1）。规则：
      1. 第一条低于 min_score 即停止（后面的只会更低）；
      2. 相邻分数比值低于 ratio 视为断崖，就此截断；
      3. 最多保留 max_keep 条；
      4. 一条都不剩但 top1 仍高于 fallback_floor 时，保底给 2 条候选，
         由 answer 的 prompt（资料不足要说明）兜底，避免直接答"没找到"。
    """
    if not ranked:
        return []

    kept: List[int] = []
    prev_score = None
    for index, score in ranked:
        if score < min_score:
            break
        if prev_score is not None and prev_score > 0 and (score / prev_score) < ratio:
            logger.info(
                f"断崖检测在第 {len(kept) + 1} 条截断："
                f"{prev_score:.3f} -> {score:.3f}（比值 {score / prev_score:.2f}）"
            )
            break
        kept.append(index)
        prev_score = score
        if len(kept) >= max_keep:
            break

    if not kept and ranked[0][1] >= fallback_floor:
        logger.warning(
            f"所有候选均低于阈值 {min_score}，按软下限 {fallback_floor} 保底取前 2 条"
        )
        kept = [index for index, _ in ranked[:2]]

    return kept