"""
RRF融合排序节点
将多路召回的结果进行融合排序
"""
import sys
from typing import List, Dict, Any, Tuple

from app.core.logger import logger
from app.query_process.agent.state import QueryGraphState
from app.utils.task_utils import add_running_task, add_done_task


def node_rrf(state: QueryGraphState) -> QueryGraphState:
    """
    RRF (Reciprocal Rank Fusion) 融合排序节点

    功能：
        1. 合并多路召回结果（向量检索、HyDE检索、联网搜索）
        2. 使用RRF算法重新排序
        3. 去重并返回Top-K结果
    """
    func_name = sys._getframe().f_code.co_name
    logger.info(f">>> 开始执行节点: {func_name}")

    add_running_task(state.get("session_id", ""), func_name, state.get("is_stream", False))

    # 获取各路召回结果
    embedding_chunks = state.get("embedding_chunks", [])
    hyde_chunks = state.get("hyde_embedding_chunks", [])
    web_docs = state.get("web_search_docs", [])

    logger.info(f"RRF输入: 向量检索={len(embedding_chunks)}, HyDE检索={len(hyde_chunks)}, 联网搜索={len(web_docs)}")

    # 如果没有结果，直接返回
    if not embedding_chunks and not hyde_chunks and not web_docs:
        state["rrf_chunks"] = []
        add_done_task(state.get("session_id", ""), func_name, state.get("is_stream", False))
        return state

    # 标准化文档格式
    norm_embedding = _normalize_chunks(embedding_chunks, "embedding")
    norm_hyde = _normalize_chunks(hyde_chunks, "hyde")
    norm_web = _normalize_web_docs(web_docs)

    # RRF融合（可调整权重）
    source_weights = [
        (norm_embedding, 1.0),
        (norm_hyde, 0.8),
        (norm_web, 0.6),
    ]

    fused_results = _reciprocal_rank_fusion(source_weights, k=60, max_results=10)

    # 保留 RRF 分数：之前这里把分数直接丢掉了，导致无法解释排序依据、也无法做评测对比
    rrf_chunks = []
    for doc, score in fused_results:
        item = dict(doc)
        item["rrf_score"] = score
        rrf_chunks.append(item)

    state["rrf_chunks"] = rrf_chunks
    logger.info(f"RRF输出: {len(rrf_chunks)} 个文档")

    add_done_task(state.get("session_id", ""), func_name, state.get("is_stream", False))
    logger.info(f">>> 节点执行完成: {func_name}")

    return state


def _normalize_chunks(chunks: List[Dict], source: str) -> List[Dict]:
    """标准化文档格式。

    Milvus 的 hybrid_search 返回 {"id": 物理主键, "distance": 分数, "entity": {...}}，
    entity 里才是业务字段。这里把业务字段摊平，供 RRF 之后的溯源与引用使用。
    """
    result = []
    for i, chunk in enumerate(chunks):
        if not chunk:
            continue
        if isinstance(chunk, dict):
            entity = chunk.get("entity") if isinstance(chunk.get("entity"), dict) else {}
            meta = entity or chunk

            content = meta.get("content", "")
            if not content and isinstance(chunk.get("entity"), dict):
                content = chunk["entity"].get("content", "")

            biz_chunk_id = str(meta.get("biz_chunk_id") or "")
            doc_id = biz_chunk_id or chunk.get("biz_chunk_id") or chunk.get("chunk_id") or chunk.get("id") or f"{source}_{i}"
            score = chunk.get("distance", chunk.get("score", 0.0))
            result.append(
                {
                    "id": str(doc_id),
                    "biz_chunk_id": biz_chunk_id,
                    "content": content,
                    "title": meta.get("title", ""),
                    "parent_title": meta.get("parent_title", ""),
                    "file_title": meta.get("file_title", ""),
                    "product_name": meta.get("product_name", ""),
                    "page_no": meta.get("page_no", 0),
                    "score": score,
                    "source": source,
                    "raw": chunk,
                }
            )
        elif hasattr(chunk, "entity"):
            doc_id = getattr(chunk, "id", f"{source}_{i}")
            entity = getattr(chunk, "entity", {})
            content = entity.get("content", "") if isinstance(entity, dict) else ""
            result.append(
                {
                    "id": str(doc_id),
                    "biz_chunk_id": str(entity.get("biz_chunk_id", "")) if isinstance(entity, dict) else "",
                    "content": content,
                    "title": entity.get("title", "") if isinstance(entity, dict) else "",
                    "file_title": entity.get("file_title", "") if isinstance(entity, dict) else "",
                    "page_no": entity.get("page_no", 0) if isinstance(entity, dict) else 0,
                    "score": getattr(chunk, "distance", 0.0),
                    "source": source,
                    "raw": chunk,
                }
            )
    return result


def _normalize_web_docs(docs: List[Dict]) -> List[Dict]:
    """标准化联网搜索结果格式。

    网页摘要里常带图片占位和示例链接，不清就会跟着上下文进答案（实测答案尾部
    出现过【图片】+ 四行 <https://example.com/...jpg>）。
    """
    from app.utils.text_clean_utils import clean_web_snippet

    result = []
    for i, doc in enumerate(docs):
        content = clean_web_snippet(doc.get("snippet", doc.get("content", "")))
        if content:
            result.append({
                "id": f"web_{i}",
                "content": content,
                "score": 0.5,
                "source": "web",
                "raw": doc,
                "title": doc.get("title", ""),
                "url": doc.get("url", "")
            })
    return result


def _reciprocal_rank_fusion(
        source_weights: List[Tuple[List[Dict], float]],
        k: int = 60,
        max_results: int = None
) -> List[Tuple[Dict, float]]:
    """带权重的RRF算法"""
    score_map = {}
    doc_map = {}

    for docs, weight in source_weights:
        for rank, doc in enumerate(docs, start=1):
            doc_id = doc.get("id", "")
            if not doc_id:
                continue
            rrf_score = weight * (1.0 / (k + rank))
            score_map[doc_id] = score_map.get(doc_id, 0.0) + rrf_score
            if doc_id not in doc_map:
                doc_map[doc_id] = doc

    merged = [(doc_map[doc_id], score) for doc_id, score in score_map.items()]
    merged.sort(key=lambda x: x[1], reverse=True)

    if max_results:
        merged = merged[:max_results]

    return merged