"""评测用的检索管线。

把线上链路拆成四条可独立对比的管线，共用同一批候选池与同一个 golden set：
  dense   纯稠密（基线）
  hybrid  稠密 + 稀疏混合（Milvus WeightedRanker）
  hyde    hybrid + HyDE 多路召回 + RRF 融合
  full    hyde + Cross-Encoder 重排 + 断崖动态 TopK

简历里的「召回准确率提升 X%」就是这个对比的产出，基线口径必须写进报告。
"""
from typing import Any, Dict, List

from app.clients.milvus_utils import (
    dense_search,
    get_milvus_client,
    search_with_optional_sparse,
)
from app.conf.milvus_config import milvus_config
from app.core.logger import logger
from app.lm.provider_factory import get_embedding_provider, get_rerank_provider

PIPELINES = ("dense", "hybrid", "hyde", "full")


def _flatten(res) -> List[Dict[str, Any]]:
    """把 Milvus 的返回结构摊平成评测用的文档结构。"""
    chunks = res[0] if res else []
    documents: List[Dict[str, Any]] = []
    for chunk in chunks:
        entity = chunk.get("entity") or {}
        documents.append(
            {
                "biz_chunk_id": entity.get("biz_chunk_id", ""),
                "file_title": entity.get("file_title", ""),
                "title": entity.get("title", ""),
                "page_no": entity.get("page_no", 0),
                "content": entity.get("content", ""),
                "score": chunk.get("distance", 0.0),
            }
        )
    return documents


def _search(query: str, *, use_sparse: bool, limit: int) -> List[Dict[str, Any]]:
    provider = get_embedding_provider()
    record = provider.embed([query], text_type="query")[0]

    client = get_milvus_client()
    if client is None:
        raise RuntimeError("Milvus 未连接，请先配置 MILVUS_URL / MILVUS_TOKEN")

    collection = milvus_config.chunks_collection
    if use_sparse and record.sparse:
        res = search_with_optional_sparse(
            client=client,
            collection_name=collection,
            dense_vector=record.dense,
            sparse_vector=record.sparse,
            limit=limit,
            ranker_weights=(milvus_config.dense_weight, milvus_config.sparse_weight),
        )
    else:
        res = dense_search(
            client=client,
            collection_name=collection,
            dense_vector=record.dense,
            limit=limit,
        )
    return _flatten(res)


def _fuse(result_lists: List[List[Dict[str, Any]]], k: int = 60) -> List[Dict[str, Any]]:
    """对多路结果做 RRF 融合（与线上 node_rrf 同一套公式）。"""
    scores: Dict[str, float] = {}
    docs: Dict[str, Dict[str, Any]] = {}

    for documents in result_lists:
        for rank, doc in enumerate(documents, start=1):
            doc_id = doc.get("biz_chunk_id") or doc.get("content", "")[:80]
            if not doc_id:
                continue
            scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (k + rank)
            docs.setdefault(doc_id, doc)

    fused = []
    for doc_id, score in sorted(scores.items(), key=lambda item: item[1], reverse=True):
        item = dict(docs[doc_id])
        item["rrf_score"] = score
        fused.append(item)
    return fused


def _rerank(query: str, documents: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    from app.query_process.agent.nodes.node_rerank import select_by_cliff

    if not documents:
        return []
    provider = get_rerank_provider()
    ranked = provider.rerank(
        query, [doc.get("content", "") for doc in documents], top_n=len(documents)
    )
    if not ranked:
        return documents

    keep = select_by_cliff(ranked)
    score_map = dict(ranked)
    selected = []
    for index in keep:
        item = dict(documents[index])
        item["rerank_score"] = score_map.get(index, 0.0)
        selected.append(item)
    return selected


def retrieve(query: str, pipeline: str, top_k: int = 5) -> List[Dict[str, Any]]:
    """按指定管线检索，返回 top_k 文档。"""
    if pipeline not in PIPELINES:
        raise ValueError(f"未知管线 {pipeline}，可选 {PIPELINES}")

    if pipeline == "dense":
        return _search(query, use_sparse=False, limit=top_k)

    base = _search(query, use_sparse=True, limit=top_k * 2)
    if pipeline == "hybrid":
        return base[:top_k]

    from app.query_process.agent.nodes.node_search_embedding_hyde import (
        step_generate_hypothetical_doc,
    )

    hyde_doc = step_generate_hypothetical_doc(query)
    hyde_results = _search(hyde_doc, use_sparse=True, limit=top_k * 2) if hyde_doc else []
    fused = _fuse([base, hyde_results])

    if pipeline == "hyde":
        return fused[:top_k]

    reranked = _rerank(query, fused)
    if not reranked:
        logger.warning("重排无达标候选，回退到 RRF 顺序")
        return fused[:top_k]
    return reranked[:top_k]
