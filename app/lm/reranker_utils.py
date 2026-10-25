"""Reranker 工具 - 本地 BGE-Reranker。

只有在 RERANK_PROVIDER=local 时才会被调用；默认走云端供应商。
是否强制离线由 .env 的 HF_HUB_OFFLINE 控制，不写死任何人的本机路径。
"""
import os

from app.conf.reranker_config import reranker_config
from app.core.logger import logger

if os.getenv("HF_HUB_OFFLINE") == "1":
    # 必须在导入 FlagEmbedding 之前设置
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ["HF_HUB_OFFLINE"] = "1"

_hf_home = os.getenv("HF_HOME")
if _hf_home:
    os.environ["HF_HOME"] = _hf_home

from FlagEmbedding import FlagReranker  # noqa: E402

_reranker_model = None


def get_reranker_model():
    """获取 Reranker 模型单例。"""
    global _reranker_model
    if _reranker_model is None:
        model_path = reranker_config.bge_reranker_large
        if not model_path:
            raise RuntimeError(
                "RERANK_PROVIDER=local 需要在 .env 配置 BGE_RERANKER_LARGE 指向本地模型目录"
            )

        logger.info(f"正在加载 Reranker 模型: {model_path}")
        try:
            _reranker_model = FlagReranker(
                model_name_or_path=model_path,
                device=reranker_config.bge_reranker_device or "cpu",
                use_fp16=reranker_config.bge_reranker_fp16,
            )
            logger.info("Reranker 模型加载成功")
        except Exception as e:
            logger.error(f"Reranker 模型加载失败: {e}")
            raise
    return _reranker_model


def rerank(query: str, documents: list) -> list:
    """对文档重排序，返回按分数降序的文档列表。"""
    if not documents:
        return []

    try:
        model = get_reranker_model()
        texts = [doc.get("content", "") for doc in documents]
        scores = model.compute_score([[query, text] for text in texts])
        if isinstance(scores, (int, float)):
            scores = [scores]

        for doc, score in zip(documents, scores):
            doc["rerank_score"] = float(score)

        documents.sort(key=lambda x: x.get("rerank_score", 0), reverse=True)
        logger.info(f"Rerank 完成，最高分: {documents[0].get('rerank_score', 0):.4f}")
        return documents

    except Exception as e:
        logger.error(f"Rerank 失败: {e}")
        return documents
