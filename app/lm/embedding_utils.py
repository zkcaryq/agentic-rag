"""向量生成工具 - 本地 BGE-M3 模型。

只有在 EMBEDDING_PROVIDER=local 时才会被调用；默认走云端供应商。
模型路径全部来自 .env（BGE_M3_PATH），不写死任何人的本机目录。
"""
import os

from app.conf.embedding_config import embedding_config
from app.core.logger import logger

_bge_m3_ef = None


def get_bge_m3_ef():
    """获取 BGE-M3 模型单例。"""
    global _bge_m3_ef
    if _bge_m3_ef is None:
        # 延迟导入：默认不装本地推理依赖
        from pymilvus.model.hybrid import BGEM3EmbeddingFunction

        model_path = embedding_config.bge_m3_path
        if not model_path:
            raise RuntimeError(
                "EMBEDDING_PROVIDER=local 需要在 .env 配置 BGE_M3_PATH 指向本地 bge-m3 目录"
            )

        hf_home = os.getenv("HF_HOME")
        if hf_home:
            os.environ["HF_HOME"] = hf_home

        logger.info(f"加载本地 BGE-M3 模型: {model_path}")
        _bge_m3_ef = BGEM3EmbeddingFunction(
            model_name=model_path,
            device=embedding_config.bge_device or "cpu",
            use_fp16=embedding_config.bge_fp16,
            normalize_embeddings=True,
        )
        logger.info("BGE-M3 模型加载成功")
    return _bge_m3_ef


def generate_embeddings(texts):
    """为文本列表生成稠密 + 稀疏向量。"""
    if not isinstance(texts, list) or len(texts) == 0:
        raise ValueError("参数 texts 必须是包含文本的非空列表")

    logger.info(f"开始为 {len(texts)} 条文本生成 BGE-M3 向量...")

    try:
        model = get_bge_m3_ef()
        embeddings = model.encode_documents(texts)

        dense_vectors = [emb.tolist() for emb in embeddings["dense"]]

        # CSR 稀疏矩阵转成 {index: value}
        sparse_vectors = []
        for i in range(len(texts)):
            indices = embeddings["sparse"].indices[
                embeddings["sparse"].indptr[i]: embeddings["sparse"].indptr[i + 1]
            ].tolist()
            data = embeddings["sparse"].data[
                embeddings["sparse"].indptr[i]: embeddings["sparse"].indptr[i + 1]
            ].tolist()
            sparse_vectors.append(dict(zip(indices, data)))

        logger.info(f"成功生成 {len(texts)} 条向量（稠密维度: {len(dense_vectors[0])}）")
        return {"dense": dense_vectors, "sparse": sparse_vectors}

    except Exception as e:
        logger.error(f"BGE-M3 向量生成失败: {e}")
        raise
