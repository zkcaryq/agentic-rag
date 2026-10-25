"""本地模型实现（FlagEmbedding）。

默认不启用：需要预先下载 BGE-M3 与 bge-reranker-large 到本地磁盘。
保留它是为了断网或需要完全离线时仍能跑，以及做供应商横向对比。
"""
import math

from app.conf.provider_config import (
    EmbeddingProviderConfig,
    RerankProviderConfig,
)
from app.lm.providers.base import EmbeddingProvider, EmbedRecord, RerankProvider


class LocalEmbeddingProvider(EmbeddingProvider):
    name = "local"
    supports_sparse = True

    def __init__(self, cfg: EmbeddingProviderConfig) -> None:
        self._cfg = cfg

    @property
    def dimension(self) -> int:
        return self._cfg.dimension

    def embed(self, texts: list[str], text_type: str = "document") -> list[EmbedRecord]:
        if not texts:
            return []
        # 延迟导入：没装 torch/flagembedding 的环境不应因为这个文件而在 import 阶段失败
        from app.lm.embedding_utils import generate_embeddings

        result = generate_embeddings(texts)
        dense_vectors = result["dense"]
        sparse_vectors = result["sparse"]
        records: list[EmbedRecord] = []
        for dense, sparse in zip(dense_vectors, sparse_vectors):
            records.append(
                EmbedRecord(
                    dense=[float(x) for x in dense],
                    sparse={int(k): float(v) for k, v in sparse.items()} or None,
                )
            )
        return records


class LocalRerankProvider(RerankProvider):
    name = "local"

    def __init__(self, cfg: RerankProviderConfig) -> None:
        self._cfg = cfg

    def rerank(
        self,
        query: str,
        documents: list[str],
        top_n: int | None = None,
    ) -> list[tuple[int, float]]:
        if not documents:
            return []
        from app.lm.reranker_utils import get_reranker_model

        model = get_reranker_model()
        pairs = [[query, text] for text in documents]
        scores = model.compute_score(pairs)
        if isinstance(scores, (int, float)):
            scores = [scores]
        # FlagReranker 输出的是 logit，统一到 0~1 后再交给上层断崖检测
        ranked = [
            (index, 1.0 / (1.0 + math.exp(-float(score))))
            for index, score in enumerate(scores)
        ]
        ranked.sort(key=lambda pair: pair[1], reverse=True)
        return ranked[: top_n or len(ranked)]
