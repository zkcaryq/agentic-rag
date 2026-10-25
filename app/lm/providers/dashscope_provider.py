"""百炼（DashScope）原生接口实现。

必须走原生接口而不是 OpenAI 兼容模式：
  - 稀疏向量（sparse_embedding）与 text_type 参数只在原生接口支持；
  - 兼容模式地址是 /compatible-mode/v1，调用会拿不到稀疏字段。
"""
from app.conf.provider_config import (
    EmbeddingProviderConfig,
    RerankProviderConfig,
)
from app.lm.providers.base import EmbeddingProvider, EmbedRecord, RerankProvider
from app.lm.providers.http_utils import post_json

_EMBEDDING_PATH = "/services/embeddings/text-embedding/text-embedding"
_RERANK_PATH = "/services/rerank/text-rerank/text-rerank"


class DashScopeEmbeddingProvider(EmbeddingProvider):
    name = "dashscope"

    def __init__(self, cfg: EmbeddingProviderConfig) -> None:
        self._cfg = cfg
        self.supports_sparse = cfg.want_sparse

    @property
    def dimension(self) -> int:
        return self._cfg.dimension

    def embed(self, texts: list[str], text_type: str = "document") -> list[EmbedRecord]:
        if not texts:
            return []
        if not self._cfg.api_key:
            raise RuntimeError("缺少百炼 API key：请配置 EMBEDDING_API_KEY 或 OPENAI_API_KEY")

        records: list[EmbedRecord] = []
        for start in range(0, len(texts), self._cfg.batch_size):
            batch = texts[start : start + self._cfg.batch_size]
            records.extend(self._embed_batch(batch, text_type))
        return records

    def _embed_batch(self, batch: list[str], text_type: str) -> list[EmbedRecord]:
        output_type = "dense&sparse" if self.supports_sparse else "dense"
        payload = {
            "model": self._cfg.model,
            "input": {"texts": batch},
            "parameters": {
                "dimension": self._cfg.dimension,
                "output_type": output_type,
                "text_type": "query" if text_type == "query" else "document",
            },
        }
        data = post_json(
            self._cfg.base_url + _EMBEDDING_PATH,
            self._cfg.api_key,
            payload,
            vendor="百炼",
        )

        embeddings = (data.get("output") or {}).get("embeddings") or []
        if len(embeddings) != len(batch):
            raise RuntimeError(
                f"百炼返回条数与输入不一致：输入 {len(batch)} 条，返回 {len(embeddings)} 条"
            )

        # 响应里的 text_index 才是与输入对齐的下标，不能假定顺序
        embeddings.sort(key=lambda item: item.get("text_index", 0))

        records: list[EmbedRecord] = []
        for item in embeddings:
            dense = item.get("embedding") or []
            if len(dense) != self._cfg.dimension:
                raise RuntimeError(
                    f"百炼稠密向量维度异常：期望 {self._cfg.dimension}，实际 {len(dense)}"
                )
            sparse = None
            if self.supports_sparse:
                raw_sparse = item.get("sparse_embedding") or []
                sparse = {int(entry["index"]): float(entry["value"]) for entry in raw_sparse}
                if not sparse:
                    raise RuntimeError(
                        "百炼未返回稀疏向量，请确认 EMBEDDING_MODEL 为 text-embedding-v3/v4 "
                        "且走的是 DashScope 原生接口"
                    )
            records.append(EmbedRecord(dense=[float(x) for x in dense], sparse=sparse))
        return records


class DashScopeRerankProvider(RerankProvider):
    name = "dashscope"

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
        if not self._cfg.api_key:
            raise RuntimeError("缺少百炼 API key：请配置 RERANK_API_KEY 或 OPENAI_API_KEY")

        payload = {
            "model": self._cfg.model,
            "input": {"query": query, "documents": documents},
            "parameters": {
                "return_documents": False,
                "top_n": top_n or len(documents),
            },
        }
        data = post_json(
            self._cfg.base_url + _RERANK_PATH,
            self._cfg.api_key,
            payload,
            vendor="百炼",
            timeout=self._cfg.timeout_seconds,
        )
        results = (data.get("output") or {}).get("results") or []
        # gte-rerank-v2 的 relevance_score 本身就在 0~1，直接使用
        ranked = [
            (int(item["index"]), float(item.get("relevance_score", 0.0))) for item in results
        ]
        ranked.sort(key=lambda pair: pair[1], reverse=True)
        return ranked
