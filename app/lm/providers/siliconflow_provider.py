"""SiliconFlow 实现（OpenAI 兼容 embeddings + /rerank 扩展接口）。

注意：这条路的 bge-m3 只返回稠密向量，稀疏侧为 None，上层会自动降级为
dense-only 检索。想保留「稠密+稀疏双向量」就必须用百炼。
"""
from app.conf.provider_config import (
    EmbeddingProviderConfig,
    RerankProviderConfig,
)
from app.lm.providers.base import EmbeddingProvider, EmbedRecord, RerankProvider
from app.lm.providers.http_utils import post_json


class SiliconFlowEmbeddingProvider(EmbeddingProvider):
    name = "siliconflow"
    supports_sparse = False

    def __init__(self, cfg: EmbeddingProviderConfig) -> None:
        self._cfg = cfg

    @property
    def dimension(self) -> int:
        return self._cfg.dimension

    def embed(self, texts: list[str], text_type: str = "document") -> list[EmbedRecord]:
        if not texts:
            return []
        if not self._cfg.api_key:
            raise RuntimeError("缺少 SiliconFlow API key：请配置 EMBEDDING_API_KEY")

        records: list[EmbedRecord] = []
        for start in range(0, len(texts), self._cfg.batch_size):
            batch = texts[start : start + self._cfg.batch_size]
            payload = {
                "model": self._cfg.model,
                "input": batch,
                "encoding_format": "float",
            }
            data = post_json(
                self._cfg.base_url + "/embeddings",
                self._cfg.api_key,
                payload,
                vendor="SiliconFlow",
            )
            items = data.get("data") or []
            if len(items) != len(batch):
                raise RuntimeError(
                    f"SiliconFlow 返回条数与输入不一致：输入 {len(batch)} 条，返回 {len(items)} 条"
                )
            items.sort(key=lambda item: item.get("index", 0))
            for item in items:
                dense = [float(x) for x in item.get("embedding") or []]
                if len(dense) != self._cfg.dimension:
                    raise RuntimeError(
                        f"SiliconFlow 稠密向量维度异常：期望 {self._cfg.dimension}，实际 {len(dense)}"
                    )
                records.append(EmbedRecord(dense=dense, sparse=None))
        return records


class SiliconFlowRerankProvider(RerankProvider):
    name = "siliconflow"

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
            raise RuntimeError("缺少 SiliconFlow API key：请配置 RERANK_API_KEY")

        payload = {
            "model": self._cfg.model,
            "query": query,
            "documents": documents,
            "top_n": top_n or len(documents),
            "return_documents": False,
        }
        data = post_json(
            self._cfg.base_url + "/rerank",
            self._cfg.api_key,
            payload,
            vendor="SiliconFlow",
            timeout=self._cfg.timeout_seconds,
        )
        results = data.get("results") or []
        # 兼容两种字段名（SiliconFlow 文档与 OpenAI 兼容层写法不一致）
        ranked = [
            (
                int(item["index"]),
                float(item.get("relevance_score", item.get("score", 0.0))),
            )
            for item in results
        ]
        ranked.sort(key=lambda pair: pair[1], reverse=True)
        return ranked
