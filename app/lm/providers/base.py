"""Embedding / Rerank 供应商抽象。

业务节点只依赖这里的接口，不感知具体供应商；切换供应商只改 .env。
"""
from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass
class EmbedRecord:
    """单条文本的向量结果。

    sparse 为 None 表示该供应商不支持稀疏向量，上层应降级为 dense-only 检索。
    """

    dense: list[float]
    sparse: dict[int, float] | None = None


class EmbeddingProvider(ABC):
    name: str = "base"
    supports_sparse: bool = False

    @property
    @abstractmethod
    def dimension(self) -> int:
        """稠密向量维度，用于建 collection 与校验。"""

    @abstractmethod
    def embed(self, texts: list[str], text_type: str = "document") -> list[EmbedRecord]:
        """text_type: 'document' 入库 / 'query' 查询。

        部分供应商（百炼）对查询和文档用不同的编码方式，不支持区分时可忽略。
        """


class RerankProvider(ABC):
    name: str = "base"

    @abstractmethod
    def rerank(
        self,
        query: str,
        documents: list[str],
        top_n: int | None = None,
    ) -> list[tuple[int, float]]:
        """返回 [(documents 中的原始下标, 分数)]，按分数降序。

        分数必须在实现里统一到 0~1，上层断崖检测直接比较，不再各处理一套量纲。
        """


class NoopRerankProvider(RerankProvider):
    """不重排：原样保留顺序，所有分数相同。

    用于本地演示或供应商不可用时的显式降级——保证 downstream 的
    断崖检测不会因为分数单调递减而只保留 1 条。
    """

    name = "none"

    def rerank(
        self,
        query: str,
        documents: list[str],
        top_n: int | None = None,
    ) -> list[tuple[int, float]]:
        return [(index, 1.0) for index in range(len(documents))]
