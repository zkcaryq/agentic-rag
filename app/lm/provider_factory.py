"""按 .env 装配 embedding / rerank provider，进程内单例。

业务代码只调 get_embedding_provider() / get_rerank_provider()，
换供应商只改 EMBEDDING_PROVIDER / RERANK_PROVIDER。
"""
from functools import lru_cache

from app.conf.provider_config import (
    embedding_provider_config,
    rerank_provider_config,
)
from app.core.logger import logger
from app.lm.providers.base import EmbeddingProvider, RerankProvider

_SUPPORTED = ("dashscope", "siliconflow", "local", "lexical")


@lru_cache(maxsize=1)
def get_embedding_provider() -> EmbeddingProvider:
    name = embedding_provider_config.provider
    if name not in _SUPPORTED:
        raise ValueError(
            f"不支持的 EMBEDDING_PROVIDER={name}，可选：{'/'.join(_SUPPORTED)}"
        )

    if name == "dashscope":
        from app.lm.providers.dashscope_provider import DashScopeEmbeddingProvider

        provider: EmbeddingProvider = DashScopeEmbeddingProvider(embedding_provider_config)
    elif name == "siliconflow":
        from app.lm.providers.siliconflow_provider import SiliconFlowEmbeddingProvider

        provider = SiliconFlowEmbeddingProvider(embedding_provider_config)
    elif name == "lexical":
        # 本地词法向量：无依赖、无 key，仅用于演示与链路验证
        from app.lm.providers.lexical_provider import LexicalEmbeddingProvider

        provider = LexicalEmbeddingProvider(embedding_provider_config)
    elif name == "local":
        # 本地推理依赖默认不装（tensor 栈体积大、在部分 Windows 上 DLL 会加载失败）
        try:
            import torch  # noqa: F401
        except Exception as exc:  # noqa: BLE001
            raise RuntimeError(
                "EMBEDDING_PROVIDER=local 需要本地推理依赖，请先执行："
                "uv add flagembedding pymilvus-model torch torchvision"
            ) from exc
        from app.lm.providers.local_provider import LocalEmbeddingProvider

        provider = LocalEmbeddingProvider(embedding_provider_config)

    logger.info(
        f"Embedding provider={provider.name} model={embedding_provider_config.model} "
        f"dim={provider.dimension} sparse={provider.supports_sparse}"
    )
    if not provider.supports_sparse:
        logger.warning(
            "当前 embedding 供应商不支持稀疏向量，检索将降级为 dense-only；"
            "如需「稠密+稀疏双向量」请使用 EMBEDDING_PROVIDER=dashscope"
        )
    return provider


@lru_cache(maxsize=1)
def get_rerank_provider() -> RerankProvider:
    name = rerank_provider_config.provider
    rerank_supported = ("dashscope", "siliconflow", "local", "none")
    if name not in rerank_supported:
        raise ValueError(
            f"不支持的 RERANK_PROVIDER={name}，可选：{'/'.join(rerank_supported)}"
        )

    if name == "none":
        from app.lm.providers.base import NoopRerankProvider

        logger.info("Rerank provider=none（不重排，保留 RRF 顺序）")
        return NoopRerankProvider()

    if name == "dashscope":
        from app.lm.providers.dashscope_provider import DashScopeRerankProvider

        provider: RerankProvider = DashScopeRerankProvider(rerank_provider_config)
    elif name == "siliconflow":
        from app.lm.providers.siliconflow_provider import SiliconFlowRerankProvider

        provider = SiliconFlowRerankProvider(rerank_provider_config)
    else:
        try:
            import torch  # noqa: F401
        except Exception as exc:  # noqa: BLE001
            raise RuntimeError(
                "RERANK_PROVIDER=local 需要本地推理依赖，请先执行："
                "uv add flagembedding pymilvus-model torch torchvision"
            ) from exc
        from app.lm.providers.local_provider import LocalRerankProvider

        provider = LocalRerankProvider(rerank_provider_config)

    logger.info(f"Rerank provider={provider.name} model={rerank_provider_config.model}")
    return provider


def reset_providers() -> None:
    """测试或热切换配置后清空单例。"""
    get_embedding_provider.cache_clear()
    get_rerank_provider.cache_clear()
