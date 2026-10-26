"""Embedding / Rerank 供应商配置。

切换供应商只改 .env 里的 *_PROVIDER，业务节点代码不动。

  EMBEDDING_PROVIDER=dashscope | siliconflow | local
  RERANK_PROVIDER=dashscope | siliconflow | local

dashscope 是唯一能同时给出稠密 + 稀疏向量的路径（且必须是 DashScope 原生接口，
OpenAI 兼容接口不支持稀疏输出与 text_type）。siliconflow 的 bge-m3 只有稠密，
稀疏侧为 None，上层会自动降级为 dense-only 检索。
"""
from dataclasses import dataclass
import os

from dotenv import load_dotenv

load_dotenv()


def _env_bool(name: str, default: bool = False) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in ("1", "true", "yes", "on")


@dataclass
class EmbeddingProviderConfig:
    provider: str
    model: str
    base_url: str
    api_key: str
    dimension: int
    batch_size: int
    want_sparse: bool


@dataclass
class RerankProviderConfig:
    provider: str
    model: str
    base_url: str
    api_key: str
    timeout_seconds: int
    # 联网结果的分倍率。<1 表示"私有知识库优先"：通用教程网页在 Cross-Encoder
    # 下常常比本型号手册更像"直接答案"（实测 web 0.88 vs 手册落榜），不加权就会
    # 让公网内容挤占上下文，与"企业私有化知识问答"的定位相反。设 1.0 即不降权。
    web_score_penalty: float


# 百炼原生接口（注意：不是兼容模式地址，兼容模式 /compatible-mode/v1 不支持稀疏向量）
_DASHSCOPE_NATIVE_BASE = "https://dashscope.aliyuncs.com/api/v1"
# SiliconFlow 的 OpenAI 兼容地址（rerank 走同域下的 /rerank 扩展接口）
_SILICONFLOW_BASE = "https://api.siliconflow.cn/v1"

embedding_provider_config = EmbeddingProviderConfig(
    provider=os.getenv("EMBEDDING_PROVIDER", "dashscope").strip().lower(),
    model=os.getenv("EMBEDDING_MODEL", "text-embedding-v4"),
    base_url=os.getenv("EMBEDDING_BASE_URL", _DASHSCOPE_NATIVE_BASE),
    # 默认复用已有的 OPENAI_API_KEY（现有 .env 里就是百炼的 key）
    api_key=os.getenv("EMBEDDING_API_KEY") or os.getenv("OPENAI_API_KEY", ""),
    dimension=int(os.getenv("EMBEDDING_DIM", "1024")),
    batch_size=int(os.getenv("EMBEDDING_BATCH_SIZE", "10")),
    want_sparse=_env_bool("EMBEDDING_WANT_SPARSE", True),
)

rerank_provider_config = RerankProviderConfig(
    provider=os.getenv("RERANK_PROVIDER", "dashscope").strip().lower(),
    model=os.getenv("RERANK_MODEL", "gte-rerank-v2"),
    base_url=os.getenv("RERANK_BASE_URL", _DASHSCOPE_NATIVE_BASE),
    api_key=(
        os.getenv("RERANK_API_KEY")
        or os.getenv("EMBEDDING_API_KEY")
        or os.getenv("OPENAI_API_KEY", "")
    ),
    timeout_seconds=int(os.getenv("RERANK_TIMEOUT_SECONDS", "30")),
    web_score_penalty=float(os.getenv("RERANK_WEB_SCORE_PENALTY", "0.7")),
)
