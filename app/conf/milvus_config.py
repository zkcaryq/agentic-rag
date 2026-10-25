# 导入核心依赖（和其他配置类共用，只需导入一次）
from dataclasses import dataclass
import os
from dotenv import load_dotenv

# 提前加载.env配置文件（全局执行一次即可，无需重复写）
load_dotenv()

# ===================== 其他配置类（LLM/Embedding）可放在上方，保持原有代码不变 =====================
# ... 你的LLMConfig、EmbeddingConfig代码 ...

# 定义Milvus向量数据库配置类
@dataclass
class MilvusConfig:
    milvus_url: str          # Milvus服务端连接地址（Zilliz Cloud 填集群 URI）
    milvus_token: str        # Zilliz Cloud 的 token / api key；自建 Milvus 留空
    chunks_collection: str   # 存储切片的集合名称
    dense_weight: float      # 混合检索中稠密向量的权重
    sparse_weight: float     # 混合检索中稀疏向量的权重
    search_top_k: int        # 单次召回最终返回条数

# 实例化Milvus配置对象（和其他配置对象命名风格统一）
milvus_config = MilvusConfig(
    milvus_url=os.getenv("MILVUS_URL"),
    milvus_token=os.getenv("MILVUS_TOKEN", ""),
    chunks_collection=os.getenv("CHUNKS_COLLECTION", "kb_chunks"),
    dense_weight=float(os.getenv("MILVUS_DENSE_WEIGHT", "0.8")),
    sparse_weight=float(os.getenv("MILVUS_SPARSE_WEIGHT", "0.2")),
    search_top_k=int(os.getenv("MILVUS_SEARCH_TOP_K", "5")),
)