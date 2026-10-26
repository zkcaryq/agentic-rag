import os
from pymilvus import MilvusClient, AnnSearchRequest, WeightedRanker
from app.conf.milvus_config import milvus_config
from app.core.logger import logger

# 全局Milvus客户端实例，实现单例复用
_milvus_client = None

# 检索/回捞必须带回的核心切片字段。
# 混合检索的默认字段曾经只有 product_name，于是"切片级溯源"只在有稀疏向量时失效
# （纯稠密路径字段是齐的），三个默认值分开写就是这么埋的雷，统一引用这一个常量。
CORE_OUTPUT_FIELDS = [
    "biz_chunk_id",
    "content",
    "title",
    "parent_title",
    "file_title",
    "product_name",
    "page_no",
]


# 已确认加载过的集合（Milvus Lite / Milvus 都需要 load 才能 query/search）
_loaded_collections: set[str] = set()


def ensure_collection_loaded(client, collection_name: str) -> None:
    """保证集合处于 loaded 状态。

    集合在进程重启后通常是 released，此时 query/search 会直接报
    "Collection is in state 'released'"。这里在拿到客户端时主动 load 一次，
    避免只有检索路径做了兜底、而 query 路径静默失败。
    """
    if client is None or not collection_name:
        return
    if collection_name in _loaded_collections:
        return
    try:
        # 集合还不存在时先退掉：直接 load 会让 pymilvus 自己打一整段
        # "collection not found" 的 traceback，首次建库时看着像故障
        if collection_name not in client.list_collections():
            return
        client.load_collection(collection_name=collection_name)
        _loaded_collections.add(collection_name)
        logger.info(f"集合 {collection_name} 已加载")
    except Exception as e:
        logger.debug(f"load_collection 调用失败（可能已加载）: {e}")


def get_milvus_client():
    """
    Milvus客户端单例获取方法
    实现客户端连接复用，避免重复创建连接消耗资源
    :return: MilvusClient实例，连接失败返回None
    """
    try:
        global _milvus_client
        # 单例判断：未初始化则创建新连接
        if _milvus_client is None:
            milvus_uri = milvus_config.milvus_url
            # 校验Milvus连接地址配置
            if not milvus_uri:
                logger.error("Milvus客户端连接失败：缺少MILVUS_URL环境变量配置")
                return None
            # 初始化Milvus客户端（Zilliz Cloud 需要额外传 token）
            if milvus_config.milvus_token:
                _milvus_client = MilvusClient(
                    uri=milvus_uri, token=milvus_config.milvus_token
                )
            else:
                _milvus_client = MilvusClient(uri=milvus_uri)
            logger.info("Milvus客户端连接成功")
            ensure_collection_loaded(_milvus_client, milvus_config.chunks_collection)
        return _milvus_client
    except Exception as e:
        logger.error(f"Milvus客户端连接异常：{str(e)}", exc_info=True)
        return None


def close_milvus_client():
    """关闭并丢弃单例。

    Milvus Lite 是单进程文件锁，同一个 db 文件不能被两个进程同时打开，
    所以本地演示里启动独立服务前必须先释放。
    """
    global _milvus_client
    if _milvus_client is not None:
        try:
            _milvus_client.close()
        except Exception:
            pass
        _milvus_client = None


def create_hybrid_search_requests(dense_vector, sparse_vector, dense_params=None, sparse_params=None, expr=None,
                                  limit=5):
    """
    构建Milvus混合搜索请求对象
    分别创建稠密/稀疏向量的搜索请求，用于后续混合搜索融合
    :param dense_vector: 文本生成的稠密向量
    :param sparse_vector: 文本生成的稀疏向量
    :param dense_params: 稠密向量搜索参数，默认使用余弦相似度
    :param sparse_params: 稀疏向量搜索参数，默认使用内积相似度
    :param expr: 搜索过滤表达式，用于精准筛选数据
    :param limit: 单向量搜索返回结果数量，默认5
    :return: 搜索请求列表，包含[dense_req, sparse_req]
    """
    # 稠密向量默认搜索参数：余弦相似度（COSINE），适配BGE-M3稠密向量并与建库参数保持一致
    if dense_params is None:
        dense_params = {"metric_type": "COSINE"}
    # 稀疏向量默认搜索参数：内积（IP），适配BGE-M3稀疏向量
    if sparse_params is None:
        sparse_params = {"metric_type": "IP"}

    # 构建稠密向量搜索请求，关联Milvus的dense_vector字段 近似最近邻（ANN）检索请求的核心类
    dense_req = AnnSearchRequest(
        data=[dense_vector],
        anns_field="dense_vector",
        param=dense_params,
        expr=expr,
        limit=limit
    )

    # 构建稀疏向量搜索请求，关联Milvus的sparse_vector字段
    sparse_req = AnnSearchRequest(
        data=[sparse_vector],
        anns_field="sparse_vector",
        param=sparse_params,
        expr=expr,
        limit=limit
    )

    return [dense_req, sparse_req]


def _is_released_error(exc: Exception) -> bool:
    return "released" in str(exc).lower()


def hybrid_search(client, collection_name, reqs, ranker_weights=(0.5, 0.5), norm_score=False, limit=5,
                  output_fields=None, search_params=None):
    """稠密 + 稀疏双路检索，用 WeightedRanker 加权融合。搜索失败返回 None。"""
    ensure_collection_loaded(client, collection_name)
    try:
        # 初始化加权排名器：按权重融合稠密/稀疏向量的搜索结果
        # norm_score=True：先将两个向量评分归一化到0~1区间，再加权计算
        rerank = WeightedRanker(ranker_weights[0], ranker_weights[1], norm_score=norm_score)

        # 默认返回字段：与纯稠密一致，保证溯源字段不丢
        if output_fields is None:
            output_fields = list(CORE_OUTPUT_FIELDS)

        # 执行混合搜索：融合稠密+稀疏向量结果，按权重重新排序
        res = client.hybrid_search(
            collection_name=collection_name,
            reqs=reqs,
            ranker=rerank,
            limit=limit,
            output_fields=output_fields,
            search_params=search_params
        )

        logger.info(f"Milvus混合搜索完成，集合[{collection_name}]共检索到{len(res[0])}条结果")
        return res
    except Exception as e:
        if _is_released_error(e):
            # 集合被释放：load 后重试一次
            ensure_collection_loaded(client, collection_name)
            try:
                res = client.hybrid_search(
                    collection_name=collection_name,
                    reqs=reqs,
                    ranker=WeightedRanker(ranker_weights[0], ranker_weights[1], norm_score=norm_score),
                    limit=limit,
                    output_fields=output_fields,
                    search_params=search_params,
                )
                return res
            except Exception as retry_error:
                logger.error(f"Milvus混合搜索重试仍失败：{retry_error}", exc_info=True)
                return None
        logger.error(f"Milvus混合搜索执行失败，集合[{collection_name}]：{str(e)}", exc_info=True)
        return None


def dense_search(client, collection_name, dense_vector, limit=5, output_fields=None,
                 expr=None, search_params=None):
    """纯稠密检索。

    当 embedding 供应商不支持稀疏向量时退化为这一条路，保证换供应商不会直接炸。
    """
    ensure_collection_loaded(client, collection_name)
    try:
        if output_fields is None:
            output_fields = list(CORE_OUTPUT_FIELDS)
        res = client.search(
            collection_name=collection_name,
            data=[dense_vector],
            anns_field="dense_vector",
            search_params=search_params or {"metric_type": "COSINE"},
            limit=limit,
            filter=expr or "",
            output_fields=output_fields,
        )
        count = len(res[0]) if res else 0
        logger.info(f"Milvus纯稠密搜索完成，集合[{collection_name}]共检索到{count}条结果")
        return res
    except Exception as e:
        if _is_released_error(e):
            ensure_collection_loaded(client, collection_name)
            try:
                return client.search(
                    collection_name=collection_name,
                    data=[dense_vector],
                    anns_field="dense_vector",
                    search_params=search_params or {"metric_type": "COSINE"},
                    limit=limit,
                    filter=expr or "",
                    output_fields=output_fields,
                )
            except Exception as retry_error:
                logger.error(f"Milvus纯稠密搜索重试仍失败：{retry_error}", exc_info=True)
                return None
        logger.error(f"Milvus纯稠密搜索失败，集合[{collection_name}]：{str(e)}", exc_info=True)
        return None


def search_chunks_with_fallback(client, collection_name, dense_vector, sparse_vector,
                                expr, limit=5, ranker_weights=(0.8, 0.2), label="检索"):
    """按产品过滤检索；过滤后为空则退回全库再检索一次。

    产品名口径不一致是常态（用户问「奔图 P3030 打印机」，库里存的可能是
    "Pantum P3030 User Guide zh_CN V1_2"），硬过滤一旦不匹配就会让私有知识库
    整体失忆——实测出现过最终上下文 8 条全是联网结果的故障形态。宁可放宽范围，
    也不要返回空。
    """
    chunks = search_with_optional_sparse(
        client=client,
        collection_name=collection_name,
        dense_vector=dense_vector,
        sparse_vector=sparse_vector,
        expr=expr,
        limit=limit,
        ranker_weights=ranker_weights,
    )
    chunks = (chunks[0] if chunks else []) or []
    if expr and not chunks:
        logger.warning(f"{label}按产品过滤后无结果，退回不带过滤重检: {expr}")
        chunks = search_with_optional_sparse(
            client=client,
            collection_name=collection_name,
            dense_vector=dense_vector,
            sparse_vector=sparse_vector,
            expr=None,
            limit=limit,
            ranker_weights=ranker_weights,
        )
        chunks = chunks[0] if chunks else []
    return chunks


def search_with_optional_sparse(client, collection_name, dense_vector, sparse_vector,
                                expr=None, limit=5, output_fields=None,
                                ranker_weights=(0.8, 0.2), norm_score=True):
    """有稀疏向量就走混合检索，没有就退化成纯稠密检索。

    供应商不支持稀疏时不会报错，只是召回质量下降，日志里会有明示。
    """
    if sparse_vector:
        reqs = create_hybrid_search_requests(
            dense_vector=dense_vector,
            sparse_vector=sparse_vector,
            expr=expr,
            limit=limit * 2,
        )
        return hybrid_search(
            client=client,
            collection_name=collection_name,
            reqs=reqs,
            ranker_weights=ranker_weights,
            norm_score=norm_score,
            limit=limit,
            output_fields=output_fields,
        )
    logger.warning("无稀疏向量，本次检索降级为纯稠密检索")
    return dense_search(
        client=client,
        collection_name=collection_name,
        dense_vector=dense_vector,
        limit=limit,
        output_fields=output_fields,
        expr=expr,
    )