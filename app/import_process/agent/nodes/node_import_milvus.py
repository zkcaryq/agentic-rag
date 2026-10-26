"""
Milvus入库节点
将向量化后的切片数据存入 Milvus 数据库
"""
import sys
import json
from pathlib import Path
from typing import List, Dict

from pymilvus import MilvusClient, DataType

from app.core.logger import logger
from app.import_process.agent.state import ImportGraphState
from app.lm.provider_factory import get_embedding_provider
from app.utils.task_utils import add_running_task, add_done_task
from app.utils.escape_milvus_string_utils import escape_milvus_string


def node_import_milvus(state: ImportGraphState) -> ImportGraphState:
    """
    Milvus入库节点

    核心流程：
        1. 获取切片数据
        2. 连接Milvus
        3. 清理旧数据（幂等性）
        4. 批量插入新数据
    """
    func_name = sys._getframe().f_code.co_name
    logger.info(f">>> 开始执行节点: {func_name}")

    add_running_task(state.get("task_id", ""), func_name)

    try:
        chunks = state.get("chunks", [])
        if not chunks:
            logger.warning("无切片数据，跳过入库")
            add_done_task(state.get("task_id", ""), func_name)
            return state

        product_name = state.get("product_name", "unknown")
        logger.info(f"开始入库，产品: {product_name}, 切片数: {len(chunks)}")

        # 延迟导入，避免没有 Milvus 的环境在 import 阶段就失败
        from app.clients.milvus_utils import get_milvus_client
        from app.conf.milvus_config import milvus_config

        # 单例客户端，Zilliz Cloud 的 token 在 milvus_utils 里统一处理
        client = get_milvus_client()
        if client is None:
            raise RuntimeError("Milvus 客户端不可用，请检查 MILVUS_URL / MILVUS_TOKEN")

        collection_name = milvus_config.chunks_collection
        if not _collection_exists(client, collection_name):
            _create_collection(client, collection_name)
        else:
            _ensure_schema_compatible(client, collection_name)

        _clean_old_data(client, collection_name, product_name)

        inserted_count = _insert_chunks(client, collection_name, chunks)
        logger.info(f"成功入库 {inserted_count} 条数据")
        if inserted_count != len(chunks):
            raise RuntimeError(
                f"入库条数不符：切片 {len(chunks)}，实际写入 {inserted_count}"
            )

        _save_import_result(state, chunks)

    except Exception as e:
        state["failed_stage"] = "milvus"
        logger.exception(f"Milvus入库失败: {e}")
        add_done_task(state.get("task_id", ""), func_name)
        raise

    add_done_task(state.get("task_id", ""), func_name)
    logger.info(f">>> 节点执行完成: {func_name}")

    return state


def _collection_exists(client: MilvusClient, collection_name: str) -> bool:
    """检查集合是否存在"""
    try:
        collections = client.list_collections()
        return collection_name in collections
    except Exception as e:
        logger.warning(f"检查集合失败: {e}")
        return False


def _ensure_schema_compatible(client: MilvusClient, collection_name: str) -> None:
    """老集合缺少溯源字段时给出明确处置提示，而不是让插入静默失败。"""
    try:
        desc = client.describe_collection(collection_name=collection_name)
        fields = {item.get("name") for item in desc.get("fields", [])}
    except Exception as e:
        logger.warning(f"读取集合 schema 失败，跳过兼容性检查: {e}")
        return

    missing = {"biz_chunk_id", "page_no"} - fields
    if missing:
        raise RuntimeError(
            f"集合 {collection_name} 的 schema 过旧，缺少字段 {sorted(missing)}。"
            f"执行 `python scripts/milvus_admin.py --recreate` 重建后重新入库。"
        )


def _create_collection(client: MilvusClient, collection_name: str):
    """创建集合（稠密必建，稀疏取决于当前 embedding 供应商是否支持）。"""
    provider = get_embedding_provider()
    dim = provider.dimension
    with_sparse = provider.supports_sparse

    schema = client.create_schema(auto_id=True, enable_dynamic_field=True)

    # 物理主键（Milvus 自增）
    schema.add_field(
        field_name="chunk_id", datatype=DataType.INT64, is_primary=True, auto_id=True
    )
    # 业务切片 ID：切分节点生成，用于「切片 ID 贯穿全链路」与逐段溯源
    schema.add_field(field_name="biz_chunk_id", datatype=DataType.VARCHAR, max_length=512)
    schema.add_field(field_name="content", datatype=DataType.VARCHAR, max_length=65535)
    schema.add_field(field_name="title", datatype=DataType.VARCHAR, max_length=1024)
    schema.add_field(field_name="parent_title", datatype=DataType.VARCHAR, max_length=1024)
    schema.add_field(field_name="file_title", datatype=DataType.VARCHAR, max_length=512)
    schema.add_field(field_name="product_name", datatype=DataType.VARCHAR, max_length=512)
    # 页码溯源：0 表示未知或不适用（例如 Excel 只有 sheet 名）
    schema.add_field(field_name="page_no", datatype=DataType.INT64)
    schema.add_field(field_name="dense_vector", datatype=DataType.FLOAT_VECTOR, dim=dim)
    if with_sparse:
        schema.add_field(
            field_name="sparse_vector", datatype=DataType.SPARSE_FLOAT_VECTOR
        )

    index_params = client.prepare_index_params()
    index_params.add_index(
        field_name="dense_vector",
        index_name="dense_index",
        index_type="HNSW",
        metric_type="COSINE",
        params={"M": 16, "efConstruction": 200},
    )
    if with_sparse:
        index_params.add_index(
            field_name="sparse_vector",
            index_name="sparse_index",
            index_type="SPARSE_INVERTED_INDEX",
            metric_type="IP",
            params={"inverted_index_algo": "DAAT_MAXSCORE"},
        )

    client.create_collection(
        collection_name=collection_name, schema=schema, index_params=index_params
    )
    # 建完立刻 load，避免服务重启后集合处于 released 状态导致检索报错
    try:
        client.load_collection(collection_name=collection_name)
    except Exception as e:
        logger.warning(f"集合 {collection_name} 加载失败（检索时会重试）: {e}")
    logger.info(f"集合 {collection_name} 创建成功（dim={dim}, sparse={with_sparse}）")


def _clean_old_data(client: MilvusClient, collection_name: str, product_name: str):
    """清理旧数据（幂等性）"""
    try:
        # 检查集合是否存在
        if collection_name not in client.list_collections():
            return

        # 删除指定产品的数据。product_name 来自模型对上传内容的识别，属于外部输入，
        # 而这是一条 delete 表达式——不转义就等于让输入决定删谁的数据
        client.delete(
            collection_name=collection_name,
            filter=f'product_name == "{escape_milvus_string(product_name)}"'
        )
        logger.info(f"已清理产品 {product_name} 的旧数据")
    except Exception as e:
        logger.warning(f"清理旧数据失败: {e}")


def _insert_chunks(client: MilvusClient, collection_name: str, chunks: List[Dict]) -> int:
    """插入切片数据。失败直接抛出，避免「入库失败仍算完成」。"""
    with_sparse_fields = _collection_has_sparse(client, collection_name)

    data = []
    for chunk in chunks:
        doc = {
            "biz_chunk_id": str(chunk.get("chunk_id", "")),
            "content": chunk.get("content", ""),
            "title": chunk.get("title", ""),
            "parent_title": chunk.get("parent_title", ""),
            "file_title": chunk.get("file_title", ""),
            "product_name": chunk.get("product_name", ""),
            "page_no": int(chunk.get("page_no") or 0),
            "dense_vector": chunk.get("dense_vector", []),
        }
        if with_sparse_fields:
            sparse = chunk.get("sparse_vector") or {}
            if not sparse:
                raise RuntimeError(
                    f"切片 {doc['biz_chunk_id']} 缺少稀疏向量，但集合 schema 含稀疏字段；"
                    f"请确认 EMBEDDING_PROVIDER 为 dashscope"
                )
            doc["sparse_vector"] = sparse
        data.append(doc)

    result = client.insert(collection_name=collection_name, data=data)
    return result.get("insert_count", 0)


def _collection_has_sparse(client: MilvusClient, collection_name: str) -> bool:
    """判断集合是否声明了稀疏字段（决定插入时是否带 sparse_vector）。"""
    try:
        desc = client.describe_collection(collection_name=collection_name)
        fields = {item.get("name") for item in desc.get("fields", [])}
        return "sparse_vector" in fields
    except Exception:
        return False


def _save_import_result(state: ImportGraphState, chunks: List[Dict]):
    """保存入库结果到本地文件（只留元数据，不写向量，避免文件过大）。"""
    local_dir = state.get("local_dir", "")
    if not local_dir:
        return

    result_path = Path(local_dir) / "import_result.json"
    try:
        result = {
            "task_id": state.get("task_id"),
            "file_title": state.get("file_title"),
            "product_name": state.get("product_name"),
            "chunk_count": len(chunks),
            "chunks": [
                {
                    "chunk_id": c.get("chunk_id"),
                    "title": c.get("title"),
                    "page_no": c.get("page_no"),
                    "content_length": len(c.get("content", "")),
                }
                for c in chunks
            ],
        }
        with open(result_path, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False)
        logger.info(f"入库结果已保存: {result_path}")
    except Exception as e:
        logger.warning(f"保存入库结果失败: {e}")


# ==================== 测试代码 ====================
if __name__ == "__main__":
    print("=" * 50)
    print("node_import_milvus 模块")
    print("=" * 50)