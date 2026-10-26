#!/usr/bin/env python
"""Milvus / Zilliz 集合维护。

用法：
    .venv/Scripts/python.exe scripts/milvus_admin.py --stats
    .venv/Scripts/python.exe scripts/milvus_admin.py --recreate    # 删集合重建（会丢向量）

注意：--recreate 会删除整个 chunk 集合，只在确定要全量重灌时使用。
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

from app.clients.milvus_utils import get_milvus_client  # noqa: E402
from app.conf.milvus_config import milvus_config  # noqa: E402


def show_stats(client, collection: str) -> None:
    exists = collection in client.list_collections()
    print(f"集合: {collection}")
    print(f"存在: {exists}")
    if not exists:
        return

    desc = client.describe_collection(collection_name=collection)
    fields = [f"{item.get('name')}:{item.get('type')}" for item in desc.get("fields", [])]
    print(f"字段: {', '.join(fields)}")

    try:
        stats = client.get_collection_stats(collection_name=collection)
        print(f"统计: {stats}")
    except Exception as e:
        print(f"统计读取失败: {e}")

    try:
        rows = client.query(
            # Zilliz 上 filter="" 的 count 会返回 0（空过滤不被支持），
            # 必须用恒真的主键条件才拿得到真实行数
            collection_name=collection,
            filter="chunk_id > 0",
            output_fields=["count(*)"],
        )
        print(f"实体数: {rows}")
    except Exception as e:
        print(f"count(*) 查询失败: {e}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Milvus 集合维护")
    parser.add_argument("--stats", action="store_true", help="打印集合统计")
    parser.add_argument("--recreate", action="store_true", help="删除并重建 chunk 集合")
    args = parser.parse_args()

    client = get_milvus_client()
    if client is None:
        print("Milvus 客户端不可用，请检查 MILVUS_URL / MILVUS_TOKEN")
        return 1

    collection = milvus_config.chunks_collection

    if args.recreate:
        if collection in client.list_collections():
            client.drop_collection(collection_name=collection)
            print(f"已删除集合 {collection}")
        else:
            print(f"集合 {collection} 不存在，跳过删除")
        from app.import_process.agent.nodes.node_import_milvus import _create_collection

        _create_collection(client, collection)
        print("重建完成")
        return 0

    if args.stats or not (args.recreate):
        show_stats(client, collection)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
