#!/usr/bin/env python
"""复用 output/ 里已有的切分结果重新向量化并灌库，不重跑 MinerU。

原来的 output/ 里保存了 80+ 份文档的 chunks.json，但当时写进去的是随机向量。
重跑 MinerU 既慢又要花 token，所以这里直接读 chunks.json，只重算向量。

用法：
    .venv/Scripts/python.exe scripts/reembed_existing.py --dry-run
    .venv/Scripts/python.exe scripts/reembed_existing.py --limit 3
    .venv/Scripts/python.exe scripts/reembed_existing.py            # 全量
    .venv/Scripts/python.exe scripts/reembed_existing.py --include-placeholder
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import uuid
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

from app.import_process.agent.nodes.node_bge_embedding import node_bge_embedding  # noqa: E402
from app.import_process.agent.nodes.node_document_split import attach_page_numbers  # noqa: E402
from app.import_process.agent.nodes.node_import_milvus import node_import_milvus  # noqa: E402
from app.import_process.agent.state import create_default_state  # noqa: E402

PLACEHOLDER_MARKER = "这是从PDF转换而来的Markdown内容"
VECTOR_FIELDS = ("dense_vector", "sparse_vector")


def collect_chunks_files(output_dir: Path) -> list[Path]:
    """收集所有 chunks.json，同一文档只保留体积最大的那一份。"""
    best: dict[str, Path] = {}
    for path in output_dir.rglob("chunks.json"):
        try:
            size = path.stat().st_size
        except OSError:
            continue
        key = path.parent.name
        if key not in best or size > best[key].stat().st_size:
            best[key] = path
    return sorted(best.values())


def is_placeholder(chunks: list[dict]) -> bool:
    if not chunks:
        return True
    if len(chunks) <= 1 and len(chunks[0].get("content", "")) < 200:
        return True
    return any(PLACEHOLDER_MARKER in chunk.get("content", "") for chunk in chunks)


def build_state(chunks_path: Path, chunks: list[dict]) -> dict:
    local_dir = chunks_path.parent
    first = chunks[0]
    file_title = first.get("file_title") or local_dir.name
    product_name = first.get("product_name") or file_title

    cleaned = []
    for index, chunk in enumerate(chunks):
        item = {k: v for k, v in chunk.items() if k not in VECTOR_FIELDS}
        item.setdefault("chunk_id", f"{file_title}_{index}")
        item.setdefault("file_title", file_title)
        item.setdefault("product_name", product_name)
        cleaned.append(item)

    attach_page_numbers(cleaned, str(local_dir))

    state = create_default_state(
        task_id=f"reembed_{uuid.uuid4().hex[:8]}",
        local_dir=str(local_dir),
        local_file_path=str(local_dir),
    )
    state["file_title"] = file_title
    state["product_name"] = product_name
    state["chunks"] = cleaned
    return state


def main() -> int:
    parser = argparse.ArgumentParser(description="复用已有 chunks 重灌向量")
    parser.add_argument("--output-dir", default=str(ROOT / "output"))
    parser.add_argument("--dry-run", action="store_true", help="只统计，不调 API")
    parser.add_argument("--limit", type=int, default=0, help="只处理前 N 份文档")
    parser.add_argument(
        "--include-placeholder",
        action="store_true",
        help="连占位/空文档一起处理（默认跳过）",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    if not output_dir.exists():
        print(f"目录不存在: {output_dir}")
        return 1

    planned = []
    skipped_placeholder = 0
    total_chunks = 0

    for path in collect_chunks_files(output_dir):
        try:
            chunks = json.loads(path.read_text(encoding="utf-8"))
        except Exception as e:
            print(f"[跳过] 读取失败 {path}: {e}")
            continue
        if not isinstance(chunks, list) or not chunks:
            continue
        if is_placeholder(chunks) and not args.include_placeholder:
            skipped_placeholder += 1
            continue
        planned.append((path, chunks))
        total_chunks += len(chunks)

    if args.limit:
        planned = planned[: args.limit]
        total_chunks = sum(len(chunks) for _, chunks in planned)

    print(f"待处理文档: {len(planned)}，切片总数: {total_chunks}")
    print(f"跳过占位/空文档: {skipped_placeholder}")

    if args.dry_run:
        for path, chunks in planned[:10]:
            print(f"  - {path.parent.name}: {len(chunks)} chunks")
        if len(planned) > 10:
            print(f"  ... 另外 {len(planned) - 10} 份")
        return 0

    succeeded = 0
    failed = 0
    started = time.perf_counter()

    for index, (path, chunks) in enumerate(planned, start=1):
        state = build_state(path, chunks)
        try:
            node_bge_embedding(state)
            node_import_milvus(state)
            succeeded += 1
            print(
                f"[{index}/{len(planned)}] OK {state['file_title']} "
                f"({len(state['chunks'])} chunks)"
            )
        except Exception as e:  # noqa: BLE001
            failed += 1
            print(f"[{index}/{len(planned)}] FAIL {path.parent.name}: {e}")

    elapsed = time.perf_counter() - started
    print(f"\n完成：成功 {succeeded}，失败 {failed}，耗时 {elapsed:.1f}s")
    if failed:
        print("失败文档请查看日志，确认 EMBEDDING_PROVIDER / Milvus 配置")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
