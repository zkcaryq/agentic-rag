#!/usr/bin/env python
"""本地零凭据演示：把整条链路在你机器上跑起来。

用到的都是真实实现，只是换成本地可跑的形态：
  向量库  Milvus Lite（单文件，无需 Docker）
  向量    lexical（本地词法向量，Hashing TF；不是随机数，但没有语义泛化）
  重排    none（不重排，保留 RRF 顺序）
  存储    本地文件（output/demo/storage）
  历史    SQLite（output/demo/history.db）
  队列    关闭（进程内线程兜底）

用法：
    python scripts/demo_local.py --limit 8            # 灌 8 份文档并跑 3 个问题
    python scripts/demo_local.py --limit 8 --reset    # 先清空集合再灌
    python scripts/demo_local.py --skip-ingest        # 只跑问题
    python scripts/demo_local.py --serve              # 起聊天页面（先跑过 ingest）

换成真实云端只需要改 .env：EMBEDDING_PROVIDER=dashscope + 百炼 key
+ MILVUS_URL 指向 Zilliz，代码不用动。
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

DEMO_DIR = ROOT / "output" / "demo"

QUESTIONS = [
    "Pantum P3030 打印机怎么更换硒鼓",
    "H3C ER2100 路由器怎么恢复出厂设置",
    "华为 MateBook B3-410 怎么进入 BIOS 设置",
]


def configure_env() -> dict:
    """在导入任何 app 模块之前锁定演示配置。"""
    from app.conf.demo_config import configure_demo_env

    return configure_demo_env()


def print_profile(profile: dict) -> None:
    print("演示配置：")
    for key in (
        "EMBEDDING_PROVIDER",
        "MILVUS_URL",
        "STORAGE_BACKEND",
        "HISTORY_BACKEND",
        "RERANK_PROVIDER",
    ):
        value = os.environ.get(key, "")
        if key == "MILVUS_URL":
            value = Path(value).name
        print(f"  {key:<20} {value}")
    print()


def reset_collection() -> None:
    from app.clients.milvus_utils import get_milvus_client
    from app.conf.milvus_config import milvus_config

    client = get_milvus_client()
    if client is None:
        raise SystemExit("Milvus 连接失败")
    if client.has_collection(milvus_config.chunks_collection):
        client.drop_collection(milvus_config.chunks_collection)
        print(f"已删除旧集合 {milvus_config.chunks_collection}")


def collect_documents(limit: int) -> list:
    from reembed_existing import collect_chunks_files, is_placeholder

    documents = []
    for path in collect_chunks_files(ROOT / "output"):
        if DEMO_DIR in path.parents:
            continue
        try:
            chunks = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        if not isinstance(chunks, list) or not chunks:
            continue
        if is_placeholder(chunks):
            continue
        documents.append((path, chunks))
        if len(documents) >= limit:
            break
    return documents


def ingest(limit: int) -> int:
    from reembed_existing import build_state

    from app.import_process.agent.nodes.node_bge_embedding import node_bge_embedding
    from app.import_process.agent.nodes.node_import_milvus import node_import_milvus

    documents = collect_documents(limit)
    if not documents:
        raise SystemExit(
            "output/ 下没有可用的 chunks.json。请先把文档放到 doc/ 并通过导入服务解析，"
            "或直接跑 scripts/reembed_existing.py 复用已有产物。"
        )

    total_chunks = sum(len(chunks) for _, chunks in documents)
    print(f"准备入库 {len(documents)} 份文档 / {total_chunks} 个切片\n")

    inserted = 0
    for index, (path, chunks) in enumerate(documents, start=1):
        state = build_state(path, chunks)
        node_bge_embedding(state)
        node_import_milvus(state)
        inserted += len(state["chunks"])
        print(f"  [{index}/{len(documents)}] {state['file_title']} — {len(state['chunks'])} 切片")

    print(f"\n入库完成，共 {inserted} 个切片\n")
    return inserted


def run_queries(top_k: int) -> None:
    from app.eval.pipelines import retrieve

    print("=" * 70)
    for question in QUESTIONS:
        docs = retrieve(question, "hybrid", top_k=top_k)
        print(f"\n问题：{question}")
        if not docs:
            print("  （未召回任何文档）")
            continue
        for rank, doc in enumerate(docs, start=1):
            score = doc.get("score", 0.0) or 0.0
            print(
                f"  [{rank}] {doc.get('file_title', '')}"
                f"  | 第 {doc.get('page_no') or '-'} 页"
                f"  | biz_chunk_id={doc.get('biz_chunk_id') or '-'}"
                f"  | score={score:.4f}"
            )
    print("\n" + "=" * 70)


SERVICES = {
    "query": (
        "app.query_process.api.query_service:app",
        8001,
        "聊天页面",
        "http://127.0.0.1:8001/chat.html",
    ),
    "import": (
        "app.import_process.api.file_import_service:app",
        8002,
        "上传页面",
        "http://127.0.0.1:8002/import.html",
    ),
}


def serve(target: str = "query") -> None:
    from app.clients.milvus_utils import close_milvus_client

    # Milvus Lite 是单进程文件锁，起独立服务前必须先释放本进程的句柄。
    # 同理：query 与 import 两个服务也不能同时起，它们抢同一个 db 文件。
    close_milvus_client()

    module, port, label, url = SERVICES[target]
    process = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "uvicorn",
            module,
            "--host",
            "127.0.0.1",
            "--port",
            str(port),
        ],
        cwd=str(ROOT),
        env=os.environ.copy(),
    )

    print()
    print(f"{label}已启动： {url}")
    if target == "query":
        print("（问答走抽取式降级；配上大模型 key 后会变成生成式回答）")
    else:
        print("（.md / .docx / .xlsx 不需要任何凭据即可入库；.pdf 需要 MinerU token）")
    print("按 Ctrl+C 停止\n")

    try:
        while process.poll() is None:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\n正在停止服务…")
        process.terminate()
        try:
            process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            process.kill()


def main() -> int:
    parser = argparse.ArgumentParser(description="本地零凭据演示")
    parser.add_argument("--limit", type=int, default=8, help="入库文档数，默认 8")
    parser.add_argument("--reset", action="store_true", help="入库前清空集合")
    parser.add_argument("--skip-ingest", action="store_true", help="跳过入库，只跑查询")
    parser.add_argument(
        "--serve",
        nargs="?",
        const="query",
        choices=("query", "import"),
        help="启动服务：--serve 起问答页，--serve import 起上传页",
    )
    parser.add_argument("--top-k", type=int, default=3)
    args = parser.parse_args()

    print_profile(configure_env())

    if args.serve:
        # 关键：本进程绝不能打开 Milvus Lite —— 它是单进程文件锁，
        # 父进程一旦持有句柄，子进程（uvicorn）就起不来。
        print("启动服务（本进程不打开 Milvus，避免文件锁冲突）")
        print("两个服务不能同时运行；如需重新入库请先停止服务再跑 --limit N --reset")
        serve(args.serve)
        return 0

    if args.reset:
        reset_collection()

    if not args.skip_ingest:
        ingest(args.limit)

    run_queries(args.top_k)

    print("想看网页效果：先结束本进程，再执行 python scripts/demo_local.py --serve")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
