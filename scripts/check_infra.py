#!/usr/bin/env python
"""基础设施连通性自检。

用法（在仓库根目录）：
    .venv/Scripts/python.exe scripts/check_infra.py
    .venv/Scripts/python.exe scripts/check_infra.py --only milvus,pg,redis
    .venv/Scripts/python.exe scripts/check_infra.py --list

退出码：0 = 全部通过或跳过；1 = 至少一项 FAIL。
输出里不会出现任何凭据、token 或带口令的连接串。
"""
from __future__ import annotations

import argparse
import sys
import time
import uuid
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Windows 控制台默认 GBK，强制 UTF-8 以免中文输出乱码
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass


class SkipCheck(Exception):
    """缺少配置或依赖，跳过而不是判失败。"""


@dataclass
class Result:
    name: str
    status: str
    detail: str
    elapsed_ms: int = 0


def check_milvus() -> str:
    from app.conf.milvus_config import milvus_config

    if not milvus_config.milvus_url:
        raise SkipCheck("未配置 MILVUS_URL")
    try:
        from pymilvus import MilvusClient
    except ImportError:
        raise SkipCheck("未安装 pymilvus")

    kwargs = {"uri": milvus_config.milvus_url}
    if milvus_config.milvus_token:
        kwargs["token"] = milvus_config.milvus_token
    client = MilvusClient(**kwargs)
    collections = client.list_collections()

    version = ""
    try:
        version = str(client.get_server_version())
    except Exception:
        pass
    suffix = f" version={version}" if version else ""
    return f"connected collections={len(collections)}{suffix}"


def check_oss() -> str:
    from app.conf.oss_config import oss_config

    if not oss_config.enabled:
        raise SkipCheck("OSS_ENABLED=false")
    if not (oss_config.endpoint and oss_config.bucket and oss_config.access_key_id):
        raise SkipCheck("OSS 配置不完整")

    try:
        import oss2
    except ImportError:
        raise SkipCheck("未安装 oss2（pip install oss2）")
    try:
        import requests
    except ImportError:
        raise SkipCheck("未安装 requests")

    auth = oss2.Auth(oss_config.access_key_id, oss_config.access_key_secret)
    bucket = oss2.Bucket(auth, oss_config.endpoint, oss_config.bucket)
    # oss2 的 Bucket 没有 exists()，要用 get_bucket_info()——这一步同时验凭证与桶权限
    info = bucket.get_bucket_info()
    if info.location.replace("oss-", "") not in oss_config.endpoint:
        raise RuntimeError(
            f"桶实际在 {info.location}，与 OSS_ENDPOINT={oss_config.endpoint} 不匹配"
        )

    # 写 -> 签名 URL 读 -> 删，同时验证私有 bucket 的签名链路
    key = f"{oss_config.upload_prefix}/__healthcheck__/{uuid.uuid4().hex}.txt"
    bucket.put_object(key, b"healthcheck")
    try:
        signed = bucket.sign_url("GET", key, 60)
        resp = requests.get(signed, timeout=10)
        if resp.status_code != 200:
            raise RuntimeError(f"签名 URL 读取失败 HTTP {resp.status_code}")
    finally:
        bucket.delete_object(key)

    return f"bucket={oss_config.bucket} location={info.location} 写读删+签名URL 正常"


def check_pg() -> str:
    from app.conf.infra_config import infra_config_pg as cfg

    if not cfg.enabled:
        raise SkipCheck("PG_ENABLED=false")
    if not cfg.dsn:
        raise SkipCheck("未配置 PG_DSN")
    try:
        import psycopg
    except ImportError:
        raise SkipCheck("未安装 psycopg（pip install psycopg[binary]）")

    # 与应用侧保持一致：锁 UTC。psycopg 不认 server_settings，只能走 options。
    with psycopg.connect(cfg.dsn, connect_timeout=5, options="-c TimeZone=UTC") as conn:
        with conn.cursor() as cur:
            cur.execute("show timezone")
            tz = cur.fetchone()[0]
            cur.execute("select current_database(), version()")
            db, ver = cur.fetchone()

    if tz != "UTC":
        raise RuntimeError(f"会话时区不是 UTC（实际 {tz}），naive TIMESTAMP 会偏 8 小时")
    return f"db={db} tz={tz} {ver.split(',')[0]}"


def check_redis() -> str:
    from app.conf.infra_config import REDIS_KEY_PREFIX
    from app.conf.infra_config import infra_config_redis as cfg

    if not cfg.enabled:
        raise SkipCheck("REDIS_ENABLED=false")
    if not cfg.url:
        raise SkipCheck("未配置 REDIS_URL")
    try:
        import redis
    except ImportError:
        raise SkipCheck("未安装 redis（pip install redis）")

    client = redis.from_url(cfg.url, socket_connect_timeout=5, socket_timeout=5)
    client.ping()

    key = f"{REDIS_KEY_PREFIX}__healthcheck__"
    client.set(key, "1", ex=30)
    value = client.get(key)
    client.delete(key)
    if value not in (b"1", "1"):
        raise RuntimeError("健康检查 key 读写不一致")

    # 反向验证 ACL 隔离是否真的生效（过期过宽的 ACL 只提示，不判失败）
    probe = f"__no_permission_probe__{uuid.uuid4().hex[:6]}"
    try:
        client.set(probe, "1", ex=10)
        client.delete(probe)
        acl_note = "ACL 未拦截非 rag: 前缀（可能过宽）"
    except Exception:
        acl_note = "ACL 已拦截非 rag: 前缀"

    # 跨进程 SSE 事件总线（MQ worker → API 进程）用的就是这对原语，
    # 直接调生产函数验，别在检查脚本里抄一份 RPUSH/BLPOP。
    from app.clients import redis_utils

    if not redis_utils.publish_event("__healthcheck__", "ping", {"ok": True}):
        raise RuntimeError("SSE 事件写入失败，MQ 派发查询会丢事件")
    event = redis_utils.pop_event("__healthcheck__", timeout=3)
    redis_utils.clear_events("__healthcheck__")
    if not event or event.get("event") != "ping":
        raise RuntimeError("SSE 事件写进去读不回来（BLPOP 不通），跨进程查询链路不通")

    return f"ping+读写删 OK；RPUSH/BLPOP 往返 OK；{acl_note}"


def check_rabbitmq() -> str:
    from app.conf.infra_config import MQ_PREFIX
    from app.conf.infra_config import infra_config_mq as cfg

    if not cfg.enabled:
        raise SkipCheck("RABBITMQ_ENABLED=false")
    if not cfg.url:
        raise SkipCheck("未配置 RABBITMQ_URL")
    try:
        import pika
    except ImportError:
        raise SkipCheck("未安装 pika（pip install pika）")

    params = pika.URLParameters(cfg.url)
    params.connection_attempts = 1
    params.socket_timeout = 5
    connection = pika.BlockingConnection(params)
    suffix = uuid.uuid4().hex[:8]
    exchange = f"{MQ_PREFIX}healthcheck.{suffix}"
    queue = f"{MQ_PREFIX}healthcheck.q.{suffix}"
    payload = b'{"healthcheck":true}'

    try:
        channel = connection.channel()
        channel.exchange_declare(exchange=exchange, exchange_type="direct", auto_delete=True)
        channel.queue_declare(queue=queue, exclusive=True, auto_delete=True)
        channel.queue_bind(queue=queue, exchange=exchange, routing_key=exchange)
        channel.basic_publish(exchange=exchange, routing_key=exchange, body=payload)
        method, _props, body = channel.basic_get(queue=queue, auto_ack=True)
        if method is None or body != payload:
            raise RuntimeError("消息发布后未取回，队列链路不通")
        channel.queue_delete(queue=queue)
        channel.exchange_delete(exchange=exchange)
    finally:
        connection.close()

    return f"vhost 连接 + publish/consume 往返 OK（{MQ_PREFIX}* 权限正常）"


def check_embedding() -> str:
    from app.conf.provider_config import embedding_provider_config as cfg
    from app.lm.provider_factory import get_embedding_provider

    if not cfg.api_key:
        raise SkipCheck(f"未配置 {cfg.provider} 的 API key")

    provider = get_embedding_provider()
    texts = ["打印机卡纸怎么处理", "路由器无法连接网络"]
    records = provider.embed(texts, text_type="query")

    if len(records) != len(texts):
        raise RuntimeError(f"返回条数不符：期望 {len(texts)}，实际 {len(records)}")
    dim = len(records[0].dense)
    if dim != provider.dimension:
        raise RuntimeError(f"稠密维度不符：期望 {provider.dimension}，实际 {dim}")
    # 关键校验：防止再次出现「随机向量 / 常量向量」的假实现
    if records[0].dense == records[1].dense:
        raise RuntimeError("两条不同文本返回了相同稠密向量，向量化实现有问题")

    sparse_note = "no"
    if provider.supports_sparse:
        if not records[0].sparse:
            raise RuntimeError("声明支持稀疏向量但返回为空")
        if records[0].sparse == records[1].sparse:
            raise RuntimeError("两条不同文本返回了相同稀疏向量，向量化实现有问题")
        sparse_note = f"yes(nnz={len(records[0].sparse)})"

    return f"{provider.name}/{cfg.model} dim={dim} sparse={sparse_note}"


def check_rerank() -> str:
    from app.conf.provider_config import rerank_provider_config as cfg
    from app.lm.provider_factory import get_rerank_provider

    if not cfg.api_key:
        raise SkipCheck(f"未配置 {cfg.provider} 的 API key")

    provider = get_rerank_provider()
    documents = [
        "更换硒鼓并重置计数器后重新打印测试页",
        "无线网络连接失败时请检查 SSID 与加密方式",
        "显示器亮度可在 OSD 菜单中调节",
    ]
    ranked = provider.rerank("打印机怎么换硒鼓", documents, top_n=3)

    if not ranked:
        raise RuntimeError("没有返回排序结果")
    scores = [score for _, score in ranked]
    if scores != sorted(scores, reverse=True):
        raise RuntimeError(f"分数未按降序排列：{scores}")
    if not all(0.0 <= score <= 1.0 for score in scores):
        raise RuntimeError(f"分数量纲不在 0~1：{scores}")
    if ranked[0][0] != 0:
        raise RuntimeError("相关性最高的不是打印机那条，重排结果可疑")

    return f"{provider.name}/{cfg.model} top1={scores[0]:.3f} 排序正确"


def check_mineru() -> str:
    import os

    token = (os.getenv("MINERU_API_TOKEN") or "").strip()
    base_url = (os.getenv("MINERU_BASE_URL") or "").strip()
    if not token:
        raise SkipCheck("未配置 MINERU_API_TOKEN（MinerU 真跑必须要有）")
    if not base_url:
        raise SkipCheck("未配置 MINERU_BASE_URL")
    return f"token 已配置（{len(token)} 字符）base={base_url}"


def check_langsmith() -> str:
    import os

    from app.conf.infra_config import LangSmithConfig, infra_config_langsmith as cfg

    if not cfg.enabled:
        raise SkipCheck(
            f"未开启 tracing（{'/'.join(LangSmithConfig.TRACING_ALIASES)} 全部为 false）"
        )
    key = next(
        (
            os.getenv(name).strip()
            for name in LangSmithConfig.KEY_ALIASES
            if (os.getenv(name) or "").strip()
        ),
        "",
    )
    if not key:
        raise SkipCheck(
            f"已开启 tracing 但未配置密钥（{'/'.join(LangSmithConfig.KEY_ALIASES)}）"
        )

    # 不只看变量存不存在，真去 API 取一次数据：密钥错/无权限在这里就会暴露
    try:
        from langsmith import Client
    except ImportError:
        raise SkipCheck("未安装 langsmith（uv add langsmith）")
    client = Client(api_key=key)
    list(client.list_runs(limit=1))
    return f"project={cfg.project}，API 鉴权与读取正常"


CHECKS = {
    "milvus": check_milvus,
    "oss": check_oss,
    "pg": check_pg,
    "redis": check_redis,
    "rabbitmq": check_rabbitmq,
    "embedding": check_embedding,
    "rerank": check_rerank,
    "mineru": check_mineru,
    "langsmith": check_langsmith,
}


def run_one(name: str, fn) -> Result:
    started = time.perf_counter()
    try:
        detail = fn()
        status = "OK"
    except SkipCheck as exc:
        detail, status = str(exc), "SKIP"
    except Exception as exc:  # noqa: BLE001 - 自检脚本要把任何异常都转成报告
        detail, status = f"{exc.__class__.__name__}: {exc}", "FAIL"
    elapsed = int((time.perf_counter() - started) * 1000)
    return Result(name=name, status=status, detail=detail, elapsed_ms=elapsed)


def main() -> int:
    parser = argparse.ArgumentParser(description="基础设施连通性自检")
    parser.add_argument("--only", help="只跑指定项，逗号分隔，如 milvus,pg,redis")
    parser.add_argument("--list", action="store_true", help="列出所有检查项")
    args = parser.parse_args()

    if args.list:
        for name in CHECKS:
            print(name)
        return 0

    selected = list(CHECKS)
    if args.only:
        wanted = [item.strip() for item in args.only.split(",") if item.strip()]
        unknown = [item for item in wanted if item not in CHECKS]
        if unknown:
            print(f"未知检查项：{', '.join(unknown)}")
            return 1
        selected = wanted

    results = [run_one(name, CHECKS[name]) for name in selected]

    print()
    width = max(len(item.name) for item in results)
    for item in results:
        print(f"[{item.status:<4}] {item.name:<{width}} {item.elapsed_ms:>6}ms  {item.detail}")

    failed = [item for item in results if item.status == "FAIL"]
    skipped = [item for item in results if item.status == "SKIP"]
    print()
    print(f"合计 {len(results)} 项：通过 {len(results) - len(failed) - len(skipped)}，"
          f"跳过 {len(skipped)}，失败 {len(failed)}")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
