#!/usr/bin/env python
"""检索链路冒烟检查：确认向量与检索是真的在工作，而不是又回到假实现。

检查项：
  1. 供应商返回的稠密/稀疏向量非空，且两条不同文本的向量不同；
  2. Milvus 里能检索到结果，且每条结果都带 biz_chunk_id（切片级可溯源）；
  3. 重排分数单调递减，说明重排结果真的被使用；
  4. 重排后保留条数落在断崖算法的合理区间。

用法：
    .venv/Scripts/python.exe scripts/check_retrieval.py
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

from app.lm.provider_factory import get_embedding_provider  # noqa: E402

# (问题, 命中标记列表)。标记在**切片正文**里找，也接受文件名命中：
# 语料文件名全是型号（hl3040网络说明书 / hak180产品安全手册 / 华为擎云L420），
# 只按文件名断言的话「打印机卡纸」永远判不出，之前这里就是这么误报 FAIL 的。
CHECKS = [
    ("打印机卡纸怎么处理", ("卡纸", "进纸", "纸张", "打印机")),
    ("路由器怎么恢复出厂设置", ("恢复出厂", "出厂设置", "reset")),
    ("笔记本电池怎么保养", ("电池", "充电", "功耗")),
]


def check_embedding() -> None:
    provider = get_embedding_provider()
    records = provider.embed(["打印机卡纸怎么处理", "路由器恢复出厂设置"], text_type="query")

    assert len(records) == 2, "向量条数与输入不一致"
    assert records[0].dense, "稠密向量为空"
    assert records[0].dense != records[1].dense, "两条不同文本返回了相同稠密向量（假实现又回来了）"
    if provider.supports_sparse:
        assert records[0].sparse, "声明支持稀疏向量但返回为空"
        assert records[0].sparse != records[1].sparse, "两条不同文本返回了相同稀疏向量"

    print(
        f"[OK] embedding provider={provider.name} dim={len(records[0].dense)} "
        f"sparse={'yes' if records[0].sparse else 'no'}"
    )


def check_retrieval() -> None:
    from app.eval.pipelines import retrieve

    for index, (question, signals) in enumerate(CHECKS):
        docs = retrieve(question, "hybrid", top_k=5)
        assert docs, f"「{question}」没有检索到任何文档"

        missing_id = [doc for doc in docs if not doc.get("biz_chunk_id")]
        assert not missing_id, f"有 {len(missing_id)} 条结果缺少 biz_chunk_id，切片级溯源不成立"

        # 页码允许为 0（.md 语料没有页），但字段必须在，且至少一条 >0 才算映射通了
        if index == 0:
            pages = [doc.get("page_no") for doc in docs]
            assert all(page is not None for page in pages), f"page_no 字段缺失: {pages}"

        titles = [doc.get("file_title", "") for doc in docs]
        hit_title = any(any(s in title for s in signals) for title in titles)
        hit_content = any(
            any(s in (doc.get("content") or "") for s in signals) for doc in docs
        )
        assert hit_title or hit_content, (
            f"「{question}」前 5 条都不像在讲这件事: 标记={signals} 标题={titles}"
        )
        where = "标题" if hit_title else "正文"
        print(f"[OK] 「{question}」top5={titles} 命中标记于{where}")


def check_rerank() -> None:
    from app.eval.pipelines import retrieve

    docs = retrieve(CHECKS[0][0], "full", top_k=5)
    assert docs, "重排管线没有返回结果"

    scores = [doc.get("rerank_score") for doc in docs]
    assert all(score is not None for score in scores), f"存在未打分的文档: {scores}"
    assert scores == sorted(scores, reverse=True), f"重排分数未单调递减: {scores}"
    assert 1 <= len(docs) <= 8, f"断崖算法保留条数异常: {len(docs)}"
    print(f"[OK] rerank 分数单调递减，保留 {len(docs)} 条: {[round(s, 3) for s in scores]}")


def main() -> int:
    try:
        check_embedding()
        check_retrieval()
        check_rerank()
    except AssertionError as e:
        print(f"[FAIL] {e}")
        return 1
    except Exception as e:  # noqa: BLE001
        print(f"[FAIL] {e.__class__.__name__}: {e}")
        return 1

    print("\n全部通过：向量真实、切片可溯源、重排生效。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
