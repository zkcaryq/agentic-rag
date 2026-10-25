"""检索质量指标：Hit@K / Recall@K / Precision@K / MRR / NDCG@K。

纯函数、零第三方依赖，可在没有 Milvus / 没有 API key 的环境下单测。

两处口径是刻意设计的，防止指标虚高：
  1. 文档级去重：同一文档的多个 chunk 只按首次出现计一次；
  2. 每个相关条目最多命中一次：子串匹配下同一关键词可能被多个文档命中，
     不去重会让 recall 超过 1（contract-review 曾实测到 1.055）。

与 contract-review 的实现差异：matcher 一律通过参数传入，
不用模块级全局变量，避免测试之间互相污染。
"""
from __future__ import annotations

import math
from typing import Callable, Iterable, Sequence

Matcher = Callable[[str, str], bool]


def default_matcher(entry: str, doc: str) -> bool:
    """精确匹配：检索结果与相关文档完全相同。"""
    return entry == doc


def keyword_matcher(entry: str, doc: str) -> bool:
    """关键词子串匹配：相关文档写标题关键词，命中检索结果的标题即算相关。"""
    return bool(entry) and entry in (doc or "")


def dedupe_docs(docs: Sequence[str]) -> list[str]:
    """文档级去重，保留首次出现的位置。"""
    seen: set[str] = set()
    result: list[str] = []
    for doc in docs:
        if doc and doc not in seen:
            seen.add(doc)
            result.append(doc)
    return result


def _matched_entries(
    relevant: Iterable[str], retrieved: Sequence[str], matcher: Matcher
) -> list[str]:
    """按检索顺序列出被命中的相关条目，每个条目最多计一次。"""
    matched: list[str] = []
    for entry in relevant:
        for doc in retrieved:
            if matcher(entry, doc):
                matched.append(entry)
                break
    return matched


def hit_at_k(relevant, retrieved, k: int, matcher: Matcher = default_matcher) -> int:
    return int(any(matcher(entry, doc) for entry in relevant for doc in retrieved[:k]))


def recall_at_k(relevant, retrieved, k: int, matcher: Matcher = default_matcher) -> float:
    relevant = list(relevant)
    if not relevant:
        return 0.0
    return len(_matched_entries(relevant, retrieved[:k], matcher)) / len(relevant)


def precision_at_k(relevant, retrieved, k: int, matcher: Matcher = default_matcher) -> float:
    if k <= 0 or not retrieved:
        return 0.0
    matched = len(_matched_entries(list(relevant), retrieved[:k], matcher))
    return matched / min(k, len(retrieved))


def mrr(relevant, retrieved, matcher: Matcher = default_matcher) -> float:
    relevant = list(relevant)
    for index, doc in enumerate(retrieved):
        if any(matcher(entry, doc) for entry in relevant):
            return 1.0 / (index + 1)
    return 0.0


def ndcg_at_k(relevant, retrieved, k: int, matcher: Matcher = default_matcher) -> float:
    relevant = list(relevant)
    top = retrieved[:k]
    matched = _matched_entries(relevant, top, matcher)
    if not matched:
        return 0.0

    dcg = 0.0
    for entry in matched:
        for index, doc in enumerate(top):
            if matcher(entry, doc):
                dcg += 1.0 / math.log2(index + 2)
                break

    ideal = sum(1.0 / math.log2(i + 2) for i in range(min(len(relevant), k)))
    return dcg / ideal if ideal > 0 else 0.0


def evaluate_retrieval(
    cases: Sequence[tuple[set[str], list[str]]],
    k: int = 5,
    matcher: Matcher = default_matcher,
) -> dict:
    """cases: [(relevant_docs, retrieved_docs)] → 聚合指标。"""
    empty = {"hit@k": 0.0, "recall@k": 0.0, "precision@k": 0.0, "mrr": 0.0, "ndcg@k": 0.0}
    if not cases:
        return empty

    normalised = [(list(relevant), dedupe_docs(retrieved)) for relevant, retrieved in cases]
    total = len(normalised)
    return {
        "hit@k": sum(hit_at_k(r, d, k, matcher) for r, d in normalised) / total,
        "recall@k": sum(recall_at_k(r, d, k, matcher) for r, d in normalised) / total,
        "precision@k": sum(precision_at_k(r, d, k, matcher) for r, d in normalised) / total,
        "mrr": sum(mrr(r, d, matcher) for r, d in normalised) / total,
        "ndcg@k": sum(ndcg_at_k(r, d, k, matcher) for r, d in normalised) / total,
    }
