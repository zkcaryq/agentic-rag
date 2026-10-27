"""检索指标纯函数单测（不依赖 Milvus / 网络 / API key）。"""
import math

from app.eval.retrieval_metrics import (
    dedupe_docs,
    evaluate_retrieval,
    hit_at_k,
    keyword_matcher,
    mrr,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
)


def test_dedupe_keeps_first_occurrence():
    assert dedupe_docs(["a", "a", "b", "", "b", "c"]) == ["a", "b", "c"]


def test_hit_at_k():
    relevant = {"docA"}
    assert hit_at_k(relevant, ["docX", "docA"], 5) == 1
    assert hit_at_k(relevant, ["docX", "docA"], 1) == 0


def test_recall_counts_each_entry_once():
    # 同一相关条目被两个文档命中，只能算一次，否则 recall 会超过 1
    retrieved = ["Pantum P3030 manual", "Pantum P3030 user guide"]
    recall = recall_at_k(["P3030"], retrieved, 5, matcher=keyword_matcher)
    assert recall == 1.0


def test_recall_never_exceeds_one():
    retrieved = ["P3030 a", "P3030 b", "P3030 c"]
    assert recall_at_k(["P3030", "P3500"], retrieved, 5, matcher=keyword_matcher) == 0.5


def test_precision_at_k():
    retrieved = ["P3030 a", "unrelated", "P3030 b"]
    precision = precision_at_k(["P3030"], retrieved, 3, matcher=keyword_matcher)
    # 去重后只命中 1 个相关条目，分母是 min(k, len(retrieved)) = 3
    assert precision == 1 / 3


def test_mrr():
    assert mrr(["docA"], ["docX", "docA"]) == 0.5
    assert mrr(["docA"], ["docA"]) == 1.0
    assert mrr(["docA"], ["docX"]) == 0.0


def test_ndcg_perfect_ranking_is_one():
    retrieved = ["docA", "docB"]
    assert ndcg_at_k(["docA", "docB"], retrieved, 2) == 1.0


def test_ndcg_partial():
    # docB 命中在第 1 位（DCG 权重 1/log2(2)=1），理想排序是两条都命中
    value = ndcg_at_k(["docA", "docB"], ["docB", "docX"], 2)
    expected = 1.0 / (1 + 1 / math.log2(3))
    assert abs(value - expected) < 1e-9


def test_evaluate_aggregates():
    cases = [
        ({"docA"}, ["docA", "docX"]),
        ({"docB"}, ["docX", "docY"]),
    ]
    metrics = evaluate_retrieval(cases, k=2)
    assert metrics["hit@k"] == 0.5
    assert metrics["recall@k"] == 0.5
    assert metrics["mrr"] == 0.5


def test_empty_cases_returns_zeros():
    metrics = evaluate_retrieval([], k=5)
    assert all(value == 0.0 for value in metrics.values())


def test_matcher_is_not_global_state():
    """两次调用用不同 matcher，结果不能互相污染（所以 matcher 只能按参数传，不能有模块级状态）。"""
    retrieved = ["BAAI/bge-m3 spec"]
    exact = recall_at_k(["bge-m3"], retrieved, 5)
    keyword = recall_at_k(["bge-m3"], retrieved, 5, matcher=keyword_matcher)
    assert exact == 0.0
    assert keyword == 1.0
    assert recall_at_k(["bge-m3"], retrieved, 5) == 0.0
