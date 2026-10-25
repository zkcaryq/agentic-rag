#!/usr/bin/env python
"""检索评测脚本：在 golden set 上对比四条管线的指标。

用法（仓库根目录）：
    .venv/Scripts/python.exe scripts/eval_retrieval.py --pipelines all --k 5
    .venv/Scripts/python.exe scripts/eval_retrieval.py --pipelines dense,hybrid --limit 20

产出：
    output/eval/retrieval_<时间戳>.json   完整明细
    docs/rag-eval.md                      对比表（追加一节）

指标用关键词子串匹配 file_title，并做文档级去重 + 每个相关条目只计一次，
避免同一文档多个 chunk 或同一关键词多次命中把指标撑高。
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

from app.eval.golden_set import GOLDEN_SET  # noqa: E402
from app.eval.pipelines import PIPELINES, retrieve  # noqa: E402
from app.eval.retrieval_metrics import evaluate_retrieval, keyword_matcher  # noqa: E402

OUTPUT_DIR = ROOT / "output" / "eval"
DOCS_DIR = ROOT / "docs"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="检索管线对比评测")
    parser.add_argument("--pipelines", default="all", help="all 或逗号分隔，如 dense,hybrid")
    parser.add_argument("--k", type=int, default=5, help="评估截断位置，默认 5")
    parser.add_argument("--limit", type=int, default=0, help="只跑前 N 条用例（调试用）")
    parser.add_argument("--no-write", action="store_true", help="只打印，不写文件")
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    pipelines = list(PIPELINES) if args.pipelines == "all" else [
        item.strip() for item in args.pipelines.split(",") if item.strip()
    ]
    unknown = [item for item in pipelines if item not in PIPELINES]
    if unknown:
        print(f"未知管线: {unknown}，可选 {PIPELINES}")
        return 1

    cases = GOLDEN_SET[: args.limit] if args.limit else GOLDEN_SET
    print(f"用例数: {len(cases)}，管线: {pipelines}，K={args.k}")

    report = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "k": args.k,
        "case_count": len(cases),
        "pipelines": {},
        "details": {},
    }

    for pipeline in pipelines:
        started = time.perf_counter()
        metric_cases = []
        details = []
        failures = 0

        for index, case in enumerate(cases, start=1):
            try:
                docs = retrieve(case.question, pipeline, top_k=args.k)
            except Exception as e:  # noqa: BLE001 - 评测要跑完全部用例
                failures += 1
                print(f"  [{pipeline}] {case.id} 检索失败: {e}")
                docs = []

            retrieved_titles = [doc.get("file_title", "") for doc in docs]
            metric_cases.append((set(case.relevant_docs), retrieved_titles))
            details.append(
                {
                    "id": case.id,
                    "category": case.category,
                    "question": case.question,
                    "relevant_docs": list(case.relevant_docs),
                    "retrieved": [
                        {
                            "file_title": doc.get("file_title", ""),
                            "biz_chunk_id": doc.get("biz_chunk_id", ""),
                            "page_no": doc.get("page_no", 0),
                            "rerank_score": doc.get("rerank_score"),
                        }
                        for doc in docs
                    ],
                }
            )
            if index % 10 == 0:
                print(f"  [{pipeline}] {index}/{len(cases)}")

        metrics = evaluate_retrieval(metric_cases, k=args.k, matcher=keyword_matcher)
        elapsed = time.perf_counter() - started
        metrics["elapsed_seconds"] = round(elapsed, 1)
        metrics["failures"] = failures
        report["pipelines"][pipeline] = metrics
        report["details"][pipeline] = details

        print(
            f"[{pipeline}] Hit@{args.k}={metrics['hit@k']:.4f} "
            f"Recall@{args.k}={metrics['recall@k']:.4f} MRR={metrics['mrr']:.4f} "
            f"NDCG@{args.k}={metrics['ndcg@k']:.4f} 用时 {elapsed:.1f}s"
        )

    print_relative_table(report)
    if not args.no_write:
        write_outputs(report)
    return 0


def print_relative_table(report: dict) -> None:
    """打印相对基线的提升，这是简历数字的直接来源。"""
    baseline = report["pipelines"].get("dense")
    if not baseline:
        print("\n（未跑 dense 基线，无法给出相对提升）")
        return

    print("\n相对纯稠密基线的提升：")
    print(f"{'管线':<10}{'Recall@K':>12}{'提升':>10}{'Hit@K':>10}{'MRR':>10}")
    for name, metrics in report["pipelines"].items():
        recall = metrics["recall@k"]
        base_recall = baseline["recall@k"]
        lift = (recall - base_recall) / base_recall * 100 if base_recall else 0.0
        print(
            f"{name:<10}{recall:>12.4f}{lift:>9.1f}%{metrics['hit@k']:>10.4f}{metrics['mrr']:>10.4f}"
        )


def write_outputs(report: dict) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    json_path = OUTPUT_DIR / f"retrieval_{stamp}.json"
    json_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"\n明细已写入: {json_path}")

    DOCS_DIR.mkdir(parents=True, exist_ok=True)
    md_path = DOCS_DIR / "rag-eval.md"
    md_path.write_text(render_markdown(report), encoding="utf-8")
    print(f"报告已写入: {md_path}")


def render_markdown(report: dict) -> str:
    k = report["k"]
    lines = [
        "# RAG 检索评测报告",
        "",
        f"- 生成时间：{report['generated_at']}",
        f"- 用例数：{report['case_count']}（golden set，见 `app/eval/golden_set.py`）",
        f"- 评估截断：K = {k}",
        "- 匹配口径：关键词子串匹配 `file_title`，文档级去重，每个相关条目最多命中一次",
        "",
        "## 管线对比",
        "",
        f"| 管线 | Hit@{k} | Recall@{k} | Precision@{k} | MRR | NDCG@{k} | 耗时(s) |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]
    for name, m in report["pipelines"].items():
        lines.append(
            f"| {name} | {m['hit@k']:.4f} | {m['recall@k']:.4f} | {m['precision@k']:.4f} "
            f"| {m['mrr']:.4f} | {m['ndcg@k']:.4f} | {m['elapsed_seconds']} |"
        )

    baseline = report["pipelines"].get("dense")
    if baseline and baseline["recall@k"]:
        lines += [
            "",
            "## 相对基线（dense）的召回提升",
            "",
            "| 管线 | Recall@K | 相对提升 |",
            "| --- | --- | --- |",
        ]
        for name, m in report["pipelines"].items():
            lift = (m["recall@k"] - baseline["recall@k"]) / baseline["recall@k"] * 100
            lines.append(f"| {name} | {m['recall@k']:.4f} | {lift:+.1f}% |")

    lines += [
        "",
        "## 口径说明",
        "",
        "- **基线是纯稠密检索（dense）**，不含稀疏、不含 HyDE、不含重排；",
        "- 报告里的提升只对同一批用例、同一个 golden set 成立，扩充语料后必须重新标注并复测；",
        "- 若实测提升不足预期，以实测数字为准，不调口径凑数。",
        "",
    ]
    return "\n".join(lines)


if __name__ == "__main__":
    raise SystemExit(main())
