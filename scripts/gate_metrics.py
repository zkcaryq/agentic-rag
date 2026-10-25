#!/usr/bin/env python
"""指标门禁：读取最近一次评测结果，低于阈值即失败（可接 CI）。

用法：
    .venv/Scripts/python.exe scripts/gate_metrics.py
    .venv/Scripts/python.exe scripts/gate_metrics.py --pipeline full \
        --min-recall 0.6 --min-mrr 0.4 --min-ndcg 0.5
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EVAL_DIR = ROOT / "output" / "eval"


def latest_report() -> Path | None:
    if not EVAL_DIR.exists():
        return None
    reports = sorted(EVAL_DIR.glob("retrieval_*.json"))
    return reports[-1] if reports else None


def main() -> int:
    parser = argparse.ArgumentParser(description="检索指标门禁")
    parser.add_argument("--pipeline", default="full", help="要校验的管线，默认 full")
    parser.add_argument("--min-recall", type=float, default=0.60)
    parser.add_argument("--min-mrr", type=float, default=0.40)
    parser.add_argument("--min-ndcg", type=float, default=0.50)
    parser.add_argument("--report", help="指定评测结果文件，默认取最新")
    args = parser.parse_args()

    path = Path(args.report) if args.report else latest_report()
    if not path or not path.exists():
        print("找不到评测结果，先跑 scripts/eval_retrieval.py")
        return 1

    report = json.loads(path.read_text(encoding="utf-8"))
    metrics = report.get("pipelines", {}).get(args.pipeline)
    if not metrics:
        print(f"报告里没有管线 {args.pipeline}，可用: {list(report.get('pipelines', {}))}")
        return 1

    checks = [
        ("Recall@K", metrics["recall@k"], args.min_recall),
        ("MRR", metrics["mrr"], args.min_mrr),
        ("NDCG@K", metrics["ndcg@k"], args.min_ndcg),
    ]

    failed = False
    print(f"报告: {path.name}  管线: {args.pipeline}")
    for name, actual, threshold in checks:
        status = "OK" if actual >= threshold else "FAIL"
        if actual < threshold:
            failed = True
        print(f"  [{status:<4}] {name:<10} 实际 {actual:.4f}  阈值 {threshold:.4f}")

    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
