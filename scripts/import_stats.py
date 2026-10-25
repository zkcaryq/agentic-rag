#!/usr/bin/env python
"""导入成功率 / 失败率统计（数据来自 PostgreSQL import_tasks 表）。

用法：
    .venv/Scripts/python.exe scripts/import_stats.py
    .venv/Scripts/python.exe scripts/import_stats.py --json

口径：
    分母 = completed + failed（不含 pending/processing）
    partial_success 单列，不计入分母
    失败按 failed_stage 归因
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

from app.clients.pg_task_utils import aggregate_stats  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="导入失败率统计")
    parser.add_argument("--json", action="store_true", help="以 JSON 输出")
    args = parser.parse_args()

    try:
        stats = aggregate_stats()
    except Exception as e:  # noqa: BLE001
        print(f"读取任务表失败: {e}")
        return 1

    if args.json:
        print(json.dumps(stats, ensure_ascii=False, indent=2, default=str))
        return 0

    print("按状态统计：")
    for status, info in stats["by_status"].items():
        print(f"  {status:<16} 任务 {info['count']:>4}  切片 {info['chunks']:>6}")

    print(f"\n分母（completed + failed）: {stats['denominator']}")
    print(f"失败任务数: {stats['failed']}")
    print(f"失败率: {stats['failure_rate'] * 100:.2f}%")

    if stats["failed_stage_distribution"]:
        print("\n失败阶段分布：")
        for stage, count in stats["failed_stage_distribution"].items():
            print(f"  {stage:<12} {count}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
