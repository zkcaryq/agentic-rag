#!/usr/bin/env python
"""golden set 静态校验（不需要 API key、不需要 Milvus），可直接进 CI。

检查：
  1. id 唯一、必填字段非空、relevant_docs 非空；
  2. 每个相关文档关键词必须能在 doc/ 的真实文件名里命中，否则这条用例不可回答；
  3. 用例总数与「图片/表格型」用例数量达到下限。

用法：
    .venv/Scripts/python.exe scripts/check_golden.py
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

from app.eval.golden_set import GOLDEN_SET, get_categories, get_image_cases  # noqa: E402

MIN_CASES = 50
MIN_IMAGE_CASES = 15
DOC_DIR = ROOT / "doc"
CORPUS_SUFFIXES = {".pdf", ".docx", ".xlsx", ".md"}


def main() -> int:
    errors: list[str] = []

    seen_ids: set[str] = set()
    for case in GOLDEN_SET:
        if case.id in seen_ids:
            errors.append(f"重复 id: {case.id}")
        seen_ids.add(case.id)

        if not case.question.strip():
            errors.append(f"{case.id}: question 为空")
        if not case.expected_answer.strip():
            errors.append(f"{case.id}: expected_answer 为空")
        if not case.relevant_docs:
            errors.append(f"{case.id}: relevant_docs 为空")
        if not case.category.strip():
            errors.append(f"{case.id}: category 为空")

    # 关键词必须能在真实语料文件名里命中，否则这条用例根本不可能被召回
    corpus_files = []
    if DOC_DIR.exists():
        corpus_files = [
            path
            for path in DOC_DIR.iterdir()
            if path.is_file()
            and path.suffix.lower() in CORPUS_SUFFIXES
            and path.name.lower() != "readme.md"
        ]
    if corpus_files:
        titles = [path.stem for path in corpus_files]
        for case in GOLDEN_SET:
            for keyword in case.relevant_docs:
                if not any(keyword in title for title in titles):
                    errors.append(
                        f"{case.id}: 关键词「{keyword}」在 doc/ 里找不到对应文档"
                    )
    else:
        print(f"[警告] {DOC_DIR} 中没有实际语料文件，跳过关键词核对")

    if len(GOLDEN_SET) < MIN_CASES:
        errors.append(f"用例数 {len(GOLDEN_SET)} 少于下限 {MIN_CASES}")
    image_cases = len(get_image_cases())
    if image_cases < MIN_IMAGE_CASES:
        errors.append(f"图片/表格型用例 {image_cases} 条，少于下限 {MIN_IMAGE_CASES}")

    categories = get_categories()
    print(f"用例总数: {len(GOLDEN_SET)}")
    print(f"分类分布: {categories}")
    print(f"图片/表格型: {image_cases}")

    if errors:
        print("\n校验失败：")
        for message in errors:
            print(f"  - {message}")
        return 1

    print("\ngolden set 校验通过")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
