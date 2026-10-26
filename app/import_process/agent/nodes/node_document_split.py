"""
文档切分节点
将长文档切分成适合检索的文本块（Chunk）
"""
import os
import sys
import re
import json
from pathlib import Path
from typing import List, Dict, Any, Tuple

from app.core.logger import logger
from app.import_process.agent.state import ImportGraphState
from app.utils.task_utils import add_running_task, add_done_task

# 配置参数
DEFAULT_MAX_CONTENT_LENGTH = 2000  # 单个Chunk最大字符数
MIN_CONTENT_LENGTH = 500  # 短Chunk合并阈值


def node_document_split(state: ImportGraphState) -> ImportGraphState:
    """
    文档切分节点

    核心流程：
        1. 获取MD内容
        2. 按标题切分
        3. 处理超长/过短Chunk
        4. 保存切分结果

    Args:
        state: 导入流程状态

    Returns:
        更新后的状态，新增 chunks 字段
    """
    func_name = sys._getframe().f_code.co_name
    logger.info(f">>> 开始执行节点: {func_name}")

    add_running_task(state.get("task_id", ""), func_name)

    try:
        # 步骤1：获取输入
        content, file_title, max_len = step_1_get_inputs(state)
        if not content:
            logger.warning("无MD内容，跳过切分")
            return state

        # 步骤2：按标题切分
        sections = step_2_split_by_titles(content, file_title)

        # 步骤3：精细化处理
        chunks = step_3_refine_chunks(sections, max_len)

        # 步骤3.5：把页码带到每个 chunk（MinerU 的 content_list.json 里有 page_idx）
        chunks = attach_page_numbers(chunks, state.get("local_dir", ""))

        # 步骤4：保存结果
        state["chunks"] = chunks
        step_4_save_chunks(state, chunks)

        logger.info(f"切分完成，共生成 {len(chunks)} 个Chunk")

    except Exception as e:
        logger.exception(f"文档切分失败: {e}")

    finally:
        add_done_task(state.get("task_id", ""), func_name)
        logger.info(f">>> 节点执行完成: {func_name}")

    return state


def step_1_get_inputs(state: ImportGraphState) -> Tuple[str, str, int]:
    """
    步骤1：获取并预处理输入数据
    """
    content = state.get("md_content", "")
    file_title = state.get("file_title", "Unknown")
    max_len = DEFAULT_MAX_CONTENT_LENGTH

    if not content:
        # 尝试从文件读取
        md_path = state.get("md_path", "")
        if md_path and os.path.exists(md_path):
            with open(md_path, "r", encoding="utf-8") as f:
                content = f.read()
            logger.info(f"从文件读取MD内容: {md_path}")

    if content:
        # 统一换行符
        content = content.replace("\r\n", "\n").replace("\r", "\n")
        logger.info(f"MD内容长度: {len(content)} 字符")

    return content, file_title, max_len


def step_2_split_by_titles(content: str, file_title: str) -> List[Dict[str, Any]]:
    """
    步骤2：按Markdown标题切分
    """
    # 正则匹配标题行
    title_pattern = r'^(#{1,6})\s+(.+)$'

    lines = content.split("\n")
    sections = []
    current_title = ""
    current_content = []
    in_code_block = False

    def flush_section():
        if current_content:
            sections.append({
                "title": current_title or "无标题",
                "content": "\n".join(current_content).strip(),
                "file_title": file_title,
                "parent_title": ""
            })

    for line in lines:
        stripped = line.strip()

        # 处理代码块
        if stripped.startswith("```") or stripped.startswith("~~~"):
            in_code_block = not in_code_block
            current_content.append(line)
            continue

        # 识别标题
        if not in_code_block:
            match = re.match(title_pattern, line)
            if match:
                flush_section()
                current_title = line.strip()
                current_content = [line]
                continue

        current_content.append(line)

    # 处理最后一个章节
    flush_section()

    logger.info(f"按标题切分完成，共 {len(sections)} 个章节")
    return sections


def step_3_refine_chunks(sections: List[Dict[str, Any]], max_len: int) -> List[Dict[str, Any]]:
    """
    步骤3：精细化处理Chunk
    - 切分超长章节
    - 合并过短章节
    """
    # 切分超长章节
    refined = []
    for sec in sections:
        chunks = _split_long_chunk(sec, max_len)
        refined.extend(chunks)

    # 合并过短章节
    final = _merge_short_chunks(refined)

    # 添加索引
    for i, chunk in enumerate(final):
        chunk["chunk_index"] = i
        chunk["chunk_id"] = f"{chunk.get('file_title', 'unknown')}_{i}"

    logger.info(f"精细化处理完成: {len(sections)} -> {len(refined)} -> {len(final)}")
    return final


def _split_long_chunk(chunk: Dict[str, Any], max_len: int) -> List[Dict[str, Any]]:
    """
    切分超长Chunk
    """
    content = chunk.get("content", "")
    if len(content) <= max_len:
        return [chunk]

    title = chunk.get("title", "")
    file_title = chunk.get("file_title", "")

    # 按段落切分
    paragraphs = content.split("\n\n")
    sub_chunks = []
    current = ""

    for para in paragraphs:
        if len(current) + len(para) + 2 <= max_len:
            if current:
                current += "\n\n" + para
            else:
                current = para
        else:
            if current:
                sub_chunks.append({
                    "title": f"{title}-{len(sub_chunks) + 1}" if title else f"chunk-{len(sub_chunks) + 1}",
                    "content": current,
                    "file_title": file_title,
                    "parent_title": title
                })
            current = para

    if current:
        sub_chunks.append({
            "title": f"{title}-{len(sub_chunks) + 1}" if title else f"chunk-{len(sub_chunks) + 1}",
            "content": current,
            "file_title": file_title,
            "parent_title": title
        })

    return sub_chunks


def _merge_short_chunks(chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    合并过短Chunk
    """
    if len(chunks) <= 1:
        return chunks

    merged = []
    current = chunks[0].copy()

    for next_chunk in chunks[1:]:
        # 检查是否可以合并
        if len(current["content"]) < MIN_CONTENT_LENGTH:
            # 合并
            current["content"] += "\n\n" + next_chunk["content"]
            # 更新标题
            if next_chunk.get("title"):
                current["title"] = next_chunk["title"]
        else:
            merged.append(current)
            current = next_chunk.copy()

    merged.append(current)
    return merged


def attach_page_numbers(
    chunks: List[Dict[str, Any]], local_dir: str
) -> List[Dict[str, Any]]:
    """给每个 chunk 标注页码，用于答案里的逐段溯源。

    数据来源是 MinerU 产物里的 content_list.json：每一条内容带 page_idx（从 0 开始）。
    做法是把每条内容的正文压缩成 40 字的前缀做锚点，按顺序在 chunk 正文里找，
    命中即取该条目的页码。找不到的 chunk 记 0（表示未知），不猜测。
    """
    items = _load_content_list(local_dir)
    if not items:
        for chunk in chunks:
            chunk.setdefault("page_no", 0)
        return chunks

    anchors: List[tuple[str, int]] = []
    for item in items:
        text = (item.get("text") or "").strip()
        page_idx = item.get("page_idx")
        if not text or page_idx is None:
            continue
        needle = re.sub(r"\s+", "", text)[:40]
        if len(needle) < 10:
            continue
        # page_idx 从 0 开始，这里转成人类习惯的 1-based 页码
        anchors.append((needle, int(page_idx) + 1))

    if not anchors:
        for chunk in chunks:
            chunk.setdefault("page_no", 0)
        return chunks

    cursor = 0
    matched = 0
    for chunk in chunks:
        content = re.sub(r"\s+", "", chunk.get("content", ""))
        page_no = 0
        for index in range(cursor, len(anchors)):
            needle, page = anchors[index]
            if needle in content:
                page_no = page
                cursor = index
                matched += 1
                break
        chunk["page_no"] = page_no

    logger.info(f"页码映射完成：{matched}/{len(chunks)} 个 chunk 命中页码锚点")
    return chunks


def _load_content_list(local_dir: str) -> List[Dict[str, Any]]:
    """读取 MinerU 的 content_list.json；没有就返回空列表。"""
    if not local_dir:
        return []
    try:
        candidates = sorted(Path(local_dir).rglob("*content_list.json"))
    except Exception:
        return []
    if not candidates:
        return []

    # 一个任务目录下可能有多个文档，取体积最大的那份作为主文档
    target = max(candidates, key=lambda p: p.stat().st_size)
    try:
        with open(target, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, list) else []
    except Exception as e:
        logger.warning(f"读取 content_list.json 失败: {e}")
        return []


def step_4_save_chunks(state: ImportGraphState, chunks: List[Dict[str, Any]]):
    """
    步骤4：保存切分结果到文件
    """
    local_dir = state.get("local_dir", "")
    if not local_dir:
        return

    chunks_path = Path(local_dir) / "chunks.json"
    try:
        with open(chunks_path, "w", encoding="utf-8") as f:
            json.dump(chunks, f, ensure_ascii=False, indent=2)
        logger.info(f"Chunk结果已保存: {chunks_path}")
    except Exception as e:
        logger.warning(f"保存Chunk失败: {e}")
