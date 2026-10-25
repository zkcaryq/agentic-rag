"""
Office 转 Markdown 节点
把 .docx / .xlsx 转成结构化 Markdown，接入后续的切分与向量化链路。

选 python-docx + openpyxl 而不是 unstructured：后者会带一堆重依赖
（含 torch 系），而这里只需要标题层级、段落、表格和 sheet 名。
"""
import sys
from pathlib import Path

from app.core.logger import logger
from app.import_process.agent.state import ImportGraphState
from app.utils.task_utils import add_running_task, add_done_task

# Word 里这些样式名视为标题级别
_HEADING_PREFIX = "heading"


def node_office_to_md(state: ImportGraphState) -> ImportGraphState:
    """把 Word / Excel 转成 Markdown 并写盘。"""
    func_name = sys._getframe().f_code.co_name
    logger.info(f">>> 开始执行节点: {func_name}")

    add_running_task(state.get("task_id", ""), func_name)

    try:
        file_path = Path(state.get("local_file_path", ""))
        if not file_path.exists():
            raise FileNotFoundError(f"文件不存在: {file_path}")

        suffix = file_path.suffix.lower()
        if suffix == ".docx":
            markdown = docx_to_markdown(str(file_path))
        elif suffix == ".xlsx":
            markdown = xlsx_to_markdown(str(file_path))
        else:
            raise ValueError(f"不支持的文件类型: {suffix}")

        if not markdown.strip():
            raise RuntimeError("Office 解析结果为空")

        local_dir = Path(state.get("local_dir") or file_path.parent)
        local_dir.mkdir(parents=True, exist_ok=True)
        md_path = local_dir / f"{file_path.stem}.md"
        md_path.write_text(markdown, encoding="utf-8")

        state["md_path"] = str(md_path)
        state["md_content"] = markdown
        logger.info(f"Office 转 Markdown 完成: {md_path}，{len(markdown)} 字符")

    except Exception as e:
        state["failed_stage"] = "office"
        logger.error(f"Office 解析失败: {e}", exc_info=True)
        add_done_task(state.get("task_id", ""), func_name)
        raise

    add_done_task(state.get("task_id", ""), func_name)
    logger.info(f">>> 节点执行完成: {func_name}")
    return state


def docx_to_markdown(path: str) -> str:
    """Word → Markdown：保留标题层级、段落和表格。"""
    from docx import Document
    from docx.table import Table
    from docx.text.paragraph import Paragraph

    document = Document(path)
    lines: list[str] = [f"# {Path(path).stem}", ""]

    for block in _iter_docx_blocks(document):
        if isinstance(block, Paragraph):
            text = (block.text or "").strip()
            if not text:
                continue
            level = _heading_level(block.style.name if block.style else "")
            lines.append(f"{'#' * level} {text}" if level else text)
            lines.append("")
        elif isinstance(block, Table):
            lines.extend(_table_to_markdown(block))
            lines.append("")

    return "\n".join(lines).strip() + "\n"


def _iter_docx_blocks(document):
    """按文档顺序遍历段落与表格（python-docx 默认把两者分开存）。"""
    from docx.table import Table
    from docx.text.paragraph import Paragraph

    body = document.element.body
    for child in body.iterchildren():
        tag = child.tag.split("}")[-1]
        if tag == "p":
            yield Paragraph(child, document)
        elif tag == "tbl":
            yield Table(child, document)


def _heading_level(style_name: str) -> int:
    """把 Word 的 Heading N 样式映射成 Markdown 的 # 数量。"""
    name = (style_name or "").strip().lower()
    if not name.startswith(_HEADING_PREFIX):
        return 0
    tail = name[len(_HEADING_PREFIX):].strip()
    if tail.isdigit():
        return max(1, min(int(tail), 6))
    # 中文版 Word 可能是 "标题 1"
    for index in range(1, 7):
        if str(index) in name:
            return index
    return 1


def _table_to_markdown(table) -> list[str]:
    rows = []
    for row in table.rows:
        cells = [((cell.text or "").strip().replace("\n", " ")) for cell in row.cells]
        if any(cells):
            rows.append(cells)
    if not rows:
        return []

    header = rows[0]
    lines = ["| " + " | ".join(header) + " |",
             "| " + " | ".join(["---"] * len(header)) + " |"]
    for row in rows[1:]:
        padded = row + [""] * (len(header) - len(row))
        lines.append("| " + " | ".join(padded[: len(header)]) + " |")
    return lines


def xlsx_to_markdown(path: str) -> str:
    """Excel → Markdown：每个 sheet 一个标题，表格带行号便于溯源。"""
    from openpyxl import load_workbook

    workbook = load_workbook(path, read_only=True, data_only=True)
    lines: list[str] = [f"# {Path(path).stem}", ""]

    try:
        for sheet in workbook.worksheets:
            lines.append(f"## Sheet: {sheet.title}")
            lines.append("")

            rows = []
            for row_index, row in enumerate(sheet.iter_rows(values_only=True), start=1):
                values = ["" if cell is None else str(cell).strip() for cell in row]
                if any(values):
                    rows.append((row_index, values))

            if not rows:
                lines.append("（空表）")
                lines.append("")
                continue

            width = max(len(values) for _, values in rows)
            header = [f"列{i + 1}" for i in range(width)]
            lines.append("| 行号 | " + " | ".join(header) + " |")
            lines.append("| --- | " + " | ".join(["---"] * width) + " |")
            for row_index, values in rows:
                padded = values + [""] * (width - len(values))
                lines.append(f"| {row_index} | " + " | ".join(padded) + " |")
            lines.append("")
    finally:
        workbook.close()

    return "\n".join(lines).strip() + "\n"
