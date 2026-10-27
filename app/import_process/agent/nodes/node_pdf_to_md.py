"""
MinerU 解析节点
把 PDF / Word / Excel 交给 MinerU 云端 API 转成结构化 Markdown。

文件名仍叫 node_pdf_to_md（图节点名，改动会牵动断点快照里的节点名），
但职责不止 PDF：实测 MinerU 也收 .docx/.xlsx 并返回 Markdown（表格为 <table> HTML）。
Office 走不走 MinerU 由 OFFICE_PARSE_ENGINE 控制，见 app/conf/mineru_config.py。
"""
import os
import sys
import time
import json
import requests
import zipfile
import shutil
from pathlib import Path

from app.core.logger import logger
from app.import_process.agent.state import ImportGraphState
from app.utils.task_utils import add_running_task, add_done_task
from app.conf.mineru_config import mineru_config


# MinerU配置
MINERU_BASE_URL = mineru_config.base_url
MINERU_API_TOKEN = mineru_config.api_key


def node_pdf_to_md(state: ImportGraphState) -> ImportGraphState:
    """
    PDF转Markdown节点

    核心流程：
        1. 校验PDF路径和输出目录
        2. 上传PDF至MinerU并轮询解析结果（超过单次页数上限时切段逐段提交）
        3. 下载ZIP包并提取MD文件（分段时合并为单本，页码按段偏移校正）
        4. 读取MD内容并更新状态

    Args:
        state: 导入流程状态

    Returns:
        更新后的状态，新增 md_path 和 md_content
    """
    func_name = sys._getframe().f_code.co_name
    logger.info(f">>> 开始执行节点: {func_name}")

    # 记录任务开始
    add_running_task(state.get("task_id", ""), func_name)

    try:
        # 步骤1：校验路径
        pdf_path, output_dir = step_1_validate_paths(state)

        # 步骤2+3：提交 MinerU 并取回 Markdown（超过单次页数上限会自动分段再拼回单本）
        md_path, submit_count = parse_with_page_limit(pdf_path, output_dir)
        if submit_count > 1:
            logger.info(f"【{func_name}】本份文档分 {submit_count} 段提交后已合并为单一 Markdown")

        # 更新状态
        state["md_path"] = md_path
        logger.info(f"【{func_name}】MD文件生成成功: {md_path}")

        # 读取MD内容
        try:
            with open(md_path, "r", encoding="utf-8") as f:
                state["md_content"] = f.read()
            logger.info(f"【{func_name}】MD内容读取成功，长度: {len(state['md_content'])} 字符")
        except Exception as e:
            logger.error(f"【{func_name}】读取MD文件失败: {e}")

    except Exception as e:
        target = Path(state.get("pdf_path") or state.get("local_file_path", ""))
        if target.suffix.lower() in (".docx", ".xlsx") and mineru_config.office_fallback_native:
            # Word/Excel 本地解析器兜得住，退一步比整单失败强（PDF 没有本地兜底，照旧抛出）。
            # failed_stage 记成 mineru_fallback：任务算"部分成功"，不混进失败率分母，
            # 但降级这件事在统计里看得见。
            logger.exception(f"【{func_name}】MinerU 解析 Office 失败，退回本地解析器: {e}")
            state["failed_stage"] = "mineru_fallback"
            from app.import_process.agent.nodes.node_office_to_md import node_office_to_md

            return node_office_to_md(state)

        state["failed_stage"] = "mineru"
        logger.exception(f"【{func_name}】MinerU 解析失败: {e}")
        raise

    finally:
        add_done_task(state.get("task_id", ""), func_name)
        logger.info(f">>> 节点执行完成: {func_name}")

    return state


def step_1_validate_paths(state: ImportGraphState):
    """步骤1：校验待解析文件路径和输出目录。

    这里的"文件"不一定是 PDF：Office 文件默认也走 MinerU（实测它收 docx/xlsx），
    而 node_entry 只给 PDF 设 `pdf_path`，所以取不到时回退到 `local_file_path`。
    """
    pdf_path = state.get("pdf_path") or state.get("local_file_path", "")
    local_dir = state.get("local_dir", "")

    if not pdf_path:
        raise ValueError("待解析文件路径为空")
    if not local_dir:
        raise ValueError("输出目录为空")

    pdf_path_obj = Path(pdf_path)
    output_dir_obj = Path(local_dir)

    if not pdf_path_obj.exists():
        raise FileNotFoundError(f"待解析文件不存在: {pdf_path}")

    output_dir_obj.mkdir(parents=True, exist_ok=True)

    logger.info(f"PDF路径: {pdf_path_obj}")
    logger.info(f"输出目录: {output_dir_obj}")

    return pdf_path_obj, output_dir_obj


def parse_with_page_limit(pdf_path_obj: Path, output_dir_obj: Path) -> tuple[str, int]:
    """提交 MinerU 解析，返回 (Markdown 路径, 提交次数)。

    MinerU 服务端对单个任务有页数硬上限（实测 200 页，报
    "number of pages exceeds limit (200 pages), please split the file and try
    again"）。语料里 4 本原厂手册因此全部导入失败，其中一本 630 页——这不是
    边角情况，所以在这里按上限切段逐段提交，再把各段产物拼回「单本」的样子，
    下游节点不必知道切过段。
    """
    pages = _pdf_page_count(pdf_path_obj)
    limit = mineru_config.max_pages_per_task
    if limit <= 0 or pages is None or pages <= limit:
        zip_url = step_2_upload_and_poll(pdf_path_obj, output_dir_obj)
        return step_3_download_and_extract(zip_url, output_dir_obj, pdf_path_obj.stem), 1

    ranges = _segment_ranges(pages, limit)
    logger.info(f"PDF 共 {pages} 页，超过 MinerU 单次上限 {limit} 页，切为 {len(ranges)} 段")

    parts: list[dict] = []
    try:
        for order, (start, end) in enumerate(ranges, start=1):
            part_name = f"{pdf_path_obj.stem}_part{order:02d}"
            part_path = output_dir_obj / f"{part_name}.pdf"
            _write_pdf_range(pdf_path_obj, part_path, start, end)

            zip_url = step_2_upload_and_poll(part_path, output_dir_obj)
            part_md = step_3_download_and_extract(zip_url, output_dir_obj, part_name)
            parts.append(
                {
                    "md_path": part_md,
                    "extract_dir": str(output_dir_obj / part_name),
                    "page_offset": start,
                }
            )
            logger.info(f"第 {order}/{len(ranges)} 段解析完成（原文件第 {start + 1}-{end} 页）")

        md_path = merge_part_outputs(
            parts, output_dir_obj / pdf_path_obj.stem, pdf_path_obj.stem
        )
    finally:
        _cleanup_parts(output_dir_obj, pdf_path_obj.stem, parts)

    return md_path, len(ranges)


def _pdf_page_count(pdf_path_obj: Path) -> int | None:
    """读页数。读不出来就返回 None，按"不分段"处理，让服务端去报真实错误。"""
    if pdf_path_obj.suffix.lower() != ".pdf":
        return None
    try:
        from pypdf import PdfReader

        return len(PdfReader(str(pdf_path_obj), strict=False).pages)
    except Exception as e:
        logger.warning(f"读取 PDF 页数失败，按整本提交: {e}")
        return None


def _segment_ranges(total_pages: int, limit: int) -> list[tuple[int, int]]:
    """页区间列表，(首页, 尾页) 左闭右开、0-based。

    limit <= 0 视为"没配上限"，整本一段返回——不能把 0 当步长，那是死循环。
    """
    if total_pages <= 0:
        return []
    if limit <= 0:
        return [(0, total_pages)]
    return [
        (start, min(start + limit, total_pages))
        for start in range(0, total_pages, limit)
    ]


def _write_pdf_range(pdf_path_obj: Path, part_path: Path, start: int, end: int) -> None:
    from pypdf import PdfReader, PdfWriter

    reader = PdfReader(str(pdf_path_obj), strict=False)
    writer = PdfWriter()
    for index in range(start, end):
        writer.add_page(reader.pages[index])
    with open(part_path, "wb") as f:
        writer.write(f)


def _cleanup_parts(output_dir_obj: Path, stem: str, parts: list[dict]) -> None:
    """删掉分段中间产物。

    必须删：attach_page_numbers 是在任务目录里 rglob content_list.json 取最大的
    那份，留着各段自己的 json，页码就会捡到某一段的局部页码（第 201 页被写成第 1 页）。
    """
    for part in parts:
        shutil.rmtree(Path(part["extract_dir"]), ignore_errors=True)
    for part_pdf in output_dir_obj.glob(f"{stem}_part*.pdf"):
        part_pdf.unlink(missing_ok=True)


def step_2_upload_and_poll(pdf_path_obj: Path, output_dir_obj: Path):
    """
    步骤2：上传PDF至MinerU并轮询结果
    """
    logger.info(f"【step_2】开始处理PDF: {pdf_path_obj.name}")

    # 配置缺失不再降级造假数据：直接失败，由上层记录 failed_stage 并计入失败率
    if not MINERU_BASE_URL or not MINERU_API_TOKEN:
        raise RuntimeError(
            "MinerU 配置缺失：请在 .env 配置 MINERU_BASE_URL 与 MINERU_API_TOKEN"
        )

    try:
        # 请求头
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {MINERU_API_TOKEN}"
        }

        # 1. 获取上传URL
        url_get_upload = f"{MINERU_BASE_URL}/file-urls/batch"
        req_data = {
            "files": [{"name": pdf_path_obj.name}],
            "model_version": mineru_config.model_version
        }

        logger.info(f"请求上传URL: {url_get_upload}")
        resp = requests.post(url_get_upload, headers=headers, json=req_data, timeout=30)

        if resp.status_code != 200:
            raise Exception(f"获取上传URL失败: {resp.status_code}")

        resp_data = resp.json()
        if resp_data.get("code") != 0:
            raise Exception(f"API错误: {resp_data}")

        signed_url = resp_data["data"]["file_urls"][0]
        batch_id = resp_data["data"]["batch_id"]
        logger.info(f"获取上传URL成功, batch_id: {batch_id}")

        # 2. 上传文件
        with open(pdf_path_obj, "rb") as f:
            file_data = f.read()

        # 使用Session禁用代理
        session = requests.Session()
        session.trust_env = False

        put_resp = session.put(signed_url, data=file_data, timeout=60)
        if put_resp.status_code != 200:
            # 重试：指定Content-Type
            headers_pdf = {"Content-Type": "application/pdf"}
            put_resp = session.put(signed_url, data=file_data, headers=headers_pdf, timeout=60)
            if put_resp.status_code != 200:
                raise Exception(f"上传失败: {put_resp.status_code}")

        session.close()
        logger.info("文件上传成功")

        # 3. 轮询任务状态
        poll_url = f"{MINERU_BASE_URL}/extract-results/batch/{batch_id}"
        start_time = time.time()
        timeout_seconds = 600  # 10分钟超时

        while True:
            elapsed = time.time() - start_time
            if elapsed > timeout_seconds:
                raise TimeoutError(f"任务超时: {int(elapsed)}秒")

            poll_resp = requests.get(poll_url, headers=headers, timeout=10)
            if poll_resp.status_code != 200:
                time.sleep(3)
                continue

            poll_data = poll_resp.json()
            if poll_data.get("code") != 0:
                time.sleep(3)
                continue

            results = poll_data.get("data", {}).get("extract_result", [])
            if not results:
                time.sleep(3)
                continue

            result = results[0]
            state_status = result.get("state")

            if state_status == "done":
                zip_url = result.get("full_zip_url")
                if not zip_url:
                    raise Exception("未返回ZIP下载链接")
                logger.info(f"任务完成，耗时: {int(elapsed)}秒")
                return zip_url
            elif state_status == "failed":
                err_msg = result.get("err_msg", "未知错误")
                raise Exception(f"任务失败: {err_msg}")
            else:
                logger.debug(f"处理中... 状态: {state_status}")
                time.sleep(3)

    except Exception as e:
        logger.error(f"MinerU API调用失败: {e}")
        raise


def step_3_download_and_extract(zip_url: str, output_dir_obj: Path, pdf_stem: str) -> str:
    """
    步骤3：下载ZIP并提取MD文件
    """
    logger.info(f"【step_3】开始处理ZIP: {zip_url}")

    # 创建解压目录
    extract_dir = output_dir_obj / pdf_stem
    extract_dir.mkdir(parents=True, exist_ok=True)

    # 下载ZIP
    zip_path = extract_dir / f"{pdf_stem}_result.zip"

    try:
        # 尝试下载
        if zip_url.startswith("http"):
            resp = requests.get(zip_url, timeout=120)
            if resp.status_code == 200:
                with open(zip_path, "wb") as f:
                    f.write(resp.content)
                logger.info(f"ZIP下载成功: {zip_path}")

                # 解压
                with zipfile.ZipFile(zip_path, 'r') as zf:
                    zf.extractall(extract_dir)
                logger.info(f"ZIP解压成功: {extract_dir}")

                # 查找MD文件
                md_files = list(extract_dir.rglob("*.md"))
                if md_files:
                    # 优先选择与PDF同名的
                    target_md = None
                    for md in md_files:
                        if md.stem == pdf_stem:
                            target_md = md
                            break
                    if not target_md:
                        target_md = md_files[0]

                    # 重命名
                    final_md = extract_dir / f"{pdf_stem}.md"
                    if target_md != final_md:
                        shutil.move(str(target_md), str(final_md))

                    logger.info(f"MD文件: {final_md}")
                    return str(final_md)
    except Exception as e:
        logger.error(f"下载/解压失败: {e}")
        raise RuntimeError(f"MinerU 结果下载或解压失败: {e}") from e

    # 曾经这里会写一段假的 "这是从PDF转换而来的Markdown内容"，导致入库的是占位文本。
    # 现在找不到 MD 就直接失败，宁可让导入任务记失败，也不制造假数据。
    raise RuntimeError(f"未在 MinerU 结果包中找到 Markdown 文件: {zip_url}")


def merge_part_outputs(parts: list[dict], final_dir: Path, stem: str) -> str:
    """把各段产物拼成单本：一份 Markdown + 一份页码已校正的 content_list.json。

    段内 page_idx 是从 0 开始的局部页码，不叠加段偏移，答案里的"第几页"就是错的；
    图片文件名是内容哈希，段与段理论上不撞名，但撞上了不能互相覆盖——改名后同步
    改写正文引用和 img_path。
    """
    final_dir.mkdir(parents=True, exist_ok=True)
    images_dir = final_dir / "images"
    images_dir.mkdir(parents=True, exist_ok=True)

    md_chunks: list[str] = []
    merged_items: list[dict] = []

    for order, part in enumerate(parts, start=1):
        extract_dir = Path(part["extract_dir"])
        offset = int(part.get("page_offset") or 0)
        text = Path(part["md_path"]).read_text(encoding="utf-8", errors="ignore")

        renames: dict[str, str] = {}
        part_images = extract_dir / "images"
        if part_images.exists():
            for image in sorted(p for p in part_images.iterdir() if p.is_file()):
                target_name = image.name
                dest = images_dir / target_name
                if dest.exists() and dest.stat().st_size != image.stat().st_size:
                    target_name = f"p{order}-{image.name}"
                    renames[image.name] = target_name
                shutil.copy2(image, images_dir / target_name)
        for old_name, new_name in renames.items():
            text = text.replace(old_name, new_name)

        for item in _load_part_content_list(extract_dir):
            if isinstance(item.get("page_idx"), int):
                item["page_idx"] += offset
            img_path = item.get("img_path")
            if isinstance(img_path, str):
                base = Path(img_path).name
                if base in renames:
                    item["img_path"] = img_path.replace(base, renames[base])
            merged_items.append(item)

        md_chunks.append(text.strip())

    final_md = final_dir / f"{stem}.md"
    final_md.write_text("\n\n".join(md_chunks) + "\n", encoding="utf-8")
    if merged_items:
        (final_dir / f"{stem}_content_list.json").write_text(
            json.dumps(merged_items, ensure_ascii=False), encoding="utf-8"
        )
    logger.info(
        f"分段合并完成: {len(parts)} 段 -> {final_md}, 页锚点 {len(merged_items)} 条"
    )
    return str(final_md)


def _load_part_content_list(extract_dir: Path) -> list[dict]:
    """读取某一段的 content_list.json（MinerU 产物里带 page_idx 的那份）。"""
    candidates = sorted(extract_dir.rglob("*content_list.json"))
    if not candidates:
        return []
    target = max(candidates, key=lambda p: p.stat().st_size)
    try:
        with open(target, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception as e:
        logger.warning(f"读取分段 content_list.json 失败 {target}: {e}")
        return []
    return [item for item in data if isinstance(item, dict)] if isinstance(data, list) else []
