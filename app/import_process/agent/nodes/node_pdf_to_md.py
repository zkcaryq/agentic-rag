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
        2. 上传PDF至MinerU并轮询解析结果
        3. 下载ZIP包并提取MD文件
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

        # 步骤2：上传并轮询
        zip_url = step_2_upload_and_poll(pdf_path, output_dir)

        # 步骤3：下载并提取MD文件
        md_path = step_3_download_and_extract(zip_url, output_dir, pdf_path.stem)

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
            "model_version": "vlm"
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
