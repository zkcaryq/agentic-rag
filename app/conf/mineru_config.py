# 导入核心依赖：数据类、环境变量读取、路径处理
from dataclasses import dataclass
import os
from dotenv import load_dotenv

# 提前加载.env配置文件（必须在读取环境变量前执行，确保os.getenv能获取到值）
# 若.env不在项目根目录，可指定路径：load_dotenv(dotenv_path=Path(__file__).parent / ".env")
load_dotenv()


# 定义minerU服务配置
@dataclass
class MineruConfig:
    base_url: str
    api_key : str
    # Office 文件走哪条解析路：mineru（默认，与简历"基于 MinerU 完成 PDF/Word/Excel
    # 解析转换"一致，实测 MinerU 收 docx/xlsx 并返回 Markdown+HTML 表格）
    # 或 native（python-docx / openpyxl，零网络、表格转 Markdown 更干净、保留 Excel 行号）
    office_parse_engine: str
    # MinerU 解析引擎：vlm（默认）或 pipeline，见 MineruConfig.model_version 的说明
    model_version: str
    # MinerU 失败时是否退回本地解析器：Word/Excel 本地能解，退一步比整单失败好；
    # PDF 没有本地兜底，仍然如实报错。
    office_fallback_native: bool
    # 单次任务的页数上限。MinerU 服务端会直接拒收超页 PDF（实测报
    # "number of pages exceeds limit (200 pages), please split the file"），
    # 所以本地先按这个值切段、逐段提交，再把产物拼回来。
    max_pages_per_task: int


def _env_bool(name: str, default: bool = False) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in ("1", "true", "yes", "on")

def _mineru_base_url() -> str:
    """归一化成 API 根地址。

    代码里拼的是 {base}/file-urls/batch 与 {base}/extract-results/batch/{id}，
    所以 base 必须是根。原仓库 .env 传下来的是 .../api/v4/extract/task
    （那是单个建任务接口的完整地址），照抄进新配置就会让 MinerU 全链路 404。
    """
    raw = (os.getenv("MINERU_BASE_URL") or "").strip().rstrip("/")
    for suffix in ("/extract/task", "/extract-results", "/file-urls"):
        if raw.endswith(suffix):
            return raw[: -len(suffix)]
    return raw


mineru_config = MineruConfig(
    base_url=_mineru_base_url(),
    api_key=os.getenv("MINERU_API_TOKEN"),
    office_parse_engine=(os.getenv("OFFICE_PARSE_ENGINE") or "mineru").strip().lower(),
    model_version=(os.getenv("MINERU_MODEL_VERSION") or "vlm").strip().lower(),
    office_fallback_native=_env_bool("OFFICE_PARSE_FALLBACK_NATIVE", True),
    max_pages_per_task=int(os.getenv("MINERU_MAX_PAGES_PER_TASK") or 200),
)