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
    api_key=os.getenv("MINERU_API_TOKEN")
)