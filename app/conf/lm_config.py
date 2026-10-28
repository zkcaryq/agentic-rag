# 导入核心依赖：数据类、环境变量读取、路径处理
from dataclasses import dataclass
import os
from dotenv import load_dotenv

# 提前加载.env配置文件（必须在读取环境变量前执行，确保os.getenv能获取到值）
# 若.env不在项目根目录，可指定路径：load_dotenv(dotenv_path=Path(__file__).parent / ".env")
load_dotenv()


# 定义minerU服务配置
@dataclass
class LLMConfig:
    base_url: str
    api_key : str
    lv_model: str
    llm_model: str
    llm_temperature: float
    context_budget_tokens: int  # 拼上下文时的预算上限（token 近似值）
    # 图片摘要的调用速率（次/分钟，按 worker 进程计）。批量导入时这一段的耗时是
    # 线性的：431 张图 @9 次/分 ≈ 48 分钟，所以必须能按额度调，而不是写死。
    vlm_summary_rpm: int

lm_config = LLMConfig(
    base_url=os.getenv("OPENAI_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1"),
    api_key=os.getenv("OPENAI_API_KEY", ""),
    lv_model=os.getenv("VL_MODEL", "qwen-vl-max"),
    llm_model=os.getenv("LLM_DEFAULT_MODEL", "qwen-plus"),
    llm_temperature=float(os.getenv("LLM_DEFAULT_TEMPERATURE", "0.3")),
    context_budget_tokens=int(os.getenv("CONTEXT_BUDGET_TOKENS", "6000")),
    vlm_summary_rpm=int(os.getenv("VLM_SUMMARY_RPM", "9")),
)
