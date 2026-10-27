"""
导入流程状态定义
所有导入节点共享的数据结构
"""
from typing import TypedDict, List, Dict, Any, Optional


class ImportGraphState(TypedDict):
    """
    导入流程图的状态定义，包含所有节点产生和消费的数据字段。
    TypedDict 让我们在代码中能有自动补全和类型检查。
    """

    # ==================== 基础标识 ====================
    task_id: str  # 任务唯一ID，用于追踪日志
    file_title: str  # 文件标题（文件名去后缀）

    # ==================== 路径相关 ====================
    local_dir: str  # 当前工作目录或输出目录
    local_file_path: str  # 原始输入文件路径
    pdf_path: str  # PDF 文件路径 (如果输入是PDF)
    md_path: str  # Markdown 文件路径 (转换后或直接输入的)

    # ==================== 流程控制标记 ====================
    is_md_read_enabled: bool  # 是否启用 Markdown 读取路径
    is_pdf_read_enabled: bool  # 是否启用 PDF 读取路径
    is_office_read_enabled: bool  # 是否启用 Word/Excel 读取路径

    # ==================== 内容数据 ====================
    md_content: str  # Markdown 的全文内容
    chunks: List[Dict[str, Any]]  # 切片后的文本列表，包含 metadata
    product_name: str  # 识别出的主体名称 (如: "万用表")

    # ==================== 向量相关 ====================
    embeddings_content: List[Dict[str, Any]]  # 包含向量数据的列表，准备写入 Milvus

    # ==================== 失败归因 ====================
    # 记录失败发生在哪个阶段（mineru/image/split/embedding/milvus/office），
    # 供导入任务表统计失败率与归因使用；成功时为 None
    failed_stage: Optional[str]

    # 这次执行是不是接着断点跑。必须写进 schema：LangGraph 只传递声明过的键，
    # 之前 task_runner 设了 resumable 但 schema 里没有它，节点包装器读到的永远是
    # None，"跳过已完成节点"这件事在真实图里从来没生效过（MinerU 被白调了一遍）。
    resumable: bool


# ==================== 默认状态配置 ====================
def get_default_state() -> ImportGraphState:
    """
    获取默认状态，用于初始化
    """
    return {
        "task_id": "",
        "file_title": "",
        "local_dir": "",
        "local_file_path": "",
        "pdf_path": "",
        "md_path": "",
        "is_md_read_enabled": False,
        "is_pdf_read_enabled": False,
        "is_office_read_enabled": False,
        "md_content": "",
        "chunks": [],
        "product_name": "",
        "embeddings_content": [],
        "failed_stage": None,
        "resumable": False,
    }


def create_default_state(**overrides) -> ImportGraphState:
    """
    创建默认状态，支持覆盖特定字段

    Args:
        **overrides: 要覆盖的字段（关键字参数）

    Returns:
        新的状态实例

    Examples:
        state = create_default_state(
            task_id="task_001",
            local_file_path="doc/test.pdf"
        )
    """
    state = get_default_state()
    state.update(overrides)
    return state
