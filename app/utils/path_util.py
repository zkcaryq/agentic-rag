"""项目根目录定位。

模块导入时就会 load_dotenv()，这也是演示档必须**直接赋值**而不是 setdefault
的原因（见 app/conf/demo_config.py）。
"""
import os
from pathlib import Path

from dotenv import load_dotenv


def get_project_root(identifier: str = ".env") -> Path:
    # 第一步：优先读取环境变量（生产环境用）
    env_root = os.getenv("PROJECT_ROOT")
    if env_root and Path(env_root).absolute().exists():
        return Path(env_root).absolute()

    # 第二步：加载根目录的.env文件（为了后续逻辑，也可省略）
    current_dir = Path(__file__).absolute().parent
    while current_dir != current_dir.parent:
        if (current_dir / identifier).exists():
            load_dotenv(dotenv_path=current_dir / identifier)
            break
        current_dir = current_dir.parent

    # 第三步：递归查找标识（兜底，开发环境用）
    current_dir = Path(__file__).absolute().parent
    while current_dir != current_dir.parent:
        if (current_dir / identifier).exists():
            return current_dir
        current_dir = current_dir.parent

    raise FileNotFoundError(f"未找到项目根目录标识「{identifier}」，且环境变量PROJECT_ROOT未配置")


PROJECT_ROOT = get_project_root(".env")