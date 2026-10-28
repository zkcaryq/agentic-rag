"""项目根目录定位。

模块导入时就会 load_dotenv()，这也是演示档必须**直接赋值**而不是 setdefault
的原因（见 app/conf/demo_config.py）。
"""
import os
from pathlib import Path

from dotenv import load_dotenv


def get_project_root(identifier: str = "pyproject.toml") -> Path:
    # 第一步：优先读取环境变量（生产环境用）
    env_root = os.getenv("PROJECT_ROOT")
    if env_root and Path(env_root).absolute().exists():
        return Path(env_root).absolute()

    # 第二步：用受版本控制的文件定位项目根目录；.env 只负责可选配置，
    # 不能作为根目录标识，否则 CI / 容器这类不带 .env 的环境会在导入期失败。
    current_dir = Path(__file__).absolute().parent
    while current_dir != current_dir.parent:
        if (current_dir / identifier).exists():
            env_file = current_dir / ".env"
            if env_file.exists():
                load_dotenv(dotenv_path=env_file)
            return current_dir
        current_dir = current_dir.parent

    raise FileNotFoundError(f"未找到项目根目录标识「{identifier}」，且环境变量PROJECT_ROOT未配置")


PROJECT_ROOT = get_project_root()
