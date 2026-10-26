"""本地演示档的默认配置。

不依赖任何云端凭据：Milvus Lite + 词法向量 + SQLite 历史 + 本地文件存储，
Redis / RabbitMQ 关闭后走进程内兜底。

必须在导入任何 app 模块**之前**调用 configure_demo_env()。
各配置模块内部用的是 load_dotenv()，它不会覆盖已存在的环境变量，
所以这里设置的值优先于 .env。
"""
import os
from pathlib import Path

from app.utils.path_util import PROJECT_ROOT


def demo_dir() -> Path:
    path = PROJECT_ROOT / "output" / "demo"
    path.mkdir(parents=True, exist_ok=True)
    return path


def configure_demo_env() -> dict:
    root = demo_dir()
    defaults = {
        "EMBEDDING_PROVIDER": "lexical",
        "EMBEDDING_DIM": "1024",
        "EMBEDDING_WANT_SPARSE": "false",
        "RERANK_PROVIDER": "none",
        "MILVUS_URL": str(root / "milvus_lite.db"),
        "MILVUS_TOKEN": "",
        "STORAGE_BACKEND": "local",
        "LOCAL_STORAGE_DIR": str(root / "storage"),
        "HISTORY_BACKEND": "sqlite",
        "SQLITE_HISTORY_PATH": str(root / "history.db"),
        "PG_ENABLED": "false",
        "REDIS_ENABLED": "false",
        "RABBITMQ_ENABLED": "false",
        # 演示档没有任何凭据，Office 只能走本地解析器（生产默认是 mineru）
        "OFFICE_PARSE_ENGINE": "native",
        "LANGCHAIN_TRACING_V2": "false",
        "LOG_CONSOLE_LEVEL": "WARNING",
        "LOG_FILE_ENABLE": "false",
    }
    # 注意：必须直接赋值而不是 setdefault。
    # app/utils/path_util 在导入时就会 load_dotenv()，.env 里的值已经进了 os.environ，
    # setdefault 会被 .env 覆盖，导致演示档实际用的是云端配置。
    for key, value in defaults.items():
        os.environ[key] = value
    return defaults
