"""对象存储门面：oss | local。

生产用阿里云 OSS 私有 bucket（storage_key + 签名 URL）；
本地演示用 output/storage 下的文件，对外接口完全一致。
正文里存的一律是 storage_key，不存任何 bucket 的外链。
"""
import os
import shutil
from pathlib import Path
from typing import Optional

from app.conf.oss_config import oss_config
from app.core.logger import logger

BACKEND_OSS = "oss"
BACKEND_LOCAL = "local"


def backend() -> str:
    return os.getenv("STORAGE_BACKEND", BACKEND_OSS).strip().lower()


def local_root() -> Path:
    return Path(os.getenv("LOCAL_STORAGE_DIR", "output/storage"))


def is_enabled() -> bool:
    if backend() == BACKEND_LOCAL:
        return True
    from app.clients import oss_utils

    return oss_utils.is_enabled()


def build_storage_key(file_title: str, filename: str, sub_dir: str = "images") -> str:
    from app.clients import oss_utils

    return oss_utils.build_storage_key(file_title, filename, sub_dir=sub_dir)


def upload_file(local_path: str, storage_key: str) -> Optional[str]:
    if backend() == BACKEND_LOCAL:
        target = local_root() / storage_key
        try:
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(local_path, target)
            return storage_key
        except Exception as e:
            logger.error(f"本地存储写入失败 {storage_key}: {e}")
            return None

    from app.clients import oss_utils

    return oss_utils.upload_file(local_path, storage_key)


def upload_bytes(data: bytes, storage_key: str) -> Optional[str]:
    if backend() == BACKEND_LOCAL:
        target = local_root() / storage_key
        try:
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(data)
            return storage_key
        except Exception as e:
            logger.error(f"本地存储写入失败 {storage_key}: {e}")
            return None

    from app.clients import oss_utils

    return oss_utils.upload_bytes(data, storage_key)


def resolve_local(storage_key: str) -> Optional[Path]:
    """本地后端返回磁盘路径；OSS 后端返回 None。"""
    if backend() != BACKEND_LOCAL:
        return None
    path = local_root() / storage_key
    return path if path.is_file() else None


def sign_url(storage_key: str, ttl_seconds: Optional[int] = None) -> Optional[str]:
    """生成可访问地址。本地后端返回本站 /media 路径。"""
    if backend() == BACKEND_LOCAL:
        return f"/media/{storage_key}" if resolve_local(storage_key) else None

    from app.clients import oss_utils

    return oss_utils.sign_url(storage_key, ttl_seconds)
